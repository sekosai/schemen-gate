"""Concrete VectorStore implementations for external databases.

PgVectorStore — Level 1 regime-tagged vectors. Uses a ``regime_id``
column and partition-key filtering (``WHERE regime_id = $1``). The PostgreSQL
server must have the pgvector extension installed; the Python client dependency
alone is not sufficient. Structural isolation comes from the Gate at inference
time, not from the database extension.

Requires ``pip install schemen-gate[rag]`` for psycopg.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, cast

import numpy as np

from schemen_gate._cargo import _clone_json_object
from schemen_gate._rag import RetrievedDoc, _bounded_text, _vector_copy


class PgVectorStore:
    """Vector store backed by PostgreSQL + pgvector.

    Level 1 isolation: regime_id column with partition filtering.
    The table must already exist with the expected schema::

        CREATE TABLE IF NOT EXISTS schemen_vectors (
            id          TEXT PRIMARY KEY,
            regime_id   TEXT NOT NULL,
            embedding   vector({dim}),
            content     TEXT,
            kind        TEXT DEFAULT 'document',
            metadata    JSONB DEFAULT '{{}}'::jsonb
        );
        CREATE INDEX ON schemen_vectors USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX ON schemen_vectors (regime_id);

    Parameters
    ----------
    conninfo : str
        PostgreSQL connection string.
    table : str
        Table name (default ``schemen_vectors``).
    dim : int
        Embedding dimensionality (for vector cast validation).
    """

    def __init__(
        self,
        conninfo: str,
        *,
        table: str = "schemen_vectors",
        dim: int = 768,
    ) -> None:
        _bounded_text(conninfo, "conninfo")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("table must be a simple PostgreSQL identifier")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        try:
            from psycopg import sql
        except ImportError as exc:
            raise ImportError(
                "psycopg required for PgVectorStore. Install with: pip install schemen-gate[rag]"
            ) from exc
        self._conninfo = conninfo
        self._sql = sql
        self._table = table
        self._dim = dim
        self._conn: Optional[Any] = None
        self._conn_lock = threading.RLock()

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """Serialize one complete operation on the shared psycopg connection."""

        with self._conn_lock:
            if self._conn is None or self._conn.closed:
                import psycopg

                self._conn = psycopg.connect(self._conninfo, autocommit=True)
            yield self._conn

    def _embedding_literal(self, embedding: np.ndarray) -> str:
        array = np.asarray(embedding, dtype=np.float64)
        if array.ndim != 1 or array.shape[0] != self._dim:
            raise ValueError(f"embedding must be a 1D vector with dimension {self._dim}")
        if not np.all(np.isfinite(array)):
            raise ValueError("embedding must contain only finite values")
        return "[" + ",".join(str(value) for value in array.tolist()) + "]"

    def _embedding_from_row(self, value: Any) -> np.ndarray:
        """Decode and validate one database vector without synthesizing data."""
        try:
            if isinstance(value, str):
                decoded = json.loads(value)
            elif isinstance(value, (bytes, bytearray, dict)) or not hasattr(value, "__iter__"):
                raise ValueError("database embedding has an unsupported type")
            else:
                decoded = list(value)
            return _vector_copy(
                decoded,
                "database embedding",
                expected_dimensions=self._dim,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("database returned an invalid embedding") from exc

    @staticmethod
    def _metadata_from_row(value: Any) -> Dict[str, Any]:
        """Decode one database JSON object and return an independent copy."""
        try:
            decoded = json.loads(value) if isinstance(value, str) else value
            return _clone_json_object(
                {} if decoded is None else decoded,
                name="database metadata",
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("database returned invalid metadata") from exc

    def ensure_table(self) -> None:
        """Create the vectors table and indexes after proving pgvector exists."""
        with self._connection() as conn:
            extension = conn.execute("SELECT to_regtype('vector')").fetchone()
            if not extension or extension[0] is None:
                raise RuntimeError(
                    "PostgreSQL pgvector extension is required before ensure_table()"
                )
            conn.execute(
                self._sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    id          TEXT PRIMARY KEY,
                    regime_id   TEXT NOT NULL,
                    embedding   vector({dim}),
                    content     TEXT,
                    kind        TEXT DEFAULT 'document',
                    metadata    JSONB DEFAULT '{{}}'::jsonb
                )
            """).format(
                    table=self._sql.Identifier(self._table),
                    dim=self._sql.SQL(str(self._dim)),
                )
            )
            conn.execute(
                self._sql.SQL("""
                CREATE INDEX IF NOT EXISTS {index}
                ON {table} USING hnsw (embedding vector_cosine_ops)
            """).format(
                    index=self._sql.Identifier(f"{self._table}_hnsw_idx"),
                    table=self._sql.Identifier(self._table),
                )
            )
            conn.execute(
                self._sql.SQL("""
                CREATE INDEX IF NOT EXISTS {index}
                ON {table} (regime_id)
            """).format(
                    index=self._sql.Identifier(f"{self._table}_regime_idx"),
                    table=self._sql.Identifier(self._table),
                )
            )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        partition_key: str,
        top_k: int = 10,
        *,
        kind: Optional[str] = None,
    ) -> List[RetrievedDoc]:
        _bounded_text(partition_key, "partition_key")
        _bounded_text(kind, "kind", allow_none=True)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        emb_literal = self._embedding_literal(query_embedding)

        kind_clause = ""
        params: dict[str, Any] = {
            "regime_id": partition_key,
            "top_k": top_k,
        }
        if kind is not None:
            kind_clause = "AND kind = %(kind)s"
            params["kind"] = kind

        statement = self._sql.SQL("""
            SELECT id, content, embedding, kind, metadata,
                   1 - (embedding <=> %(emb)s::vector) AS score
            FROM {table}
            WHERE regime_id = %(regime_id)s {kind_clause}
            ORDER BY embedding <=> %(emb)s::vector
            LIMIT %(top_k)s
        """).format(
            table=self._sql.Identifier(self._table),
            kind_clause=self._sql.SQL(kind_clause),
        )
        with self._connection() as conn:
            rows = conn.execute(
                statement,
                {**params, "emb": emb_literal},
            ).fetchall()

        results = []
        for row in rows:
            doc_id, content, emb_raw, row_kind, meta, score = row
            _bounded_text(doc_id, "database doc_id")
            _bounded_text(
                "" if content is None else content,
                "database content",
                allow_empty=True,
            )
            resolved_kind = "document" if row_kind is None else row_kind
            _bounded_text(resolved_kind, "database kind")
            resolved_score = float(score) if score is not None else 0.0
            if not math.isfinite(resolved_score):
                raise ValueError("database returned a non-finite retrieval score")
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    content="" if content is None else content,
                    embedding=self._embedding_from_row(emb_raw),
                    score=resolved_score,
                    partition_key=partition_key,
                    kind=resolved_kind,
                    metadata=self._metadata_from_row(meta),
                )
            )
        return results

    def insert(
        self,
        embedding: np.ndarray,
        document: str,
        partition_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        _bounded_text(document, "document", allow_empty=True)
        _bounded_text(partition_key, "partition_key")
        self._embedding_literal(embedding)
        with self._connection() as conn:
            return self._insert_one(
                conn,
                embedding,
                document,
                partition_key,
                metadata,
            )

    def _insert_one(
        self,
        conn: Any,
        embedding: np.ndarray,
        document: str,
        partition_key: str,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        emb_literal = self._embedding_literal(embedding)
        _bounded_text(document, "document", allow_empty=True)
        _bounded_text(partition_key, "partition_key")
        meta = _clone_json_object(
            metadata if metadata is not None else {},
            name="metadata",
        )
        doc_id = meta.pop("doc_id", str(uuid.uuid4()))
        kind = meta.pop("kind", "document")
        _bounded_text(doc_id, "doc_id")
        _bounded_text(kind, "kind")
        statement = self._sql.SQL("""
            INSERT INTO {table} (id, regime_id, embedding, content, kind, metadata)
            VALUES (%(id)s, %(regime_id)s, %(emb)s::vector, %(content)s, %(kind)s, %(meta)s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                content = EXCLUDED.content,
                kind = EXCLUDED.kind,
                metadata = EXCLUDED.metadata
            WHERE {table}.regime_id = EXCLUDED.regime_id
            RETURNING id
        """).format(table=self._sql.Identifier(self._table))
        row = conn.execute(
            statement,
            {
                "id": doc_id,
                "regime_id": partition_key,
                "emb": emb_literal,
                "content": document,
                "kind": kind,
                "meta": json.dumps(meta),
            },
        ).fetchone()
        if row is None:
            raise ValueError("document id already belongs to another partition")
        return cast(str, doc_id)

    def insert_many(
        self,
        items: list[tuple[np.ndarray, str, Optional[Dict[str, Any]]]],
        partition_key: str,
    ) -> List[str]:
        """Insert a Cargo batch in one PostgreSQL transaction."""
        _bounded_text(partition_key, "partition_key")
        for embedding, document, metadata in items:
            self._embedding_literal(embedding)
            _bounded_text(document, "document", allow_empty=True)
            _clone_json_object(
                metadata if metadata is not None else {},
                name="metadata",
            )
        with self._connection() as conn:
            with conn.transaction():
                return [
                    self._insert_one(
                        conn,
                        embedding,
                        document,
                        partition_key,
                        metadata,
                    )
                    for embedding, document, metadata in items
                ]

    def list_by_kind(
        self,
        partition_key: str,
        kind: str,
    ) -> List[RetrievedDoc]:
        _bounded_text(partition_key, "partition_key")
        _bounded_text(kind, "kind")
        statement = self._sql.SQL("""
            SELECT id, content, embedding, metadata
            FROM {table}
            WHERE regime_id = %(regime_id)s AND kind = %(kind)s
        """).format(table=self._sql.Identifier(self._table))
        with self._connection() as conn:
            rows = conn.execute(
                statement,
                {"regime_id": partition_key, "kind": kind},
            ).fetchall()

        results = []
        for row in rows:
            doc_id, content, emb_raw, meta = row
            _bounded_text(doc_id, "database doc_id")
            _bounded_text(
                "" if content is None else content,
                "database content",
                allow_empty=True,
            )
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    content="" if content is None else content,
                    embedding=self._embedding_from_row(emb_raw),
                    score=0.0,
                    partition_key=partition_key,
                    kind=kind,
                    metadata=self._metadata_from_row(meta),
                )
            )
        return results

    def count(
        self,
        partition_key: str,
        *,
        kind: Optional[str] = None,
    ) -> int:
        _bounded_text(partition_key, "partition_key")
        _bounded_text(kind, "kind", allow_none=True)
        with self._connection() as conn:
            if kind is not None:
                statement = self._sql.SQL("""
                    SELECT COUNT(*) FROM {table}
                    WHERE regime_id = %(regime_id)s AND kind = %(kind)s
                """).format(table=self._sql.Identifier(self._table))
                row = conn.execute(
                    statement,
                    {"regime_id": partition_key, "kind": kind},
                ).fetchone()
            else:
                statement = self._sql.SQL("""
                    SELECT COUNT(*) FROM {table}
                    WHERE regime_id = %(regime_id)s
                """).format(table=self._sql.Identifier(self._table))
                row = conn.execute(
                    statement,
                    {"regime_id": partition_key},
                ).fetchone()
        return int(row[0]) if row else 0

    def delete_partition(self, partition_key: str) -> int:
        """Delete all vectors for a partition. Returns count deleted."""
        _bounded_text(partition_key, "partition_key")
        statement = self._sql.SQL("""
            WITH deleted AS (
                DELETE FROM {table}
                WHERE regime_id = %(regime_id)s
                RETURNING 1
            )
            SELECT COUNT(*) FROM deleted
        """).format(table=self._sql.Identifier(self._table))
        with self._connection() as conn:
            row = conn.execute(
                statement,
                {"regime_id": partition_key},
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._conn_lock:
            if self._conn and not self._conn.closed:
                self._conn.close()
