"""Real-model Cargo/RAG and whole-model authorization test.

Customer corpora stay in keyed vector-store partitions.  A tenant-and-regime
access key must successfully dock before retrieval context is supplied to the
Transformer.  A separate tenant + sub-id + model + operation gate treats the
complete generator as an atomic resource. Wrong keys are rejected before the
generation model is invoked.

This tests runtime/corpus separation, not attention-coordinate isolation:
attention remains a normal shared Transformer component and sees only context
released by the authorized Cargo session.

    modal run experiments/modal_cargo_transformer_authorization.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from library_provenance import collect_experiment_provenance
from modal_schemen_image import (
    assert_remote_schemen_versions,
    install_current_schemen,
)

APP_NAME = "cdp-cargo-transformer-authorization"
VOLUME_NAME = "cdp-cargo-transformer-results"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GENERATION_MODEL = "google/flan-t5-small"
GENERATION_MODEL_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
MODEL_DIGEST = hashlib.sha256(
    (
        f"{EMBED_MODEL}@{EMBED_MODEL_REVISION}|"
        f"{GENERATION_MODEL}@{GENERATION_MODEL_REVISION}"
    ).encode()
).hexdigest()
GENERATION_MODEL_DIGEST = hashlib.sha256(
    f"{GENERATION_MODEL}@{GENERATION_MODEL_REVISION}".encode()
).hexdigest()
app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = install_current_schemen(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.10.1",
        "sentence-transformers==5.5.1",
        "sentencepiece==0.2.1",
        "numpy==2.4.6",
        "cryptography==50.0.0",
    ),
    launcher=Path(__file__),
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def source_metadata() -> dict[str, Any]:
    return collect_experiment_provenance(Path(__file__))


@app.function(
    max_containers=3,
    image=image,
    gpu="T4",
    timeout=5 * 60,
    volumes={"/results": results_volume},
)
def run_test(source: dict[str, Any]) -> dict[str, Any]:
    import os
    from importlib import metadata

    import numpy as np
    import torch
    from execution_preflight import ExecutionPreflightError, GateExecutionPreflight
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from schemen_gate import (
        BusTerminal,
        CargoAuthenticationError,
        CargoItem,
        EmbeddingSpec,
        GatedRAGAdapter,
        InMemoryVectorStore,
        PartitionMap,
        PartitionMode,
        compute_payload_hash,
        create_manifest,
        derive_cargo_access_key,
        hash_vocabulary,
    )

    source = assert_remote_schemen_versions(
        source,
        launcher_name=Path(__file__).name,
    )

    device = torch.device("cuda")
    embedder = SentenceTransformer(
        EMBED_MODEL,
        revision=EMBED_MODEL_REVISION,
        device=str(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL,
        revision=GENERATION_MODEL_REVISION,
    )
    generator = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL,
        revision=GENERATION_MODEL_REVISION,
    ).to(device)
    generator.eval()
    dimensions = int(embedder.get_embedding_dimension())

    embedding_spec = EmbeddingSpec(
        model_id=EMBED_MODEL,
        dimensions=dimensions,
        vocabulary_hash=hash_vocabulary(list(tokenizer.get_vocab().keys())),
        pooling="mean",
    )
    master_key = hashlib.sha256(b"cdp-cargo-real-model-suite").digest()
    store = InMemoryVectorStore()
    partition_map = PartitionMap(master_key, dimensions, 4)

    def embed(text: str) -> np.ndarray:
        return np.asarray(
            embedder.encode(text, normalize_embeddings=True),
            dtype=np.float64,
        )

    adapter = GatedRAGAdapter(store, partition_map, embed_fn=embed)
    terminal = BusTerminal(adapter, master_key, embedding_spec)

    tenants = [
        {
            "partition": "customer-alpha",
            "tenant_id": "customer-alpha",
            "sub_id": "claims-team",
            "regime": 0,
            "facts": [
                (
                    "Alpha's confidential launch code is ORBIT-731.",
                    "What is Alpha's confidential launch code?",
                    "ORBIT-731",
                ),
                (
                    "Alpha's private renewal month is November.",
                    "What is Alpha's private renewal month?",
                    "November",
                ),
            ],
        },
        {
            "partition": "customer-beta",
            "tenant_id": "customer-beta",
            "sub_id": "renewals-team",
            "regime": 1,
            "facts": [
                (
                    "Beta's confidential launch code is NEBULA-284.",
                    "What is Beta's confidential launch code?",
                    "NEBULA-284",
                ),
                (
                    "Beta's private renewal month is February.",
                    "What is Beta's private renewal month?",
                    "February",
                ),
            ],
        },
    ]

    access_keys: dict[str, dict[str, bytes]] = {}
    load_receipts_valid = []
    for tenant in tenants:
        partition_map.register(
            tenant["partition"],
            regime_id=tenant["regime"],
            mode=PartitionMode.READ_WRITE,
        )
        terminal.register_bus(tenant["partition"], tenant["regime"])
        items = [
            CargoItem(
                content=fact,
                embedding=embed(fact),
                doc_id=f"{tenant['tenant_id']}-{index}",
            )
            for index, (fact, _question, _answer) in enumerate(tenant["facts"])
        ]
        manifest = create_manifest(
            tenant_id=tenant["tenant_id"],
            regime_id=tenant["regime"],
            embedding_spec=embedding_spec,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=len(items),
            subject_id=tenant["sub_id"],
            model_digest=MODEL_DIGEST,
            operation="load",
            policy_version="cargo-policy-v1",
            partition_key=tenant["partition"],
            gate_embeddings_at_rest=False,
        )
        access_keys[tenant["tenant_id"]] = {
            operation: derive_cargo_access_key(
                master_key,
                tenant["regime"],
                tenant["tenant_id"],
                subject_id=tenant["sub_id"],
                model_digest=MODEL_DIGEST,
                operation=operation,
                policy_version="cargo-policy-v1",
                partition_key=tenant["partition"],
            )
            for operation in ("load", "retrieve")
        }
        access_key = access_keys[tenant["tenant_id"]]["load"]
        session = terminal.get_bus(tenant["partition"]).dock(
            manifest,
            gate_key_secret=access_key,
        )
        session.load_cargo(items)
        receipt = session.depart()
        load_receipts_valid.append(
            terminal.get_bus(tenant["partition"]).verify_receipt(
                receipt,
                expected_manifest=manifest,
            )
        )

    alias_partition = "customer-alpha-same-regime-alias"
    partition_map.register(
        alias_partition,
        regime_id=tenants[0]["regime"],
        mode=PartitionMode.IMMUTABLE,
    )
    terminal.register_bus(alias_partition, tenants[0]["regime"])

    model_calls = 0

    @torch.no_grad()
    def answer(context: str, question: str) -> str:
        nonlocal model_calls
        model_calls += 1
        prompt = (
            "Answer using only the supplied context. "
            f"Context: {context} Question: {question}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated = generator.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def authorize_runtime(
        tenant: dict[str, Any],
        *,
        frame_model: str = GENERATION_MODEL_DIGEST,
        frame_dimensions: int = 1,
        frame_regime: int | None = None,
        authorized_regimes: list[int] | None = None,
    ) -> Any:
        authority = (
            [tenant["regime"]]
            if authorized_regimes is None
            else authorized_regimes
        )
        if not authority:
            raise ExecutionPreflightError(
                "no Regime authority is configured",
                status_code=403,
            )
        surface = GateExecutionPreflight(
            model_id=GENERATION_MODEL_DIGEST,
            dimensions=1,
            authorized_regime_ids=authority,
        )
        return surface.authorize(
            tenant["regime"] if frame_regime is None else frame_regime,
            model_id=frame_model,
            dimensions=frame_dimensions,
        )

    def invoke_model(
        tenant: dict[str, Any], context: str, question: str
    ) -> str:
        authorize_runtime(tenant)
        return answer(context, question)

    owning_rows = []
    retrieval_receipts_valid = []
    for tenant in tenants:
        bus = terminal.get_bus(tenant["partition"])
        for _fact, question, expected in tenant["facts"]:
            read_manifest = create_manifest(
                tenant_id=tenant["tenant_id"],
                regime_id=tenant["regime"],
                embedding_spec=embedding_spec,
                payload_hash=compute_payload_hash([]),
                payload_kind="rag_documents",
                item_count=0,
                subject_id=tenant["sub_id"],
                model_digest=MODEL_DIGEST,
                operation="retrieve",
                policy_version="cargo-policy-v1",
                partition_key=tenant["partition"],
                gate_embeddings_at_rest=False,
            )
            retrieve_key = access_keys[tenant["tenant_id"]]["retrieve"]
            session = bus.dock(
                read_manifest,
                gate_key_secret=retrieve_key,
            )
            retrieved = session.unload_cargo(question, top_k=1)
            context = retrieved.docs[0].content if retrieved.docs else ""
            response = invoke_model(tenant, context, question)
            receipt = session.depart()
            retrieval_receipts_valid.append(
                bus.verify_receipt(receipt, expected_manifest=read_manifest)
            )
            owning_rows.append(
                {
                    "tenant_id": tenant["tenant_id"],
                    "sub_id": tenant["sub_id"],
                    "question": question,
                    "expected": expected,
                    "response": response,
                    "retrieved_partition": (
                        retrieved.docs[0].partition_key
                        if retrieved.docs
                        else None
                    ),
                    "correct": expected.lower() in response.lower(),
                }
            )

    rejection_rows = []
    for target in tenants:
        bus = terminal.get_bus(target["partition"])
        manifest = create_manifest(
            tenant_id=target["tenant_id"],
            regime_id=target["regime"],
            embedding_spec=embedding_spec,
            payload_hash=compute_payload_hash([]),
            payload_kind="rag_documents",
            item_count=0,
            subject_id=target["sub_id"],
            model_digest=MODEL_DIGEST,
            operation="retrieve",
            policy_version="cargo-policy-v1",
            partition_key=target["partition"],
            gate_embeddings_at_rest=False,
        )
        attempts = [
            ("random", os.urandom(32)),
            (
                "other_operation_same_scope",
                access_keys[target["tenant_id"]]["load"],
            ),
            (
                "other_tenant",
                access_keys[
                    next(
                        tenant["tenant_id"]
                        for tenant in tenants
                        if tenant["tenant_id"] != target["tenant_id"]
                    )
                ]["retrieve"],
            ),
            (
                "other_sub_id_same_tenant",
                derive_cargo_access_key(
                    master_key,
                    target["regime"],
                    target["tenant_id"],
                    subject_id="unauthorized-subject",
                    model_digest=MODEL_DIGEST,
                    operation="retrieve",
                    policy_version="cargo-policy-v1",
                    partition_key=target["partition"],
                ),
            ),
            (
                "other_regime_same_tenant",
                derive_cargo_access_key(
                    master_key,
                    (target["regime"] + 1) % 4,
                    target["tenant_id"],
                    subject_id=target["sub_id"],
                    model_digest=MODEL_DIGEST,
                    operation="retrieve",
                    policy_version="cargo-policy-v1",
                    partition_key=target["partition"],
                ),
            ),
            (
                "other_model_same_identity",
                derive_cargo_access_key(
                    master_key,
                    target["regime"],
                    target["tenant_id"],
                    subject_id=target["sub_id"],
                    model_digest="0" * 64,
                    operation="retrieve",
                    policy_version="cargo-policy-v1",
                    partition_key=target["partition"],
                ),
            ),
            (
                "other_policy_same_identity",
                derive_cargo_access_key(
                    master_key,
                    target["regime"],
                    target["tenant_id"],
                    subject_id=target["sub_id"],
                    model_digest=MODEL_DIGEST,
                    operation="retrieve",
                    policy_version="cargo-policy-v0",
                    partition_key=target["partition"],
                ),
            ),
        ]
        for attack, key in attempts:
            calls_before = model_calls
            rejected = False
            try:
                session = bus.dock(manifest, gate_key_secret=key)
                retrieved = session.unload_cargo("confidential code", top_k=1)
                context = retrieved.docs[0].content if retrieved.docs else ""
                answer(context, "What is the confidential code?")
                session.depart()
            except CargoAuthenticationError:
                rejected = True
            rejection_rows.append(
                {
                    "target": target["tenant_id"],
                    "target_sub_id": target["sub_id"],
                    "attack": attack,
                    "rejected": rejected,
                    "model_calls": model_calls - calls_before,
                }
            )

    alpha = tenants[0]
    alpha_manifest = create_manifest(
        tenant_id=alpha["tenant_id"],
        regime_id=alpha["regime"],
        embedding_spec=embedding_spec,
        payload_hash=compute_payload_hash([]),
        payload_kind="rag_documents",
        item_count=0,
        subject_id=alpha["sub_id"],
        model_digest=MODEL_DIGEST,
        operation="retrieve",
        policy_version="cargo-policy-v1",
        partition_key=alpha["partition"],
        gate_embeddings_at_rest=False,
    )
    calls_before = model_calls
    same_regime_partition_rejected = False
    try:
        terminal.get_bus(alias_partition).dock(
            alpha_manifest,
            gate_key_secret=access_keys[alpha["tenant_id"]]["retrieve"],
        )
    except CargoAuthenticationError:
        same_regime_partition_rejected = True
    rejection_rows.append(
        {
            "target": alias_partition,
            "target_sub_id": "claims-team",
            "attack": "other_partition_same_regime_valid_origin_key",
            "rejected": same_regime_partition_rejected,
            "model_calls": model_calls - calls_before,
        }
    )

    for target in tenants:
        attempts = [
            (
                "runtime_other_model_digest",
                {"frame_model": "0" * 64},
            ),
            (
                "runtime_other_regime",
                {"frame_regime": (target["regime"] + 1) % 4},
            ),
            (
                "runtime_wrong_dimensions",
                {"frame_dimensions": 2},
            ),
            (
                "runtime_empty_authority",
                {"authorized_regimes": []},
            ),
        ]
        for attack, runtime_changes in attempts:
            calls_before = model_calls
            rejected = False
            try:
                authorize_runtime(target, **runtime_changes)
                answer(
                    "Public control context.", "What is the context?"
                )
            except ExecutionPreflightError:
                rejected = True
            rejection_rows.append(
                {
                    "target": target["tenant_id"],
                    "target_sub_id": target["sub_id"],
                    "attack": attack,
                    "rejected": rejected,
                    "model_calls": model_calls - calls_before,
                }
            )

    owning_recall = float(
        np.mean([row["correct"] for row in owning_rows])
    )
    all_rejected = all(row["rejected"] for row in rejection_rows)
    zero_unauthorized_calls = all(
        row["model_calls"] == 0 for row in rejection_rows
    )
    result = {
        "schema_version": 3,
        "experiment": "cargo_transformer_corpus_authorization",
        "status": (
            "pass"
            if owning_recall == 1.0
            and all_rejected
            and zero_unauthorized_calls
            and all(load_receipts_valid)
            and all(retrieval_receipts_valid)
            else "failure"
        ),
        "source": source,
        "assets": {
            "embedding_model": EMBED_MODEL,
            "embedding_model_revision": EMBED_MODEL_REVISION,
            "generation_model": GENERATION_MODEL,
            "generation_model_revision": GENERATION_MODEL_REVISION,
            "generation_model_digest": GENERATION_MODEL_DIGEST,
            "model_digest": MODEL_DIGEST,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "schemen_gate": metadata.version("schemen-gate"),
            "execution_preflight": "schemen-gate/research-execution-preflight-v1",
        },
        "owning_exact_recall": owning_recall,
        "all_unauthorized_docks_rejected": all_rejected,
        "unauthorized_model_calls": sum(
            row["model_calls"] for row in rejection_rows
        ),
        "load_receipts_valid": all(load_receipts_valid),
        "retrieval_receipts_valid": all(retrieval_receipts_valid),
        "owning_rows": owning_rows,
        "rejection_rows": rejection_rows,
        "threat_boundary": (
            "The vector store, complete generator, and master key remain "
            "trusted runtime assets. Corpus release passes through RegimeBus; "
            "generator invocation separately passes through the bundled "
            "research execution preflight. The released Gate binds "
            "tenant + subject + regime + model + operation + policy + exact "
            "partition. This experiment does not include an IdP, policy "
            "authority, durable use ledger, or root-key custody deployment."
        ),
    }
    remote_path = Path("/results") / f"cargo_transformer_{timestamp()}.json"
    remote_path.write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    result["remote_artifact"] = str(remote_path)
    return result


@app.local_entrypoint()
def main() -> None:
    started = time.perf_counter()
    result = run_test.remote(source_metadata())
    artifact = {
        "schema_version": 3,
        "experiment": "cargo_transformer_corpus_authorization_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "result": result,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"cargo_transformer_{timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
