"""Partition mutation and identifier admission regressions."""

from __future__ import annotations

import numpy as np
import pytest

from schemen_gate import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
    PartitionModeError,
)


def test_immutable_partition_rejects_destruction_without_touching_store():
    store = InMemoryVectorStore()
    store.insert(np.ones(4), "retained", "partition", {"doc_id": "existing"})
    partitions = PartitionMap(b"k" * 32, n_dims=4, n_regimes=1)
    partitions.register("partition", regime_id=0)
    adapter = GatedRAGAdapter(store, partitions)
    with pytest.raises(PartitionModeError, match="IMMUTABLE"):
        adapter.destroy("partition")
    assert store.count("partition") == 1


@pytest.mark.parametrize("mode", [PartitionMode.FRESH, PartitionMode.READ_WRITE])
def test_writable_partition_can_be_destroyed(mode):
    store = InMemoryVectorStore()
    store.insert(np.ones(4), "retained", "partition")
    partitions = PartitionMap(b"k" * 32, n_dims=4, n_regimes=1)
    partitions.register("partition", regime_id=0, mode=mode)
    GatedRAGAdapter(store, partitions).destroy("partition")
    assert store.count("partition") == 0


def test_existing_document_id_rejected_atomically_across_batches():
    store = InMemoryVectorStore()
    store.insert(np.ones(4), "original", "partition", {"doc_id": "existing"})
    with pytest.raises(ValueError, match="document ids"):
        store.insert_many(
            [
                (np.ones(4), "new", {"doc_id": "new"}),
                (np.ones(4), "collision", {"doc_id": "existing"}),
            ],
            "partition",
        )
    assert store.count("partition") == 1
    assert store.retrieve(np.ones(4), "partition")[0].content == "original"
    # IDs remain partition-local; another partition is unaffected.
    assert store.insert(np.ones(4), "separate", "other", {"doc_id": "existing"}) == "existing"
