"""Schemen Gate cryptographic masks, authority tokens, and lockboxes.

Use this standalone package to integrate an identity-bound Gate into NumPy or
an AI framework without a model server or control-plane dependency.

Quick start with pre-computed masks (zero crypto dependency)::

    from schemen_gate import GateMask

    mask = GateMask.from_file("masks/regime_0.npy")
    gated = mask.apply(hidden_activations)

Deterministic mask derivation is available in the NumPy-only core::

    from schemen_gate import GateMask

    mask = GateMask.derive(key_bytes, regime_id=0, n_dims=768, n_regimes=4)

Encrypted tokens and signatures require
``pip install schemen-gate[crypto]``::

    from schemen_gate import GateKey, issue_adapter_token

    token = issue_adapter_token(key, "tenant", 0, 768, 4, weights, ...)

With lockbox (``pip install schemen-gate[lockbox]``)::

    from schemen_gate import create_lockbox, seal_grant, sign_lockbox

Token-based operations (adapter tokens, mask tokens, gate rights) require
the ``[crypto]`` extra. The authenticated contract fields are bound as AAD;
any mismatch causes cryptographic verification to fail.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Keep the runtime package import lightweight while presenting the complete,
    # concrete public API to static type checkers.  These imports are deliberately
    # guarded: several modules require optional dependencies at runtime.
    from schemen_gate._cargo import (
        CargoAuthenticationError,
        CargoError,
        CargoExpiredError,
        CargoItem,
        CargoManifest,
        CargoManifestOperation,
        CargoOperation,
        CargoPayloadKind,
        CargoReceipt,
        CargoSessionClosedError,
        CompletionCondition,
        CompletionKind,
        DockingSession,
        EmbeddingSpec,
        InferenceStream,
        InferenceStreamFrame,
        LoadResult,
        OperationKind,
        RegimeBus,
        UnloadResult,
        VectorBridge,
        VectorPayload,
        compute_items_hash,
        compute_payload_hash,
        hash_vocabulary,
    )
    from schemen_gate._cargo_impl import (
        BusTerminal,
        DefaultRegimeBus,
        LiveDockingSession,
        create_manifest,
        derive_cargo_access_key,
    )
    from schemen_gate._crypto import derive_gate_mask_raw, derive_partition
    from schemen_gate._key_wrapping import KeyWrapper, WrappedKey, X25519AESGCMWrapper
    from schemen_gate._lockbox import (
        AccessLevel,
        Authority,
        Grant,
        GrantSpec,
        HierarchyDef,
        InMemoryKeyProvider,
        KeyProvider,
        Lockbox,
        ProvenanceResult,
        RegimeCapability,
        RevocationCheck,
        RevocationPolicy,
        SealedKey,
        SerializedToken,
        SpiffeWorkloadClient,
        WorkloadIdentity,
        attest_model,
        bind_model_artifact,
        check_certificate_revocation,
        compute_access_fingerprint,
        compute_lockbox_hash,
        compute_model_artifact_hash,
        compute_model_graph_hash,
        create_lockbox,
        extract_spiffe_id,
        fingerprint_from_x509,
        generate_authority_cert,
        generate_self_signed_x25519_cert,
        generate_svid,
        load_lockbox,
        public_key_from_x509,
        reconstitute_gate_key,
        reissue_lockbox,
        resolve_access,
        resolve_access_by_scopes,
        save_lockbox,
        seal_grant,
        seal_grant_spiffe,
        sign_lockbox,
        split_gate_key,
        trust_bundle_fingerprints,
        unseal_grant,
        validate_hierarchy,
        validate_svid,
        verify_attestation_spiffe,
        verify_authority,
        verify_grant_provenance,
        verify_model_attestation,
        verify_provenance_spiffe,
    )
    from schemen_gate._operation_gate import (
        LEARNED_ROUTE,
        NATIVE_ROUTE,
        OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA,
        OPERATION_GATE_REDEMPTION_SCHEMA,
        OPERATION_GATE_TOKEN_SCHEMA,
        OPERATION_TRANSITION_AAD_SCHEMA,
        OperationGateAttenuationError,
        OperationGateError,
        OperationGatePublicAttestation,
        OperationGateRedemptionReceipt,
        OperationGateReplayError,
        OperationGateToken,
        OperationGateVerifier,
        OperationProposalOrigin,
        OperationTransitionAAD,
        authenticate_operation_gate,
        derive_operation_gate_key,
        derive_subordinate_operation_key,
        issue_operation_gate,
        issue_operation_subgate,
        sign_operation_redemption,
        verify_operation_public_attestation,
        verify_operation_public_attestation_self_consistency,
        verify_operation_redemption,
    )
    from schemen_gate._pkcs12 import Pkcs12KeyProvider
    from schemen_gate._rag import (
        CachePolicy,
        GatedRAGAdapter,
        GatedRetrievalResult,
        InMemoryVectorStore,
        PartitionMap,
        PartitionMode,
        PartitionModeError,
        PartitionNotFoundError,
        RetrievedDoc,
        VectorStore,
    )
    from schemen_gate._rag_arch import (
        ArchitectureSpec,
        CompressionResult,
        PartitionAnalysis,
        analyze_vectors,
        compress_vectors,
    )
    from schemen_gate._rag_stores import PgVectorStore
    from schemen_gate._regime0_fold import (
        FoldedRepresentation,
        fold_matrix,
        fold_vector,
        reconstruction_quality,
        unfold_matrix,
        unfold_vector,
    )
    from schemen_gate._tokens import (
        AdapterToken,
        DerivedKey,
        GateKey,
        GateRights,
        MaskToken,
        RegimePermissionError,
        SchemenTokenError,
        StoreCapacityError,
        TokenAuthenticationError,
        TokenExpiredError,
        WeightIntegrityError,
        adapter_topology_string,
        canonical_permissions,
        derive_regime_key,
        derive_tenant_key,
        hash_adapter_weights,
        hkdf_expand_sha256,
        issue_adapter_token,
        issue_mask_token,
        issue_use_only_token,
        redeem_adapter_token,
        redeem_mask_token,
    )
    from schemen_gate._vector_dispatch import DispatchResult, SkillFingerprint, SkillRegistry
    from schemen_gate._von import (
        QuantizationLevel,
        VONFrame,
    )
    from schemen_gate._von import (
        compression_ratio as von_compression_ratio,
    )
    from schemen_gate._von import (
        cosine_at_level as von_cosine_at_level,
    )
    from schemen_gate._von import (
        encode as von_encode,
    )
    from schemen_gate._von import (
        encode_batch as von_encode_batch,
    )
    from schemen_gate._von import (
        fidelity_curve as von_fidelity_curve,
    )
    from schemen_gate._von import (
        gate_levels as von_gate_levels,
    )
    from schemen_gate._von import (
        ranking_preservation as von_ranking_preservation,
    )
    from schemen_gate._von import (
        strip_levels as von_strip_levels,
    )
    from schemen_gate.capability import (
        AttestationReceipt,
        CapabilityToken,
        DelegationCertificate,
        PhaseGateReceipt,
        RevocationNotice,
        delegation_policy_verify_key,
        delegation_runtime_verify_key,
        derive_bound_policy_key,
        derive_policy_key,
        derive_policy_verify_key,
        make_nonce,
        sign_attestation,
        sign_capability,
        sign_delegation,
        sign_phase_gate,
        sign_revocation,
        verify_attestation,
        verify_attestation_self_consistency,
        verify_capability,
        verify_delegation,
        verify_delegation_self_consistency,
        verify_enforced_attestation,
        verify_enforced_delegation,
        verify_phase_gate,
        verify_revocation,
    )

from schemen_gate._release import (
    GATE_PACKAGE,
    GATE_RELEASE_IDENTITY_SCHEMA,
    GATE_SOURCE_REPOSITORY,
    GATE_VERSION,
    GateReleaseIdentity,
    ReleaseIdentityError,
    current_release_identity,
    release_identity_matches,
)

__version__ = GATE_VERSION

from schemen_gate._mask import GateMask

__all__ = [  # noqa: RUF022 - grouped public API is intentionally documented by subsystem
    "GateMask",
    "__version__",
    # Authenticated software release identity
    "GATE_PACKAGE",
    "GATE_RELEASE_IDENTITY_SCHEMA",
    "GATE_SOURCE_REPOSITORY",
    "GATE_VERSION",
    "GateReleaseIdentity",
    "ReleaseIdentityError",
    "current_release_identity",
    "release_identity_matches",
    # Crypto primitives (require [crypto] extra)
    "derive_partition",
    "derive_gate_mask_raw",
    # Error hierarchy
    "SchemenTokenError",
    "TokenAuthenticationError",
    "TokenExpiredError",
    "WeightIntegrityError",
    "RegimePermissionError",
    "StoreCapacityError",
    # Token types and operations (require [crypto] extra)
    "GateKey",
    "DerivedKey",
    "MaskToken",
    "AdapterToken",
    "GateRights",
    "hkdf_expand_sha256",
    "derive_regime_key",
    "derive_tenant_key",
    "issue_mask_token",
    "redeem_mask_token",
    "issue_adapter_token",
    "issue_use_only_token",
    "redeem_adapter_token",
    "canonical_permissions",
    "hash_adapter_weights",
    "adapter_topology_string",
    # Key wrapping (require [crypto] extra)
    "KeyWrapper",
    "WrappedKey",
    "X25519AESGCMWrapper",
    # Lockbox (require [lockbox] extra)
    "Lockbox",
    "AccessLevel",
    "RegimeCapability",
    "HierarchyDef",
    "Grant",
    "SealedKey",
    "SerializedToken",
    "Authority",
    "ProvenanceResult",
    "RevocationCheck",
    "RevocationPolicy",
    "KeyProvider",
    "InMemoryKeyProvider",
    "Pkcs12KeyProvider",
    "create_lockbox",
    "seal_grant",
    "unseal_grant",
    "sign_lockbox",
    "verify_authority",
    "check_certificate_revocation",
    "compute_lockbox_hash",
    "compute_access_fingerprint",
    "validate_hierarchy",
    "reissue_lockbox",
    "GrantSpec",
    "resolve_access",
    "resolve_access_by_scopes",
    "split_gate_key",
    "reconstitute_gate_key",
    "save_lockbox",
    "load_lockbox",
    "generate_authority_cert",
    "verify_grant_provenance",
    "extract_spiffe_id",
    "validate_svid",
    "generate_svid",
    "trust_bundle_fingerprints",
    "fingerprint_from_x509",
    "public_key_from_x509",
    "generate_self_signed_x25519_cert",
    "compute_model_artifact_hash",
    "compute_model_graph_hash",
    "bind_model_artifact",
    "attest_model",
    "verify_model_attestation",
    "seal_grant_spiffe",
    "verify_provenance_spiffe",
    "verify_attestation_spiffe",
    "WorkloadIdentity",
    "SpiffeWorkloadClient",
    # RAG adapter (require [rag] extra for PgVectorStore)
    "PartitionMode",
    "PartitionModeError",
    "PartitionNotFoundError",
    "CachePolicy",
    "RetrievedDoc",
    "GatedRetrievalResult",
    "VectorStore",
    "InMemoryVectorStore",
    "PartitionMap",
    "GatedRAGAdapter",
    # RAG architecture (require [rag] extra for compression analysis)
    "ArchitectureSpec",
    "PartitionAnalysis",
    "CompressionResult",
    "analyze_vectors",
    "compress_vectors",
    # RAG stores (require [rag] extra)
    "PgVectorStore",
    # Cargo Mode (require [crypto] extra)
    "EmbeddingSpec",
    "CompletionKind",
    "CompletionCondition",
    "CargoManifestOperation",
    "CargoPayloadKind",
    "CargoManifest",
    "CargoItem",
    "OperationKind",
    "CargoOperation",
    "LoadResult",
    "UnloadResult",
    "CargoReceipt",
    "DockingSession",
    "RegimeBus",
    "CargoError",
    "CargoAuthenticationError",
    "CargoExpiredError",
    "CargoSessionClosedError",
    "compute_items_hash",
    "compute_payload_hash",
    "hash_vocabulary",
    # Cargo Mode vector bridge (direct vector-to-vector comms)
    "VectorBridge",
    "VectorPayload",
    # Inference stream (vector transport to GPU backbones)
    "InferenceStreamFrame",
    "InferenceStream",
    # Cargo Mode implementation (require [crypto] extra)
    "DefaultRegimeBus",
    "LiveDockingSession",
    "BusTerminal",
    "create_manifest",
    "derive_cargo_access_key",
    # Vector dispatch (semantic skill routing)
    "SkillFingerprint",
    "DispatchResult",
    "SkillRegistry",
    # Lossless row folding (encoding only; not a security boundary)
    "FoldedRepresentation",
    "fold_vector",
    "unfold_vector",
    "fold_matrix",
    "unfold_matrix",
    "reconstruction_quality",
    # VON -- Vector Object Notation (precision-gated transport)
    "VONFrame",
    "QuantizationLevel",
    "von_encode",
    "von_encode_batch",
    "von_cosine_at_level",
    "von_fidelity_curve",
    "von_ranking_preservation",
    "von_strip_levels",
    "von_gate_levels",
    "von_compression_ratio",
    # Capability attestation (require [crypto] extra)
    "DelegationCertificate",
    "CapabilityToken",
    "AttestationReceipt",
    "PhaseGateReceipt",
    "RevocationNotice",
    "sign_delegation",
    "verify_delegation",
    "verify_delegation_self_consistency",
    "verify_enforced_delegation",
    "sign_capability",
    "verify_capability",
    "make_nonce",
    "sign_attestation",
    "verify_attestation",
    "verify_attestation_self_consistency",
    "verify_enforced_attestation",
    "sign_phase_gate",
    "verify_phase_gate",
    "sign_revocation",
    "verify_revocation",
    "derive_policy_key",
    "derive_policy_verify_key",
    "derive_bound_policy_key",
    "delegation_policy_verify_key",
    "delegation_runtime_verify_key",
    # Exact finite-operation gates (require [crypto] extra)
    "LEARNED_ROUTE",
    "NATIVE_ROUTE",
    "OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA",
    "OPERATION_GATE_REDEMPTION_SCHEMA",
    "OPERATION_GATE_TOKEN_SCHEMA",
    "OPERATION_TRANSITION_AAD_SCHEMA",
    "OperationGateAttenuationError",
    "OperationGateError",
    "OperationGateRedemptionReceipt",
    "OperationGatePublicAttestation",
    "OperationGateReplayError",
    "OperationGateToken",
    "OperationGateVerifier",
    "OperationProposalOrigin",
    "OperationTransitionAAD",
    "authenticate_operation_gate",
    "derive_operation_gate_key",
    "derive_subordinate_operation_key",
    "issue_operation_gate",
    "issue_operation_subgate",
    "sign_operation_redemption",
    "verify_operation_public_attestation",
    "verify_operation_public_attestation_self_consistency",
    "verify_operation_redemption",
]


# Public names served lazily so that ``import schemen_gate`` stays NumPy-only.
# Each entry maps an exported name to ``(module, attribute)``; the table is
# reconciled with ``__all__`` by tests/test_public_api_surface.py.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # _crypto
    "derive_gate_mask_raw": ("_crypto", "derive_gate_mask_raw"),
    "derive_partition": ("_crypto", "derive_partition"),
    # _tokens
    "AdapterToken": ("_tokens", "AdapterToken"),
    "DerivedKey": ("_tokens", "DerivedKey"),
    "GateKey": ("_tokens", "GateKey"),
    "GateRights": ("_tokens", "GateRights"),
    "MaskToken": ("_tokens", "MaskToken"),
    "RegimePermissionError": ("_tokens", "RegimePermissionError"),
    "SchemenTokenError": ("_tokens", "SchemenTokenError"),
    "StoreCapacityError": ("_tokens", "StoreCapacityError"),
    "TokenAuthenticationError": ("_tokens", "TokenAuthenticationError"),
    "TokenExpiredError": ("_tokens", "TokenExpiredError"),
    "WeightIntegrityError": ("_tokens", "WeightIntegrityError"),
    "adapter_topology_string": ("_tokens", "adapter_topology_string"),
    "canonical_permissions": ("_tokens", "canonical_permissions"),
    "derive_regime_key": ("_tokens", "derive_regime_key"),
    "derive_tenant_key": ("_tokens", "derive_tenant_key"),
    "hash_adapter_weights": ("_tokens", "hash_adapter_weights"),
    "hkdf_expand_sha256": ("_tokens", "hkdf_expand_sha256"),
    "issue_adapter_token": ("_tokens", "issue_adapter_token"),
    "issue_mask_token": ("_tokens", "issue_mask_token"),
    "issue_use_only_token": ("_tokens", "issue_use_only_token"),
    "redeem_adapter_token": ("_tokens", "redeem_adapter_token"),
    "redeem_mask_token": ("_tokens", "redeem_mask_token"),
    # _key_wrapping
    "KeyWrapper": ("_key_wrapping", "KeyWrapper"),
    "WrappedKey": ("_key_wrapping", "WrappedKey"),
    "X25519AESGCMWrapper": ("_key_wrapping", "X25519AESGCMWrapper"),
    # _lockbox
    "AccessLevel": ("_lockbox", "AccessLevel"),
    "Authority": ("_lockbox", "Authority"),
    "Grant": ("_lockbox", "Grant"),
    "GrantSpec": ("_lockbox", "GrantSpec"),
    "HierarchyDef": ("_lockbox", "HierarchyDef"),
    "InMemoryKeyProvider": ("_lockbox", "InMemoryKeyProvider"),
    "KeyProvider": ("_lockbox", "KeyProvider"),
    "Lockbox": ("_lockbox", "Lockbox"),
    "ProvenanceResult": ("_lockbox", "ProvenanceResult"),
    "RegimeCapability": ("_lockbox", "RegimeCapability"),
    "RevocationCheck": ("_lockbox", "RevocationCheck"),
    "RevocationPolicy": ("_lockbox", "RevocationPolicy"),
    "SealedKey": ("_lockbox", "SealedKey"),
    "SerializedToken": ("_lockbox", "SerializedToken"),
    "SpiffeWorkloadClient": ("_lockbox", "SpiffeWorkloadClient"),
    "WorkloadIdentity": ("_lockbox", "WorkloadIdentity"),
    "attest_model": ("_lockbox", "attest_model"),
    "bind_model_artifact": ("_lockbox", "bind_model_artifact"),
    "check_certificate_revocation": ("_lockbox", "check_certificate_revocation"),
    "compute_access_fingerprint": ("_lockbox", "compute_access_fingerprint"),
    "compute_lockbox_hash": ("_lockbox", "compute_lockbox_hash"),
    "compute_model_artifact_hash": ("_lockbox", "compute_model_artifact_hash"),
    "compute_model_graph_hash": ("_lockbox", "compute_model_graph_hash"),
    "create_lockbox": ("_lockbox", "create_lockbox"),
    "extract_spiffe_id": ("_lockbox", "extract_spiffe_id"),
    "fingerprint_from_x509": ("_lockbox", "fingerprint_from_x509"),
    "generate_authority_cert": ("_lockbox", "generate_authority_cert"),
    "generate_self_signed_x25519_cert": ("_lockbox", "generate_self_signed_x25519_cert"),
    "generate_svid": ("_lockbox", "generate_svid"),
    "load_lockbox": ("_lockbox", "load_lockbox"),
    "public_key_from_x509": ("_lockbox", "public_key_from_x509"),
    "reconstitute_gate_key": ("_lockbox", "reconstitute_gate_key"),
    "reissue_lockbox": ("_lockbox", "reissue_lockbox"),
    "resolve_access": ("_lockbox", "resolve_access"),
    "resolve_access_by_scopes": ("_lockbox", "resolve_access_by_scopes"),
    "save_lockbox": ("_lockbox", "save_lockbox"),
    "seal_grant": ("_lockbox", "seal_grant"),
    "seal_grant_spiffe": ("_lockbox", "seal_grant_spiffe"),
    "sign_lockbox": ("_lockbox", "sign_lockbox"),
    "split_gate_key": ("_lockbox", "split_gate_key"),
    "trust_bundle_fingerprints": ("_lockbox", "trust_bundle_fingerprints"),
    "unseal_grant": ("_lockbox", "unseal_grant"),
    "validate_hierarchy": ("_lockbox", "validate_hierarchy"),
    "validate_svid": ("_lockbox", "validate_svid"),
    "verify_attestation_spiffe": ("_lockbox", "verify_attestation_spiffe"),
    "verify_authority": ("_lockbox", "verify_authority"),
    "verify_grant_provenance": ("_lockbox", "verify_grant_provenance"),
    "verify_model_attestation": ("_lockbox", "verify_model_attestation"),
    "verify_provenance_spiffe": ("_lockbox", "verify_provenance_spiffe"),
    # _pkcs12
    "Pkcs12KeyProvider": ("_pkcs12", "Pkcs12KeyProvider"),
    # _rag
    "CachePolicy": ("_rag", "CachePolicy"),
    "GatedRAGAdapter": ("_rag", "GatedRAGAdapter"),
    "GatedRetrievalResult": ("_rag", "GatedRetrievalResult"),
    "InMemoryVectorStore": ("_rag", "InMemoryVectorStore"),
    "PartitionMap": ("_rag", "PartitionMap"),
    "PartitionMode": ("_rag", "PartitionMode"),
    "PartitionModeError": ("_rag", "PartitionModeError"),
    "PartitionNotFoundError": ("_rag", "PartitionNotFoundError"),
    "RetrievedDoc": ("_rag", "RetrievedDoc"),
    "VectorStore": ("_rag", "VectorStore"),
    # _rag_arch
    "ArchitectureSpec": ("_rag_arch", "ArchitectureSpec"),
    "CompressionResult": ("_rag_arch", "CompressionResult"),
    "PartitionAnalysis": ("_rag_arch", "PartitionAnalysis"),
    "analyze_vectors": ("_rag_arch", "analyze_vectors"),
    "compress_vectors": ("_rag_arch", "compress_vectors"),
    # _rag_stores
    "PgVectorStore": ("_rag_stores", "PgVectorStore"),
    # _cargo
    "CargoAuthenticationError": ("_cargo", "CargoAuthenticationError"),
    "CargoError": ("_cargo", "CargoError"),
    "CargoExpiredError": ("_cargo", "CargoExpiredError"),
    "CargoItem": ("_cargo", "CargoItem"),
    "CargoManifest": ("_cargo", "CargoManifest"),
    "CargoManifestOperation": ("_cargo", "CargoManifestOperation"),
    "CargoOperation": ("_cargo", "CargoOperation"),
    "CargoPayloadKind": ("_cargo", "CargoPayloadKind"),
    "CargoReceipt": ("_cargo", "CargoReceipt"),
    "CargoSessionClosedError": ("_cargo", "CargoSessionClosedError"),
    "CompletionCondition": ("_cargo", "CompletionCondition"),
    "CompletionKind": ("_cargo", "CompletionKind"),
    "DockingSession": ("_cargo", "DockingSession"),
    "EmbeddingSpec": ("_cargo", "EmbeddingSpec"),
    "InferenceStream": ("_cargo", "InferenceStream"),
    "InferenceStreamFrame": ("_cargo", "InferenceStreamFrame"),
    "LoadResult": ("_cargo", "LoadResult"),
    "OperationKind": ("_cargo", "OperationKind"),
    "RegimeBus": ("_cargo", "RegimeBus"),
    "UnloadResult": ("_cargo", "UnloadResult"),
    "VectorBridge": ("_cargo", "VectorBridge"),
    "VectorPayload": ("_cargo", "VectorPayload"),
    "compute_items_hash": ("_cargo", "compute_items_hash"),
    "compute_payload_hash": ("_cargo", "compute_payload_hash"),
    "hash_vocabulary": ("_cargo", "hash_vocabulary"),
    # _cargo_impl
    "BusTerminal": ("_cargo_impl", "BusTerminal"),
    "DefaultRegimeBus": ("_cargo_impl", "DefaultRegimeBus"),
    "LiveDockingSession": ("_cargo_impl", "LiveDockingSession"),
    "create_manifest": ("_cargo_impl", "create_manifest"),
    "derive_cargo_access_key": ("_cargo_impl", "derive_cargo_access_key"),
    # _vector_dispatch
    "DispatchResult": ("_vector_dispatch", "DispatchResult"),
    "SkillFingerprint": ("_vector_dispatch", "SkillFingerprint"),
    "SkillRegistry": ("_vector_dispatch", "SkillRegistry"),
    # _regime0_fold
    "FoldedRepresentation": ("_regime0_fold", "FoldedRepresentation"),
    "fold_matrix": ("_regime0_fold", "fold_matrix"),
    "fold_vector": ("_regime0_fold", "fold_vector"),
    "reconstruction_quality": ("_regime0_fold", "reconstruction_quality"),
    "unfold_matrix": ("_regime0_fold", "unfold_matrix"),
    "unfold_vector": ("_regime0_fold", "unfold_vector"),
    # _von
    "QuantizationLevel": ("_von", "QuantizationLevel"),
    "VONFrame": ("_von", "VONFrame"),
    "von_compression_ratio": ("_von", "compression_ratio"),
    "von_cosine_at_level": ("_von", "cosine_at_level"),
    "von_encode": ("_von", "encode"),
    "von_encode_batch": ("_von", "encode_batch"),
    "von_fidelity_curve": ("_von", "fidelity_curve"),
    "von_gate_levels": ("_von", "gate_levels"),
    "von_ranking_preservation": ("_von", "ranking_preservation"),
    "von_strip_levels": ("_von", "strip_levels"),
    # capability
    "AttestationReceipt": ("capability", "AttestationReceipt"),
    "CapabilityToken": ("capability", "CapabilityToken"),
    "DelegationCertificate": ("capability", "DelegationCertificate"),
    "PhaseGateReceipt": ("capability", "PhaseGateReceipt"),
    "RevocationNotice": ("capability", "RevocationNotice"),
    "delegation_policy_verify_key": ("capability", "delegation_policy_verify_key"),
    "delegation_runtime_verify_key": ("capability", "delegation_runtime_verify_key"),
    "derive_bound_policy_key": ("capability", "derive_bound_policy_key"),
    "derive_policy_key": ("capability", "derive_policy_key"),
    "derive_policy_verify_key": ("capability", "derive_policy_verify_key"),
    "make_nonce": ("capability", "make_nonce"),
    "sign_attestation": ("capability", "sign_attestation"),
    "sign_capability": ("capability", "sign_capability"),
    "sign_delegation": ("capability", "sign_delegation"),
    "sign_phase_gate": ("capability", "sign_phase_gate"),
    "sign_revocation": ("capability", "sign_revocation"),
    "verify_attestation": ("capability", "verify_attestation"),
    "verify_attestation_self_consistency": ("capability", "verify_attestation_self_consistency"),
    "verify_capability": ("capability", "verify_capability"),
    "verify_delegation": ("capability", "verify_delegation"),
    "verify_delegation_self_consistency": ("capability", "verify_delegation_self_consistency"),
    "verify_enforced_attestation": ("capability", "verify_enforced_attestation"),
    "verify_enforced_delegation": ("capability", "verify_enforced_delegation"),
    "verify_phase_gate": ("capability", "verify_phase_gate"),
    "verify_revocation": ("capability", "verify_revocation"),
    # _operation_gate
    "LEARNED_ROUTE": ("_operation_gate", "LEARNED_ROUTE"),
    "NATIVE_ROUTE": ("_operation_gate", "NATIVE_ROUTE"),
    "OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA": (
        "_operation_gate",
        "OPERATION_GATE_PUBLIC_ATTESTATION_SCHEMA",
    ),
    "OPERATION_GATE_REDEMPTION_SCHEMA": ("_operation_gate", "OPERATION_GATE_REDEMPTION_SCHEMA"),
    "OPERATION_GATE_TOKEN_SCHEMA": ("_operation_gate", "OPERATION_GATE_TOKEN_SCHEMA"),
    "OPERATION_TRANSITION_AAD_SCHEMA": ("_operation_gate", "OPERATION_TRANSITION_AAD_SCHEMA"),
    "OperationGateAttenuationError": ("_operation_gate", "OperationGateAttenuationError"),
    "OperationGateError": ("_operation_gate", "OperationGateError"),
    "OperationGatePublicAttestation": ("_operation_gate", "OperationGatePublicAttestation"),
    "OperationGateRedemptionReceipt": ("_operation_gate", "OperationGateRedemptionReceipt"),
    "OperationGateReplayError": ("_operation_gate", "OperationGateReplayError"),
    "OperationGateToken": ("_operation_gate", "OperationGateToken"),
    "OperationGateVerifier": ("_operation_gate", "OperationGateVerifier"),
    "OperationProposalOrigin": ("_operation_gate", "OperationProposalOrigin"),
    "OperationTransitionAAD": ("_operation_gate", "OperationTransitionAAD"),
    "authenticate_operation_gate": ("_operation_gate", "authenticate_operation_gate"),
    "derive_operation_gate_key": ("_operation_gate", "derive_operation_gate_key"),
    "derive_subordinate_operation_key": ("_operation_gate", "derive_subordinate_operation_key"),
    "issue_operation_gate": ("_operation_gate", "issue_operation_gate"),
    "issue_operation_subgate": ("_operation_gate", "issue_operation_subgate"),
    "sign_operation_redemption": ("_operation_gate", "sign_operation_redemption"),
    "verify_operation_public_attestation": (
        "_operation_gate",
        "verify_operation_public_attestation",
    ),
    "verify_operation_public_attestation_self_consistency": (
        "_operation_gate",
        "verify_operation_public_attestation_self_consistency",
    ),
    "verify_operation_redemption": ("_operation_gate", "verify_operation_redemption"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load optional modules to keep the base install lightweight."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'schemen_gate' has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(f"schemen_gate.{module_name}")
    return getattr(module, attribute)
