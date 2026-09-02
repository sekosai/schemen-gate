"""Capability token signing, verification, and key-derivation tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from schemen_gate.capability import (
    AttestationReceipt,
    CapabilityToken,
    DelegationCertificate,
    PhaseGateReceipt,
    RevocationNotice,
    _public_key_hex,
    canonicalize,
    delegation_policy_verify_key,
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def da_key():
    """Deterministic DA signing key from fixed seed."""
    from schemen_gate import hkdf_expand_sha256

    seed = b"test-gate-seed-32-bytes-long!!!!!"[:32]
    material = hkdf_expand_sha256(seed, b"test-da-key:v1", length=32)
    return Ed25519PrivateKey.from_private_bytes(material)


@pytest.fixture
def other_key():
    """A different Ed25519 key for wrong-key tests."""
    from schemen_gate import hkdf_expand_sha256

    seed = b"other-seed-32-bytes-long!!!!!!!!"[:32]
    material = hkdf_expand_sha256(seed, b"test-other:v1", length=32)
    return Ed25519PrivateKey.from_private_bytes(material)


@pytest.fixture
def runtime_key():
    """Simulated runtime signing key (same library, different key)."""
    from schemen_gate import hkdf_expand_sha256

    seed = b"runtime-seed-32-bytes-long!!!!!!"[:32]
    material = hkdf_expand_sha256(seed, b"test-runtime:v1", length=32)
    return Ed25519PrivateKey.from_private_bytes(material)


@pytest.fixture
def sample_delegation(da_key):
    """An unsigned DelegationCertificate with realistic fields."""
    now = datetime.now(timezone.utc)
    return DelegationCertificate(
        mission_id="test-mission-001",
        principal="release-test@example.invalid",
        granted_phases=["plan", "code", "verify"],
        granted_tools=["read_file", "write_file", "run_command", "edit_file"],
        granted_paths=["src/", "tests/", "PLAN.md"],
        gate_rights={
            "can_use": True,
            "can_update": True,
            "can_delegate": False,
            "can_destroy": False,
        },
        max_iterations=50,
        issued_at=now.isoformat(),
        expires_at=(now + timedelta(hours=2)).isoformat(),
        da_public_key=_public_key_hex(da_key.public_key()),
    )


@pytest.fixture
def signed_delegation(sample_delegation, da_key):
    return sign_delegation(sample_delegation, da_key)


@pytest.fixture
def policy_key(signed_delegation):
    return derive_policy_key(signed_delegation.signature)


@pytest.fixture
def sample_capability(signed_delegation):
    return CapabilityToken(
        action="write_file",
        target="src/foo.py",
        phase="code",
        iteration=1,
        nonce=make_nonce(),
        delegation_ref=signed_delegation.ref(),
        constraints={"max_bytes": 10000},
        issued_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def signed_capability(sample_capability, policy_key):
    return sign_capability(sample_capability, policy_key)


# ===================================================================
# DelegationCertificate (8 tests)
# ===================================================================


class TestDelegationCertificate:
    def test_canonical_serialization_deterministic(self, sample_delegation):
        d1 = sample_delegation.to_dict()
        d2 = {k: d1[k] for k in reversed(d1)}
        assert canonicalize(d1) == canonicalize(d2)

    def test_sign_verify_roundtrip(self, signed_delegation):
        assert signed_delegation.signature
        assert verify_delegation(
            signed_delegation,
            expected_da_public_key=signed_delegation.da_public_key,
        )
        assert verify_delegation_self_consistency(signed_delegation)

    def test_tamper_mission_id_rejected(self, signed_delegation):
        d = signed_delegation.to_dict()
        d["mission_id"] = "tampered-mission"
        tampered = DelegationCertificate.from_dict(d)
        assert not verify_delegation(
            tampered, expected_da_public_key=signed_delegation.da_public_key
        )

    def test_tamper_granted_paths_rejected(self, signed_delegation):
        d = signed_delegation.to_dict()
        d["granted_paths"].append("/etc/shadow")
        tampered = DelegationCertificate.from_dict(d)
        assert not verify_delegation(
            tampered, expected_da_public_key=signed_delegation.da_public_key
        )

    def test_wrong_key_rejected(self, sample_delegation, other_key):
        signed = sign_delegation(sample_delegation, other_key)
        # signature is valid for other_key but da_public_key points to da_key
        assert not verify_delegation(signed, expected_da_public_key=sample_delegation.da_public_key)

    def test_expired_delegation_fields(self, signed_delegation):
        """The enforced verifier checks expiry; serialization preserves it."""
        d = signed_delegation.to_dict()
        rt = DelegationCertificate.from_dict(d)
        assert rt.expires_at == signed_delegation.expires_at

    def test_serialization_roundtrip(self, signed_delegation):
        d = signed_delegation.to_dict()
        rt = DelegationCertificate.from_dict(d)
        assert rt == signed_delegation

    def test_signature_is_hex_ed25519(self, signed_delegation):
        sig = signed_delegation.signature
        assert len(sig) == 128  # 64 bytes = 128 hex chars
        bytes.fromhex(sig)  # must be valid hex


# ===================================================================
# CapabilityToken (7 tests)
# ===================================================================


class TestCapabilityToken:
    def test_sign_verify_roundtrip(self, signed_capability, policy_key):
        pub = policy_key.public_key()
        assert signed_capability.signature
        assert verify_capability(signed_capability, pub)

    def test_tamper_action_rejected(self, signed_capability, policy_key):
        d = signed_capability.to_dict()
        d["action"] = "run_command"
        tampered = CapabilityToken.from_dict(d)
        assert not verify_capability(tampered, policy_key.public_key())

    def test_tamper_target_rejected(self, signed_capability, policy_key):
        d = signed_capability.to_dict()
        d["target"] = "/etc/passwd"
        tampered = CapabilityToken.from_dict(d)
        assert not verify_capability(tampered, policy_key.public_key())

    def test_wrong_key_rejected(self, signed_capability, other_key):
        assert not verify_capability(signed_capability, other_key.public_key())

    def test_nonce_uniqueness(self):
        n1 = make_nonce()
        n2 = make_nonce()
        assert n1 != n2
        assert len(n1) == 32  # 16 bytes = 32 hex chars

    def test_delegation_ref_is_sha256(self, signed_delegation, sample_capability):
        expected_ref = signed_delegation.ref()
        assert sample_capability.delegation_ref == expected_ref
        assert len(expected_ref) == 64  # SHA-256 hex

    def test_serialization_roundtrip(self, signed_capability):
        d = signed_capability.to_dict()
        rt = CapabilityToken.from_dict(d)
        assert rt == signed_capability


# ===================================================================
# AttestationReceipt (5 tests)
# ===================================================================


class TestAttestationReceipt:
    @pytest.mark.parametrize("missing", ["success", "executed_at"])
    def test_parse_rejects_missing_execution_contract(
        self, signed_capability, runtime_key, missing
    ):
        receipt = sign_attestation(
            AttestationReceipt(
                action="write_file",
                target="src/foo.py",
                capability_ref=signed_capability.ref(),
                result_hash=hashlib.sha256(b"content").hexdigest(),
                success=True,
                executed_at=datetime.now(timezone.utc).isoformat(),
            ),
            runtime_key,
        )
        document = receipt.to_dict()
        document.pop(missing)
        with pytest.raises(ValueError, match="missing or unknown"):
            AttestationReceipt.from_dict(document)

    def test_verification_requires_external_runtime_key(self, signed_capability, runtime_key):
        receipt = sign_attestation(
            AttestationReceipt(
                action="write_file",
                target="src/foo.py",
                capability_ref=signed_capability.ref(),
                result_hash=hashlib.sha256(b"content").hexdigest(),
                success=True,
                executed_at=datetime.now(timezone.utc).isoformat(),
            ),
            runtime_key,
        )
        with pytest.raises(TypeError):
            verify_attestation(receipt)  # type: ignore[call-arg]

    def test_sign_verify_roundtrip(self, signed_capability, runtime_key):
        receipt = AttestationReceipt(
            action="write_file",
            target="src/foo.py",
            capability_ref=signed_capability.ref(),
            result_hash=hashlib.sha256(b"file content").hexdigest(),
            content_hash=hashlib.sha256(b"file content").hexdigest(),
            success=True,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_attestation(receipt, runtime_key)
        assert signed.signature
        assert signed.runtime_public_key
        assert verify_attestation(signed, _public_key_hex(runtime_key.public_key()))
        assert verify_attestation_self_consistency(signed)

    def test_tamper_result_hash_rejected(self, signed_capability, runtime_key):
        receipt = AttestationReceipt(
            action="write_file",
            target="src/foo.py",
            capability_ref=signed_capability.ref(),
            result_hash=hashlib.sha256(b"real content").hexdigest(),
            success=True,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_attestation(receipt, runtime_key)
        d = signed.to_dict()
        d["result_hash"] = hashlib.sha256(b"fake content").hexdigest()
        tampered = AttestationReceipt.from_dict(d)
        assert not verify_attestation(tampered, _public_key_hex(runtime_key.public_key()))

    def test_capability_ref_matches(self, signed_capability, runtime_key):
        receipt = AttestationReceipt(
            action="write_file",
            target="src/foo.py",
            capability_ref=signed_capability.ref(),
            result_hash=hashlib.sha256(b"content").hexdigest(),
            success=True,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_attestation(receipt, runtime_key)
        assert signed.capability_ref == signed_capability.ref()

    def test_wrong_key_rejected(self, signed_capability, runtime_key, other_key):
        receipt = AttestationReceipt(
            action="write_file",
            target="src/foo.py",
            capability_ref=signed_capability.ref(),
            result_hash=hashlib.sha256(b"content").hexdigest(),
            success=True,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_attestation(receipt, runtime_key)
        d = signed.to_dict()
        d["runtime_public_key"] = _public_key_hex(other_key.public_key())
        tampered = AttestationReceipt.from_dict(d)
        assert not verify_attestation(tampered, _public_key_hex(runtime_key.public_key()))

    def test_serialization_roundtrip(self, signed_capability, runtime_key):
        receipt = AttestationReceipt(
            action="write_file",
            target="src/foo.py",
            capability_ref=signed_capability.ref(),
            result_hash=hashlib.sha256(b"content").hexdigest(),
            success=True,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_attestation(receipt, runtime_key)
        d = signed.to_dict()
        rt = AttestationReceipt.from_dict(d)
        assert rt == signed


# ===================================================================
# PhaseGateReceipt (3 tests)
# ===================================================================


class TestPhaseGateReceipt:
    def test_sign_verify_roundtrip(self, policy_key):
        gate = PhaseGateReceipt(
            from_phase="plan",
            to_phase="code",
            required_attestations=["ref-abc123", "ref-def456"],
            satisfied_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_phase_gate(gate, policy_key)
        assert verify_phase_gate(signed, policy_key.public_key())

    def test_tamper_from_phase_rejected(self, policy_key):
        gate = PhaseGateReceipt(
            from_phase="plan",
            to_phase="code",
            required_attestations=["ref-abc123"],
            satisfied_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_phase_gate(gate, policy_key)
        d = signed.to_dict()
        d["from_phase"] = "verify"
        tampered = PhaseGateReceipt.from_dict(d)
        assert not verify_phase_gate(tampered, policy_key.public_key())

    def test_serialization_roundtrip(self, policy_key):
        gate = PhaseGateReceipt(
            from_phase="code",
            to_phase="verify",
            required_attestations=["ref-1", "ref-2"],
            satisfied_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_phase_gate(gate, policy_key)
        d = signed.to_dict()
        rt = PhaseGateReceipt.from_dict(d)
        assert rt == signed


# ===================================================================
# RevocationNotice (3 tests)
# ===================================================================


class TestRevocationNotice:
    def test_sign_verify_roundtrip(self, da_key):
        notice = RevocationNotice(
            scope="tool:write_file",
            reason="Security concern",
            issued_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_revocation(notice, da_key)
        assert verify_revocation(signed, da_key.public_key())

    def test_tamper_scope_rejected(self, da_key):
        notice = RevocationNotice(
            scope="tool:write_file",
            reason="Security concern",
            issued_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_revocation(notice, da_key)
        d = signed.to_dict()
        d["scope"] = "phase:code"
        tampered = RevocationNotice.from_dict(d)
        assert not verify_revocation(tampered, da_key.public_key())

    def test_serialization_roundtrip(self, da_key):
        notice = RevocationNotice(
            scope="capability:some-ref",
            reason="Revoking stale capability",
            issued_at=datetime.now(timezone.utc).isoformat(),
        )
        signed = sign_revocation(notice, da_key)
        d = signed.to_dict()
        rt = RevocationNotice.from_dict(d)
        assert rt == signed


# ===================================================================
# Key Derivation (4 tests)
# ===================================================================


class TestKeyDerivation:
    @pytest.mark.parametrize(
        ("authority_secret", "mission_id"),
        [
            (bytearray(b"a" * 32), "mission"),
            (b"a" * 32, True),
            (b"a" * 32, "mission\x00admin"),
        ],
    )
    def test_bound_policy_key_rejects_noncanonical_inputs(self, authority_secret, mission_id):
        with pytest.raises(ValueError):
            derive_bound_policy_key(authority_secret, mission_id)

    @staticmethod
    def _v2_delegation(
        sample_delegation,
        da_key,
        runtime_key,
        *,
        issued_at=None,
        expires_at=None,
        policy_public_key=None,
        runtime_public_key=None,
    ):
        policy_key = derive_bound_policy_key(
            b"test-authority-secret-material-32-bytes",
            sample_delegation.mission_id,
        )
        values = sample_delegation.to_dict()
        values.update(
            authority_version=2,
            policy_public_key=(policy_public_key or _public_key_hex(policy_key.public_key())),
            runtime_public_key=(runtime_public_key or _public_key_hex(runtime_key.public_key())),
            issued_at=issued_at or sample_delegation.issued_at,
            expires_at=expires_at or sample_delegation.expires_at,
            signature="",
        )
        return sign_delegation(DelegationCertificate.from_dict(values), da_key)

    def test_enforced_delegation_rejects_self_signed_rogue_da(
        self, sample_delegation, da_key, other_key, runtime_key
    ):
        policy_key = derive_bound_policy_key(
            b"rogue-authority-secret-material-32", sample_delegation.mission_id
        )
        values = sample_delegation.to_dict()
        values.update(
            da_public_key=_public_key_hex(other_key.public_key()),
            authority_version=2,
            policy_public_key=_public_key_hex(policy_key.public_key()),
            runtime_public_key=_public_key_hex(runtime_key.public_key()),
            signature="",
        )
        rogue = sign_delegation(DelegationCertificate.from_dict(values), other_key)

        assert verify_delegation_self_consistency(rogue)
        assert not verify_delegation(
            rogue,
            expected_da_public_key=_public_key_hex(da_key.public_key()),
        )
        assert not verify_enforced_delegation(
            rogue,
            expected_da_public_key=_public_key_hex(da_key.public_key()),
        )

    def test_delegation_verification_requires_external_da_key(self, signed_delegation):
        with pytest.raises(TypeError):
            verify_delegation(signed_delegation)  # type: ignore[call-arg]

    def test_canonicalization_rejects_non_json_objects(self):
        with pytest.raises(ValueError, match="canonical JSON"):
            canonicalize({"unsafe": object()})

    def test_v2_policy_key_requires_secret_authority_custody(
        self, sample_delegation, da_key, runtime_key
    ):
        secret = b"authority-secret-material-is-not-public"
        policy_key = derive_bound_policy_key(secret, sample_delegation.mission_id)
        d = sample_delegation.to_dict()
        d.update(
            authority_version=2,
            policy_public_key=_public_key_hex(policy_key.public_key()),
            runtime_public_key=_public_key_hex(runtime_key.public_key()),
            signature="",
        )
        signed = sign_delegation(DelegationCertificate.from_dict(d), da_key)

        assert verify_enforced_delegation(signed, expected_da_public_key=signed.da_public_key)
        assert delegation_policy_verify_key(
            signed, expected_da_public_key=signed.da_public_key
        ).public_bytes(Encoding.Raw, PublicFormat.Raw) == policy_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        # Public certificate material is insufficient to reproduce the key.
        legacy_public_derived = derive_policy_key(signed.signature)
        assert legacy_public_derived.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ) != policy_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def test_v1_delegation_is_readable_but_not_enforceable(self, signed_delegation):
        assert verify_delegation(
            signed_delegation,
            expected_da_public_key=signed_delegation.da_public_key,
        )
        assert not verify_enforced_delegation(
            signed_delegation,
            expected_da_public_key=signed_delegation.da_public_key,
        )

    def test_attestation_must_match_delegated_runtime(self, runtime_key, other_key):
        receipt = sign_attestation(
            AttestationReceipt(
                action="read_file",
                target="src/a.py",
                capability_ref="c" * 64,
                result_hash="d" * 64,
                success=True,
                executed_at=datetime.now(timezone.utc).isoformat(),
            ),
            runtime_key,
        )
        assert verify_attestation(receipt, _public_key_hex(runtime_key.public_key()))
        assert not verify_attestation(receipt, _public_key_hex(other_key.public_key()))

    def test_enforced_attestation_requires_external_runtime_pin(self, runtime_key, other_key):
        receipt = sign_attestation(
            AttestationReceipt(
                action="read_file",
                target="src/a.py",
                capability_ref="c" * 64,
                result_hash="d" * 64,
                success=True,
                executed_at=datetime.now(timezone.utc).isoformat(),
            ),
            runtime_key,
        )

        assert verify_enforced_attestation(
            receipt,
            expected_runtime_public_key=_public_key_hex(runtime_key.public_key()),
            expected_action="read_file",
            expected_target="src/a.py",
            expected_capability_ref="c" * 64,
            expected_success=True,
        )
        assert not verify_enforced_attestation(
            receipt,
            expected_runtime_public_key=_public_key_hex(other_key.public_key()),
            expected_action="read_file",
            expected_target="src/a.py",
            expected_capability_ref="c" * 64,
            expected_success=True,
        )
        assert not verify_enforced_attestation(
            receipt,
            expected_runtime_public_key="",
            expected_action="read_file",
            expected_target="src/a.py",
            expected_capability_ref="c" * 64,
            expected_success=True,
        )

    def test_enforced_delegation_rejects_expired_or_future_certificates(
        self, sample_delegation, da_key, runtime_key
    ):
        now = datetime.now(timezone.utc)
        expired = self._v2_delegation(
            sample_delegation,
            da_key,
            runtime_key,
            issued_at=(now - timedelta(hours=2)).isoformat(),
            expires_at=(now - timedelta(hours=1)).isoformat(),
        )
        future = self._v2_delegation(
            sample_delegation,
            da_key,
            runtime_key,
            issued_at=(now + timedelta(minutes=1)).isoformat(),
            expires_at=(now + timedelta(hours=1)).isoformat(),
        )

        for certificate in (expired, future):
            assert not verify_enforced_delegation(
                certificate,
                expected_da_public_key=certificate.da_public_key,
                now=now,
            )

    @pytest.mark.parametrize("duplicate", ["policy", "runtime"])
    def test_enforced_delegation_requires_three_distinct_authority_keys(
        self, sample_delegation, da_key, runtime_key, duplicate
    ):
        da_public_key = _public_key_hex(da_key.public_key())
        kwargs = {
            "policy_public_key": da_public_key if duplicate == "policy" else None,
            "runtime_public_key": da_public_key if duplicate == "runtime" else None,
        }
        certificate = self._v2_delegation(sample_delegation, da_key, runtime_key, **kwargs)

        assert not verify_enforced_delegation(
            certificate,
            expected_da_public_key=da_public_key,
        )

    def test_enforced_attestation_resolves_from_public_package(self):
        from schemen_gate import verify_enforced_attestation as public_api

        assert public_api is verify_enforced_attestation

    def test_policy_key_derived_deterministically(self, signed_delegation):
        k1 = derive_policy_key(signed_delegation.signature)
        k2 = derive_policy_key(signed_delegation.signature)
        pub1 = k1.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        pub2 = k2.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        assert pub1 == pub2

    def test_policy_key_different_delegations_differ(self, sample_delegation, da_key, other_key):
        # Sign the same delegation with two different keys → different signatures → different policy keys
        s1 = sign_delegation(sample_delegation, da_key)
        # Need a delegation with da_public_key matching other_key
        d = sample_delegation.to_dict()
        d["da_public_key"] = _public_key_hex(other_key.public_key())
        other_deleg = DelegationCertificate.from_dict(d)
        s2 = sign_delegation(other_deleg, other_key)
        assert s1.signature != s2.signature
        pk1 = (
            derive_policy_key(s1.signature)
            .public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
        )
        pk2 = (
            derive_policy_key(s2.signature)
            .public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
        )
        assert pk1 != pk2

    def test_policy_key_is_32_bytes(self, signed_delegation):
        pk = derive_policy_key(signed_delegation.signature)
        raw = pk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        assert len(raw) == 32

    def test_derive_verify_key_matches(self, signed_delegation):
        priv = derive_policy_key(signed_delegation.signature)
        pub_direct = priv.public_key()
        pub_via_fn = derive_policy_verify_key(signed_delegation.signature)
        assert pub_direct.public_bytes(Encoding.Raw, PublicFormat.Raw) == pub_via_fn.public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
