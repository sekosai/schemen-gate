"""Tests for token operations: GateKey, MaskToken, AdapterToken, GateRights.

Covers issuance, redemption, AAD binding (every contract field), key
derivation hierarchy, permission canonicalization, weight hashing,
token expiry, error types, and the convenience factory.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import replace

import numpy as np
import pytest

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

# ---------------------------------------------------------------------------
# GateKey
# ---------------------------------------------------------------------------


class TestGateKey:
    def test_generate(self):
        k = GateKey.generate()
        assert len(k.secret) == 32

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError):
            GateKey(secret=b"\x00" * 16)

    @pytest.mark.parametrize("secret", [bytearray(32), memoryview(bytes(32)), "0" * 32])
    def test_mutable_or_nonbyte_key_material_is_rejected(self, secret):
        with pytest.raises(ValueError, match="key material"):
            GateKey(secret=secret)

    def test_derived_key_rejects_an_invalid_context(self):
        with pytest.raises(ValueError, match="context"):
            DerivedKey(secret=bytes(32), context="")

    def test_repr_redacted(self):
        k = GateKey.generate()
        assert "REDACTED" in repr(k)


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


class TestKeyDerivation:
    @pytest.mark.parametrize("length", [-1, 0, True, 33, 1.5])
    def test_hkdf_rejects_invalid_lengths(self, length):
        with pytest.raises(ValueError, match="length"):
            hkdf_expand_sha256(bytes(32), b"context", length)

    @pytest.mark.parametrize("regime_id", [-1, True, 1.5, "1"])
    def test_regime_key_rejects_noncanonical_ids(self, regime_id):
        with pytest.raises(ValueError, match="regime_id"):
            derive_regime_key(GateKey(secret=b"\x01" * 32), regime_id)

    def test_regime_key_deterministic(self):
        master = GateKey(secret=b"\x01" * 32)
        k1 = derive_regime_key(master, 0)
        k2 = derive_regime_key(master, 0)
        assert k1.secret == k2.secret

    def test_different_regimes_different_keys(self):
        master = GateKey(secret=b"\x01" * 32)
        k0 = derive_regime_key(master, 0)
        k1 = derive_regime_key(master, 1)
        assert k0.secret != k1.secret

    def test_tenant_key_deterministic(self):
        master = GateKey(secret=b"\x01" * 32)
        t1 = derive_tenant_key(master, 0, "alice")
        t2 = derive_tenant_key(master, 0, "alice")
        assert t1.secret == t2.secret

    def test_different_tenants_different_keys(self):
        master = GateKey(secret=b"\x01" * 32)
        t1 = derive_tenant_key(master, 0, "alice")
        t2 = derive_tenant_key(master, 0, "bob")
        assert t1.secret != t2.secret

    def test_tenant_id_colon_rejected(self):
        master = GateKey(secret=b"\x01" * 32)
        with pytest.raises(ValueError):
            derive_tenant_key(master, 0, "a:b")

    def test_tenant_id_empty_rejected(self):
        master = GateKey(secret=b"\x01" * 32)
        with pytest.raises(ValueError):
            derive_tenant_key(master, 0, "")

    @pytest.mark.parametrize("tenant_id", [True, b"alice", "alice\x00admin"])
    def test_tenant_id_type_and_nul_are_rejected(self, tenant_id):
        master = GateKey(secret=b"\x01" * 32)
        with pytest.raises(ValueError, match="tenant_id"):
            derive_tenant_key(master, 0, tenant_id)


# ---------------------------------------------------------------------------
# MaskToken
# ---------------------------------------------------------------------------


class TestMaskToken:
    def test_issue_and_redeem(self):
        master = GateKey(secret=b"\x02" * 32)
        token = issue_mask_token(master, "alice", 0, 64, 4)
        key = derive_tenant_key(master, 0, "alice")
        mask = redeem_mask_token(token, key)
        assert mask.shape == (64,)
        assert int(mask.sum()) == 16
        assert set(np.unique(mask)) == {0.0, 1.0}

    def test_wrong_key_fails(self):
        master = GateKey(secret=b"\x02" * 32)
        token = issue_mask_token(master, "alice", 0, 64, 4)
        wrong_key = derive_tenant_key(GateKey(secret=b"\x03" * 32), 0, "alice")
        with pytest.raises(TokenAuthenticationError):
            redeem_mask_token(token, wrong_key)

    def test_tampered_n_dims_fails(self):
        master = GateKey(secret=b"\x02" * 32)
        token = issue_mask_token(master, "alice", 0, 64, 4)
        tampered = MaskToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=128,
            n_regimes=token.n_regimes,
        )
        key = derive_tenant_key(master, 0, "alice")
        with pytest.raises(TokenAuthenticationError):
            redeem_mask_token(tampered, key)


# ---------------------------------------------------------------------------
# AdapterToken — AAD binding (the contract IS the channel)
# ---------------------------------------------------------------------------


class TestAdapterToken:
    @pytest.fixture
    def setup(self):
        master = GateKey(secret=b"\x04" * 32)
        weights = os.urandom(512)
        topo = adapter_topology_string([768, 256, 768])
        token = issue_adapter_token(
            master,
            "tenant_a",
            0,
            256,
            4,
            weights,
            topo,
            version=1,
            permissions=["use", "update"],
            parent_lockbox_hash=hashlib.sha256(b"lockbox_v1").hexdigest(),
        )
        key = derive_tenant_key(master, 0, "tenant_a")
        return master, weights, topo, token, key

    def test_issue_and_redeem(self, setup):
        _, weights, _, token, key = setup
        recovered = redeem_adapter_token(token, key)
        assert recovered == weights

    def test_wrong_key_fails(self, setup):
        _master, _, _, token, _ = setup
        wrong = derive_tenant_key(GateKey.generate(), 0, "tenant_a")
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(token, wrong)

    def test_wrong_tenant_fails(self, setup):
        master, _, _, token, _ = setup
        wrong = derive_tenant_key(master, 0, "impostor")
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(token, wrong)

    def test_tampered_n_dims_fails(self, setup):
        _, _, _, token, key = setup
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=512,
            n_regimes=token.n_regimes,
            topology=token.topology,
            weight_hash=token.weight_hash,
            version=token.version,
            permissions=token.permissions,
            parent_lockbox_hash=token.parent_lockbox_hash,
        )
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_tampered_topology_fails(self, setup):
        _, _, _, token, key = setup
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            topology="128:64:128",
            weight_hash=token.weight_hash,
            version=token.version,
            permissions=token.permissions,
            parent_lockbox_hash=token.parent_lockbox_hash,
        )
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_tampered_version_fails(self, setup):
        _, _, _, token, key = setup
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            topology=token.topology,
            weight_hash=token.weight_hash,
            version=99,
            permissions=token.permissions,
            parent_lockbox_hash=token.parent_lockbox_hash,
        )
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_tampered_permissions_fails(self, setup):
        _, _, _, token, key = setup
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            topology=token.topology,
            weight_hash=token.weight_hash,
            version=token.version,
            permissions="delegate,destroy,update,use",
            parent_lockbox_hash=token.parent_lockbox_hash,
        )
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_tampered_lockbox_fails(self, setup):
        _, _, _, token, key = setup
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            topology=token.topology,
            weight_hash=token.weight_hash,
            version=token.version,
            permissions=token.permissions,
            parent_lockbox_hash=hashlib.sha256(b"wrong_lockbox").hexdigest(),
        )
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_delimiter_collision_cannot_reinterpret_permissions(self):
        master = GateKey(secret=b"\x04" * 32)
        weights = os.urandom(64)
        token = issue_adapter_token(
            master,
            "tenant_a",
            0,
            64,
            4,
            weights,
            adapter_topology_string([64, 32, 64]),
            permissions="use:alpha",
            parent_lockbox_hash=hashlib.sha256(b"beta").hexdigest(),
        )
        tampered = replace(
            token,
            permissions="use",
            parent_lockbox_hash=hashlib.sha256(b"alpha:beta").hexdigest(),
        )
        key = derive_tenant_key(master, 0, "tenant_a")

        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_tampered_ciphertext_fails(self, setup):
        _, _, _, token, key = setup
        bad_ct = bytearray(token.ciphertext)
        bad_ct[len(bad_ct) // 2] ^= 0x01
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=bytes(bad_ct),
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            topology=token.topology,
            weight_hash=token.weight_hash,
            version=token.version,
            permissions=token.permissions,
            parent_lockbox_hash=token.parent_lockbox_hash,
        )
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)


# ---------------------------------------------------------------------------
# Token expiry
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    def test_non_expired_token_redeems(self):
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(256)
        topo = adapter_topology_string([64, 32, 64])
        future = int(time.time()) + 3600
        token = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
            expires_epoch=future,
        )
        key = derive_tenant_key(master, 0, "alice")
        recovered = redeem_adapter_token(token, key)
        assert recovered == weights

    def test_expired_token_rejected(self):
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(256)
        topo = adapter_topology_string([64, 32, 64])
        past = int(time.time()) - 1
        token = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
            expires_epoch=int(time.time()) + 3600,
        )
        token = replace(token, expires_epoch=past)
        key = derive_tenant_key(master, 0, "alice")
        with pytest.raises(TokenExpiredError):
            redeem_adapter_token(token, key)

    def test_invalid_adapter_contract_fields_are_rejected_at_issuance(self):
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(32)

        with pytest.raises(ValueError, match="topology"):
            issue_adapter_token(
                master,
                "alice",
                0,
                64,
                4,
                weights,
                "64::64",
            )
        with pytest.raises(ValueError, match="parent_lockbox_hash"):
            issue_adapter_token(
                master,
                "alice",
                0,
                64,
                4,
                weights,
                "64:32:64",
                parent_lockbox_hash="not-a-hash",
            )
        with pytest.raises(ValueError, match="already-expired"):
            issue_adapter_token(
                master,
                "alice",
                0,
                64,
                4,
                weights,
                "64:32:64",
                expires_epoch=int(time.time()) - 1,
            )

    def test_omitted_expiry_gets_finite_default(self):
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(256)
        topo = adapter_topology_string([64, 32, 64])
        token = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
        )
        assert token.expires_epoch is not None
        assert token.expires_epoch > int(time.time())
        key = derive_tenant_key(master, 0, "alice")
        assert redeem_adapter_token(token, key) == weights

    def test_non_expiring_requires_explicit_policy(self):
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(256)
        token = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            adapter_topology_string([64, 32, 64]),
            expires_epoch=None,
            allow_non_expiring=True,
        )
        assert token.expires_epoch is None

    def test_tampered_expiry_fails(self):
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(256)
        topo = adapter_topology_string([64, 32, 64])
        future = int(time.time()) + 3600
        token = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
            expires_epoch=future,
        )
        tampered = AdapterToken(
            tenant_id=token.tenant_id,
            regime_id=token.regime_id,
            nonce=token.nonce,
            ciphertext=token.ciphertext,
            n_dims=token.n_dims,
            n_regimes=token.n_regimes,
            topology=token.topology,
            weight_hash=token.weight_hash,
            version=token.version,
            permissions=token.permissions,
            parent_lockbox_hash=token.parent_lockbox_hash,
            expires_epoch=future + 999999,
        )
        key = derive_tenant_key(master, 0, "alice")
        with pytest.raises(TokenAuthenticationError):
            redeem_adapter_token(tampered, key)

    def test_expires_epoch_in_aad(self):
        """Tokens with and without expiry must not be interchangeable."""
        master = GateKey(secret=b"\x07" * 32)
        weights = os.urandom(256)
        topo = adapter_topology_string([64, 32, 64])
        token_with = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
            expires_epoch=int(time.time()) + 3600,
        )
        token_without = issue_adapter_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
        )
        assert token_with.expires_epoch is not None
        assert token_without.expires_epoch is not None
        assert token_with.ciphertext != token_without.ciphertext


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


class TestConvenienceFactory:
    def test_issue_use_only_token(self):
        master = GateKey(secret=b"\x08" * 32)
        weights = os.urandom(256)
        topo = adapter_topology_string([64, 32, 64])
        token = issue_use_only_token(
            master,
            "alice",
            0,
            64,
            4,
            weights,
            topo,
        )
        assert token.permissions == "use"
        assert token.version == 1
        key = derive_tenant_key(master, 0, "alice")
        assert redeem_adapter_token(token, key) == weights


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_all_errors_are_schemen_token_error(self):
        assert issubclass(TokenAuthenticationError, SchemenTokenError)
        assert issubclass(TokenExpiredError, SchemenTokenError)
        assert issubclass(WeightIntegrityError, SchemenTokenError)
        assert issubclass(RegimePermissionError, SchemenTokenError)
        assert issubclass(StoreCapacityError, SchemenTokenError)

    def test_permission_error_does_not_shadow_builtin(self):
        import builtins

        assert RegimePermissionError is not builtins.PermissionError


# ---------------------------------------------------------------------------
# GateRights
# ---------------------------------------------------------------------------


class TestGateRights:
    def test_omitted_permissions_deny_and_expire(self):
        rights = GateRights(regime_id=0)
        assert rights.permissions_set() == set()
        assert rights.expires_epoch is not None
        assert rights.expires_epoch > int(time.time())

    def test_sign_and_verify(self):
        key = GateKey(secret=b"\x05" * 32)
        rights = GateRights(regime_id=0, can_use=True, can_update=True)
        sig = rights.sign(key)
        assert GateRights.verify(rights, sig, key)

    def test_tampered_field_rejected(self):
        key = GateKey(secret=b"\x05" * 32)
        rights = GateRights(regime_id=0, can_use=True, can_update=False)
        sig = rights.sign(key)
        tampered = GateRights(regime_id=0, can_use=True, can_update=True)
        assert not GateRights.verify(tampered, sig, key)

    def test_wrong_key_rejected(self):
        key = GateKey(secret=b"\x05" * 32)
        rights = GateRights(regime_id=0, can_use=True)
        sig = rights.sign(key)
        wrong = GateKey(secret=b"\x06" * 32)
        assert not GateRights.verify(rights, sig, wrong)

    def test_permissions_set(self):
        r = GateRights(regime_id=0, can_use=True, can_delegate=True, can_destroy=True)
        assert r.permissions_set() == {"use", "delegate", "destroy"}

    def test_expired_rights_do_not_verify_or_return_permissions(self):
        key = GateKey(secret=b"\x05" * 32)
        rights = GateRights(regime_id=0, can_use=True, expires_epoch=100)
        signature = rights.sign(key)

        assert not GateRights.verify(rights, signature, key, now_epoch=100)
        assert rights.permissions_set(now_epoch=100) == set()
        assert not GateRights.authorizes(
            rights,
            signature,
            key,
            "use",
            expected_regime_id=0,
            expected_issuer="root",
            now_epoch=100,
        )

    @pytest.mark.parametrize(
        "now_epoch",
        [True, "99", float("nan"), float("inf"), float("-inf"), -1],
    )
    def test_invalid_verifier_clock_overrides_fail_closed(self, now_epoch):
        key = GateKey(secret=b"\x05" * 32)
        rights = GateRights(regime_id=0, can_use=True, expires_epoch=100)
        signature = rights.sign(key)

        assert not GateRights.verify(
            rights,
            signature,
            key,
            now_epoch=now_epoch,
        )
        assert not GateRights.authorizes(
            rights,
            signature,
            key,
            "use",
            expected_regime_id=0,
            expected_issuer="root",
            now_epoch=now_epoch,
        )
        with pytest.raises(ValueError, match="now_epoch"):
            rights.is_expired(now_epoch=now_epoch)
        with pytest.raises(ValueError, match="now_epoch"):
            rights.permissions_set(now_epoch=now_epoch)

    def test_authorization_binds_regime_issuer_and_schema_version(self):
        key = GateKey(secret=b"\x05" * 32)
        rights = GateRights(regime_id=7, issuer="machine-ca", can_use=True)
        signature = rights.sign(key)

        assert GateRights.authorizes(
            rights,
            signature,
            key,
            "use",
            expected_regime_id=7,
            expected_issuer="machine-ca",
        )
        assert not GateRights.authorizes(
            rights,
            signature,
            key,
            "use",
            expected_regime_id=8,
            expected_issuer="machine-ca",
        )
        assert not GateRights.authorizes(
            rights,
            signature,
            key,
            "use",
            expected_regime_id=7,
            expected_issuer="other-ca",
        )
        assert not GateRights.authorizes(
            rights,
            signature,
            key,
            "use",
            expected_regime_id=7,
            expected_issuer="machine-ca",
            expected_version=2,
        )


# ---------------------------------------------------------------------------
# GateRights Boolean algebra: AND (compose), OR (union), NOT (complement).
#
# These tests authenticate that the 6-bit permission vector forms a Boolean
# algebra under the three operations, with the standard laws:
#
#   * idempotence:        a & a = a;          a | a = a
#   * commutativity:      a & b = b & a;      a | b = b | a
#   * associativity:      same for & and |
#   * double negation:    ~~a = a (on permission bits)
#   * de Morgan:          ~(a & b) = ~a | ~b; ~(a | b) = ~a & ~b
#   * complementation:    a & ~a = all-False; a | ~a = all-True (on bits)
#   * absorption:         a & (a | b) = a;    a | (a & b) = a
#   * distributivity:     & distributes over |, and vice versa
#
# This is the same algebra as HadamardAdapter's mask layer, applied to a
# 6-element bit vector instead of an n-element ambient mask.  Functional
# completeness follows: any monotone Boolean predicate over the six
# permissions can be expressed via these three primitives.
# ---------------------------------------------------------------------------


class TestGateRightsBooleanAlgebra:
    def _example_pair(self) -> tuple[GateRights, GateRights]:
        """A and B: distinct permission profiles on the same regime."""
        a = GateRights(
            regime_id=0,
            can_use=True,
            can_update=True,
            can_delegate=False,
            can_create_subordinate=False,
            can_destroy=True,
            can_inspect=False,
        )
        b = GateRights(
            regime_id=0,
            can_use=True,
            can_update=False,
            can_delegate=True,
            can_create_subordinate=False,
            can_destroy=False,
            can_inspect=True,
        )
        return a, b

    def _bits(self, r: GateRights) -> tuple[bool, ...]:
        return (
            r.can_use,
            r.can_update,
            r.can_delegate,
            r.can_create_subordinate,
            r.can_destroy,
            r.can_inspect,
        )

    def test_compose_is_bitwise_and(self):
        a, b = self._example_pair()
        c = a.compose(b)
        assert self._bits(c) == tuple(
            x and y for x, y in zip(self._bits(a), self._bits(b), strict=True)
        )

    def test_union_is_bitwise_or(self):
        a, b = self._example_pair()
        u = a.union(b)
        assert self._bits(u) == tuple(
            x or y for x, y in zip(self._bits(a), self._bits(b), strict=True)
        )

    def test_complement_is_bitwise_not(self):
        a, _ = self._example_pair()
        n = a.complement()
        assert self._bits(n) == tuple(not x for x in self._bits(a))

    def test_double_negation(self):
        """~~a == a on the permission bits."""
        a, _ = self._example_pair()
        assert self._bits(a.complement().complement()) == self._bits(a)

    def test_de_morgan_and_to_or(self):
        """~(a & b) == ~a | ~b on the permission bits."""
        a, b = self._example_pair()
        lhs = a.compose(b).complement()
        rhs = a.complement().union(b.complement())
        assert self._bits(lhs) == self._bits(rhs)

    def test_de_morgan_or_to_and(self):
        """~(a | b) == ~a & ~b on the permission bits."""
        a, b = self._example_pair()
        lhs = a.union(b).complement()
        rhs = a.complement().compose(b.complement())
        assert self._bits(lhs) == self._bits(rhs)

    def test_idempotence_of_compose_and_union(self):
        a, _ = self._example_pair()
        assert self._bits(a.compose(a)) == self._bits(a)
        assert self._bits(a.union(a)) == self._bits(a)

    def test_commutativity_of_compose_and_union(self):
        a, b = self._example_pair()
        assert self._bits(a.compose(b)) == self._bits(b.compose(a))
        assert self._bits(a.union(b)) == self._bits(b.union(a))

    def test_associativity_of_compose_and_union(self):
        a, b = self._example_pair()
        c = GateRights(
            regime_id=0,
            can_use=False,
            can_update=True,
            can_delegate=True,
            can_create_subordinate=True,
            can_destroy=False,
            can_inspect=False,
        )
        assert self._bits(a.compose(b).compose(c)) == self._bits(a.compose(b.compose(c)))
        assert self._bits(a.union(b).union(c)) == self._bits(a.union(b.union(c)))

    def test_absorption_laws(self):
        a, b = self._example_pair()
        assert self._bits(a.compose(a.union(b))) == self._bits(a)
        assert self._bits(a.union(a.compose(b))) == self._bits(a)

    def test_distributivity(self):
        a, b = self._example_pair()
        c = GateRights(
            regime_id=0,
            can_use=True,
            can_update=False,
            can_delegate=False,
            can_create_subordinate=True,
            can_destroy=True,
            can_inspect=True,
        )
        assert self._bits(a.compose(b.union(c))) == self._bits(a.compose(b).union(a.compose(c)))
        assert self._bits(a.union(b.compose(c))) == self._bits(a.union(b).compose(a.union(c)))

    def test_complementation_to_zero_and_one(self):
        """a & ~a is all-False; a | ~a is all-True.  The two halves
        of the spectral decomposition of the identity, on bits."""
        a, _ = self._example_pair()
        zero = a.compose(a.complement())
        one_ = a.union(a.complement())
        assert self._bits(zero) == (False,) * 6
        assert self._bits(one_) == (True,) * 6

    def test_compose_rejects_different_regimes(self):
        """AND-composing rights for different regimes is a category
        error: rights are per-regime."""
        a = GateRights(regime_id=0, can_use=True)
        b = GateRights(regime_id=1, can_use=True)
        with pytest.raises(ValueError, match="different regimes"):
            a.compose(b)
        with pytest.raises(ValueError, match="different regimes"):
            a.union(b)

    def test_compose_takes_min_expiry(self):
        """AND tightens expiry: result expires when EITHER operand
        would have expired."""
        a = GateRights(regime_id=0, can_use=True, expires_epoch=100)
        b = GateRights(regime_id=0, can_use=True, expires_epoch=200)
        assert a.compose(b).expires_epoch == 100

    def test_union_takes_max_expiry(self):
        """OR loosens expiry: result expires only after BOTH would
        have expired."""
        a = GateRights(regime_id=0, can_use=True, expires_epoch=100)
        b = GateRights(regime_id=0, can_use=True, expires_epoch=200)
        assert a.union(b).expires_epoch == 200

    def test_compose_treats_none_expiry_as_no_constraint(self):
        """expires_epoch=None means 'never expires'; AND with a
        finite expiry uses the finite one (the more restrictive
        constraint)."""
        a = GateRights(regime_id=0, can_use=True, expires_epoch=None)
        b = GateRights(regime_id=0, can_use=True, expires_epoch=200)
        assert a.compose(b).expires_epoch == 200
        assert b.compose(a).expires_epoch == 200

    def test_union_treats_none_expiry_as_dominant(self):
        """At OR, expires_epoch=None ('never expires') wins because
        if either grant lasts forever, the union does too."""
        a = GateRights(regime_id=0, can_use=True, expires_epoch=None)
        b = GateRights(regime_id=0, can_use=True, expires_epoch=200)
        assert a.union(b).expires_epoch is None
        assert b.union(a).expires_epoch is None

    def test_issuer_records_composition_history(self):
        """The derived GateRights' issuer string makes the composition
        readable: ``compose(a,b)``, ``union(a,b)``, ``complement(a)``."""
        a = GateRights(regime_id=0, can_use=True, issuer="alice")
        b = GateRights(regime_id=0, can_update=True, issuer="bob")
        assert a.compose(b).issuer == "compose(alice,bob)"
        assert a.union(b).issuer == "union(alice,bob)"
        assert a.complement().issuer == "complement(alice)"

    def test_metadata_records_operand_fingerprints(self):
        """Audit metadata: ``compose_of`` and ``union_of`` carry the
        operand fingerprints; ``complement_of`` carries the single
        operand's fingerprint.  An auditor can recover which two
        rights states fed each composition."""
        a, b = self._example_pair()
        c = a.compose(b)
        assert "compose_of" in c.metadata
        assert len(c.metadata["compose_of"]) == 2
        u = a.union(b)
        assert "union_of" in u.metadata
        assert len(u.metadata["union_of"]) == 2
        n = a.complement()
        assert "complement_of" in n.metadata
        assert isinstance(n.metadata["complement_of"], str)

    def test_operator_overloads_match_method_calls(self):
        """``a & b``, ``a | b``, ``~a`` produce the same permission
        bits as the explicit method calls."""
        a, b = self._example_pair()
        assert self._bits(a & b) == self._bits(a.compose(b))
        assert self._bits(a | b) == self._bits(a.union(b))
        assert self._bits(~a) == self._bits(a.complement())

    def test_three_source_intersection_is_strictest_axis(self):
        """The ratchet semantics: AND-of-three-sources collapses to
        the strictest bit on every axis.

        ``effective = delegation & partition & situational`` where any
        False in any source on any axis makes that bit False in the
        effective rights -- exactly the read-only-partition,
        read-only-inference, deny-write composition used by the Gate's
        permission model."""
        delegation = GateRights(
            regime_id=0,
            can_use=True,
            can_update=True,
            can_delegate=True,
            can_create_subordinate=False,
            can_destroy=False,
            can_inspect=True,
        )
        partition = GateRights(  # read-only partition
            regime_id=0,
            can_use=True,
            can_update=False,
            can_delegate=False,
            can_create_subordinate=False,
            can_destroy=False,
            can_inspect=True,
        )
        situational = GateRights(  # heavily restricted situation
            regime_id=0,
            can_use=True,
            can_update=False,
            can_delegate=False,
            can_create_subordinate=False,
            can_destroy=False,
            can_inspect=False,
        )
        effective = delegation.compose(partition).compose(situational)
        # Only can_use survives: True everywhere.
        # can_update: True & False & False = False (partition denies).
        # can_inspect: True & True & False = False (situation denies).
        # Everything else: False at the source.
        assert effective.can_use is True
        assert effective.can_update is False
        assert effective.can_delegate is False
        assert effective.can_create_subordinate is False
        assert effective.can_destroy is False
        assert effective.can_inspect is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_canonical_permissions_sorts(self):
        assert canonical_permissions(["use", "delegate", "update"]) == "delegate,update,use"

    def test_canonical_permissions_dedupes(self):
        assert canonical_permissions(["use", "use", "update"]) == "update,use"

    def test_canonical_permissions_string_input(self):
        assert canonical_permissions("use, update, delegate") == "delegate,update,use"

    def test_adapter_topology_string(self):
        assert adapter_topology_string([768, 256, 768]) == "768:256:768"

    def test_hash_adapter_weights_deterministic(self):
        w = b"\x42" * 100
        assert hash_adapter_weights(w) == hash_adapter_weights(w)

    def test_hash_adapter_weights_different(self):
        assert hash_adapter_weights(b"\x01") != hash_adapter_weights(b"\x02")
