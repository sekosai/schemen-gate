"""Asymmetric key wrapping for zero-trust key distribution.

Provides an abstract KeyWrapper interface and a concrete X25519 + HKDF +
AES-256-GCM implementation (ECIES pattern).  The authority wraps a secret
key under the recipient's public key; only the recipient's private key can
unwrap it.  No shared secrets cross trust boundaries.

Wrapping flow:
    1. Generate ephemeral X25519 keypair
    2. ECDH(ephemeral_private, recipient_public) -> shared_secret
    3. HKDF(shared_secret, info) -> wrapping_key (32 bytes)
    4. AES-256-GCM(wrapping_key, plaintext_key) -> (nonce, ciphertext)
    5. Output: (ephemeral_public, nonce, ciphertext)

Unwrapping is the reverse: recipient performs ECDH with the ephemeral
public key to recover the same shared secret and wrapping key.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEYWRAP_INFO = b"schemen:v1:keywrap"


@dataclass(frozen=True)
class WrappedKey:
    """Output of a key wrapping operation."""

    ephemeral_public_key: bytes
    nonce: bytes
    ciphertext: bytes


class KeyWrapper(ABC):
    """Abstract interface for asymmetric key wrapping."""

    @property
    @abstractmethod
    def algorithm_id(self) -> str:
        """Short identifier for the wrapping algorithm."""

    @abstractmethod
    def wrap(self, plaintext_key: bytes, recipient_public_key: bytes) -> WrappedKey:
        """Wrap *plaintext_key* so only *recipient_public_key*'s holder can unwrap."""

    @abstractmethod
    def unwrap(self, wrapped: WrappedKey, recipient_private_key: bytes) -> bytes:
        """Unwrap using the recipient's private key."""


class X25519AESGCMWrapper(KeyWrapper):
    """X25519 ECDH + HKDF-SHA256 + AES-256-GCM key wrapping (ECIES pattern)."""

    @property
    def algorithm_id(self) -> str:
        return "X25519-HKDF-SHA256-AES256GCM"

    def wrap(self, plaintext_key: bytes, recipient_public_key: bytes) -> WrappedKey:
        if len(plaintext_key) != 32:
            raise ValueError("invalid key material")
        recipient_pub = X25519PublicKey.from_public_bytes(recipient_public_key)

        ephemeral_priv = X25519PrivateKey.generate()
        ephemeral_pub_bytes = ephemeral_priv.public_key().public_bytes_raw()

        shared_secret = ephemeral_priv.exchange(recipient_pub)

        wrapping_key = HKDF(
            algorithm=SHA256(),
            length=32,
            salt=None,
            info=KEYWRAP_INFO,
        ).derive(shared_secret)

        nonce = os.urandom(12)
        ciphertext = AESGCM(wrapping_key).encrypt(nonce, plaintext_key, ephemeral_pub_bytes)

        return WrappedKey(
            ephemeral_public_key=ephemeral_pub_bytes,
            nonce=nonce,
            ciphertext=ciphertext,
        )

    def unwrap(self, wrapped: WrappedKey, recipient_private_key: bytes) -> bytes:
        try:
            recipient_priv = X25519PrivateKey.from_private_bytes(recipient_private_key)
            ephemeral_pub = X25519PublicKey.from_public_bytes(wrapped.ephemeral_public_key)
            shared_secret = recipient_priv.exchange(ephemeral_pub)
            wrapping_key = HKDF(
                algorithm=SHA256(),
                length=32,
                salt=None,
                info=KEYWRAP_INFO,
            ).derive(shared_secret)
            plaintext = AESGCM(wrapping_key).decrypt(
                wrapped.nonce,
                wrapped.ciphertext,
                wrapped.ephemeral_public_key,
            )
        except Exception:
            raise ValueError("Key unwrapping failed — invalid or tampered key material") from None
        if len(plaintext) != 32:
            raise ValueError("invalid key material")
        return plaintext


def generate_x25519_keypair() -> tuple[bytes, bytes]:
    """Generate an X25519 keypair. Returns (private_key_bytes, public_key_bytes)."""
    private_key = X25519PrivateKey.generate()
    return (
        private_key.private_bytes_raw(),
        private_key.public_key().public_bytes_raw(),
    )
