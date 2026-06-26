"""
AES-256-GCM authenticated encryption for fuzzy vault data.
Provides confidentiality and integrity in a single operation.
"""

import os
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import AES_KEY_SIZE, AES_NONCE_SIZE


class AESEncryption:
    """
    AES-256-GCM authenticated encryption.

    GCM mode provides both encryption and authentication (integrity check)
    in a single pass — no need for separate HMAC.
    """

    def __init__(self, key_size: int = AES_KEY_SIZE):
        self.key_size = key_size

    def generate_key(self) -> bytes:
        """Generate a random AES-256 key."""
        return os.urandom(self.key_size)

    def encrypt(
        self,
        plaintext: bytes,
        key: bytes = None,
        associated_data: bytes = None,
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt
            key: AES key (generated if None)
            associated_data: Optional authenticated but unencrypted data

        Returns:
            (ciphertext, nonce, key)
        """
        if key is None:
            key = self.generate_key()

        nonce = os.urandom(AES_NONCE_SIZE)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        return ciphertext, nonce, key

    def decrypt(
        self,
        ciphertext: bytes,
        nonce: bytes,
        key: bytes,
        associated_data: bytes = None,
    ) -> bytes:
        """
        Decrypt data using AES-256-GCM.

        Args:
            ciphertext: Encrypted data (includes GCM auth tag)
            nonce: 96-bit nonce used during encryption
            key: AES key
            associated_data: Must match the data used during encryption

        Returns:
            Decrypted plaintext

        Raises:
            cryptography.exceptions.InvalidTag: If authentication fails
        """
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data)
        return plaintext
