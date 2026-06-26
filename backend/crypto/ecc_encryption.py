"""
ECC (Elliptic Curve Cryptography) for AES key wrapping.
Uses ECIES (Elliptic Curve Integrated Encryption Scheme) via eciespy.

ECC is preferred over RSA for:
  - Smaller key sizes (256-bit ECC ~ 3072-bit RSA security)
  - Faster operations
  - Lower computational overhead

Design: ECC only wraps the small AES key, never the bulk vault data.
"""

import os
import json
from typing import Tuple, Dict, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


class ECCKeyManager:
    """
    ECC key management and AES key wrapping using ECIES.
    Falls back to RSA-2048 if eciespy is not available.
    """

    def __init__(self):
        self.use_ecc = True

        try:
            import eciespy
            self._ecies = eciespy
        except ImportError:
            print("Warning: eciespy not available. Using RSA-2048 fallback.")
            self.use_ecc = False

    def generate_keypair(self) -> Tuple[str, str]:
        """
        Generate an ECC key pair (secp256k1).

        Returns:
            (private_key_hex, public_key_hex)
        """
        if self.use_ecc:
            private_key = self._ecies.utils.generate_eth_key()
            private_key_hex = private_key.to_hex()
            public_key_hex = private_key.public_key.to_hex()
            return private_key_hex, public_key_hex
        else:
            return self._generate_rsa_keypair()

    def encrypt_key(self, aes_key: bytes, public_key_hex: str) -> bytes:
        """
        Encrypt an AES key using ECC public key (ECIES).

        Args:
            aes_key: The AES-256 key to wrap (32 bytes)
            public_key_hex: Recipient's ECC public key (hex)

        Returns:
            Encrypted AES key bytes
        """
        if self.use_ecc:
            return self._ecies.encrypt(public_key_hex, aes_key)
        else:
            return self._rsa_encrypt(aes_key, public_key_hex)

    def decrypt_key(self, encrypted_key: bytes, private_key_hex: str) -> bytes:
        """
        Decrypt an AES key using ECC private key (ECIES).

        Args:
            encrypted_key: ECIES-encrypted AES key
            private_key_hex: Recipient's ECC private key (hex)

        Returns:
            Decrypted AES key (32 bytes)
        """
        if self.use_ecc:
            return self._ecies.decrypt(private_key_hex, encrypted_key)
        else:
            return self._rsa_decrypt(encrypted_key, private_key_hex)

    # ── RSA Fallback ──────────────────────────────────────────────────

    def _generate_rsa_keypair(self) -> Tuple[str, str]:
        """Generate RSA-2048 key pair as fallback."""
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import serialization

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        return private_pem, public_pem

    def _rsa_encrypt(self, data: bytes, public_key_pem: str) -> bytes:
        """Encrypt data with RSA public key."""
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes, serialization

        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def _rsa_decrypt(self, data: bytes, private_key_pem: str) -> bytes:
        """Decrypt data with RSA private key."""
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes, serialization

        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )
        return private_key.decrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def export_keys(self, private_key: str, public_key: str) -> Dict[str, str]:
        """Export key pair as a JSON-serializable dictionary."""
        return {
            "private_key": private_key,
            "public_key": public_key,
            "algorithm": "ECC-secp256k1" if self.use_ecc else "RSA-2048",
        }
