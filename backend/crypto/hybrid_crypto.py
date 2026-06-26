"""
Hybrid encryption pipeline: AES-256-GCM + ECC (ECIES).

Flow:
  Encrypt: AES encrypts vault data -> ECC encrypts AES key
  Decrypt: ECC decrypts AES key -> AES decrypts vault data

Design rationale:
  - AES handles bulk encryption (fast, suitable for large vault data)
  - ECC wraps only the small AES key (avoids asymmetric crypto overhead)
  - This is the standard hybrid encryption pattern used in TLS, PGP, etc.
"""

import json
import base64
from typing import Dict, Any, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from crypto.aes_encryption import AESEncryption
from crypto.ecc_encryption import ECCKeyManager


class HybridCrypto:
    """
    Hybrid AES + ECC encryption pipeline.

    Usage:
        crypto = HybridCrypto()
        priv, pub = crypto.generate_keypair()

        # Encrypt
        encrypted = crypto.encrypt(vault_json, pub)

        # Decrypt
        decrypted = crypto.decrypt(encrypted, priv)
    """

    def __init__(self):
        self.aes = AESEncryption()
        self.ecc = ECCKeyManager()

    def generate_keypair(self) -> Tuple[str, str]:
        """Generate ECC key pair for asymmetric key wrapping."""
        return self.ecc.generate_keypair()

    def encrypt(
        self,
        plaintext: str,
        public_key: str,
    ) -> Dict[str, str]:
        """
        Hybrid encrypt: AES encrypts data, ECC wraps AES key.

        Args:
            plaintext: Data to encrypt (typically JSON vault string)
            public_key: Recipient's ECC public key (hex)

        Returns:
            Dictionary with encrypted components (base64-encoded):
              - ciphertext: AES-encrypted vault data
              - nonce: AES-GCM nonce
              - encrypted_key: ECC-encrypted AES key
              - algorithm: Encryption algorithm identifier
        """
        plaintext_bytes = plaintext.encode("utf-8")

        # Step 1: AES encrypt the vault data
        ciphertext, nonce, aes_key = self.aes.encrypt(plaintext_bytes)

        # Step 2: ECC encrypt the AES key
        encrypted_aes_key = self.ecc.encrypt_key(aes_key, public_key)

        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "encrypted_key": base64.b64encode(encrypted_aes_key).decode(),
            "algorithm": "AES-256-GCM+ECIES",
        }

    def decrypt(
        self,
        encrypted_data: Dict[str, str],
        private_key: str,
    ) -> str:
        """
        Hybrid decrypt: ECC unwraps AES key, AES decrypts data.

        Args:
            encrypted_data: Dictionary from encrypt()
            private_key: Recipient's ECC private key (hex)

        Returns:
            Decrypted plaintext string
        """
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])
        encrypted_aes_key = base64.b64decode(encrypted_data["encrypted_key"])

        # Step 1: ECC decrypt the AES key
        aes_key = self.ecc.decrypt_key(encrypted_aes_key, private_key)

        # Step 2: AES decrypt the vault data
        plaintext = self.aes.decrypt(ciphertext, nonce, aes_key)

        return plaintext.decode("utf-8")

    def encrypt_vault(
        self,
        vault: Dict[str, Any],
        public_key: str,
    ) -> Dict[str, str]:
        """Convenience method: encrypt a vault dictionary."""
        vault_json = json.dumps(vault)
        return self.encrypt(vault_json, public_key)

    def decrypt_vault(
        self,
        encrypted_data: Dict[str, str],
        private_key: str,
    ) -> Dict[str, Any]:
        """Convenience method: decrypt to a vault dictionary."""
        vault_json = self.decrypt(encrypted_data, private_key)
        return json.loads(vault_json)
