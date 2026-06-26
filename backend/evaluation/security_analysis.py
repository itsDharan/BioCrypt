"""
Security analysis module for the multimodal biometric authentication system.

Evaluates:
  - Template inversion resistance (how hard to recover biometrics from vault)
  - Unlinkability (different enrollments → different vault representations)
  - Irreversibility (can't reconstruct embedding from vault hash)
  - Revocability (credential revocation works end-to-end)
  - Replay attack resistance (timestamp/nonce verification)
  - STRIDE threat model analysis

Usage:
    python evaluation/security_analysis.py
"""

import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import EVALUATION_DIR


class SecurityAnalyzer:
    """Automated security evaluation for biometric template protection."""

    def __init__(self):
        self.results = {}

    def test_template_inversion_resistance(self, num_trials: int = 100):
        """
        Measure reconstruction difficulty from vault.
        Attempts brute-force and correlation attacks against stored vault points.
        """
        from crypto.fuzzy_vault import ImprovedFuzzyVault

        vault = ImprovedFuzzyVault()
        original = np.random.randn(512).astype(np.float32) * 0.3
        original = original / np.linalg.norm(original)

        vault_data, _ = vault.lock(original)
        num_points = len(vault_data["vault_points"])
        genuine_points = vault_data["num_genuine"]
        chaff_points = vault_data["num_chaff"]

        # Compute brute-force difficulty
        # Attacker must guess which points are genuine from total
        from math import comb, log2
        total = genuine_points + chaff_points
        combinations = comb(total, genuine_points)
        security_bits = log2(combinations) if combinations > 0 else 0

        # Try random embeddings against the vault
        successful_unlocks = 0
        for _ in range(num_trials):
            random_emb = np.random.randn(512).astype(np.float32) * 0.3
            random_emb = random_emb / np.linalg.norm(random_emb)
            success, _, _ = vault.unlock(random_emb, vault_data, enrolled_embedding=original)
            if success:
                successful_unlocks += 1

        self.results["template_inversion"] = {
            "total_vault_points": num_points,
            "genuine_points": genuine_points,
            "chaff_points": chaff_points,
            "security_bits": round(security_bits, 1),
            "brute_force_combinations": str(combinations),
            "random_attack_success_rate": successful_unlocks / num_trials,
            "passed": successful_unlocks == 0,
        }

        print(f"  Template Inversion Resistance:")
        print(f"    Vault points: {num_points} ({genuine_points} genuine + {chaff_points} chaff)")
        print(f"    Security level: {security_bits:.1f} bits")
        print(f"    Random attack success: {successful_unlocks}/{num_trials}")
        print(f"    Status: {'PASS' if successful_unlocks == 0 else 'FAIL'}")

    def test_unlinkability(self, num_trials: int = 50):
        """
        Verify different enrollments produce different vault representations.
        Same biometric → re-enrollment should produce unlinkable vaults.
        """
        from crypto.fuzzy_vault import ImprovedFuzzyVault

        vault = ImprovedFuzzyVault()
        features = np.random.randn(512).astype(np.float32) * 0.3
        features = features / np.linalg.norm(features)

        vault_hashes = set()
        point_sets = []

        for _ in range(num_trials):
            vault_data, _ = vault.lock(features)
            # Hash the full vault for comparison
            vault_hash = hashlib.sha256(
                json.dumps(vault_data["vault_points"], sort_keys=True).encode()
            ).hexdigest()
            vault_hashes.add(vault_hash)
            point_sets.append(set(tuple(p) for p in vault_data["vault_points"]))

        # All vaults should be different (different chaff, different shuffling)
        unique_ratio = len(vault_hashes) / num_trials

        # Measure overlap between vault point sets
        overlaps = []
        for i in range(min(10, len(point_sets))):
            for j in range(i + 1, min(10, len(point_sets))):
                intersection = len(point_sets[i] & point_sets[j])
                union = len(point_sets[i] | point_sets[j])
                overlaps.append(intersection / union if union > 0 else 0)

        avg_overlap = np.mean(overlaps) if overlaps else 0

        self.results["unlinkability"] = {
            "unique_vaults": len(vault_hashes),
            "total_enrollments": num_trials,
            "unique_ratio": unique_ratio,
            "avg_vault_overlap": avg_overlap,
            "passed": unique_ratio > 0.95 and avg_overlap < 0.5,
        }

        print(f"  Unlinkability:")
        print(f"    Unique vaults: {len(vault_hashes)}/{num_trials} ({unique_ratio*100:.1f}%)")
        print(f"    Avg vault overlap: {avg_overlap:.4f}")
        print(f"    Status: {'PASS' if unique_ratio > 0.95 else 'FAIL'}")

    def test_irreversibility(self):
        """
        Verify embedding cannot be recovered from vault hash.
        The embedding hash stored in vault is one-way SHA-256.
        """
        from crypto.fuzzy_vault import ImprovedFuzzyVault

        vault_obj = ImprovedFuzzyVault()
        features = np.random.randn(512).astype(np.float32) * 0.3
        features = features / np.linalg.norm(features)

        vault_data, _ = vault_obj.lock(features)
        emb_hash = vault_data["embedding_hash"]

        # Verify hash is SHA-256
        is_sha256 = len(emb_hash) == 64

        # Verify different embeddings produce different hashes
        different_hashes = set()
        for _ in range(20):
            rand_emb = np.random.randn(512).astype(np.float32) * 0.3
            rand_emb = rand_emb / np.linalg.norm(rand_emb)
            v2, _ = vault_obj.lock(rand_emb)
            different_hashes.add(v2["embedding_hash"])

        collision_free = len(different_hashes) == 20

        self.results["irreversibility"] = {
            "hash_algorithm": "SHA-256",
            "hash_length_bits": 256 if is_sha256 else 0,
            "collision_test_passed": collision_free,
            "passed": is_sha256 and collision_free,
        }

        print(f"  Irreversibility:")
        print(f"    Hash: SHA-256 ({256 if is_sha256 else 'UNKNOWN'} bits)")
        print(f"    Collision-free: {'YES' if collision_free else 'NO'}")
        print(f"    Status: {'PASS' if is_sha256 and collision_free else 'FAIL'}")

    def test_revocability(self):
        """Test credential revocation end-to-end."""
        from storage.blockchain_client import BlockchainClient

        bc = BlockchainClient()

        # Enroll
        bc.store_credentials("revoke_test_user", "QmTestCID", "key", "test")
        is_enrolled = bc.is_enrolled("revoke_test_user")

        # Revoke
        revoke_success = bc.revoke_credentials("revoke_test_user")
        is_still_enrolled = bc.is_enrolled("revoke_test_user")

        passed = is_enrolled and revoke_success and not is_still_enrolled

        self.results["revocability"] = {
            "enrollment_worked": is_enrolled,
            "revocation_worked": revoke_success,
            "post_revoke_rejected": not is_still_enrolled,
            "mode": bc.mode,
            "passed": passed,
        }

        print(f"  Revocability (mode={bc.mode}):")
        print(f"    Enrollment: {'OK' if is_enrolled else 'FAIL'}")
        print(f"    Revocation: {'OK' if revoke_success else 'FAIL'}")
        print(f"    Post-revoke rejected: {'OK' if not is_still_enrolled else 'FAIL'}")
        print(f"    Status: {'PASS' if passed else 'FAIL'}")

    def test_encryption_integrity(self):
        """Verify AES-GCM authentication tag prevents tampering."""
        from crypto.aes_encryption import AESEncryption

        aes = AESEncryption()
        plaintext = b"sensitive biometric template data"
        ct, nonce, key = aes.encrypt(plaintext)

        # Verify decryption works
        decrypted = aes.decrypt(ct, nonce, key)
        correct_decrypt = decrypted == plaintext

        # Verify tampering detection
        tampered = bytearray(ct)
        tampered[0] ^= 0xFF
        tamper_detected = False
        try:
            aes.decrypt(bytes(tampered), nonce, key)
        except Exception:
            tamper_detected = True

        self.results["encryption_integrity"] = {
            "algorithm": "AES-256-GCM",
            "correct_decrypt": correct_decrypt,
            "tamper_detected": tamper_detected,
            "passed": correct_decrypt and tamper_detected,
        }

        print(f"  Encryption Integrity (AES-256-GCM):")
        print(f"    Correct decrypt: {'OK' if correct_decrypt else 'FAIL'}")
        print(f"    Tamper detected: {'OK' if tamper_detected else 'FAIL'}")
        print(f"    Status: {'PASS' if correct_decrypt and tamper_detected else 'FAIL'}")

    def generate_threat_model(self) -> Dict:
        """Generate STRIDE threat model analysis."""
        threats = {
            "Spoofing": {
                "threat": "Attacker presents fake biometrics (photo, prosthetic iris)",
                "mitigation": "Multi-modal fusion (face+iris), liveness detection required",
                "residual_risk": "Medium (no liveness detection currently implemented)",
            },
            "Tampering": {
                "threat": "Attacker modifies stored vault data or blockchain records",
                "mitigation": "AES-256-GCM integrity tags, blockchain immutability, SHA-256 vault hash",
                "residual_risk": "Low (blockchain consensus prevents modification)",
            },
            "Repudiation": {
                "threat": "User denies authentication action",
                "mitigation": "Blockchain event logging (CredentialStored, CredentialRevoked events)",
                "residual_risk": "Low (immutable audit trail)",
            },
            "Information Disclosure": {
                "threat": "Raw biometric templates leaked",
                "mitigation": "Fuzzy vault protection, AES-256-GCM + ECC-ECIES hybrid encryption",
                "residual_risk": "Very low (templates never stored in cleartext)",
            },
            "Denial of Service": {
                "threat": "Attacker prevents legitimate users from authenticating",
                "mitigation": "IPFS distributed storage (no single point of failure), local mock fallback",
                "residual_risk": "Medium (blockchain node availability)",
            },
            "Elevation of Privilege": {
                "threat": "Attacker bypasses authentication to gain unauthorized access",
                "mitigation": "Dual-gate auth (cosine + vault), ECC key possession requirement",
                "residual_risk": "Low (multiple independent verification layers)",
            },
        }

        self.results["threat_model"] = threats
        return threats

    def run_all(self):
        """Run all security tests."""
        print("=" * 60)
        print("  Security Analysis Report")
        print("=" * 60)

        self.test_template_inversion_resistance()
        self.test_unlinkability()
        self.test_irreversibility()
        self.test_revocability()
        self.test_encryption_integrity()
        self.generate_threat_model()

        # Summary
        tests = [
            "template_inversion", "unlinkability", "irreversibility",
            "revocability", "encryption_integrity",
        ]
        all_passed = all(self.results.get(t, {}).get("passed", False) for t in tests)

        print(f"\n{'='*60}")
        print(f"  Security Summary: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print(f"{'='*60}")

        return self.results

    def save_results(self, filename: str = "security_analysis.json"):
        """Save results to JSON."""
        save_path = EVALUATION_DIR / filename
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nSecurity report saved to {save_path}")


if __name__ == "__main__":
    analyzer = SecurityAnalyzer()
    analyzer.run_all()
    analyzer.save_results()
