"""
Authentication pipeline for verifying biometric identity.
Retrieves stored credentials from IPFS/blockchain and performs
MANDATORY triple-stage biometric verification.

Security Model:
  Gate 0 (MANDATORY): Independent per-modality verification.
         Face and iris embeddings are each compared SEPARATELY against
         their enrolled counterparts. BOTH must pass their individual
         thresholds. This prevents one modality from "carrying" the other.
  Gate 1 (MANDATORY): Cosine similarity between query FUSED embedding and
         the enrolled FUSED embedding stored in the encrypted vault.
         This is the overall biometric discrimination gate.
  Gate 2 (MANDATORY): Fuzzy vault structural verification.
         This ensures the vault template matches.

  ALL gates must pass for authentication to succeed.
  Having the ECC private key alone is NOT sufficient — the biometric
  features must match the enrolled template.

Attack Prevention:
  - Same face, different iris: Gate 0 rejects (iris cosine < threshold)
  - Same iris, different face: Gate 0 rejects (face cosine < threshold)
  - Impersonation with stolen ECC key: Gate 1 rejects (cosine < threshold)
  - Wrong user biometrics: Gates 0+1 reject (different embeddings)
  - Unenrolled user: Blockchain lookup fails ("User not enrolled")
"""

import time
import json
import base64
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DEVICE, SAVED_MODELS_DIR, SIMILARITY_THRESHOLD,
    FACE_SIMILARITY_THRESHOLD, IRIS_SIMILARITY_THRESHOLD,
)
from data.preprocessing import FacePreprocessor, IrisPreprocessor
from models.face_model import FaceFeatureExtractor
from models.iris_model import IrisFeatureExtractor
from models.fusion import create_fusion
from crypto.fuzzy_vault import ImprovedFuzzyVault
from crypto.hybrid_crypto import HybridCrypto
from storage.ipfs_client import IPFSClient
from storage.blockchain_client import BlockchainClient


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class AuthenticationManager:
    """
    Manages biometric authentication with MANDATORY triple-stage verification.

    Security invariant: Authentication ALWAYS requires biometric matching
    of EACH modality independently AND the fused embedding.
    Possessing the ECC private key is necessary but NOT sufficient.
    """

    def __init__(
        self,
        fusion_method: str = "weighted",
        threshold: float = SIMILARITY_THRESHOLD,
        face_threshold: float = FACE_SIMILARITY_THRESHOLD,
        iris_threshold: float = IRIS_SIMILARITY_THRESHOLD,
    ):
        # Preprocessors
        self.face_preprocessor = FacePreprocessor()
        self.iris_preprocessor = IrisPreprocessor()

        # Feature extractors
        self.face_model = FaceFeatureExtractor()
        self.face_model.eval()

        self.iris_model = None
        iris_model_path = SAVED_MODELS_DIR / "iris_model_best.pth"
        if iris_model_path.exists():
            self.iris_model = IrisFeatureExtractor.load_trained(str(iris_model_path))
        else:
            self.iris_model = IrisFeatureExtractor().to(DEVICE)
            self.iris_model.eval()
            print("Warning: No trained iris model found.")

        # Fusion
        self.fusion = create_fusion(fusion_method)

        # Crypto
        self.vault = ImprovedFuzzyVault()
        self.crypto = HybridCrypto()

        # Storage
        self.ipfs = IPFSClient()
        self.blockchain = BlockchainClient()

        # Thresholds
        self.threshold = threshold
        self.face_threshold = face_threshold
        self.iris_threshold = iris_threshold

    def authenticate(
        self,
        user_id: str,
        face_input,
        iris_input,
        ecc_private_key: str = "",
    ) -> Dict[str, Any]:
        """
        Authenticate a user with face + iris biometrics.

        This method enforces MANDATORY per-modality + fused biometric matching:
        1. Retrieves the encrypted vault from IPFS (via blockchain)
        2. Decrypts it with the user's ECC private key
        3. Extracts enrolled face, iris, and fused embeddings
        4. Gate 0: Verifies face AND iris INDEPENDENTLY (both must pass)
        5. Gate 1: Verifies fused embedding cosine similarity
        6. Gate 2: Runs fuzzy vault structural verification
        7. ALL gates must pass for authentication

        Args:
            user_id: User identifier to verify against
            face_input: Face image (PIL, numpy, or path)
            iris_input: Iris/eye image (PIL, numpy, or path)
            ecc_private_key: User's ECC private key for decryption

        Returns:
            Authentication result dictionary
        """
        t0 = time.time()
        result = {
            "user_id": user_id,
            "authenticated": False,
            "stages": {},
            "steps": {},
        }

        try:
            # ── Step 1: Check enrollment ──
            t_bc = time.time()
            credentials = self.blockchain.get_credentials(user_id)
            if credentials is None:
                return {**result, "error": f"User '{user_id}' not enrolled"}
            if not credentials.get("active", False):
                return {**result, "error": f"User '{user_id}' credentials have been revoked"}
            result["steps"]["blockchain_retrieve"] = time.time() - t_bc

            # ── Step 2: Preprocess query biometrics ──
            t1 = time.time()
            face_tensor = self.face_preprocessor.preprocess(face_input)
            iris_tensor = self.iris_preprocessor.preprocess(iris_input)

            if face_tensor is None:
                return {**result, "error": "Face detection failed in query image"}
            result["steps"]["preprocess"] = time.time() - t1

            # ── Step 3: Extract query features ──
            t2 = time.time()
            face_embedding = self.face_model.extract(face_tensor)
            iris_embedding = self.iris_model.extract(iris_tensor)
            result["steps"]["feature_extraction"] = time.time() - t2

            # ── Step 4: Fuse query features ──
            t3 = time.time()
            query_fused = self.fusion.fuse_numpy(face_embedding, iris_embedding) \
                if hasattr(self.fusion, 'fuse_numpy') else \
                self._fuse_tensors(face_embedding, iris_embedding)
            result["steps"]["fusion"] = time.time() - t3

            # ── Step 5: Retrieve encrypted vault from IPFS ──
            t5 = time.time()
            ipfs_cid = credentials["ipfs_cid"]
            encrypted_json = self.ipfs.retrieve(ipfs_cid)
            encrypted_data = json.loads(encrypted_json)
            result["steps"]["ipfs_retrieve"] = time.time() - t5

            # ── Step 6: Resolve ECC private key ──
            # If user didn't supply the key, auto-retrieve it from storage
            resolved_key = ecc_private_key.strip() if ecc_private_key else ""
            if not resolved_key:
                resolved_key = self.blockchain.get_ecc_private_key(user_id) or ""
                if resolved_key:
                    result["key_source"] = "auto-retrieved from secure storage"
                else:
                    return {
                        **result,
                        "error": (
                            "No ECC private key provided and none found in storage. "
                            "This user may have been enrolled before key storage was enabled. "
                            "Please re-enroll."
                        ),
                    }
            else:
                result["key_source"] = "user-provided"

            # ── Step 7: Decrypt vault with ECC private key ──
            t6 = time.time()
            try:
                vault_data = self.crypto.decrypt_vault(encrypted_data, resolved_key)
            except Exception as e:
                return {
                    **result,
                    "error": f"Decryption failed: invalid ECC private key. {str(e)}",
                    "stages": {"decryption": {"success": False, "error": str(e)}},
                }
            result["steps"]["decryption"] = time.time() - t6

            # ── Gate 0: MANDATORY Per-Modality Verification ──
            # Check face and iris INDEPENDENTLY against enrolled embeddings.
            # This prevents attacks where one matching modality (e.g. same face)
            # "carries" a non-matching modality (e.g. different iris) past
            # the fused similarity threshold.
            t_gate0 = time.time()
            enrolled_face = self._extract_embedding(
                vault_data, "enrolled_face_embedding", "face_embedding_dim"
            )
            enrolled_iris = self._extract_embedding(
                vault_data, "enrolled_iris_embedding", "iris_embedding_dim"
            )

            if enrolled_face is not None and enrolled_iris is not None:
                face_sim = _cosine_similarity(face_embedding, enrolled_face)
                iris_sim = _cosine_similarity(iris_embedding, enrolled_iris)

                face_passed = face_sim >= self.face_threshold
                iris_passed = iris_sim >= self.iris_threshold

                print(
                    f"[AUTH] User '{user_id}' per-modality check:\n"
                    f"       Face cosine: {face_sim:.4f} (threshold: {self.face_threshold}) "
                    f"→ {'PASS' if face_passed else 'REJECT'}\n"
                    f"       Iris cosine: {iris_sim:.4f} (threshold: {self.iris_threshold}) "
                    f"→ {'PASS' if iris_passed else 'REJECT'}"
                )

                result["stages"]["per_modality"] = {
                    "face_similarity": round(face_sim, 4),
                    "face_threshold": self.face_threshold,
                    "face_passed": face_passed,
                    "iris_similarity": round(iris_sim, 4),
                    "iris_threshold": self.iris_threshold,
                    "iris_passed": iris_passed,
                    "both_passed": face_passed and iris_passed,
                }

                # ── SECURITY: If EITHER modality fails, REJECT immediately ──
                if not face_passed or not iris_passed:
                    failed_modality = []
                    if not face_passed:
                        failed_modality.append(f"face ({face_sim:.4f} < {self.face_threshold})")
                    if not iris_passed:
                        failed_modality.append(f"iris ({iris_sim:.4f} < {self.iris_threshold})")

                    result["authenticated"] = False
                    result["error"] = (
                        f"Per-modality biometric verification failed: "
                        f"{', '.join(failed_modality)}. "
                        f"Each biometric modality must independently match "
                        f"the enrolled template."
                    )
                    result["total_time"] = time.time() - t0
                    return result
            else:
                # Legacy vault without per-modality embeddings — log warning
                print(
                    f"[AUTH] User '{user_id}': No per-modality embeddings found in vault. "
                    f"Re-enroll for enhanced security. Falling back to fused-only check."
                )
                result["stages"]["per_modality"] = {
                    "skipped": True,
                    "reason": "Legacy vault — re-enroll for per-modality verification",
                }

            result["steps"]["per_modality_check"] = time.time() - t_gate0

            # ── Gate 1: MANDATORY Fused Cosine Similarity Gate ──
            # Extract enrolled FUSED embedding from decrypted vault
            t7 = time.time()
            enrolled_embedding = self._extract_embedding(
                vault_data, "enrolled_embedding", "embedding_dim"
            )

            if enrolled_embedding is None:
                # Legacy vault without embedded enrollment — REJECT for security
                result["stages"]["cosine_similarity"] = {
                    "score": 0.0,
                    "threshold": self.threshold,
                    "passed": False,
                    "error": "No enrolled embedding found in vault (re-enroll required)",
                }
                result["error"] = (
                    "Vault was created without stored biometric template. "
                    "Please re-enroll to enable biometric verification."
                )
                return result

            # Compute cosine similarity
            similarity = _cosine_similarity(query_fused, enrolled_embedding)
            result["steps"]["cosine_check"] = time.time() - t7

            cosine_passed = similarity >= self.threshold
            print(
                f"[AUTH] User '{user_id}' fused cosine similarity: {similarity:.4f} "
                f"(threshold: {self.threshold}) → {'PASS' if cosine_passed else 'REJECT'}"
            )
            result["stages"]["cosine_similarity"] = {
                "score": round(similarity, 4),
                "threshold": self.threshold,
                "passed": cosine_passed,
            }

            # ── SECURITY: If fused cosine fails, REJECT immediately ──
            if not cosine_passed:
                result["authenticated"] = False
                result["stages"]["fuzzy_vault"] = {
                    "success": False,
                    "matching_ratio": 0,
                    "key_recovered": False,
                    "skipped": True,
                    "reason": "Biometric mismatch detected - vault check skipped",
                }
                result["error"] = (
                    f"Biometric verification failed. "
                    f"Similarity score {similarity:.4f} is below "
                    f"threshold {self.threshold}. "
                    f"The face/iris images do not match the enrolled biometrics."
                )
                result["total_time"] = time.time() - t0
                return result

            # ── Gate 2: Fuzzy vault unlock ──
            t8 = time.time()
            vault_success, vault_score, recovered_key = self.vault.unlock(
                query_fused, vault_data, enrolled_embedding=enrolled_embedding
            )
            result["steps"]["vault_unlock"] = time.time() - t8
            result["stages"]["fuzzy_vault"] = {
                "success": vault_success,
                "matching_ratio": vault_score if isinstance(vault_score, (int, float)) else 0,
                "key_recovered": recovered_key is not None,
            }

            # ── Final Decision: ALL gates must pass ──
            result["authenticated"] = cosine_passed and vault_success
            result["total_time"] = time.time() - t0

        except Exception as e:
            result["error"] = str(e)
            import traceback
            result["traceback"] = traceback.format_exc()

        return result

    def _extract_embedding(
        self, vault_data: Dict, key: str, dim_key: str
    ) -> Optional[np.ndarray]:
        """
        Extract an embedding from the decrypted vault data.

        The enrollment process stores embeddings as base64-encoded
        float32 arrays inside the vault dictionary.

        Args:
            vault_data: Decrypted vault dictionary
            key: Key name for the base64-encoded embedding
            dim_key: Key name for the expected embedding dimension

        Returns:
            numpy array of the embedding, or None if not found.
        """
        emb_b64 = vault_data.get(key)
        emb_dim = vault_data.get(dim_key, 512)

        if emb_b64 is None:
            return None

        try:
            raw_bytes = base64.b64decode(emb_b64)
            embedding = np.frombuffer(raw_bytes, dtype=np.float32)
            if len(embedding) != emb_dim:
                print(f"Warning: Expected {emb_dim}D embedding for '{key}', got {len(embedding)}D")
            return embedding
        except Exception as e:
            print(f"Error extracting embedding '{key}': {e}")
            return None

    # Keep backward-compatible alias
    def _extract_enrolled_embedding(self, vault_data: Dict) -> Optional[np.ndarray]:
        """Extract the enrolled fused embedding (backward compatibility)."""
        return self._extract_embedding(vault_data, "enrolled_embedding", "embedding_dim")

    def _fuse_tensors(self, face_emb, iris_emb):
        """Fallback tensor-based fusion."""
        face_t = torch.tensor(face_emb, dtype=torch.float32).unsqueeze(0)
        iris_t = torch.tensor(iris_emb, dtype=torch.float32).unsqueeze(0)
        fused_t = self.fusion(face_t, iris_t)
        return fused_t.detach().cpu().numpy().squeeze()
