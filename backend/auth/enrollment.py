"""
Enrollment pipeline for multimodal biometric authentication.
Handles the complete process from raw images to decentralized storage.

Flow:
  Raw Face + Iris Images
    -> Preprocess (MTCNN align, CLAHE, resize)
    -> Extract Features (FaceNet 512D + ResNet18 512D)
    -> Fuse (Weighted: alpha*face + (1-alpha)*iris)
    -> Fuzzy Vault Lock (Reed-Solomon + chaff)
    -> Hybrid Encrypt (AES-256-GCM + ECC key wrap)
    -> Store (IPFS + Blockchain)

Security:
  The enrolled embedding is stored ENCRYPTED inside the vault data
  so that authentication can perform cosine similarity verification
  against the query embedding. Without this, the vault has no
  biometric gate and accepts any ECC-key holder.
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
from config.settings import DEVICE, SAVED_MODELS_DIR
from data.preprocessing import FacePreprocessor, IrisPreprocessor
from models.face_model import FaceFeatureExtractor
from models.iris_model import IrisFeatureExtractor
from models.fusion import create_fusion
from crypto.fuzzy_vault import ImprovedFuzzyVault
from crypto.hybrid_crypto import HybridCrypto
from storage.ipfs_client import IPFSClient
from storage.blockchain_client import BlockchainClient


class EnrollmentManager:
    """
    Manages the complete biometric enrollment process.
    """

    def __init__(
        self,
        fusion_method: str = "weighted",
        use_ecc: bool = True,
    ):
        # Preprocessors
        self.face_preprocessor = FacePreprocessor()
        self.iris_preprocessor = IrisPreprocessor()

        # Feature extractors
        self.face_model = FaceFeatureExtractor()
        self.face_model.eval()

        # Load trained iris model
        self.iris_model = None
        iris_model_path = SAVED_MODELS_DIR / "iris_model_best.pth"
        if iris_model_path.exists():
            self.iris_model = IrisFeatureExtractor.load_trained(str(iris_model_path))
            print("Enrollment: Loaded trained iris model")
        else:
            self.iris_model = IrisFeatureExtractor().to(DEVICE)
            self.iris_model.eval()
            print("Warning: No trained iris model found. Using untrained model.")

        # Fusion
        self.fusion = create_fusion(fusion_method)

        # Crypto
        self.vault = ImprovedFuzzyVault()
        self.crypto = HybridCrypto()

        # Storage
        self.ipfs = IPFSClient()
        self.blockchain = BlockchainClient()

    def enroll(
        self,
        user_id: str,
        face_input,
        iris_input,
    ) -> Dict[str, Any]:
        """
        Enroll a user with face + iris biometrics.

        The enrolled fused embedding is stored INSIDE the encrypted vault
        payload so it can be retrieved during authentication for cosine
        similarity verification. This is critical for security — without
        the enrolled embedding, the vault cannot distinguish genuine users
        from impostors who have the ECC key.

        Args:
            user_id: Unique user identifier
            face_input: Face image (PIL, numpy, or path)
            iris_input: Iris/eye image (PIL, numpy, or path)

        Returns:
            Enrollment result dictionary
        """
        t0 = time.time()
        result = {"user_id": user_id, "success": False, "steps": {}}

        try:
            # Step 1: Preprocess
            t1 = time.time()
            face_tensor = self.face_preprocessor.preprocess(face_input)
            iris_tensor = self.iris_preprocessor.preprocess(iris_input)

            if face_tensor is None:
                return {**result, "error": "Face detection failed"}

            result["steps"]["preprocess"] = time.time() - t1

            # Step 2: Extract features
            t2 = time.time()
            face_embedding = self.face_model.extract(face_tensor)
            iris_embedding = self.iris_model.extract(iris_tensor)
            result["steps"]["feature_extraction"] = time.time() - t2

            # Step 3: Fuse
            t3 = time.time()
            fused = self.fusion.fuse_numpy(face_embedding, iris_embedding) \
                if hasattr(self.fusion, 'fuse_numpy') else \
                self._fuse_tensors(face_embedding, iris_embedding)
            result["steps"]["fusion"] = time.time() - t3

            # Step 4: Fuzzy vault lock
            t4 = time.time()
            vault_data, secret_key = self.vault.lock(fused)
            result["steps"]["vault_lock"] = time.time() - t4

            # ── SECURITY CRITICAL ──
            # Store the enrolled embedding inside the vault payload.
            # This is encrypted with AES+ECC so only the private key holder
            # can access it. During authentication, this embedding is
            # retrieved and compared against the query embedding using
            # cosine similarity. This prevents impersonation attacks
            # where an attacker has the ECC key but different biometrics.
            vault_data["enrolled_embedding"] = base64.b64encode(
                fused.astype(np.float32).tobytes()
            ).decode("ascii")
            vault_data["embedding_dim"] = len(fused)

            # ── PER-MODALITY EMBEDDINGS (for independent verification) ──
            # Storing face and iris embeddings separately enables the
            # authentication pipeline to verify EACH modality independently.
            # This prevents attacks where one matching modality (e.g. same face)
            # "carries" a non-matching modality (e.g. different iris) past
            # the fused similarity threshold.
            vault_data["enrolled_face_embedding"] = base64.b64encode(
                face_embedding.astype(np.float32).tobytes()
            ).decode("ascii")
            vault_data["enrolled_iris_embedding"] = base64.b64encode(
                iris_embedding.astype(np.float32).tobytes()
            ).decode("ascii")
            vault_data["face_embedding_dim"] = len(face_embedding)
            vault_data["iris_embedding_dim"] = len(iris_embedding)

            # Generate a fresh ECC keypair for this enrollment
            ecc_private, ecc_public = self.crypto.generate_keypair()

            # Step 5: Hybrid encrypt (vault data INCLUDING enrolled embedding)
            t5 = time.time()
            encrypted = self.crypto.encrypt_vault(vault_data, ecc_public)
            result["steps"]["encryption"] = time.time() - t5

            # Step 6: Store on IPFS
            t6 = time.time()
            encrypted_json = json.dumps(encrypted)
            ipfs_cid = self.ipfs.upload(encrypted_json, f"vault_{user_id}.json")
            result["steps"]["ipfs_upload"] = time.time() - t6

            # Step 7: Store on blockchain
            t7 = time.time()
            tx_result = self.blockchain.store_credentials(
                user_id=user_id,
                ipfs_cid=ipfs_cid,
                encrypted_key=encrypted["encrypted_key"],
                metadata=json.dumps({
                    "timestamp": int(time.time()),
                    "vault_points": vault_data["num_genuine"] + vault_data["num_chaff"],
                }),
                ecc_private_key=ecc_private,
                ecc_public_key=ecc_public,
            )
            result["steps"]["blockchain_store"] = time.time() - t7

            # Success
            result["success"] = True
            result["ipfs_cid"] = ipfs_cid
            result["tx_result"] = tx_result
            result["key_hash"] = vault_data["key_hash"]
            result["total_time"] = time.time() - t0
            result["ecc_public_key"] = ecc_public
            result["ecc_private_key"] = ecc_private
            result["private_key"] = ecc_private
            result["enrolled_private_key"] = ecc_private

        except Exception as e:
            result["error"] = str(e)
            import traceback
            result["traceback"] = traceback.format_exc()

        return result

    def _fuse_tensors(self, face_emb, iris_emb):
        """Fallback tensor-based fusion."""
        face_t = torch.tensor(face_emb, dtype=torch.float32).unsqueeze(0)
        iris_t = torch.tensor(iris_emb, dtype=torch.float32).unsqueeze(0)
        fused_t = self.fusion(face_t, iris_t)
        return fused_t.detach().cpu().numpy().squeeze()
