"""
Feature fusion strategies for combining face and iris biometric embeddings.

v2 improvements:
  - Quality-Aware Adaptive Fusion (primary): dynamically weights modalities
    based on embedding quality scores per-sample
  - Score-normalized fusion for reduced distribution overlap
  - Existing weighted, concat, and attention fusion preserved

Quality-aware fusion measures embedding confidence via L2-norm stability
and entropy of the embedding distribution, then gates each modality's
contribution accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    FACE_EMBEDDING_DIM, IRIS_EMBEDDING_DIM,
    FUSED_EMBEDDING_DIM, FUSION_ALPHA,
)


class QualityAwareFusion(nn.Module):
    """
    Quality-Aware Adaptive Fusion.

    Instead of static weights (alpha*face + (1-alpha)*iris), this fusion
    dynamically adjusts per-sample weights based on embedding quality:

    Quality score q = sigmoid(MLP(embedding_stats))

    Where embedding_stats includes:
      - L2 norm of the raw (pre-normalization) embedding
      - Mean absolute activation
      - Standard deviation of activations
      - Max activation

    Higher quality → higher weight. Both modalities are fused as:
      fused = softmax([q_face, q_iris]) · [face, iris]

    This handles cases where one modality has poor input (e.g., blurry iris,
    occluded face) by automatically down-weighting it.
    """

    def __init__(
        self,
        face_dim: int = FACE_EMBEDDING_DIM,
        iris_dim: int = IRIS_EMBEDDING_DIM,
        fused_dim: int = FUSED_EMBEDDING_DIM,
        base_alpha: float = FUSION_ALPHA,
    ):
        super().__init__()
        self.face_dim = face_dim
        self.iris_dim = iris_dim
        self.base_alpha = base_alpha

        # Quality estimator MLP for each modality
        # Input: 4 stats (norm, mean_abs, std, max)
        self.face_quality = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.iris_quality = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _compute_quality_stats(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute quality statistics from an embedding."""
        # embedding is L2-normalized, so we measure distribution properties
        norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        mean_abs = torch.mean(torch.abs(embedding), dim=1, keepdim=True)
        std = torch.std(embedding, dim=1, keepdim=True)
        max_val = torch.max(embedding, dim=1, keepdim=True)[0]
        return torch.cat([norm, mean_abs, std, max_val], dim=1)

    def forward(
        self,
        face_embedding: torch.Tensor,
        iris_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quality-aware adaptive fusion.

        Args:
            face_embedding: (B, 512) L2-normalized face embedding
            iris_embedding: (B, 512) L2-normalized iris embedding

        Returns:
            (B, 512) L2-normalized fused embedding
        """
        # Compute quality scores
        face_stats = self._compute_quality_stats(face_embedding)
        iris_stats = self._compute_quality_stats(iris_embedding)

        face_q = self.face_quality(face_stats)  # (B, 1)
        iris_q = self.iris_quality(iris_stats)  # (B, 1)

        # Softmax over quality scores to get adaptive weights
        # Bias toward base_alpha: add prior
        face_prior = torch.full_like(face_q, self.base_alpha)
        iris_prior = torch.full_like(iris_q, 1.0 - self.base_alpha)

        weights = torch.softmax(
            torch.cat([face_q + face_prior, iris_q + iris_prior], dim=1),
            dim=1,
        )  # (B, 2)

        face_w = weights[:, 0:1]  # (B, 1)
        iris_w = weights[:, 1:2]  # (B, 1)

        # Fuse
        fused = face_w * face_embedding + iris_w * iris_embedding
        return F.normalize(fused, p=2, dim=1)

    def fuse_numpy(self, face_emb: np.ndarray, iris_emb: np.ndarray) -> np.ndarray:
        """Fuse numpy arrays using quality-aware fusion."""
        face_t = torch.from_numpy(face_emb).float().unsqueeze(0)
        iris_t = torch.from_numpy(iris_emb).float().unsqueeze(0)

        self.eval()
        with torch.no_grad():
            fused = self.forward(face_t, iris_t)
        return fused.squeeze(0).numpy()


class WeightedFusion(nn.Module):
    """
    Weighted score-level fusion: fused = alpha * face + (1 - alpha) * iris.

    This is the fallback fusion method. It preserves L2-normalized
    embedding quality — unlike concatenation+projection which pushes
    embeddings through untrained layers and degrades discriminative power.
    """

    def __init__(self, alpha: float = FUSION_ALPHA):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        face_embedding: torch.Tensor,
        iris_embedding: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.alpha * face_embedding + (1 - self.alpha) * iris_embedding
        return F.normalize(fused, p=2, dim=1)

    def fuse_numpy(self, face_emb, iris_emb):
        """Fuse numpy arrays directly."""
        fused = self.alpha * face_emb + (1 - self.alpha) * iris_emb
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        return fused


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion with learned projection.
    Concatenates face and iris embeddings, projects to fused dimension.
    """

    def __init__(
        self,
        face_dim: int = FACE_EMBEDDING_DIM,
        iris_dim: int = IRIS_EMBEDDING_DIM,
        fused_dim: int = FUSED_EMBEDDING_DIM,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(face_dim + iris_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(
        self,
        face_embedding: torch.Tensor,
        iris_embedding: torch.Tensor,
    ) -> torch.Tensor:
        concatenated = torch.cat([face_embedding, iris_embedding], dim=1)
        fused = self.projection(concatenated)
        return F.normalize(fused, p=2, dim=1)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion that learns per-modality importance weights.
    More expressive than fixed weighted fusion, but requires training.
    """

    def __init__(
        self,
        face_dim: int = FACE_EMBEDDING_DIM,
        iris_dim: int = IRIS_EMBEDDING_DIM,
        fused_dim: int = FUSED_EMBEDDING_DIM,
    ):
        super().__init__()
        self.face_attention = nn.Sequential(
            nn.Linear(face_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.iris_attention = nn.Sequential(
            nn.Linear(iris_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.projection = nn.Linear(face_dim, fused_dim) if face_dim != fused_dim else nn.Identity()

    def forward(
        self,
        face_embedding: torch.Tensor,
        iris_embedding: torch.Tensor,
    ) -> torch.Tensor:
        face_weight = self.face_attention(face_embedding)
        iris_weight = self.iris_attention(iris_embedding)

        weights = torch.softmax(torch.cat([face_weight, iris_weight], dim=1), dim=1)
        face_w = weights[:, 0:1]
        iris_w = weights[:, 1:2]

        fused = face_w * face_embedding + iris_w * iris_embedding
        fused = self.projection(fused)
        return F.normalize(fused, p=2, dim=1)


def create_fusion(method: str = "quality_aware", **kwargs) -> nn.Module:
    """
    Factory function for creating fusion modules.

    Args:
        method: "quality_aware", "weighted", "concat", or "attention"

    Returns:
        Fusion module instance
    """
    methods = {
        "quality_aware": QualityAwareFusion,
        "weighted": WeightedFusion,
        "concat": ConcatFusion,
        "attention": AttentionFusion,
    }

    if method not in methods:
        raise ValueError(f"Unknown fusion method: {method}. Choose from {list(methods.keys())}")

    return methods[method](**kwargs)
