"""
Face feature extractor using pretrained FaceNet (InceptionResnetV1).
Generates 512-dimensional L2-normalized embeddings from face images.

No training required — uses VGGFace2 pretrained weights (3.31M face images, 9131 identities).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import FACE_EMBEDDING_DIM, DEVICE


class FaceFeatureExtractor(nn.Module):
    """
    FaceNet-based face feature extractor.
    Uses InceptionResnetV1 pretrained on VGGFace2 for 512D embeddings.
    """

    def __init__(self, embedding_dim: int = FACE_EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim

        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained="vggface2").eval()
        self.model.to(DEVICE)

        # Freeze all parameters (no training needed)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding.

        Args:
            x: Input tensor (B, 3, H, W), normalized to [-1, 1]

        Returns:
            L2-normalized embedding (B, 512)
        """
        with torch.no_grad():
            embedding = self.model(x.to(DEVICE))
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def extract(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract embedding as numpy array.

        Args:
            face_tensor: Preprocessed face tensor (3, H, W) or (B, 3, H, W)

        Returns:
            Embedding numpy array (512,) or (B, 512)
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)

        embedding = self.forward(face_tensor)
        return embedding.cpu().numpy().squeeze()
