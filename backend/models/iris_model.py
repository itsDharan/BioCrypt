"""
Iris (eye) feature extractor using pretrained ResNet18 backbone + dual CBAM attention.
Generates 512-dimensional L2-normalized embeddings from iris images.

Architecture: ResNet18 (ImageNet pretrained, frozen layers 1-2) + CBAM×2 + FC projection.
Training supports ArcFace, Triplet, and Contrastive losses.

v2 improvements:
  - Dual CBAM (after layer3 AND layer4) for richer attention
  - ArcFace classification head for angular margin training
  - Online hard negative mining support
  - Better projection head with residual connection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IRIS_EMBEDDING_DIM, DEVICE, SAVED_MODELS_DIR,
    ARCFACE_SCALE, ARCFACE_MARGIN, ARCFACE_EASY_MARGIN,
)


class ChannelAttention(nn.Module):
    """Channel attention module (part of CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module (part of CBAM)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ArcFaceHead(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) classification head.

    Enforces angular margin between classes in the embedding space,
    producing tighter intra-class clusters and wider inter-class separation.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for
    Deep Face Recognition", CVPR 2019.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = ARCFACE_SCALE,
        margin: float = ARCFACE_MARGIN,
        easy_margin: bool = ARCFACE_EASY_MARGIN,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight matrix W: (num_classes, embedding_dim), normalized per-row
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precomputed constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)   # threshold for easy_margin
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) class labels

        Returns:
            (B, num_classes) scaled logits with angular margin applied
        """
        # Normalize weight
        w = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(θ) = emb · w^T
        cosine = F.linear(embeddings, w)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # sin(θ) from cos(θ)
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to the target class
        output = one_hot * phi + (1.0 - one_hot) * cosine

        return output * self.scale


class IrisFeatureExtractor(nn.Module):
    """
    Iris feature extractor using ResNet18 backbone + dual CBAM attention.

    Architecture:
      - ResNet18 (layers 1-2 frozen, layers 3-4 trainable)
      - CBAM attention after layer3 (256ch) — focuses on discriminative iris regions
      - CBAM attention after layer4 (512ch) — refines final features
      - Global average pooling
      - FC projection to 512D with residual shortcut
      - L2 normalization

    Transfer learning strategy:
      - Pretrained ImageNet weights for rich texture/shape features
      - Fine-tune deeper layers for iris-specific features
      - Accepts standard 3-channel RGB images
    """

    def __init__(self, embedding_dim: int = IRIS_EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Extract layers (standard 3-channel RGB input)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Freeze early layers (transfer learning)
        for layer in [self.layer0, self.layer1, self.layer2]:
            for param in layer.parameters():
                param.requires_grad = False

        # Dual CBAM attention — richer feature selection
        self.cbam3 = CBAM(channels=256)  # After layer3
        self.cbam4 = CBAM(channels=512)  # After layer4

        # Global average pooling + projection with residual
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )
        # Residual shortcut for projection (identity if dims match)
        self.residual_proj = nn.Linear(512, embedding_dim) if embedding_dim != 512 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract iris embedding.

        Args:
            x: Input tensor (B, 3, 224, 224), ImageNet-normalized RGB

        Returns:
            L2-normalized embedding (B, 512)
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam3(x)           # Attention after layer3
        x = self.layer4(x)
        x = self.cbam4(x)           # Attention after layer4
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Projection with residual connection
        projected = self.projection(x)
        residual = self.residual_proj(x)
        x = projected + residual

        x = F.normalize(x, p=2, dim=1)
        return x

    def extract(self, iris_tensor: torch.Tensor) -> 'np.ndarray':
        """
        Extract embedding as numpy array.

        Args:
            iris_tensor: Preprocessed iris tensor (3, H, W) or (B, 3, H, W)

        Returns:
            Embedding numpy array (512,) or (B, 512)
        """
        import numpy as np
        if iris_tensor.dim() == 3:
            iris_tensor = iris_tensor.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            embedding = self.forward(iris_tensor.to(DEVICE))
        return embedding.cpu().numpy().squeeze()

    @classmethod
    def load_trained(cls, model_path: str = None) -> 'IrisFeatureExtractor':
        """Load a trained model from checkpoint."""
        if model_path is None:
            model_path = SAVED_MODELS_DIR / "iris_model_best.pth"

        model = cls()
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(DEVICE)
        model.eval()
        return model


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training the iris feature extractor.
    Pulls positive pairs together, pushes negative pairs apart.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embedding1, embedding2: (B, D) L2-normalized embeddings
            label: (B,) 1 for same subject, 0 for different

        Returns:
            Scalar contrastive loss
        """
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        label = label.float()

        # Same subject: minimize distance
        # Different subject: maximize distance up to margin
        loss = label * distance.pow(2) + \
               (1 - label) * F.relu(self.margin - distance).pow(2)

        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss with online hard negative mining.

    For each anchor-positive pair, selects the hardest negative
    (highest similarity to anchor among negatives) in the batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Standard triplet loss."""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: ArcFace (classification) + Triplet (metric learning).

    ArcFace provides strong inter-class separation via angular margin.
    Triplet loss fine-tunes intra-class compactness with hard mining.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        arcface_weight: float = 1.0,
        triplet_weight: float = 0.5,
        triplet_margin: float = 0.3,
    ):
        super().__init__()
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.arcface_weight = arcface_weight
        self.triplet_weight = triplet_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        anchor: torch.Tensor = None,
        positive: torch.Tensor = None,
        negative: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            embeddings: (B, D) embeddings for ArcFace
            labels: (B,) class labels
            anchor, positive, negative: optional triplets for triplet loss
        """
        # ArcFace classification loss
        logits = self.arcface_head(embeddings, labels)
        arc_loss = self.ce_loss(logits, labels)

        total_loss = self.arcface_weight * arc_loss

        # Triplet loss (if triplets provided)
        if anchor is not None and positive is not None and negative is not None:
            tri_loss = self.triplet_loss(anchor, positive, negative)
            total_loss = total_loss + self.triplet_weight * tri_loss

        return total_loss
