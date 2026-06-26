"""
Training script for the iris feature extraction model (ResNet18 + CBAM backbone).

v2 improvements:
  - ArcFace loss (primary) + Triplet loss (auxiliary) combined training
  - Online hard negative mining within each batch
  - Data augmentation: rotation, flip, color jitter, blur, random erasing
  - Cosine annealing warm-restart LR schedule
  - Embedding quality monitoring (intra-class var, inter-class dist, d-prime)
  - Subject-aware batch sampling for ArcFace training

Usage:
    python models/train.py
    python models/train.py --epochs 20 --batch-size 32
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DEVICE, LEARNING_RATE, SAVED_MODELS_DIR,
    IRIS_DATASET_DIR, CONTRASTIVE_MARGIN, TRIPLET_MARGIN,
    ARCFACE_SCALE, ARCFACE_MARGIN,
    TRAIN_NUM_PAIRS, TRAIN_POSITIVE_RATIO, TRAIN_HARD_MINING,
    TRAIN_EPOCHS, AUGMENT_ROTATION, AUGMENT_FLIP,
    AUGMENT_COLOR_JITTER, AUGMENT_GAUSSIAN_BLUR, AUGMENT_RANDOM_ERASING,
)
from data.dataset import IrisDataset
from models.iris_model import (
    IrisFeatureExtractor, CombinedLoss, ContrastiveLoss, TripletLoss,
)


# ── Data Augmentation Pipeline ────────────────────────────────────────

def build_augmentation_transform(target_size=224):
    """Build training augmentation pipeline for iris images."""
    aug_list = [
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
    ]

    if AUGMENT_ROTATION:
        aug_list.append(transforms.RandomRotation(AUGMENT_ROTATION))

    if AUGMENT_FLIP:
        aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if AUGMENT_COLOR_JITTER:
        aug_list.append(transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        ))

    if AUGMENT_GAUSSIAN_BLUR:
        aug_list.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        ], p=0.3))

    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if AUGMENT_RANDOM_ERASING:
        aug_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)))

    return transforms.Compose(aug_list)


# ── Subject-Aware Batch Dataset for ArcFace ───────────────────────────

class ArcFaceBatchDataset(Dataset):
    """
    Dataset that yields (image_tensor, subject_label) pairs.
    Subject labels are 0-indexed integers for ArcFace classification.
    """

    def __init__(self, iris_dataset: IrisDataset, augment: bool = True):
        self.iris_dataset = iris_dataset
        self.subjects = iris_dataset.subjects  # subject_id -> [paths]

        # Build flat list of (path, label_idx) pairs
        self.samples = []
        self.subject_to_idx = {}
        self.idx_to_subject = {}

        for idx, (sid, paths) in enumerate(sorted(self.subjects.items())):
            self.subject_to_idx[sid] = idx
            self.idx_to_subject[idx] = sid
            for path in paths:
                self.samples.append((path, idx))

        self.num_classes = len(self.subject_to_idx)

        # Augmentation
        if augment:
            self.transform = build_augmentation_transform()
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image as PILImage
        try:
            img = PILImage.open(path).convert("RGB")
            img_np = np.array(img)
            tensor = self.transform(img_np)
        except Exception:
            # Return a zero tensor on failure
            tensor = torch.zeros(3, 224, 224)
        return tensor, label


# ── Hard Negative Mining ──────────────────────────────────────────────

def mine_hard_triplets(embeddings, labels, num_triplets=None):
    """
    Online hard negative mining within a batch.

    For each anchor-positive pair, finds the hardest negative
    (closest to anchor among different-class samples).

    Args:
        embeddings: (B, D) L2-normalized embeddings
        labels: (B,) class labels

    Returns:
        (anchor_idx, positive_idx, negative_idx) tuples
    """
    B = embeddings.size(0)
    device = embeddings.device

    # Pairwise cosine similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.t())  # (B, B)

    triplets = []
    label_np = labels.cpu().numpy()

    for i in range(B):
        # Find positives (same class, different sample)
        pos_mask = (label_np == label_np[i]) & (np.arange(B) != i)
        neg_mask = (label_np != label_np[i])

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        for pos_idx in pos_indices:
            # Hardest negative: highest similarity to anchor among negatives
            neg_sims = sim_matrix[i, neg_indices]
            hardest_neg_local = torch.argmax(neg_sims).item()
            hardest_neg_idx = neg_indices[hardest_neg_local]

            triplets.append((i, pos_idx, hardest_neg_idx))

    if num_triplets and len(triplets) > num_triplets:
        triplets = random.sample(triplets, num_triplets)

    return triplets


# ── Embedding Quality Metrics ─────────────────────────────────────────

def compute_embedding_quality(embeddings, labels):
    """
    Compute embedding quality metrics for monitoring training.

    Returns:
        dict with intra_class_variance, inter_class_distance, d_prime
    """
    label_np = labels.cpu().numpy()
    emb_np = embeddings.detach().cpu().numpy()

    unique_labels = np.unique(label_np)
    if len(unique_labels) < 2:
        return {"intra_var": 0.0, "inter_dist": 0.0, "d_prime": 0.0}

    # Intra-class: average pairwise distance within each class
    intra_dists = []
    class_centroids = {}
    for lab in unique_labels:
        mask = label_np == lab
        class_embs = emb_np[mask]
        if len(class_embs) >= 2:
            centroid = class_embs.mean(axis=0)
            class_centroids[lab] = centroid
            dists = np.linalg.norm(class_embs - centroid, axis=1)
            intra_dists.extend(dists.tolist())

    # Inter-class: average distance between class centroids
    inter_dists = []
    centroid_list = list(class_centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            d = np.linalg.norm(centroid_list[i] - centroid_list[j])
            inter_dists.append(d)

    intra_var = np.mean(intra_dists) if intra_dists else 0.0
    inter_dist = np.mean(inter_dists) if inter_dists else 0.0

    # d-prime (decidability index)
    if intra_var > 0:
        d_prime = inter_dist / (intra_var + 1e-8)
    else:
        d_prime = 0.0

    return {
        "intra_var": float(intra_var),
        "inter_dist": float(inter_dist),
        "d_prime": float(d_prime),
    }


# ── Main Training Function ───────────────────────────────────────────

def train_iris_model(
    dataset_dir: str = None,
    epochs: int = TRAIN_EPOCHS,
    lr: float = LEARNING_RATE,
    save_name: str = "iris_model_best.pth",
    batch_size: int = 32,
    use_arcface: bool = True,
):
    """
    Train the iris feature extractor using ArcFace + Triplet combined loss.

    Args:
        dataset_dir: Path to iris dataset
        epochs: Number of training epochs
        lr: Base learning rate
        save_name: Filename for saved model
        batch_size: Training batch size
        use_arcface: Whether to use ArcFace (True) or basic contrastive (False)
    """
    print(f"Device: {DEVICE}")
    print(f"Training iris model for {epochs} epochs, lr={lr}")
    print(f"Loss: {'ArcFace + Triplet' if use_arcface else 'Contrastive'}")
    print(f"Hard mining: {TRAIN_HARD_MINING}")

    # Load dataset
    base_dataset = IrisDataset(root_dir=dataset_dir)
    if len(base_dataset) == 0:
        print("Error: No data loaded. Check dataset path.")
        return None, None

    num_subjects = len(base_dataset.subjects)
    print(f"Dataset: {len(base_dataset)} images from {num_subjects} subjects")

    if use_arcface and num_subjects < 2:
        print("Warning: ArcFace requires >=2 subjects. Falling back to contrastive.")
        use_arcface = False

    # Create datasets
    if use_arcface:
        train_dataset = ArcFaceBatchDataset(base_dataset, augment=True)
        val_dataset = ArcFaceBatchDataset(base_dataset, augment=False)

        # Split into train/val (85/15)
        val_size = max(1, int(len(train_dataset) * 0.15))
        train_size = len(train_dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
            drop_last=True,  # Important for BatchNorm with small batches
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
            drop_last=False,
        )

        num_classes = train_dataset.num_classes
        print(f"ArcFace classes: {num_classes}")
    else:
        # Fallback to contrastive pairs
        from data.dataset import ContrastivePairDataset
        pair_dataset = ContrastivePairDataset(
            base_dataset, num_pairs=TRAIN_NUM_PAIRS,
            positive_ratio=TRAIN_POSITIVE_RATIO,
        )
        val_size = max(1, int(len(pair_dataset) * 0.15))
        train_size = len(pair_dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            pair_dataset, [train_size, val_size]
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        num_classes = 0

    print(f"Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} samples ({len(val_loader)} batches)")

    # Model
    model = IrisFeatureExtractor().to(DEVICE)

    # Loss
    if use_arcface:
        criterion = CombinedLoss(
            num_classes=num_classes,
            arcface_weight=1.0,
            triplet_weight=0.5,
            triplet_margin=TRIPLET_MARGIN,
        ).to(DEVICE)
    else:
        criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)

    # Optimizer with differential learning rates
    backbone_params = list(model.layer3.parameters()) + list(model.layer4.parameters())
    attention_params = list(model.cbam3.parameters()) + list(model.cbam4.parameters())
    new_params = list(model.projection.parameters()) + list(model.residual_proj.parameters())

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},        # Pretrained: slower
        {"params": attention_params, "lr": lr * 0.5},        # CBAM: medium
        {"params": new_params, "lr": lr},                     # New layers: full LR
    ]

    if use_arcface:
        param_groups.append({"params": criterion.parameters(), "lr": lr})

    optimizer = optim.Adam(param_groups, weight_decay=1e-5)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float("inf")
    best_d_prime = 0.0
    history = {"train_loss": [], "val_loss": [], "d_prime": []}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        if use_arcface:
            criterion.train()
        train_loss = 0.0
        t0 = time.time()
        all_embeddings = []
        all_labels = []

        if use_arcface:
            # ArcFace training loop
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                embeddings = model(images)

                # Mine hard triplets within batch
                triplet_kwargs = {}
                if TRAIN_HARD_MINING and embeddings.size(0) >= 4:
                    triplets = mine_hard_triplets(embeddings.detach(), labels)
                    if len(triplets) >= 2:
                        t_indices = random.sample(triplets, min(len(triplets), batch_size // 2))
                        a_idx = [t[0] for t in t_indices]
                        p_idx = [t[1] for t in t_indices]
                        n_idx = [t[2] for t in t_indices]
                        triplet_kwargs = {
                            "anchor": embeddings[a_idx],
                            "positive": embeddings[p_idx],
                            "negative": embeddings[n_idx],
                        }

                loss = criterion(embeddings, labels, **triplet_kwargs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

                # Collect for quality metrics
                all_embeddings.append(embeddings.detach())
                all_labels.append(labels.detach())

                if (batch_idx + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    avg_time = elapsed / (batch_idx + 1)
                    remaining = avg_time * (len(train_loader) - batch_idx - 1)
                    print(
                        f"  Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"ETA: {remaining:.0f}s"
                    )
        else:
            # Contrastive training loop
            for batch_idx, (img1, img2, label) in enumerate(train_loader):
                img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                label = label.to(DEVICE)

                emb1 = model(img1)
                emb2 = model(img2)
                loss = criterion(emb1, emb2, label)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

                if (batch_idx + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    avg_time = elapsed / (batch_idx + 1)
                    remaining = avg_time * (len(train_loader) - batch_idx - 1)
                    print(
                        f"  Epoch {epoch+1}/{epochs} | "
                        f"Batch {batch_idx+1}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"ETA: {remaining:.0f}s"
                    )

        train_loss /= max(1, len(train_loader))

        # ── Compute Embedding Quality ──
        quality = {"d_prime": 0.0}
        if use_arcface and all_embeddings:
            cat_emb = torch.cat(all_embeddings[:50], dim=0)  # Limit for speed
            cat_lab = torch.cat(all_labels[:50], dim=0)
            quality = compute_embedding_quality(cat_emb, cat_lab)

        # ── Validate ──
        model.eval()
        if use_arcface:
            criterion.eval()
        val_loss = 0.0
        with torch.no_grad():
            if use_arcface:
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    embeddings = model(images)
                    loss = criterion(embeddings, labels)
                    val_loss += loss.item()
            else:
                for img1, img2, label in val_loader:
                    img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                    label = label.to(DEVICE)
                    emb1 = model(img1)
                    emb2 = model(img2)
                    loss = criterion(emb1, emb2, label)
                    val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        scheduler.step()

        epoch_time = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["d_prime"].append(quality.get("d_prime", 0.0))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"d': {quality.get('d_prime', 0):.2f} | "
            f"Time: {epoch_time:.0f}s"
        )

        # Save best model (prefer better d-prime, fallback to val loss)
        save_this = False
        if quality.get("d_prime", 0) > best_d_prime and quality.get("d_prime", 0) > 0:
            best_d_prime = quality["d_prime"]
            save_this = True
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            save_this = True

        if save_this:
            save_path = SAVED_MODELS_DIR / save_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "d_prime": quality.get("d_prime", 0),
                "architecture": "resnet18_dual_cbam_iris_v2",
                "num_classes": num_classes,
                "loss_type": "arcface+triplet" if use_arcface else "contrastive",
            }, save_path)
            print(f"  [SAVED] best model (val_loss={val_loss:.4f}, d'={quality.get('d_prime', 0):.2f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}, Best d': {best_d_prime:.2f}")
    return model, history


if __name__ == "__main__":
    train_iris_model()
