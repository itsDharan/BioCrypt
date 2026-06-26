"""
PyTorch Dataset classes for LFW face and iris biometrics.
Supports multimodal pairing, contrastive pair generation, and train/val/test splits.

Dataset loading strategies:
  - LFW: Subject subdirectories (standard format)
  - Iris: Flat directory with subject prefix in filename (e.g., "1-IMG_xxx.JPG")
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import FACE_DATASET_DIR, IRIS_DATASET_DIR, IMAGE_SIZE
from data.preprocessing import FacePreprocessor, IrisPreprocessor


class LFWDataset(Dataset):
    """
    LFW (Labeled Faces in the Wild) dataset loader.
    Expects directory structure: lfw-deepfunneled/<subject_name>/<image>.jpg
    """

    def __init__(
        self,
        root_dir: str = None,
        min_samples: int = 2,
        transform=None,
    ):
        self.root_dir = Path(root_dir or FACE_DATASET_DIR)
        self.preprocessor = FacePreprocessor()
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []      # (path, subject_id)
        self.subjects: Dict[int, List[str]] = {}       # subject_id -> [paths]
        self.subject_names: Dict[int, str] = {}        # subject_id -> name
        self._load_dataset(min_samples)

    def _load_dataset(self, min_samples: int):
        """Load LFW face images from subject subdirectories."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        if not self.root_dir.exists():
            print(f"Warning: LFW dataset not found at {self.root_dir}")
            return

        subject_id = 0
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            images = [
                str(f) for f in sorted(subject_dir.iterdir())
                if f.suffix.lower() in extensions
            ]

            if len(images) >= min_samples:
                self.subjects[subject_id] = images
                self.subject_names[subject_id] = subject_dir.name
                for img_path in images:
                    self.samples.append((img_path, subject_id))
                subject_id += 1

        print(f"LFW: Loaded {len(self.samples)} images from {len(self.subjects)} subjects")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, subject_id = self.samples[idx]
        tensor = self.preprocessor.preprocess(img_path)
        if tensor is None:
            # Fallback: return zeros if face detection fails
            tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        return tensor, subject_id


class IrisDataset(Dataset):
    """
    Iris/eye image dataset loader for the CASIA eye dataset.
    Supports two loading strategies:
      1. Subject prefix in filename: "1-IMG_xxx.JPG" where "1" is the subject ID
      2. Subject subdirectories: each subfolder = one subject
    
    If only one subject is found (as in the CASIA-Diabetes dataset),
    synthetic subjects are created by grouping images for contrastive learning.
    """

    IMAGES_PER_SUBJECT = 18  # For synthetic subject grouping

    def __init__(
        self,
        root_dir: str = None,
        min_samples: int = 2,
        transform=None,
    ):
        self.root_dir = Path(root_dir or IRIS_DATASET_DIR)
        self.preprocessor = IrisPreprocessor()
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.subjects: Dict[int, List[str]] = {}
        self.subject_names: Dict[int, str] = {}
        self._load_dataset(min_samples)

    def _load_dataset(self, min_samples: int):
        """Load iris images using best available strategy."""
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        if not self.root_dir.exists():
            print(f"Warning: Iris dataset not found at {self.root_dir}")
            return

        # Strategy 1: Check for subject subdirectories
        subdirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if subdirs and not any(f.suffix.lower() in extensions for f in self.root_dir.iterdir() if f.is_file()):
            self._load_from_subject_dirs(subdirs, extensions, min_samples)
        else:
            # Strategy 2: Flat directory — parse subject ID from filename prefix
            self._load_from_flat_dir(extensions, min_samples)

    def _load_from_subject_dirs(self, subdirs, extensions, min_samples):
        """Load from subject-organized subdirectories (e.g., 001/, 002/)."""
        subject_id = 0
        for subject_dir in sorted(subdirs):
            images = [
                str(f) for f in sorted(subject_dir.rglob("*"))
                if f.suffix.lower() in extensions
            ]

            if len(images) >= min_samples:
                self.subjects[subject_id] = images
                self.subject_names[subject_id] = subject_dir.name
                for img_path in images:
                    self.samples.append((img_path, subject_id))
                subject_id += 1

        print(f"Iris: Loaded {len(self.samples)} images from {len(self.subjects)} subjects (subdirs)")

    def _load_from_flat_dir(self, extensions, min_samples):
        """
        Load from flat directory. Parse subject from filename prefix (e.g., '1-IMG_xxx.JPG').
        If only one real subject exists, create synthetic subjects by grouping images.
        """
        all_images = sorted([
            f for f in self.root_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])

        if not all_images:
            print("Iris: No images found")
            return

        # Try to parse subject IDs from filename prefix (e.g., "1-xxx.JPG" -> subject "1")
        subject_map: Dict[str, List[str]] = {}
        for f in all_images:
            parts = f.name.split("-", 1)
            if len(parts) >= 2:
                subject_key = parts[0]
            else:
                subject_key = "0"
            subject_map.setdefault(subject_key, []).append(str(f))

        if len(subject_map) >= 5:
            # Multiple real subjects — use as-is
            subject_id = 0
            for key in sorted(subject_map.keys()):
                images = subject_map[key]
                if len(images) >= min_samples:
                    self.subjects[subject_id] = images
                    self.subject_names[subject_id] = f"iris_subject_{key}"
                    for img_path in images:
                        self.samples.append((img_path, subject_id))
                    subject_id += 1
            print(f"Iris: Loaded {len(self.samples)} images from {len(self.subjects)} subjects (prefix)")
        else:
            # Single or few subjects — create synthetic subjects for contrastive learning
            all_image_paths = [str(f) for f in all_images]
            random.seed(42)  # Reproducible grouping
            random.shuffle(all_image_paths)
            self._assign_synthetic_subjects(all_image_paths, min_samples)

    def _assign_synthetic_subjects(self, all_images, min_samples):
        """Group images into synthetic subjects (every IMAGES_PER_SUBJECT images)."""
        if not all_images:
            print("Iris: No images found")
            return

        subject_id = 0
        for i in range(0, len(all_images), self.IMAGES_PER_SUBJECT):
            group = all_images[i : i + self.IMAGES_PER_SUBJECT]
            if len(group) >= min_samples:
                self.subjects[subject_id] = group
                self.subject_names[subject_id] = f"iris_subject_{subject_id:04d}"
                for img_path in group:
                    self.samples.append((img_path, subject_id))
                subject_id += 1

        print(f"Iris: Loaded {len(self.samples)} images from {len(self.subjects)} synthetic subjects")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, subject_id = self.samples[idx]
        tensor = self.preprocessor.preprocess(img_path)
        return tensor, subject_id


class ContrastivePairDataset(Dataset):
    """
    Generates contrastive pairs (positive/negative) from a base dataset.
    Used for training the iris feature extractor.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_pairs: int = 5000,
        positive_ratio: float = 0.5,
    ):
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        self.positive_ratio = positive_ratio
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """Generate (idx1, idx2, label) pairs. label=1 for same subject, 0 for different."""
        subjects = self.base_dataset.subjects
        if len(subjects) < 2:
            return []

        subject_ids = list(subjects.keys())
        pairs = []
        num_positive = int(self.num_pairs * self.positive_ratio)

        # Positive pairs (same subject)
        for _ in range(num_positive):
            sid = random.choice(subject_ids)
            while len(subjects[sid]) < 2:
                sid = random.choice(subject_ids)
            img1, img2 = random.sample(subjects[sid], 2)
            idx1 = next(i for i, (p, s) in enumerate(self.base_dataset.samples) if p == img1)
            idx2 = next(i for i, (p, s) in enumerate(self.base_dataset.samples) if p == img2)
            pairs.append((idx1, idx2, 1))

        # Negative pairs (different subjects)
        for _ in range(self.num_pairs - num_positive):
            sid1, sid2 = random.sample(subject_ids, 2)
            img1 = random.choice(subjects[sid1])
            img2 = random.choice(subjects[sid2])
            idx1 = next(i for i, (p, s) in enumerate(self.base_dataset.samples) if p == img1)
            idx2 = next(i for i, (p, s) in enumerate(self.base_dataset.samples) if p == img2)
            pairs.append((idx1, idx2, 0))

        random.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        idx1, idx2, label = self.pairs[idx]
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        return img1, img2, label
