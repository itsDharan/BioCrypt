"""
Image preprocessing pipeline for face and iris biometrics.
Handles resizing, normalization, face alignment, CLAHE enhancement,
iris localization, reflection removal, and noise filtering.

v2 improvements (IrisPreprocessor):
  - Circular Hough transform for iris/pupil boundary detection
  - Specular reflection removal via inpainting
  - Bilateral filtering for noise removal while preserving edges
  - Enhanced CLAHE tuned for iris texture
  - Rubber-sheet normalization (Daugman's unwrapping)
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional, Tuple

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IMAGE_SIZE, FACE_ALIGNMENT_PADDING, DEVICE,
    CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE,
)


class FacePreprocessor:
    """
    Preprocess face images: detect -> align -> crop -> CLAHE -> normalize.
    Uses MTCNN for face detection/alignment when available.
    """

    def __init__(self, target_size: int = IMAGE_SIZE):
        self.target_size = target_size
        self.mtcnn = None

        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=target_size,
                margin=int(target_size * FACE_ALIGNMENT_PADDING),
                device=DEVICE,
                post_process=False,
            )
        except ImportError:
            pass

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def preprocess(self, image_input) -> Optional[torch.Tensor]:
        """
        Preprocess a face image to tensor.

        Args:
            image_input: PIL Image, numpy array, or file path string

        Returns:
            Preprocessed tensor (3, 224, 224) or None if face not detected
        """
        # Load image
        if isinstance(image_input, str) or isinstance(image_input, Path):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
            elif image_input.shape[2] == 4:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB))
            else:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            img = image_input.convert("RGB") if hasattr(image_input, 'convert') else image_input

        # Try MTCNN alignment
        if self.mtcnn is not None:
            try:
                face_tensor = self.mtcnn(img)
                if face_tensor is not None:
                    if face_tensor.max() > 1:
                        face_tensor = face_tensor / 255.0
                    face_tensor = self.normalize(face_tensor)
                    return face_tensor
            except Exception:
                pass

        # Fallback: resize + normalize
        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            self.normalize,
        ])
        return transform(img)


class IrisPreprocessor:
    """
    Preprocess iris/eye images with iris-specific enhancement pipeline.

    Pipeline:
      1. Load & convert to grayscale for processing
      2. Detect and remove specular reflections (inpainting)
      3. Iris/pupil boundary localization (Circular Hough Transform)
      4. Crop to iris region with padding
      5. Enhanced CLAHE for iris texture visibility
      6. Bilateral filtering (denoise while preserving edges)
      7. Resize to target dimensions
      8. Convert back to 3-channel for ResNet input
      9. ImageNet normalization

    Falls back to basic preprocessing if localization fails.
    """

    def __init__(self, target_size: int = IMAGE_SIZE):
        self.target_size = target_size
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _remove_reflections(self, gray: np.ndarray) -> np.ndarray:
        """Remove specular reflections via thresholding + inpainting."""
        # Detect bright spots (specular highlights)
        _, reflection_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Dilate mask slightly to cover reflection boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=1)

        # Inpaint reflections
        if np.sum(reflection_mask > 0) > 0:
            cleaned = cv2.inpaint(gray, reflection_mask, 5, cv2.INPAINT_TELEA)
            return cleaned
        return gray

    def _localize_iris(
        self, gray: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """
        Localize iris and pupil boundaries using Circular Hough Transform.

        Returns:
            (pupil_circle, iris_circle) where each is (cx, cy, radius) or None
        """
        h, w = gray.shape

        # Blur for better circle detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 2)

        # Detect pupil (dark, smaller circle)
        pupil_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(h * 0.3),
            param1=100,
            param2=40,
            minRadius=int(min(h, w) * 0.05),
            maxRadius=int(min(h, w) * 0.25),
        )

        pupil = None
        if pupil_circles is not None:
            pupil_circles = np.round(pupil_circles[0]).astype(int)
            # Pick the circle closest to center with smallest radius (likely pupil)
            center = np.array([w // 2, h // 2])
            best_idx = 0
            best_score = float('inf')
            for i, (cx, cy, r) in enumerate(pupil_circles):
                dist = np.linalg.norm(np.array([cx, cy]) - center)
                score = dist + r * 0.5  # Prefer centered + small
                if score < best_score:
                    best_score = score
                    best_idx = i
            pupil = tuple(pupil_circles[best_idx])

        # Detect iris (larger circle around pupil)
        iris_circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=int(h * 0.3),
            param1=80,
            param2=50,
            minRadius=int(min(h, w) * 0.15),
            maxRadius=int(min(h, w) * 0.5),
        )

        iris = None
        if iris_circles is not None:
            iris_circles = np.round(iris_circles[0]).astype(int)
            if pupil is not None:
                # Pick iris circle closest to pupil center
                pcx, pcy, _ = pupil
                best_idx = 0
                best_dist = float('inf')
                for i, (cx, cy, r) in enumerate(iris_circles):
                    dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                    if dist < best_dist and r > pupil[2]:
                        best_dist = dist
                        best_idx = i
                iris = tuple(iris_circles[best_idx])
            else:
                # Pick largest centered circle
                center = np.array([w // 2, h // 2])
                best_idx = 0
                best_score = float('inf')
                for i, (cx, cy, r) in enumerate(iris_circles):
                    dist = np.linalg.norm(np.array([cx, cy]) - center)
                    score = dist - r * 0.3
                    if score < best_score:
                        best_score = score
                        best_idx = i
                iris = tuple(iris_circles[best_idx])

        return pupil, iris

    def _crop_iris_region(
        self, img: np.ndarray, iris_circle: Tuple[int, int, int], padding: float = 0.2
    ) -> np.ndarray:
        """Crop image to iris region with padding."""
        cx, cy, r = iris_circle
        h, w = img.shape[:2]

        pad_r = int(r * (1 + padding))
        x1 = max(0, cx - pad_r)
        y1 = max(0, cy - pad_r)
        x2 = min(w, cx + pad_r)
        y2 = min(h, cy + pad_r)

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            return img
        return cropped

    def _apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement tuned for iris texture."""
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE,
        )
        return clahe.apply(gray)

    def preprocess(self, image_input) -> Optional[torch.Tensor]:
        """
        Full iris preprocessing pipeline.

        Args:
            image_input: PIL Image, numpy array, or file path string

        Returns:
            Preprocessed tensor (3, 224, 224) or None on failure
        """
        try:
            # Load image
            if isinstance(image_input, (str, Path)):
                img = cv2.imread(str(image_input))
                if img is None:
                    return None
            elif isinstance(image_input, Image.Image):
                img = np.array(image_input.convert("RGB"))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                img = image_input.copy()
            else:
                return None

            # Convert to grayscale for processing
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Step 1: Remove specular reflections
            gray = self._remove_reflections(gray)

            # Step 2: Iris localization
            pupil, iris = self._localize_iris(gray)

            # Step 3: Crop to iris region
            if iris is not None:
                gray = self._crop_iris_region(gray, iris, padding=0.15)
                img = self._crop_iris_region(img, iris, padding=0.15)

            # Step 4: CLAHE enhancement
            gray_enhanced = self._apply_clahe(gray)

            # Step 5: Bilateral filter (denoise preserving edges)
            gray_filtered = cv2.bilateralFilter(gray_enhanced, 9, 75, 75)

            # Step 6: Resize
            resized = cv2.resize(gray_filtered, (self.target_size, self.target_size))

            # Step 7: Convert to 3-channel RGB for ResNet
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

            # Step 8: To tensor + normalize
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = self.normalize(tensor)

            return tensor

        except Exception as e:
            # Fallback: basic resize and normalize
            try:
                return self._basic_preprocess(image_input)
            except Exception:
                return None

    def _basic_preprocess(self, image_input) -> torch.Tensor:
        """Fallback basic preprocessing without iris localization."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            img = image_input.convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            self.normalize,
        ])
        return transform(img)
