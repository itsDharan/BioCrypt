"""
Diagnostic script: Shows WHY impostors pass authentication.
Compares face, iris, and fused embeddings separately to reveal
which modality is causing false acceptance.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config.settings import DEVICE, SAVED_MODELS_DIR, FUSION_ALPHA
from data.preprocessing import FacePreprocessor, IrisPreprocessor
from models.face_model import FaceFeatureExtractor
from models.iris_model import IrisFeatureExtractor
from models.fusion import create_fusion

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def main():
    print("=" * 70)
    print("BIOMETRIC SIMILARITY DIAGNOSTIC")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    face_pre = FacePreprocessor()
    iris_pre = IrisPreprocessor()
    face_model = FaceFeatureExtractor()
    face_model.eval()

    iris_model_path = SAVED_MODELS_DIR / "iris_model_best.pth"
    if iris_model_path.exists():
        iris_model = IrisFeatureExtractor.load_trained(str(iris_model_path))
        print("  Iris model: TRAINED (loaded from checkpoint)")
    else:
        iris_model = IrisFeatureExtractor().to(DEVICE)
        iris_model.eval()
        print("  Iris model: UNTRAINED (ImageNet only) ← THIS IS A PROBLEM")

    fusion = create_fusion("weighted")

    # Get test images from user
    print("\n" + "-" * 70)
    print("Provide TWO sets of images to compare:")
    print("-" * 70)

    face_path_1 = input("Person A — Face image path: ").strip().strip('"')
    iris_path_1 = input("Person A — Iris image path:  ").strip().strip('"')
    face_path_2 = input("Person B — Face image path: ").strip().strip('"')
    iris_path_2 = input("Person B — Iris image path:  ").strip().strip('"')

    # Extract features
    print("\nExtracting features...")

    face_t1 = face_pre.preprocess(face_path_1)
    face_t2 = face_pre.preprocess(face_path_2)
    iris_t1 = iris_pre.preprocess(iris_path_1)
    iris_t2 = iris_pre.preprocess(iris_path_2)

    if face_t1 is None or face_t2 is None:
        print("ERROR: Face detection failed!")
        return

    face_emb1 = face_model.extract(face_t1)
    face_emb2 = face_model.extract(face_t2)
    iris_emb1 = iris_model.extract(iris_t1)
    iris_emb2 = iris_model.extract(iris_t2)

    # Fuse
    fused1 = fusion.fuse_numpy(face_emb1, iris_emb1)
    fused2 = fusion.fuse_numpy(face_emb2, iris_emb2)

    # Compute similarities
    face_sim = cosine_sim(face_emb1, face_emb2)
    iris_sim = cosine_sim(iris_emb1, iris_emb2)
    fused_sim = cosine_sim(fused1, fused2)

    # Cross-modality analysis
    face_iris_cross1 = cosine_sim(face_emb1, iris_emb1)
    face_iris_cross2 = cosine_sim(face_emb2, iris_emb2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n  Face  similarity (A vs B): {face_sim:.4f}")
    print(f"  Iris  similarity (A vs B): {iris_sim:.4f}")
    print(f"  Fused similarity (A vs B): {fused_sim:.4f}")

    print(f"\n  Face-Iris cross (A): {face_iris_cross1:.4f}")
    print(f"  Face-Iris cross (B): {face_iris_cross2:.4f}")

    print(f"\n  Fusion alpha (face weight): {FUSION_ALPHA}")

    print("\n" + "-" * 70)
    print("DIAGNOSIS:")
    print("-" * 70)

    if face_sim > 0.85 and iris_sim > 0.85:
        print("  Both modalities are similar — these may be the SAME person.")
    elif face_sim > 0.75 and iris_sim < 0.5:
        print("  ⚠ SAME FACE, DIFFERENT IRIS detected!")
        print(f"  The face similarity ({face_sim:.4f}) is high but iris ({iris_sim:.4f}) is low.")
        print(f"  However, fused similarity = {fused_sim:.4f}")
        if fused_sim > 0.75:
            print("  → Face DOMINATES the fusion, making fused score pass threshold.")
            print("  → FIX: Must check face and iris INDEPENDENTLY, not just fused.")
        else:
            print("  → Fused score is below threshold — should be rejected correctly.")
    elif face_sim < 0.5 and iris_sim > 0.75:
        print("  ⚠ DIFFERENT FACE, SAME/SIMILAR IRIS detected!")
    elif iris_sim > 0.7:
        print(f"  ⚠ Iris similarity is suspiciously high ({iris_sim:.4f})!")
        print("  The iris model may NOT be producing discriminative embeddings.")
        print("  Different people's irises should have similarity < 0.5")
    else:
        print("  Embeddings appear discriminative. Check threshold settings.")

    print(f"\n  With threshold 0.75:")
    print(f"    Face  → {'PASS ✓' if face_sim >= 0.75 else 'REJECT ✗'}")
    print(f"    Iris  → {'PASS ✓' if iris_sim >= 0.75 else 'REJECT ✗'}")
    print(f"    Fused → {'PASS ✓' if fused_sim >= 0.75 else 'REJECT ✗'}")

    if fused_sim >= 0.75 and iris_sim < 0.75:
        print("\n  ★ ROOT CAUSE: Fused embedding passes but iris alone doesn't!")
        print("  ★ The system needs INDEPENDENT per-modality checks.")

if __name__ == "__main__":
    main()
