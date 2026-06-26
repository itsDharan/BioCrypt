"""
Full evaluation script v4: trains models, runs authentication pipeline,
computes biometric metrics (FAR, FRR, EER, d-prime, min-DCF, ROC, DET),
and generates publication-quality reports.

v4 improvements:
  - d-prime and min-DCF metrics
  - Z-score normalization of similarity scores
  - Platt score calibration
  - Bootstrap confidence intervals for EER
  - Confusion matrix generation
  - Threshold sweep analysis (FAR/FRR crossover)
  - Before/after comparison with baseline
  - Enhanced reporting with all new metrics

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --skip-training
    python evaluation/run_evaluation.py --num-genuine 2000 --num-impostor 2000
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DEVICE, SAVED_MODELS_DIR, EVALUATION_DIR,
    FACE_DATASET_DIR, IRIS_DATASET_DIR,
    FUSION_ALPHA, SCORE_NORMALIZATION,
    TARGET_ACCURACY, TARGET_FAR, TARGET_FRR, TARGET_EER,
    TARGET_VAULT_TAR, TARGET_VAULT_TRR,
)
from data.dataset import LFWDataset, IrisDataset
from data.preprocessing import FacePreprocessor, IrisPreprocessor
from models.face_model import FaceFeatureExtractor
from models.iris_model import IrisFeatureExtractor
from models.fusion import WeightedFusion
from crypto.fuzzy_vault import ImprovedFuzzyVault
from evaluation.metrics import BiometricEvaluator

# ── Previous baseline metrics (for comparison) ──
PREVIOUS_METRICS = {
    "accuracy": 0.9478,
    "far": 0.0525,
    "frr": 0.0520,
    "eer": 0.0522,
    "d_prime": 0.0,
    "eer_threshold": 0.4443,
    "score_separation": 0.6208,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Biometric System Evaluation v4")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip iris model training")
    parser.add_argument("--num-genuine", type=int, default=2000,
                        help="Number of genuine pairs to evaluate")
    parser.add_argument("--num-impostor", type=int, default=2000,
                        help="Number of impostor pairs to evaluate")
    parser.add_argument("--face-dir", type=str, default=None,
                        help="Face dataset directory")
    parser.add_argument("--iris-dir", type=str, default=None,
                        help="Iris dataset directory")
    parser.add_argument("--vault-trials", type=int, default=100,
                        help="Number of fuzzy vault test trials (default 100)")
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="Number of bootstrap samples for EER CI")
    return parser.parse_args()


def extract_embeddings(face_dataset, iris_dataset, face_model, iris_model,
                       face_preprocessor, iris_preprocessor, max_subjects=None):
    """Extract embeddings for all subjects."""
    print("\nExtracting embeddings...")

    # Get shared subject count
    num_face_subjects = len(face_dataset.subjects)
    num_iris_subjects = len(iris_dataset.subjects)
    num_subjects = min(num_face_subjects, num_iris_subjects)
    if max_subjects:
        num_subjects = min(num_subjects, max_subjects)

    print(f"  Face subjects: {num_face_subjects}")
    print(f"  Iris subjects: {num_iris_subjects}")
    print(f"  Using: {num_subjects} paired subjects")

    # Extract embeddings per subject
    subject_embeddings = {}  # subject_id -> list of fused embeddings
    fusion = WeightedFusion(alpha=FUSION_ALPHA)

    face_subject_ids = sorted(face_dataset.subjects.keys())[:num_subjects]
    iris_subject_ids = sorted(iris_dataset.subjects.keys())[:num_subjects]

    for i in range(num_subjects):
        face_sid = face_subject_ids[i]
        iris_sid = iris_subject_ids[i]

        face_images = face_dataset.subjects[face_sid]
        iris_images = iris_dataset.subjects[iris_sid]

        embeddings = []
        # Pair images: use min of available images per subject
        num_pairs = min(len(face_images), len(iris_images))

        for j in range(num_pairs):
            try:
                # Preprocess
                face_tensor = face_preprocessor.preprocess(face_images[j])
                iris_tensor = iris_preprocessor.preprocess(iris_images[j % len(iris_images)])

                if face_tensor is None:
                    continue

                # Extract
                face_emb = face_model.extract(face_tensor)
                iris_emb = iris_model.extract(iris_tensor)

                # Fuse
                fused = fusion.fuse_numpy(face_emb, iris_emb)
                embeddings.append(fused)

            except Exception as e:
                continue

        if len(embeddings) >= 2:
            subject_embeddings[i] = embeddings

        if (i + 1) % 50 == 0 or i == num_subjects - 1:
            print(f"  Processed {i+1}/{num_subjects} subjects "
                  f"({len(subject_embeddings)} with 2+ images)")

    print(f"  Total subjects with embeddings: {len(subject_embeddings)}")
    return subject_embeddings


def compute_scores(subject_embeddings, num_genuine, num_impostor):
    """Compute genuine and impostor similarity scores."""
    print(f"\nComputing scores ({num_genuine} genuine, {num_impostor} impostor)...")

    subject_ids = list(subject_embeddings.keys())
    if len(subject_ids) < 2:
        print("Error: Need at least 2 subjects with embeddings")
        return np.array([]), np.array([])

    genuine_scores = []
    impostor_scores = []

    # Genuine pairs: same subject, different images
    subjects_with_multiple = [
        sid for sid in subject_ids if len(subject_embeddings[sid]) >= 2
    ]

    for _ in range(num_genuine):
        if not subjects_with_multiple:
            break
        sid = random.choice(subjects_with_multiple)
        embs = subject_embeddings[sid]
        idx1, idx2 = random.sample(range(len(embs)), 2)
        score = float(np.dot(embs[idx1], embs[idx2]) / (
            np.linalg.norm(embs[idx1]) * np.linalg.norm(embs[idx2]) + 1e-8
        ))
        genuine_scores.append(score)

    # Impostor pairs: different subjects
    for _ in range(num_impostor):
        sid1, sid2 = random.sample(subject_ids, 2)
        emb1 = random.choice(subject_embeddings[sid1])
        emb2 = random.choice(subject_embeddings[sid2])
        score = float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        ))
        impostor_scores.append(score)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"  Genuine: n={len(genuine_scores)}, "
          f"mean={np.mean(genuine_scores):.4f}, std={np.std(genuine_scores):.4f}")
    print(f"  Impostor: n={len(impostor_scores)}, "
          f"mean={np.mean(impostor_scores):.4f}, std={np.std(impostor_scores):.4f}")
    print(f"  Separation: {np.mean(genuine_scores) - np.mean(impostor_scores):.4f}")

    return genuine_scores, impostor_scores


def normalize_scores(genuine_scores, impostor_scores):
    """Z-score normalize scores for better threshold calibration."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    mu = np.mean(all_scores)
    sigma = np.std(all_scores) + 1e-8

    gen_norm = (genuine_scores - mu) / sigma
    imp_norm = (impostor_scores - mu) / sigma

    # Shift to [0, 1] range using sigmoid-like transform
    gen_norm = 1 / (1 + np.exp(-gen_norm))
    imp_norm = 1 / (1 + np.exp(-imp_norm))

    return gen_norm, imp_norm


def test_fuzzy_vault(subject_embeddings, num_tests=100):
    """Test fuzzy vault genuine unlock and impostor rejection rates."""
    print(f"\nTesting fuzzy vault ({num_tests} genuine + {num_tests} impostor trials)...")

    vault = ImprovedFuzzyVault()
    subject_ids = list(subject_embeddings.keys())
    subjects_with_multiple = [
        sid for sid in subject_ids if len(subject_embeddings[sid]) >= 2
    ]

    genuine_unlocks = 0
    genuine_total = 0
    impostor_rejects = 0
    impostor_total = 0
    genuine_ratios = []
    impostor_ratios = []

    # Genuine unlock tests
    print("  Running genuine unlock tests...")
    for trial in range(num_tests):
        if not subjects_with_multiple:
            break
        sid = random.choice(subjects_with_multiple)
        embs = subject_embeddings[sid]
        idx1, idx2 = random.sample(range(len(embs)), 2)

        # Lock with one embedding, unlock with another (pass enrolled for cosine gate)
        vault_data, secret_key = vault.lock(embs[idx1])
        success, ratio, recovered = vault.unlock(embs[idx2], vault_data, enrolled_embedding=embs[idx1])

        genuine_total += 1
        genuine_ratios.append(ratio)
        if success:
            genuine_unlocks += 1

        if (trial + 1) % 25 == 0:
            print(f"    Genuine: {trial+1}/{num_tests} done (TAR so far: {genuine_unlocks}/{genuine_total})")

    # Impostor rejection tests
    print("  Running impostor rejection tests...")
    for trial in range(num_tests):
        sid1, sid2 = random.sample(subject_ids, 2)
        enroll_emb = random.choice(subject_embeddings[sid1])
        query_emb = random.choice(subject_embeddings[sid2])

        vault_data, secret_key = vault.lock(enroll_emb)
        success, ratio, recovered = vault.unlock(query_emb, vault_data, enrolled_embedding=enroll_emb)

        impostor_total += 1
        impostor_ratios.append(ratio)
        if not success:
            impostor_rejects += 1

        if (trial + 1) % 25 == 0:
            print(f"    Impostor: {trial+1}/{num_tests} done (TRR so far: {impostor_rejects}/{impostor_total})")

    tar = genuine_unlocks / max(1, genuine_total)
    trr = impostor_rejects / max(1, impostor_total)

    print(f"  Genuine Unlock Rate (TAR): {tar:.4f} ({genuine_unlocks}/{genuine_total})")
    print(f"  Impostor Reject Rate (TRR): {trr:.4f} ({impostor_rejects}/{impostor_total})")

    # Diagnostic: match ratio distributions
    if genuine_ratios:
        gen_arr = np.array(genuine_ratios)
        print(f"  Genuine match ratios:  min={gen_arr.min():.3f}, mean={gen_arr.mean():.3f}, max={gen_arr.max():.3f}")
    if impostor_ratios:
        imp_arr = np.array(impostor_ratios)
        print(f"  Impostor match ratios: min={imp_arr.min():.3f}, mean={imp_arr.mean():.3f}, max={imp_arr.max():.3f}")

    return {
        "tar": tar,
        "trr": trr,
        "genuine_unlocks": genuine_unlocks,
        "genuine_total": genuine_total,
        "impostor_rejects": impostor_rejects,
        "impostor_total": impostor_total,
    }


def check_targets(metrics, vault_results):
    """Check if metrics meet target benchmarks."""
    print("\n=== Target Verification ===")
    targets = [
        ("Accuracy > {:.1f}%".format(TARGET_ACCURACY * 100),
         metrics["accuracy"] >= TARGET_ACCURACY),
        ("FAR < {:.1f}%".format(TARGET_FAR * 100),
         metrics["far"] <= TARGET_FAR),
        ("FRR < {:.1f}%".format(TARGET_FRR * 100),
         metrics["frr"] <= TARGET_FRR),
        ("EER < {:.1f}%".format(TARGET_EER * 100),
         metrics["eer"] <= TARGET_EER),
        ("Vault TAR > {:.1f}%".format(TARGET_VAULT_TAR * 100),
         vault_results["tar"] >= TARGET_VAULT_TAR),
        ("Vault TRR > {:.1f}%".format(TARGET_VAULT_TRR * 100),
         vault_results["trr"] >= TARGET_VAULT_TRR),
    ]

    all_met = True
    for label, met in targets:
        status = "PASS" if met else "FAIL"
        print(f"    [{status}] {label}")
        if not met:
            all_met = False

    if all_met:
        print("\n  ==> ALL TARGETS MET!")
    else:
        print("\n  ==> Some targets not met. Further tuning needed.")

    return all_met


def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 60)
    print("  Multimodal Biometric Authentication System - Evaluation v4")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Step 1: Train iris model if needed ──
    if not args.skip_training:
        model_path = SAVED_MODELS_DIR / "iris_model_best.pth"
        if not model_path.exists():
            print("\n--- Training Iris Model ---")
            from models.train import train_iris_model
            train_iris_model()
        else:
            print(f"\nIris model already trained: {model_path}")
    else:
        print("\nSkipping training (--skip-training)")

    # ── Step 2: Load models ──
    print("\n--- Loading Models ---")
    face_model = FaceFeatureExtractor()
    print("  Face: FaceNet (VGGFace2 pretrained)")

    try:
        iris_model = IrisFeatureExtractor.load_trained()
        print("  Iris: ResNet18+CBAM (trained)")
    except Exception:
        iris_model = IrisFeatureExtractor().to(DEVICE)
        iris_model.eval()
        print("  Iris: ResNet18+CBAM (untrained - results will be poor)")

    # ── Step 3: Load datasets ──
    print("\n--- Loading Datasets ---")
    face_dir = args.face_dir or str(FACE_DATASET_DIR)
    iris_dir = args.iris_dir or str(IRIS_DATASET_DIR)

    face_dataset = LFWDataset(root_dir=face_dir)
    iris_dataset = IrisDataset(root_dir=iris_dir)

    if len(face_dataset) == 0 or len(iris_dataset) == 0:
        print("\nWarning: Datasets not found. Using SYNTHETIC data for evaluation.")
        print("(Results will be approximate. Download real datasets for accurate metrics.)")
        return run_synthetic_evaluation()

    # ── Step 4: Extract embeddings ──
    face_preprocessor = FacePreprocessor()
    iris_preprocessor = IrisPreprocessor()
    subject_embeddings = extract_embeddings(
        face_dataset, iris_dataset,
        face_model, iris_model,
        face_preprocessor, iris_preprocessor,
    )

    if len(subject_embeddings) < 2:
        print("Error: Not enough subjects with embeddings. Check datasets.")
        return

    # ── Step 5: Compute scores ──
    genuine_scores, impostor_scores = compute_scores(
        subject_embeddings, args.num_genuine, args.num_impostor
    )

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print("Error: Could not compute scores.")
        return

    # ── Step 5b: Score normalization (optional) ──
    if SCORE_NORMALIZATION:
        print("\n--- Applying Z-Score Normalization ---")
        genuine_norm, impostor_norm = normalize_scores(genuine_scores, impostor_scores)
        print(f"  Normalized genuine:  mean={np.mean(genuine_norm):.4f}, std={np.std(genuine_norm):.4f}")
        print(f"  Normalized impostor: mean={np.mean(impostor_norm):.4f}, std={np.std(impostor_norm):.4f}")
    else:
        genuine_norm = genuine_scores
        impostor_norm = impostor_scores

    # ── Step 6: Evaluate metrics ──
    print("\n--- Computing Metrics ---")
    evaluator = BiometricEvaluator()

    # Compute at EER threshold (primary - standard biometric evaluation)
    eer, eer_threshold = evaluator.compute_eer(genuine_norm, impostor_norm)
    metrics = evaluator.full_evaluation(genuine_norm, impostor_norm, threshold=eer_threshold)

    # Also compute at max-accuracy threshold for comparison
    opt = evaluator.compute_optimal_threshold(genuine_norm, impostor_norm)
    metrics_max_acc = evaluator.full_evaluation(genuine_norm, impostor_norm, threshold=opt["max_accuracy_threshold"])

    # d-prime
    d_prime = evaluator.compute_d_prime(genuine_norm, impostor_norm)

    # min-DCF
    min_dcf, dcf_threshold = evaluator.compute_min_dcf(genuine_norm, impostor_norm)

    print(f"\n{'='*55}")
    print(f"  === At EER Threshold ({eer_threshold:.4f}) ===")
    print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  FAR:            {metrics['far']:.4f} ({metrics['far']*100:.2f}%)")
    print(f"  FRR:            {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)")
    print(f"  EER:            {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"  d-prime:        {d_prime:.4f}")
    print(f"  min-DCF:        {min_dcf:.6f}")
    print(f"")
    print(f"  === At Max-Accuracy Threshold ({opt['max_accuracy_threshold']:.4f}) ===")
    print(f"  Accuracy:       {metrics_max_acc['accuracy']:.4f} ({metrics_max_acc['accuracy']*100:.2f}%)")
    print(f"  FAR:            {metrics_max_acc['far']:.4f} ({metrics_max_acc['far']*100:.2f}%)")
    print(f"  FRR:            {metrics_max_acc['frr']:.4f} ({metrics_max_acc['frr']*100:.2f}%)")
    print(f"")
    print(f"  Score Sep.:     {metrics['score_separation']:.4f}")
    print(f"  Genuine Mean:   {metrics['genuine_mean']:.4f}")
    print(f"  Impostor Mean:  {metrics['impostor_mean']:.4f}")
    print(f"{'='*55}")

    # ── Step 6b: Bootstrap EER confidence intervals ──
    print("\n--- Bootstrap EER Confidence Intervals ---")
    bootstrap_ci = evaluator.bootstrap_eer(
        genuine_norm, impostor_norm, n_bootstrap=args.bootstrap
    )
    print(f"  EER (mean ± std): {bootstrap_ci['eer_mean']*100:.2f}% ± {bootstrap_ci['eer_std']*100:.2f}%")
    print(f"  95% CI: [{bootstrap_ci['eer_lower']*100:.2f}%, {bootstrap_ci['eer_upper']*100:.2f}%]")

    # ── Step 7: Plot curves ──
    print("\n--- Generating Plots ---")
    roc_path = evaluator.plot_roc_curve(genuine_norm, impostor_norm)
    det_path = evaluator.plot_det_curve(genuine_norm, impostor_norm)
    dist_path = evaluator.plot_score_distributions(
        genuine_norm, impostor_norm, metrics["threshold"]
    )
    sweep_path = evaluator.plot_threshold_sweep(genuine_norm, impostor_norm)
    cm_path = evaluator.plot_confusion_matrix(
        genuine_norm, impostor_norm, metrics["threshold"]
    )
    print(f"  ROC: {roc_path}")
    print(f"  DET: {det_path}")
    print(f"  Dist: {dist_path}")
    print(f"  Sweep: {sweep_path}")
    print(f"  CM: {cm_path}")

    # ── Step 8: Fuzzy vault test ──
    vault_results = test_fuzzy_vault(subject_embeddings, num_tests=args.vault_trials)

    # ── Step 9: Target verification ──
    all_targets_met = check_targets(metrics, vault_results)

    # ── Step 10: Comparison with previous ──
    comparison = evaluator.compare_results(PREVIOUS_METRICS, metrics)
    print(comparison)

    # ── Step 11: Save report ──
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "metrics": metrics,
        "metrics_max_accuracy": {
            "accuracy": metrics_max_acc["accuracy"],
            "far": metrics_max_acc["far"],
            "frr": metrics_max_acc["frr"],
            "threshold": metrics_max_acc["threshold"],
        },
        "d_prime": d_prime,
        "min_dcf": min_dcf,
        "bootstrap_eer": bootstrap_ci,
        "score_normalization": SCORE_NORMALIZATION,
        "vault_results": vault_results,
        "all_targets_met": all_targets_met,
        "targets": {
            "accuracy": TARGET_ACCURACY,
            "far": TARGET_FAR,
            "frr": TARGET_FRR,
            "eer": TARGET_EER,
            "vault_tar": TARGET_VAULT_TAR,
            "vault_trr": TARGET_VAULT_TRR,
        },
        "total_time_seconds": time.time() - t_start,
    }

    report_path = EVALUATION_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    print(f"Total evaluation time: {time.time() - t_start:.1f}s")


def run_synthetic_evaluation():
    """Run evaluation with synthetic data when real datasets are unavailable."""
    print("\n--- Synthetic Evaluation ---")
    print("Generating synthetic embeddings for demonstration...")

    # Simulate high-quality embeddings
    np.random.seed(42)
    num_subjects = 100
    embeddings_per_subject = 3

    subject_embeddings = {}
    for i in range(num_subjects):
        base = np.random.randn(512).astype(np.float32)
        base = base / np.linalg.norm(base)
        embs = []
        for _ in range(embeddings_per_subject):
            noise = np.random.randn(512) * 0.05
            emb = base + noise
            emb = emb / np.linalg.norm(emb)
            embs.append(emb)
        subject_embeddings[i] = embs

    genuine_scores, impostor_scores = compute_scores(
        subject_embeddings, 2000, 2000
    )

    evaluator = BiometricEvaluator()
    metrics = evaluator.full_evaluation(genuine_scores, impostor_scores)

    print(f"\n{'='*40}")
    print(f"  [SYNTHETIC] Accuracy: {metrics['accuracy']:.4f}")
    print(f"  [SYNTHETIC] FAR:      {metrics['far']:.4f}")
    print(f"  [SYNTHETIC] FRR:      {metrics['frr']:.4f}")
    print(f"  [SYNTHETIC] EER:      {metrics['eer']:.4f}")
    print(f"  [SYNTHETIC] d-prime:  {metrics['d_prime']:.4f}")
    print(f"{'='*40}")

    # Plots
    evaluator.plot_roc_curve(genuine_scores, impostor_scores, "roc_curve_synthetic.png")
    evaluator.plot_score_distributions(
        genuine_scores, impostor_scores, metrics["threshold"],
        "score_distributions_synthetic.png"
    )
    evaluator.plot_threshold_sweep(genuine_scores, impostor_scores, "threshold_sweep_synthetic.png")

    vault_results = test_fuzzy_vault(subject_embeddings, num_tests=100)
    check_targets(metrics, vault_results)

    print("\nNote: These are SYNTHETIC results. Download real datasets for accurate metrics.")


if __name__ == "__main__":
    main()
