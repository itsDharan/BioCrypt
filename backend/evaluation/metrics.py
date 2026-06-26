"""
Biometric evaluation metrics: Accuracy, FAR, FRR, EER, d-prime, min-DCF,
ROC, DET curves, confusion matrix, score calibration, and threshold sweep.

v2 improvements:
  - d-prime (decidability index) for score separation quality
  - min-DCF (Detection Cost Function) at multiple operating points
  - Confusion matrix generation
  - Threshold sweep analysis with FAR/FRR crossover plot
  - Score calibration via Platt scaling (logistic regression)
  - Bootstrap confidence intervals for EER
  - Publication-quality matplotlib styling
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, det_curve, confusion_matrix as sk_confusion_matrix
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import EVALUATION_DIR

# Publication-quality plot styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


class BiometricEvaluator:
    """
    Computes biometric authentication metrics:
      - FAR (False Acceptance Rate)
      - FRR (False Rejection Rate)
      - EER (Equal Error Rate) with bootstrap CIs
      - d-prime (decidability index)
      - min-DCF (Detection Cost Function)
      - Confusion matrix
      - ROC / DET / threshold sweep curves
      - Score distributions with overlap visualization
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or EVALUATION_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_far_frr(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float]:
        """
        Compute FAR and FRR at a given threshold.

        FAR = fraction of impostors accepted (score >= threshold)
        FRR = fraction of genuine users rejected (score < threshold)
        """
        far = np.mean(impostor_scores >= threshold) if len(impostor_scores) > 0 else 0.0
        frr = np.mean(genuine_scores < threshold) if len(genuine_scores) > 0 else 0.0
        return float(far), float(frr)

    def compute_eer(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) — where FAR == FRR.

        Returns:
            (eer, eer_threshold)
        """
        thresholds = np.linspace(0, 1, 10000)
        min_diff = float("inf")
        eer = 0.0
        eer_threshold = 0.5

        for t in thresholds:
            far, frr = self.compute_far_frr(genuine_scores, impostor_scores, t)
            diff = abs(far - frr)
            if diff < min_diff:
                min_diff = diff
                eer = (far + frr) / 2
                eer_threshold = t

        return float(eer), float(eer_threshold)

    def compute_d_prime(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> float:
        """
        Compute d-prime (decidability index).

        d' = |μ_genuine - μ_impostor| / sqrt(0.5 * (σ²_genuine + σ²_impostor))

        Higher d-prime → better separation. d' > 3 is considered excellent.
        """
        if len(genuine_scores) < 2 or len(impostor_scores) < 2:
            return 0.0

        mu_gen = np.mean(genuine_scores)
        mu_imp = np.mean(impostor_scores)
        var_gen = np.var(genuine_scores)
        var_imp = np.var(impostor_scores)

        denominator = np.sqrt(0.5 * (var_gen + var_imp))
        if denominator < 1e-8:
            return 0.0

        return float(abs(mu_gen - mu_imp) / denominator)

    def compute_min_dcf(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute minimum Detection Cost Function (min-DCF).

        DCF = c_miss * p_target * FRR + c_fa * (1 - p_target) * FAR

        Returns:
            (min_dcf, optimal_threshold)
        """
        thresholds = np.linspace(0, 1, 10000)
        min_dcf = float("inf")
        opt_threshold = 0.5

        for t in thresholds:
            far, frr = self.compute_far_frr(genuine_scores, impostor_scores, t)
            dcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
            if dcf < min_dcf:
                min_dcf = dcf
                opt_threshold = t

        return float(min_dcf), float(opt_threshold)

    def compute_optimal_threshold(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        target_far: float = 0.01,
    ) -> Dict[str, float]:
        """
        Compute multiple optimal thresholds.

        Returns dict with:
          - eer_threshold: Where FAR == FRR
          - max_accuracy_threshold: Maximizes overall accuracy
          - target_far_threshold: Achieves target FAR
        """
        thresholds = np.linspace(0, 1, 10000)

        # EER threshold
        eer, eer_threshold = self.compute_eer(genuine_scores, impostor_scores)

        # Max-accuracy threshold
        best_acc = 0.0
        max_acc_threshold = 0.5

        for t in thresholds:
            genuine_correct = np.sum(genuine_scores >= t)
            impostor_correct = np.sum(impostor_scores < t)
            total = len(genuine_scores) + len(impostor_scores)
            acc = (genuine_correct + impostor_correct) / total if total > 0 else 0
            if acc > best_acc:
                best_acc = acc
                max_acc_threshold = t

        # Target-FAR threshold
        target_far_threshold = 0.9
        for t in reversed(thresholds):
            far = np.mean(impostor_scores >= t) if len(impostor_scores) > 0 else 0
            if far <= target_far:
                target_far_threshold = t
                break

        return {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "max_accuracy": best_acc,
            "max_accuracy_threshold": max_acc_threshold,
            "target_far_threshold": target_far_threshold,
            "target_far": target_far,
        }

    def calibrate_scores(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate scores using Platt scaling (logistic regression).

        Maps raw cosine similarity scores to calibrated probabilities,
        improving threshold selection.

        Returns:
            (calibrated_genuine, calibrated_impostor, calibrated_threshold)
        """
        scores = np.concatenate([genuine_scores, impostor_scores]).reshape(-1, 1)
        labels = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores)),
        ])

        # Fit logistic regression for score calibration
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(scores, labels)

        # Calibrated probabilities
        cal_genuine = lr.predict_proba(genuine_scores.reshape(-1, 1))[:, 1]
        cal_impostor = lr.predict_proba(impostor_scores.reshape(-1, 1))[:, 1]

        # Calibrated threshold at 0.5 probability
        cal_threshold = 0.5

        return cal_genuine, cal_impostor, cal_threshold

    def bootstrap_eer(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute EER with bootstrap confidence intervals.

        Returns:
            dict with eer_mean, eer_lower, eer_upper, eer_std
        """
        eers = []
        n_gen = len(genuine_scores)
        n_imp = len(impostor_scores)

        for _ in range(n_bootstrap):
            gen_sample = np.random.choice(genuine_scores, n_gen, replace=True)
            imp_sample = np.random.choice(impostor_scores, n_imp, replace=True)
            eer, _ = self.compute_eer(gen_sample, imp_sample)
            eers.append(eer)

        eers = np.array(eers)
        alpha = (1 - confidence) / 2

        return {
            "eer_mean": float(np.mean(eers)),
            "eer_std": float(np.std(eers)),
            "eer_lower": float(np.percentile(eers, alpha * 100)),
            "eer_upper": float(np.percentile(eers, (1 - alpha) * 100)),
        }

    def full_evaluation(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float = None,
    ) -> Dict[str, float]:
        """Compute all metrics. Uses max-accuracy threshold if none specified."""
        eer, eer_threshold = self.compute_eer(genuine_scores, impostor_scores)

        # Use max-accuracy threshold by default
        if threshold is None:
            opt = self.compute_optimal_threshold(genuine_scores, impostor_scores)
            threshold = opt["max_accuracy_threshold"]

        far, frr = self.compute_far_frr(genuine_scores, impostor_scores, threshold)

        # Accuracy
        genuine_correct = np.sum(genuine_scores >= threshold)
        impostor_correct = np.sum(impostor_scores < threshold)
        total = len(genuine_scores) + len(impostor_scores)
        accuracy = (genuine_correct + impostor_correct) / total if total > 0 else 0

        # d-prime
        d_prime = self.compute_d_prime(genuine_scores, impostor_scores)

        # min-DCF
        min_dcf, _ = self.compute_min_dcf(genuine_scores, impostor_scores)

        # Score statistics
        gen_mean = float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0
        imp_mean = float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0
        separation = gen_mean - imp_mean

        return {
            "accuracy": float(accuracy),
            "far": float(far),
            "frr": float(frr),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "threshold": float(threshold),
            "d_prime": float(d_prime),
            "min_dcf": float(min_dcf),
            "genuine_mean": gen_mean,
            "impostor_mean": imp_mean,
            "score_separation": separation,
            "genuine_std": float(np.std(genuine_scores)) if len(genuine_scores) > 0 else 0,
            "impostor_std": float(np.std(impostor_scores)) if len(impostor_scores) > 0 else 0,
            "num_genuine": len(genuine_scores),
            "num_impostor": len(impostor_scores),
        }

    def compare_results(
        self,
        previous: Dict[str, float],
        current: Dict[str, float],
    ) -> str:
        """Generate before/after comparison table."""
        lines = [
            "\n=== Performance Comparison ===",
            f"{'Metric':<20} {'Previous':>10} {'Current':>10} {'Change':>10} {'Status':>8}",
            "-" * 62,
        ]

        comparisons = [
            ("Accuracy", "accuracy", True),
            ("FAR", "far", False),
            ("FRR", "frr", False),
            ("EER", "eer", False),
            ("d-prime", "d_prime", True),
            ("Threshold", "threshold", None),
            ("Score Sep.", "score_separation", True),
        ]

        for label, key, higher_better in comparisons:
            prev_val = previous.get(key, 0)
            curr_val = current.get(key, 0)
            diff = curr_val - prev_val

            if higher_better is True:
                indicator = "[+]" if diff > 0 else "[-]"
            elif higher_better is False:
                indicator = "[+]" if diff < 0 else "[-]"
            else:
                indicator = "[=]"

            lines.append(
                f"{label:<20} {prev_val:>10.4f} {curr_val:>10.4f} "
                f"{diff:>+10.4f} {indicator:>8}"
            )

        return "\n".join(lines)

    # ── Plotting ──────────────────────────────────────────────────────

    def plot_roc_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        filename: str = "roc_curve.png",
    ) -> str:
        """Plot ROC curve and compute AUC."""
        labels = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores)),
        ])
        scores = np.concatenate([genuine_scores, impostor_scores])

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate (FAR)")
        ax.set_ylabel("True Positive Rate (1 - FRR)")
        ax.set_title("ROC Curve — Multimodal Biometric System")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / filename
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)

    def plot_det_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        filename: str = "det_curve.png",
    ) -> str:
        """Plot Detection Error Tradeoff (DET) curve."""
        labels = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores)),
        ])
        scores = np.concatenate([genuine_scores, impostor_scores])

        fpr, fnr, _ = det_curve(labels, scores)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(fpr * 100, fnr * 100, "b-", linewidth=2)
        ax.set_xlabel("False Acceptance Rate (%)")
        ax.set_ylabel("False Rejection Rate (%)")
        ax.set_title("DET Curve — Multimodal Biometric System")
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / filename
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)

    def plot_score_distributions(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float = None,
        filename: str = "score_distributions.png",
    ) -> str:
        """Plot genuine vs impostor score distributions with overlap region."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if len(genuine_scores) > 0:
            ax.hist(genuine_scores, bins=60, alpha=0.6, color="green",
                    label=f"Genuine (n={len(genuine_scores)}, μ={np.mean(genuine_scores):.3f})",
                    density=True)
        if len(impostor_scores) > 0:
            ax.hist(impostor_scores, bins=60, alpha=0.6, color="red",
                    label=f"Impostor (n={len(impostor_scores)}, μ={np.mean(impostor_scores):.3f})",
                    density=True)

        if threshold is not None:
            ax.axvline(x=threshold, color="blue", linestyle="--", linewidth=2,
                       label=f"Threshold = {threshold:.4f}")

        # d-prime annotation
        d_prime = self.compute_d_prime(genuine_scores, impostor_scores)
        ax.text(0.02, 0.95, f"d' = {d_prime:.2f}",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel("Cosine Similarity Score")
        ax.set_ylabel("Density")
        ax.set_title("Score Distribution: Genuine vs Impostor")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / filename
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)

    def plot_threshold_sweep(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        filename: str = "threshold_sweep.png",
    ) -> str:
        """Plot FAR and FRR as functions of threshold (crossover = EER)."""
        thresholds = np.linspace(0, 1, 500)
        fars, frrs = [], []

        for t in thresholds:
            far, frr = self.compute_far_frr(genuine_scores, impostor_scores, t)
            fars.append(far)
            frrs.append(frr)

        fars, frrs = np.array(fars), np.array(frrs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(thresholds, fars * 100, "r-", linewidth=2, label="FAR (%)")
        ax.plot(thresholds, frrs * 100, "b-", linewidth=2, label="FRR (%)")

        # Mark EER crossover
        eer_idx = np.argmin(np.abs(fars - frrs))
        eer_threshold = thresholds[eer_idx]
        eer_value = (fars[eer_idx] + frrs[eer_idx]) / 2
        ax.plot(eer_threshold, eer_value * 100, "ko", markersize=10,
                label=f"EER = {eer_value*100:.2f}% at τ={eer_threshold:.3f}")

        ax.set_xlabel("Threshold (τ)")
        ax.set_ylabel("Error Rate (%)")
        ax.set_title("FAR vs FRR — Threshold Sweep Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / filename
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)

    def plot_confusion_matrix(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float,
        filename: str = "confusion_matrix.png",
    ) -> str:
        """Plot confusion matrix at a given threshold."""
        # True labels: 1=genuine, 0=impostor
        y_true = np.concatenate([
            np.ones(len(genuine_scores), dtype=int),
            np.zeros(len(impostor_scores), dtype=int),
        ])
        # Predictions based on threshold
        y_pred = np.concatenate([
            (genuine_scores >= threshold).astype(int),
            (impostor_scores >= threshold).astype(int),
        ])

        cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Reject", "Accept"],
                    yticklabels=["Impostor", "Genuine"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (threshold={threshold:.4f})")

        save_path = self.output_dir / filename
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)
