"""
Performance benchmarking for the biometric authentication system.
Measures timing across all pipeline stages and compares modality combinations.
"""

import time
import numpy as np
import torch
import json
from typing import Dict, List, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import DEVICE, EVALUATION_DIR


class SystemBenchmark:
    """
    Benchmark timing for each pipeline stage:
      - Face feature extraction
      - Iris feature extraction
      - Feature fusion
      - Fuzzy vault lock/unlock
      - AES encryption/decryption
      - ECC key wrap/unwrap
      - IPFS upload/retrieve
    """

    def __init__(self):
        self.results: Dict[str, List[float]] = {}

    def _time_operation(self, name: str, func, *args, num_runs: int = 5, **kwargs):
        """Time an operation over multiple runs."""
        times = []
        result = None
        for _ in range(num_runs):
            t0 = time.time()
            result = func(*args, **kwargs)
            times.append(time.time() - t0)

        self.results[name] = {
            "mean_ms": float(np.mean(times) * 1000),
            "std_ms": float(np.std(times) * 1000),
            "min_ms": float(np.min(times) * 1000),
            "max_ms": float(np.max(times) * 1000),
            "num_runs": num_runs,
        }
        return result

    def benchmark_feature_extraction(self, num_runs: int = 5):
        """Benchmark face and iris feature extraction."""
        from models.face_model import FaceFeatureExtractor
        from models.iris_model import IrisFeatureExtractor

        # Dummy input
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)

        # Face
        face_model = FaceFeatureExtractor()
        self._time_operation(
            "face_extraction", face_model.forward, dummy, num_runs=num_runs
        )

        # Iris
        try:
            iris_model = IrisFeatureExtractor.load_trained()
        except Exception:
            iris_model = IrisFeatureExtractor().to(DEVICE)
            iris_model.eval()

        with torch.no_grad():
            self._time_operation(
                "iris_extraction", iris_model.forward, dummy, num_runs=num_runs
            )

    def benchmark_fusion(self, num_runs: int = 10):
        """Benchmark fusion methods."""
        from models.fusion import WeightedFusion

        face_emb = torch.randn(1, 512).to(DEVICE)
        iris_emb = torch.randn(1, 512).to(DEVICE)

        fusion = WeightedFusion()
        self._time_operation(
            "weighted_fusion", fusion.forward, face_emb, iris_emb, num_runs=num_runs
        )

    def benchmark_crypto(self, num_runs: int = 10):
        """Benchmark cryptographic operations."""
        from crypto.fuzzy_vault import ImprovedFuzzyVault
        from crypto.hybrid_crypto import HybridCrypto

        vault_gen = ImprovedFuzzyVault()
        crypto = HybridCrypto()

        # Fuzzy vault lock
        dummy_features = np.random.randn(512).astype(np.float32) * 0.5
        vault_data, secret_key = self._time_operation(
            "vault_lock", vault_gen.lock, dummy_features, num_runs=num_runs
        )

        # Fuzzy vault unlock (genuine)
        self._time_operation(
            "vault_unlock_genuine", vault_gen.unlock,
            dummy_features + np.random.randn(512) * 0.01,  # Small noise
            vault_data, num_runs=num_runs
        )

        # Hybrid encrypt
        priv, pub = crypto.generate_keypair()
        vault_json = json.dumps(vault_data)

        encrypted = self._time_operation(
            "hybrid_encrypt", crypto.encrypt, vault_json, pub, num_runs=num_runs
        )

        # Hybrid decrypt
        self._time_operation(
            "hybrid_decrypt", crypto.decrypt, encrypted, priv, num_runs=num_runs
        )

    def benchmark_storage(self, num_runs: int = 3):
        """Benchmark IPFS storage operations."""
        from storage.ipfs_client import IPFSClient

        ipfs = IPFSClient()
        test_data = json.dumps({"test": "data", "size": "x" * 1000})

        # Upload
        cid = self._time_operation(
            "ipfs_upload", ipfs.upload, test_data, "benchmark_test.json",
            num_runs=num_runs
        )

        # Retrieve
        self._time_operation(
            "ipfs_retrieve", ipfs.retrieve, cid, num_runs=num_runs
        )

    def run_all(self, num_runs: int = 5) -> Dict[str, Any]:
        """Run all benchmarks."""
        print("Benchmarking feature extraction...")
        self.benchmark_feature_extraction(num_runs)

        print("Benchmarking fusion...")
        self.benchmark_fusion(num_runs)

        print("Benchmarking crypto...")
        self.benchmark_crypto(num_runs)

        print("Benchmarking storage...")
        self.benchmark_storage(min(num_runs, 3))

        return self.results

    def print_results(self):
        """Print formatted benchmark results."""
        print("\n=== System Benchmark Results ===")
        print(f"{'Operation':<25} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10}")
        print("-" * 60)

        for name, data in self.results.items():
            print(
                f"{name:<25} "
                f"{data['mean_ms']:>10.2f} "
                f"{data['std_ms']:>10.2f} "
                f"{data['min_ms']:>10.2f}"
            )

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON."""
        save_path = EVALUATION_DIR / filename
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Benchmark saved to {save_path}")
        return str(save_path)
