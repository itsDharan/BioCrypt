"""
Global configuration and hyperparameters for the
Multimodal Biometric Authentication System.

All tunable parameters are centralized here for reproducibility.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
FACE_DATASET_DIR = DATA_DIR / "lfw-deepfunneled"
IRIS_DATASET_DIR = DATA_DIR / "CASIA-Iris-Thousand"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
VAULT_STORAGE_DIR = BASE_DIR / "vault_storage"
EVALUATION_DIR = BASE_DIR / "evaluation_results"

# Create directories
for d in [SAVED_MODELS_DIR, VAULT_STORAGE_DIR, EVALUATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Image Preprocessing ───────────────────────────────────────────────
IMAGE_SIZE = 224                  # Resize all images to 224x224
FACE_ALIGNMENT_PADDING = 0.3     # Padding around detected face for alignment
CLAHE_CLIP_LIMIT = 2.0           # CLAHE contrast enhancement clip limit
CLAHE_GRID_SIZE = (8, 8)         # CLAHE tile grid size

# ── Model Architecture ────────────────────────────────────────────────
FACE_EMBEDDING_DIM = 512         # FaceNet InceptionResnetV1 output dim
IRIS_EMBEDDING_DIM = 512         # ResNet18+CBAM iris model output dim
FUSED_EMBEDDING_DIM = 512        # Fusion output preserves dimensionality

# ── Training Hyperparameters ──────────────────────────────────────────
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5
CONTRASTIVE_MARGIN = 1.0
TRIPLET_MARGIN = 0.3
NUM_WORKERS = 0                  # 0 for Windows compatibility
FUSION_ALPHA = 0.6               # Weight for face in weighted fusion

# ArcFace loss hyperparameters
ARCFACE_SCALE = 30.0             # Feature scaling factor (s)
ARCFACE_MARGIN = 0.5             # Additive angular margin (m) in radians
ARCFACE_EASY_MARGIN = False      # Use easy margin variant

# Training strategy
TRAIN_NUM_PAIRS = 8000           # Contrastive/triplet pairs per epoch
TRAIN_POSITIVE_RATIO = 0.5      # Ratio of positive pairs
TRAIN_HARD_MINING = True         # Enable online hard negative mining
TRAIN_EPOCHS = 15                # Total training epochs

# Data augmentation flags
AUGMENT_ROTATION = 15            # Max rotation degrees (±)
AUGMENT_FLIP = True              # Horizontal flip
AUGMENT_COLOR_JITTER = True      # Brightness, contrast, saturation jitter
AUGMENT_GAUSSIAN_BLUR = True     # Random Gaussian blur
AUGMENT_RANDOM_ERASING = True    # Random erasing (cutout)

# ── Authentication Thresholds ─────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.75      # Cosine similarity threshold (hardened to reject impostors)
FACE_SIMILARITY_THRESHOLD = 0.65 # Independent face cosine threshold (FaceNet is discriminative)
IRIS_SIMILARITY_THRESHOLD = 0.80 # Independent iris cosine threshold (must reject different iris)
USE_DYNAMIC_THRESHOLD = True     # Use EER/max-accuracy computed threshold at eval time
SCORE_NORMALIZATION = True       # Z-score normalize scores before thresholding
USE_CALIBRATED_THRESHOLD = True  # Use Platt-scaled calibrated threshold

# ── Fuzzy Vault Parameters (tuned for >95% TAR) ──────────────────────
VAULT_FEATURE_POINTS = 40        # Number of biometric feature points to use
VAULT_CHAFF_MULTIPLIER = 8       # chaff_count = feature_points * multiplier (320 chaff)
VAULT_FIELD_SIZE = 65536         # Large field for chaff placement (must be >> feature_points)
RS_NSYM = 20                     # Reed-Solomon error correction symbols (↑ from 16)
VAULT_SECRET_KEY_LENGTH = 16     # 128-bit secret key (must fit in RS: key_len + rs_nsym < 256)
MIN_MATCHING_POINTS = 6          # Minimum matching points for vault unlock (↑ from 5 for security)
VAULT_QUANTIZE_BINS = 63         # Coarse quantization for robust genuine matching
VAULT_QUANTIZE_RANGE = (-0.6, 0.6)  # Wider range to capture more embedding variance (↑ from ±0.5)
VAULT_MATCH_TOLERANCE = 2        # Tolerance for x-coordinate matching (↓ from 3 for security)
VAULT_MIN_CHAFF_DISTANCE = 10    # Minimum distance between chaff and genuine (in spread space)
VAULT_BIN_SPACING = 100          # Spread factor: x_vault = x_bin * spacing

# ── Cryptography ──────────────────────────────────────────────────────
AES_KEY_SIZE = 32                # 256-bit AES key
AES_NONCE_SIZE = 12              # 96-bit nonce for GCM

# ── Blockchain / Smart Contract ───────────────────────────────────────
BLOCKCHAIN_RPC_URL = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "")
DEPLOYER_PRIVATE_KEY = os.getenv("DEPLOYER_PRIVATE_KEY", "")
CHAIN_ID = int(os.getenv("CHAIN_ID", "31337"))  # Hardhat default

# ── IPFS ──────────────────────────────────────────────────────────────
IPFS_HOST = os.getenv("IPFS_HOST", "127.0.0.1")
IPFS_PORT = int(os.getenv("IPFS_PORT", "5001"))
IPFS_GATEWAY = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080/ipfs/")
PINATA_JWT = os.getenv("PINATA_JWT", "")

# ── Device ────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Target Metrics (for evaluation verification) ──────────────────────
TARGET_ACCURACY = 0.985
TARGET_FAR = 0.02
TARGET_FRR = 0.04
TARGET_EER = 0.03
TARGET_VAULT_TAR = 0.95
TARGET_VAULT_TRR = 0.98
