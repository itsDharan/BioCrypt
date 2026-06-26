# 🔐 Multimodal Biometric Authentication System

> A **secure biometric login system** that uses your **face + iris (eye)** to verify your identity — protected by encryption and stored on blockchain.

---

## 📌 What Does This Project Do? (Simple Explanation)

Imagine a system where instead of using passwords (which can be stolen), you use your **face** and **eye (iris)** to log in. This project does exactly that, but with multiple layers of security:

1. **Takes your face photo + eye photo** 📸
2. **Converts them into unique number codes** (called embeddings) 🔢
3. **Mixes both codes together** into one combined identity 🔀
4. **Locks this identity in a mathematical vault** (Fuzzy Vault) 🔒
5. **Encrypts everything** with military-grade encryption (AES + ECC) 🛡️
6. **Stores it on decentralized storage** (IPFS + Blockchain) 🌐

When you want to log in later, it takes fresh photos, creates new codes, and checks if they match the stored ones. **No passwords, no central database to hack.**

---

## 🗂️ Project Structure (What's Where)

```
Captain/
│
├── backend/                    ← 🧠 Main Python code (the brain)
│   ├── config/settings.py      ← ⚙️ All settings and parameters in one place
│   ├── data/                   ← 📦 Image loading & preprocessing
│   │   ├── preprocessing.py    ← 🖼️ Face & iris image enhancement
│   │   └── dataset.py          ← 📂 Dataset loading for training
│   ├── models/                 ← 🤖 AI models for feature extraction
│   │   ├── face_model.py       ← 👤 Face recognition (FaceNet)
│   │   ├── iris_model.py       ← 👁️ Iris recognition (ResNet18 + CBAM)
│   │   ├── fusion.py           ← 🔀 Combining face + iris features
│   │   └── train.py            ← 🏋️ Training script for iris model
│   ├── crypto/                 ← 🔐 Cryptography & security
│   │   ├── fuzzy_vault.py      ← 🏦 Fuzzy Vault (biometric template protection)
│   │   ├── aes_encryption.py   ← 🔑 AES-256 encryption
│   │   ├── ecc_encryption.py   ← 🔑 ECC (Elliptic Curve) key management
│   │   └── hybrid_crypto.py    ← 🔑 Combined AES + ECC pipeline
│   ├── auth/                   ← ✅ Enrollment & authentication logic
│   │   ├── enrollment.py       ← 📝 User registration flow
│   │   └── authentication.py   ← 🔓 User login/verification flow
│   ├── storage/                ← 💾 Decentralized storage
│   │   ├── ipfs_client.py      ← 📡 IPFS (file storage)
│   │   └── blockchain_client.py← ⛓️ Ethereum blockchain client
│   ├── evaluation/             ← 📊 Performance testing & benchmarks
│   ├── api/main.py             ← 🌐 FastAPI web server (REST API)
│   ├── frontend/index.html     ← 💻 Web UI for browser demo
│   └── requirements.txt        ← 📋 Python dependencies list
│
├── blockchain/                 ← ⛓️ Smart contract code
│   ├── contracts/
│   │   └── BiometricVault.sol  ← 📜 Solidity smart contract
│   ├── hardhat.config.js       ← ⚙️ Blockchain dev environment config
│   └── scripts/                ← 🚀 Deployment scripts
│
└── Help/Report/                ← 📄 Research paper / documentation
```

---

## 🔄 How The System Works — Step by Step

### 🟢 STEP 1: Image Preprocessing (`data/preprocessing.py`)

**What it does:** Takes raw photos and cleans them up so the AI can understand them better.

#### For Face Images (FacePreprocessor):
```
Raw Photo → Detect Face (MTCNN) → Align & Crop → Apply CLAHE → Normalize → Tensor
```
- **MTCNN** = A face detector that finds where the face is in the photo and aligns it straight
- **CLAHE** = A contrast enhancement technique that makes facial features more visible (especially in poor lighting)
- **Normalize** = Converts pixel values to a standard range that the AI model expects
- **Output**: A clean 224×224 pixel face image as a number tensor

#### For Iris/Eye Images (IrisPreprocessor):
```
Raw Photo → Remove Reflections → Find Iris Circle → Crop → CLAHE → Denoise → Tensor
```
- **Remove Reflections** = Finds bright spots (light reflections on the eye) and fills them using inpainting
- **Circular Hough Transform** = A mathematical method to find the circular boundary of the iris and pupil in the image
- **Bilateral Filter** = Removes noise (graininess) from the image while keeping the iris texture sharp
- **Output**: A clean 224×224 pixel iris image as a number tensor

> **Why?** Garbage in = garbage out. If the images are noisy, blurry, or have reflections, the AI will produce bad embeddings. Preprocessing ensures consistent, high-quality input.

---

### 🟢 STEP 2: Feature Extraction (`models/face_model.py` & `models/iris_model.py`)

**What it does:** Converts the cleaned images into **512-dimensional number vectors** (embeddings) — a unique "fingerprint" of your biometrics.

#### Face Feature Extraction (FaceNet):
```
Clean Face Image → FaceNet (InceptionResnetV1) → 512 numbers → L2 Normalize
```
- Uses **FaceNet**, a pretrained deep learning model trained on **3.31 million face images** from **9,131 different people** (VGGFace2 dataset)
- **No training needed** — the model already knows how to recognize faces
- **L2 Normalization** = Makes all embeddings the same length (unit vectors), so we can compare them fairly using cosine similarity
- **Output**: A vector of 512 numbers that uniquely represents your face

#### Iris Feature Extraction (ResNet18 + CBAM):
```
Clean Iris Image → ResNet18 → CBAM Attention → CBAM Attention → Pool → Project → 512 numbers
```
- **ResNet18** = A pretrained image recognition model (trained on ImageNet — 1.2M images). We freeze early layers and fine-tune the deeper ones for iris patterns
- **CBAM (Convolutional Block Attention Module)** = An attention mechanism that tells the model: "Focus on the important iris texture patterns and ignore the rest." Applied twice:
  - After layer3 (256 channels) — focuses on discriminative iris regions
  - After layer4 (512 channels) — refines the final features
- **ArcFace Training** = The iris model is trained using ArcFace loss, which pushes different people's embeddings far apart and pulls same-person embeddings close together (angular margin learning)
- **Output**: A vector of 512 numbers that uniquely represents your iris pattern

> **Simple analogy:** Think of embeddings like a unique barcode. Two photos of the same person → similar barcodes. Photos of different people → very different barcodes.

---

### 🟢 STEP 3: Feature Fusion (`models/fusion.py`)

**What it does:** Combines the face embedding (512 numbers) and iris embedding (512 numbers) into **one single fused embedding** (512 numbers).

#### Why Fuse?
- Using **two biometrics** is more secure than one (harder to fake both face AND iris)
- Fusion gives **better accuracy** than either modality alone

#### Quality-Aware Adaptive Fusion (Primary Method):
```
Face Embedding ──→ Quality Score ──→ ┐
                                     ├──→ Adaptive Weights → Weighted Sum → Fused Embedding
Iris Embedding ──→ Quality Score ──→ ┘
```

**How it works:**
1. For each embedding, compute 4 **quality statistics**: L2 norm, mean absolute value, standard deviation, and max value
2. Feed these stats into a small MLP (mini neural network) to get a **quality score**
3. Use **softmax** on both scores to get adaptive weights (e.g., face=0.65, iris=0.35)
4. **Multiply and add**: `fused = 0.65 × face + 0.35 × iris`
5. **L2 normalize** the result

**Why adaptive?** If your face photo is blurry but your iris photo is clear, the system automatically gives MORE weight to the iris and LESS weight to the face. Smart! 🧠

#### Other Available Fusion Methods:
| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **Weighted** | `0.6×face + 0.4×iris` (fixed weights) | Simple, reliable default |
| **Concat** | Joins both vectors (1024D → project to 512D) | When you can train the projection layer |
| **Attention** | Learns weights using attention network | When you have lots of training data |

---

### 🟢 STEP 4: Fuzzy Vault Lock (`crypto/fuzzy_vault.py`)

**What it does:** Protects the biometric template using a mathematical scheme called a **Fuzzy Vault**. The vault stores a secret key that can ONLY be recovered if you present similar enough biometrics.

#### How the Fuzzy Vault Works (Simple Explanation):

**During Enrollment (Locking):**
```
Fused Embedding → Quantize to 40 points → Put on polynomial → Add 320 chaff (fake) points → Shuffle → Vault
```

1. **Quantize**: Convert the 512 floating-point numbers into 40 integer coordinate points (x-values)
2. **Generate Secret Key**: Create a random 128-bit secret key
3. **Encode as Polynomial**: Use Reed-Solomon coding to encode this key as polynomial coefficients
4. **Evaluate Polynomial**: For each of the 40 genuine x-values, compute the y-value using the polynomial → these 40 (x, y) pairs are the **genuine points** that lie ON the polynomial
5. **Add 320 Chaff Points**: Generate 320 FAKE (x, y) pairs that do NOT lie on the polynomial
6. **Shuffle**: Mix genuine and chaff points randomly → now it's 360 total points, and an attacker can't tell which are real

**During Authentication (Unlocking):**
```
New Embedding → Quantize → Match against vault points → If enough match → Recover polynomial → Recover key
```

1. **Dual-Gate Security:**
   - **Gate 1 (Cosine Similarity)**: Compare query embedding against enrolled embedding. If similarity < 0.50 → **REJECT immediately**
   - **Gate 2 (Vault Matching)**: Quantize query, find matching x-coordinates in the vault. If ≥5 genuine points match → try to decode the polynomial using Reed-Solomon error correction

> **Analogy:** Imagine a room with 360 treasure chests. Only 40 contain real gold. If you have the right "biometric key," you can identify which 40 are real and recover the treasure (secret key). An attacker sees 360 identical chests and has no idea which are real.

---

### 🟢 STEP 5: Hybrid Encryption (`crypto/hybrid_crypto.py`)

**What it does:** Encrypts the entire vault data so that it can be safely stored on the internet without anyone being able to read it.

#### The Two-Layer Encryption Pipeline:

```
Vault Data (JSON) ──→ AES-256-GCM ──→ Encrypted Data
Random AES Key ──→ ECC (ECIES) ──→ Encrypted AES Key
```

**Step by step:**
1. **Generate a random AES-256 key** (32 bytes of random numbers)
2. **AES-256-GCM encrypts the vault data** (fast, handles large data)
   - GCM mode provides both **encryption** (secrecy) AND **authentication** (tamper detection) in one operation
   - Uses a random 96-bit nonce (number used once) to ensure the same data encrypts differently each time
3. **ECC (ECIES) encrypts the AES key** (only 32 bytes, so asymmetric crypto is fine)
   - Uses **secp256k1** elliptic curve (same as Bitcoin/Ethereum)
   - Only the person with the **ECC private key** can decrypt the AES key

**Why two layers?**
- **AES** is fast for big data but uses the same key for encrypt/decrypt (symmetric)
- **ECC** can use different keys for encrypt/decrypt (asymmetric) but is slow for big data
- **Hybrid** = Best of both: AES handles the heavy lifting, ECC protects the small AES key

**Components:**

| File | What It Does |
|------|-------------|
| `aes_encryption.py` | AES-256-GCM encrypt/decrypt operations |
| `ecc_encryption.py` | ECC key generation + ECIES key wrapping (falls back to RSA-2048 if eciespy is unavailable) |
| `hybrid_crypto.py` | Orchestrates the full pipeline: AES encrypts data, ECC encrypts the AES key |

---

### 🟢 STEP 6: Decentralized Storage (`storage/`)

**What it does:** Stores the encrypted vault on **IPFS** (distributed file storage) and records the reference on **Ethereum blockchain** (immutable ledger).

#### IPFS Storage (`ipfs_client.py`):
```
Encrypted Vault JSON → Upload to IPFS → Get back CID (content hash)
```
- **IPFS (InterPlanetary File System)** = A distributed file system where files are stored across many computers, not one central server
- **CID (Content Identifier)** = A unique hash of the file contents. If the content changes, the CID changes → **tamper-proof**
- Supports 3 modes:
  1. **Local IPFS daemon** (if running on your machine)
  2. **Pinata cloud** (IPFS hosting service, needs API key)
  3. **Mock storage** (saves files locally for development/testing)

#### Blockchain Storage (`blockchain_client.py`):
```
CID + Encrypted Key + User ID → Smart Contract → Ethereum Transaction
```
- The **Ethereum smart contract** (`BiometricVault.sol`) stores:
  - The IPFS CID (pointer to where the vault is stored)
  - The encrypted AES key
  - User metadata and timestamp
  - Active/revoked status
- **Why blockchain?** Once data is written, it **cannot be altered or deleted** → provides an immutable audit trail
- Also supports **mock mode** (local JSON file) for development without running a blockchain

#### Smart Contract (`BiometricVault.sol`):
Key functions:
| Function | What It Does |
|----------|-------------|
| `storeCredentials()` | Save a user's encrypted vault reference on-chain |
| `getCredentials()` | Retrieve a user's vault reference |
| `revokeCredentials()` | Disable a user's credentials (can't authenticate anymore) |
| `logAuthentication()` | Record auth attempts for audit trail |
| `isEnrolled()` | Check if a user exists and is active |

---

### 🟢 STEP 7: Enrollment — Putting It All Together (`auth/enrollment.py`)

**The complete enrollment (registration) flow:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER ENROLLMENT                              │
│                                                                     │
│  Face Photo + Eye Photo                                             │
│       │                                                             │
│       ▼                                                             │
│  ① Preprocess (MTCNN align, CLAHE, reflection removal, resize)     │
│       │                                                             │
│       ▼                                                             │
│  ② Extract Features (FaceNet → 512D, ResNet18+CBAM → 512D)         │
│       │                                                             │
│       ▼                                                             │
│  ③ Fuse (alpha × face + (1-alpha) × iris → 512D)                   │
│       │                                                             │
│       ▼                                                             │
│  ④ Fuzzy Vault Lock (embed in polynomial + add chaff points)       │
│       │                                                             │
│       ▼                                                             │
│  ⑤ Store enrolled embedding INSIDE vault (for cosine gate later)   │
│       │                                                             │
│       ▼                                                             │
│  ⑥ Hybrid Encrypt (AES encrypts vault → ECC encrypts AES key)     │
│       │                                                             │
│       ▼                                                             │
│  ⑦ Upload to IPFS → Get CID                                        │
│       │                                                             │
│       ▼                                                             │
│  ⑧ Store CID on Blockchain → Get transaction receipt                │
│       │                                                             │
│       ▼                                                             │
│  ✅ Return: IPFS CID + ECC Private Key (user must keep this safe!) │
└─────────────────────────────────────────────────────────────────────┘
```

> **⚠️ Security Note:** The enrolled fused embedding is stored INSIDE the encrypted vault. This is critical — during authentication, the system compares the new face/iris against this stored embedding. Without it, anyone with the ECC key could bypass biometric checks.

---

### 🟢 STEP 8: Authentication — Verifying Identity (`auth/authentication.py`)

**The complete authentication (login) flow with MANDATORY dual-stage verification:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                       USER AUTHENTICATION                            │
│                                                                      │
│  Face Photo + Eye Photo + ECC Private Key                            │
│       │                                                              │
│       ▼                                                              │
│  ① Check blockchain: Is user enrolled and active?                    │
│       │ (If not → REJECT)                                            │
│       ▼                                                              │
│  ② Preprocess query images (same pipeline as enrollment)             │
│       │                                                              │
│       ▼                                                              │
│  ③ Extract query features (FaceNet + ResNet18+CBAM → 512D each)     │
│       │                                                              │
│       ▼                                                              │
│  ④ Fuse query features → query_fused (512D)                          │
│       │                                                              │
│       ▼                                                              │
│  ⑤ Retrieve encrypted vault from IPFS using CID from blockchain     │
│       │                                                              │
│       ▼                                                              │
│  ⑥ Decrypt vault using ECC private key                               │
│       │ (If key is wrong → REJECT)                                   │
│       ▼                                                              │
│  ⑦ ★ GATE 1: Cosine Similarity Check ★                              │
│     • Extract enrolled embedding from decrypted vault                │
│     • Compute: similarity = dot(query, enrolled) / (||q|| × ||e||)  │
│     • If similarity < 0.52 → REJECT (biometric mismatch!)           │
│       │                                                              │
│       ▼                                                              │
│  ⑧ ★ GATE 2: Fuzzy Vault Unlock ★                                   │
│     • Quantize query features → match against vault points           │
│     • Try Reed-Solomon decode to recover secret key                  │
│     • If not enough matching points → REJECT                        │
│       │                                                              │
│       ▼                                                              │
│  ✅ BOTH gates passed → AUTHENTICATED!                               │
│  ❌ Either gate failed → REJECTED!                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Attack Prevention:**
| Attack | How It's Stopped |
|--------|-----------------|
| Stolen ECC key, wrong person | Gate 1 rejects (cosine similarity too low) |
| Right person, wrong key | Decryption fails (can't read vault) |
| Unenrolled person | Blockchain lookup fails |
| Tampered vault on IPFS | IPFS CID changes if content is modified → lookup fails |

---

### 🟢 STEP 9: REST API (`api/main.py`)

**What it does:** Provides web endpoints so any application (web, mobile, etc.) can use the biometric system over HTTP.

Built with **FastAPI** (modern Python web framework).

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/` | GET | Health check — is the system running? |
| `/ui` | GET | Opens the browser demo page |
| `/status` | GET | Shows system components and their status |
| `/enroll` | POST | Register a new user (upload face + iris images) |
| `/authenticate` | POST | Verify a user (upload face + iris images + optional key) |
| `/revoke/{user_id}` | POST | Disable a user's credentials |
| `/metrics` | GET | Show target performance metrics |

**How to run the API:**
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```
Then open `http://localhost:8000/ui` in your browser for the demo interface.

---

### 🟢 STEP 10: Configuration (`config/settings.py`)

**What it does:** One central file where ALL parameters and settings live. Change anything here and it affects the whole system.

Key settings explained:

| Setting | Value | What It Means |
|---------|-------|--------------|
| `IMAGE_SIZE` | 224 | All images are resized to 224×224 pixels |
| `FACE_EMBEDDING_DIM` | 512 | Face embeddings are 512 numbers long |
| `IRIS_EMBEDDING_DIM` | 512 | Iris embeddings are 512 numbers long |
| `FUSION_ALPHA` | 0.6 | Face gets 60% weight, iris gets 40% |
| `SIMILARITY_THRESHOLD` | 0.52 | Minimum cosine similarity to pass Gate 1 |
| `VAULT_FEATURE_POINTS` | 40 | Number of genuine points in the vault |
| `VAULT_CHAFF_MULTIPLIER` | 8 | 40 × 8 = 320 fake chaff points added |
| `MIN_MATCHING_POINTS` | 5 | Need at least 5 matching points to unlock |
| `AES_KEY_SIZE` | 32 | 256-bit AES encryption key |
| `ARCFACE_SCALE` | 30.0 | Angular scaling for ArcFace loss |
| `ARCFACE_MARGIN` | 0.5 | Angular margin (radians) between classes |

---

## 🚀 How to Run This Project

### Prerequisites
- Python 3.9+
- Node.js 18+ (for blockchain, optional)

### 1. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. (Optional) Train the Iris Model
```bash
python models/train.py
```
This trains the ResNet18+CBAM model on your iris dataset. Without training, the system uses untrained (ImageNet) weights.

### 3. Start the API Server
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```

### 4. Open the Demo UI
Go to `http://localhost:8000/ui` in your browser.

### 5. (Optional) Start Blockchain
```bash
cd blockchain
npm install
npx hardhat node                    # Start local blockchain
npx hardhat run scripts/deploy.js   # Deploy smart contract
```

---

## 📊 Target Performance Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| **Accuracy** | > 98.5% | Correctly accepts genuine users + rejects impostors |
| **FAR** (False Accept Rate) | < 2% | How often an impostor is wrongly accepted |
| **FRR** (False Reject Rate) | < 4% | How often a genuine user is wrongly rejected |
| **EER** (Equal Error Rate) | < 3% | The point where FAR = FRR (lower is better) |
| **Vault TAR** | > 95% | True Accept Rate of the fuzzy vault for genuine users |
| **Vault TRR** | > 98% | True Reject Rate of the fuzzy vault for impostors |

---

## 🛡️ Security Layers Summary

```
Layer 1: Biometric (your face + iris are unique to you)
    ↓
Layer 2: Fuzzy Vault (mathematical protection of biometric template)
    ↓
Layer 3: AES-256-GCM (military-grade symmetric encryption)
    ↓
Layer 4: ECC/ECIES (asymmetric key wrapping — only you have the private key)
    ↓
Layer 5: IPFS (content-addressable storage — tamper-detectable)
    ↓
Layer 6: Blockchain (immutable ledger — can't alter enrollment records)
```

**Key principle:** Raw biometric data is NEVER stored anywhere. Only encrypted, vault-protected representations exist.

---

## 🧰 Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Deep Learning** | PyTorch, FaceNet, ResNet18 | Feature extraction from images |
| **Attention** | CBAM | Focus on discriminative iris regions |
| **Loss Function** | ArcFace + Triplet | Angular margin training for better separation |
| **Template Protection** | Fuzzy Vault + Reed-Solomon | Protect biometric templates mathematically |
| **Encryption** | AES-256-GCM, ECIES (secp256k1) | Encrypt vault data and wrap keys |
| **Storage** | IPFS, Ethereum Blockchain | Decentralized, tamper-proof storage |
| **API** | FastAPI + Uvicorn | REST API for web/mobile integration |
| **Smart Contract** | Solidity (Hardhat) | On-chain credential management |
| **Image Processing** | OpenCV, MTCNN | Face detection, iris localization, CLAHE |

---

## 📚 Datasets

| Dataset | Used For | Details |
|---------|----------|---------|
| **LFW (Labeled Faces in the Wild)** | Face recognition testing | 13,000+ face images of 5,700+ people |
| **CASIA-Iris-Thousand** | Iris model training & testing | 20,000 iris images from 1,000 subjects |
| **VGGFace2** (pretrained) | Face embedding model weights | 3.31M images, 9,131 identities |

---

## 📝 License

This project is built for academic/research purposes as part of a biometric authentication study.

---

> **Built with ❤️ using PyTorch, FastAPI, Ethereum, and IPFS**
