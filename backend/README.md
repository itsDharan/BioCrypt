# Multimodal Biometric Authentication System

A research-grade multimodal biometric authentication system combining **face** and **dorsal hand** biometrics with advanced cryptographic protection and decentralized storage.

## Architecture

```
Face Image + Hand Image
       |
Preprocessing (MTCNN alignment, CLAHE, resize 224x224)
       |
Feature Extraction
  |-- Face: FaceNet (InceptionResnetV1, VGGFace2 pretrained) -> 512D
  |-- Hand: ResNet18 (ImageNet pretrained) + CBAM attention -> 512D
       |
Feature Fusion: Weighted (0.6*face + 0.4*hand)
       |
Fused 512D Embedding
       |
Fuzzy Vault (Reed-Solomon + chaff points)
       |
Hybrid Encryption (AES-256-GCM + ECC/ECIES key wrap)
       |
Decentralized Storage
  |-- IPFS -> encrypted vault data
  |-- Ethereum Smart Contract -> IPFS CID + encrypted AES key
```

## Target Metrics

| Metric | Target |
|--------|--------|
| Accuracy | >98.5% |
| FAR | <2% |
| FRR | <4% |
| EER | <3% |
| Vault TAR | >95% |
| Vault TRR | >98% |

## Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Download Datasets

- **Faces**: [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) -> `backend/datasets/lfw-deepfunneled/`
- **Hands**: [Hands & Palm Dataset](https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset) -> `backend/datasets/Hands/`

### 3. Train the Hand Model

```bash
cd backend
python models/train.py
```

> The face model uses **pretrained FaceNet** — no training needed.

### 4. Run Evaluation

```bash
python evaluation/run_evaluation.py --skip-training
```

### 5. Start API Server

```bash
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for Swagger UI.

### 6. Smart Contract (Optional)

```bash
cd blockchain
npm install
npx hardhat compile
npx hardhat test
npx hardhat node  # Start local chain
npx hardhat run scripts/deploy.js --network localhost
```

## Project Structure

```
Captain/
|-- backend/
|   |-- api/              # FastAPI REST endpoints
|   |-- auth/             # Enrollment & authentication pipelines
|   |-- config/           # Global configuration
|   |-- crypto/           # Fuzzy vault, AES, ECC, hybrid encryption
|   |-- data/             # Preprocessing & dataset loaders
|   |-- evaluation/       # Metrics, benchmarks, evaluation runner
|   |-- models/           # Face/hand feature extractors, fusion, training
|   |-- storage/          # IPFS & blockchain clients
|   |-- datasets/         # LFW + Hands datasets (not tracked)
|   |-- saved_models/     # Trained model checkpoints
|   |-- vault_storage/    # Local vault storage (mock IPFS/blockchain)
|   |-- evaluation_results/ # Generated plots and reports
|   |-- requirements.txt
|   |-- .env.example
|   |-- README.md
|
|-- blockchain/
|   |-- contracts/        # BiometricVault.sol
|   |-- scripts/          # Deployment script
|   |-- test/             # Smart contract tests
|   |-- hardhat.config.js
|   |-- package.json
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System info / health check |
| GET | `/status` | Component health status |
| POST | `/enroll` | Enroll user (face + hand images) |
| POST | `/authenticate` | Verify user identity |
| POST | `/revoke/{user_id}` | Revoke credentials |
| GET | `/metrics` | Target performance metrics |

## Tech Stack

- **Backend**: Python, FastAPI
- **ML**: PyTorch, FaceNet, ResNet18, CBAM
- **Crypto**: AES-256-GCM, ECC (ECIES), Reed-Solomon
- **Blockchain**: Solidity, Hardhat, Web3.py
- **Storage**: IPFS (local/Pinata), Ethereum
