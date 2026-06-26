"""
FastAPI backend for the Multimodal Biometric Authentication System.
Provides REST API endpoints for enrollment, authentication, and system management.
"""

import io
import cv2
import numpy as np
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


# ── Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal Biometric Authentication API",
    description=(
        "Secure biometric authentication using face + iris "
        "with fuzzy vault, hybrid AES+ECC encryption, and "
        "IPFS/blockchain decentralized storage."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (videos, images, etc.) from the frontend directory
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

# Lazy-loaded managers (avoids loading models at import time)
_enrollment_manager = None
_auth_manager = None


def _get_enrollment_manager():
    global _enrollment_manager
    if _enrollment_manager is None:
        from auth.enrollment import EnrollmentManager
        _enrollment_manager = EnrollmentManager()
    return _enrollment_manager


def _get_auth_manager():
    global _auth_manager
    if _auth_manager is None:
        from auth.authentication import AuthenticationManager
        _auth_manager = AuthenticationManager()
    return _auth_manager


async def _read_image(file: UploadFile) -> np.ndarray:
    """Read uploaded image file to numpy array."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail=f"Empty file received: {file.filename}")
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}. Ensure it is a valid JPEG/PNG image.")
        return img
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image read error for {file.filename}: {str(e)}")


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """System welcome / health check."""
    return {
        "system": "Multimodal Biometric Authentication",
        "version": "2.0.0",
        "status": "online",
        "endpoints": ["/ui", "/enroll", "/authenticate", "/status", "/revoke/{user_id}"],
    }


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui_page():
    """Serve the browser demo UI."""
    html_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/status", tags=["System"])
async def system_status():
    """Get system status and component health."""
    from config.settings import SAVED_MODELS_DIR, DEVICE

    iris_model_exists = (SAVED_MODELS_DIR / "iris_model_best.pth").exists()

    return {
        "status": "online",
        "device": str(DEVICE),
        "iris_model_trained": iris_model_exists,
        "components": {
            "face_extractor": "FaceNet (VGGFace2 pretrained)",
            "iris_extractor": "ResNet18 + CBAM (Iris)",
            "fusion": "Weighted (alpha=0.6)",
            "vault": "Improved Fuzzy Vault (RS + Chaff)",
            "encryption": "AES-256-GCM + ECC (ECIES)",
            "storage": "IPFS + Ethereum (mock fallback)",
        },
    }


@app.post("/enroll", tags=["Biometric"])
async def enroll_user(
    user_id: str = Form(..., description="Unique user identifier"),
    face_image: UploadFile = File(..., description="Face image file"),
    iris_image: UploadFile = File(..., description="Iris/eye image file"),
):
    """
    Enroll a new user with face + iris biometrics.

    Process:
    1. Preprocess images (MTCNN alignment, CLAHE, resize)
    2. Extract features (FaceNet 512D + ResNet18 512D)
    3. Fuse features (weighted fusion)
    4. Lock in fuzzy vault (Reed-Solomon + chaff)
    5. Encrypt (AES-256 + ECC key wrap)
    6. Store on IPFS + blockchain
    """
    try:
        manager = _get_enrollment_manager()

        face_img = await _read_image(face_image)
        iris_img = await _read_image(iris_image)

        result = manager.enroll(user_id, face_img, iris_img)

        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"User {user_id} enrolled successfully",
                    "ipfs_cid": result["ipfs_cid"],
                    "key_hash": result["key_hash"],
                    "total_time_ms": round(result["total_time"] * 1000, 1),
                    "step_times_ms": {
                        k: round(v * 1000, 1) for k, v in result["steps"].items()
                    },
                    "ecc_public_key": result["ecc_public_key"],
                    "ecc_private_key": result.get("ecc_private_key"),
                    "private_key": result.get("private_key", result.get("ecc_private_key")),
                    "enrolled_private_key": result.get("private_key", result.get("ecc_private_key")),
                },
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": result.get("error", "Enrollment failed")},
            )
    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"error": he.detail},
        )
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Enrollment failed: {str(e)}",
                "type": type(e).__name__,
            },
        )


@app.post("/authenticate", tags=["Biometric"])
async def authenticate_user(
    user_id: str = Form(..., description="User identifier to verify"),
    face_image: UploadFile = File(..., description="Face image file"),
    iris_image: UploadFile = File(..., description="Iris/eye image file"),
    ecc_private_key: Optional[str] = Form("", description="User's ECC private key (optional — auto-retrieved if omitted)"),
):
    """
    Authenticate a user via MANDATORY dual-stage biometric verification.

    Security model:
    1. User must be enrolled (blockchain lookup)
    2. ECC private key must decrypt the vault (possession check)
    3. Face + Iris biometrics must match enrolled template (cosine similarity)
    4. Fuzzy vault structure must verify (polynomial check)

    ALL stages must pass. Having the ECC key alone is NOT sufficient.
    """
    manager = _get_auth_manager()

    face_img = await _read_image(face_image)
    iris_img = await _read_image(iris_image)

    # Basic key format validation (only if key was manually provided)
    stripped_key = (ecc_private_key or "").strip()

    if stripped_key and stripped_key.startswith("Qm") and len(stripped_key) in (46, 59):
        raise HTTPException(
            status_code=400,
            detail=(
                "The value provided appears to be an IPFS CID, not the private key. "
                "You can leave the key field empty — the system will auto-retrieve it."
            ),
        )

    result = manager.authenticate(
        user_id=user_id,
        face_input=face_img,
        iris_input=iris_img,
        ecc_private_key=stripped_key,
    )

    return JSONResponse(
        status_code=200,
        content={
            "user_id": user_id,
            "authenticated": result["authenticated"],
            "stages": result.get("stages", {}),
            "total_time_ms": round(result.get("total_time", 0) * 1000, 1),
            "error": result.get("error"),
        },
    )


@app.post("/revoke/{user_id}", tags=["Management"])
async def revoke_credentials(user_id: str):
    """Revoke a user's biometric credentials."""
    manager = _get_enrollment_manager()
    success = manager.blockchain.revoke_credentials(user_id)

    if success:
        return {"message": f"Credentials for {user_id} revoked successfully"}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found or already revoked",
        )


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """Get system performance metrics."""
    from config.settings import (
        TARGET_ACCURACY, TARGET_FAR, TARGET_FRR, TARGET_EER,
        TARGET_VAULT_TAR, TARGET_VAULT_TRR,
    )
    return {
        "target_metrics": {
            "accuracy": f">{TARGET_ACCURACY*100:.1f}%",
            "far": f"<{TARGET_FAR*100:.1f}%",
            "frr": f"<{TARGET_FRR*100:.1f}%",
            "eer": f"<{TARGET_EER*100:.1f}%",
            "vault_tar": f">{TARGET_VAULT_TAR*100:.1f}%",
            "vault_trr": f">{TARGET_VAULT_TRR*100:.1f}%",
        },
        "note": "Run evaluation/run_evaluation.py for actual computed metrics",
    }
