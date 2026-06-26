"""
Security test: Verify the biometric auth system correctly:
1. Rejects unenrolled users
2. Accepts genuine users (same biometrics)
3. Rejects impostors (different biometrics + stolen key)
"""
import requests
import json
import os

BASE = "http://localhost:8000"

# Dataset paths
FACE_DIR = r"C:\Users\murli\Desktop\Captain\backend\datasets\lfw-deepfunneled"
IRIS_DIR = r"C:\Users\murli\Desktop\Captain\backend\datasets\eye"

# Find two different face images
face_dirs = sorted(os.listdir(FACE_DIR))
face1_dir = os.path.join(FACE_DIR, face_dirs[0])
face1_img = os.path.join(face1_dir, os.listdir(face1_dir)[0])
face2_dir = os.path.join(FACE_DIR, face_dirs[1])
face2_img = os.path.join(face2_dir, os.listdir(face2_dir)[0])

# Find iris images
iris_imgs = sorted([f for f in os.listdir(IRIS_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
iris1_img = os.path.join(IRIS_DIR, iris_imgs[0])

print(f"Face 1: {face1_img}")
print(f"Face 2: {face2_img}")
print(f"Iris 1: {iris1_img}")
print()

# ═══════════════════════════════════════════════════════════
# TEST 1: Enroll "Professor Demo"
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Enroll 'Professor Demo'")
print("=" * 60)

r = requests.post(
    f"{BASE}/enroll",
    data={"user_id": "Professor Demo"},
    files={
        "face_image": ("face.jpg", open(face1_img, "rb"), "image/jpeg"),
        "iris_image": ("iris.jpg", open(iris1_img, "rb"), "image/jpeg"),
    },
)
enroll_result = r.json()
print(f"Status: {r.status_code}")
print(f"Success: {enroll_result.get('message', enroll_result.get('error'))}")

# Save the private key for later tests
private_key = enroll_result.get("private_key") or enroll_result.get("ecc_private_key")
ipfs_cid = enroll_result.get("ipfs_cid")
print(f"IPFS CID: {ipfs_cid}")
print(f"Private key received: {'Yes' if private_key else 'No'}")
print()

if not private_key:
    print("ERROR: No private key returned from enrollment!")
    exit(1)

# ═══════════════════════════════════════════════════════════
# TEST 2: Authenticate with SAME images (should PASS)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Authenticate 'Professor Demo' with SAME images")
print("=" * 60)

r = requests.post(
    f"{BASE}/authenticate",
    data={
        "user_id": "Professor Demo",
        "ecc_private_key": private_key,
    },
    files={
        "face_image": ("face.jpg", open(face1_img, "rb"), "image/jpeg"),
        "iris_image": ("iris.jpg", open(iris1_img, "rb"), "image/jpeg"),
    },
)
auth_result = r.json()
print(f"Authenticated: {auth_result['authenticated']}")
print(f"Stages: {json.dumps(auth_result.get('stages', {}), indent=2)}")
if auth_result.get("error"):
    print(f"Error: {auth_result['error']}")
assert auth_result["authenticated"] == True, "FAIL: Should accept genuine user!"
print(">>> PASSED: Genuine user accepted")
print()

# ═══════════════════════════════════════════════════════════
# TEST 3: Authenticate with DIFFERENT face (should FAIL)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Authenticate with DIFFERENT person's face + stolen key")
print("=" * 60)

r = requests.post(
    f"{BASE}/authenticate",
    data={
        "user_id": "Professor Demo",
        "ecc_private_key": private_key,  # Using the STOLEN key
    },
    files={
        "face_image": ("face.jpg", open(face2_img, "rb"), "image/jpeg"),  # DIFFERENT face
        "iris_image": ("iris.jpg", open(iris1_img, "rb"), "image/jpeg"),  # Same iris
    },
)
auth_result = r.json()
print(f"Authenticated: {auth_result['authenticated']}")
print(f"Stages: {json.dumps(auth_result.get('stages', {}), indent=2)}")
if auth_result.get("error"):
    print(f"Error: {auth_result['error']}")
assert auth_result["authenticated"] == False, "FAIL: Should reject impostor!"
print(">>> PASSED: Impostor with stolen key REJECTED")
print()

# ═══════════════════════════════════════════════════════════
# TEST 4: Authenticate unenrolled user (should FAIL)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: Authenticate unenrolled 'Different Person'")
print("=" * 60)

r = requests.post(
    f"{BASE}/authenticate",
    data={
        "user_id": "Different Person",
        "ecc_private_key": private_key,
    },
    files={
        "face_image": ("face.jpg", open(face1_img, "rb"), "image/jpeg"),
        "iris_image": ("iris.jpg", open(iris1_img, "rb"), "image/jpeg"),
    },
)
auth_result = r.json()
print(f"Authenticated: {auth_result['authenticated']}")
if auth_result.get("error"):
    print(f"Error: {auth_result['error']}")
assert auth_result["authenticated"] == False, "FAIL: Should reject unenrolled user!"
assert "not enrolled" in auth_result.get("error", "").lower(), "FAIL: Should say 'not enrolled'"
print(">>> PASSED: Unenrolled user rejected with correct message")
print()

# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("ALL 4 SECURITY TESTS PASSED!")
print("=" * 60)
