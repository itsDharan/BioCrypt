"""
IPFS client for uploading and retrieving encrypted fuzzy vaults.
Supports local IPFS daemon, Pinata cloud service, and mock file storage.
"""

import json
import os
import hashlib
import requests
from typing import Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IPFS_HOST, IPFS_PORT, IPFS_GATEWAY,
    PINATA_JWT, VAULT_STORAGE_DIR,
)


class IPFSClient:
    """
    IPFS client with automatic fallback.
    Priority: Local IPFS daemon -> Pinata cloud -> Local mock storage
    """

    def __init__(self):
        self.mode = "mock"
        self.mock_dir = VAULT_STORAGE_DIR / "ipfs_mock"
        self.mock_dir.mkdir(parents=True, exist_ok=True)

        # Try local IPFS daemon
        try:
            resp = requests.post(
                f"http://{IPFS_HOST}:{IPFS_PORT}/api/v0/id",
                timeout=3,
            )
            if resp.status_code == 200:
                self.mode = "local"
                print("IPFS: Connected to local daemon")
                return
        except Exception:
            pass

        # Try Pinata
        if PINATA_JWT:
            try:
                resp = requests.get(
                    "https://api.pinata.cloud/data/testAuthentication",
                    headers={"Authorization": f"Bearer {PINATA_JWT}"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    self.mode = "pinata"
                    print("IPFS: Connected to Pinata cloud")
                    return
            except Exception:
                pass

        print("IPFS: Using local mock storage (no IPFS daemon or Pinata)")

    def upload(self, data: str, filename: str = "vault.json") -> str:
        """
        Upload data to IPFS.

        Args:
            data: String data to upload (typically JSON)
            filename: Optional filename for the upload

        Returns:
            IPFS CID (Content Identifier) or mock hash
        """
        import time as _time
        t0 = _time.time()

        if self.mode == "local":
            cid = self._upload_local(data, filename)
        elif self.mode == "pinata":
            cid = self._upload_pinata(data, filename)
        else:
            cid = self._upload_mock(data, filename)

        self._last_upload_latency_ms = (_time.time() - t0) * 1000
        self._last_upload_size_bytes = len(data.encode())
        return cid

    def retrieve(self, cid: str) -> str:
        """
        Retrieve data from IPFS by CID.

        Args:
            cid: IPFS Content Identifier

        Returns:
            Retrieved string data
        """
        import time as _time
        t0 = _time.time()

        if self.mode == "local":
            data = self._retrieve_local(cid)
        elif self.mode == "pinata":
            data = self._retrieve_pinata(cid)
        else:
            data = self._retrieve_mock(cid)

        self._last_retrieve_latency_ms = (_time.time() - t0) * 1000
        return data

    def pin(self, cid: str) -> bool:
        """Pin a CID to ensure persistence."""
        if self.mode == "local":
            try:
                resp = requests.post(
                    f"http://{IPFS_HOST}:{IPFS_PORT}/api/v0/pin/add?arg={cid}",
                    timeout=10,
                )
                return resp.status_code == 200
            except Exception:
                return False
        return True  # Mock/Pinata already persistent

    def unpin(self, cid: str) -> bool:
        """Unpin a CID (for revocation)."""
        if self.mode == "local":
            try:
                resp = requests.post(
                    f"http://{IPFS_HOST}:{IPFS_PORT}/api/v0/pin/rm?arg={cid}",
                    timeout=10,
                )
                return resp.status_code == 200
            except Exception:
                return False
        elif self.mode == "mock":
            mock_file = self.mock_dir / f"{cid}.json"
            if mock_file.exists():
                mock_file.unlink()
                return True
        return False

    # ── Local IPFS ────────────────────────────────────────────────────

    def _upload_local(self, data: str, filename: str) -> str:
        resp = requests.post(
            f"http://{IPFS_HOST}:{IPFS_PORT}/api/v0/add",
            files={"file": (filename, data.encode())},
            timeout=30,
        )
        result = resp.json()
        return result["Hash"]

    def _retrieve_local(self, cid: str) -> str:
        resp = requests.post(
            f"http://{IPFS_HOST}:{IPFS_PORT}/api/v0/cat?arg={cid}",
            timeout=30,
        )
        return resp.text

    # ── Pinata Cloud ──────────────────────────────────────────────────

    def _upload_pinata(self, data: str, filename: str) -> str:
        resp = requests.post(
            "https://api.pinata.cloud/pinning/pinJSONToIPFS",
            headers={
                "Authorization": f"Bearer {PINATA_JWT}",
                "Content-Type": "application/json",
            },
            json={
                "pinataContent": json.loads(data) if data.startswith("{") else {"data": data},
                "pinataMetadata": {"name": filename},
            },
            timeout=30,
        )
        result = resp.json()
        return result["IpfsHash"]

    def _retrieve_pinata(self, cid: str) -> str:
        resp = requests.get(
            f"{IPFS_GATEWAY}{cid}",
            timeout=30,
        )
        return resp.text

    # ── Mock Storage ──────────────────────────────────────────────────

    def _upload_mock(self, data: str, filename: str) -> str:
        """Store locally and return SHA-256 hash as mock CID."""
        cid = "Qm" + hashlib.sha256(data.encode()).hexdigest()[:44]
        mock_file = self.mock_dir / f"{cid}.json"
        mock_file.write_text(data, encoding="utf-8")
        return cid

    def _retrieve_mock(self, cid: str) -> str:
        """Retrieve from local mock storage."""
        mock_file = self.mock_dir / f"{cid}.json"
        if not mock_file.exists():
            raise FileNotFoundError(f"Mock IPFS: CID {cid} not found")
        return mock_file.read_text(encoding="utf-8")
