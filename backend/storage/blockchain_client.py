"""
Blockchain client for interacting with the BiometricVault smart contract.
Uses Web3.py to connect to Ethereum (Hardhat/Ganache local network).
Falls back to mock storage for development without running blockchain.
"""

import json
import os
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    BLOCKCHAIN_RPC_URL, CONTRACT_ADDRESS,
    DEPLOYER_PRIVATE_KEY, CHAIN_ID,
    VAULT_STORAGE_DIR,
)


class BlockchainClient:
    """
    Ethereum blockchain client for BiometricVault smart contract.
    Falls back to local JSON mock storage if blockchain is not available.
    """

    def __init__(self, contract_address: str = None, abi_path: str = None):
        self.mode = "mock"
        self.web3 = None
        self.contract = None
        self.account = None

        # Mock storage
        self.mock_dir = VAULT_STORAGE_DIR / "blockchain_mock"
        self.mock_dir.mkdir(parents=True, exist_ok=True)
        self.mock_db_path = self.mock_dir / "registry.json"

        # Try connecting to blockchain
        contract_addr = contract_address or CONTRACT_ADDRESS
        if contract_addr:
            try:
                from web3 import Web3
                self.web3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_RPC_URL))

                if self.web3.is_connected():
                    # Load contract ABI
                    abi = self._load_abi(abi_path)
                    if abi:
                        self.contract = self.web3.eth.contract(
                            address=Web3.to_checksum_address(contract_addr),
                            abi=abi,
                        )
                        # Setup account
                        if DEPLOYER_PRIVATE_KEY:
                            self.account = self.web3.eth.account.from_key(DEPLOYER_PRIVATE_KEY)
                        else:
                            self.account = self.web3.eth.accounts[0] if self.web3.eth.accounts else None

                        self.mode = "blockchain"
                        print(f"Blockchain: Connected to {BLOCKCHAIN_RPC_URL}")
                        return
            except Exception as e:
                print(f"Blockchain connection failed: {e}")

        print("Blockchain: Using local mock storage")

    def _load_abi(self, abi_path: str = None) -> Optional[list]:
        """Load contract ABI from compiled artifacts."""
        search_paths = [
            abi_path,
            str(Path(__file__).resolve().parent.parent.parent / "blockchain" / "artifacts" /
                "contracts" / "BiometricVault.sol" / "BiometricVault.json"),
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                return data.get("abi", data)

        return None

    def _load_mock_db(self) -> Dict:
        """Load mock registry, defaulting to empty users dict if file is missing or empty."""
        try:
            if self.mock_db_path.exists():
                content = self.mock_db_path.read_text(encoding="utf-8").strip()
                if content:
                    return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: {self.mock_db_path} contains invalid JSON, creating new registry")
        except Exception as e:
            print(f"Warning: Error loading mock registry: {e}")
        
        return {"users": {}}

    def _save_mock_db(self, db: Dict):
        """Save mock registry."""
        self.mock_db_path.write_text(json.dumps(db, indent=2), encoding="utf-8")

    # ── Store Credentials ─────────────────────────────────────────────

    def store_credentials(
        self,
        user_id: str,
        ipfs_cid: str,
        encrypted_key: str,
        metadata: str = "",
        ecc_private_key: str = "",
        ecc_public_key: str = "",
    ) -> Dict[str, Any]:
        """
        Store biometric credentials on-chain.

        Args:
            user_id: Unique user identifier
            ipfs_cid: IPFS CID of the encrypted vault
            encrypted_key: Base64 ECC-encrypted AES key
            metadata: Optional metadata string

        Returns:
            Transaction result dictionary
        """
        t0 = time.time()
        if self.mode == "blockchain":
            result = self._store_blockchain(user_id, ipfs_cid, encrypted_key, metadata)
        else:
            result = self._store_mock(user_id, ipfs_cid, encrypted_key, metadata)
        result["latency_ms"] = (time.time() - t0) * 1000

        # Store ECC keys in a separate secure key-store file
        # (kept separate from blockchain/mock registry for security layering)
        if ecc_private_key:
            self._store_ecc_key(user_id, ecc_private_key, ecc_public_key)
        return result

    def _store_blockchain(self, user_id, ipfs_cid, encrypted_key, metadata):
        """Store on actual blockchain."""
        tx = self.contract.functions.storeCredentials(
            user_id, ipfs_cid, encrypted_key, metadata
        ).build_transaction({
            "chainId": CHAIN_ID,
            "from": self.account.address if hasattr(self.account, 'address') else self.account,
            "nonce": self.web3.eth.get_transaction_count(
                self.account.address if hasattr(self.account, 'address') else self.account
            ),
            "gas": 500000,
        })

        if hasattr(self.account, 'sign_transaction'):
            signed = self.account.sign_transaction(tx)
            tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = self.web3.eth.send_transaction(tx)

        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "success": receipt["status"] == 1,
            "tx_hash": receipt["transactionHash"].hex(),
            "block": receipt["blockNumber"],
            "gas_used": receipt["gasUsed"],
        }

    def _store_mock(self, user_id, ipfs_cid, encrypted_key, metadata):
        """Store in local mock registry."""
        db = self._load_mock_db()
        db["users"][user_id] = {
            "ipfs_cid": ipfs_cid,
            "encrypted_key": encrypted_key,
            "metadata": metadata,
            "timestamp": int(time.time()),
            "active": True,
        }
        self._save_mock_db(db)

        return {
            "success": True,
            "tx_hash": f"mock_tx_{user_id}_{int(time.time())}",
            "block": 0,
            "gas_used": 0,
        }

    # ── Retrieve Credentials ──────────────────────────────────────────

    def get_credentials(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored credentials for a user.

        Returns:
            Dictionary with ipfs_cid, encrypted_key, etc. or None
        """
        t0 = time.time()
        if self.mode == "blockchain":
            result = self._get_blockchain(user_id)
        else:
            result = self._get_mock(user_id)
        if result is not None:
            result["latency_ms"] = (time.time() - t0) * 1000
        return result

    def _get_blockchain(self, user_id):
        """Retrieve from blockchain."""
        try:
            result = self.contract.functions.getCredentials(user_id).call()
            return {
                "ipfs_cid": result[0],
                "encrypted_key": result[1],
                "metadata": result[2],
                "timestamp": result[3],
                "active": result[4],
            }
        except Exception:
            return None

    def _get_mock(self, user_id):
        """Retrieve from mock registry."""
        db = self._load_mock_db()
        return db["users"].get(user_id)

    # ── Revoke Credentials ────────────────────────────────────────────

    def revoke_credentials(self, user_id: str) -> bool:
        """Revoke a user's stored credentials."""
        if self.mode == "blockchain":
            try:
                tx = self.contract.functions.revokeCredentials(user_id).build_transaction({
                    "chainId": CHAIN_ID,
                    "from": self.account.address if hasattr(self.account, 'address') else self.account,
                    "nonce": self.web3.eth.get_transaction_count(
                        self.account.address if hasattr(self.account, 'address') else self.account
                    ),
                    "gas": 200000,
                })

                if hasattr(self.account, 'sign_transaction'):
                    signed = self.account.sign_transaction(tx)
                    tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(tx)

                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt["status"] == 1
            except Exception:
                return False
        else:
            db = self._load_mock_db()
            if user_id in db["users"]:
                db["users"][user_id]["active"] = False
                self._save_mock_db(db)
                return True
            return False

    # ── ECC Key Storage ───────────────────────────────────────────────

    def _ecc_keystore_path(self) -> Path:
        """Path to the ECC key store file."""
        return self.mock_dir / "ecc_keystore.json"

    def _load_ecc_keystore(self) -> Dict:
        """Load the ECC key store."""
        path = self._ecc_keystore_path()
        try:
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: ECC keystore load error: {e}")
        return {}

    def _save_ecc_keystore(self, keystore: Dict):
        """Save the ECC key store."""
        self._ecc_keystore_path().write_text(
            json.dumps(keystore, indent=2), encoding="utf-8"
        )

    def _store_ecc_key(self, user_id: str, private_key: str, public_key: str = ""):
        """Store ECC keys for a user in the local key store."""
        ks = self._load_ecc_keystore()
        ks[user_id] = {
            "private_key": private_key,
            "public_key": public_key,
            "timestamp": int(time.time()),
        }
        self._save_ecc_keystore(ks)

    def get_ecc_private_key(self, user_id: str) -> Optional[str]:
        """
        Retrieve the stored ECC private key for a user.

        Returns:
            The ECC private key hex string, or None if not found.
        """
        ks = self._load_ecc_keystore()
        entry = ks.get(user_id)
        if entry:
            return entry.get("private_key")
        return None

    # ── Check Enrollment ──────────────────────────────────────────────

    def is_enrolled(self, user_id: str) -> bool:
        """Check if a user is enrolled and active."""
        creds = self.get_credentials(user_id)
        if creds is None:
            return False
        return creds.get("active", False)

    # ── Authentication Logging ────────────────────────────────────────

    def log_authentication(self, user_id: str, success: bool) -> Dict[str, Any]:
        """
        Log an authentication attempt on-chain.

        Args:
            user_id: User identifier
            success: Whether authentication succeeded

        Returns:
            Transaction result with gas and latency
        """
        t0 = time.time()
        if self.mode == "blockchain":
            try:
                tx = self.contract.functions.logAuthentication(
                    user_id, success
                ).build_transaction({
                    "chainId": CHAIN_ID,
                    "from": self.account.address if hasattr(self.account, 'address') else self.account,
                    "nonce": self.web3.eth.get_transaction_count(
                        self.account.address if hasattr(self.account, 'address') else self.account
                    ),
                    "gas": 200000,
                })

                if hasattr(self.account, 'sign_transaction'):
                    signed = self.account.sign_transaction(tx)
                    tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(tx)

                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return {
                    "success": receipt["status"] == 1,
                    "gas_used": receipt["gasUsed"],
                    "latency_ms": (time.time() - t0) * 1000,
                }
            except Exception as e:
                return {"success": False, "error": str(e), "latency_ms": (time.time() - t0) * 1000}
        else:
            # Mock: append to registry
            db = self._load_mock_db()
            if "auth_logs" not in db:
                db["auth_logs"] = {}
            if user_id not in db["auth_logs"]:
                db["auth_logs"][user_id] = []
            db["auth_logs"][user_id].append({
                "timestamp": int(time.time()),
                "success": success,
            })
            self._save_mock_db(db)
            return {
                "success": True,
                "gas_used": 0,
                "latency_ms": (time.time() - t0) * 1000,
            }
