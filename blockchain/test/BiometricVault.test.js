/**
 * Tests for BiometricVault smart contract.
 */

const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("BiometricVault", function () {
  let vault;
  let owner, user1, user2, unauthorized;

  beforeEach(async function () {
    [owner, user1, user2, unauthorized] = await ethers.getSigners();
    const BiometricVault = await ethers.getContractFactory("BiometricVault");
    vault = await BiometricVault.deploy();
    await vault.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the deployer as owner", async function () {
      expect(await vault.owner()).to.equal(owner.address);
    });

    it("Should start with zero users", async function () {
      expect(await vault.getUserCount()).to.equal(0);
    });
  });

  describe("Store Credentials", function () {
    it("Should store new credentials", async function () {
      await expect(
        vault.connect(user1).storeCredentials(
          "user001",
          "QmTestHash123",
          "encrypted_key_base64",
          "test metadata"
        )
      ).to.emit(vault, "CredentialStored");

      expect(await vault.getUserCount()).to.equal(1);
      expect(await vault.isEnrolled("user001")).to.be.true;
    });

    it("Should reject empty user ID", async function () {
      await expect(
        vault.storeCredentials("", "QmHash", "key", "")
      ).to.be.revertedWith("User ID cannot be empty");
    });

    it("Should reject empty IPFS CID", async function () {
      await expect(
        vault.storeCredentials("user001", "", "key", "")
      ).to.be.revertedWith("IPFS CID cannot be empty");
    });

    it("Should reject empty encrypted key", async function () {
      await expect(
        vault.storeCredentials("user001", "QmHash", "", "")
      ).to.be.revertedWith("Encrypted key cannot be empty");
    });

    it("Should allow owner to update existing credentials", async function () {
      await vault.connect(user1).storeCredentials(
        "user001", "QmHash1", "key1", ""
      );

      // Original creator can update
      await expect(
        vault.connect(user1).storeCredentials(
          "user001", "QmHash2", "key2", "updated"
        )
      ).to.emit(vault, "CredentialUpdated");
    });

    it("Should prevent unauthorized updates", async function () {
      await vault.connect(user1).storeCredentials(
        "user001", "QmHash1", "key1", ""
      );

      await expect(
        vault.connect(unauthorized).storeCredentials(
          "user001", "QmHash2", "key2", ""
        )
      ).to.be.revertedWith("Cannot overwrite existing credential");
    });
  });

  describe("Retrieve Credentials", function () {
    beforeEach(async function () {
      await vault.connect(user1).storeCredentials(
        "user001", "QmTestCid", "enc_key_123", "some metadata"
      );
    });

    it("Should retrieve stored credentials", async function () {
      const [cid, key, meta, timestamp, active] = 
        await vault.getCredentials("user001");

      expect(cid).to.equal("QmTestCid");
      expect(key).to.equal("enc_key_123");
      expect(meta).to.equal("some metadata");
      expect(active).to.be.true;
      expect(timestamp).to.be.gt(0);
    });

    it("Should fail for non-existent user", async function () {
      await expect(
        vault.getCredentials("nonexistent")
      ).to.be.revertedWith("Credential does not exist");
    });
  });

  describe("Revoke Credentials", function () {
    beforeEach(async function () {
      await vault.connect(user1).storeCredentials(
        "user001", "QmTestCid", "enc_key", ""
      );
    });

    it("Should allow credential owner to revoke", async function () {
      await expect(
        vault.connect(user1).revokeCredentials("user001")
      ).to.emit(vault, "CredentialRevoked");

      expect(await vault.isEnrolled("user001")).to.be.false;
    });

    it("Should allow contract owner to revoke", async function () {
      await vault.connect(owner).revokeCredentials("user001");
      expect(await vault.isEnrolled("user001")).to.be.false;
    });

    it("Should prevent unauthorized revocation", async function () {
      await expect(
        vault.connect(unauthorized).revokeCredentials("user001")
      ).to.be.revertedWith("Not authorized to manage this credential");
    });

    it("Should prevent access to revoked credentials", async function () {
      await vault.connect(user1).revokeCredentials("user001");

      await expect(
        vault.getCredentials("user001")
      ).to.be.revertedWith("Credential has been revoked");
    });
  });

  describe("Enrollment Check", function () {
    it("Should return false for non-enrolled user", async function () {
      expect(await vault.isEnrolled("nobody")).to.be.false;
    });

    it("Should return true for enrolled user", async function () {
      await vault.storeCredentials("user001", "QmHash", "key", "");
      expect(await vault.isEnrolled("user001")).to.be.true;
    });

    it("Should return false for revoked user", async function () {
      await vault.storeCredentials("user001", "QmHash", "key", "");
      await vault.revokeCredentials("user001");
      expect(await vault.isEnrolled("user001")).to.be.false;
    });
  });

  describe("Ownership", function () {
    it("Should transfer ownership", async function () {
      await vault.transferOwnership(user1.address);
      expect(await vault.owner()).to.equal(user1.address);
    });

    it("Should prevent non-owner from transferring", async function () {
      await expect(
        vault.connect(user1).transferOwnership(user2.address)
      ).to.be.revertedWith("Only contract owner can call this");
    });

    it("Should prevent transfer to zero address", async function () {
      await expect(
        vault.transferOwnership(ethers.ZeroAddress)
      ).to.be.revertedWith("New owner cannot be zero address");
    });
  });
});
