/**
 * Deployment script for BiometricVault smart contract.
 * Deploys to local Hardhat/Ganache network.
 */

const hre = require("hardhat");

async function main() {
  console.log("Deploying BiometricVault contract...");

  const [deployer] = await hre.ethers.getSigners();
  console.log("Deployer address:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Deployer balance:", hre.ethers.formatEther(balance), "ETH");

  // Deploy contract
  const BiometricVault = await hre.ethers.getContractFactory("BiometricVault");
  const vault = await BiometricVault.deploy();
  await vault.waitForDeployment();

  const contractAddress = await vault.getAddress();
  console.log("BiometricVault deployed to:", contractAddress);
  console.log("\nAdd this to your .env file:");
  console.log(`CONTRACT_ADDRESS=${contractAddress}`);

  // Verify deployment
  const owner = await vault.owner();
  console.log("Contract owner:", owner);

  const userCount = await vault.getUserCount();
  console.log("Initial user count:", userCount.toString());
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Deployment failed:", error);
    process.exit(1);
  });
