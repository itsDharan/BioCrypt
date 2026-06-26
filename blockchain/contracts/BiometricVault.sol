// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title BiometricVault
 * @dev Smart contract for decentralized biometric credential storage.
 * 
 * Stores:
 *   - IPFS CID (hash of encrypted fuzzy vault)
 *   - Encrypted AES key (ECC-wrapped)
 *   - User metadata
 *
 * Features:
 *   - Per-user access control
 *   - Credential revocation mechanism
 *   - Event logging for audit trail
 *   - Admin management
 */
contract BiometricVault {
    
    // ── Structs ──────────────────────────────────────────────────────
    
    struct Credential {
        string ipfsCid;          // IPFS CID of encrypted vault
        string encryptedKey;     // ECC-encrypted AES key (base64)
        string metadata;         // Optional metadata
        uint256 timestamp;       // Creation timestamp
        uint256 version;         // Enrollment version (increments on re-enrollment)
        bool active;             // Revocation flag
        bool exists;             // Existence flag
    }
    
    struct AuthLog {
        uint256 timestamp;       // When authentication was attempted
        bool success;            // Whether authentication succeeded
        uint256 gasUsed;         // Gas consumed by the log transaction
    }
    
    // ── State Variables ──────────────────────────────────────────────
    
    address public owner;
    mapping(string => Credential) private credentials;  // userId -> Credential
    mapping(string => address) private userOwners;       // userId -> owner address
    mapping(string => AuthLog[]) private authHistory;    // userId -> auth logs
    string[] private userIds;                             // List of all user IDs
    uint256 public totalUsers;
    
    // ── Events ───────────────────────────────────────────────────────
    
    event CredentialStored(
        string indexed userId,
        string ipfsCid,
        uint256 timestamp
    );
    
    event CredentialRevoked(
        string indexed userId,
        uint256 timestamp
    );
    
    event CredentialUpdated(
        string indexed userId,
        string newIpfsCid,
        uint256 timestamp
    );
    
    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );
    
    event AuthenticationLogged(
        string indexed userId,
        bool success,
        uint256 timestamp
    );
    
    // ── Modifiers ────────────────────────────────────────────────────
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only contract owner can call this");
        _;
    }
    
    modifier onlyCredentialOwner(string memory userId) {
        require(
            userOwners[userId] == msg.sender || msg.sender == owner,
            "Not authorized to manage this credential"
        );
        _;
    }
    
    modifier credentialExists(string memory userId) {
        require(credentials[userId].exists, "Credential does not exist");
        _;
    }
    
    modifier credentialActive(string memory userId) {
        require(credentials[userId].active, "Credential has been revoked");
        _;
    }
    
    // ── Constructor ──────────────────────────────────────────────────
    
    constructor() {
        owner = msg.sender;
    }
    
    // ── Core Functions ───────────────────────────────────────────────
    
    /**
     * @dev Store biometric credentials for a user.
     * @param userId Unique user identifier
     * @param ipfsCid IPFS CID of the encrypted vault
     * @param encryptedKey ECC-encrypted AES key (base64)
     * @param metadata Optional metadata string
     */
    function storeCredentials(
        string memory userId,
        string memory ipfsCid,
        string memory encryptedKey,
        string memory metadata
    ) external {
        require(bytes(userId).length > 0, "User ID cannot be empty");
        require(bytes(ipfsCid).length > 0, "IPFS CID cannot be empty");
        require(bytes(encryptedKey).length > 0, "Encrypted key cannot be empty");
        
        // If credential already exists, only the owner can update
        if (credentials[userId].exists) {
            require(
                userOwners[userId] == msg.sender || msg.sender == owner,
                "Cannot overwrite existing credential"
            );
            
            credentials[userId].ipfsCid = ipfsCid;
            credentials[userId].encryptedKey = encryptedKey;
            credentials[userId].metadata = metadata;
            credentials[userId].timestamp = block.timestamp;
            credentials[userId].active = true;
            
            emit CredentialUpdated(userId, ipfsCid, block.timestamp);
        } else {
            credentials[userId] = Credential({
                ipfsCid: ipfsCid,
                encryptedKey: encryptedKey,
                metadata: metadata,
                timestamp: block.timestamp,
                version: 1,
                active: true,
                exists: true
            });
            
            userOwners[userId] = msg.sender;
            userIds.push(userId);
            totalUsers++;
            
            emit CredentialStored(userId, ipfsCid, block.timestamp);
        }
    }
    
    /**
     * @dev Retrieve credentials for a user.
     * @param userId User identifier to look up
     * @return ipfsCid, encryptedKey, metadata, timestamp, active
     */
    function getCredentials(string memory userId) 
        external 
        view 
        credentialExists(userId)
        credentialActive(userId)
        returns (
            string memory ipfsCid,
            string memory encryptedKey,
            string memory metadata,
            uint256 timestamp,
            bool active
        ) 
    {
        Credential storage cred = credentials[userId];
        return (
            cred.ipfsCid,
            cred.encryptedKey,
            cred.metadata,
            cred.timestamp,
            cred.active
        );
    }
    
    /**
     * @dev Revoke a user's credentials.
     * @param userId User identifier to revoke
     */
    function revokeCredentials(string memory userId) 
        external 
        onlyCredentialOwner(userId)
        credentialExists(userId) 
    {
        credentials[userId].active = false;
        emit CredentialRevoked(userId, block.timestamp);
    }
    
    /**
     * @dev Check if a user is enrolled and active.
     * @param userId User identifier to check
     * @return enrolled True if user has active credentials
     */
    function isEnrolled(string memory userId) 
        external 
        view 
        returns (bool enrolled) 
    {
        return credentials[userId].exists && credentials[userId].active;
    }
    
    /**
     * @dev Get the owner address of a credential.
     * @param userId User identifier
     * @return ownerAddress Address that created the credential
     */
    function getCredentialOwner(string memory userId) 
        external 
        view 
        returns (address ownerAddress) 
    {
        return userOwners[userId];
    }
    
    /**
     * @dev Transfer contract ownership.
     * @param newOwner Address of the new owner
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "New owner cannot be zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
    
    /**
     * @dev Get total number of registered users.
     * @return count Total user count
     */
    function getUserCount() external view returns (uint256 count) {
        return totalUsers;
    }
    
    /**
     * @dev Log an authentication attempt for audit trail.
     * @param userId User identifier
     * @param success Whether authentication succeeded
     */
    function logAuthentication(
        string memory userId,
        bool success
    ) external {
        uint256 gasStart = gasleft();
        
        authHistory[userId].push(AuthLog({
            timestamp: block.timestamp,
            success: success,
            gasUsed: gasStart - gasleft()
        }));
        
        emit AuthenticationLogged(userId, success, block.timestamp);
    }
    
    /**
     * @dev Get authentication history for a user.
     * @param userId User identifier
     * @return timestamps, successes, gasUsed arrays
     */
    function getAuthenticationHistory(string memory userId)
        external
        view
        returns (
            uint256[] memory timestamps,
            bool[] memory successes,
            uint256[] memory gasUsedArr
        )
    {
        AuthLog[] storage logs = authHistory[userId];
        uint256 length = logs.length;
        
        timestamps = new uint256[](length);
        successes = new bool[](length);
        gasUsedArr = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            timestamps[i] = logs[i].timestamp;
            successes[i] = logs[i].success;
            gasUsedArr[i] = logs[i].gasUsed;
        }
        
        return (timestamps, successes, gasUsedArr);
    }
    
    /**
     * @dev Get enrollment version for a user.
     * @param userId User identifier
     * @return version Current enrollment version number
     */
    function getEnrollmentVersion(string memory userId)
        external
        view
        returns (uint256 version)
    {
        return credentials[userId].version;
    }
}
