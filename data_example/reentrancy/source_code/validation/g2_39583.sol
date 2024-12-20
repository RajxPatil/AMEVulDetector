pragma solidity ^0.4.24;
contract ProofOfExistence {
  mapping (string => uint) private proofs;

  function storeProof(string sha256) {
        proofs[sha256] = block.timestamp;
  }
}