pragma solidity ^0.4.24;
contract CHAltar {
  uint256 public genesis;

  function _getCurrentAltarRecordId() internal view returns (uint256) {
    return (block.timestamp - genesis) / 86400;
  }
}