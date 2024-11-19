pragma solidity ^0.4.24;
contract TokenTimelock {
  uint64 public releaseTime;
  uint64 public amount;

  function release() public {
    require(uint64(block.timestamp) >= releaseTime);
    require(amount > 0);
    return;
  }
}
