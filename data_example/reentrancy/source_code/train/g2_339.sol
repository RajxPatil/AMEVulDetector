pragma solidity ^0.4.24;
contract TimedCrowdsale {
  uint256 public closingTime;

  function hasClosed() public view returns (bool) {
    return block.timestamp > closingTime;
  }
}