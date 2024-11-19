pragma solidity ^0.4.24;
contract Crowdsale{
  uint256 public endTime;

  function validPurchase() internal view returns (bool) {
    bool withinPeriod = block.timestamp <= endTime;
    return withinPeriod;
  }
}