pragma solidity ^0.4.24;
contract LifMarketValidationMechanism {

 uint256 public startTimestamp;

  function getCurrentPeriodIndex() public constant returns(uint256) {
    assert(block.timestamp >= startTimestamp);
    return startTimestamp;
  }
}