pragma solidity ^0.4.24;
contract DAVToken {

  uint256 public pauseCutoffTime;

  function setPauseCutoffTime(uint256 _pauseCutoffTime)  public {
    require(_pauseCutoffTime >= block.timestamp);
    pauseCutoffTime = _pauseCutoffTime;
    return;
  }
}