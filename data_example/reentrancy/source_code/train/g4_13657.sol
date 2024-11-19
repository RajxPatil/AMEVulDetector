pragma solidity ^0.4.24;
contract NeuroChainClausius {
  event FreezeStatusChanged(bool toStatus, uint timestamp);
  bool public tradingLive = false;

  function setTradingStatus(bool isLive) public {
    tradingLive = isLive;
    FreezeStatusChanged(tradingLive, block.timestamp);
    return;
  }
}