pragma solidity ^0.4.24;
contract RiskSharingToken {

  function setTokenController(address addr) public {
    if( !addr.delegatecall(bytes4(sha3("init()"))) ){ revert(); }
  }
}