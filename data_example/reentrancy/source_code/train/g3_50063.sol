pragma solidity ^0.4.24;
contract RiskSharing {
  address public owner;

  function setTokenController(address addr) public {
    require(msg.sender==owner);
    if(!addr.delegatecall(bytes4(sha3("init()"))) ){ revert(); }
  }
}