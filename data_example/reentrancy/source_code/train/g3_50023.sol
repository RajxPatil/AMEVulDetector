pragma solidity ^0.4.24;
contract D {

  function delegatecallSetN(address _e, uint _n) {
      _e.delegatecall(bytes4(keccak256("setN(uint256)")), _n);
  }
}