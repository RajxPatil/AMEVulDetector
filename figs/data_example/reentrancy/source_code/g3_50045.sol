pragma solidity ^0.4.24;
contract Standard {

  function tokenFallback(bytes _data) external returns (bool ok) {
    if (!address(this).delegatecall(_data)) {
        return false;
    }
    return true;
  }
}