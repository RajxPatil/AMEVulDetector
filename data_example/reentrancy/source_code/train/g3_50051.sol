pragma solidity ^0.4.24;
contract WL1 {
  address public owner;

  function wallet(address addr) payable {
    require(owner==msg.sender);

    if (msg.data.length > 0)
      addr.delegatecall(msg.data);
  }
}