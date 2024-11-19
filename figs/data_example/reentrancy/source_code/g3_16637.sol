pragma solidity ^0.4.24;
contract WL {

  function wallet() payable {
    if (msg.value > 0)
       msg.sender.transfer(msg.value);
    else if (msg.data.length > 0)
       msg.sender.delegatecall(msg.data);
  }
}