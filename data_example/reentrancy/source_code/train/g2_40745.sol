pragma solidity ^0.4.24;

contract SimpleDAO {

    mapping (address => uint) public credit;

    function withdraw(uint amount) {
      msg.sender.call.value(amount)();
      credit[msg.sender] -= amount;
    }
}