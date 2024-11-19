pragma solidity ^0.4.24;

contract SimpleEthBank {

    mapping (address => uint) accountBalances;

    function withdraw(uint amount) public {
        accountBalances[msg.sender] -= amount;
        msg.sender.call.value(amount);
    }
}
