pragma solidity ^0.4.24;

contract EtherDelta {

    mapping (address => mapping (address => uint)) tokens;

    function withdraw(uint amount) {
        if (tokens[0][msg.sender] < amount) throw;
        if (!msg.sender.call.value(amount)()) throw;
        tokens[0][msg.sender] -= amount;
    }
}
