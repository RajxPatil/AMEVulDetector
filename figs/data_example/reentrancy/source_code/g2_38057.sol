pragma solidity ^0.4.24;

contract TokenStore {

    mapping (address => mapping (address => uint)) public tokens;

    function withdraw(uint _amount) {
        tokens[0][msg.sender] = tokens[0][msg.sender] - _amount;
        if (!msg.sender.call.value(_amount)()) { revert(); }
    }
}
