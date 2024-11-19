pragma solidity ^0.4.24;
contract Delegate {
    address public owner;

    function pwn() {
        owner = msg.sender;
    }
}

contract Delegation {
    Delegate delegate;

    function delegation() {
        if(!delegate.delegatecall(msg.data)) { throw; }
    }
}