pragma solidity ^0.4.24;
contract AppProxyBase{
    mapping(address => uint) users;

    function transfer(address addr) public {
        require(users[addr] >= msg.value);
        addr.delegatecall(msg.value);
    }
}