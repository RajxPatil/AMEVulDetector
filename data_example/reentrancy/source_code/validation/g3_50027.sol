pragma solidity ^0.4.24;
contract LightContract {
    address owner;

    function delegate(address addr) {
        require(addr == msg.sender);
        require(addr.delegatecall(msg.data));
    }
}
