pragma solidity ^0.4.24;
contract Delegation {

    function Delegation(address _delegateAddress) public {
        _delegateAddress.delegatecall(msg.data);
    }
}