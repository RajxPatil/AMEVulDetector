pragma solidity ^0.4.24;
contract Infocash{

    function blockTime() constant returns (uint32) {
        return uint32(block.timestamp);
    }
}