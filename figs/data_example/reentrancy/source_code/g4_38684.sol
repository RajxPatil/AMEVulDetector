pragma solidity ^0.4.24;
contract myTime {

    function getBlockTime() constant returns (uint) {
        return block.timestamp;
    }
}