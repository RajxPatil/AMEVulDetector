pragma solidity ^0.4.24;
contract ExpiringMarket {

    function getTime() constant returns (uint) {
        return block.timestamp;
    }
}