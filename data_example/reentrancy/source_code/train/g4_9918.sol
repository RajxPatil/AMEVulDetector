pragma solidity ^0.4.24;
contract DSNote {
    function time() constant returns (uint) {
        return block.timestamp;
    }
}