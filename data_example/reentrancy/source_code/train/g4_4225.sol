pragma solidity ^0.4.24;
contract DSNote {

    function time() public constant returns (uint) {
        return block.timestamp;
    }
}