pragma solidity ^0.4.24;
contract wbcSale {
    function blockTime() public view returns (uint32) {
        return uint32(block.timestamp);
    }
}