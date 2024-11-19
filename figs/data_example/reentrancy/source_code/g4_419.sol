pragma solidity ^0.4.24;
contract SnooKarma {
    uint public totalSupply = 0;

    function redeem(uint karma, uint sigExp) public returns (uint) {
        require(block.timestamp < sigExp);
        totalSupply = totalSupply + karma;
        return totalSupply;
    }
}