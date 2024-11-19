pragma solidity ^0.4.24;
contract TMTG {
    uint256 public openingTime;

    address public owner;
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    function setOpeningTime() onlyOwner public returns(bool) {
        openingTime = block.timestamp;
        return false;
    }
}