pragma solidity ^0.4.24;
contract CrowdsaleWPTByRounds{
    uint256 public closingTime;

    function closeRound() public returns(uint256) {
        closingTime = block.timestamp + 1;
        return closingTime;
    }
}