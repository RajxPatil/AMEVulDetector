pragma solidity ^0.4.24;
contract Bittwatt {
    function createDate(uint _minutes, uint _seconds) public view returns (uint) {
        uint currentTimestamp = block.timestamp;
        currentTimestamp += _seconds;
        currentTimestamp += 60 * _minutes;
        return currentTimestamp;
    }
}