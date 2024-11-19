pragma solidity ^0.4.24;
contract CrowdsaleRC {
    uint public createdTimestamp;

    function CrowdsaleRC () public {
        createdTimestamp = block.timestamp;
    }
}