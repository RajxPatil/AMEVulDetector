pragma solidity ^0.4.24;
contract BlocklancerToken {
    uint public fundingStart;

    function TimeLeftBeforeCrowdsale() external constant returns (uint256) {
        if(fundingStart > block.timestamp)
            return fundingStart;
        else
            return 0;
    }
}