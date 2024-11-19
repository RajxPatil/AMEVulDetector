pragma solidity ^0.4.24;
contract CrowdsaleExt {
    uint public startsAt;
    enum State {PreFunding, Failure}

    function getState() public constant returns (State) {
        if (block.timestamp < startsAt) return State.PreFunding;
        else return State.Failure;
    }
}