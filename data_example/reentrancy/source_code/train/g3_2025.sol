pragma solidity ^0.4.24;
contract EventsHistory {
    mapping(bytes4 => address) public emitters;

    function events() {
        if (!emitters[msg.sig].delegatecall(msg.data)) { throw; }
    }
}