pragma solidity ^0.4.24;

contract PoormansHoneyPot {

    mapping (address => uint) public balances;

    function withdraw() public{
        assert(msg.sender.call.value(balances[msg.sender])()) ;
        balances[msg.sender] = 0;
    }
}