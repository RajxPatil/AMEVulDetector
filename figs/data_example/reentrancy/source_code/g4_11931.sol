pragma solidity ^0.4.24;
contract Fluzcoin {
    uint public launched = 0;  

    function launch() public returns(uint){
        launched = block.timestamp;
        return launched;
    }
}