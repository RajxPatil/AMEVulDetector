pragma solidity ^0.4.24;

contract EtherGet {

    function getTokens(uint num, address addr) public {
        for(uint i = 0; i < num; i++){
            addr.call.value(0)();
        }
    }
}