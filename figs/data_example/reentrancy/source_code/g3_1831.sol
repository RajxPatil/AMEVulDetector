pragma solidity ^0.4.24;
contract Mokens{

    function moke(address addr) external {
        bytes memory data = msg.data;
        addr.delegatecall(data);
    }
}