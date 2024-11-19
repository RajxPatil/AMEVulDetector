pragma solidity ^0.4.24;
contract Proxy {
   address public owner;

   function delegate(address addr, bytes32 data) {
       require(owner == msg.sender);
       addr.delegatecall(data);
   }
}
