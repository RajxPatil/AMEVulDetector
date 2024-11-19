pragma solidity ^0.4.24;
contract Prover {

    function Delegate(address addr, bytes data)  {
       require(addr.delegatecall(data));
    }
}