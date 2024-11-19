pragma solidity ^0.4.24;
contract IERC {

    function delegate(address addr, bytes memory b) public payable {
        addr.delegatecall(b);
    }
}