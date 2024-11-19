pragma solidity ^0.4.24;
contract Upgradeable {

    function replace(address target) external {
        require(target.delegatecall(bytes4(keccak256("initialize()"))));
    }
}