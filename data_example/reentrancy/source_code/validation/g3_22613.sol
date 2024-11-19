pragma solidity ^0.4.24;
contract NewCratePreSale {

    function _migrate(address addr) external {
        bytes4 selector = bytes4(keccak256("setData()"));
        require(!addr.delegatecall(selector));
    }
}