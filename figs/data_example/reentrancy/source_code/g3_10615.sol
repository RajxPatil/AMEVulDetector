pragma solidity ^0.4.24;
contract OwnedUpgradeabilityProxy {

    function upgradeToAndCall(address implementation, bytes data) payable public {
        require(implementation.delegatecall(data));
    }
}
