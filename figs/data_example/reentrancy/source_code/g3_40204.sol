pragma solidity ^0.4.24;
contract Safe {

    function requireStackDepth(address addr, bytes32 data) {
        if (data=='') { throw; }
        if (!addr.delegatecall(data)){ throw; }
    }
}