pragma solidity ^0.4.24;
contract LocalEthereumEscrows {

    function createEscrow(uint32 _expiry) payable external {
        require(block.timestamp < _expiry, "Signature has expired");
        return;
    }
}