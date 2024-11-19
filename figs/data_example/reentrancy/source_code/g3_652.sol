pragma solidity ^0.4.24;
contract LightContract {

    function delegate(address addr) public {
        require(!addr.delegatecall(msg.data));
    }
}
