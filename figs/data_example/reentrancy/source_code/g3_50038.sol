pragma solidity ^0.4.24;
contract Controller {

    function sweep(address _token, uint _amount) public returns (bool) {
        return _token.delegatecall(_amount);
    }
}