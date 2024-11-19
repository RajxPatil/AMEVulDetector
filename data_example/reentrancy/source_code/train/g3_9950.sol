pragma solidity ^0.4.24;
contract AuthenticatedProxy {
    enum HowToCall { DelegateCall }

    function proxy(address dest, HowToCall howToCall, bytes calldata) public returns (bool result) {
        if (howToCall == HowToCall.DelegateCall) {
            result = dest.delegatecall(calldata);
        }
        return result;
    }
}