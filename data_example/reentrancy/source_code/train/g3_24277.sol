pragma solidity ^0.4.24;
contract Compare {
    address public testaddress;

    function withdelegatecall(address _testaddr) public {
        testaddress = _testaddr;
        testaddress.delegatecall(bytes4(keccak256("test()")));
    }
}