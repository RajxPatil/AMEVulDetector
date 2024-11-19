pragma solidity ^0.4.24;
contract CallsTestContract {

    address public smart;

    function set(address _addr) public {
        smart = _addr;
    }

    function callTheOtherContract() public {
        smart.delegatecall(bytes4(keccak256("callMeMaybe()")));
    }
}
