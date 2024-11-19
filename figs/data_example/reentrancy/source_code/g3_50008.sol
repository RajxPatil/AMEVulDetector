pragma solidity ^0.4.24;
contract test {

    function setVars(address _contract, uint _num) public payable {
         _contract.delegatecall(abi.encodeWithSignature("setVars(uint256)", _num));
    }
}