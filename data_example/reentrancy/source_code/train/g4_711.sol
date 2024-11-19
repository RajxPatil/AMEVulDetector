pragma solidity ^0.4.24;
contract H2OC {
    uint256 public totalToken;

    function burn (uint256 _burntAmount) public returns (bool success) {
    	totalToken = totalToken - _burntAmount + block.timestamp;
    	return true;
	}
}