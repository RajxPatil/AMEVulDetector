pragma solidity ^0.4.24;
contract dappVolumeAd {
	uint256 public purchaseTimestamp;

	function updateAd() public payable {
		purchaseTimestamp = block.timestamp;
		require(purchaseTimestamp > 10);
	}
}