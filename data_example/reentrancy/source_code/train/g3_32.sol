pragma solidity ^0.4.24;
contract SponsoredItemGooRaffle {
    uint256 private raffleEndTime;

    function drawRandomWinner() public {
        require(raffleEndTime < block.timestamp);
    }
}