pragma solidity ^0.4.24;
contract CardsRaffle {
  uint256 private raffleTicketsBought;
  uint256 private raffleTicketThatWon;

  function drawRandomWinner() public returns (uint256) {
    uint256 seed = raffleTicketsBought + block.timestamp;
    raffleTicketThatWon = addmod(uint256(block.blockhash(block.number-1)), seed, raffleTicketsBought);
    return raffleTicketThatWon;
  }
}