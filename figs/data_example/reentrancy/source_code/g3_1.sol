pragma solidity ^0.4.24;
contract TeamToken  {

    uint64 public gameTime;

    function test() payable public {
        if (gameTime > 1514764800) {
            require(gameTime > block.timestamp);
        }
    }
}