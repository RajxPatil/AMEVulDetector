pragma solidity ^0.4.24;
contract KuaiMintableToken {
    uint256 public lastMintTime = 0;
    uint256 public createTime;

    function KuaiMintableToken() public {
        createTime = block.timestamp;
        lastMintTime = createTime;
        return;
    }
}