pragma solidity ^0.4.24;
contract SaiVox {

    function era() public view returns (uint) {
        return block.timestamp;
    }
}
