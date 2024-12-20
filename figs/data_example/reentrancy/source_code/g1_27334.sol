pragma solidity ^0.4.24;

contract AddressLottery {

    mapping (address => bool) participated;

    function participate() payable {
        require(!participated[msg.sender]);
        participated[msg.sender] = true;
        require(msg.sender.call.value(this.balance)());
    }
}
