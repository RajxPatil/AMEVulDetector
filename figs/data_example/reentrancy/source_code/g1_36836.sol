pragma solidity ^0.4.24;

contract EtherShot{

    mapping (address => uint) playerPendingWithdrawals;

    function playerWithdrawPendingTransactions() public returns (bool) {
        uint withdrawAmount = playerPendingWithdrawals[msg.sender];
        playerPendingWithdrawals[msg.sender] = 0;

        if (msg.sender.call.value(withdrawAmount)()) {
            return true;
        }
        else {
            playerPendingWithdrawals[msg.sender] = withdrawAmount;
            return false;
        }
    }
}
