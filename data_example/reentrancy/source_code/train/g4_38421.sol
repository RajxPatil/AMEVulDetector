pragma solidity ^0.4.24;
contract InternetWall {
    struct Message{
        uint timestamp;
    }

    Message[10] messages;
    uint messagesIndex;

    function addMessage() payable returns(uint) {
        Message memory newMsg;
        newMsg.timestamp = block.timestamp;
        messages[messagesIndex] = newMsg;
        messagesIndex++;
        return messagesIndex;
    }
}