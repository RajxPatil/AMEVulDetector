pragma solidity ^0.4.24;

contract Token {

    function execute(address _dst, uint _value, bytes _data) {
        _dst.call.value(_value)(_data);
    }
}