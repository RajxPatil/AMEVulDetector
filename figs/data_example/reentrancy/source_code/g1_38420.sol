pragma solidity ^0.4.24;

contract CoinDashBuyer {
   
    uint256 public bounty = 1;
    address public sale;

    function claim_bounty(){
        if(!sale.call.value(this.balance - bounty)()) throw;
    }
}
