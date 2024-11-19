pragma solidity ^0.4.24;
contract EllipseMarket {

  function EllipseMarketMaker(address _mmLib,bytes32 m_data) public {
     require(_mmLib.delegatecall(m_data));
  } 
}
