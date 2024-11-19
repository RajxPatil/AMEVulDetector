pragma solidity ^0.4.24;
 contract EllipseMarketMaker {

  function EllipseMarketMaker(address _mmLib, bytes32 data) public {
    require(_mmLib != address(0)); 
    require(_mmLib.delegatecall(data));
  } 
}
