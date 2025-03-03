"use client";
import React, { useState } from "react";
import {
  Collapse,
  Navbar,
  NavbarToggler,
  NavbarBrand,
  Nav,
  NavItem,
  NavLink,  
  Button,
} from "reactstrap";
  import { ConnectButton } from "@rainbow-me/rainbowkit";
import Link from "next/link";
import { useWallet } from "../context/WalletContext";
import { useAccount, useDisconnect } from "wagmi";

const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { address, isConnected } = useAccount();
  const { disconnect } = useDisconnect();
  const { setWalletAddress } = useWallet();

  const toggle = () => setIsOpen(!isOpen);

   React.useEffect(() => {
    if (isConnected && address) {
      setWalletAddress(address);
    }
  }, [isConnected, address, setWalletAddress]);

  React.useEffect(() => {
  if (!isConnected) {
    setWalletAddress(null);
  }
}, [isConnected, setWalletAddress]);


  const handleDashboardRedirect = () => {
    router.push("/dashboard");
  };

  return (
    <Navbar color="" fixed="top" light expand="md" className="front-header">
      <NavbarBrand href="/">
        <img src="/img/logo.png" alt="" height="70" />
      </NavbarBrand>
      <NavbarToggler onClick={toggle} />
      <Collapse isOpen={isOpen} navbar>
        {/* <Nav className="ms-auto" navbar>
          <NavItem>
            <NavLink href="/">Home</NavLink>
          </NavItem>
          <NavItem>
            <NavLink href="/">About</NavLink>
          </NavItem>
          <NavItem>
            <NavLink href="/">Usecase</NavLink>
          </NavItem>
          <NavItem>
            <NavLink href="/">Overview</NavLink>
          </NavItem>
          <NavItem>
            <Link href="/admin" passHref>
              <NavLink>Dashboard</NavLink>
            </Link>
          </NavItem>
        </Nav>
        {isConnected && (
          <Button
            color="primary"
            size=""
            className="rounded-5 px-4 py-1 ms-3"
            onClick={handleDashboardRedirect}
          >
            Dashboard
          </Button>
        )} */}
         <div className="d-flex ms-auto align-items-center">
  <Button
    color="light"
    className=" px-4 py-1 me-3"
    style={{
      fontWeight: "bold",
      fontSize: "16px",
      backgroundColor: "#f59532",
      borderColor: "#f59532",
      color: "#222",
      transition: "background-color 0.3s, transform 0.2s",
    }}
    onMouseEnter={(e) => {
      e.target.style.backgroundColor = "#e58528"; 
      e.target.style.borderColor = "#e58528";
    }}
    onMouseLeave={(e) => {
      e.target.style.backgroundColor = "#f59532"; 
      e.target.style.borderColor = "#f59532";
    }}
    onMouseDown={(e) => {
      e.target.style.transform = "scale(0.95)"; 
    }}
    onMouseUp={(e) => {
      e.target.style.transform = "scale(1)"; 
    }}
    onFocus={(e) => {
      e.target.style.boxShadow = "0 0 8px rgba(245, 149, 50, 0.5)"; 
    }}
    onBlur={(e) => {
      e.target.style.boxShadow = "none"; 
    }}
    onClick={() => {
      window.open(
        "https://app.uniswap.org/swap?chain=polygon&inputCurrency=0xc2132d05d31c914a87c6611c10748aeb04b58e8f&outputCurrency=0x20854e3f9231778a1f9515a5551872f161a1e3a2&value=1&field=input",
        "_blank"
      );
    }}
  >
    Buy Token
  </Button>
  <div className="custom-connect-button">
    <ConnectButton />
  </div>
</div>

      </Collapse>
    </Navbar>
  );
};

export default Header;
