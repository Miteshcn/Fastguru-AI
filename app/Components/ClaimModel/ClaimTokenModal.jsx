import React, { useEffect, useState } from "react";
import Modal from "react-modal";
import "./ModalStyles.css"; 
import { ConnectButton } from "@rainbow-me/rainbowkit";

const ClaimTokenModel = () => {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    setIsOpen(true);
  }, []);

   const closeModal = () => {
    setIsOpen(false); 
  };

  return (
    <Modal
      isOpen={isOpen}
      onRequestClose={closeModal}
      contentLabel="Ad Modal"
      className="modal-container"
      overlayClassName="modal-overlay"
    >
      <div className="modal-content">
        <h2>🎉 Claim Your 1000 FTC Coins! 🎉</h2>
        <p>
          Welcome to Web3Space! As a new user, you're eligible to claim free FTC Coins. 
          Connect your wallet and start exploring the decentralized future!
        </p>
<div className="connect-button-container">
          <ConnectButton />
        </div>
         <button className="close-button" onClick={closeModal}>
          Close
        </button>
      </div>
    </Modal>
  );
};

export default ClaimTokenModel;
