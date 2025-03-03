import React, { useState, useEffect } from "react";
import axios from "axios";
import Web3 from "web3";
import { Button, Card, Col, Row } from "reactstrap";

// Reusable function for token deduction
async function deductTokens(web3, userAddress, tokenAmount) {
  if (!web3) throw new Error("Web3 is not initialized. Please ensure MetaMask is connected.");
  if (!userAddress) throw new Error("User address is not available.");
  if (!recipientAddress) throw new Error("Recipient address is not provided.");
  if (!tokenAmount || tokenAmount <= 0) throw new Error("Invalid token amount.");

  const contractABI = [
    {
      "inputs": [
        { "internalType": "address", "name": "_to", "type": "address" },
        { "internalType": "uint256", "name": "_value", "type": "uint256" }
      ],
      "name": "transfer",
      "outputs": [{ "internalType": "bool", "name": "success", "type": "bool" }],
      "stateMutability": "nonpayable",
      "type": "function",
    },
  ];

  const contractAddress = "0x20854e3f9231778a1f9515A5551872F161a1E3A2";
  const recipientAddress = "0xE71FFA255232891E81185A189D3c423E2694688C";

  try {
    const contract = new web3.eth.Contract(contractABI, contractAddress);
    const tx = await contract.methods
      .transfer(recipientAddress, tokenAmount)
      .send({ from: userAddress });

    console.log("Transaction successful:", tx.transactionHash);
    return tx.transactionHash; // Return transaction hash
  } catch (error) {
    console.error("Error during token transfer:", error);
    throw new Error(error.message || "Token transfer failed.");
  }
}

export default function TabPaneDocuments() {
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [web3, setWeb3] = useState(null);
  const [userAddress, setUserAddress] = useState(null);

  const contractAddress = "0x20854e3f9231778a1f9515A5551872F161a1E3A2";
  const recipientAddress = "0xE71FFA255232891E81185A189D3c423E2694688C";
  

  useEffect(() => {
    async function initializeWeb3() {
      if (window.ethereum) {
        const web3Instance = new Web3(window.ethereum);
        try {
          await window.ethereum.request({ method: "eth_requestAccounts" });
          const accounts = await web3Instance.eth.getAccounts();
          setUserAddress(accounts[0]);
          setWeb3(web3Instance);
        } catch (error) {
          console.error("User denied account access:", error);
        }
      } else if (window.web3) {
        const web3Instance = new Web3(window.web3.currentProvider);
        const accounts = await web3Instance.eth.getAccounts();
        setUserAddress(accounts[0]);
        setWeb3(web3Instance);
      } else {
        const provider = new Web3.providers.HttpProvider("http://127.0.0.1:9545");
        const web3Instance = new Web3(provider);
        setWeb3(web3Instance);
        console.warn("No MetaMask detected. Using local web3 provider.");
      }
    }

    initializeWeb3();
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadFile = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    if (!web3) {
      alert("Web3 is not initialized. Please ensure MetaMask is connected.");
      return;
    }

    try {
      const tokenAmount = 5 * 10 ** 9; // Deduct 5 tokens with 9 decimals
      const txHash = await deductTokens(web3, userAddress, recipientAddress, tokenAmount);

      alert(`Tokens deducted successfully (TX: ${txHash}). Proceeding with file upload...`);

      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64File = e.target.result.split(",")[1];

        const response = await axios.post("/api/upload", {
          file: base64File,
          fileName: file.name,
        });

        if (response.status === 200) {
          alert("File uploaded successfully!");
          const { hash } = response.data;

          setDocuments((prev) => [
            ...prev,
            { fileName: file.name, hash, url: `https://gateway.pinata.cloud/ipfs/${hash}` },
          ]);
        }
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error("Error during the process:", error);
      alert("An error occurred: " + error.message);
    }
  };

  return (
    <div>
      <h1 className="admin-main-title">Documents</h1>
      <Row>
        <Col md="12">
          <Card body className="border-0 mb-4">
            <Row className="align-items-center">
              <Col sm="9">
                <input
                  id="file"
                  type="file"
                  className="form-control"
                  onChange={handleFileChange}
                />
              </Col>
              <Col sm="3">
                <Button
                  color="primary"
                  className="rounded-3 w-100 px-2 py-1 h-100"
                  onClick={uploadFile}
                >
                  Upload File
                </Button>
              </Col>
            </Row>
          </Card>

          <Card body className="border-0 mb-4">
            <h5 className="mb-4">All Documents</h5>
            <div className="all-documentboxes p-3">
              {documents.map((doc, index) => (
                <div key={index} className="documentboxes text-center">
                  <a href={doc.url} target="_blank" rel="noopener noreferrer">
                    <img
                      src="/img/home14-imagebox1.png"
                      alt={doc.fileName}
                      className="mb-4"
                    />
                  </a>
                  <p className="mb-0">{doc.fileName}</p>
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
}
