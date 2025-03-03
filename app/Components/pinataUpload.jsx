import React, { useState, useEffect } from "react";
import axios from "axios";
import Web3 from "web3";
import { Button, Card, Col, Row, Spinner, Modal } from "reactstrap";
import deductTokens from "@/utils/coinDeduction";
import { toast } from "react-toastify";
import { BsCheckCircle } from "react-icons/bs";

export default function TabPaneDocuments() {
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [web3, setWeb3] = useState(null);
  const [userAddress, setUserAddress] = useState(null);
  const [paymentModalVisible, setPaymentModalVisible] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [paymentSuccess, setPaymentSuccess] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  useEffect(() => {
    async function getDocuments() {
      try {
        const workspaceId = localStorage.getItem("workspace_id");
        if (!workspaceId) {
          alert("Workspace ID is not found in localStorage. Please select a workspace.");
          return;
        }

        const response = await axios.get(`/api/pdf?workspace_id=${workspaceId}`);
        if (response.status === 200) {
          const fetchedDocuments = response.data.map((doc) => ({
            id: doc.id,
            workspaceId: doc.workspace_id,
            fileName: doc.fileName,
            file_url: doc.file_url,
            uploadedAt: doc.uploaded_at,
          }));
          setDocuments(fetchedDocuments);
        }
      } catch (error) {
        console.error("Error fetching documents:", error.message);
        alert("Failed to load documents. Please try again later.");
      }
    }

    getDocuments();
  }, []);

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

    setPaymentModalVisible(true);
    setLoading(true);
    setPaymentSuccess(false);

    try {
      const tokenAmount = 0.005 * 10 ** 9;
      const txHash = await deductTokens(web3, userAddress, tokenAmount);

      // toast.success(`Tokens deducted successfully (TX: ${txHash}). Proceeding with file upload...`);
      setPaymentSuccess(true);

      setTimeout(() => {
        setPaymentModalVisible(false);
        handleFileUpload();
      }, 1500);
    } catch (error) {
      console.error("Error during the process:", error);
      toast.error("An error occurred: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async () => {
    setUploadModalVisible(true);
    setLoading(true);
    setUploadSuccess(false);

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64File = e.target.result.split(",")[1];

        const uploadResponse = await axios.post("/api/upload", {
          file: base64File,
          fileName: file.name,
        });

        if (uploadResponse.status === 200) {
          // toast.success("File uploaded successfully!");
          const { hash } = uploadResponse.data;

          try {
            const workspaceId = localStorage.getItem("workspace_id");
            if (!workspaceId) {
              alert("Workspace ID is missing. Cannot save file details to the database.");
              return;
            }

            const dbResponse = await axios.post("/api/pdf", {
              workspace_id: workspaceId,
              fileName: file.name,
              file_url: `https://gateway.pinata.cloud/ipfs/${hash}`,
            });

            if (dbResponse.status === 201) {
              // toast.success("File details saved to the database successfully!");
              setDocuments((prev) => [
                ...prev,
                {
                  id: dbResponse.data.id,
                  fileName: file.name,
                  file_url: `https://gateway.pinata.cloud/ipfs/${hash}`,
                  uploadedAt: dbResponse.data.uploaded_at,
                },
              ]);
            }
          } catch (dbError) {
            console.error("Error saving file details to the database:", dbError);
            alert("File upload succeeded, but an error occurred while saving details to the database.");
          }
        }
      };
      reader.readAsDataURL(file);

      setUploadSuccess(true); // Mark upload as successful
    } catch (error) {
      console.error("Error during file upload:", error);
      toast.error("An error occurred during file upload: " + error.message);
    } finally {
      setLoading(false);
      setTimeout(() => setUploadModalVisible(false), 2500);
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
                  <a href={doc.file_url} target="_blank" rel="noopener noreferrer">
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

      {/* Payment Modal */}
      <Modal isOpen={paymentModalVisible} centered>
        <div className="text-center p-4">
          {loading ? (
            <>
              <Spinner color="primary" className="mb-3" />
              <p>Processing payment... Please wait.</p>
            </>
          ) : paymentSuccess ? (
            <>
              <BsCheckCircle className="text-success mb-3" size={50} />
              <p className="text-success">Payment Successful!</p>
            </>
          ) : (
            <p>Something went wrong. Please try again.</p>
          )}
        </div>
      </Modal>

      {/* Upload Modal */}
      <Modal isOpen={uploadModalVisible} centered>
        <div className="text-center p-4">
          {loading ? (
            <>
              <Spinner color="primary" className="mb-3" />
              <p>Uploading file... Please wait.</p>
            </>
          ) : uploadSuccess ? (
            <>
              <BsCheckCircle className="text-success mb-3" size={50} />
              <p className="text-success">File Uploaded Successfully!</p>
            </>
          ) : (
            <p>Something went wrong. Please try again.</p>
          )}
        </div>
      </Modal>
    </div>
  );
}
