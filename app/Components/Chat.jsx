import React, { useState, useEffect, useRef } from "react";
import Web3 from "web3";
import deductTokens from "@/utils/coinDeduction";
import { toast } from "react-toastify";
import { Modal, Spinner } from "react-bootstrap";
import { BsCheckCircle, BsUpload } from "react-icons/bs";
import { AiOutlineFileDone } from "react-icons/ai";
import { useCrm } from "../context/CrmContext";
import axios from "axios";
import { getDocument } from "pdfjs-dist";

export default function Chat({ onRegister }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [paymentSuccess, setPaymentSuccess] = useState(false);
  const [web3, setWeb3] = useState(null);
  const [userAddress, setUserAddress] = useState(null);
  const [paymentLoading, setPaymentLoading] = useState(false);
  const [fileUploading, setFileUploading] = useState(false);
  const [fileUploadDone, setFileUploadDone] = useState(false);
  const [realEstateAgent, setRealEstateAgent] = useState(false);
  // const { crmName, setCrmName } = useCrm();
  const [accessToken, setAccessToken] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [cToken, setCsrfToken] = useState("");

  const chatEndRef = useRef(null);
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

  useEffect(() => {
    async function fetchChatHistory() {
      const workspaceId = localStorage.getItem("workspace_id");
      if (!workspaceId) return;

      try {
        const response = await fetch("/api/chat", {
          method: "GET",
          headers: {
            workspace_id: workspaceId,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setMessages(data.chats || []);
        } else {
          console.error("Failed to fetch chat history");
        }
      } catch (error) {
        console.error("Error fetching chat history:", error);
      }
    }
    initializeWeb3();
    fetchChatHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (documentUrl) => {
    try {
      // Fetch the document from the provided URL
      const response = await fetch(documentUrl);
      const blob = await response.blob();

      // Read the blob as a Base64 string
      const reader = new FileReader();
      reader.onload = async (e) => {
        const base64File = e.target.result.split(",")[1]; // Extract base64 content

        // Step 1: Handle payment process
        setPaymentLoading(true); // Start payment loading
        setModalVisible(true); // Show modal for payment
        setPaymentSuccess(false); // Reset payment success state

        try {
          const tokenAmount = 0.001 * 10 ** 9;
          const txHash = await deductTokens(web3, userAddress, tokenAmount);

          toast.success(`Tokens deducted successfully. TX: ${txHash}`);
          setPaymentLoading(false); // Stop payment loading
          setPaymentSuccess(true); // Mark payment as successful

          // Wait a moment before transitioning to the file upload state
          setTimeout(() => {
            setPaymentSuccess(false); // Reset payment success
            setFileUploading(true); // Start file uploading
          }, 1000);
        } catch (paymentError) {
          setPaymentLoading(false);
          setPaymentSuccess(false);
          console.error("Error during payment:", paymentError);
          alert("Payment failed. Please try again.");
          return; // Stop further execution if payment fails
        }

        // Step 2: Handle file upload
        try {
          const uploadResponse = await axios.post("/api/upload", {
            file: base64File,
            fileName: "generated-document.doc",
          });

          if (uploadResponse.status === 200) {
            const { hash } = uploadResponse.data;

            const workspaceId = localStorage.getItem("workspace_id");
            if (!workspaceId) {
              alert(
                "Workspace ID is missing. Cannot save file details to the database."
              );
              return;
            }

            // Save the file URL and metadata to your database
            const dbResponse = await axios.post("/api/pdf", {
              workspace_id: workspaceId,
              fileName: "generated-document.doc",
              file_url: `https://gateway.pinata.cloud/ipfs/${hash}`,
            });

            if (dbResponse.status === 201) {
              setFileUploading(false); // Stop file uploading
              setFileUploadDone(true); // Mark file upload as done
              setTimeout(() => setModalVisible(false), 1000); // Close modal after a short delay
              alert("File successfully saved to Web3 Storage and database!");
            }
          }
        } catch (uploadError) {
          setFileUploading(false);
          console.error("Error during file upload:", uploadError);
          alert("An error occurred during file upload: " + uploadError.message);
        }
      };

      reader.readAsDataURL(blob);
    } catch (error) {
      console.error("Error fetching or processing file:", error);
      alert("An error occurred: " + error.message);
    }
  };

  const handleDownload = async (documentUrl) => {
    try {
      setPaymentLoading(true);
      setModalVisible(true);
      setPaymentSuccess(false);

      const tokenAmount = 0.001 * 10 ** 9;

      const txHash = await deductTokens(web3, userAddress, tokenAmount);

      toast.success(`Tokens deducted successfully. TX: ${txHash}`);
      setPaymentLoading(false);
      setPaymentSuccess(true);
      setTimeout(() => setModalVisible(false), 1000);

      const response = await fetch(documentUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.downloaddownload = "generated-document.doc";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (error) {
      console.error("Error during download:", error);
      toast.error(
        "An error occurred during the download process: " + error.message
      );
    } finally {
      setModalVisible(false);
    }
  };

  const handleModify = async (documentUrl) => {
    const userInput = prompt("Enter text to append:");
    if (!userInput) return;

    try {
      const pdf = await getDocument(documentUrl).promise;

      let extractedText = "";

      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
        const page = await pdf.getPage(pageNumber);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map((item) => item.str).join(" ");
        extractedText += ` ${pageText}`;
      }

      const combinedText = `${extractedText.trim()} ${userInput}`;

      await handleSend(null, combinedText);
    } catch (error) {
      console.error("Error modifying the PDF document:", error);
      toast.error("Failed to modify the PDF document. Please try again.");
    }
  };

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = async (crmName, modifyInput) => {
    let web3Instance;
    let userAddress;

    console.log("handleSend >>>>>>");

    if (window.ethereum) {
      web3Instance = new Web3(window.ethereum);
      try {
        await window.ethereum.request({ method: "eth_requestAccounts" });
        const accounts = await web3Instance.eth.getAccounts();
        setUserAddress(accounts[0]);
        userAddress = accounts[0];
        setWeb3(web3Instance);
        console.log("MetaMask detected. Using injected web3 provider.");
      } catch (error) {
        console.error("User denied account access:", error);
      }
    } else if (window.web3) {
      web3Instance = new Web3(window.web3.currentProvider);
      const accounts = await web3Instance.eth.getAccounts();
      setUserAddress(accounts[0]);
      userAddress = accounts[0];
      setWeb3(web3Instance);
    } else {
      const provider = new Web3.providers.HttpProvider("http://127.0.0.1:9545");
      web3Instance = new Web3(provider);
      setWeb3(web3Instance);
      console.warn("No MetaMask detected. Using local web3 provider.");
    }

    if (!crmName && !input.trim() && !modifyInput) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    // Show modal for payment processing
    // setModalVisible(true);
    setLoading(true);
    // setPaymentSuccess(false);
    // setPaymentLoading(true);

    try {
      const tokenAmount = 1 * 10 ** 9;
      console.log(web3Instance, userAddress, "aouyoo")
      const txHash = await deductTokens(web3Instance, userAddress, tokenAmount);

      toast.success(`Tokens deducted successfully. TX: ${txHash}`);
      setPaymentLoading(false);
      setPaymentSuccess(true);
      setTimeout(() => setModalVisible(false), 1000);
if (crmName === "Realestate Agent") {
  console.log("Connecting to Realestate Agent...");

  // Add the message for "Connecting to Realestate Agent"
  const connectingMessage = {
    role: "assistant",
    content: "Connecting to Realestate Agent...",
  };
  setMessages((prev) => [...prev, connectingMessage]);

  try {
    console.log("Requesting Realestate Agent token...");

    // Step 1: Get JWT token
    const response = await fetch(
      "https://zillow.fastpeer.ai/dashboard/proxy/token/",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: "Real estate Agent enabled. You can now ask questions.",
        }),
      }
    );


    if (!response.ok) {
      throw new Error("Failed to connect to Realestate Agent");
    }

    const data = await response.json();
    console.log(data, "data")
    const { access_token, session_id, CSRFToken } = data;


  
    console.log("Realestate Agent connected with token:", access_token);

    // Show success message
    const botMessage = {
      role: "assistant",
      content: "Real estate Agent enabled. You can now ask questions.",
    };
    setMessages((prev) => [...prev, botMessage]);

    // Save the access token
    setRealEstateAgent(true);
    setAccessToken(access_token);
    setCsrfToken(CSRFToken);


  } catch (error) {
    console.error("Error connecting to Realestate Agent:", error);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content:
          "Failed to connect to Realestate Agent. Please try again later.",
      },
    ]);
  }
} else if (realEstateAgent && input) {
  console.log("Sending query to Realestate Agent...");
  console.log("Access Token:", accessToken);

  try {
    const response = await fetch(
      "https://zillow.fastpeer.ai/dashboard/proxy/qa/",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },

        body: JSON.stringify({ query: input }),
      }
    );

    const data = await response.json();
    console.log("Response from Realestate Agent:", data);

    if (response.ok) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.result },
      ]);
    } else {
      throw new Error("Something went wrong with Realestate Agent.");
    }
  } catch (error) {
    console.error("Error fetching Realestate Agent response:", error);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content:
          "Sorry, something went wrong with the Realestate Agent. Please try again later.",
      },
    ]);
  }
} else if (crmName && crmName != "Realestate Agent") {
  console.log("hello crm", crmName);
  const botMessage = {
    role: "assistant",
    content: `${crmName} connected successfully.`,
  };
  setMessages((prev) => [...prev, botMessage]);
} else if (modifyInput || input) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      workspace_id: localStorage.getItem("workspace_id"),
    },
    body: JSON.stringify({ message: modifyInput ? modifyInput : input }),
  });
  if (response.headers.get("Content-Type").includes("application/pdf")) {
    console.log("pdf");
    // Handle PDF byte response
    const arrayBuffer = await response.arrayBuffer(); // Read response as arrayBuffer
    const blob = new Blob([arrayBuffer], { type: "application/pdf" }); // Convert to PDF Blob
    const url = URL.createObjectURL(blob); // Create a URL for the Blob

    const botMessage = {
      role: "assistant",
      content: "Your PDF document has been generated. Click below to download:",
      documentUrl: url, // URL for PDF download
    };
    setMessages((prev) => [...prev, botMessage]);
  } else {
    const data = await response.json();

    if (response.ok) {
      const botMessage = { role: "assistant", content: data.reply };
      setMessages((prev) => [...prev, botMessage]);
    } else {
      const errorMessage = {
        role: "assistant",
        content: "Sorry, something went wrong. Please try again later.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  }
}
    } catch (error) {
      toast.error("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (onRegister) {
      onRegister(() => handleSend);
    }
  }, [onRegister]);

  return (
    <div>
      <ul className="list-unstyled chat-details">
        {messages.map((msg, index) => (
          <li
            key={index}
            className={`d-flex justify-content-${
              msg.role === "user" ? "end" : "start"
            } mb-4`}
          >
            {msg.role === "assistant" && (
              <img
                src="https://zillow.fastpeer.ai/static/images/image_male.jpg"
                alt="avatar"
                className="rounded-circle d-flex align-self-start me-3 shadow-1-strong"
                width="40"
              />
            )}
            <div
              className="card"
              style={{
                backgroundColor:
                  msg.role === "user" ? "#6C757D" : "transparent",
                color: msg.role === "user" ? "#ffffff" : "#ffffff",
                border: "none",
                borderRadius: "20px", // More rounded corners
                padding: "1px 1px", // Reduced padding
                maxWidth: "75%", // Prevents messages from stretching too wide
              }}
            >
              <div className="card-body">
                <p className="mb-0">{msg.content}</p>
                {msg.documentUrl && (
                  <div className="d-flex mt-2">
                    {/* Download Button */}
                    <button
                      className="btn btn-primary ms-3"
                      onClick={() => {
                        window.open(msg.documentUrl);
                      }}
                    >
                      Show Document
                    </button>
                    <button
                      className="btn btn-primary ms-3"
                      onClick={() => handleDownload(msg.documentUrl)}
                    >
                      Download Document
                    </button>

                    {/* Save to Web3 Storage Button */}
                    <button
                      className="btn btn-primary ms-3"
                      onClick={() => handleFileUpload(msg.documentUrl)}
                    >
                      Save to Web3 Storage
                    </button>

                    <button
                      className="btn btn-primary ms-3"
                      onClick={() => handleModify(msg.documentUrl)}
                    >
                      Modify
                    </button>
                  </div>
                )}
              </div>
            </div>
            {msg.role === "user"}
          </li>
        ))}

        <div ref={chatEndRef}></div>

        {loading && (
          <li className="d-flex justify-content-start mb-4">
            <div className="card">
              <div className="card-body">
                <p className="mb-0">
                  <span className="typing-bubble"></span>
                  <span className="typing-bubble"></span>
                  <span className="typing-bubble"></span>
                </p>
              </div>
            </div>
          </li>
        )}

        <li className="mb-3">
          <div className="form-outline position-relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              className="form-control bg-primary-light rounded-5"
              placeholder="Type a message..."
            />
            <div className="right-12 float-end position-absolute d-flex align-items-center">
              <button type="button" className="btn p-0 px-1">
                <i className="bi bi-mic"></i>
              </button>

              <label className="btn p-0 px-1">
                <i className="bi bi-paperclip"></i>
                <input type="file" accept=".pdf" style={{ display: "none" }} />
              </label>

              <button
                type="button"
                onClick={handleSend}
                className="btn p-0 px-1"
              >
                <i className="bi bi-send"></i>
              </button>
            </div>
          </div>
        </li>
      </ul>

      {/* Modal */}
      <Modal show={modalVisible} centered>
        <Modal.Body className="text-center">
          {paymentLoading ? (
            <>
              <Spinner animation="border" role="status" className="mb-3">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
              <p>Processing payment... Please wait.</p>
            </>
          ) : paymentSuccess ? (
            <>
              <BsCheckCircle className="text-success mb-3" size={50} />
              <p className="text-success">Payment Successful!</p>
            </>
          ) : fileUploading ? (
            <>
              <Spinner animation="border" role="status" className="mb-3">
                <span className="visually-hidden">Uploading...</span>
              </Spinner>
              <p>Uploading file... Please wait.</p>
            </>
          ) : fileUploadDone ? (
            <>
              <AiOutlineFileDone className="text-primary mb-3" size={50} />
              <p className="text-primary">File uploaded successfully!</p>
            </>
          ) : (
            <p>Something went wrong.</p>
          )}
        </Modal.Body>
      </Modal>

      <style jsx>{`
        .typing-bubble {
          width: 12px;
          height: 12px;
          margin-right: 6px;
          border-radius: 50%;
          background-color: #bbb;
          display: inline-block;
          animation: typing 1.5s infinite ease-in-out;
        }

        .typing-bubble:nth-child(1) {
          animation-delay: 0s;
        }

        .typing-bubble:nth-child(2) {
          animation-delay: 0.3s;
        }

        .typing-bubble:nth-child(3) {
          animation-delay: 0.6s;
        }

        @keyframes typing {
          0% {
            opacity: 0.1;
            transform: scale(0.7);
          }
          50% {
            opacity: 1;
            transform: scale(1);
          }
          100% {
            opacity: 0.1;
            transform: scale(0.7);
          }
        }
      `}</style>
    </div>
  );
}
