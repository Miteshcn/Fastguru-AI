import React, { useState } from "react";
import { Collapse, Button, CardBody, Card } from "reactstrap";
import CrmPopup from "./CrmPopup/CrmPopup";
import { useRealestate } from "@/app/context/RealestateContext";
import { useActiveTab } from "../context/ActiveTab";
function EnterprisesModules({ args, handleSend }) {
  const [isOpen, setIsOpen] = useState(false);
  const [showPopup, setShowPopup] = useState(false);
  const { setRealestate } = useRealestate(); // Use the setRealestate function to update state
  const { setActiveTab } = useActiveTab(); // Use the setActiveTab function to update state
  const handleButtonClick = () => {
    setShowPopup(true);
  };

  const handleRealestateClick = () => {
    setRealestate(true); // Set realestate state to true when clicked
    handleSend("Realestate Agent"); // Call handleSend with the message
    setActiveTab("2");
  };

  const handleClosePopup = () => {
    setShowPopup(false);
  };

  const toggle = () => setIsOpen(!isOpen);

  return (
    <>
      <div className="dropdown-container">
        <Button
          className="mb-0 w-100 text-start d-flex justify-content-start align-items-center"
          color="secondary"
          onClick={toggle}
          style={{
            marginBottom: "1rem",
            backgroundColor: "transparent", // Remove background
            color: "#ffffff", // White text
            borderRadius: "10px",
            border: "none", // Remove border
          }}
        >
          <span>{isOpen ? "▲" : "▼"}</span> {/* Chevron on the left */}
          <span style={{ marginLeft: "10px" }}>Enterprise Agent</span>
        </Button>

        <Collapse isOpen={isOpen} {...args}>
          <Card>
            <CardBody className="bg-dark flex">
              <button
                type="button"
                className="btn custom-btn-no-hover me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#contentCreatorModal"
              >
                Knowledgebase Agent
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#contentCreatorModal"
              >
                Report Agent
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#contentCreatorModal"
              >
                Willow Agent
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#contentCreatorModal"
                onClick={handleRealestateClick}
              >
                Realestate Agent
              </button>
              <button
                type="button"
                className="btn custom-btn-no-hover me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#contentCreatorModal"
              >
                ManageDoc Agent
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#contentCreatorModal"
              >
                Legal Contracts
              </button>
              <div>
                <button
                  className="btn me-1 mb-2 btn-sm fs-10"
                  id="connect-crm-btn-business"
                  onClick={handleButtonClick}
                >
                  Connect to CRM
                </button>
                {showPopup && (
                  <CrmPopup
                    handleSend={handleSend}
                    onClose={handleClosePopup}
                  />
                )}
              </div>
              <a
                href="/dashboard/workspace-supported-platform/"
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="connect-zoho-business"
              >
                Connect to Zoho
              </a>
              {/* <a
                href="/dashboard/workspace-supported-platform/"
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="connect-salesforce-business"
              >
                Connect to Salesforce
              </a> */}
              {/* <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="connect-lms-learning"
                onClick={() => showConnectingMessage("lms")}
              >
                Connect to LMS
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="digital-document"
                onClick={() => showConnectingMessage("digital_document")}
              >
                Connect to Digital Document
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="scanned-document"
                onClick={() => showConnectingMessage("scanned_document")}
              >
                Scanned Documents
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#LegalContractModal"
              >
                Legal Contract Creation
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="legal-contract"
                data-bs-toggle="modal"
                data-bs-target="#LegalContractModal"
              >
                Legal Contract Review
              </button> */}
            </CardBody>
          </Card>
        </Collapse>
      </div>
    </>
  );
}

export default EnterprisesModules;
