import React, { useState } from "react";
import { Collapse, Button, CardBody, Card } from "reactstrap";

function DocumentsAndFiles(args) {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <>
      <Button
        className="mb-0 w-100 d-flex justify-content-start align-items-center"
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
        <span style={{ marginLeft: "10px" }}>Documents & Files</span>
      </Button>

      <Collapse isOpen={isOpen} {...args}>
        <Card>
          <CardBody className="bg-dark flex">
            <div className="card-body">
              <div className="col-lg-12 mb-4 mb-lg-0">
                <div className="document bg-danger">
                  <div className="document-body">
                    <i className="fas fa-file-pdf text-danger"></i>
                  </div>
                  <div className="document-footer">
                    <span className="document-name">PDF file 2017</span>
                    <span className="document-description">5.3 MB</span>
                  </div>
                </div>
              </div>
              <div className="col-lg-12 mb-4 mb-lg-0">
                <div className="document bg-success">
                  <div className="document-body">
                    <i className="fas fa-file-excel text-success"></i>
                  </div>
                  <div className="document-footer">
                    <span className="document-name">Excel file 2017</span>
                    <span className="document-description">2.7 MB</span>
                  </div>
                </div>
              </div>
              <div className="col-lg-12 mb-4 mb-lg-0">
                <div className="document bg-primary">
                  <div className="document-body">
                    <i className="fas fa-file-word text-primary"></i>
                  </div>
                  <div className="document-footer">
                    <span className="document-name">Word file 2017</span>
                    <span className="document-description">1.2 MB</span>
                  </div>
                </div>
              </div>
            </div>
          </CardBody>
        </Card>
      </Collapse>
    </>
  );
}

export default DocumentsAndFiles;
