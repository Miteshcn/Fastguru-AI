import React, { useState } from "react";
import { Collapse, Button, CardBody, Card } from "reactstrap";

function MyAgentSpace(args) {
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
        <span style={{ marginLeft: "10px" }}>My Agentspace</span>
      </Button>

      <Collapse isOpen={isOpen} {...args}>
        <Card>
          <CardBody className="bg-dark flex">
            <div className="card-body px-2">
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="schedule-meeting-general"
              >
                Document Generator
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="ai-form-fillup-task"
              >
                Resume Builder
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="set-reminder-general"
                onClick={() => showConnectingMessage("reminder")}
              >
                Lesson Planner
              </button>
            </div>
          </CardBody>
        </Card>
      </Collapse>
    </>
  );
}

export default MyAgentSpace;
