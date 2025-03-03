import React, { useState } from "react";
import { Collapse, Button, CardBody, Card } from "reactstrap";

function PersonalModules(args) {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

  return (
    <>
      <Button
        className="w-100 d-flex justify-content-start align-items-center mb-0"
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
        <span style={{ marginLeft: "10px" }}>Personal Agent</span>
      </Button>

      <Collapse isOpen={isOpen} {...args}>
        <Card>
          <CardBody className="bg-dark flex">
            <div className="card-body px-2">
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="ai-form-fillup-task"
                onClick={() => showConnectingMessage("ffu")}
              >
                AI-Form Fill-Up
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="set-reminder-general"
                onClick={() => showConnectingMessage("reminder")}
              >
                Set Reminder
              </button>
              <button
                type="button"
                className="btn me-1 mb-2 btn-sm fs-10"
                id="schedule-meeting-general"
                onClick={() => showConnectingMessage("meeting")}
              >
                Schedule Meeting
              </button>
            </div>
          </CardBody>
        </Card>
      </Collapse>
    </>
  );
}

export default PersonalModules;
