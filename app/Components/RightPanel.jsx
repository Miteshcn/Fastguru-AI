import React from 'react';
import { Modal, Tab, Tabs, Collapse, Button, Card } from 'react-bootstrap';
import { FaFilePdf, FaFileExcel, FaFileWord } from 'react-icons/fa';

function RightPanel() {
  const [showModal, setShowModal] = React.useState(false);
  const [openCollapse, setOpenCollapse] = React.useState(false);

  return (
    <div className="right-panel">
      <div className="right-panel-card bhoechie-tab border border-secondary rounded-4 p-3 min-height-100">
        {/* Business Features */}
        <h5
          className="text-white d-flex align-items-center justify-content-between bg-dark px-3 py-2 h6 rounded-2 cursor-pointer"
          onClick={() => setOpenCollapse(!openCollapse)}
        >
          Enterprises Modules <i className="fa fa-angle-down"></i>
        </h5>
        <Collapse in={openCollapse}>
          <div>
            <Button
              className="btn btn-warning me-1 mb-2 btn-sm fs-10"
              onClick={() => setShowModal(true)}
            >
              Content Creator
            </Button>
          </div>
        </Collapse>

        {/* Modal */}
        <Modal show={showModal} onHide={() => setShowModal(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Content Creator</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Tabs defaultActiveKey="source" id="uncontrolled-tab-example" className="mb-3">
              <Tab eventKey="source" title="Source">
                <div className="chat-container">
                  <form className="from-local mb-2">
                    <label className="text-white fw-bold mb-3 fs-10">
                      Load single or multiple PDF...
                    </label>
                    <input
                      type="file"
                      className="form-control"
                      accept="application/pdf"
                      multiple
                    />
                  </form>
                </div>
              </Tab>
              <Tab eventKey="preview" title="Preview">
                <div>Preview Content</div>
              </Tab>
              <Tab eventKey="target" title="Target">
                <div>Target Content</div>
              </Tab>
            </Tabs>
          </Modal.Body>
        </Modal>

        {/* Documents Section */}
        <h5
          className="text-white d-flex align-items-center justify-content-between bg-dark px-3 py-2 h6 rounded-2 cursor-pointer"
        >
          Documents & Files
        </h5>
        <Card bg="dark" className="text-white">
          <Card.Body className="d-flex">
            <div className="me-3">
              <FaFilePdf size={30} className="text-danger" />
            </div>
            <div>
              <p>PDF File 2017</p>
              <p>5.3 MB</p>
            </div>
          </Card.Body>
        </Card>
        <Card bg="dark" className="text-white mt-2">
          <Card.Body className="d-flex">
            <div className="me-3">
              <FaFileExcel size={30} className="text-success" />
            </div>
            <div>
              <p>Excel File 2017</p>
              <p>2.7 MB</p>
            </div>
          </Card.Body>
        </Card>
      </div>
    </div>
  );
}

export default RightPanel;
