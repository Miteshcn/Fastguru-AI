"use client";
import React, { useState } from "react";
import {
  Button,
  Card,
  CardBody,
  CardText,
  CardTitle,
  Col,
  Container,
  Nav,
  NavItem,
  NavLink,
  Row,
  TabContent,
  TabPane,
} from "reactstrap";
import classnames from "classnames";
import Collapsebtn from "../Components/Collapsebtn";
import Navbar from "../Components/Navbar";
import Documentreportsummary from "../Components/Documentreportsummary";
import Usercard from "../Components/Usercard";
import Chat from "../Components/Chat";
import Bots from "../Components/Bots";
import Pricing from "../Components/Pricing";
import Mediagallery from "../Components/Mediagallery";
import Contentcreator from "../Components/Contentcreator";
import Features from "../Components/Features";
import TabPaneDocuments from "../Components/pinataUpload";
import TransactionHistory from "../Components/transactionHistory";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import EnterprisesModules from "../Components/EnterprisesModules";
import PersonalModules from "../Components/PersonalModules";
import MyAgentSpace from "../Components/MyAgentSpace";
import DocumentsAndFiles from "../Components/DocumentsAndFiles";
import MeetigsAndEvents from "../Components/MeetngAndEvents";
import { useActiveTab } from "../context/ActiveTab";


export default function Index() {
  const {activeTab , setActiveTab} = useActiveTab()
  // const [activeTab, setActiveTab] = useState("1"); // State to track active tab

  const toggleTab = (tab) => {
    if (activeTab !== tab) {
      setActiveTab(tab);
    }
  };
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);

   const [childFunction, setChildFunction] = useState(null);

  const handleSend = (crmName)=>{
    if (childFunction) {
       childFunction(crmName); 
    }
  }

  return (
    <>
      <ToastContainer />
      <Navbar />
      <section className="chat-section bg-dark">
        <div className="px-3">
          <Nav tabs className="mb-3 border-0">
            <NavItem>
              {/* <NavLink
                className={classnames({ active: activeTab === "1" })}
                onClick={() => toggleTab("1")}
              >
                <i className="bi bi-person"></i> Education Worksapce
              </NavLink> */}
            </NavItem>
            <NavItem>
              <NavLink
                className={classnames({ active: activeTab === "2" })}
                onClick={() => toggleTab("2")}
              >
                <i className="bi bi-robot"></i> My Coworker
              </NavLink>
            </NavItem>
            <NavItem>
              <NavLink
                className={classnames({ active: activeTab === "4" })}
                onClick={() => toggleTab("4")}
              >
                <i className="bi bi-people"></i> Agent Collaboration
              </NavLink>
            </NavItem>
            <NavItem>
              <NavLink
                className={classnames({ active: activeTab === "5" })}
                onClick={() => toggleTab("5")}
              >
                <i className="bi bi-person"></i> Agentspace
              </NavLink>
            </NavItem>
            {/* <NavItem>
              <NavLink
                className={classnames({ active: activeTab === "6" })}
                onClick={() => toggleTab("6")}
              >
                <i className="bi bi-robot"></i> Train Your AI Coworker
              </NavLink>
            </NavItem> */}
            <NavItem>
              <NavLink
                className={classnames({ active: activeTab === "7" })}
                onClick={() => toggleTab("7")}
              >
                <i className="bi bi-robot"></i> Web3 Storage
              </NavLink>
            </NavItem>
            <NavItem>
              <NavLink
                // href="https://polygonscan.com/token/0x20854e3f9231778a1f9515a5551872f161a1e3a2"
                // target="_blank"
                className={classnames({ active: activeTab === "8" })}
                onClick={() => toggleTab("8")}
              >
                <i class="bi bi-repeat"></i> Transaction History
              </NavLink>
            </NavItem>
          </Nav>
          <Row>
            <Col md="3">
              <div className="border rounded-3 p-3 h-100 border-secondary">
                <Collapsebtn />
                <video width="100%" autoplay muted loop className="rounded-3">
                  <source src="/img/user-video.mp4" type="video/mp4" />
                </video>
              </div>
            </Col>
            <Col md="6">
              <TabContent
                activeTab={activeTab}
                className="border rounded-3 p-3 border-secondary"
              >
                <TabPane tabId="1">
                  <Row className="g-3">
                    <Col md="6">
                      <Usercard />
                    </Col>
                    <Col md="6">
                      <Usercard />
                    </Col>
                    <Col md="6">
                      <Usercard />
                    </Col>
                    <Col md="6">
                      <Usercard />
                    </Col>
                  </Row>
                </TabPane>
                <TabPane tabId="2">
                  <Row>
                    <Col sm="12">
                      <div class="chat-header d-flex justify-content-between align-items-center mb-3 bg-secondary p-2 rounded-3">
                        <div class="d-flex align-items-center justify-content-between w-100 py-1">
                          <div class=" pl-3 text-white">
                            <img
                              src="https://fastguruaidev.blob.core.windows.net/workspace/unnamed.png"
                              class="rounded-circle me-2"
                              alt="AI Coworker"
                              width="40"
                              height="40"
                            />
                            <strong className="small text-white">
                              0xE980aC9c9B77A57BcF7638C72c1DCac2f7ee7G8d
                            </strong>
                          </div>
                          <i class="bi bi-copy text-white"></i>
                        </div>
                      </div>
                      <Chat onRegister={setChildFunction} />
                    </Col>
                  </Row>
                </TabPane>
                <TabPane tabId="4">
                  <Row>
                    <Col sm="12">
                      <div class="chat-header d-flex justify-content-between align-items-center mb-3 bg-secondary p-2 rounded-3">
                        <div class="d-flex align-items-center justify-content-between w-100 py-1">
                          <div class="flex-grow-1 pl-3 text-white">
                            <img
                              src="https://fastguruaidev.blob.core.windows.net/workspace/unnamed.png"
                              class="rounded-circle me-3"
                              alt="AI Coworker"
                              width="40"
                              height="40"
                            />
                            <img
                              src="https://fastguruaidev.blob.core.windows.net/workspace/unnamed.png"
                              class="rounded-circle me-3"
                              alt="AI Coworker"
                              width="40"
                              height="40"
                            />
                            <img
                              src="https://fastguruaidev.blob.core.windows.net/workspace/unnamed.png"
                              class="rounded-circle me-3"
                              alt="AI Coworker"
                              width="40"
                              height="40"
                            />
                            <div class="text-muted small">
                              <span class="fe fe-check text-success"></span>
                            </div>
                          </div>
                          <button
                            class="btn rounded-5"
                            style={{
                              backgroundColor: "#242629",
                              color: "white",
                              border: "none",
                            }}
                            data-bs-toggle="modal"
                            data-bs-target="#addMemberModal"
                          >
                            <i class="bi bi-plus-circle"></i> Add Member
                          </button>
                        </div>
                      </div>
                      <Chat />
                    </Col>
                  </Row>
                </TabPane>
                <TabPane tabId="5">
                  <Row className="g-3">
                    <Col md="6">
                      <Bots />
                    </Col>
                    <Col md="6">
                      <Bots />
                    </Col>
                    <Col md="6">
                      <Bots />
                    </Col>
                    <Col md="6">
                      <Bots />
                    </Col>
                  </Row>
                </TabPane>
                <TabPane tabId="6">
                  <Row className="g-3">
                    <Col md="6">
                      <Pricing />
                    </Col>
                    <Col md="6">
                      <Pricing />
                    </Col>
                  </Row>
                </TabPane>
                <TabPane tabId="7">
                  {/* <h1 className="admin-main-title">Documents</h1>
                  <Row>
                    <Col md="12">
                      <Card body className="border-0 mb-4">
                        <Row className="align-items-center">
                          <Col sm="9">
                            <input
                              id="file"
                              name="email"
                              placeholder="something@idk.cool"
                              type="file"
                              className="form-control"
                            />
                          </Col>
                          <Col sm="3">
                            <button className="btn btn-primary rounded-3 w-100 px-2 py-1 h-100">
                              Upload File
                            </button>
                          </Col>
                        </Row>
                      </Card>
                      <Card body className="border-0 mb-4">
                        <h5 className="mb-4">All Documents</h5>
                        <div className="all-documentboxes p-3">
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                          <div className="documentboxes text-center">
                            <img
                              src="/img/home14-imagebox1.png"
                              alt=""
                              className="mb-4"
                            />
                            <p className="mb-0">Lorem ipsum</p>
                          </div>
                        </div>
                      </Card>
                    </Col>
                  </Row> */}
                  <TabPaneDocuments />
                </TabPane>
                <TabPane tabId="8">
                  <TransactionHistory />
                </TabPane>
              </TabContent>
            </Col>
            <Col md="3">
              <div className="border rounded-3 p-5 h-100 border-secondary">
                <EnterprisesModules handleSend={handleSend} />
                <PersonalModules />
                <MyAgentSpace />
                <DocumentsAndFiles />
                {/* <MeetigsAndEvents /> */}
                {/* <Features />
                <Documentreportsummary />
                <Mediagallery />
                <Contentcreator /> */}
              </div>
            </Col>
          </Row>
        </div>
      </section>
    </>
  );
}
