import Link from "next/link";
import React from "react";
import { Col, Container, Progress, Row } from "reactstrap";

export default function Herosection() {
  return (
    <div className="hero-section">
      <Container className="pb-5">
        <Row className="align-items-center justify-content-between">
          <Col md="6" lg="6" className="hero-text">
            <h1 className="mb-4">ICO Tokens Details & Sale Tokens.</h1>
            <p className="mb-5">
              Leverage on any tokens with a protocol trusted with billions for
              its price execution, super low fees and reliability.
            </p>
            <Link href="" className="btn btn-primary">
              Whitepaper <i class="bi bi-arrow-up-right"></i>
            </Link>
          </Col>
          <Col md="6" lg="6" className="ps-lg-5">
            <div className="hero-counter text-center">
              <h4 className="mb-4">Token Sale is Live 5% Bonus Ends In</h4>
              <Row className="mb-4">
                <Col xs="3">
                  <div className="count">
                    <span>30</span>
                    <p>Days</p>
                  </div>
                </Col>
                <Col xs="3">
                  <div className="count">
                    <span>13</span>
                    <p>Hours</p>
                  </div>
                </Col>
                <Col xs="3">
                  <div className="count">
                    <span>30</span>
                    <p>Minutes</p>
                  </div>
                </Col>
                <Col xs="3">
                  <div className="count">
                    <span>10</span>
                    <p>Seconds</p>
                  </div>
                </Col>
              </Row>
              <Link href="" className="btn btn-warning mb-5">
                Register & Buy Token Now <i class="bi bi-arrow-up-right"></i>
              </Link>
              <div className="mb-5">
                <Progress max={500} value="463" />
                <div className="d-flex align-items-center justify-content-between text-white opacity-75 mt-2">
                  <span>Raisen: $65,556.80</span>
                  <span>$100.000</span>
                </div>
              </div>
              <div className="">
                <h5 className="text-white mb-4">We Accepted:</h5>
                <Row>
                  <Col xs="3">
                    <div className="payment-mode">
                      <img src="/img/home14-mastercard.png" alt="" />
                    </div>
                  </Col>
                  <Col xs="3">
                    <div className="payment-mode">
                      <img src="/img/home14-btc.png" alt="" />
                    </div>
                  </Col>
                  <Col xs="3">
                    <div className="payment-mode">
                      <img src="/img/home14-eth.png" alt="" />
                    </div>
                  </Col>
                  <Col xs="3">
                    <div className="payment-mode">
                      <img src="/img/home14-ripple.png" alt="" />
                    </div>
                  </Col>
                </Row>
              </div>
            </div>
          </Col>
        </Row>
      </Container>
      <Container className="pt-5 text-white hero-text">
        <Row className="align-items-center">
          <Col lg="6" md="6">
            <h2 className="text-white">What FastCoin (FTC) Offers ?</h2>
            <p className="opacity-75 mb-4">
              Transforming Student, Tutor, and Organization Payments.
            </p>
            <p className="mb-4">
              <strong className="text-warning">
                Tuition Payments Made Easy:
              </strong>
              <span className="opacity-75 ">
                Swift, secure payments for tuition fees directly from the app,
                eliminating paperwork and long queues.
              </span>
            </p>
            <p className="mb-4">
              <strong className="text-warning">
                One-Stop Course Material Shop:
              </strong>
              <span className="opacity-75 ">
                Browse, purchase, and download textbooks and study materials
                hassle-free within the app.
              </span>
            </p>
          </Col>
          <Col lg="6" md="6">
            <p className="mb-4">
              <strong className="text-warning">
              Streamlined Service Management: 
              </strong>
              <span className="opacity-75 ">
              Set availability, manage appointments, and offer services seamlessly through the app.
              </span>
            </p>
            <p className="mb-4">
              <strong className="text-warning">
              Instant Earnings: 
              </strong>
              <span className="opacity-75 ">
              Prompt and secure payments ensure tutors focus on teaching, not tracking down fees.
              </span>
            </p>
            <p className="mb-4">
              <strong className="text-warning">
              Effortless Fee Collection: 
              </strong>
              <span className="opacity-75 ">
              Simplified student fee processing with a secure and transparent platform.
              </span>
            </p>
            <p>
              <strong className="text-warning">
              Smooth Staff Payments: 
              </strong>
              <span className="opacity-75 ">
              Manage payroll and reimbursements efficiently, enhancing administrative processes.
              </span>
            </p>
          </Col>
        </Row>
      </Container>
    </div>
  );
}
