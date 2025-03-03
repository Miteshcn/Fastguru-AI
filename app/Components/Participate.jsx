import Link from "next/link";
import React from "react";
import {
  Button,
  Col,
  Container,
  Form,
  FormGroup,
  Input,
  Row,
} from "reactstrap";

export default function Participate() {
  return (
    <>
      <div className="participate-section text-white">
        <Container>
          <Row className="align-items-center mb-5">
            <Col md="6" lg="6">
              <h2 className="fs-1">
                Participate in the FastCoin ICO and support the future of
                education!
              </h2>
            </Col>
            <Col md="6" lg="6">
              <img src="/img/home14-image2.png" alt="" className="w-100" />
            </Col>
          </Row>
          <Row>
            <Col sm="6" md="3" lg="3">
              <div className="text-center percent-box">
                <span>50%</span>
                <p>Pre Sale</p>
              </div>
            </Col>
            <Col sm="6" md="3" lg="3">
              <div className="text-center percent-box">
                <span>30%</span>
                <p>Dec 2024</p>
              </div>
            </Col>
            <Col sm="6" md="3" lg="3">
              <div className="text-center percent-box">
                <span>10%</span>
                <p>April 2025</p>
              </div>
            </Col>
            <Col sm="6" md="3" lg="3">
              <div className="text-center percent-box">
                <span>10%</span>
                <p>Oct 2025</p>
              </div>
            </Col>
          </Row>
        </Container>
      </div>
      <div className="fee-collection text-white">
        <Container>
          <Row className="justify-content-between align-items-center">
            <Col lg="5" md="5">
              <div className="coinbox">
                <h3 className="text-warning mb-0">1 USD = 100 FTC</h3>
                <h6 className="mb-5 ">Fastcoin Time Efficiency</h6>
                <div className="d-flex align-items-center justify-content-around">
                  <p className="mb-0">
                    <i class="bi bi-check2"></i> Transaction Completed
                  </p>
                  <span>9:40 AM</span>
                </div>
                <div className="d-flex align-items-center justify-content-around active">
                  <p className="mb-0">
                    <i class="bi bi-check2"></i> Transaction Completed
                  </p>
                  <span>9:40 AM</span>
                </div>
                <div className="d-flex align-items-center justify-content-around">
                  <p className="mb-0">
                    <i class="bi bi-check2"></i> Transaction Completed
                  </p>
                  <span>9:40 AM</span>
                </div>
                <div className="d-flex align-items-center justify-content-around">
                  <p className="mb-0">
                    <i class="bi bi-check2"></i> Transaction Completed
                  </p>
                  <span>9:40 AM</span>
                </div>
              </div>
            </Col>
            <Col lg="6" md="6">
              <h1 className="mb-4 fw-bold">
                Efficient Fee <br /> Collection
              </h1>
              <p className="opacity-75 mb-4">
                Fastcoin redefines education finance with a dedicated mobile app
                designed for students, tutors, and educational organizations.
                Here are four key features of Fastcoin:
              </p>
              <ul className="mb-5">
                <li>Student Convenience</li>
                <li>Tutor Efficiency Management</li>
                <li>Organizational Effortless Fee Collection</li>
                <li>Top-Tier Security</li>
              </ul>
              <Link href="" className="btn btn-light w-75 mb-5">
                Invest Now & Grow with us <i class="bi bi-arrow-up-right"></i>
              </Link>
            </Col>
          </Row>
        </Container>
      </div>
      <div className="fastcoin text-white">
        <Container>
          <Row className="g-4">
            <Col lg="12">
              <h1 className="text-center fw-bold mb-5">Why Fastcoin</h1>
            </Col>
            <Col lg="3" md="3" sm="6">
              <div className="fastcoin-box text-center">
                <img src="/img/home14-imagebox1.png" alt="" className="mb-3" />
                <h6>Lightning-fast transactions</h6>
                <p>
                  FastCoin ensures that educational transactions are processed
                  in seconds, eliminating delays and streamlining administrative
                  processes.
                </p>
              </div>
            </Col>
            <Col lg="3" md="3" sm="6">
              <div className="fastcoin-box text-center">
                <img src="/img/home14-imagebox2.png" alt="" className="mb-3" />
                <h6>Maximum security</h6>
                <p>
                  Our platform prioritizes the security of educational
                  transactions, utilizing advanced encryption techniques to
                  safeguard sensitive information and financial data.
                </p>
              </div>
            </Col>
            <Col lg="3" md="3" sm="6">
              <div className="fastcoin-box text-center">
                <img src="/img/home14-imagebox3.png" alt="" className="mb-3" />
                <h6>Cost-effective</h6>
                <p>
                  FastCoin offers competitive transaction fees, allowing
                  educational institutions to save money on administrative costs
                  and allocate resources more efficiently.
                </p>
              </div>
            </Col>
            <Col lg="3" md="3" sm="6">
              <div className="fastcoin-box text-center">
                <img src="/img/home14-imagebox4.png" alt="" className="mb-3" />
                <h6>User-friendly</h6>
                <p>
                  Interface: Our intuitive platform is designed to be
                  user-friendly, making it easy for educators, students, and
                  administrators to navigate and utilize the FastCoin ecosystem.
                </p>
              </div>
            </Col>
          </Row>
        </Container>
        <div className="our-mission text-white">
          <Container>
            <Row className="align-items-center">
              <Col lg="5" md="5">
                <h1 className="fw-bold mb-4">
                  Our Mission to <br /> prioritizes security and <br /> ease of
                  use
                </h1>
                <p className="opacity-75">
                  Fastcoin intuitive interface ensures that students, tutors,
                  and organizations can navigate the app effortlessly, making
                  FastGuru the go-to platform for all their financial needs in
                  the education sector.
                </p>
              </Col>
              <Col lg="7" md="7" className="ps-lg-5">
                <img src="/img/home14-image3.png" alt="" className="w-100" />
              </Col>
            </Row>
          </Container>
        </div>
      </div>
      <div className="mobile-app text-white">
        <Container>
          <Row className="justify-content-between align-items-center">
            <Col lg="5" md="5">
              <img src="/img/fastg.jpg" alt="" className="w-100" />
            </Col>
            <Col lg="6" md="6">
              <h1 className="fw-bold mb-4">FastGuru Mobile App:</h1>
              <h1 className="fw-bold mb-4">
                Revolutionizing Student, <br /> Tutor, and Organization <br />{" "}
                Payments.
              </h1>
              <p className="opacity-75">
                FastGuru introduces an innovative mobile app tailored for the
                dynamic needs of students, tutors, and educational
                organizations. With its user-friendly interface and robust
                features, FastGuru simplifies financial transactions within the
                education sector, ensuring a seamless experience for all users.
              </p>
              <ul className="list-unstyled">
                <li>
                  <a href="">
                    <i className="bi bi-check2"></i> Empowering Students
                  </a>
                </li>
                <li>
                  <a href="">
                    <i className="bi bi-check2"></i> Academics Success
                  </a>
                </li>
              </ul>
              <p className="opacity-75 mb-4">
                FastGuru stands out as the premier mobile app for students,
                tutors, and educational organizations alike. Whether you’re a
                student managing tuition payments, a tutor offering your
                expertise, or an organization streamlining financial processes,
                Download FastGuru today and revolutionize the way you handle
                education-related payments!{" "}
              </p>
              <Link href="" className="btn btn-warning ">
                Download App <i class="bi bi-arrow-up-right"></i>
              </Link>
            </Col>
          </Row>
        </Container>
      </div>
      <div className="footer-section">
        <Container>
          <Row>
            <Col lg="12" className="text-center">
              <img
                src="/img/logo.png"
                alt="logo"
                className="mb-4"
                height="60"
              />
              <h1 className="fw-bold mb-4">
                Ready to revolutionize the education industry?
              </h1>
              <h6 className="fw-bold mb-5">
                Join the FastCoin ICO today and become a part of the movement to
                transform how financial transactions are conducted in education.
                With FastCoin, educators, students, and institutions alike can
                enjoy fast, secure, and cost-effective transactions tailored to
                their unique needs.
              </h6>
              <Row className="justify-content-center">
                <Col lg="4" md="4">
                  <input type="text" className="form-control mb-4 text-center" placeholder="Enter Your Email" />
                  <button type="" className="btn btn-dark w-100 ">
                    Join the Mailing List
                  </button>
                </Col>
              </Row>
            </Col>
          </Row>
        </Container>
      </div>
    </>
  );
}
