import React from "react";
import { Col, Container, Row } from "reactstrap";

export default function Footer() {
  return (
    <footer>      
      <Container>
        <Row>
          <Col lg="6" md="6">
            <ul className="list-unstyled nav mb-0">
              <li>
                <a href="">Home</a>
              </li>
              <li>
                <a href="">Service</a>
              </li>
              <li>
                <a href="">Projects</a>
              </li>
              <li>
                <a href="">Contacts</a>
              </li>
            </ul>
          </Col>
          <Col lg="6" md="6">
            <p className="text-end mb-0">© 2024 Edu-Fast Network. All rights reserved</p>
          </Col>
        </Row>
      </Container>
    </footer>
  );
}
