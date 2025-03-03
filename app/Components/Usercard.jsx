import React from "react";
import { Card, CardText, CardTitle, Col, Row } from "reactstrap";

export default function Usercard() {
  return (
    <>
      <Card body className="bg-secondary text-white">
        <CardTitle className="fw-bold text-center">Connect To LMS</CardTitle>
        <CardText>
          <Row>
            <Col xs="4" className="pe-0">
              <img
                src="/img/connect-lms.jpg"
                alt=""
                className="w-100 rounded-3"
              />
            </Col>
            <Col xs="8">
              <p className="mb-0 small">Student Enrollment</p>
              <p className="mb-0 small">Course Material Upload</p>
              <p className="mb-0 small">Progress Tracking</p>
              <p className="mb-0 small">Interactive Assesments</p>
              <p className="mb-0 small">Reporting & Analytics</p>
            </Col>
          </Row>
        </CardText>
      </Card>
    </>
  );
}
