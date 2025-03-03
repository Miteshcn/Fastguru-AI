import React from "react";
import { Card } from "reactstrap";

export default function Pricing() {
  return (
    <>
      <Card body className="bg-secondary">
        <div class="text-light">
          <h2 class="fw-bold mb-0">Pro Plan</h2>
          <h5 class="mb-4">15$/Month</h5>
          <hr />
          <ul class="ps-3 ms-1">
            <li>Project Builder</li>
            <li>Fastguru LLM Endpoint</li>
            <li>Customized Bot Builder</li>
            <li>Developer Bot Builder</li>
            <li>Regular Bot Builder</li>
            <li>Regular Videos Builder</li>
            <li>Fast Assistent (Ask Ishaan)</li>
            <li>Fast AI-Training</li>
            <li>Fast Tutor</li>
            <li>Multi Language</li>
          </ul>
          <a href="" class="btn btn-warning">
            Upgrade
          </a>
        </div>
      </Card>
    </>
  );
}
