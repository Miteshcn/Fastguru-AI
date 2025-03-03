import React from "react";
import { Button } from "reactstrap";

export default function Signup() {
  return (
    <div className="login-box">
      <h1 className="mb-4">Sign In</h1>
      <form>
        <div className="form-group mb-3">
          <input type="text" className="form-control" placeholder="Full Name" />
        </div>
        <div className="form-group mb-3">
          <input type="email" className="form-control" placeholder="Email" />
        </div>
        <div className="form-group mb-3">
          <input type="text" className="form-control" placeholder="Password" />
        </div>
        <div className="form-group mb-4">
          <input
            type="text"
            className="form-control"
            placeholder="Confirm Password"
          />
        </div>
        <div className="form-group">
          <Button type="submit" className="py-2 w-100">
            Submit
          </Button>
        </div>
      </form>
    </div>
  );
}
