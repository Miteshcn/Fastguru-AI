import React from "react";
import { Button } from "reactstrap";

export default function Login() {
  return (
    <div className="login-box">
      <h1 className="mb-4">Login</h1>
      <form action="admin">
        <div className="form-group mb-3">
          <input
            type="text"
            className="form-control"
            placeholder="Username or Email"
          />
        </div>
        <div className="form-group mb-4">
          <input type="text" className="form-control" placeholder="Password" />
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
