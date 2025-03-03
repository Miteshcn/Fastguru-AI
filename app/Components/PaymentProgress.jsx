import React from "react";

const PaymentProgress = ({ status, message }) => {
  return (
    <div className="payment-progress">
      <div className="spinner-border text-primary" role="status" />
      <div className="mt-3 text-center">
        <h5>{status}</h5>
        <p>{message}</p>
      </div>
      <style jsx>{`
        .payment-progress {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100px;
          width: 100%;
        }
      `}</style>
    </div>
  );
};

export default PaymentProgress;
