import React, { useEffect, useState } from "react";
import { Card, Row, Col, Modal, ModalHeader, ModalBody, Button } from "reactstrap";

export default function TransactionHistory() {
  const [transactions, setTransactions] = useState([]);
  const [selectedTransaction, setSelectedTransaction] = useState(null); 
  const [modalOpen, setModalOpen] = useState(false); 

  useEffect(() => {
    const storedTransactions = localStorage.getItem("transactions");
    if (storedTransactions) {
      const parsedTransactions = JSON.parse(storedTransactions);
      const sortedTransactions = parsedTransactions.sort(
        (a, b) => new Date(b.time) - new Date(a.time)
      );
      setTransactions(sortedTransactions);
    }
  }, []);

  const toggleModal = () => setModalOpen(!modalOpen);

  const handleTransactionClick = (transaction) => {
    setSelectedTransaction(transaction); 
    toggleModal();
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString(); 
  };

  return (
    <div>
      <Row>
        <Col md="12">
          <Card body className="border-0 mb-4">
            <h5 className="mb-4">Transaction History</h5>
            <div className="transaction-list">
              {transactions.length > 0 ? (
                transactions.map((transaction, index) => (
                  <div
                    key={index}
                    className="transaction-card d-flex align-items-center justify-content-between border p-3 mb-3"
                    style={{
                      borderRadius: "8px",
                      backgroundColor: "#f8f9fa",
                    }}
                  >
                    <div
                      style={{
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                      onClick={() => handleTransactionClick(transaction)}
                      className="transaction-info"
                    >
                      <p className="mb-1">
                        <strong>Transaction ID:</strong>{" "}
                        <span style={{ whiteSpace: "nowrap" }}>{transaction.transactionHash}</span>
                      </p>
                      <p className="mb-1">
                        <strong>Amount:</strong> {(transaction.tokenAmount / 10 ** 9).toFixed(9)}
                      </p>
                      <p className="mb-1">
                        <strong>Recipient Address:</strong>{" "}
                        <span style={{ whiteSpace: "nowrap" }}>{transaction.recipientAddress}</span>
                      </p>
                      <p className="mb-1">
                        <strong>Time:</strong> {formatDate(transaction.timestamp)}
                      </p>
                    </div>
                    <Button
                      color="primary"
                      href={`https://polygonscan.com/tx/${transaction.transactionHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Invoice
                    </Button>
                  </div>
                ))
              ) : (
                <p className="text-center">No transactions found</p>
              )}
            </div>
          </Card>
        </Col>
      </Row>

      {selectedTransaction && (
        <Modal isOpen={modalOpen} toggle={toggleModal} size="lg">
          <ModalHeader toggle={toggleModal}>Transaction Details</ModalHeader>
          <ModalBody>
            <p><strong>Transaction ID:</strong> {selectedTransaction.transactionHash}</p>
            <p><strong>Amount:</strong> ${selectedTransaction.tokenAmount}</p>
            <p><strong>Recipient Address:</strong> {selectedTransaction.recipientAddress}</p>
            <p><strong>Time:</strong> {formatDate(selectedTransaction.timestamp)}</p>
            <Button color="primary" onClick={toggleModal} className="mt-3">
              Close
            </Button>
          </ModalBody>
        </Modal>
      )}
    </div>
  );
}
