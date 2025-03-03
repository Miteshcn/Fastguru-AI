import React from 'react';
import './CrmPopup.css';
import { useActiveTab } from '@/app/context/ActiveTab';
import { useCrm } from '@/app/context/CrmContext';


const CrmPopup = ({ onClose , handleSend }) => {
  const {setActiveTab} = useActiveTab()
  const {setCrmName } = useCrm();

  const crms = [
    { name: 'GPS CRM', image: 'https://via.placeholder.com/50' },
    { name: 'Salesforce CRM', image: 'https://via.placeholder.com/50' },
    { name: 'Zoho CRM', image: 'https://via.placeholder.com/50' },
    { name: 'Microsoft ERP', image: 'https://via.placeholder.com/50' },
  ];

  const handleCrmSelection = (crmName) => {
    // alert(`Connecting to ${crmName}`);
    setCrmName(crmName);
    setActiveTab("2")
    handleSend(crmName)
    onClose(); 
  };

  return (
    <div className="crm-popup-overlay">
      <div className="crm-popup">
        <h3>Connect to a CRM</h3>
        <div className="crm-options">
          {crms.map((crm) => (
            <div className="crm-option" key={crm.name}>
              <img src={crm.image} alt={`${crm.name} logo`} className="crm-image" />
              <span className="crm-name">{crm.name}</span>
              <button
                className="crm-connect-button"
                onClick={() => handleCrmSelection(crm.name)}
              >
                Connect
              </button>
            </div>
          ))}
        </div>
        <button className="close-popup" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default CrmPopup;
