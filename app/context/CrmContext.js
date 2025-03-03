"use client";
import React, { createContext, useState, useContext } from "react";

const CrmContext = createContext();

export const CrmProvider = ({ children }) => {
  const [crmName, setCrmName] = useState(null);

  return (
    <CrmContext.Provider value={{ crmName, setCrmName }}>
      {children}
    </CrmContext.Provider>
  );
};

export const useCrm = () => useContext(CrmContext);
