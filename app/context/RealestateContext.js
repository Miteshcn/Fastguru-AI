"use client"
import React, { createContext, useState, useContext } from "react";

const RealestateContext = createContext();

export const RealestateProvider = ({ children }) => {
  const [realestate, setRealestate] = useState(false); // Initialize as false (not clicked)

  return (
    <RealestateContext.Provider value={{ realestate, setRealestate }}>
      {children}
    </RealestateContext.Provider>
  );
};

export const useRealestate = () => useContext(RealestateContext);
