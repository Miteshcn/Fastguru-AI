"use client";
import Navbar from '@/app/Components/Navbar';
import Sidebar from '@/app/Components/Sidebar'
import React from 'react'

export default function Index() {
  return (
    <div className="page">
      <Navbar />
      <Sidebar />
      <div className="content">
        <h1 className='admin-main-title'>Document List</h1>
      </div>     
    </div>
  )
}
