"use client";
import Navbar from '@/app/Components/Navbar';
import Sidebar from '@/app/Components/Sidebar'
import React from 'react'

export default function Profile() {
  return (
    <div className="page">
      <Navbar />
      <Sidebar />
      <div className="content">
        <h1 className='admin-main-title'>My Profile</h1>
      </div>
      <style jsx>{`
       
      `}</style>
    </div>
  )
}
