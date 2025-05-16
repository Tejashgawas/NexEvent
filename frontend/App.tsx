import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Footer from './components/Footer';
import Event from './pages/Event'; 
import Theme from './pages/Theme';
import Login from './pages/Login';
import Service from './pages/Service';
import SignUp from './pages/SignUp';
import { AuthProvider } from './providers/AuthProvider';
import EventLogForm from './pages/EventLogForm';
import EventView from './pages/EventView';
import OrganizerFeedback from './components/organizer_feedback';
import UploadParticipantsCSV from './components/test';







function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen flex flex-col">
          <Navbar />
          <main className="flex-grow pt-16">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/service" element={<Service />} />
              <Route path="/event" element={<Event />} />
              <Route path="/theme" element={<Theme />} />
              <Route path="/login" element={<Login />} />
              <Route path="/SignUp" element={<SignUp />} />
              <Route path="/event-log" element={<EventLogForm />} />
              <Route path="/events" element={<EventView />} />
              <Route path="/feedback" element={<OrganizerFeedback />} />
              <Route path="/upload-participants" element={<UploadParticipantsCSV />} />


         
              {/* Add more routes as needed */}
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </AuthProvider>
  );
}


export default App;