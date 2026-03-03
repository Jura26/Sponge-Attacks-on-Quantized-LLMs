import React from 'react';
import Header from './components/Header';
import SystemStats from './components/SystemStats';
import SpongeAttack from './components/SpongeAttack';
import './App.css';

function App() {
  return (
    <div className="app">
      <Header />
      <main className="main-content">
        <div className="section-block">
          <div className="section-label">System Overview</div>
          <SystemStats />
        </div>
        <div className="section-block">
          <div className="section-label">Sponge Attack Console</div>
          <SpongeAttack />
        </div>
      </main>
    </div>
  );
}

export default App;
