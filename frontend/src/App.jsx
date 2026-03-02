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
        <SystemStats />
        <SpongeAttack />
      </main>
    </div>
  );
}

export default App;
