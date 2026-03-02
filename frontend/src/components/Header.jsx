import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
          <span className="header-title">System Monitor</span>
          <span className="header-version">v1.0</span>
        </div>
        <div className="header-status">
          <span className="status-dot" />
          <span className="status-text">Live</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
