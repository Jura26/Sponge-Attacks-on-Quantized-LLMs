import React, { useState, useEffect } from 'react';
import './SystemStats.css';

const SystemStats = () => {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/stats');
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        setStats(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatTime = (seconds) => {
    if (!seconds || seconds < 0) return '';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m remaining`;
  };

  const getBarColor = (percent) => {
    if (percent < 50) return 'var(--accent-success)';
    if (percent < 80) return 'var(--accent-warning)';
    return 'var(--accent-danger)';
  };

  if (error) {
    return (
      <div className="stats-error">
        <div className="stats-error-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>
        <div>
          <p className="stats-error-title">Failed to load system stats</p>
          <p className="stats-error-desc">{error}</p>
          <p className="stats-error-hint">Make sure the backend server (main.py) is running on port 8000.</p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="stats-loading">
        <div className="loading-spinner" />
        <span>Loading system stats...</span>
      </div>
    );
  }

  return (
    <div className="stats-container">
      <div className="stats-header">
        <h2 className="stats-title">System Overview</h2>
        <span className="stats-subtitle">Real-time hardware monitoring</span>
      </div>

      {/* CPU Section */}
      <div className="stats-card stats-card-full">
        <div className="card-header">
          <div className="card-header-left">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="4" y="4" width="16" height="16" rx="2" ry="2" />
              <rect x="9" y="9" width="6" height="6" />
              <line x1="9" y1="1" x2="9" y2="4" />
              <line x1="15" y1="1" x2="15" y2="4" />
              <line x1="9" y1="20" x2="9" y2="23" />
              <line x1="15" y1="20" x2="15" y2="23" />
              <line x1="20" y1="9" x2="23" y2="9" />
              <line x1="20" y1="14" x2="23" y2="14" />
              <line x1="1" y1="9" x2="4" y2="9" />
              <line x1="1" y1="14" x2="4" y2="14" />
            </svg>
            <h3>CPU</h3>
          </div>
          <span className="card-badge">{stats.cpu_percent}%</span>
        </div>
        <div className="cpu-cores-grid">
          {stats.cpu_per_core && stats.cpu_per_core.map((core, index) => (
            <div key={index} className="cpu-core">
              <div className="cpu-core-bar-bg">
                <div
                  className="cpu-core-bar-fill"
                  style={{
                    height: `${core}%`,
                    backgroundColor: getBarColor(core),
                  }}
                />
              </div>
              <span className="cpu-core-label">{'#'}{index}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="stats-grid-2">
        {/* Memory */}
        <div className="stats-card">
          <div className="card-header">
            <div className="card-header-left">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M6 19v-8a6 6 0 0 1 12 0v8" />
                <rect x="2" y="19" width="20" height="2" rx="1" />
              </svg>
              <h3>Memory</h3>
            </div>
            <span className="card-badge">{stats.memory_percent}%</span>
          </div>
          <p className="card-value">{formatBytes(stats.memory_used)} <span className="card-value-dim">/ {formatBytes(stats.memory_total)}</span></p>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${stats.memory_percent}%`,
                backgroundColor: getBarColor(stats.memory_percent),
              }}
            />
          </div>
        </div>

        {/* Disk */}
        <div className="stats-card">
          <div className="card-header">
            <div className="card-header-left">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3" />
                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
              </svg>
              <h3>Disk</h3>
            </div>
            <span className="card-badge">{stats.disk_percent}%</span>
          </div>
          <p className="card-value">{formatBytes(stats.disk_free)} <span className="card-value-dim">free</span></p>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${stats.disk_percent}%`,
                backgroundColor: 'var(--accent-info)',
              }}
            />
          </div>
        </div>
      </div>

      {/* Battery */}
      {stats.battery && stats.battery.percent !== null && (
        <div className="stats-card">
          <div className="card-header">
            <div className="card-header-left">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="1" y="6" width="18" height="12" rx="2" ry="2" />
                <line x1="23" y1="13" x2="23" y2="11" />
              </svg>
              <h3>Battery</h3>
            </div>
            {stats.battery.power_plugged && (
              <span className="card-badge card-badge-success">Charging</span>
            )}
          </div>
          <div className="battery-row">
            <div className="battery-visual">
              <div className="battery-shell">
                <div
                  className="battery-fill"
                  style={{
                    width: `${stats.battery.percent}%`,
                    backgroundColor: stats.battery.percent < 20 ? 'var(--accent-danger)' : 'var(--accent-success)',
                  }}
                />
              </div>
              <div className="battery-cap" />
            </div>
            <span className="battery-percent">{Math.round(stats.battery.percent)}%</span>
            {!stats.battery.power_plugged && stats.battery.secsleft > 0 && (
              <span className="battery-time">{formatTime(stats.battery.secsleft)}</span>
            )}
          </div>
        </div>
      )}

      {/* Temperatures */}
      <div className="stats-card">
        <div className="card-header">
          <div className="card-header-left">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
            </svg>
            <h3>Thermal Sensors</h3>
          </div>
        </div>
        {stats.temperatures.error ? (
          <p className="temp-error">Could not read temperatures: {stats.temperatures.error}</p>
        ) : (Object.keys(stats.temperatures).length === 0 ? (
          <p className="temp-empty">No sensors detected.</p>
        ) : null)}

        <div className="temp-sensors">
          {Object.entries(stats.temperatures).map(([sensorName, readings]) => (
            Array.isArray(readings) && readings.length > 0 && (
              <div key={sensorName} className="temp-sensor-group">
                <span className="temp-sensor-name">{sensorName}</span>
                <div className="temp-readings">
                  {readings.map((reading, idx) => (
                    <div key={idx} className="temp-reading">
                      <span className="temp-label">{reading.label || `#${idx + 1}`}</span>
                      <span
                        className="temp-value"
                        style={{
                          color: reading.current > 75 ? 'var(--accent-danger)' : 'var(--text-primary)',
                        }}
                      >
                        {reading.current}°C
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )
          ))}
        </div>
      </div>
    </div>
  );
};

export default SystemStats;
