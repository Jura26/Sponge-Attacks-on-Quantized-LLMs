import React, { useState } from 'react';
import { useEffect } from 'react';
import './SystemStats.css';

const Ico = ({ children, size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
);

const SystemStats = () => {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [thermalOpen, setThermalOpen] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/stats');
        if (!res.ok) throw new Error('Network response was not ok');
        setStats(await res.json());
        setError(null);
      } catch (err) { setError(err.message); }
    };
    fetchStats();
    const iv = setInterval(fetchStats, 2000);
    return () => clearInterval(iv);
  }, []);

  const fmtBytes = (b) => {
    if (!b) return '0 B';
    const k = 1024, sizes = ['B','KB','MB','GB','TB'];
    const i = Math.floor(Math.log(b) / Math.log(k));
    return `${(b / k ** i).toFixed(1)} ${sizes[i]}`;
  };

  const fmtTime = (s) => {
    if (!s || s < 0) return '';
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  };

  const barColor = (p) => p < 50 ? 'var(--green)' : p < 80 ? 'var(--yellow)' : 'var(--red)';

  if (error) return (
    <div className="stats-error">
      <div className="stats-error-icon">
        <Ico><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></Ico>
      </div>
      <div>
        <p className="stats-error-title">BACKEND UNREACHABLE</p>
        <p className="stats-error-desc">{error}</p>
        <p className="stats-error-hint">Start main.py on port 8000 to connect.</p>
      </div>
    </div>
  );

  if (!stats) return (
    <div className="stats-loading">
      <div className="loading-spinner" />
      Connecting to backend...
    </div>
  );

  // Flatten all thermal readings into a single list
  const thermalReadings = [];
  if (stats.temperatures) {
    Object.entries(stats.temperatures).forEach(([sensorName, readings]) => {
      if (!Array.isArray(readings) || sensorName.startsWith('_')) return;
      readings.forEach((r, i) => {
        thermalReadings.push({
          group: sensorName.replace(/_/g, ' '),
          label: r.label || `#${i + 1}`,
          current: r.current,
          high: r.high,
          unit: r.unit || '°C',
          source: r.source,
        });
      });
    });
  }

  return (
    <div className="stats-container">

      {/* Row 1: CPU full width */}
      <div className="stats-card">
        <div className="card-header">
          <div className="card-header-left">
            <Ico>
              <rect x="4" y="4" width="16" height="16" rx="1"/>
              <rect x="9" y="9" width="6" height="6"/>
              <line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>
              <line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>
              <line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>
              <line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>
            </Ico>
            <h3>CPU</h3>
          </div>
          <span className="card-badge">{stats.cpu_percent}%</span>
        </div>
        <div className="cpu-cores-grid">
          {stats.cpu_per_core?.map((c, i) => (
            <div key={i} className="cpu-core">
              <div className="cpu-core-bar-bg">
                <div className="cpu-core-bar-fill" style={{ height: `${c}%`, backgroundColor: barColor(c) }} />
              </div>
              <span className="cpu-core-label">#{i}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Row 2: Memory + Disk */}
      <div className="stats-grid-2">
        <div className="stats-card">
          <div className="card-header">
            <div className="card-header-left">
              <Ico><path d="M6 19v-8a6 6 0 0 1 12 0v8"/><rect x="2" y="19" width="20" height="2" rx="1"/></Ico>
              <h3>Memory</h3>
            </div>
            <span className="card-badge">{stats.memory_percent}%</span>
          </div>
          <p className="card-value">{fmtBytes(stats.memory_used)} <span className="card-value-dim">/ {fmtBytes(stats.memory_total)}</span></p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${stats.memory_percent}%`, backgroundColor: barColor(stats.memory_percent) }} />
          </div>
        </div>

        <div className="stats-card">
          <div className="card-header">
            <div className="card-header-left">
              <Ico>
                <ellipse cx="12" cy="5" rx="9" ry="3"/>
                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
              </Ico>
              <h3>Disk</h3>
            </div>
            <span className="card-badge">{stats.disk_percent}%</span>
          </div>
          <p className="card-value">{fmtBytes(stats.disk_free)} <span className="card-value-dim">free</span></p>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${stats.disk_percent}%`, backgroundColor: 'var(--amber)' }} />
          </div>
        </div>
      </div>

      {/* Row 3: Battery (conditional) */}
      {stats.battery?.percent != null && (
        <div className="stats-card stats-card-inline">
          <div className="card-header" style={{ marginBottom: 0 }}>
            <div className="card-header-left">
              <Ico><rect x="1" y="6" width="18" height="12" rx="2"/><line x1="23" y1="13" x2="23" y2="11"/></Ico>
              <h3>Battery</h3>
            </div>
            <div className="battery-row">
              <div className="battery-visual">
                <div className="battery-shell">
                  <div className="battery-fill" style={{ width: `${stats.battery.percent}%`, backgroundColor: stats.battery.percent < 20 ? 'var(--red)' : 'var(--green)' }} />
                </div>
                <div className="battery-cap" />
              </div>
              <span className="battery-percent">{Math.round(stats.battery.percent)}%</span>
              {!stats.battery.power_plugged && stats.battery.secsleft > 0 && (
                <span className="battery-time">{fmtTime(stats.battery.secsleft)} remaining</span>
              )}
              {stats.battery.power_plugged && <span className="card-badge card-badge-success">Charging</span>}
            </div>
          </div>
        </div>
      )}

      {/* Row 4: Thermal — compact scrollable table */}
      <div className="stats-card">
        <button className="thermal-toggle" onClick={() => setThermalOpen(o => !o)}>
          <div className="card-header-left">
            <Ico><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></Ico>
            <h3>Thermal Sensors</h3>
            {thermalReadings.length > 0 && (
              <span className="thermal-count">{thermalReadings.length} sensors</span>
            )}
          </div>
          <span className="thermal-chevron" style={{ transform: thermalOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>
            <Ico size={12}>
              <polyline points="18 15 12 9 6 15"/>
            </Ico>
          </span>
        </button>

        {thermalOpen && (
          <>
            {stats.temperatures?.error && <p className="temp-error">Could not read: {stats.temperatures.error}</p>}
            {stats.temperatures?._hint && <p className="temp-hint">{stats.temperatures._hint}</p>}
            {thermalReadings.length === 0 && !stats.temperatures?.error && (
              <p className="temp-empty">No sensors detected.</p>
            )}
            {thermalReadings.length > 0 && (
              <div className="thermal-table-wrap">
                <table className="thermal-table">
                  <thead>
                    <tr>
                      <th>Sensor</th>
                      <th>Label</th>
                      <th>Temp</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {thermalReadings.map((r, i) => (
                      <tr key={i}>
                        <td className="thermal-group">{r.group}</td>
                        <td className="thermal-label">{r.label}</td>
                        <td className="thermal-val" style={{ color: typeof r.current === 'number' && r.current > 75 ? 'var(--red)' : 'var(--text-primary)' }}>
                          {typeof r.current === 'number' ? `${r.current}${r.unit}` : r.current}
                        </td>
                        <td className="thermal-max">{r.high != null ? `${r.high}${r.unit}` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

    </div>
  );
};

export default SystemStats;
