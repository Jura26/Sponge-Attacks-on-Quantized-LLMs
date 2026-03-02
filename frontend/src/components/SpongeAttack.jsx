import React, { useState, useEffect, useRef } from 'react';
import './SpongeAttack.css';

const SpongeAttack = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState([]);
  const [bestResult, setBestResult] = useState(null);
  const [generations, setGenerations] = useState(5);
  const [population, setPopulation] = useState(10);
  const terminalRef = useRef(null);

  const scrollToBottom = () => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/attack/status');
        if (response.ok) {
          const data = await response.json();
          setLogs(data.logs || []);
          setBestResult(data.best_result);
          setIsRunning(data.is_running);
          setStatus(data.status);
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const startAttack = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/attack/start?gens=${generations}&pop=${population}`,
        { method: 'POST' }
      );

      if (response.ok) {
        setLogs([]);
        setBestResult(null);
      } else {
        const err = await response.json();
        alert(`Error: ${err.error || 'Failed to start'}`);
      }
    } catch (error) {
      console.error('Error starting attack:', error);
    }
  };

  const getStatusClass = () => {
    if (status === 'running') return 'status-running';
    if (status === 'completed') return 'status-completed';
    return 'status-idle';
  };

  return (
    <div className="attack-container">
      <div className="attack-header">
        <div className="attack-title-row">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="4 17 10 11 4 5" />
            <line x1="12" y1="19" x2="20" y2="19" />
          </svg>
          <h2 className="attack-title">Sponge Attack Console</h2>
        </div>
        <p className="attack-desc">Genetic algorithm-based resource exhaustion testing</p>
      </div>

      <div className="attack-controls">
        <div className="control-group">
          <label className="control-label">Generations</label>
          <input
            type="number"
            className="control-input"
            value={generations}
            onChange={(e) => setGenerations(parseInt(e.target.value))}
            disabled={isRunning}
            min="1"
          />
        </div>
        <div className="control-group">
          <label className="control-label">Population</label>
          <input
            type="number"
            className="control-input"
            value={population}
            onChange={(e) => setPopulation(parseInt(e.target.value))}
            disabled={isRunning}
            min="2"
          />
        </div>
        <button
          className={`attack-btn ${isRunning ? 'attack-btn-running' : ''}`}
          onClick={startAttack}
          disabled={isRunning}
        >
          {isRunning ? (
            <>
              <span className="btn-spinner" />
              Attack in Progress...
            </>
          ) : (
            'Start Sponge Attack'
          )}
        </button>
      </div>

      <div className="attack-status-bar">
        <div className="attack-status-left">
          <span className={`attack-status-indicator ${getStatusClass()}`} />
          <span className="attack-status-label">{status.toUpperCase()}</span>
        </div>
        {bestResult && (
          <span className="attack-status-score">
            Best Score: <strong>{bestResult.score.toFixed(4)}</strong>
          </span>
        )}
      </div>

      <div className="attack-dashboard">
        {/* Log Terminal */}
        <div className="attack-terminal">
          <div className="terminal-header">
            <div className="terminal-dots">
              <span className="dot dot-red" />
              <span className="dot dot-yellow" />
              <span className="dot dot-green" />
            </div>
            <span className="terminal-title">Attack Logs</span>
          </div>
          <div className="terminal-body" ref={terminalRef}>
            {logs.length === 0 ? (
              <div className="terminal-empty">
                <span className="terminal-prompt">{'$'}</span> Waiting for attack to start...
              </div>
            ) : (
              logs.map((log, index) => (
                <div key={index} className="terminal-line">
                  <span className="terminal-line-num">{String(index + 1).padStart(3, ' ')}</span>
                  {log}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Results Panel */}
        <div className="attack-results">
          <h3 className="results-title">Best Result</h3>
          {bestResult ? (
            <div className="results-content">
              <div className="result-metrics">
                <div className="metric-item">
                  <span className="metric-label">Total Score</span>
                  <span className="metric-value metric-value-primary">{bestResult.score.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Avg CPU Load</span>
                  <span className="metric-value">{Number(bestResult.avg_cpu).toFixed(2)}%</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Duration</span>
                  <span className="metric-value">{bestResult.duration.toFixed(4)}s</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Input Tokens</span>
                  <span className="metric-value">{bestResult.input_tokens}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Output Tokens</span>
                  <span className="metric-value">{bestResult.output_tokens}</span>
                </div>
              </div>

              <div className="result-data">
                <label className="data-label">Trigger Prompt</label>
                <textarea readOnly value={bestResult.prompt} className="data-textarea" />
              </div>

              <div className="result-data">
                <label className="data-label">Model Output</label>
                <textarea readOnly value={bestResult.output} className="data-textarea data-textarea-output" />
              </div>
            </div>
          ) : (
            <div className="results-empty">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p>No results yet</p>
              <span>Start an attack to find resource-heavy prompts.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpongeAttack;
