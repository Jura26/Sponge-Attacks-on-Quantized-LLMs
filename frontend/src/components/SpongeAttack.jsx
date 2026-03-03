import React, { useState, useEffect, useRef } from 'react';
import './SpongeAttack.css';

const SpongeAttack = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState([]);
  const [bestResult, setBestResult] = useState(null);
  const [generations, setGenerations] = useState(5);
  const [population, setPopulation] = useState(10);
  const [selectedModel, setSelectedModel] = useState('gpt2');
  const terminalRef = useRef(null);

  const models = [
    { id: 'gpt2', label: 'GPT-2 Small (124M)', size: '~1 GB' },
    { id: 'gpt2-medium', label: 'GPT-2 Medium (355M)', size: '~1.5 GB' },
    { id: 'gpt2-large', label: 'GPT-2 Large (774M)', size: '~3–4 GB' },
    { id: 'gpt2-xl', label: 'GPT-2 XL (1.5B)', size: '~6–8 GB' },
    { id: 'facebook/opt-2.7b', label: 'OPT-2.7B', size: '~14 GB' },
    { id: 'facebook/opt-6.7b', label: 'OPT-6.7B', size: '~20 GB' },
    { id: 'facebook/opt-13b', label: 'OPT-13B', size: '~32 GB' },
    { id: 'mistralai/Mistral-7B-v0.1', label: 'Mistral-7B', size: '~12 GB' },
    { id: 'meta-llama/Llama-2-7b-hf', label: 'LLaMA 2-7B', size: '~14 GB' },
  ];

  useEffect(() => {
    if (terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/api/attack/status');
        if (res.ok) {
          const data = await res.json();
          setLogs(data.logs || []);
          setBestResult(data.best_result);
          setIsRunning(data.is_running);
          setStatus(data.status);
        }
      } catch {}
    }, 1000);
    return () => clearInterval(iv);
  }, []);

  const startAttack = async () => {
    try {
      const res = await fetch(
        `http://localhost:8000/api/attack/start?model_id=${encodeURIComponent(selectedModel)}&gens=${generations}&pop=${population}`,
        { method: 'POST' }
      );
      if (res.ok) { setLogs([]); setBestResult(null); }
      else { const e = await res.json(); alert(`Error: ${e.error || 'Failed to start'}`); }
    } catch (e) { console.error(e); }
  };

  const statusClass = status === 'running' ? 'status-running' : status === 'completed' ? 'status-completed' : '';

  return (
    <div className="attack-container">

      {/* Controls */}
      <div className="attack-controls">
        <div className="control-group">
          <label className="control-label">Target Model</label>
          <select className="control-input" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} disabled={isRunning}>
            {models.map(m => <option key={m.id} value={m.id}>{m.label} ({m.size})</option>)}
          </select>
        </div>
        <div className="control-group">
          <label className="control-label">Generations</label>
          <input type="number" className="control-input" value={generations} onChange={e => setGenerations(parseInt(e.target.value))} disabled={isRunning} min="1" />
        </div>
        <div className="control-group">
          <label className="control-label">Population</label>
          <input type="number" className="control-input" value={population} onChange={e => setPopulation(parseInt(e.target.value))} disabled={isRunning} min="2" />
        </div>
        <button className={`attack-btn${isRunning ? ' attack-btn-running' : ''}`} onClick={startAttack} disabled={isRunning}>
          {isRunning ? <><span className="btn-spinner" />Running...</> : 'Start Attack'}
        </button>
      </div>

      {/* Status */}
      <div className="attack-status-bar">
        <div className="attack-status-left">
          <span className={`attack-status-indicator ${statusClass}`} />
          <span className="attack-status-label">{status.toUpperCase()}</span>
        </div>
        {bestResult && (
          <span className="attack-status-score">Best score: <strong>{bestResult.score.toFixed(4)}</strong></span>
        )}
      </div>

      {/* Dashboard */}
      <div className="attack-dashboard">

        {/* Terminal */}
        <div className="attack-terminal">
          <div className="terminal-header">
            <div className="terminal-dots">
              <span className="dot dot-red" /><span className="dot dot-yellow" /><span className="dot dot-green" />
            </div>
            <span className="terminal-title">Attack Log</span>
          </div>
          <div className="terminal-body" ref={terminalRef}>
            {logs.length === 0 ? (
              <div className="terminal-empty"><span className="terminal-prompt">$</span>Waiting for attack to start...</div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="terminal-line">
                  <span className="terminal-line-num">{String(i + 1).padStart(3, ' ')}</span>
                  {log}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Results */}
        <div className="attack-results">
          <div className="results-title">Best Result</div>
          {bestResult ? (
            <div className="results-content">
              <div className="result-metrics">
                <div className="metric-item">
                  <span className="metric-label">Total Score</span>
                  <span className="metric-value metric-value-primary">{bestResult.score.toFixed(4)}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">{bestResult.avg_gpu > 0 ? 'Avg GPU Load' : 'Avg CPU Load'}</span>
                  <span className="metric-value">{(bestResult.avg_gpu > 0 ? bestResult.avg_gpu : bestResult.avg_cpu).toFixed(2)}%</span>
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
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <p>No results yet</p>
              <span>Run an attack to see results here.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpongeAttack;
