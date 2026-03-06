import React, { useState, useEffect, useRef } from 'react';
import './SpongeAttack.css';

const ResultCard = ({ title, result, tag }) => (
  <div className="compare-card">
    <div className="compare-card-header">
      <span className="compare-card-title">{title}</span>
      {tag && <span className="compare-card-tag">{tag}</span>}
    </div>
    {result ? (
      <div className="results-content">
        <div className="result-metrics">
          <div className="metric-item">
            <span className="metric-label">Total Score</span>
            <span className="metric-value metric-value-primary">{result.score?.toFixed(4) ?? '—'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">{result.avg_gpu > 0 ? 'Avg GPU Load' : 'Avg CPU Load'}</span>
            <span className="metric-value">{Math.min(result.avg_gpu > 0 ? result.avg_gpu : result.avg_cpu, 100)?.toFixed(2) ?? '—'}%</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Duration</span>
            <span className="metric-value">{result.duration?.toFixed(4) ?? '—'}s</span>
          </div>
          {result.avg_power > 0 && (
            <div className="metric-item">
              <span className="metric-label">Avg Power Draw</span>
              <span className="metric-value">{result.avg_power?.toFixed(1) ?? '—'}W</span>
            </div>
          )}
          {result.energy_joules > 0 && (
            <div className="metric-item">
              <span className="metric-label">Energy Consumed</span>
              <span className="metric-value">{result.energy_joules?.toFixed(1) ?? '—'}J</span>
            </div>
          )}
          <div className="metric-item">
            <span className="metric-label">Input Tokens</span>
            <span className="metric-value">{result.input_tokens ?? '—'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Output Tokens</span>
            <span className="metric-value">{result.output_tokens ?? '—'}</span>
          </div>
        </div>
        <div className="result-data">
          <label className="data-label">Trigger Prompt</label>
          <textarea readOnly value={result.prompt ?? ''} className="data-textarea" />
        </div>
        <div className="result-data">
          <label className="data-label">Model Output</label>
          <textarea readOnly value={result.output ?? ''} className="data-textarea data-textarea-output" />
        </div>
      </div>
    ) : (
      <div className="results-empty">
        <p>Waiting…</p>
      </div>
    )}
  </div>
);

const DeltaBadge = ({ label, regular, quantized, suffix = '', invert = false }) => {
  if (regular == null || quantized == null || regular === 0) return null;
  const diff = quantized - regular;
  const pct = ((diff / Math.abs(regular)) * 100).toFixed(1);
  // For score: higher quantized = green (attack works better). For duration: higher = green too.
  // invert=true flips the colour logic (e.g. for TPS where lower = worse for the defender = "good" for attack)
  const positive = invert ? diff < 0 : diff > 0;
  return (
    <div className={`delta-badge ${positive ? 'delta-positive' : 'delta-negative'}`}>
      <span className="delta-label">{label}</span>
      <span className="delta-value">{diff > 0 ? '+' : ''}{pct}%{suffix}</span>
    </div>
  );
};

const SpongeAttack = () => {
  // ── Single-attack state ──
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState([]);
  const [bestResult, setBestResult] = useState(null);

  // ── Comparison state ──
  const [isComparing, setIsComparing] = useState(false);
  const [comparePhase, setComparePhase] = useState('idle');
  const [regularResult, setRegularResult] = useState(null);
  const [quantizedResult, setQuantizedResult] = useState(null);
  const [regularLogs, setRegularLogs] = useState([]);
  const [quantizedLogs, setQuantizedLogs] = useState([]);

  // ── Controls ──
  const [generations, setGenerations] = useState(5);
  const [population, setPopulation] = useState(10);
  const [selectedModel, setSelectedModel] = useState('gpt2');
  const [attackType, setAttackType] = useState('evolutionary');
  const [numRequests, setNumRequests] = useState(10);
  const [autoDoSIterations, setAutoDoSIterations] = useState(3);
  const [treeDepth, setTreeDepth] = useState(3);
  const [treeBreadth, setTreeBreadth] = useState(4);

  const terminalRef = useRef(null);
  const compareTerminalRef = useRef(null);

  // Models — comparison runs the same model in fp16 vs bitsandbytes NF4 4-bit
  const models = [
    { id: 'gpt2', label: 'GPT-2 Small (124M)', size: '~1 GB', canQuantize: true },
    { id: 'gpt2-medium', label: 'GPT-2 Medium (355M)', size: '~1.5 GB', canQuantize: true },
    { id: 'gpt2-large', label: 'GPT-2 Large (774M)', size: '~3–4 GB', canQuantize: true },
    { id: 'gpt2-xl', label: 'GPT-2 XL (1.5B)', size: '~6–8 GB', canQuantize: true },
    { id: 'facebook/opt-2.7b', label: 'OPT-2.7B', size: '~5 GB', canQuantize: true },
    { id: 'facebook/opt-6.7b', label: 'OPT-6.7B', size: '~13 GB', canQuantize: true },
    { id: 'facebook/opt-13b', label: 'OPT-13B', size: '~25 GB', canQuantize: true },
    { id: 'mistralai/Mistral-7B-v0.1', label: 'Mistral-7B', size: '~14 GB', canQuantize: true },
    { id: 'meta-llama/Llama-2-7b-hf', label: 'LLaMA 2-7B', size: '~14 GB', canQuantize: true },
  ];

  const selectedModelObj = models.find(m => m.id === selectedModel);
  const hasQuantized = selectedModelObj?.canQuantize ?? false;

  // Auto-scroll terminals
  useEffect(() => {
    if (terminalRef.current) terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  }, [logs]);
  useEffect(() => {
    if (compareTerminalRef.current) compareTerminalRef.current.scrollTop = compareTerminalRef.current.scrollHeight;
  }, [regularLogs, quantizedLogs]);

  // Poll single-attack status
  useEffect(() => {
    const iv = setInterval(async () => {
      if (isComparing) return; // don't poll single attack while comparing
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
  }, [isComparing]);

  // Poll comparison status
  useEffect(() => {
    if (!isComparing) return;
    const iv = setInterval(async () => {
      try {
        const res = await fetch('http://localhost:8000/api/attack/compare/status');
        if (res.ok) {
          const data = await res.json();
          setComparePhase(data.phase);
          setRegularResult(data.regular_result);
          setQuantizedResult(data.quantized_result);
          setRegularLogs(data.regular_logs || []);
          setQuantizedLogs(data.quantized_logs || []);
          if (!data.is_running) setIsComparing(false);
        }
      } catch {}
    }, 1000);
    return () => clearInterval(iv);
  }, [isComparing]);

  const anyRunning = isRunning || isComparing;

  const startAttack = async () => {
    try {
      const res = await fetch(
        `http://localhost:8000/api/attack/start?model_id=${encodeURIComponent(selectedModel)}&gens=${generations}&pop=${population}&attack_type=${attackType}&num_requests=${numRequests}&autodos_iterations=${autoDoSIterations}&tree_depth=${treeDepth}&tree_breadth=${treeBreadth}`,
        { method: 'POST' }
      );
      if (res.ok) { setLogs([]); setBestResult(null); }
      else { const e = await res.json(); alert(`Error: ${e.error || 'Failed to start'}`); }
    } catch (e) { console.error(e); }
  };

  const startComparison = async () => {
    if (!hasQuantized) return;
    try {
      const res = await fetch(
        `http://localhost:8000/api/attack/compare?model_id=${encodeURIComponent(selectedModel)}&gens=${generations}&pop=${population}&attack_type=${attackType}&num_requests=${numRequests}&autodos_iterations=${autoDoSIterations}&tree_depth=${treeDepth}&tree_breadth=${treeBreadth}`,
        { method: 'POST' }
      );
      if (res.ok) {
        setIsComparing(true);
        setComparePhase('queued');
        setRegularResult(null);
        setQuantizedResult(null);
        setRegularLogs([]);
        setQuantizedLogs([]);
      } else {
        const e = await res.json();
        alert(`Error: ${e.error || 'Failed to start comparison'}`);
      }
    } catch (e) { console.error(e); }
  };

  const statusClass = status === 'running' ? 'status-running' : status === 'completed' ? 'status-completed' : '';
  const compareStatusClass = comparePhase === 'regular' || comparePhase === 'quantized'
    ? 'status-running'
    : comparePhase === 'complete' ? 'status-completed' : '';
  const compareLogs = [...regularLogs, ...quantizedLogs];

  return (
    <div className="attack-container">

      {/* Controls */}
      <div className="attack-controls">
        <div className="control-group">
          <label className="control-label">Target Model</label>
          <select className="control-input" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} disabled={anyRunning}>
            {models.map(m => <option key={m.id} value={m.id}>{m.label} ({m.size})</option>)}
          </select>
        </div>
        <div className="control-group">
          <label className="control-label">Attack Type</label>
          <select className="control-input" value={attackType} onChange={e => setAttackType(e.target.value)} disabled={anyRunning}>
            <option value="evolutionary">Evolutionary Sponge</option>
            <option value="context_exhaustion">Context Exhaustion</option>
            <option value="autodos">AutoDoS (Tree-based)</option>
          </select>
        </div>
        {attackType === 'evolutionary' ? (
          <>
            <div className="control-group">
              <label className="control-label">Generations</label>
              <input type="number" className="control-input" value={generations} onChange={e => setGenerations(parseInt(e.target.value))} disabled={anyRunning} min="1" />
            </div>
            <div className="control-group">
              <label className="control-label">Population</label>
              <input type="number" className="control-input" value={population} onChange={e => setPopulation(parseInt(e.target.value))} disabled={anyRunning} min="2" />
            </div>
          </>
        ) : attackType === 'context_exhaustion' ? (
          <div className="control-group">
            <label className="control-label">Num Requests</label>
            <input type="number" className="control-input" value={numRequests} onChange={e => setNumRequests(parseInt(e.target.value))} disabled={anyRunning} min="1" />
          </div>
        ) : (
          <>
            <div className="control-group">
              <label className="control-label">Iterations</label>
              <input type="number" className="control-input" value={autoDoSIterations} onChange={e => setAutoDoSIterations(parseInt(e.target.value))} disabled={anyRunning} min="1" />
            </div>
            <div className="control-group">
              <label className="control-label">Tree Depth</label>
              <input type="number" className="control-input" value={treeDepth} onChange={e => setTreeDepth(parseInt(e.target.value))} disabled={anyRunning} min="1" max="10" />
            </div>
            <div className="control-group">
              <label className="control-label">Tree Breadth</label>
              <input type="number" className="control-input" value={treeBreadth} onChange={e => setTreeBreadth(parseInt(e.target.value))} disabled={anyRunning} min="1" max="10" />
            </div>
          </>
        )}
        <button className={`attack-btn${isRunning ? ' attack-btn-running' : ''}`} onClick={startAttack} disabled={anyRunning}>
          {isRunning ? <><span className="btn-spinner" />Running...</> : 'Start Attack'}
        </button>
        <button
          className={`attack-btn attack-btn-compare${isComparing ? ' attack-btn-running' : ''}`}
          onClick={startComparison}
          disabled={anyRunning || !hasQuantized}
          title={hasQuantized ? `Compare ${selectedModelObj.label}: FP16 vs 4-bit` : 'Quantized comparison not available'}
        >
          {isComparing ? <><span className="btn-spinner" />Comparing...</> : 'Compare FP16 vs 4-bit'}
        </button>
      </div>

      {/* ── Single Attack View ── */}
      {!isComparing && comparePhase === 'idle' && (
        <>
          {/* Status */}
          <div className="attack-status-bar">
            <div className="attack-status-left">
              <span className={`attack-status-indicator ${statusClass}`} />
              <span className="attack-status-label">{status.toUpperCase()}</span>
            </div>
            {bestResult && (
              <span className="attack-status-score">Best score: <strong>{bestResult.score?.toFixed(4)}</strong></span>
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
                      <span className="metric-value metric-value-primary">{bestResult.score?.toFixed(4)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">{bestResult.avg_gpu > 0 ? 'Avg GPU Load' : 'Avg CPU Load'}</span>
                      <span className="metric-value">{(bestResult.avg_gpu > 0 ? bestResult.avg_gpu : bestResult.avg_cpu)?.toFixed(2)}%</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Duration</span>
                      <span className="metric-value">{bestResult.duration?.toFixed(4)}s</span>
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
        </>
      )}

      {/* ── Comparison View ── */}
      {(isComparing || comparePhase !== 'idle') && (
        <>
          {/* Status */}
          <div className="attack-status-bar">
            <div className="attack-status-left">
              <span className={`attack-status-indicator ${compareStatusClass}`} />
              <span className="attack-status-label">
                {comparePhase === 'regular' && `PHASE 1/2 — ${selectedModelObj?.label || 'REGULAR'}`}
                {comparePhase === 'quantized' && `PHASE 2/2 — ${selectedModelObj?.label || 'MODEL'} (4-bit)`}
                {comparePhase === 'complete' && 'COMPARISON COMPLETE'}
                {comparePhase === 'queued' && 'QUEUED'}
                {comparePhase === 'error' && 'ERROR'}
                {comparePhase === 'idle' && 'IDLE'}
              </span>
            </div>
          </div>

          {/* Comparison Terminal */}
          <div className="attack-terminal compare-terminal">
            <div className="terminal-header">
              <div className="terminal-dots">
                <span className="dot dot-red" /><span className="dot dot-yellow" /><span className="dot dot-green" />
              </div>
              <span className="terminal-title">Comparison Log</span>
            </div>
            <div className="terminal-body" ref={compareTerminalRef}>
              {compareLogs.length === 0 ? (
                <div className="terminal-empty"><span className="terminal-prompt">$</span>Waiting for comparison to start...</div>
              ) : (
                compareLogs.map((log, i) => (
                  <div key={i} className="terminal-line">
                    <span className="terminal-line-num">{String(i + 1).padStart(3, ' ')}</span>
                    {log}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Delta badges */}
          {regularResult && quantizedResult && (
            <div className="delta-row">
              <DeltaBadge label="Score" regular={regularResult.score} quantized={quantizedResult.score} />
              <DeltaBadge label="Duration" regular={regularResult.duration} quantized={quantizedResult.duration} />
              <DeltaBadge label="CPU Load" regular={regularResult.avg_cpu} quantized={quantizedResult.avg_cpu} />
              {regularResult.avg_gpu > 0 && (
                <DeltaBadge label="GPU Load" regular={regularResult.avg_gpu} quantized={quantizedResult.avg_gpu} />
              )}
              {regularResult.avg_power > 0 && (
                <DeltaBadge label="Power Draw" regular={regularResult.avg_power} quantized={quantizedResult.avg_power} />
              )}
              {regularResult.energy_joules > 0 && (
                <DeltaBadge label="Energy (J)" regular={regularResult.energy_joules} quantized={quantizedResult.energy_joules} />
              )}
            </div>
          )}

          {/* Side-by-side results */}
          <div className="compare-grid">
            <ResultCard title="Regular" result={regularResult} tag={regularResult?.quant_label ?? 'fp16'} />
            <ResultCard title="Quantized (int4)" result={quantizedResult} tag={quantizedResult?.quant_label ?? 'int4'} />
          </div>
        </>
      )}
    </div>
  );
};

export default SpongeAttack;
