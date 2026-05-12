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
            <span className="metric-label">Input Chars</span>
            <span className="metric-value">{result.input_chars ?? '—'}</span>
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
  const [selectedModelA, setSelectedModelA] = useState('gpt2');
  const [selectedModelB, setSelectedModelB] = useState('gpt2');
  const [attackType, setAttackType] = useState('evolutionary');
  const [numRequests, setNumRequests] = useState(10);
  const [autoDoSIterations, setAutoDoSIterations] = useState(3);
  const [treeDepth, setTreeDepth] = useState(3);
  const [treeBreadth, setTreeBreadth] = useState(4);
  const [quantCapabilities, setQuantCapabilities] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);

  const compareTerminalRef = useRef(null);

  // Models — prefer local GGUF files when available.
  const fallbackModels = [
    { id: 'gpt2', label: 'GPT-2 (GGUF)', size: 'local', canQuantize: true },
    { id: 'mistral7b', label: 'Mistral-7B (GGUF)', size: 'local', canQuantize: true },
    { id: 'opt-6.7b', label: 'OPT-6.7B (GGUF)', size: 'local', canQuantize: true },
  ];

  const models = availableModels.length > 0 ? availableModels : fallbackModels;

  const selectedModelObjA = models.find(m => m.id === selectedModelA);
  const selectedModelObjB = models.find(m => m.id === selectedModelB);
  const hasQuantized = Boolean(selectedModelObjA?.canQuantize && selectedModelObjB?.canQuantize);

  const isModeSupported = (modeId) => {
    if (!quantCapabilities?.modes) return true;
    return quantCapabilities.modes[modeId]?.supported !== false;
  };

  const modeReason = (modeId) => {
    if (!quantCapabilities?.modes) return '';
    return quantCapabilities.modes[modeId]?.reason || '';
  };

  // Auto-scroll terminal
  useEffect(() => {
    if (compareTerminalRef.current) compareTerminalRef.current.scrollTop = compareTerminalRef.current.scrollHeight;
  }, [regularLogs, quantizedLogs]);

  // Fetch hardware/runtime capabilities and refresh periodically.
  useEffect(() => {
    const fetchCapabilities = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/capabilities');
        if (!res.ok) return;
        const data = await res.json();
        const caps = data?.quantization || null;
        setQuantCapabilities(caps);

        // Auto-adjust selections if currently unsupported.
        if (caps?.modes && caps.modes['gguf-f16']?.supported === false) {
          setQuantCapabilities(caps);
        }
      } catch {}
    };

    fetchCapabilities();
    const iv = setInterval(fetchCapabilities, 5000);
    const onFocus = () => fetchCapabilities();
    window.addEventListener('focus', onFocus);

    return () => {
      clearInterval(iv);
      window.removeEventListener('focus', onFocus);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch local GGUF file list for model selection.
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/gguf/list');
        if (!res.ok) return;
        const data = await res.json();
        const files = Array.isArray(data?.files) ? data.files : [];
        if (!files.length) return;

        const mapped = files.map(file => ({
          id: `gguf:${file.path}`,
          label: file.name,
          size: file.size_gb != null ? `${file.size_gb} GB` : 'local',
          canQuantize: true,
        }));
        setAvailableModels(mapped);

        if (!mapped.find(m => m.id === selectedModelA)) {
          setSelectedModelA(mapped[0].id);
        }
        if (!mapped.find(m => m.id === selectedModelB)) {
          setSelectedModelB(mapped[0].id);
        }
      } catch {}
    };

    fetchModels();
    const iv = setInterval(fetchModels, 10000);
    return () => clearInterval(iv);
  }, [selectedModelA, selectedModelB]);

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

  const anyRunning = isComparing;

  const startComparison = async () => {
    if (!hasQuantized) return;
    try {
      const res = await fetch(
        `http://localhost:8000/api/attack/compare?model_id_a=${encodeURIComponent(selectedModelA)}&model_id_b=${encodeURIComponent(selectedModelB)}&gens=${generations}&pop=${population}&attack_type=${attackType}&num_requests=${numRequests}&autodos_iterations=${autoDoSIterations}&tree_depth=${treeDepth}&tree_breadth=${treeBreadth}`,
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

  const compareStatusClass = comparePhase === 'regular' || comparePhase === 'quantized'
    ? 'status-running'
    : comparePhase === 'complete' ? 'status-completed' : '';
  const compareLogs = [...regularLogs, ...quantizedLogs];

  return (
    <div className="attack-container">

      {/* Controls */}
      <div className="attack-controls">
        <div className="control-group">
          <label className="control-label">Model Under Test (A)</label>
          <select className="control-input" value={selectedModelA} onChange={e => setSelectedModelA(e.target.value)} disabled={anyRunning}>
            {models.map(m => <option key={m.id} value={m.id}>{m.label} ({m.size})</option>)}
          </select>
        </div>
        <div className="control-group">
          <label className="control-label">Model Under Test (B)</label>
          <select className="control-input" value={selectedModelB} onChange={e => setSelectedModelB(e.target.value)} disabled={anyRunning}>
            {models.map(m => <option key={m.id} value={m.id}>{m.label} ({m.size})</option>)}
          </select>
        </div>
        <div className="control-group">
          <label className="control-label">Attack Scenario</label>
          <select className="control-input" value={attackType} onChange={e => setAttackType(e.target.value)} disabled={anyRunning}>
            <option value="evolutionary">Evolutionary Sponge</option>
            <option value="context_exhaustion">Context Exhaustion</option>
            <option value="autodos">AutoDoS (Tree-based)</option>
            <option value="token_busting">Token-Busting</option>
            <option value="lingoloop">LingoLoop</option>
            <option value="state_entrapment">State Entrapment</option>
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
        ) : (attackType === 'context_exhaustion' || attackType === 'token_busting' || attackType === 'lingoloop' || attackType === 'state_entrapment') ? (
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
        <button
          className={`attack-btn attack-btn-compare${isComparing ? ' attack-btn-running' : ''}`}
          onClick={startComparison}
          disabled={anyRunning || !hasQuantized || !isModeSupported('gguf-f16')}
          title={
            !hasQuantized
              ? 'Quantized comparison not available'
              : !isModeSupported('gguf-f16')
                ? modeReason('gguf-f16')
                : `Compare ${selectedModelObjA?.label || 'Model A'} vs ${selectedModelObjB?.label || 'Model B'}`
          }
        >
          {isComparing ? <><span className="btn-spinner" />Comparing A vs B...</> : 'Run A/B Comparison'}
        </button>
      </div>

      {/* ── Comparison View ── */}
      {
        <>
          {/* Status */}
          <div className="attack-status-bar">
            <div className="attack-status-left">
              <span className={`attack-status-indicator ${compareStatusClass}`} />
              <span className="attack-status-label">
                {comparePhase === 'regular' && `PHASE 1/2 — ${selectedModelObjA?.label || 'MODEL A'}`}
                {comparePhase === 'quantized' && `PHASE 2/2 — ${selectedModelObjB?.label || 'MODEL B'}`}
                {comparePhase === 'complete' && 'COMPARISON COMPLETE'}
                {comparePhase === 'queued' && 'QUEUED'}
                {comparePhase === 'error' && 'ERROR'}
                {comparePhase === 'idle' && 'IDLE'}
              </span>
            </div>
            <span className="attack-status-score compare-legend">A = Model A, B = Model B</span>
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
              <div className="delta-caption">Delta shown as B vs A</div>
              <DeltaBadge label="Score" regular={regularResult.score} quantized={quantizedResult.score} />
              <DeltaBadge label="Duration" regular={regularResult.duration} quantized={quantizedResult.duration} />
            
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
            <ResultCard title={`A: ${selectedModelObjA?.label || 'Model A'}`} result={regularResult} tag={regularResult?.quant_label ?? 'A'} />
            <ResultCard title={`B: ${selectedModelObjB?.label || 'Model B'}`} result={quantizedResult} tag={quantizedResult?.quant_label ?? 'B'} />
          </div>
        </>
      }
    </div>
  );
};

export default SpongeAttack;
