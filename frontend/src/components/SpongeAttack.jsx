import React, { useState, useEffect, useRef, useMemo } from 'react';
import API_BASE from '../api';
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
            <span className="metric-label">Ukupni rezultat</span>
            <span className="metric-value metric-value-primary">{result.score?.toFixed(4) ?? '—'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">{result.context_mode ? 'Ukupno trajanje' : 'Trajanje'}</span>
            <span className="metric-value">{result.duration?.toFixed(4) ?? '—'}s</span>
          </div>
          {result.avg_power > 0 && (
            <div className="metric-item">
              <span className="metric-label">Prosjecna snaga</span>
              <span className="metric-value">{result.avg_power?.toFixed(1) ?? '—'}W</span>
            </div>
          )}
          {result.energy_joules > 0 && (
            <div className="metric-item">
              <span className="metric-label">Energija</span>
              <span className="metric-value">{result.energy_joules?.toFixed(1) ?? '—'}J</span>
            </div>
          )}
        </div>
        <div className="result-data">
          <label className="data-label">Prompt</label>
          <textarea readOnly value={result.prompt ?? ''} className="data-textarea" />
        </div>
        <div className="result-data">
          <label className="data-label">Izlaz</label>
          <textarea readOnly value={result.output ?? ''} className="data-textarea data-textarea-output" />
        </div>
      </div>
    ) : (
      <div className="results-empty"><p>Cekam…</p></div>
    )}
  </div>
);

const DeltaBadge = ({ label, regular, quantized }) => {
  if (regular == null || quantized == null || regular === 0) return null;
  const diff = quantized - regular;
  const pct = ((diff / Math.abs(regular)) * 100).toFixed(1);
  const positive = diff > 0;
  return (
    <div className={`delta-badge ${positive ? 'delta-positive' : 'delta-negative'}`}>
      <span className="delta-label">{label}</span>
      <span className="delta-value">{diff > 0 ? '+' : ''}{pct}%</span>
    </div>
  );
};

const PhasePanel = ({
  phaseLabel,
  familyId,
  onFamilyChange,
  backend,
  onBackendChange,
  ggufVariant,
  onGgufVariantChange,
  families,
  disabled,
}) => {
  const family = families.find(f => f.id === familyId) || families[0];
  const ggufOptions = family?.gguf_variants || [];
  const firstAvailable = ggufOptions.find(v => v.available)?.id || ggufOptions[0]?.id || 'q4_k_m';

  useEffect(() => {
    if (backend !== 'gguf') return;
    const current = ggufOptions.find(v => v.id === ggufVariant);
    if (!current?.available && firstAvailable) {
      onGgufVariantChange(firstAvailable);
    }
  }, [familyId, backend, ggufOptions, ggufVariant, firstAvailable, onGgufVariantChange]);

  if (!family) {
    return <div className="phase-panel phase-panel-empty">Ucitavam modele…</div>;
  }

  return (
    <div className="phase-panel">
      <div className="phase-panel-header">
        <span className="phase-badge">{phaseLabel}</span>
        <span className="phase-panel-title">Konfiguracija modela</span>
      </div>

      <div className="phase-field">
        <label className="control-label">Model</label>
        <select
          className="control-input control-input-wide"
          value={familyId}
          onChange={e => onFamilyChange(e.target.value)}
          disabled={disabled}
        >
          {families.map(f => (
            <option key={f.id} value={f.id}>{f.label}</option>
          ))}
        </select>
      </div>

      <div className="phase-field">
        <label className="control-label">Metoda kvantizacije</label>
        <div className="backend-toggle">
          <button
            type="button"
            className={`backend-btn ${backend === 'gguf' ? 'backend-btn-active' : ''}`}
            onClick={() => onBackendChange('gguf')}
            disabled={disabled}
          >
            GGUF (llama.cpp)
          </button>
          <button
            type="button"
            className={`backend-btn ${backend === 'gptq' ? 'backend-btn-active' : ''}`}
            onClick={() => onBackendChange('gptq')}
            disabled={disabled || !family.gptq?.available}
            title={family.gptq?.error || ''}
          >
            GPTQ (GPTQModel)
          </button>
          <button
            type="button"
            className={`backend-btn ${backend === 'hf' ? 'backend-btn-active' : ''}`}
            onClick={() => onBackendChange('hf')}
            disabled={disabled || !family.hf?.available}
            title={family.hf?.error || ''}
          >
            HF FP16 (transformers)
          </button>
        </div>
      </div>

      {backend === 'gguf' ? (
        <div className="phase-field">
          <label className="control-label">GGUF varijanta</label>
          <select
            className="control-input control-input-wide"
            value={ggufVariant}
            onChange={e => onGgufVariantChange(e.target.value)}
            disabled={disabled}
          >
            {ggufOptions.map(v => (
              <option key={v.id} value={v.id} disabled={!v.available}>
                {v.label}{!v.available ? ' — nema datoteke' : ''}
              </option>
            ))}
          </select>
        </div>
      ) : backend === 'gptq' ? (
        <div className="phase-field phase-gptq-info">
          <span className="control-label">Preciznost</span>
          <p className="gptq-fixed">4-bit (GPTQModel)</p>
          {family.gptq?.repo && (
            <p className="phase-hint">{family.gptq.repo}</p>
          )}
          {!family.gptq?.available && (
            <p className="phase-hint phase-hint-warn">{family.gptq?.error || 'GPTQ nije dostupan'}</p>
          )}
        </div>
      ) : (
        <div className="phase-field phase-gptq-info">
          <span className="control-label">Preciznost</span>
          <p className="gptq-fixed">FP16 (transformers)</p>
          {family.hf?.repo && (
            <p className="phase-hint">{family.hf.repo}</p>
          )}
          {!family.hf?.available && (
            <p className="phase-hint phase-hint-warn">{family.hf?.error || 'HF FP16 nije dostupan'}</p>
          )}
        </div>
      )}
    </div>
  );
};

const SpongeAttack = () => {
  const [catalog, setCatalog] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [comparePhase, setComparePhase] = useState('idle');
  const [regularResult, setRegularResult] = useState(null);
  const [quantizedResult, setQuantizedResult] = useState(null);
  const [regularLogs, setRegularLogs] = useState([]);
  const [quantizedLogs, setQuantizedLogs] = useState([]);
  const [phaseADisplay, setPhaseADisplay] = useState('');
  const [phaseBDisplay, setPhaseBDisplay] = useState('');

  const [familyA, setFamilyA] = useState('mistral7b');
  const [familyB, setFamilyB] = useState('mistral7b');
  const [backendA, setBackendA] = useState('gguf');
  const [backendB, setBackendB] = useState('gptq');
  const [variantA, setVariantA] = useState('f16');
  const [variantB, setVariantB] = useState('q4_k_m');

  const [generations, setGenerations] = useState(5);
  const [population, setPopulation] = useState(10);
  const [attackType, setAttackType] = useState('context_exhaustion');
  const [numRequests, setNumRequests] = useState(1);
  const [autoDoSIterations, setAutoDoSIterations] = useState(3);
  const [treeDepth, setTreeDepth] = useState(3);
  const [treeBreadth, setTreeBreadth] = useState(4);
  const [contextMode, setContextMode] = useState('combined');

  const compareTerminalRef = useRef(null);
  const families = catalog?.families || [];

  const phaseAReady = useMemo(() => {
    const fam = families.find(f => f.id === familyA);
    if (!fam) return false;
    if (backendA === 'gptq') return fam.gptq?.available;
    if (backendA === 'hf') return fam.hf?.available;
    return fam.gguf_variants?.some(v => v.id === variantA && v.available);
  }, [families, familyA, backendA, variantA]);

  const phaseBReady = useMemo(() => {
    const fam = families.find(f => f.id === familyB);
    if (!fam) return false;
    if (backendB === 'gptq') return fam.gptq?.available;
    if (backendB === 'hf') return fam.hf?.available;
    return fam.gguf_variants?.some(v => v.id === variantB && v.available);
  }, [families, familyB, backendB, variantB]);

  const runDisabledReason = () => {
    if (!catalog) return 'Ucitavam katalog modela…';
    if (!phaseAReady) return 'Faza A: provjeri model i kvantizaciju';
    if (!phaseBReady) return 'Faza B: provjeri model i kvantizaciju';
    return '';
  };

  useEffect(() => {
    if (compareTerminalRef.current) {
      compareTerminalRef.current.scrollTop = compareTerminalRef.current.scrollHeight;
    }
  }, [regularLogs, quantizedLogs]);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/models/catalog`);
        if (res.ok) {
          const data = await res.json();
          setCatalog(data);
          if (data.families?.length) {
            const ids = data.families.map(f => f.id);
            if (!ids.includes(familyA)) setFamilyA(data.families[0].id);
            if (!ids.includes(familyB)) setFamilyB(data.families[0].id);
          }
        }
      } catch { /* backend offline */ }
    };
    load();
    const iv = setInterval(load, 10000);
    return () => clearInterval(iv);
  }, [familyA, familyB]);

  useEffect(() => {
    if (!isComparing) return;
    const iv = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/attack/compare/status`);
        if (res.ok) {
          const data = await res.json();
          setComparePhase(data.phase);
          setRegularResult(data.regular_result);
          setQuantizedResult(data.quantized_result);
          setRegularLogs(data.regular_logs || []);
          setQuantizedLogs(data.quantized_logs || []);
          if (data.phase_a_display) setPhaseADisplay(data.phase_a_display);
          if (data.phase_b_display) setPhaseBDisplay(data.phase_b_display);
          if (!data.is_running) setIsComparing(false);
        }
      } catch { /* ignore */ }
    }, 1000);
    return () => clearInterval(iv);
  }, [isComparing]);

  const startComparison = async () => {
    if (!phaseAReady || !phaseBReady) return;
    const params = new URLSearchParams({
      model_family_a: familyA,
      model_family_b: familyB,
      phase_a_backend: backendA,
      phase_a_gguf_variant: variantA,
      phase_b_backend: backendB,
      phase_b_gguf_variant: variantB,
      gens: String(generations),
      pop: String(population),
      attack_type: attackType,
      num_requests: String(numRequests),
      autodos_iterations: String(autoDoSIterations),
      tree_depth: String(treeDepth),
      tree_breadth: String(treeBreadth),
      context_mode: contextMode,
    });
    try {
      const res = await fetch(`${API_BASE}/api/attack/compare?${params}`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        setIsComparing(true);
        setComparePhase('queued');
        setRegularResult(null);
        setQuantizedResult(null);
        setRegularLogs([]);
        setQuantizedLogs([]);
        if (data.phase_a) setPhaseADisplay(data.phase_a);
        if (data.phase_b) setPhaseBDisplay(data.phase_b);
      } else {
        alert(`Greska: ${data.error || 'Neuspjelo pokretanje'}`);
      }
    } catch (e) {
      console.error(e);
      alert('Backend nije dostupan na portu 8000');
    }
  };

  const compareStatusClass =
    comparePhase === 'regular' || comparePhase === 'quantized'
      ? 'status-running'
      : comparePhase === 'complete'
        ? 'status-completed'
        : '';
  const compareLogs = [...regularLogs, ...quantizedLogs];
  const anyRunning = isComparing;

  const famLabel = (id) => families.find(f => f.id === id)?.label || id;

  return (
    <div className="attack-container">
      <div className="console-intro">
        <p>
          Usporedi dva profila: odaberi <strong>model</strong>, zatim <strong>metodu kvantizacije</strong>
          (GGUF s preciznoscu, GPTQ 4-bit ili HF FP16), pa pokreni napad.
        </p>
        {catalog?.gguf_dir && (
          <p className="console-intro-meta">GGUF mapa: {catalog.gguf_dir}</p>
        )}
        {catalog?.hf_dir && (
          <p className="console-intro-meta">HF mapa: {catalog.hf_dir}</p>
        )}
      </div>

      <div className="console-grid">
        <PhasePanel
          phaseLabel="A"
          familyId={familyA}
          onFamilyChange={setFamilyA}
          backend={backendA}
          onBackendChange={setBackendA}
          ggufVariant={variantA}
          onGgufVariantChange={setVariantA}
          families={families}
          disabled={anyRunning}
        />
        <PhasePanel
          phaseLabel="B"
          familyId={familyB}
          onFamilyChange={setFamilyB}
          backend={backendB}
          onBackendChange={setBackendB}
          ggufVariant={variantB}
          onGgufVariantChange={setVariantB}
          families={families}
          disabled={anyRunning}
        />
        <div className="phase-panel attack-panel">
          <div className="phase-panel-header">
            <span className="phase-badge phase-badge-attack">⚡</span>
            <span className="phase-panel-title">Parametri napada</span>
          </div>

          <div className="phase-field">
            <label className="control-label">Scenarij</label>
            <select className="control-input control-input-wide" value={attackType} onChange={e => setAttackType(e.target.value)} disabled={anyRunning}>
              <option value="evolutionary">Evolucijski Sponge</option>
              <option value="context_exhaustion">Iscrpljivanje konteksta</option>
              <option value="autodos">AutoDoS</option>
              <option value="token_busting">Razbijanje tokena</option>
              <option value="lingoloop">LingoLoop</option>
              <option value="state_entrapment">Zarobljavanje stanja</option>
            </select>
          </div>

          {attackType === 'evolutionary' ? (
            <div className="attack-params-row">
              <div className="phase-field">
                <label className="control-label">Generacije</label>
                <input type="number" className="control-input" value={generations} onChange={e => setGenerations(parseInt(e.target.value, 10) || 1)} disabled={anyRunning} min="1" />
              </div>
              <div className="phase-field">
                <label className="control-label">Populacija</label>
                <input type="number" className="control-input" value={population} onChange={e => setPopulation(parseInt(e.target.value, 10) || 2)} disabled={anyRunning} min="2" />
              </div>
            </div>
          ) : (attackType === 'context_exhaustion' || attackType === 'token_busting' || attackType === 'lingoloop' || attackType === 'state_entrapment') ? (
            <>
              <div className="phase-field">
                <label className="control-label">Broj zahtjeva</label>
                <input type="number" className="control-input" value={numRequests} onChange={e => setNumRequests(parseInt(e.target.value, 10) || 1)} disabled={anyRunning} min="1" />
              </div>
              {attackType === 'context_exhaustion' && (
                <div className="phase-field">
                  <label className="control-label">Nacin mjerenja</label>
                  <select className="control-input control-input-wide" value={contextMode} onChange={e => setContextMode(e.target.value)} disabled={anyRunning}>
                    <option value="combined">Combined (prefill + decode)</option>
                    <option value="prefill_only">Prefill only</option>
                  </select>
                </div>
              )}
            </>
          ) : (
            <div className="attack-params-row">
              <div className="phase-field">
                <label className="control-label">Iteracije</label>
                <input type="number" className="control-input" value={autoDoSIterations} onChange={e => setAutoDoSIterations(parseInt(e.target.value, 10) || 1)} disabled={anyRunning} min="1" />
              </div>
              <div className="phase-field">
                <label className="control-label">Dubina</label>
                <input type="number" className="control-input" value={treeDepth} onChange={e => setTreeDepth(parseInt(e.target.value, 10) || 1)} disabled={anyRunning} min="1" max="10" />
              </div>
              <div className="phase-field">
                <label className="control-label">Sirina</label>
                <input type="number" className="control-input" value={treeBreadth} onChange={e => setTreeBreadth(parseInt(e.target.value, 10) || 1)} disabled={anyRunning} min="1" max="10" />
              </div>
            </div>
          )}

          <div className="run-summary">
            <div className="run-summary-row"><span className="plan-badge">A</span> {famLabel(familyA)} · {backendA === 'gptq' ? 'GPTQ 4-bit' : variantA.toUpperCase()}</div>
            <div className="run-summary-row"><span className="plan-badge">B</span> {famLabel(familyB)} · {backendB === 'gptq' ? 'GPTQ 4-bit' : variantB.toUpperCase()}</div>
            {runDisabledReason() && (
              <p className="phase-hint phase-hint-warn">{runDisabledReason()}</p>
            )}
          </div>

          <button
            type="button"
            className={`attack-btn attack-btn-compare${isComparing ? ' attack-btn-running' : ''}`}
            onClick={startComparison}
            disabled={anyRunning || !phaseAReady || !phaseBReady}
            title={runDisabledReason() || 'Pokreni A/B usporedbu'}
          >
            {isComparing ? <><span className="btn-spinner" />U tijeku…</> : 'Pokreni A/B usporedbu'}
          </button>
        </div>
      </div>

      <div className="attack-status-bar">
        <div className="attack-status-left">
          <span className={`attack-status-indicator ${compareStatusClass}`} />
          <span className="attack-status-label">
            {comparePhase === 'regular' && `Faza 1/2 — ${phaseADisplay || famLabel(familyA)}`}
            {comparePhase === 'quantized' && `Faza 2/2 — ${phaseBDisplay || famLabel(familyB)}`}
            {comparePhase === 'complete' && 'Usporedba zavrsena'}
            {comparePhase === 'queued' && 'U redu…'}
            {comparePhase === 'error' && 'Greska'}
            {comparePhase === 'idle' && 'Spremno'}
          </span>
        </div>
      </div>

      <div className="attack-terminal compare-terminal">
        <div className="terminal-header">
          <div className="terminal-dots">
            <span className="dot dot-red" /><span className="dot dot-yellow" /><span className="dot dot-green" />
          </div>
          <span className="terminal-title">Dnevnik</span>
        </div>
        <div className="terminal-body" ref={compareTerminalRef}>
          {compareLogs.length === 0 ? (
            <div className="terminal-empty"><span className="terminal-prompt">$</span> odaberi profile i pokreni usporedbu</div>
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

      {regularResult && quantizedResult && (
        <div className="delta-row">
          <DeltaBadge label="Rezultat" regular={regularResult.score} quantized={quantizedResult.score} />
          <DeltaBadge label="Trajanje" regular={regularResult.duration} quantized={quantizedResult.duration} />
        </div>
      )}

      <div className="compare-grid">
        <ResultCard title={`A — ${phaseADisplay || famLabel(familyA)}`} result={regularResult} tag={regularResult?.quant_label} />
        <ResultCard title={`B — ${phaseBDisplay || famLabel(familyB)}`} result={quantizedResult} tag={quantizedResult?.quant_label} />
      </div>
    </div>
  );
};

export default SpongeAttack;
