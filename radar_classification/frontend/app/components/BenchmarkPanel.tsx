'use client';




























import React, { useState } from 'react';
import { Play, Square, BarChart3, Loader2 } from 'lucide-react';
import {
  SUBTYPE_CLASS, SUBTYPE_PRESETS, type TargetSubtype,
  computeFeatureVectors, type KinematicSample,
} from './RadarSimulator';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8081';
const ALL_SUBTYPES = Object.keys(SUBTYPE_PRESETS) as TargetSubtype[];
const CLASSES: KinematicSample['type'][] = ['bird', 'drone', 'aircraft', 'anomaly'];


const SNR_PRESETS = [
  { name: 'Border', db: -10.0 },
  { name: 'Night', db: -10.0 },
  { name: 'Airport', db: -12.0 },
  { name: 'Urban', db: -14.5 },
  { name: 'Forest', db: -17.0 },
  { name: 'Heavy Noise', db: -23.0 },
].sort((a, b) => b.db - a.db); 

const FEAT11_NAMES = [
  'doppler_velocity', 'doppler_freq_shift', 'rcs_estimate', 'micro_doppler_bandwidth',
  'micro_doppler_jitter', 'range_m', 'azimuth_deg', 'altitude_m', 'velocity_variance',
  'acceleration', 'snr_db',
];
const FEAT31_NAMES = [
  'snr_db',
  'amplitude_mean', 'amplitude_std', 'amplitude_min', 'amplitude_max', 'amplitude_range',
  'amplitude_energy', 'amplitude_zcr', 'amplitude_dominant_freq', 'amplitude_spectral_bandwidth', 'amplitude_diff_var',
  'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'velocity_range',
  'velocity_energy', 'velocity_zcr', 'velocity_dominant_freq', 'velocity_spectral_bandwidth', 'velocity_diff_var',
  'energy_mean', 'energy_std', 'energy_min', 'energy_max', 'energy_range',
  'energy_energy', 'energy_zcr', 'energy_dominant_freq', 'energy_spectral_bandwidth', 'energy_diff_var',
];
const FEAT42_NAMES = [...FEAT11_NAMES, ...FEAT31_NAMES.slice(1)];

interface Variant {
  key: 'feat11' | 'feat31' | 'feat42' | 'quantum' | 'ensemble';
  label: string;
  color: string;
  supports4Class: boolean;
}
const MODEL_VARIANTS: Variant[] = [
  { key: 'feat11', label: '11-Feature (Classic)', color: '#10b981', supports4Class: true },
  { key: 'feat31', label: '31-Feature', color: '#38bdf8', supports4Class: true },
  { key: 'feat42', label: '42-Feature', color: '#f59e0b', supports4Class: true },
  { key: 'quantum', label: 'Quantum', color: '#a78bfa', supports4Class: false },
  { key: 'ensemble', label: 'Quantum + Ensemble', color: '#fb7185', supports4Class: false },
];

interface Cell {
  droneRecall: number;      
  accuracy: number | null;  
  nDrone: number;
  n: number;
}
type Results = Record<Variant['key'], Record<string, Cell>>;








function spawnSampleOfClass(cls: KinematicSample['type']): KinematicSample {
  const candidates = ALL_SUBTYPES.filter(s => SUBTYPE_CLASS[s] === cls);
  const subtype = candidates[Math.floor(Math.random() * candidates.length)];
  const preset = SUBTYPE_PRESETS[subtype];
  const angle = Math.random() * Math.PI * 2;
  const dist = 300 + Math.random() * 4000;
  const speed = preset.speedMin + Math.random() * (preset.speedMax - preset.speedMin);
  const vAngle = angle + Math.PI + (Math.random() - 0.5) * 0.8;
  return {
    x: Math.cos(angle) * dist,
    y: Math.sin(angle) * dist,
    vx: Math.cos(vAngle) * speed,
    vy: Math.sin(vAngle) * speed,
    ax: 0, ay: 0,
    altitude: preset.altMin + Math.random() * (preset.altMax - preset.altMin),
    baseRcs: preset.rcs + (Math.random() - 0.5) * 2,
    baseUdbw: preset.udbw * (0.85 + Math.random() * 0.3),
    baseUdjit: Math.min(1, Math.max(0, preset.udjit + (Math.random() - 0.5) * 0.08)),
    baseVelvar: preset.velvar * (0.8 + Math.random() * 0.4),
    type: cls,
  };
}

async function callClassicEndpoint(endpoint: string, names: string[], values: number[]): Promise<string> {
  const payload: Record<string, number> = {};
  names.forEach((n, i) => { payload[n] = values[i]; });
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`${endpoint} responded ${res.status}`);
  const data = await res.json();
  return String(data.label);
}

async function callQuantumEndpoint(features11: number[], useEnsemble: boolean): Promise<'drone' | 'not_drone'> {
  const payload = {
    doppler_velocity: features11[0],
    rcs_estimate: features11[2],
    micro_doppler_bandwidth: features11[3],
    micro_doppler_jitter: features11[4],
    use_ensemble: useEnsemble,
    mode: useEnsemble ? 'ensemble' : 'quantum',
  };
  const res = await fetch(`${API_BASE}/predict/quantum`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`/predict/quantum responded ${res.status}`);
  const data = await res.json();
  return data.prediction === 'drone' ? 'drone' : 'not_drone';
}

export default function BenchmarkPanel({ theme }: { theme: 'dark' | 'light' }) {
  const isDark = theme === 'dark';
  const [samplesPerClass, setSamplesPerClass] = useState(5);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0, label: '' });
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = React.useRef(false);

  const runBenchmark = async () => {
    setRunning(true);
    setError(null);
    cancelRef.current = false;
    const newResults: Results = { feat11: {}, feat31: {}, feat42: {}, quantum: {}, ensemble: {} };
    const total = MODEL_VARIANTS.length * SNR_PRESETS.length;
    let done = 0;
    setProgress({ done: 0, total, label: 'Starting…' });

    try {
      for (const variant of MODEL_VARIANTS) {
        for (const snr of SNR_PRESETS) {
          if (cancelRef.current) throw new Error('cancelled');
          setProgress({ done, total, label: `${variant.label} @ ${snr.name} (${snr.db} dB)` });

          const samples = CLASSES.flatMap(cls =>
            Array.from({ length: samplesPerClass }, () => ({ cls, sample: spawnSampleOfClass(cls) }))
          );

          let dronePredictedCorrect = 0, droneActual = 0, correct4 = 0, counted4 = 0;

          await Promise.all(samples.map(async ({ cls, sample }) => {
            const { featureVector11, featureVector31 } = computeFeatureVectors(sample, snr.db);
            let predictedIsDrone: boolean;
            let predictedIsCorrect4: boolean | null = null;

            if (variant.key === 'quantum' || variant.key === 'ensemble') {
              const pred = await callQuantumEndpoint(featureVector11, variant.key === 'ensemble');
              predictedIsDrone = pred === 'drone';
            } else {
              const endpoint = variant.key === 'feat11' ? '/predict/classic'
                : variant.key === 'feat31' ? '/predict/31' : '/predict/42';
              const names = variant.key === 'feat11' ? FEAT11_NAMES
                : variant.key === 'feat31' ? FEAT31_NAMES : FEAT42_NAMES;
              const values = variant.key === 'feat11' ? featureVector11
                : variant.key === 'feat31' ? featureVector31
                  : [...featureVector11, ...featureVector31.slice(1)];
              const predictedLabel = await callClassicEndpoint(endpoint, names, values);
              predictedIsDrone = predictedLabel === 'drone';
              predictedIsCorrect4 = predictedLabel === cls;
            }

            if (cls === 'drone') {
              droneActual += 1;
              if (predictedIsDrone) dronePredictedCorrect += 1;
            }
            if (predictedIsCorrect4 !== null) {
              counted4 += 1;
              if (predictedIsCorrect4) correct4 += 1;
            }
          }));

          newResults[variant.key][snr.name] = {
            droneRecall: droneActual > 0 ? (dronePredictedCorrect / droneActual) * 100 : 0,
            accuracy: counted4 > 0 ? (correct4 / counted4) * 100 : null,
            nDrone: droneActual,
            n: samples.length,
          };
          done += 1;
          setProgress({ done, total, label: `${variant.label} @ ${snr.name}` });
        }
      }
      setResults(newResults);
    } catch (err) {
      if (String(err).includes('cancelled')) {
        setError('Benchmark cancelled.');
      } else {
        setError(`Benchmark failed: ${err instanceof Error ? err.message : String(err)} — is uvicorn api.app:app running on :8081?`);
      }
    } finally {
      setRunning(false);
    }
  };

  const cancel = () => { cancelRef.current = true; };

  
  
  
  
  
  
  const CONCURRENCY = 6;
  const CLASSICAL_MS = 30, QUANTUM_MS = 60;
  const totalCalls = MODEL_VARIANTS.length * SNR_PRESETS.length * CLASSES.length * samplesPerClass;
  const perCellSamples = samplesPerClass * CLASSES.length;
  const classicalCellMs = Math.ceil(perCellSamples / CONCURRENCY) * CLASSICAL_MS;
  const quantumCellMs = Math.ceil(perCellSamples / CONCURRENCY) * QUANTUM_MS;
  const classicalVariantCount = MODEL_VARIANTS.filter(v => v.supports4Class).length;
  const quantumVariantCount = MODEL_VARIANTS.length - classicalVariantCount;
  const estimatedSeconds = (SNR_PRESETS.length *
    (classicalVariantCount * classicalCellMs + quantumVariantCount * quantumCellMs)) / 1000;

  return (
    <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${isDark ? 'bg-zinc-950/70 border-indigo-950/40' : 'bg-white border-zinc-200'}`}>
      <div className={`flex justify-between items-center border-b pb-2 ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
        <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
          <BarChart3 className="w-4 h-4 text-indigo-400" /> Model Benchmark
        </span>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-[10px] font-mono text-zinc-500" title="Total calls = 5 variants x 6 SNR levels x 4 classes x this number">
            samples/class
            <input
              type="number" min={2} max={500} value={samplesPerClass} disabled={running}
              onChange={e => setSamplesPerClass(Math.max(2, Math.min(500, parseInt(e.target.value) || 5)))}
              className={`w-14 px-1 py-0.5 rounded border text-center ${isDark ? 'bg-zinc-900 border-zinc-800 text-zinc-200' : 'bg-white border-zinc-300'}`}
            />
            <span className="text-zinc-600">
              ({totalCalls.toLocaleString()} calls, ~{estimatedSeconds < 60 ? `${estimatedSeconds.toFixed(0)}s` : `${(estimatedSeconds / 60).toFixed(1)}m`})
            </span>
          </label>
          <button
            onClick={running ? cancel : runBenchmark}
            className={`px-3 py-1.5 font-mono text-[11px] font-bold border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all ${running
              ? 'bg-red-500/20 text-red-400 border-red-500/40 hover:bg-red-500/30'
              : 'bg-indigo-500/20 text-indigo-400 border-indigo-500/40 hover:bg-indigo-500/30'
              }`}
          >
            {running ? <><Square className="w-3 h-3" /> Cancel</> : <><Play className="w-3 h-3" /> Run Full Benchmark</>}
          </button>
        </div>
      </div>

      <p className={`text-[10px] font-sans leading-relaxed ${isDark ? 'text-zinc-500' : 'text-zinc-600'}`}>
        Runs {MODEL_VARIANTS.length} model variants (11/31/42-feature classic + quantum + quantum-ensemble)
        across {SNR_PRESETS.length} SNR scenarios, {samplesPerClass}&times;4 balanced synthetic samples per
        cell &mdash; same SUBTYPE_PRESETS physics and computeFeatureVectors() synthesis as
        RadarSimulator.tsx, calling the real backend endpoints directly (no client-side WASM). One button,
        runs every combination in sequence, plots drone recall (always comparable, since the quantum
        backend is binary drone/not_drone) and overall accuracy (classical variants only) vs. SNR.
      </p>

      {running && (
        <div className="flex items-center gap-2 text-[11px] font-mono text-indigo-400">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          <span>{progress.done}/{progress.total} &mdash; {progress.label}</span>
          <div className={`flex-1 h-1.5 rounded-full overflow-hidden ${isDark ? 'bg-zinc-800' : 'bg-zinc-200'}`}>
            <div className="h-full bg-indigo-500 transition-all duration-300" style={{ width: `${progress.total ? (progress.done / progress.total) * 100 : 0}%` }} />
          </div>
        </div>
      )}

      {error && <p className="text-[11px] font-mono text-red-400">{error}</p>}

      {results && (
        <>
          <LineChart
            title="Drone Recall vs. SNR (all variants)"
            results={results} metric="droneRecall" isDark={isDark}
          />
          <LineChart
            title="Overall 4-Class Accuracy vs. SNR (classical variants only)"
            results={results} metric="accuracy" isDark={isDark}
            variants={MODEL_VARIANTS.filter(v => v.supports4Class)}
          />
        </>
      )}
    </div>
  );
}

function LineChart({
  title, results, metric, isDark, variants = MODEL_VARIANTS,
}: {
  title: string;
  results: Results;
  metric: 'droneRecall' | 'accuracy';
  isDark: boolean;
  variants?: Variant[];
}) {
  const W = 640, H = 280;
  const padL = 40, padR = 12, padT = 16, padB = 36;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = SNR_PRESETS.length;
  const xFor = (i: number) => padL + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
  const yFor = (v: number) => padT + plotH - (Math.max(0, Math.min(100, v)) / 100) * plotH;

  return (
    <div className="flex flex-col gap-2">
      <span className={`text-[10px] font-mono uppercase tracking-wider ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>{title}</span>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ minWidth: 480 }}>
          {[0, 25, 50, 75, 100].map(g => (
            <g key={g}>
              <line x1={padL} x2={W - padR} y1={yFor(g)} y2={yFor(g)} stroke={isDark ? '#27272a' : '#e4e4e7'} strokeWidth={1} />
              <text x={padL - 6} y={yFor(g) + 3} textAnchor="end" fontSize={9} fill={isDark ? '#71717a' : '#a1a1aa'} fontFamily="monospace">{g}</text>
            </g>
          ))}
          {SNR_PRESETS.map((s, i) => (
            <text key={s.name} x={xFor(i)} y={H - padB + 14} textAnchor="middle" fontSize={8.5} fill={isDark ? '#71717a' : '#a1a1aa'} fontFamily="monospace">
              {s.name}
            </text>
          ))}
          {variants.map(v => {
            const points = SNR_PRESETS
              .map((s, i) => {
                const cell = results[v.key]?.[s.name];
                const val = cell ? cell[metric] : null;
                return val === null || val === undefined ? null : `${xFor(i)},${yFor(val)}`;
              })
              .filter((p): p is string => p !== null);
            if (points.length === 0) return null;
            return (
              <g key={v.key}>
                <polyline points={points.join(' ')} fill="none" stroke={v.color} strokeWidth={2} />
                {SNR_PRESETS.map((s, i) => {
                  const cell = results[v.key]?.[s.name];
                  const val = cell ? cell[metric] : null;
                  if (val === null || val === undefined) return null;
                  return <circle key={s.name} cx={xFor(i)} cy={yFor(val)} r={3} fill={v.color} />;
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="flex flex-wrap gap-3">
        {variants.map(v => (
          <span key={v.key} className="flex items-center gap-1.5 text-[9px] font-mono" style={{ color: v.color }}>
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ background: v.color }} />
            {v.label}
          </span>
        ))}
      </div>
    </div>
  );
}
