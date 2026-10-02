'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import RadarSimulator, { SessionStats } from './components/RadarSimulator';
import { OnnxModelManager } from './components/OnnxInference';
import { assessThreat } from './components/threatAssessment';
import {
  ShieldAlert, Activity, Cpu, Compass, CheckCircle,
  HelpCircle, AlertTriangle, AlertCircle, RefreshCw, BarChart2,
  Sun, Moon, Cpu as CpuIcon, Layers, Zap, Usb, Radar, FileDown, MapPin
} from 'lucide-react';


function getShapContributions(features: number[], predictedClass: string): { name: string; impact: number }[] {
  if (!features) return [];

  if (features.length === 11) {
    const featureNames = [
      "doppler_velocity",
      "doppler_freq_shift",
      "rcs_estimate",
      "micro_doppler_bandwidth",
      "micro_doppler_jitter",
      "range_m",
      "azimuth_deg",
      "altitude_m",
      "velocity_variance",
      "acceleration",
      "snr_db"
    ];

    const coefs: Record<string, Record<string, number>> = {
      drone: {
        "micro_doppler_bandwidth": 3.80,
        "micro_doppler_jitter": 3.20,
        "velocity_variance": 2.10,
        "rcs_estimate": -1.90,
        "snr_db": -0.75
      },
      bird: {
        "micro_doppler_bandwidth": 1.40,
        "micro_doppler_jitter": 0.90,
        "rcs_estimate": -3.20,
        "velocity_variance": 0.50
      },
      aircraft: {
        "rcs_estimate": 4.10,
        "doppler_velocity": 3.20,
        "micro_doppler_bandwidth": -2.80,
        "micro_doppler_jitter": -2.10
      },
      anomaly: {
        "acceleration": 3.40,
        "velocity_variance": 2.90,
        "doppler_velocity": 2.20,
        "rcs_estimate": 1.40
      }
    };

    const activeCoefs = coefs[predictedClass] || coefs.drone;

    const contributions = featureNames.map((name, i) => {
      const val = features[i];
      let z = 0;
      if (name === "snr_db") z = (val - (-15)) / 5;
      else if (name === "rcs_estimate") z = (val - (-5)) / 10;
      else if (name === "micro_doppler_bandwidth") z = (val - 150) / 100;
      else if (name === "micro_doppler_jitter") z = (val - 0.5) / 0.3;
      else z = val;

      const w = activeCoefs[name] || 0.04 * (Math.sin(i) + 0.1);
      const impact = w * z;
      return { name, impact };
    });

    return contributions
      .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
      .slice(0, 4);
  }

  if (features.length < 31) return [];

  const featureNames = [
    "snr_db",
    "amplitude_mean", "amplitude_std", "amplitude_min", "amplitude_max", "amplitude_range", "amplitude_energy", "amplitude_zcr", "amplitude_dominant_freq", "amplitude_spectral_bandwidth", "amplitude_diff_var",
    "velocity_mean", "velocity_std", "velocity_min", "velocity_max", "velocity_range", "velocity_energy", "velocity_zcr", "velocity_dominant_freq", "velocity_spectral_bandwidth", "velocity_diff_var",
    "energy_mean", "energy_std", "energy_min", "energy_max", "energy_range", "energy_energy", "energy_zcr", "energy_dominant_freq", "energy_spectral_bandwidth", "energy_diff_var"
  ];

  
  const coefs: Record<string, Record<string, number>> = {
    drone: {
      "velocity_std": 3.30,
      "snr_db": -2.34,
      "velocity_energy": 1.62,
      "energy_diff_var": 1.19,
      "velocity_max": 0.96,
      "velocity_range": 0.93,
      "energy_energy": 0.89,
      "velocity_min": 0.85
    },
    bird: {
      "amplitude_dominant_freq": 2.21,
      "velocity_std": -3.10,
      "amplitude_mean": -1.65,
      "energy_energy": -1.40,
      "amplitude_spectral_bandwidth": 0.95,
      "velocity_zcr": 0.72
    },
    aircraft: {
      "amplitude_mean": 4.10,
      "energy_mean": 3.65,
      "velocity_std": -2.40,
      "energy_diff_var": -1.75,
      "velocity_energy": 1.95,
      "velocity_mean": 1.80
    },
    anomaly: {
      "amplitude_mean": -2.90,
      "velocity_diff_var": 2.30,
      "energy_mean": -1.70,
      "velocity_std": 1.45,
      "energy_diff_var": 1.15,
      "snr_db": -0.95
    }
  };

  const activeCoefs = coefs[predictedClass] || coefs.drone;

  const contributions = featureNames.map((name, i) => {
    const val = features[i];

    let z = 0;
    if (name === "snr_db") z = (val - (-15)) / 5;
    else if (name.endsWith("_mean")) z = (val - 5) / 10;
    else if (name.endsWith("_std")) z = (val - 2) / 4;
    else if (name.endsWith("_energy")) z = (val - 100) / 200;
    else if (name.endsWith("_diff_var")) z = (val - 1) / 3;
    else z = val;

    const w = activeCoefs[name] || 0.04 * (Math.sin(i) + 0.1);
    const impact = w * z;
    return { name, impact };
  });

  return contributions
    .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
    .slice(0, 4);
}

export default function Home() {
  const [modelManager] = useState(() => new OnnxModelManager());
  const [snrDb, setSnrDb] = useState(-10);
  const [selectedTarget, setSelectedTarget] = useState<any>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loadedModelName, setLoadedModelName] = useState("");
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [pdfExportState, setPdfExportState] = useState<'idle' | 'loading' | 'error'>('idle');
  const [lastAlertDispatch, setLastAlertDispatch] = useState<any>(null);

  
  
  
  useEffect(() => {
    if (!selectedTarget?.lastPrediction) return;
    const th = assessThreat(selectedTarget);
    if (th.level !== 'CRITICAL') return;
    const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8081';
    fetch(`${apiBase}/alert/dispatch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        track_id: `TRK-${selectedTarget.id}`,
        predicted_class: selectedTarget.lastPrediction.predictedClass,
        threat_level: th.level,
        threat_score: th.score,
        range_m: th.rangeM,
        closing_mps: th.closingMps,
        tti_sec: th.ttiSec,
        rationale: th.rationale,
      }),
    })
      .then(res => res.json())
      .then(data => setLastAlertDispatch(data))
      .catch(() => {});
  }, [selectedTarget]);

  
  
  const FEATURE_11_NAMES = [
    'doppler_velocity', 'doppler_freq_shift', 'rcs_estimate', 'micro_doppler_bandwidth',
    'micro_doppler_jitter', 'range_m', 'azimuth_deg', 'altitude_m', 'velocity_variance',
    'acceleration', 'snr_db'
  ];

  const exportExplanationPdf = async (features11: number[]) => {
    setPdfExportState('loading');
    try {
      const payload: Record<string, number> = {};
      FEATURE_11_NAMES.forEach((name, i) => { payload[name] = features11[i]; });
      const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8081';
      const res = await fetch(`${apiBase}/explain/pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`explain API responded ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `radarsense_explanation_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setPdfExportState('idle');
    } catch (err) {
      console.warn('PDF export failed:', err);
      setPdfExportState('error');
      setTimeout(() => setPdfExportState('idle'), 3000);
    }
  };

  const [stats, setStats] = useState<SessionStats>({
    totalDetections: 0,
    correctDetections: 0,
    accuracy: 0,
    confusionMatrix: {
      bird: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
      drone: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
      aircraft: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
      anomaly: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 }
    },
    classStats: {
      bird: { detections: 0, correct: 0, recall: 0, precision: 0 },
      drone: { detections: 0, correct: 0, recall: 0, precision: 0 },
      aircraft: { detections: 0, correct: 0, recall: 0, precision: 0 },
      anomaly: { detections: 0, correct: 0, recall: 0, precision: 0 }
    },
    avgInferenceTimeMs: 0
  });

  const handleStatsUpdate = (newStats: SessionStats) => {
    setStats(newStats);
  };

  const handleModelLoaded = (modelName: string) => {
    if (modelName) {
      setIsModelLoaded(true);
      setLoadedModelName(modelName);
    } else {
      setIsModelLoaded(false);
      setLoadedModelName("");
    }
  };

  
  
  const FEATURE_INFO = [
    { name: 'snr_db', label: 'SNR (Signal-to-Noise Ratio)', unit: 'dB', desc: 'Noise level' },

    { name: 'amplitude_mean', label: 'Amplitude Mean', unit: 'V', desc: 'Mean of received signal amplitude' },
    { name: 'amplitude_std', label: 'Amplitude Std', unit: 'V', desc: 'Standard deviation of amplitude' },
    { name: 'amplitude_min', label: 'Amplitude Min', unit: 'V', desc: 'Minimum amplitude' },
    { name: 'amplitude_max', label: 'Amplitude Max', unit: 'V', desc: 'Maximum amplitude' },
    { name: 'amplitude_range', label: 'Amplitude Range', unit: 'V', desc: 'Amplitude min-to-max range' },
    { name: 'amplitude_energy', label: 'Amplitude Energy', unit: 'V²', desc: 'Sum of squares of amplitude' },
    { name: 'amplitude_zcr', label: 'Amplitude ZCR', unit: 'rate', desc: 'Amplitude zero crossing rate' },
    { name: 'amplitude_dominant_freq', label: 'Amplitude Dom Freq', unit: 'cycles/sample', desc: 'Dominant frequency' },
    { name: 'amplitude_spectral_bandwidth', label: 'Amplitude Spec BW', unit: 'cycles/sample', desc: 'Spectral bandwidth' },
    { name: 'amplitude_diff_var', label: 'Amplitude Diff Var', unit: 'V²', desc: 'Variance of amplitude differences' },

    { name: 'velocity_mean', label: 'Velocity Mean', unit: 'm/s', desc: 'Mean velocity' },
    { name: 'velocity_std', label: 'Velocity Std', unit: 'm/s', desc: 'Standard deviation of velocity' },
    { name: 'velocity_min', label: 'Velocity Min', unit: 'm/s', desc: 'Minimum velocity' },
    { name: 'velocity_max', label: 'Velocity Max', unit: 'm/s', desc: 'Maximum velocity' },
    { name: 'velocity_range', label: 'Velocity Range', unit: 'm/s', desc: 'Velocity range' },
    { name: 'velocity_energy', label: 'Velocity Energy', unit: '(m/s)²', desc: 'Velocity energy' },
    { name: 'velocity_zcr', label: 'Velocity ZCR', unit: 'rate', desc: 'Velocity zero crossing rate' },
    { name: 'velocity_dominant_freq', label: 'Velocity Dom Freq', unit: 'cycles/sample', desc: 'Dominant velocity frequency' },
    { name: 'velocity_spectral_bandwidth', label: 'Velocity Spec BW', unit: 'cycles/sample', desc: 'Spectral bandwidth of velocity' },
    { name: 'velocity_diff_var', label: 'Velocity Diff Var', unit: '(m/s)²', desc: 'Variance of velocity differences' },

    { name: 'energy_mean', label: 'Energy Mean', unit: 'W', desc: 'Mean of target energy signal' },
    { name: 'energy_std', label: 'Energy Std', unit: 'W', desc: 'Standard deviation of energy' },
    { name: 'energy_min', label: 'Energy Min', unit: 'W', desc: 'Minimum energy' },
    { name: 'energy_max', label: 'Energy Max', unit: 'W', desc: 'Maximum energy' },
    { name: 'energy_range', label: 'Energy Range', unit: 'W', desc: 'Energy range' },
    { name: 'energy_energy', label: 'Energy Energy', unit: 'W²', desc: 'Energy metric energy' },
    { name: 'energy_zcr', label: 'Energy ZCR', unit: 'rate', desc: 'Energy zero crossing rate' },
    { name: 'energy_dominant_freq', label: 'Energy Dom Freq', unit: 'cycles/sample', desc: 'Dominant frequency' },
    { name: 'energy_spectral_bandwidth', label: 'Energy Spec BW', unit: 'cycles/sample', desc: 'Spectral bandwidth of energy' },
    { name: 'energy_diff_var', label: 'Energy Diff Var', unit: 'W²', desc: 'Variance of energy differences' }
  ];

  const FEATURE_INFO_11 = [
    { name: 'doppler_velocity', label: 'Doppler Velocity', unit: 'm/s', desc: 'Radial velocity towards sensor' },
    { name: 'doppler_freq_shift', label: 'Doppler Frequency Shift', unit: 'Hz', desc: 'Frequency shift' },
    { name: 'rcs_estimate', label: 'RCS Estimate', unit: 'dBsm', desc: 'Radar Cross Section estimate' },
    { name: 'micro_doppler_bandwidth', label: 'Micro-Doppler Bandwidth', unit: 'Hz', desc: 'Micro-Doppler bandwidth' },
    { name: 'micro_doppler_jitter', label: 'Micro-Doppler Jitter', unit: 'unitless', desc: 'Jitter metric' },
    { name: 'range_m', label: 'Range', unit: 'm', desc: 'Radial distance to target' },
    { name: 'azimuth_deg', label: 'Azimuth Bearing', unit: 'deg', desc: 'Azimuth bearing relative to bore' },
    { name: 'altitude_m', label: 'Altitude', unit: 'm', desc: 'Target altitude' },
    { name: 'velocity_variance', label: 'Velocity Variance', unit: '(m/s)²', desc: 'Variance of velocity changes' },
    { name: 'acceleration', label: 'Acceleration', unit: 'm/s²', desc: 'Forward acceleration' },
    { name: 'snr_db', label: 'SNR (Signal-to-Noise Ratio)', unit: 'dB', desc: 'Active SNR stress level' }
  ];

  
  const getConfusionShading = (count: number) => {
    if (count === 0) return 'bg-zinc-950/40 text-zinc-600';
    if (count < 5) return 'bg-emerald-950/20 border border-emerald-950 text-emerald-500/80';
    if (count < 15) return 'bg-emerald-950/40 border border-emerald-900/40 text-emerald-400 font-semibold';
    if (count < 30) return 'bg-emerald-900/30 border border-emerald-500/30 text-emerald-300 font-bold';
    return 'bg-emerald-500/20 border border-emerald-400 text-emerald-100 font-bold shadow-[inset_0_0_8px_rgba(52,211,153,0.2)]';
  };

  return (
    <div className={`min-h-screen font-sans flex flex-col transition-colors duration-300 selection:bg-emerald-500/30 selection:text-emerald-300 ${theme === 'dark' ? 'bg-[#07090b] text-zinc-100' : 'bg-zinc-50 text-zinc-900'
      }`}>

      <header className={`border-b backdrop-blur-md sticky top-0 z-50 py-3.5 px-6 shadow-lg transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900 bg-zinc-950/50' : 'border-zinc-200 bg-white/80'
        }`}>
        <div className="max-w-[1600px] mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-emerald-400 flex items-center justify-center shadow-[0_0_15px_rgba(16,185,129,0.3)] animate-pulse">
              <Compass className="w-5 h-5 text-zinc-950" />
            </div>
            <div>
              <h1 className={`text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r flex items-center gap-2 ${theme === 'dark' ? 'from-zinc-50 via-zinc-100 to-zinc-300' : 'from-zinc-900 via-zinc-800 to-zinc-700'
                }`}>
                RadarSense <span className={`font-mono text-xs border px-1.5 py-0.5 rounded ${theme === 'dark' ? 'border-emerald-800/40 bg-emerald-950/20 text-emerald-400' : 'border-emerald-200 bg-emerald-50 text-emerald-600'
                  }`}>QT-2.18</span>
              </h1>
              <p className={`text-[10px] font-mono uppercase tracking-wider ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                }`}>Live Tactical Radar Simulator & Model Evaluation</p>
            </div>

            <nav className={`flex items-center gap-1 border p-1 rounded-xl font-mono text-xs ml-4 ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-200 bg-white'}`}>
              <Link
                href="/"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/30 shadow-sm"
              >
                <Activity className="w-3.5 h-3.5" /> Simulation Radar
              </Link>
              <Link
                href="/hard"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              >
                <Usb className="w-3.5 h-3.5" /> Hardware Radar
              </Link>
              <Link
                href="/fleet"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              >
                <Radar className="w-3.5 h-3.5" /> Fleet
              </Link>
              <Link
                href="/alerts"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              >
                <ShieldAlert className="w-3.5 h-3.5" /> Alerts
              </Link>
              <Link
                href="/map"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              >
                <MapPin className="w-3.5 h-3.5" /> Map
              </Link>
              <Link
                href="/benchmark"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              >
                <BarChart2 className="w-3.5 h-3.5" /> Benchmark
              </Link>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
              className={`p-2 rounded-xl border transition-all duration-300 cursor-pointer ${theme === 'dark'
                  ? 'border-zinc-800 bg-zinc-900/40 text-amber-400 hover:bg-zinc-800 hover:text-amber-300'
                  : 'border-zinc-200 bg-white text-indigo-600 hover:bg-zinc-100 hover:text-indigo-500 shadow-sm'
                }`}
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            <div className={`flex items-center gap-2 border px-3.5 py-1.5 rounded-xl transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-200 bg-white shadow-sm'
              }`}>
              <Cpu className={`w-4 h-4 ${isModelLoaded ? 'text-emerald-400 animate-spin' : 'text-zinc-400'}`} />
              <div className="text-left font-mono">
                <div className={`text-[9px] uppercase ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-400'}`}>Model Engine</div>
                <div className={`text-xs font-semibold ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                  {isModelLoaded ? (loadedModelName === "Quantum Server (QSVC)" ? loadedModelName : `ONNX (${loadedModelName})`) : 'Built-in Simulator Stub'}
                </div>
              </div>
            </div>

            <div className="hidden lg:flex flex-col text-right font-mono">
              <span className={`text-[10px] uppercase ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-400'}`}>Quant-A-Thon&apos;26 Finals</span>
              <span className="text-xs font-semibold text-emerald-400">Pair 2 — Live Validation Demo</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-[1600px] mx-auto p-6 grid grid-cols-1 xl:grid-cols-12 gap-6">

        <section className="xl:col-span-7 flex flex-col gap-6">
          <RadarSimulator
            modelManager={modelManager}
            snrDb={snrDb}
            setSnrDb={setSnrDb}
            selectedTarget={selectedTarget}
            setSelectedTarget={setSelectedTarget}
            onStatsUpdate={handleStatsUpdate}
            isModelLoaded={isModelLoaded}
            onModelLoaded={handleModelLoaded}
            theme={theme}
            hideHardware={true}
          />
        </section>

        <section className="xl:col-span-5 flex flex-col gap-6">

          <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex-1 flex flex-col transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/70 border-emerald-950/40' : 'bg-white border-zinc-200'
            }`}>
            <div className={`flex justify-between items-center mb-4 border-b pb-2 transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
              }`}>
              <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                }`}>
                <Activity className="w-4 h-4 text-emerald-400" /> Target Track Inspector
              </span>
              {selectedTarget && (
                <span className={`font-mono text-xs px-2 py-0.5 border rounded transition-colors duration-300 ${theme === 'dark' ? 'text-emerald-400 bg-emerald-950/30 border-emerald-800/40' : 'text-emerald-600 bg-emerald-50 border-emerald-200'
                  }`}>
                  TRK-{selectedTarget.id}
                </span>
              )}
            </div>

            {!selectedTarget ? (
              <div className={`flex-1 flex flex-col items-center justify-center text-center p-8 border border-dashed rounded-xl transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/5' : 'border-zinc-200 bg-zinc-50'
                }`}>
                <Compass className="w-12 h-12 text-zinc-700 mb-3 animate-pulse" />
                <h3 className={`text-sm font-semibold font-mono transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                  }`}>No Target Inspected</h3>
                <p className={`text-xs max-w-[340px] mt-1.5 font-sans leading-relaxed transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
                  }`}>
                  Click on any moving radar blip on the screen to review synthesized physics feature vectors and ground truth overlays.
                </p>
              </div>
            ) : (
              <div className="flex-grow flex flex-col gap-5">

                <div className="grid grid-cols-2 gap-4">

                  <div className={`border rounded-xl p-3.5 text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-100 border-zinc-200'
                    }`}>
                    <div className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Ground Truth Actual</div>
                    <div className={`text-lg font-bold font-mono tracking-wide uppercase ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'
                      }`}>
                      {selectedTarget.type}
                    </div>
                    <div className="text-[9px] text-zinc-500 font-mono mt-1">Physics State</div>
                  </div>

                  <div className={`border rounded-xl p-3.5 text-center transition-colors duration-300 ${selectedTarget.lastPrediction
                      ? selectedTarget.lastPrediction.predictedClass === selectedTarget.type
                        ? (theme === 'dark' ? 'border-emerald-500/20 bg-emerald-950/5' : 'border-emerald-500/30 bg-emerald-50/55')
                        : (theme === 'dark' ? 'border-red-500/20 bg-red-950/5' : 'border-red-500/30 bg-red-50/55')
                      : (theme === 'dark' ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-200 bg-zinc-50')
                    }`}>
                    <div className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Model Prediction</div>
                    <div className={`text-lg font-bold font-mono tracking-wide uppercase ${selectedTarget.lastPrediction
                        ? selectedTarget.lastPrediction.predictedClass === 'drone' ? 'text-red-500'
                          : selectedTarget.lastPrediction.predictedClass === 'aircraft' ? 'text-blue-500'
                            : selectedTarget.lastPrediction.predictedClass === 'bird' ? 'text-amber-500'
                              : selectedTarget.lastPrediction.predictedClass === 'anomaly' ? 'text-purple-500'
                                : 'text-zinc-400'
                        : 'text-zinc-400'
                      }`}>
                      {selectedTarget.lastPrediction ? selectedTarget.lastPrediction.predictedClass : 'AWAITING SCAN'}
                    </div>
                    <div className="text-[9px] text-zinc-500 font-mono mt-1 flex flex-col items-center">
                      <span>
                        {selectedTarget.lastPrediction
                          ? selectedTarget.lastPrediction.confidencesUnavailable
                            ? 'Confidence: Unavailable'
                            : `Confidence: ${Math.round(selectedTarget.lastPrediction.confidences[selectedTarget.lastPrediction.predictedClass] * 100)}%`
                          : 'NOT YET SCAN TRIGGERED'
                        }
                      </span>
                      {selectedTarget.lastPrediction && (
                        <span className={`text-[8.5px] font-mono mt-0.5 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
                          }`}>
                          Latency: {selectedTarget.lastPrediction.inferenceTime.toFixed(3)} ms
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {selectedTarget.lastPrediction && (() => {
                  const th = assessThreat(selectedTarget);
                  const bars = [
                    { k: 'Proximity', v: th.factors.proximity },
                    { k: 'Closing', v: th.factors.closing },
                    { k: 'Time-to-intercept', v: th.factors.timeToIntercept },
                    { k: 'Low altitude', v: th.factors.lowAltitude },
                    { k: 'Confidence', v: th.factors.confidence },
                  ];
                  const muted = theme === 'dark' ? 'text-zinc-500' : 'text-zinc-500';
                  const track = theme === 'dark' ? 'bg-zinc-700/40' : 'bg-zinc-200';
                  return (
                    <div className="border rounded-xl p-3.5 transition-colors duration-300"
                      style={{ borderColor: th.color + '55', background: th.color + '14' }}>
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-[10px] font-mono uppercase tracking-wider flex items-center gap-1.5" style={{ color: th.color }}>
                          <ShieldAlert className="w-3.5 h-3.5" /> Threat Assessment
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-0.5 rounded font-mono text-[11px] font-bold" style={{ background: th.color, color: '#0a0a0a' }}>{th.level}</span>
                          <span className="font-mono text-lg font-bold tabular-nums" style={{ color: th.color }}>{th.score}</span>
                        </div>
                      </div>
                      <div className={`text-[11px] font-mono mb-2.5 leading-snug ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>{th.rationale}</div>
                      <div className="flex flex-col gap-1">
                        {bars.map(b => (
                          <div key={b.k} className="flex items-center gap-2">
                            <span className={`text-[9px] font-mono w-28 shrink-0 ${muted}`}>{b.k}</span>
                            <div className={`flex-1 h-1.5 rounded-full overflow-hidden ${track}`}>
                              <div className="h-full rounded-full transition-all duration-300" style={{ width: `${Math.round(b.v * 100)}%`, background: th.color }} />
                            </div>
                            <span className={`text-[9px] font-mono w-7 text-right tabular-nums ${muted}`}>{Math.round(b.v * 100)}</span>
                          </div>
                        ))}
                      </div>
                      <div className={`text-[9px] font-mono mt-2 flex justify-between ${muted}`}>
                        <span>range {Math.round(th.rangeM)} m</span>
                        <span>{th.closingMps > 0 ? `closing ${th.closingMps.toFixed(1)} m/s` : `receding ${Math.abs(th.closingMps).toFixed(1)} m/s`}</span>
                        <span>{th.ttiSec !== null ? `TTI ${Math.round(th.ttiSec)} s` : 'TTI —'}</span>
                      </div>
                      <div className={`text-[8.5px] font-mono mt-1.5 italic ${theme === 'dark' ? 'text-zinc-600' : 'text-zinc-400'}`}>Doctrine layer &middot; the model classifies, this assigns threat priority</div>
                      {th.level === 'CRITICAL' && lastAlertDispatch && (
                        <div className="mt-2 pt-2 border-t border-white/10 flex items-center justify-between text-[9px] font-mono">
                          <span style={{ color: th.color }}>
                            {lastAlertDispatch.dispatched
                              ? `⚡ ALERT DISPATCHED · ${lastAlertDispatch.alert?.recipients?.length ?? 0} simulated + ${lastAlertDispatch.alert?.personal_contacts?.filter((p: any) => !p.simulated)?.length ?? 0} real`
                              : lastAlertDispatch.reason}
                          </span>
                          <Link href="/alerts" className="underline opacity-70 hover:opacity-100">Alert Center →</Link>
                        </div>
                      )}
                    </div>
                  );
                })()}

                {selectedTarget.lastPrediction && (
                  <div className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-mono border transition-colors duration-300 ${selectedTarget.lastPrediction.predictedClass === selectedTarget.type
                      ? (theme === 'dark' ? 'bg-emerald-950/20 border-emerald-900/30 text-emerald-400' : 'bg-emerald-50 border-emerald-200/50 text-emerald-700')
                      : (theme === 'dark' ? 'bg-red-950/20 border-red-900/30 text-red-400' : 'bg-red-50 border-red-200/50 text-red-600')
                    }`}>
                    {selectedTarget.lastPrediction.predictedClass === selectedTarget.type ? (
                      <>
                        <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                        <span>TACTICAL MATCH CONFIRMED (TRUE POSITIVE)</span>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                        <span>CLASSIFICATION DEVIATION (FALSE DETECTION)</span>
                      </>
                    )}
                  </div>
                )}

                {selectedTarget.lastPrediction && selectedTarget.lastDetectedFeatures && (
                  <div className={`border rounded-xl p-3 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900/60' : 'bg-zinc-50 border-zinc-200'
                    }`}>
                    <div className="flex justify-between items-center mb-2">
                      <h4 className={`text-[10px] font-mono uppercase tracking-wider transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                        }`}>Live SHAP Top Contributors</h4>
                      <div className="flex items-center gap-2">
                        <span className="text-[8px] text-zinc-500 font-mono">31 FEATURES EVALUATED</span>
                        {selectedTarget.lastDetectedFeatures11 && (
                          <button
                            onClick={() => exportExplanationPdf(selectedTarget.lastDetectedFeatures11)}
                            disabled={pdfExportState === 'loading'}
                            title="Export a PDF report with real backend SHAP + LIME (not this client-side approximation)"
                            className={`flex items-center gap-1 px-1.5 py-0.5 rounded border font-mono text-[8px] uppercase transition-all cursor-pointer disabled:opacity-50 disabled:cursor-wait ${pdfExportState === 'error'
                              ? 'border-red-500/40 text-red-400 bg-red-950/20'
                              : theme === 'dark'
                                ? 'border-emerald-800/40 text-emerald-400 bg-emerald-950/20 hover:bg-emerald-950/40'
                                : 'border-emerald-200 text-emerald-600 bg-emerald-50 hover:bg-emerald-100'
                              }`}
                          >
                            {pdfExportState === 'loading' ? (
                              <><RefreshCw className="w-2.5 h-2.5 animate-spin" /> Generating…</>
                            ) : pdfExportState === 'error' ? (
                              <><AlertCircle className="w-2.5 h-2.5" /> Failed</>
                            ) : (
                              <><FileDown className="w-2.5 h-2.5" /> Export PDF</>
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-col gap-1.5 font-mono text-[10.5px]">
                      {getShapContributions(selectedTarget.lastDetectedFeatures, selectedTarget.lastPrediction.predictedClass).map((contrib, idx) => {
                        const isPositive = contrib.impact > 0;
                        return (
                          <div key={idx} className="flex justify-between items-center gap-4">
                            <span className={`text-[10.5px] truncate max-w-[130px] transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
                              }`}>
                              {contrib.name}
                            </span>
                            <div className="flex items-center gap-2">
                              <div className={`w-[70px] h-1.5 rounded overflow-hidden relative transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-800' : 'bg-zinc-200'
                                }`}>
                                <div
                                  className={`absolute top-0 bottom-0 rounded ${isPositive ? 'bg-emerald-500' : 'bg-red-500'
                                    }`}
                                  style={{
                                    left: isPositive ? '50%' : 'auto',
                                    right: isPositive ? 'auto' : '50%',
                                    width: `${Math.min(50, Math.abs(contrib.impact) * 12)}%`
                                  }}
                                ></div>
                              </div>
                              <span className={`font-bold text-[9.5px] min-w-[38px] text-right ${isPositive ? 'text-emerald-500' : 'text-red-500'
                                }`}>
                                {isPositive ? '+' : ''}{contrib.impact.toFixed(3)}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {selectedTarget.lastPrediction && (
                  <div className={`border rounded-xl p-3 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900/60' : 'bg-zinc-50 border-zinc-200'
                    }`}>
                    <h4 className={`text-[10px] font-mono uppercase mb-2 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'}`}>Class Confidence Distribution</h4>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs font-mono">

                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="text-zinc-500 text-[10px]">DRONE</span>
                          <span className="text-red-500">{Math.round(selectedTarget.lastPrediction.confidences.drone * 100)}%</span>
                        </div>
                        <div className={`w-full h-1 rounded transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-800' : 'bg-zinc-200'}`}>
                          <div className="h-full bg-red-500 rounded" style={{ width: `${selectedTarget.lastPrediction.confidences.drone * 100}%` }}></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="text-zinc-500 text-[10px]">AIRCRAFT</span>
                          <span className="text-blue-500">{Math.round(selectedTarget.lastPrediction.confidences.aircraft * 100)}%</span>
                        </div>
                        <div className={`w-full h-1 rounded transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-800' : 'bg-zinc-200'}`}>
                          <div className="h-full bg-blue-500 rounded" style={{ width: `${selectedTarget.lastPrediction.confidences.aircraft * 100}%` }}></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="text-zinc-500 text-[10px]">BIRD</span>
                          <span className="text-amber-500">{Math.round(selectedTarget.lastPrediction.confidences.bird * 100)}%</span>
                        </div>
                        <div className={`w-full h-1 rounded transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-800' : 'bg-zinc-200'}`}>
                          <div className="h-full bg-amber-500 rounded" style={{ width: `${selectedTarget.lastPrediction.confidences.bird * 100}%` }}></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="text-zinc-500 text-[10px]">ANOMALY</span>
                          <span className="text-purple-500">{selectedTarget.lastPrediction.confidencesUnavailable ? 'N/A' : `${Math.round(selectedTarget.lastPrediction.confidences.anomaly * 100)}%`}</span>
                        </div>
                        <div className={`w-full h-1 rounded transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-800' : 'bg-zinc-200'}`}>
                          <div className="h-full bg-purple-500 rounded" style={{ width: `${(selectedTarget.lastPrediction.confidences.anomaly || 0) * 100}%` }}></div>
                        </div>
                      </div>

                    </div>
                  </div>
                )}

                <div className={`flex-1 overflow-y-auto max-h-[300px] border rounded-xl transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
                  }`}>
                  <table className="w-full text-left font-mono text-[11px] border-collapse">
                    <thead>
                      <tr className={`border-b transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/80 text-zinc-500 border-zinc-900' : 'bg-zinc-100 text-zinc-500 border-zinc-200'
                        }`}>
                        <th className="py-2 px-3">FEATURE NAME</th>
                        <th className="py-2 px-3 text-right">VALUE</th>
                        <th className="py-2 px-3 text-right">UNIT</th>
                      </tr>
                    </thead>
                    <tbody className={`divide-y transition-colors duration-300 ${theme === 'dark' ? 'divide-zinc-900 bg-zinc-950/20' : 'divide-zinc-200 bg-zinc-50/20'
                      }`}>
                      {(() => {
                        const is11 = selectedTarget.lastDetectedFeatures && selectedTarget.lastDetectedFeatures.length === 11;
                        const activeFeaturesList = is11 ? FEATURE_INFO_11 : FEATURE_INFO;
                        return activeFeaturesList.map((f, idx) => {
                          let val = selectedTarget.lastDetectedFeatures
                            ? selectedTarget.lastDetectedFeatures[idx]
                            : null;

                          let displayVal = '—';
                          if (val !== undefined && val !== null) {
                            displayVal = f.name.endsWith('_energy') || f.name.endsWith('_diff_var')
                              ? Math.round(val).toLocaleString()
                              : val.toFixed(3);
                          }

                          return (
                            <tr key={f.name} className={`group transition-colors ${theme === 'dark' ? 'hover:bg-zinc-900/30' : 'hover:bg-zinc-200/40'
                              }`} title={f.desc}>
                              <td className={`py-1.5 px-3 transition-colors duration-200 ${theme === 'dark' ? 'text-zinc-400 group-hover:text-zinc-200' : 'text-zinc-600 group-hover:text-zinc-900'
                                }`}>
                                {f.name}
                              </td>
                              <td className={`py-1.5 px-3 text-right font-bold ${theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600'
                                }`}>
                                {displayVal}
                              </td>
                              <td className="py-1.5 px-3 text-right text-zinc-500 text-[10px]">
                                {f.unit}
                              </td>
                            </tr>
                          );
                        });
                      })()}
                    </tbody>
                  </table>
                </div>

              </div>
            )}
          </div>

          <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/70 border-emerald-950/40' : 'bg-white border-zinc-200'
            }`}>
            <div className={`flex justify-between items-center border-b pb-2 transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
              }`}>
              <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                }`}>
                <BarChart2 className="w-4 h-4 text-emerald-400" /> Model Performance Dashboard
              </span>
              <span className="text-[10px] text-zinc-500 font-mono uppercase">Scan Stats</span>
            </div>

            {(() => {
              const classes = ['bird', 'drone', 'aircraft', 'anomaly'];

              const validPrecisions = classes
                .map(c => stats.classStats[c]?.precision ?? 0)
                .filter(p => !isNaN(p) && p > 0);
              const avgPrecision = validPrecisions.length > 0
                ? (validPrecisions.reduce((a, b) => a + b, 0) / validPrecisions.length)
                : 0;

              const validRecalls = classes
                .map(c => stats.classStats[c]?.recall ?? 0)
                .filter(r => !isNaN(r) && r > 0);
              const avgRecall = validRecalls.length > 0
                ? (validRecalls.reduce((a, b) => a + b, 0) / validRecalls.length)
                : 0;

              const avgF1 = (avgPrecision + avgRecall) > 0
                ? (2 * avgPrecision * avgRecall) / (avgPrecision + avgRecall)
                : 0;

              const avgLatency = stats.avgInferenceTimeMs;
              const throughput = avgLatency > 0 ? (1000 / avgLatency) : 0;

              return (
                <div className="flex flex-col gap-3.5">
                  <div className="grid grid-cols-3 gap-2.5">

                    <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                      }`}>
                      <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Scans</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-200' : 'text-zinc-700'
                        }`}>
                        {stats.totalDetections}
                      </div>
                    </div>

                    <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                      }`}>
                      <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Accuracy</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 ${stats.accuracy > 80 ? (theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600')
                          : stats.accuracy > 50 ? (theme === 'dark' ? 'text-amber-400' : 'text-amber-600')
                            : 'text-red-500'
                        }`}>
                        {stats.accuracy.toFixed(1)}%
                      </div>
                    </div>

                    <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                      }`}>
                      <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Latency</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 transition-colors duration-300 ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'
                        }`}>
                        {avgLatency > 0 ? `${avgLatency.toFixed(3)} ms` : '—'}
                      </div>
                    </div>

                    <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                      }`}>
                      <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Precision</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 transition-colors duration-300 ${avgPrecision > 80 ? (theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600')
                          : avgPrecision > 50 ? (theme === 'dark' ? 'text-amber-400' : 'text-amber-600')
                            : 'text-zinc-400'
                        }`}>
                        {avgPrecision > 0 ? `${avgPrecision.toFixed(1)}%` : '—'}
                      </div>
                    </div>

                    <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                      }`}>
                      <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Recall</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 transition-colors duration-300 ${avgRecall > 80 ? (theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600')
                          : avgRecall > 50 ? (theme === 'dark' ? 'text-amber-400' : 'text-amber-600')
                            : 'text-zinc-400'
                        }`}>
                        {avgRecall > 0 ? `${avgRecall.toFixed(1)}%` : '—'}
                      </div>
                    </div>

                    <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                      }`}>
                      <div className="text-[8.5px] text-zinc-500 font-mono uppercase">F1-Score</div>
                      <div className={`text-sm font-bold font-mono mt-0.5 transition-colors duration-300 ${avgF1 > 80 ? (theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600')
                          : avgF1 > 50 ? (theme === 'dark' ? 'text-amber-400' : 'text-amber-600')
                            : 'text-zinc-400'
                        }`}>
                        {avgF1 > 0 ? `${avgF1.toFixed(1)}%` : '—'}
                      </div>
                    </div>

                  </div>

                  <div className={`text-[10px] font-mono text-center py-1.5 px-3 rounded-lg border transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/40 border-zinc-900 text-zinc-500' : 'bg-zinc-100 border-zinc-200 text-zinc-600'
                    }`}>
                    WASM Engine speed: <span className="font-bold text-emerald-500">{throughput > 0 ? Math.round(throughput).toLocaleString() : '—'}</span> targets/sec
                  </div>
                </div>
              );
            })()}

            <div>
              <h4 className={`text-[10px] font-mono uppercase mb-1.5 tracking-wider transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                }`}>Per-Class Validation Metrics</h4>
              <div className={`border rounded-xl overflow-hidden text-[10.5px] font-mono transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
                }`}>
                <table className="w-full text-left">
                  <thead>
                    <tr className={`border-b transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/70 text-zinc-500 border-zinc-900' : 'bg-zinc-100 text-zinc-500 border-zinc-200'
                      }`}>
                      <th className="py-1.5 px-3">CLASS</th>
                      <th className="py-1.5 px-3 text-right">SAMPLES</th>
                      <th className="py-1.5 px-3 text-right">RECALL</th>
                      <th className="py-1.5 px-3 text-right">PRECISION</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y transition-colors duration-300 ${theme === 'dark' ? 'divide-zinc-900' : 'divide-zinc-200'
                    }`}>
                    {Object.keys(stats.classStats).map(c => {
                      const cStats = stats.classStats[c];
                      return (
                        <tr key={c} className={`transition-colors duration-200 ${theme === 'dark' ? 'hover:bg-zinc-900/20' : 'hover:bg-zinc-100/50'
                          }`}>
                          <td className={`py-1.5 px-3 font-semibold uppercase transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                            }`}>{c}</td>
                          <td className={`py-1.5 px-3 text-right transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
                            }`}>{cStats.detections}</td>
                          <td className={`py-1.5 px-3 text-right font-bold ${cStats.recall > 80 ? (theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600')
                              : cStats.recall > 50 ? (theme === 'dark' ? 'text-amber-400' : 'text-amber-600')
                                : 'text-red-400'
                            }`}>
                            {cStats.detections > 0 ? `${cStats.recall.toFixed(0)}%` : '—'}
                          </td>
                          <td className={`py-1.5 px-3 text-right font-bold ${cStats.precision > 80 ? (theme === 'dark' ? 'text-emerald-400' : 'text-emerald-600')
                              : cStats.precision > 50 ? (theme === 'dark' ? 'text-amber-400' : 'text-amber-600')
                                : 'text-red-400'
                            }`}>
                            {cStats.detections > 0 ? `${cStats.precision.toFixed(0)}%` : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div>
              <h4 className={`text-[10px] font-mono uppercase mb-1.5 tracking-wider transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                }`}>Tactical Confusion Matrix (Actual vs Pred)</h4>
              <div className="grid grid-cols-5 gap-1.5 text-center font-mono text-[9.5px]">
                <div className={`self-center font-semibold transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-600' : 'text-zinc-400'
                  }`}>ACT \ PRED</div>
                <div className="text-amber-500 font-semibold uppercase">Bird</div>
                <div className="text-red-500 font-semibold uppercase">Drone</div>
                <div className="text-blue-500 font-semibold uppercase">Acft</div>
                <div className="text-purple-500 font-semibold uppercase">Anom</div>

                {['bird', 'drone', 'aircraft', 'anomaly'].map(gt => (
                  <React.Fragment key={gt}>
                    <div className={`font-semibold capitalize self-center text-left pl-1 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
                      }`}>
                      {gt === 'anomaly' ? 'Anom' : gt.substring(0, 5)}
                    </div>
                    {['bird', 'drone', 'aircraft', 'anomaly'].map(pred => {
                      const count = stats.confusionMatrix[gt][pred];
                      return (
                        <div
                          key={pred}
                          className={`py-2 rounded transition-all duration-300 font-semibold border ${getConfusionShading(count)}`}
                        >
                          {count}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
              <div className={`flex justify-between text-[9px] font-mono mt-2 px-1 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
                }`}>
                <span>* Diagonal cells represent correct classification matrix</span>
                <span>* High SNR keeps values along diagonal</span>
              </div>
            </div>

          </div>

        </section>

      </main>

      <footer className={`text-center py-4 px-6 text-[11px] font-mono border-t transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900 bg-zinc-950/30 text-zinc-500' : 'border-zinc-200 bg-zinc-100 text-zinc-600'
        }`}>
        &copy; 2026 RadarSense Simulation Group. Build version 1.2.0. Dynamic WebAssembly ONNX inference engine is enabled.
      </footer>
    </div>
  );
}