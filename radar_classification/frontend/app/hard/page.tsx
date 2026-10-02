'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import RadarSimulator, { SessionStats } from '../components/RadarSimulator';
import { OnnxModelManager } from '../components/OnnxInference';
import {
  Activity, Cpu, Compass, CheckCircle,
  AlertTriangle, BarChart2,
  Sun, Moon, Usb, Radar, ShieldAlert, MapPin
} from 'lucide-react';

function getShapContributions(features: number[], predictedClass: string): { name: string; impact: number }[] {
  if (!features) return [];

  if (features.length === 11) {
    const featureNames = [
      "doppler_velocity", "doppler_freq_shift", "rcs_estimate",
      "micro_doppler_bandwidth", "micro_doppler_jitter", "range_m",
      "azimuth_deg", "altitude_m", "velocity_variance", "acceleration", "snr_db"
    ];

    const coefs: Record<string, Record<string, number>> = {
      drone: { "micro_doppler_bandwidth": 3.80, "micro_doppler_jitter": 3.20, "velocity_variance": 2.10, "rcs_estimate": -1.90, "snr_db": -0.75 },
      bird: { "micro_doppler_bandwidth": 1.40, "micro_doppler_jitter": 0.90, "rcs_estimate": -3.20, "velocity_variance": 0.50 },
      aircraft: { "rcs_estimate": 4.10, "doppler_velocity": 3.20, "micro_doppler_bandwidth": -2.80, "micro_doppler_jitter": -2.10 },
      anomaly: { "acceleration": 3.40, "velocity_variance": 2.90, "doppler_velocity": 2.20, "rcs_estimate": 1.40 }
    };

    const activeCoefs = coefs[predictedClass] || coefs.drone;

    return featureNames.map((name, i) => {
      const val = features[i];
      let z = 0;
      if (name === "snr_db") z = (val - (-15)) / 5;
      else if (name === "rcs_estimate") z = (val - (-5)) / 10;
      else if (name === "micro_doppler_bandwidth") z = (val - 150) / 100;
      else if (name === "micro_doppler_jitter") z = (val - 0.5) / 0.3;
      else z = val;

      const w = activeCoefs[name] || 0.04 * (Math.sin(i) + 0.1);
      return { name, impact: w * z };
    }).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 4);
  }

  return [];
}

export default function HardwarePage() {
  const [modelManager] = useState(() => new OnnxModelManager());
  const [snrDb, setSnrDb] = useState(-10);
  const [selectedTarget, setSelectedTarget] = useState<any>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loadedModelName, setLoadedModelName] = useState("");
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

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

  return (
    <div className={`min-h-screen font-sans flex flex-col transition-colors duration-300 selection:bg-cyan-500/30 selection:text-cyan-300 ${theme === 'dark' ? 'bg-[#07090b] text-zinc-100' : 'bg-zinc-50 text-zinc-900'
      }`}>

      <header className={`border-b backdrop-blur-md sticky top-0 z-50 py-3.5 px-6 shadow-lg transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900 bg-zinc-950/50' : 'border-zinc-200 bg-white/80'
        }`}>
        <div className="max-w-[1600px] mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-cyan-400 flex items-center justify-center shadow-[0_0_15px_rgba(6,182,212,0.3)] animate-pulse">
              <Usb className="w-5 h-5 text-zinc-950" />
            </div>
            <div>
              <h1 className={`text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r flex items-center gap-2 ${theme === 'dark' ? 'from-zinc-50 via-zinc-100 to-zinc-300' : 'from-zinc-900 via-zinc-800 to-zinc-700'
                }`}>
                RadarSense <span className={`font-mono text-xs border px-1.5 py-0.5 rounded ${theme === 'dark' ? 'border-cyan-800/40 bg-cyan-950/20 text-cyan-400' : 'border-cyan-200 bg-cyan-50 text-cyan-600'
                  }`}>HARDWARE-ONLY</span>
              </h1>
              <p className={`text-[10px] font-mono uppercase tracking-wider ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                }`}>Physical Radar Sensor Scanning & Telemetry Inference</p>
            </div>

            <nav className={`flex items-center gap-1 border p-1 rounded-xl font-mono text-xs ml-4 ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-200 bg-white'}`}>
              <Link
                href="/"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
              >
                <Activity className="w-3.5 h-3.5" /> Simulation Radar
              </Link>
              <Link
                href="/hard"
                className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all bg-cyan-500/20 text-cyan-400 font-bold border border-cyan-500/30 shadow-sm"
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
              <Cpu className={`w-4 h-4 ${isModelLoaded ? 'text-cyan-400 animate-spin' : 'text-zinc-400'}`} />
              <div className="text-left font-mono">
                <div className={`text-[9px] uppercase ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-400'}`}>Model Engine</div>
                <div className={`text-xs font-semibold ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                  {isModelLoaded ? (loadedModelName === "Quantum Server (QSVC)" ? loadedModelName : `ONNX (${loadedModelName})`) : 'Built-in Simulator Stub'}
                </div>
              </div>
            </div>

            <div className="hidden lg:flex flex-col text-right font-mono">
              <span className={`text-[10px] uppercase ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-400'}`}>Hardware Mode</span>
              <span className="text-xs font-semibold text-cyan-400">Web Serial API Active</span>
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
            hardwareOnly={true}
          />
        </section>

        <section className="xl:col-span-5 flex flex-col gap-6">
          <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex-1 flex flex-col transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/70 border-cyan-950/40' : 'bg-white border-zinc-200'
            }`}>
            <div className={`flex justify-between items-center mb-4 border-b pb-2 transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
              }`}>
              <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                }`}>
                <Activity className="w-4 h-4 text-cyan-400" /> Hardware Target Inspector
              </span>
            </div>

            {!selectedTarget ? (
              <div className={`flex-1 flex flex-col items-center justify-center text-center p-8 border border-dashed rounded-xl transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/5' : 'border-zinc-200 bg-zinc-50'
                }`}>
                <Compass className="w-12 h-12 text-zinc-700 mb-3 animate-pulse" />
                <h3 className={`text-sm font-semibold font-mono transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                  }`}>No Hardware Target Detected</h3>
                <p className={`text-xs max-w-[340px] mt-1.5 font-sans leading-relaxed transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
                  }`}>
                  Connect Arduino / Ultrasonic sensor via Web Serial API or send manual telemetry to view live classification results.
                </p>
              </div>
            ) : (
              <div className="flex-grow flex flex-col gap-5">
                <div className={`border rounded-xl p-3.5 text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-100 border-zinc-200'
                  }`}>
                  <div className="text-[10px] text-zinc-500 font-mono uppercase mb-1">Live Classification</div>
                  <div className={`text-lg font-bold font-mono tracking-wide uppercase ${selectedTarget.lastPrediction
                      ? selectedTarget.lastPrediction.predictedClass === 'drone' ? 'text-red-500'
                        : selectedTarget.lastPrediction.predictedClass === 'aircraft' ? 'text-blue-500'
                          : selectedTarget.lastPrediction.predictedClass === 'bird' ? 'text-amber-500'
                            : 'text-purple-500'
                      : 'text-zinc-400'
                    }`}>
                    {selectedTarget.lastPrediction ? selectedTarget.lastPrediction.predictedClass : 'AWAITING SCAN'}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/70 border-cyan-950/40' : 'bg-white border-zinc-200'
            }`}>
            <div className={`flex justify-between items-center border-b pb-2 transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
              }`}>
              <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                }`}>
                <BarChart2 className="w-4 h-4 text-cyan-400" /> Hardware Detection Dials
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2.5">
              <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                }`}>
                <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Scans</div>
                <div className={`text-sm font-bold font-mono mt-0.5 ${theme === 'dark' ? 'text-zinc-200' : 'text-zinc-700'}`}>
                  {stats.totalDetections}
                </div>
              </div>

              <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                }`}>
                <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Mode</div>
                <div className="text-sm font-bold font-mono mt-0.5 text-cyan-400">
                  SERIAL
                </div>
              </div>

              <div className={`border p-2 rounded-xl text-center transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/30 border-zinc-900' : 'bg-zinc-50 border-zinc-200'
                }`}>
                <div className="text-[8.5px] text-zinc-500 font-mono uppercase">Latency</div>
                <div className="text-sm font-bold font-mono mt-0.5 text-cyan-400">
                  {stats.avgInferenceTimeMs > 0 ? `${stats.avgInferenceTimeMs.toFixed(3)} ms` : '—'}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
