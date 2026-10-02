'use client';

import React, { useRef, useEffect, useMemo, useState } from 'react';
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { OnnxModelManager, InferenceResult } from './OnnxInference';
import { assessThreat, type ThreatAssessment } from './threatAssessment';
import {
  Play, Pause, RefreshCw, Radio, Shield,
  Settings, Upload, BarChart2, Activity,
  CheckCircle, AlertTriangle, Eye, Compass,
  Plus, Minus, Trash2, Cpu, Wifi, WifiOff, Server,
  Usb, Cable, Terminal, Gauge, Zap
} from 'lucide-react';


const WAVEFORM_PARAMS = {
  bird: {
    amp_mean: 0.018, amp_std: 5.50, amp_noise_mult: 1.3, amp_freq: 0.08,
    vel_mean: -0.008, vel_std: 2.83, vel_noise_mult: 0.7, vel_freq: 0.08,
    eng_mean: 0.760, eng_std: 5.45, eng_noise_mult: 1.3, eng_freq: 0.08
  },
  drone: {
    amp_mean: 0.007, amp_std: 4.14, amp_noise_mult: 1.1, amp_freq: 0.35,
    vel_mean: 0.011, vel_std: 3.42, vel_noise_mult: 1.0, vel_freq: 0.35,
    eng_mean: 0.731, eng_std: 3.99, eng_noise_mult: 1.1, eng_freq: 0.35
  },
  aircraft: {
    amp_mean: 0.360, amp_std: 2.46, amp_noise_mult: 1.0, amp_freq: 0.05,
    vel_mean: 0.007, vel_std: 0.225, vel_noise_mult: 1.0, vel_freq: 0.05,
    eng_mean: 0.358, eng_std: 2.46, eng_noise_mult: 1.0, eng_freq: 0.05
  },
  anomaly: {
    amp_mean: -0.017, amp_std: 3.16, amp_noise_mult: 1.0, amp_freq: 0.15,
    vel_mean: 0.001, vel_std: 2.23, vel_noise_mult: 0.6, vel_freq: 0.15,
    eng_mean: 0.400, eng_std: 3.12, eng_noise_mult: 1.0, eng_freq: 0.15
  }
};


function randomNormal(mean: number, std: number): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return num * std + mean;
}


function computeDft(x: number[]): { freqs: number[], mags: number[], powers: number[] } {
  const N = x.length;
  const DT = 0.01;
  const freqs: number[] = [];
  const mags: number[] = [];
  const powers: number[] = [];
  const numBins = Math.floor(N / 2) + 1;

  for (let k = 0; k < numBins; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += x[n] * Math.cos(angle);
      im -= x[n] * Math.sin(angle);
    }
    const mag = Math.sqrt(re * re + im * im);
    freqs.push(k / (N * DT)); 
    mags.push(mag);
    powers.push(mag * mag);
  }
  return { freqs, mags, powers };
}


function extractStatsFeatures(sig: number[]): number[] {
  const N = sig.length;

  
  const mean = sig.reduce((a, b) => a + b, 0) / N;

  
  let min = sig[0];
  let max = sig[0];
  let energy = 0;
  let sumSqDiff = 0;
  for (let i = 0; i < N; i++) {
    const val = sig[i];
    if (val < min) min = val;
    if (val > max) max = val;
    energy += val * val;
    sumSqDiff += Math.pow(val - mean, 2);
  }
  const std = Math.sqrt(sumSqDiff / N);
  const range = max - min;

  
  const centered = sig.map(v => v - mean);
  const signs = centered.map(v => (v > 0 ? 1 : v < 0 ? -1 : 1));
  let zcrCount = 0;
  for (let i = 1; i < N; i++) {
    if (signs[i] !== signs[i - 1]) {
      zcrCount++;
    }
  }
  const zcr = zcrCount / (N - 1);

  
  const { freqs, powers } = computeDft(sig);

  
  let dominant_freq = 0.0;
  let spectral_bandwidth = 0.0;

  let maxPower = -1;
  let dominantIdx = 0;
  let power_sum = 0;
  let centroid_num = 0;

  for (let k = 0; k < powers.length; k++) {
    const pwr = powers[k];
    power_sum += pwr;
    centroid_num += freqs[k] * pwr;
    if (pwr > maxPower) {
      maxPower = pwr;
      dominantIdx = k;
    }
  }

  dominant_freq = freqs[dominantIdx];
  if (power_sum > 1e-12) {
    const centroid = centroid_num / power_sum;
    let sumDiffSq = 0;
    for (let k = 0; k < powers.length; k++) {
      sumDiffSq += Math.pow(freqs[k] - centroid, 2) * powers[k];
    }
    spectral_bandwidth = Math.sqrt(sumDiffSq / power_sum);
  }

  
  const diffs: number[] = [];
  let diffMean = 0;
  for (let i = 1; i < N; i++) {
    const d = sig[i] - sig[i - 1];
    diffs.push(d);
    diffMean += d;
  }
  diffMean /= (N - 1);

  let diffVarSum = 0;
  for (let i = 0; i < diffs.length; i++) {
    diffVarSum += Math.pow(diffs[i] - diffMean, 2);
  }
  const diff_var = diffVarSum / diffs.length;

  return [
    mean,
    std,
    min,
    max,
    range,
    energy,
    zcr,
    dominant_freq,
    spectral_bandwidth,
    diff_var
  ];
}






export interface KinematicSample {
  x: number; y: number; vx: number; vy: number; ax: number; ay: number;
  altitude: number;
  baseRcs: number; baseUdbw: number; baseUdjit: number; baseVelvar: number;
  type: 'bird' | 'drone' | 'aircraft' | 'anomaly';
}








export function computeFeatureVectors(
  t: KinematicSample,
  currentSnr: number
): { featureVector11: number[]; featureVector31: number[] } {
  const r = Math.sqrt(t.x * t.x + t.y * t.y);
  const azimuthRad = Math.atan2(t.x, t.y);
  const azimuthDeg = azimuthRad * 180 / Math.PI;

  const doppler_velocity = -(t.x * t.vx + t.y * t.vy) / r;
  const snr_db = currentSnr;

  const N = 100;
  const DT = 0.01;
  const cls = t.type;

  const rcs = t.baseRcs;
  const md_bw = t.baseUdbw;
  const md_jit = t.baseUdjit;

  const base_amp = Math.sqrt(Math.pow(10, rcs / 10.0));
  const mod_freq = md_bw / 4.0;
  const mod_depth = md_jit;
  const phase = Math.random() * 2 * Math.PI;

  const noise_frac = Math.max(0, (-10.0 - currentSnr) / 15.0);
  const randn = () => randomNormal(0, 1);

  const amplitude: number[] = [];
  const velocity: number[] = [];
  const energy: number[] = [];

  let theta = 0.5, sigma = 1.0;
  if (cls === 'bird') { theta = 0.6; sigma = 2.5; }
  else if (cls === 'drone') { theta = 0.9; sigma = 3.0; }
  else if (cls === 'aircraft') { theta = 0.05; sigma = 0.4; }
  else { theta = 0.3; sigma = 1.0; }

  let currentVel = doppler_velocity;
  for (let i = 0; i < N; i++) {
    const t_i = i * DT;
    const env = 1.0 + mod_depth * Math.sin(2 * Math.PI * mod_freq * t_i + phase);
    let amp_i = base_amp * Math.max(0.05, env) + 0.03 * base_amp * randn();

    const dv = theta * (doppler_velocity - currentVel) * DT + sigma * Math.sqrt(DT) * randn();
    currentVel += dv;
    let vel_i = currentVel;
    if (cls === 'bird') {
      vel_i += 1.5 * Math.sin(2 * Math.PI * 8.0 * t_i);
    }

    amp_i += noise_frac * 0.25 * (base_amp + 1e-6) * randn();
    vel_i += noise_frac * 4.0 * randn();

    const eng_i = amp_i * amp_i + noise_frac * 0.35 * (base_amp * base_amp + 1e-6) * randn();

    amplitude.push(amp_i);
    velocity.push(vel_i);
    energy.push(eng_i);
  }

  const injectNoise = (val: number, std: number) => {
    const noiseMult = Math.max(0.1, 1.0 + (10 - currentSnr) / 15.0);
    return val + randomNormal(0, std * noiseMult);
  };

  const amp_features = extractStatsFeatures(amplitude);
  const vel_features = extractStatsFeatures(velocity);
  const eng_features = extractStatsFeatures(energy);

  const featureVector31 = [snr_db, ...amp_features, ...vel_features, ...eng_features];

  const noisy_doppler_velocity = Math.max(-50, Math.min(50, injectNoise(doppler_velocity, 0.5)));
  const noisy_doppler_freq_shift = (2 * noisy_doppler_velocity * 10e9) / 3e8;
  const noisy_rcs = Math.max(-20, Math.min(20, injectNoise(t.baseRcs, 1.0)));
  const noisy_udbw = Math.max(0, Math.min(500, injectNoise(t.baseUdbw, 20.0)));
  const noisy_udjit = Math.max(0, Math.min(1, injectNoise(t.baseUdjit, 0.05)));
  const noisy_range = Math.max(50, Math.min(5000, injectNoise(r, 10.0)));
  const noisy_azimuth = Math.max(-180, Math.min(180, injectNoise(azimuthDeg, 0.5)));
  const noisy_altitude = Math.max(0, Math.min(3000, injectNoise(t.altitude, 15.0)));
  const noisy_velvar = Math.max(0, injectNoise(t.baseVelvar, 1.0));

  const speed = Math.sqrt(t.vx * t.vx + t.vy * t.vy);
  const acc = speed > 0 ? (t.ax * t.vx + t.ay * t.vy) / speed : 0;
  const noisy_acc = Math.max(-10, Math.min(10, injectNoise(acc, 0.2)));

  const featureVector11 = [
    noisy_doppler_velocity, noisy_doppler_freq_shift, noisy_rcs, noisy_udbw, noisy_udjit,
    noisy_range, noisy_azimuth, noisy_altitude, noisy_velvar, noisy_acc, snr_db,
  ];

  return { featureVector11, featureVector31 };
}






export type TargetSubtype =
  | 'Quadcopter' | 'Hexacopter' | 'Fixed-Wing UAV' | 'Nano Drone'
  | 'Airliner' | 'Fighter Jet' | 'Helicopter' | 'Cargo Plane'
  | 'Eagle' | 'Flock' | 'Seagull'
  | 'Weather Balloon' | 'Debris' | 'Unknown Object';

export const SUBTYPE_CLASS: Record<TargetSubtype, 'bird' | 'drone' | 'aircraft' | 'anomaly'> = {
  'Quadcopter': 'drone', 'Hexacopter': 'drone', 'Fixed-Wing UAV': 'drone', 'Nano Drone': 'drone',
  'Airliner': 'aircraft', 'Fighter Jet': 'aircraft', 'Helicopter': 'aircraft', 'Cargo Plane': 'aircraft',
  'Eagle': 'bird', 'Flock': 'bird', 'Seagull': 'bird',
  'Weather Balloon': 'anomaly', 'Debris': 'anomaly', 'Unknown Object': 'anomaly',
};


export interface SubtypePreset {
  rcs: number; udbw: number; udjit: number; velvar: number;
  altMin: number; altMax: number; speedMin: number; speedMax: number;
}
export const SUBTYPE_PRESETS: Record<TargetSubtype, SubtypePreset> = {
  
  'Quadcopter': { rcs: -14, udbw: 240, udjit: 0.72, velvar: 12.0, altMin: 30, altMax: 500, speedMin: 2, speedMax: 16 },
  'Hexacopter': { rcs: -11, udbw: 260, udjit: 0.78, velvar: 14.0, altMin: 50, altMax: 600, speedMin: 2, speedMax: 14 },
  'Fixed-Wing UAV': { rcs: -12, udbw: 210, udjit: 0.65, velvar: 8.0, altMin: 100, altMax: 1200, speedMin: 12, speedMax: 35 },
  'Nano Drone': { rcs: -16, udbw: 280, udjit: 0.82, velvar: 16.0, altMin: 10, altMax: 200, speedMin: 1, speedMax: 10 },

  
  'Airliner': { rcs: 16, udbw: 12, udjit: 0.06, velvar: 0.2, altMin: 2000, altMax: 3000, speedMin: 40, speedMax: 50 },
  'Fighter Jet': { rcs: 10, udbw: 18, udjit: 0.10, velvar: 0.5, altMin: 800, altMax: 2800, speedMin: 35, speedMax: 55 },
  'Helicopter': { rcs: 8, udbw: 22, udjit: 0.12, velvar: 0.8, altMin: 50, altMax: 1500, speedMin: 5, speedMax: 30 },
  'Cargo Plane': { rcs: 18, udbw: 10, udjit: 0.05, velvar: 0.1, altMin: 1500, altMax: 2800, speedMin: 30, speedMax: 45 },

  
  'Eagle': { rcs: -14, udbw: 50, udjit: 0.45, velvar: 3.2, altMin: 30, altMax: 400, speedMin: 8, speedMax: 22 },
  'Flock': { rcs: -15, udbw: 55, udjit: 0.48, velvar: 4.0, altMin: 20, altMax: 250, speedMin: 10, speedMax: 20 },
  'Seagull': { rcs: -18, udbw: 40, udjit: 0.38, velvar: 2.5, altMin: 5, altMax: 150, speedMin: 6, speedMax: 18 },

  
  'Weather Balloon': { rcs: -2, udbw: 8, udjit: 0.08, velvar: 0.4, altMin: 500, altMax: 3000, speedMin: 1, speedMax: 5 },
  'Debris': { rcs: -4, udbw: 12, udjit: 0.12, velvar: 0.6, altMin: 100, altMax: 2000, speedMin: 10, speedMax: 40 },
  'Unknown Object': { rcs: 0, udbw: 15, udjit: 0.14, velvar: 0.8, altMin: 200, altMax: 2500, speedMin: 8, speedMax: 50 },
};

export interface Target {
  id: number;
  type: 'bird' | 'drone' | 'aircraft' | 'anomaly';
  subtype: TargetSubtype;
  x: number;
  y: number;
  vx: number;
  vy: number;
  ax: number;
  ay: number;
  altitude: number;
  baseRcs: number;
  baseUdbw: number;
  baseUdjit: number;
  baseVelvar: number;
  history: { x: number; y: number; time: number }[];
  lastDetectedFeatures: number[] | null;
  
  
  
  lastDetectedFeatures11: number[] | null;
  lastPrediction: InferenceResult | null;
  lastDetectTime: number;
}


interface RadarSimulatorProps {
  modelManager: OnnxModelManager;
  snrDb: number;
  setSnrDb: (val: number) => void;
  selectedTarget: Target | null;
  setSelectedTarget: (target: Target | null) => void;
  onStatsUpdate: (stats: SessionStats) => void;
  isModelLoaded: boolean;
  onModelLoaded: (name: string) => void;
  theme: 'dark' | 'light';
  hideHardware?: boolean;
  hardwareOnly?: boolean;
}

export interface SessionStats {
  totalDetections: number;
  correctDetections: number;
  accuracy: number;
  confusionMatrix: Record<string, Record<string, number>>; 
  classStats: Record<string, { detections: number; correct: number; recall: number; precision: number }>;
  avgInferenceTimeMs: number;
}

export default function RadarSimulator({
  modelManager,
  snrDb,
  setSnrDb,
  selectedTarget,
  setSelectedTarget,
  onStatsUpdate,
  isModelLoaded,
  onModelLoaded,
  theme,
  hideHardware = false,
  hardwareOnly = false
}: RadarSimulatorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const modelManagerRef = useRef<OnnxModelManager>(modelManager);

  
  useEffect(() => {
    modelManagerRef.current = modelManager;
  }, [modelManager]);

  const [isPaused, setIsPaused] = useState(false);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [modelSourceName, setModelSourceName] = useState<string>("Built-in Simulator Stub");
  const [dragActive, setDragActive] = useState(false);
  const [displayMode, setDisplayMode] = useState<'2D' | '3D'>('3D');

  const [inferenceMode, setInferenceMode] = useState<'classical' | 'quantum'>('classical');
  const inferenceModeRef = useRef<'classical' | 'quantum'>(inferenceMode);

  const [useEnsemble, setUseEnsemble] = useState(true);
  const useEnsembleRef = useRef<boolean>(useEnsemble);

  useEffect(() => { inferenceModeRef.current = inferenceMode; }, [inferenceMode]);
  useEffect(() => { useEnsembleRef.current = useEnsemble; }, [useEnsemble]);

  const [quantumServerStatus, setQuantumServerStatus] = useState<{
    online: boolean;
    metadata: any;
    checking: boolean;
  }>({ online: false, metadata: null, checking: false });

  
  const [hardwareMode, setHardwareMode] = useState<boolean>(hardwareOnly || false);
  const hardwareModeRef = useRef(false);
  useEffect(() => { hardwareModeRef.current = hardwareMode; }, [hardwareMode]);
  const [serialConnected, setSerialConnected] = useState<boolean>(false);
  const [baudRate, setBaudRate] = useState<number>(115200);
  const [lastServoAngle, setLastServoAngle] = useState<number | null>(null);
  const [lastDistance, setLastDistance] = useState<number | null>(null);
  const [serialLogs, setSerialLogs] = useState<string[]>([]);
  const [manualSerialInput, setManualSerialInput] = useState<string>('');

  const serialPortRef = useRef<any>(null);
  const serialReaderRef = useRef<any>(null);
  const lastServoAngleRef = useRef<number | null>(null);
  const lastDistanceRef = useRef<number | null>(null);
  const lastHwPosRef = useRef<{ range: number; time: number } | null>(null);

  useEffect(() => {
    lastServoAngleRef.current = lastServoAngle;
  }, [lastServoAngle]);

  useEffect(() => {
    lastDistanceRef.current = lastDistance;
  }, [lastDistance]);

  const processSerialLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    
    if (/[^\x20-\x7E]/.test(trimmed)) return;

    setSerialLogs(prev => [trimmed, ...prev.slice(0, 49)]);

    let updatedAngle = lastServoAngleRef.current;
    let updatedDist = lastDistanceRef.current;
    let hasNewData = false;

    
    
    
    const csvMatch = trimmed.match(/^\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*$/);
    if (csvMatch) {
      const angle = parseFloat(csvMatch[1]);
      const dist = parseFloat(csvMatch[2]);
      
      if (!isNaN(angle) && angle >= 0 && angle <= 180) {
        updatedAngle = angle;
        setLastServoAngle(angle);
        hasNewData = true;
      }
      if (!isNaN(dist) && dist >= 0 && dist <= 500) {
        updatedDist = dist;
        if (dist > 0) setLastDistance(dist);
      }
    }

    
    if (!hasNewData) {
      const servoMatch = trimmed.match(/servo\s*[:=]?\s*([0-9.-]+)/i);
      if (servoMatch) {
        const angle = parseFloat(servoMatch[1]);
        if (!isNaN(angle) && angle >= 0 && angle <= 180) {
          updatedAngle = angle;
          setLastServoAngle(angle);
          hasNewData = true;
        }
      }
      const distMatch = trimmed.match(/distance\s*[:=]?\s*([0-9.-]+)/i);
      if (distMatch) {
        const dist = parseFloat(distMatch[1]);
        if (!isNaN(dist) && dist >= 0 && dist <= 500) {
          updatedDist = dist;
          if (dist > 0) setLastDistance(dist);
          hasNewData = true;
        }
      }
    }

    if (hasNewData) {
      handleHardwareReading(updatedAngle, updatedDist);
    }
  };

  const handleHardwareReading = async (angle: number | null, distance: number | null) => {
    const angleVal = angle ?? 0;
    const distVal = distance ?? 0;

    
    const hasDetection = distVal > 0;

    
    
    
    const radarBearing = (270 + angleVal * 2) % 360;
    sweepAngleRef.current = radarBearing;

    
    if (hardwareMode && targetsRef.current.some(t => t.id !== 99999)) {
      targetsRef.current = targetsRef.current.filter(t => t.id === 99999);
      setTargetCount(targetsRef.current.length);
      setSceneTargets([...targetsRef.current]);
    }

    if (!hasDetection) return;

    
    
    const rangeM = Math.max(50, Math.min(5000, distVal * 12.5));
    const altM = Math.min(1500, Math.max(80, rangeM * 0.15 + (Math.random() - 0.5) * 20));

    
    const azimuthDeg = Math.max(-180, Math.min(180, (angleVal - 45) * 2));

    
    const now = Date.now();
    let dopplerVel = 0.0;
    if (lastHwPosRef.current) {
      const dtSec = (now - lastHwPosRef.current.time) / 1000;
      if (dtSec > 0.01 && dtSec < 5.0) {
        const dRange = rangeM - lastHwPosRef.current.range;
        dopplerVel = Math.min(50, Math.max(-50, dRange / dtSec));
      }
    }
    lastHwPosRef.current = { range: rangeM, time: now };

    
    
    const dopplerFreqShift = 160 * dopplerVel;
    const speed = Math.abs(dopplerVel);

    
    const rcsEstimate = Math.max(-20, Math.min(20, -14.0 + (distVal / 100.0) * 3.0 + (Math.random() - 0.5) * 2.0));

    
    const microDopplerBw = Math.max(10, Math.min(500, 260 + speed * 5.0 + (Math.random() - 0.5) * 30));
    const microDopplerJitter = Math.max(0, Math.min(1, 0.65 + speed * 0.01 + (Math.random() - 0.5) * 0.08));

    
    const velVar = Math.max(0, Math.abs(dopplerVel) * 1.2 + (Math.random() - 0.5) * 2.0);
    const acc = 0.0; 
    const snrDb = -12.0;

    
    const radAz = (azimuthDeg * Math.PI) / 180;
    const xPos = rangeM * Math.sin(radAz);
    const yPos = rangeM * Math.cos(radAz);

    
    let hwTarget = targetsRef.current.find(t => t.id === 99999);
    if (!hwTarget) {
      const newHwTarget: Target = {
        id: 99999,
        type: 'drone',
        subtype: 'Quadcopter',
        x: xPos,
        y: yPos,
        vx: dopplerVel,
        vy: 0,
        ax: 0,
        ay: 0,
        altitude: altM,
        baseRcs: rcsEstimate,
        baseUdbw: microDopplerBw,
        baseUdjit: microDopplerJitter,
        baseVelvar: velVar,
        history: [],
        lastDetectedFeatures: [],
        lastDetectedFeatures11: null,
        lastPrediction: null,
        lastDetectTime: now
      };
      hwTarget = newHwTarget;
      targetsRef.current.push(newHwTarget);
      setTargetCount(targetsRef.current.length);
      setSceneTargets([...targetsRef.current]);
    } else {
      hwTarget.x = xPos;
      hwTarget.y = yPos;
      hwTarget.vx = dopplerVel;
      hwTarget.altitude = altM;
      hwTarget.baseRcs = rcsEstimate;
      hwTarget.baseUdbw = microDopplerBw;
      hwTarget.baseUdjit = microDopplerJitter;
      hwTarget.baseVelvar = velVar;
      hwTarget.lastDetectTime = now;
      hwTarget.history.push({ x: xPos, y: yPos, time: now });
      if (hwTarget.history.length > 50) hwTarget.history.shift();
    }

    const activeHwTarget: Target = hwTarget;

    
    const featureVector11 = [
      dopplerVel,           
      dopplerFreqShift,     
      rcsEstimate,          
      microDopplerBw,       
      microDopplerJitter,   
      rangeM,               
      azimuthDeg,           
      altM,                 
      velVar,               
      acc,                  
      snrDb                 
    ];

    
    
    const N = 100;
    const DT = 0.01;
    const base_amp = Math.sqrt(Math.pow(10, rcsEstimate / 10.0));
    const mod_freq = microDopplerBw / 4.0;
    const mod_depth = microDopplerJitter;
    const phase = Math.random() * 2 * Math.PI;

    const amplitude: number[] = [];
    const velocity: number[] = [];
    const energy: number[] = [];
    let currentVel = dopplerVel;

    for (let i = 0; i < N; i++) {
      const t_i = i * DT;
      const env = 1.0 + mod_depth * Math.sin(2 * Math.PI * mod_freq * t_i + phase);
      const amp_i = base_amp * Math.max(0.05, env) + 0.03 * base_amp * randn();
      const dv = 0.9 * (dopplerVel - currentVel) * DT + 3.0 * Math.sqrt(DT) * randn();
      currentVel += dv;
      amplitude.push(amp_i);
      velocity.push(currentVel);
      energy.push(amp_i * amp_i);
    }
    const amp_features = extractStatsFeatures(amplitude);
    const vel_features = extractStatsFeatures(velocity);
    const eng_features = extractStatsFeatures(energy);
    const featureVector31 = [snrDb, ...amp_features, ...vel_features, ...eng_features];

    
    const is11 = (modelManagerRef.current as any).expectedFeatures === 11;
    activeHwTarget.lastDetectedFeatures = is11 ? featureVector11 : featureVector31;
    activeHwTarget.lastDetectedFeatures11 = featureVector11;

    
    try {
      const result = await modelManagerRef.current.predict(
        featureVector31,
        featureVector11,
        'drone',
        snrDb,
        inferenceModeRef.current === 'quantum',
        useEnsembleRef.current
      );
      activeHwTarget.lastPrediction = result;
      
      if (result.predictedClass) {
        activeHwTarget.type = result.predictedClass as any;
      }
      setSceneTargets([...targetsRef.current]);
    } catch (e) {
      console.error("Hardware target inference error:", e);
    }
  };

  const [serialError, setSerialError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<{ title: string; message: string; type: 'error' | 'success' | 'info' } | null>(null);

  const showToast = (title: string, message: string, type: 'error' | 'success' | 'info' = 'info') => {
    setToastMessage({ title, message, type });
    setTimeout(() => {
      setToastMessage(prev => (prev?.title === title && prev?.message === message) ? null : prev);
    }, 5000);
  };

  const connectSerialPort = async () => {
    if (typeof window === 'undefined' || !('serial' in (navigator as any))) {
      showToast("Environment Error", "Web Serial API is not supported in this browser environment. Use Google Chrome or Microsoft Edge, or test with the Manual Serial Input box below.", "error");
      return;
    }

    setSerialError(null);

    
    if (serialPortRef.current) {
      await disconnectSerialPort();
    }

    try {
      const port = await (navigator as any).serial.requestPort();

      
      
      
      
      try {
        
        
        await port.open({ baudRate, bufferSize: 4096 });
      } catch (openErr: any) {
        if (openErr.name === 'NetworkError' || openErr.message?.includes('Failed to open')) {
          const msg = "Failed to open COM port. The port may still be locked by a previous session. Try: 1) Unplug and replug the USB cable, 2) Close ALL browser tabs/windows and reopen, or 3) Close Arduino IDE Serial Monitor / PuTTY if open.";
          setSerialError(msg);
          showToast("Serial Port Locked", msg, "error");
        } else {
          setSerialError(`Port Open Error: ${openErr.message || openErr}`);
          showToast("Serial Port Error", `Port Open Error: ${openErr.message || openErr}`, "error");
        }
        return;
      }

      
      
      
      
      try {
        await port.setSignals({ dataTerminalReady: false, requestToSend: false });
      } catch (sigErr) {
        console.warn("Could not set DTR/RTS signals:", sigErr);
      }

      serialPortRef.current = port;
      setSerialConnected(true);
      setHardwareMode(true);
      setSerialError(null);

      
      
      let framingRetries = 0;
      const MAX_FRAMING_RETRIES = 5;
      let linesSkipped = 0;
      const LINES_TO_SKIP = 5; 

      const readStream = async () => {
        const reader = port.readable.getReader();
        serialReaderRef.current = reader;
        const decoder = new TextDecoder('utf-8', { fatal: false });

        let buffer = '';
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) {
              reader.releaseLock();
              break;
            }
            if (value) {
              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';
              for (const line of lines) {
                
                if (linesSkipped < LINES_TO_SKIP) {
                  linesSkipped++;
                  continue;
                }
                processSerialLine(line);
              }
              
              framingRetries = 0;
            }
          }
        } catch (streamErr: any) {
          reader.releaseLock();

          if ((streamErr.name === 'FramingError' || streamErr.message?.includes('Framing error')) && framingRetries < MAX_FRAMING_RETRIES) {
            framingRetries++;
            console.warn(`FramingError #${framingRetries}/${MAX_FRAMING_RETRIES} — retrying in 1s...`);
            setSerialError(`Framing error (retry ${framingRetries}/${MAX_FRAMING_RETRIES})... waiting for clean data from board.`);
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            await readStream();
            return;
          }

          
          setSerialConnected(false);
          if (streamErr.name === 'FramingError' || streamErr.message?.includes('Framing error')) {
            setSerialError("Persistent Framing Error after retries. Check your USB cable and ensure the correct Baud Rate is selected.");
          } else {
            console.warn("Serial stream error:", streamErr);
            setSerialError(`Stream Error: ${streamErr.message || streamErr}`);
          }
        }
      };

      await readStream();
    } catch (err: any) {
      if (err.name === 'NotFoundError') {
        console.log("User cancelled serial port picker.");
      } else {
        console.error("Serial connection error:", err);
        setSerialError(err.message || "Failed to connect to COM port.");
      }
      setSerialConnected(false);
    }
  };

  const disconnectSerialPort = async () => {
    try {
      if (serialReaderRef.current) {
        await serialReaderRef.current.cancel().catch(() => { });
        serialReaderRef.current = null;
      }
      if (serialPortRef.current) {
        await serialPortRef.current.close().catch(() => { });
        serialPortRef.current = null;
      }
    } catch (e) {
      console.error("Error closing serial port:", e);
    }
    setSerialConnected(false);
    setHardwareMode(false);
    
    initTargets();
  };

  
  
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (serialReaderRef.current) {
        serialReaderRef.current.cancel().catch(() => { });
      }
      if (serialPortRef.current) {
        serialPortRef.current.close().catch(() => { });
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      
      handleBeforeUnload();
    };
  }, []);

  useEffect(() => {
    inferenceModeRef.current = inferenceMode;
  }, [inferenceMode]);


  const checkQuantumServer = async () => {
    setQuantumServerStatus(prev => ({ ...prev, checking: true }));
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8081';
      const res = await fetch(`${apiBase}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        setQuantumServerStatus({
          online: true,
          metadata: data,
          checking: false
        });
      } else {
        setQuantumServerStatus({ online: false, metadata: null, checking: false });
      }
    } catch (err) {
      setQuantumServerStatus({ online: false, metadata: null, checking: false });
    }
  };

  useEffect(() => {
    if (inferenceMode === 'quantum') {
      checkQuantumServer();
    }
  }, [inferenceMode]);

  useEffect(() => {
    if (inferenceMode !== 'quantum') return;
    const interval = setInterval(() => {
      checkQuantumServer();
    }, 10000);
    return () => clearInterval(interval);
  }, [inferenceMode]);

  
  useEffect(() => {
    if (inferenceMode === 'quantum') {
      onModelLoaded("Quantum Server (QSVC)");
    } else {
      if (modelManagerRef.current.isModelLoaded()) {
        onModelLoaded(modelSourceName);
      } else {
        onModelLoaded("");
      }
    }
  }, [inferenceMode, modelSourceName, onModelLoaded]);

  
  const targetsRef = useRef<Target[]>([]);
  const sweepAngleRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const isPausedRef = useRef(isPaused);
  const snrDbRef = useRef(snrDb);
  const selectedTargetRef = useRef<Target | null>(selectedTarget);
  const displayModeRef = useRef<'2D' | '3D'>(displayMode);
  const drawRadarRef = useRef<((ctx: CanvasRenderingContext2D, width: number, height: number) => void) | null>(null);

  const themeRef = useRef<'dark' | 'light'>(theme);
  useEffect(() => {
    themeRef.current = theme;
  }, [theme]);

  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { snrDbRef.current = snrDb; }, [snrDb]);
  useEffect(() => { selectedTargetRef.current = selectedTarget; }, [selectedTarget]);
  useEffect(() => { displayModeRef.current = displayMode; }, [displayMode]);

  const totalInferenceTimeRef = useRef<number>(0);

  
  const statsRef = useRef<SessionStats>({
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

  
  const updateStats = () => {
    const s = statsRef.current;

    
    s.accuracy = s.totalDetections > 0 ? (s.correctDetections / s.totalDetections) * 100 : 0;
    s.avgInferenceTimeMs = s.totalDetections > 0 ? (totalInferenceTimeRef.current / s.totalDetections) : 0;

    
    const classes: ('bird' | 'drone' | 'aircraft' | 'anomaly')[] = ['bird', 'drone', 'aircraft', 'anomaly'];

    classes.forEach(gt => {
      
      const actualCount = Object.values(s.confusionMatrix[gt]).reduce((a, b) => a + b, 0);

      
      let predictedCount = 0;
      classes.forEach(c => {
        predictedCount += s.confusionMatrix[c][gt];
      });

      const correctCount = s.confusionMatrix[gt][gt];

      s.classStats[gt].detections = actualCount;
      s.classStats[gt].correct = correctCount;
      s.classStats[gt].recall = actualCount > 0 ? (correctCount / actualCount) * 100 : 0;
      s.classStats[gt].precision = predictedCount > 0 ? (correctCount / predictedCount) * 100 : 0;
    });

    onStatsUpdate({ ...s });
  };

  const resetStats = () => {
    totalInferenceTimeRef.current = 0;
    statsRef.current = {
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
    };
    updateStats();
  };

  
  const randn = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  
  const injectNoise = (val: number, std: number, currentSnr: number) => {
    
    
    
    const snrRef = -10;
    const dbDifference = snrRef - currentSnr; 
    const noiseScale = Math.pow(1.2, dbDifference); 
    return val + randn() * std * noiseScale;
  };

  
  const nextIdRef = useRef<number>(1);
  const [targetCount, setTargetCount] = useState(0);
  const [sceneTargets, setSceneTargets] = useState<Target[]>([]);

  
  const createTarget = (subtype: TargetSubtype): Target => {
    const preset = SUBTYPE_PRESETS[subtype];
    const cls = SUBTYPE_CLASS[subtype];
    const id = nextIdRef.current++;
    const spawnAngle = Math.random() * Math.PI * 2;
    const spawnDist = 1500 + Math.random() * 3000;
    const speed = preset.speedMin + Math.random() * (preset.speedMax - preset.speedMin);
    
    const vAngle = spawnAngle + Math.PI + (Math.random() - 0.5) * 0.8;
    return {
      id,
      type: cls,
      subtype,
      x: Math.cos(spawnAngle) * spawnDist,
      y: Math.sin(spawnAngle) * spawnDist,
      vx: Math.cos(vAngle) * speed,
      vy: Math.sin(vAngle) * speed,
      ax: 0, ay: 0,
      altitude: preset.altMin + Math.random() * (preset.altMax - preset.altMin),
      baseRcs: preset.rcs + (Math.random() - 0.5) * 2,
      baseUdbw: preset.udbw * (0.85 + Math.random() * 0.3),
      baseUdjit: Math.min(1, Math.max(0, preset.udjit + (Math.random() - 0.5) * 0.08)),
      baseVelvar: preset.velvar * (0.8 + Math.random() * 0.4),
      history: [],
      lastDetectedFeatures: null,
      lastDetectedFeatures11: null,
      lastPrediction: null,
      lastDetectTime: 0
    };
  };

  
  const spawnTarget = (subtype: TargetSubtype) => {
    const newTarget = createTarget(subtype);
    targetsRef.current.push(newTarget);
    setTargetCount(targetsRef.current.length);
    setSceneTargets([...targetsRef.current]);
  };

  
  const removeTargetOfClass = (cls: 'bird' | 'drone' | 'aircraft' | 'anomaly') => {
    const idx = targetsRef.current.findLastIndex(t => t.type === cls);
    if (idx !== -1) {
      
      if (selectedTarget && selectedTarget.id === targetsRef.current[idx].id) {
        setSelectedTarget(null);
      }
      targetsRef.current.splice(idx, 1);
      setTargetCount(targetsRef.current.length);
      setSceneTargets([...targetsRef.current]);
    }
  };

  
  const clearAllTargets = () => {
    targetsRef.current = [];
    setSelectedTarget(null);
    setTargetCount(0);
    setSceneTargets([]);
  };

  
  const initTargets = () => {
    nextIdRef.current = 1;
    if (hardwareOnly) {
      targetsRef.current = [];
      setTargetCount(0);
      setSceneTargets([]);
      return;
    }
    targetsRef.current = [
      createTarget('Quadcopter'),
      createTarget('Airliner'),
      createTarget('Eagle'),
      createTarget('Debris'),
      createTarget('Nano Drone'),
      createTarget('Fighter Jet'),
      createTarget('Flock'),
    ];
    setTargetCount(targetsRef.current.length);
    setSceneTargets([...targetsRef.current]);
  };

  useEffect(() => {
    initTargets();
    resetStats();

    
    
    
    let loadedFrom = 'feat31_model.onnx';
    fetch('/models/feat31_model.onnx')
      .then(res => {
        if (res.ok) return res.arrayBuffer();
        loadedFrom = 'radarsense.onnx';
        return fetch('/models/radarsense.onnx').then(r => r.ok ? r.arrayBuffer() : null);
      })
      .then(async buffer => {
        if (buffer) {
          try {
            const success = await modelManagerRef.current.loadModelFromBuffer(buffer);
            if (success) {
              onModelLoaded(`${loadedFrom} (Loaded)`);
              setModelSourceName(loadedFrom);
            }
          } catch (err) {
            console.warn("Auto-load ONNX model error:", err);
          }
        }
      })
      .catch(e => console.warn("Could not auto-fetch default ONNX model:", e));
  }, []);

  
  const updateTargetPhysics = (t: Target, dt: number, currentSnr: number) => {
    const preset = SUBTYPE_PRESETS[t.subtype];
    const maxSpeed = preset.speedMax;
    const minAlt = preset.altMin;
    const maxAlt = preset.altMax;

    if (t.type === 'drone') {
      
      const jitterScale = t.subtype === 'Nano Drone' ? 18 : t.subtype === 'Fixed-Wing UAV' ? 5 : 12;
      t.ax = (Math.random() - 0.5) * jitterScale;
      t.ay = (Math.random() - 0.5) * jitterScale;
      t.altitude = Math.max(minAlt, Math.min(maxAlt, t.altitude + (Math.random() - 0.5) * 4));
      t.vx += t.ax * dt;
      t.vy += t.ay * dt;
    } else if (t.type === 'aircraft') {
      if (t.subtype === 'Helicopter') {
        
        const circleForce = 4.5;
        const angle = Math.atan2(t.vy, t.vx) + Math.PI / 2;
        t.ax = Math.cos(angle) * circleForce + (Math.random() - 0.5) * 3;
        t.ay = Math.sin(angle) * circleForce + (Math.random() - 0.5) * 3;
        t.vx += t.ax * dt;
        t.vy += t.ay * dt;
        t.altitude = Math.max(minAlt, Math.min(maxAlt, t.altitude + (Math.random() - 0.5) * 5));
      } else if (t.subtype === 'Fighter Jet') {
        
        if (Math.random() < 0.03) {
          const turnAngle = (Math.random() - 0.5) * 1.2;
          const s = Math.sqrt(t.vx * t.vx + t.vy * t.vy) || 40;
          const heading = Math.atan2(t.vy, t.vx) + turnAngle;
          t.vx = Math.cos(heading) * s;
          t.vy = Math.sin(heading) * s;
        }
        t.ax = 0; t.ay = 0;
        t.altitude = Math.max(minAlt, Math.min(maxAlt, t.altitude + (Math.random() - 0.5) * 4));
      } else {
        
        t.ax = 0; t.ay = 0;
        t.altitude = Math.max(minAlt, Math.min(maxAlt, t.altitude + (Math.random() - 0.5) * 1.5));
      }
    } else if (t.type === 'bird') {
      
      const angle = Math.atan2(t.vy, t.vx);
      const perpAngle = angle + Math.PI / 2;
      const weaveScale = t.subtype === 'Flock' ? 4.5 : 3;
      const weaveForce = Math.sin(Date.now() / (600 + t.id * 100)) * weaveScale;
      t.ax = Math.cos(perpAngle) * weaveForce;
      t.ay = Math.sin(perpAngle) * weaveForce;
      t.vx += t.ax * dt;
      t.vy += t.ay * dt;
      t.altitude = Math.max(minAlt, Math.min(maxAlt, t.altitude + (Math.random() - 0.5) * 2));
    } else if (t.type === 'anomaly') {
      if (t.subtype === 'Weather Balloon') {
        
        t.ax = (Math.random() - 0.5) * 0.5;
        t.ay = (Math.random() - 0.5) * 0.5;
        t.vx += t.ax * dt;
        t.vy += t.ay * dt;
        t.altitude = Math.min(maxAlt, t.altitude + 0.3);
      } else {
        
        if (Math.random() < 0.05) {
          t.ax = (Math.random() - 0.5) * 35;
          t.ay = (Math.random() - 0.5) * 35;
        }
        t.vx += t.ax * dt;
        t.vy += t.ay * dt;
        t.altitude = Math.max(minAlt, Math.min(maxAlt, t.altitude + (Math.random() - 0.5) * 12));
      }
    }

    
    const speed = Math.sqrt(t.vx * t.vx + t.vy * t.vy);
    if (speed > maxSpeed) {
      t.vx = (t.vx / speed) * maxSpeed;
      t.vy = (t.vy / speed) * maxSpeed;
    }

    
    t.x += t.vx * dt;
    t.y += t.vy * dt;

    
    const r = Math.sqrt(t.x * t.x + t.y * t.y);
    if (r > 5200) {
      
      const spawnAngle = Math.random() * Math.PI * 2;
      t.x = -Math.cos(spawnAngle) * 4900;
      t.y = -Math.sin(spawnAngle) * 4900;

      
      const speed = t.type === 'aircraft' ? 35 + Math.random() * 15 : 6 + Math.random() * 14;
      t.vx = Math.cos(spawnAngle) * speed + (Math.random() - 0.5) * 5;
      t.vy = Math.sin(spawnAngle) * speed + (Math.random() - 0.5) * 5;

      t.history = [];
    }

    
    t.history.push({ x: t.x, y: t.y, time: Date.now() });
    if (t.history.length > 30) {
      t.history.shift();
    }
  };

  
  const executeInferenceSweep = async (t: Target, currentSnr: number) => {
    
    const r = Math.sqrt(t.x * t.x + t.y * t.y);
    const azimuthRad = Math.atan2(t.x, t.y);
    const azimuthDeg = azimuthRad * 180 / Math.PI; 

    
    const doppler_velocity = -(t.x * t.vx + t.y * t.vy) / r; 

    
    const doppler_freq_shift = (2 * doppler_velocity * 10e9) / 3e8; 

    const snr_db = currentSnr;

    
    const N = 100;
    const DT = 0.01;
    const cls = t.type;

    const rcs = t.baseRcs;
    const md_bw = t.baseUdbw;
    const md_jit = t.baseUdjit;

    const base_amp = Math.sqrt(Math.pow(10, rcs / 10.0));
    const mod_freq = md_bw / 4.0;
    const mod_depth = md_jit;
    const phase = Math.random() * 2 * Math.PI;

    
    const noise_frac = Math.max(0, (-10.0 - currentSnr) / 15.0);

    const amplitude: number[] = [];
    const velocity: number[] = [];
    const energy: number[] = [];

    
    let theta = 0.5, sigma = 1.0;
    if (cls === 'bird') { theta = 0.6; sigma = 2.5; }
    else if (cls === 'drone') { theta = 0.9; sigma = 3.0; }
    else if (cls === 'aircraft') { theta = 0.05; sigma = 0.4; }
    else { theta = 0.3; sigma = 1.0; }

    let currentVel = doppler_velocity;
    for (let i = 0; i < N; i++) {
      const t_i = i * DT;
      
      const env = 1.0 + mod_depth * Math.sin(2 * Math.PI * mod_freq * t_i + phase);
      let amp_i = base_amp * Math.max(0.05, env) + 0.03 * base_amp * randn();

      
      const dv = theta * (doppler_velocity - currentVel) * DT + sigma * Math.sqrt(DT) * randn();
      currentVel += dv;
      let vel_i = currentVel;
      if (cls === 'bird') {
        vel_i += 1.5 * Math.sin(2 * Math.PI * 8.0 * t_i);
      }

      
      amp_i += noise_frac * 0.25 * (base_amp + 1e-6) * randn();
      vel_i += noise_frac * 4.0 * randn();

      let eng_i = amp_i * amp_i + noise_frac * 0.35 * (base_amp * base_amp + 1e-6) * randn();

      amplitude.push(amp_i);
      velocity.push(vel_i);
      energy.push(eng_i);
    }

    const injectNoise = (val: number, std: number) => {
      const noiseMult = Math.max(0.1, 1.0 + (10 - currentSnr) / 15.0);
      return val + randomNormal(0, std * noiseMult);
    };

    const amp_features = extractStatsFeatures(amplitude);
    const vel_features = extractStatsFeatures(velocity);
    const eng_features = extractStatsFeatures(energy);

    
    const featureVector31 = [
      snr_db,
      ...amp_features,
      ...vel_features,
      ...eng_features
    ];

    
    const noisy_doppler_velocity = Math.max(-50, Math.min(50, injectNoise(doppler_velocity, 0.5)));
    const noisy_doppler_freq_shift = (2 * noisy_doppler_velocity * 10e9) / 3e8;
    const noisy_rcs = Math.max(-20, Math.min(20, injectNoise(t.baseRcs, 1.0)));
    const noisy_udbw = Math.max(0, Math.min(500, injectNoise(t.baseUdbw, 20.0)));
    const noisy_udjit = Math.max(0, Math.min(1, injectNoise(t.baseUdjit, 0.05)));
    const noisy_range = Math.max(50, Math.min(5000, injectNoise(r, 10.0)));
    const noisy_azimuth = Math.max(-180, Math.min(180, injectNoise(azimuthDeg, 0.5)));
    const noisy_altitude = Math.max(0, Math.min(3000, injectNoise(t.altitude, 15.0)));
    const noisy_velvar = Math.max(0, injectNoise(t.baseVelvar, 1.0));

    const speed = Math.sqrt(t.vx * t.vx + t.vy * t.vy);
    const acc = speed > 0 ? (t.ax * t.vx + t.ay * t.vy) / speed : 0;
    const noisy_acc = Math.max(-10, Math.min(10, injectNoise(acc, 0.2)));

    const featureVector11 = [
      noisy_doppler_velocity,
      noisy_doppler_freq_shift,
      noisy_rcs,
      noisy_udbw,
      noisy_udjit,
      noisy_range,
      noisy_azimuth,
      noisy_altitude,
      noisy_velvar,
      noisy_acc,
      snr_db
    ];

    
    const is11 = (modelManagerRef.current as any).expectedFeatures === 11;
    t.lastDetectedFeatures = is11 ? featureVector11 : featureVector31;
    t.lastDetectedFeatures11 = featureVector11;
    t.lastDetectTime = Date.now();
    
    try {
      const result = await modelManagerRef.current.predict(
        featureVector31,
        featureVector11,
        t.type,
        currentSnr,
        inferenceModeRef.current === 'quantum',
        useEnsembleRef.current
      );
      t.lastPrediction = result;

      
      if (!result.isStub) {
        const s = statsRef.current;
        s.totalDetections += 1;
        totalInferenceTimeRef.current += result.inferenceTime;

        const isCorrect = result.predictedClass === t.type;
        if (isCorrect) {
          s.correctDetections += 1;
        }

        
        s.confusionMatrix[t.type][result.predictedClass] += 1;

        updateStats();
      }
    } catch (err) {
      console.error("Inference execution failed:", err);
    }
  };

  
  
  useEffect(() => {
    let animationFrameId: number;

    const resizeCanvas = () => {
      if (containerRef.current && canvasRef.current) {
        const cssSize = Math.min(
          containerRef.current.clientWidth,
          containerRef.current.clientHeight
        );
        const dpr = window.devicePixelRatio || 1;
        
        canvasRef.current.width = Math.round(cssSize * dpr);
        canvasRef.current.height = Math.round(cssSize * dpr);
        canvasRef.current.style.width = `${cssSize}px`;
        canvasRef.current.style.height = `${cssSize}px`;
      }
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    
    const tick = async (timestamp: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      let dt = (timestamp - lastTimeRef.current) / 1000;
      lastTimeRef.current = timestamp;

      
      if (dt > 0.1) dt = 0.1;

      if (!isPausedRef.current) {
        
        if (!hardwareModeRef.current) {
          targetsRef.current.forEach(t => updateTargetPhysics(t, dt, snrDbRef.current));
        }

        
        const prevSweepAngle = sweepAngleRef.current;
        if (!hardwareModeRef.current) {
          const sweepSpeed = 65;
          sweepAngleRef.current = (sweepAngleRef.current + sweepSpeed * dt) % 360;
        }
        const currentSweepAngle = sweepAngleRef.current;

        
        
        
        
        for (const t of targetsRef.current) {
          const r = Math.sqrt(t.x * t.x + t.y * t.y);
          
          
          
          let bearing = Math.atan2(t.x, t.y) * 180 / Math.PI;
          if (bearing < 0) bearing += 360;

          
          let swept = false;
          if (currentSweepAngle > prevSweepAngle) {
            swept = bearing >= prevSweepAngle && bearing < currentSweepAngle;
          } else {
            
            swept = bearing >= prevSweepAngle || bearing < currentSweepAngle;
          }

          if (swept) {
            
            
            
            void executeInferenceSweep(t, snrDbRef.current);

            
            if (selectedTargetRef.current && selectedTargetRef.current.id === t.id) {
              setSelectedTarget({ ...t });
            }
          }
        }
      }

      
      if (displayModeRef.current === '2D') {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (canvas && ctx) drawRadarRef.current?.(ctx, canvas.width, canvas.height);
      }

      animationFrameId = requestAnimationFrame(tick);
    };

    animationFrameId = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  
  function drawRadar(ctx: CanvasRenderingContext2D, w: number, h: number) {
    const dpr = window.devicePixelRatio || 1;
    const logicalW = w / dpr;
    const logicalH = h / dpr;
    const center = logicalW / 2;
    const maxRadius = logicalW / 2 - 20;
    const isDark = themeRef.current === 'dark';

    
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    
    ctx.fillStyle = isDark ? 'rgba(6, 12, 12, 0.12)' : 'rgba(244, 245, 247, 0.12)';
    ctx.fillRect(0, 0, logicalW, logicalH);

    
    ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.15)' : 'rgba(16, 185, 129, 0.25)';
    ctx.lineWidth = 1;

    
    const isHw = hardwareModeRef.current;

    
    for (let i = 1; i <= 5; i++) {
      const radius = (maxRadius / 5) * i;
      ctx.beginPath();
      if (isHw) {
        
        ctx.arc(center, center, radius, Math.PI, 2 * Math.PI);
      } else {
        ctx.arc(center, center, radius, 0, Math.PI * 2);
      }
      ctx.stroke();

      
      ctx.fillStyle = isDark ? 'rgba(16, 185, 129, 0.4)' : 'rgba(16, 185, 129, 0.7)';
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${i * 1000}m`, center, center - radius + 11);
    }

    
    ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.15)';
    const bearingStart = isHw ? 0 : 0;
    const bearingEnd = isHw ? 180 : 360;
    const bearingStep = isHw ? 15 : 30;
    for (let angle = bearingStart; angle <= bearingEnd; angle += bearingStep) {
      const rad = (angle - 90) * Math.PI / 180;
      const xEnd = center + Math.cos(rad) * maxRadius;
      const yEnd = center + Math.sin(rad) * maxRadius;

      ctx.beginPath();
      ctx.moveTo(center, center);
      ctx.lineTo(xEnd, yEnd);
      ctx.stroke();

      
      const textX = center + Math.cos(rad) * (maxRadius + 11);
      const textY = center + Math.sin(rad) * (maxRadius + 11) + 3;
      ctx.fillStyle = isDark ? 'rgba(16, 185, 129, 0.5)' : 'rgba(16, 185, 129, 0.8)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${angle}°`, textX, textY);
    }

    
    if (isHw) {
      ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.3)' : 'rgba(16, 185, 129, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(center - maxRadius, center);
      ctx.lineTo(center + maxRadius, center);
      ctx.stroke();
      ctx.lineWidth = 1;

      
      ctx.fillStyle = isDark ? 'rgba(52, 211, 153, 0.6)' : 'rgba(16, 185, 129, 0.8)';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('⚡ HARDWARE MODE — 180° SCAN', center, center + 18);
    }

    
    const currentSweepRad = (sweepAngleRef.current - 90) * Math.PI / 180;

    
    ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.85)' : 'rgba(16, 185, 129, 0.95)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(center, center);
    ctx.lineTo(
      center + Math.cos(currentSweepRad) * maxRadius,
      center + Math.sin(currentSweepRad) * maxRadius
    );
    ctx.stroke();

    
    
    ctx.save();
    for (let step = 0; step < 40; step++) {
      const stepRad = currentSweepRad - (step * 0.02);
      ctx.strokeStyle = isDark
        ? `rgba(16, 185, 129, ${0.15 * (1 - step / 40)})`
        : `rgba(16, 185, 129, ${0.2 * (1 - step / 40)})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(center, center);
      ctx.lineTo(
        center + Math.cos(stepRad) * maxRadius,
        center + Math.sin(stepRad) * maxRadius
      );
      ctx.stroke();
    }
    ctx.restore();

    
    let topThreat: { a: ThreatAssessment; t: Target } | null = null;
    const LEVEL_ABBR: Record<string, string> = { CRITICAL: 'CRIT', ELEVATED: 'ELEV', UNCERTAIN: 'UNC', MONITOR: 'MON', CLEAR: 'CLR', PENDING: '—' };
    const rrect = (x: number, y: number, w: number, h: number, r: number) => {
      ctx.beginPath();
      const c = ctx as unknown as { roundRect?: (x: number, y: number, w: number, h: number, r: number) => void };
      if (c.roundRect) c.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
    };

    
    targetsRef.current.forEach(t => {
      
      
      
      const px = center + (t.x / 5000) * maxRadius;
      const py = center - (t.y / 5000) * maxRadius; 

      
      const timeSinceDetect = Date.now() - t.lastDetectTime;
      const glowDuration = 2500; 
      const opacity = Math.max(0.18, 1 - (timeSinceDetect / glowDuration)); 

      
      let color = 'rgba(156, 163, 175, 0.4)'; 
      let glowColor = 'rgba(156, 163, 175, 0.15)';
      const pred = t.lastPrediction ? t.lastPrediction.predictedClass : 'unknown';

      if (pred === 'drone') {
        color = `rgba(239, 68, 68, ${opacity})`; 
        glowColor = 'rgba(239, 68, 68, 0.4)';
      } else if (pred === 'aircraft') {
        color = `rgba(59, 130, 246, ${opacity})`; 
        glowColor = 'rgba(59, 130, 246, 0.4)';
      } else if (pred === 'bird') {
        color = `rgba(245, 158, 11, ${opacity})`; 
        glowColor = 'rgba(245, 158, 11, 0.4)';
      } else if (pred === 'anomaly') {
        color = `rgba(168, 85, 247, ${opacity})`; 
        glowColor = 'rgba(168, 85, 247, 0.4)';
      }

      
      if (timeSinceDetect < glowDuration) {
        ctx.fillStyle = glowColor;
        ctx.beginPath();
        ctx.arc(px, py, 12 * (1 - timeSinceDetect / glowDuration * 0.4), 0, Math.PI * 2);
        ctx.fill();
      }

      
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();

      
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (pred === 'drone') {
        
        ctx.moveTo(px - 10, py); ctx.lineTo(px + 10, py);
        ctx.moveTo(px, py - 10); ctx.lineTo(px, py + 10);
      } else if (pred === 'aircraft') {
        
        ctx.moveTo(px, py - 8);
        ctx.lineTo(px - 7, py + 6);
        ctx.lineTo(px + 7, py + 6);
        ctx.closePath();
      } else if (pred === 'bird') {
        
        ctx.moveTo(px - 8, py - 2);
        ctx.quadraticCurveTo(px - 4, py - 6, px, py - 2);
        ctx.quadraticCurveTo(px + 4, py - 6, px + 8, py - 2);
      } else {
        
        ctx.strokeRect(px - 6, py - 6, 12, 12);
      }
      ctx.stroke();

      
      if (t.lastPrediction) {
        const a = assessThreat(t);
        if (!topThreat || a.priority > topThreat.a.priority) topThreat = { a, t };
        if (a.level !== 'CLEAR' && a.level !== 'PENDING') {
          const label = `${LEVEL_ABBR[a.level]} ${a.score}`;
          ctx.font = 'bold 9px monospace';
          ctx.textAlign = 'left';
          const tw = ctx.measureText(label).width;
          const bw = tw + 18, bh = 13, bx = px + 10, by = py - 22;
          ctx.fillStyle = 'rgba(8, 10, 18, 0.72)';
          rrect(bx, by, bw, bh, 3); ctx.fill();
          ctx.fillStyle = a.color;
          ctx.beginPath(); ctx.arc(bx + 6, by + bh / 2, 2.3, 0, Math.PI * 2); ctx.fill();
          ctx.fillText(label, bx + 11, by + 9.5);
        }
      }

      
      if (selectedTarget && selectedTarget.id === t.id) {
        ctx.strokeStyle = isDark ? '#34d399' : '#059669';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(px, py, 16, 0, Math.PI * 2);
        ctx.stroke();

        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = isDark ? 'rgba(52, 211, 153, 0.5)' : 'rgba(5, 150, 105, 0.6)';
        ctx.beginPath();
        ctx.arc(px, py, 22, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      
      if (showGroundTruth) {
        ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.8)' : 'rgba(9, 9, 11, 0.85)';
        ctx.font = 'bold 9px monospace';
        ctx.textAlign = 'left';

        const textYOffset = 18;
        const confidenceText = t.lastPrediction
          ? ` (${Math.round(t.lastPrediction.confidences[t.lastPrediction.predictedClass] * 100)}%)`
          : '';

        
        ctx.fillText(`TRK-${t.id}`, px + 10, py - 4);

        
        ctx.fillStyle = color;
        ctx.fillText(`P: ${pred.toUpperCase()}${confidenceText}`, px + 10, py + 6);

        
        ctx.fillStyle = isDark ? '#10b981' : '#059669';
        ctx.fillText(`A: ${t.type.toUpperCase()}`, px + 10, py + 16);
      }
    });

    
    if (topThreat) {
      const tt: { a: ThreatAssessment; t: Target } = topThreat;
      if (tt.a.level !== 'CLEAR' && tt.a.level !== 'PENDING') {
        const bl = `HIGHEST THREAT  ·  ${tt.a.level}  ·  TRK-${tt.t.id} ${(tt.t.lastPrediction?.predictedClass ?? '').toUpperCase()}  ·  ${tt.a.score}`;
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'left';
        const tw = ctx.measureText(bl).width;
        const bx = 12, by = 12, bh = 20, bw = tw + 30;
        ctx.fillStyle = 'rgba(8, 10, 18, 0.82)';
        rrect(bx, by, bw, bh, 4); ctx.fill();
        ctx.fillStyle = tt.a.color;
        rrect(bx, by, 4, bh, 2); ctx.fill();
        ctx.beginPath(); ctx.arc(bx + 16, by + bh / 2, 3, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#e8eef9';
        ctx.fillText(bl, bx + 24, by + 14);
      }
    }

    
    const gradient = ctx.createRadialGradient(center, center, 1, center, center, 6);
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.5, isDark ? '#34d399' : '#10b981');
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(center, center, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  useEffect(() => {
    drawRadarRef.current = drawRadar;
  });

  
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const clickX = (e.clientX - rect.left) * dpr;
    const clickY = (e.clientY - rect.top) * dpr;

    const center = canvas.width / 2;
    const maxRadius = canvas.width / 2 - 20;

    
    let closestTarget: Target | null = null;
    let minDistance = 25; 
    targetsRef.current.forEach(t => {
      const px = center + (t.x / 5000) * maxRadius;
      const py = center - (t.y / 5000) * maxRadius;

      const dist = Math.sqrt(Math.pow(clickX - px, 2) + Math.pow(clickY - py, 2));
      if (dist < minDistance) {
        minDistance = dist;
        closestTarget = t;
      }
    });

    setSelectedTarget(closestTarget);
  };

  
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.onnx')) {
        await loadOnnxFile(file);
      } else {
        showToast("Invalid File Format", "Please upload a valid .onnx model file.", "error");
      }
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.name.endsWith('.onnx')) {
        await loadOnnxFile(file);
      } else {
        showToast("Invalid File Format", "Please upload a valid .onnx model file.", "error");
      }
    }
  };

  const loadOnnxFile = async (file: File) => {
    try {
      setModelSourceName("Loading ONNX Model...");
      const arrayBuffer = await file.arrayBuffer();
      await modelManagerRef.current.loadModelFromBuffer(arrayBuffer);
      onModelLoaded(file.name);
      setModelSourceName(file.name);
      resetStats(); 
      showToast("Model Loaded", `Model "${file.name}" loaded successfully into onnxruntime-web! Running live client-side inference.`, "success");
    } catch (e) {
      console.error(e);
      setModelSourceName("Error loading model");
      showToast("Model Load Failed", "Failed to load ONNX model. Check browser console logs for details.", "error");
    }
  };

  const unloadCustomModel = () => {
    modelManagerRef.current.unloadModel();
    onModelLoaded("");
    setModelSourceName("Built-in Simulator Stub");
    resetStats();
  };

  return (
    <div className="flex flex-col gap-6 w-full relative">
      {toastMessage && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-xl border shadow-2xl backdrop-blur-md max-w-md transition-all font-mono text-xs ${toastMessage.type === 'error'
          ? 'bg-red-950/90 border-red-500/50 text-red-200'
          : toastMessage.type === 'success'
            ? 'bg-emerald-950/90 border-emerald-500/50 text-emerald-200'
            : 'bg-zinc-900/90 border-zinc-700 text-zinc-200'
          }`}>
          <div className="flex justify-between items-start gap-3">
            <div>
              <div className="font-bold text-sm mb-1">{toastMessage.title}</div>
              <div>{toastMessage.message}</div>
            </div>
            <button onClick={() => setToastMessage(null)} className="text-zinc-400 hover:text-white text-base">✕</button>
          </div>
        </div>
      )}

      {(!modelManagerRef.current.isModelLoaded() && inferenceMode !== 'quantum') && (
        <div className="w-full bg-amber-500/15 border border-amber-500/40 text-amber-300 px-4 py-3 rounded-xl flex items-center justify-between font-mono text-xs shadow-lg">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
            <div>
              <span className="font-bold text-amber-400 uppercase tracking-wider">DEMO MODE — SYNTHETIC STUB ACTIVE:</span> No ONNX model loaded. Live metrics recording is disabled until an ONNX model file is dragged in.
            </div>
          </div>
        </div>
      )}

      <div className={`relative rounded-2xl p-4 backdrop-blur-md shadow-2xl flex flex-col items-center border transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/70 border-emerald-950/50' : 'bg-white border-zinc-200'
        }`}>
        <div className={`flex w-full justify-between items-center mb-3 px-1 border-b pb-2 transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
          }`}>
          <div className="flex items-center gap-2">
            <Radio className={`w-4 h-4 text-emerald-400 ${!isPaused ? 'animate-pulse' : ''}`} />
            <span className={`font-mono text-xs tracking-wider uppercase transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
              }`}>Live Tactical Sweep</span>
            <select
              value={displayMode}
              onChange={(e) => setDisplayMode(e.target.value as '2D' | '3D')}
              className={`ml-3 font-mono text-[10px] px-2 py-0.5 rounded focus:outline-none focus:border-emerald-500 cursor-pointer transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900 border-zinc-800 text-zinc-300' : 'bg-zinc-100 border-zinc-200 text-zinc-700'
                }`}
            >
              <option value="2D">2D PLANE VIEW</option>
              <option value="3D">3D PERSPECTIVE</option>
            </select>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setIsPaused(!isPaused)}
              className={`p-1.5 rounded-lg border transition-all font-mono text-xs flex items-center gap-1.5 cursor-pointer ${theme === 'dark'
                ? 'border-zinc-800 text-zinc-400 hover:text-emerald-400 hover:border-emerald-900/50'
                : 'border-zinc-200 text-zinc-600 bg-zinc-50 hover:bg-zinc-100 hover:text-emerald-600'
                }`}
              title={isPaused ? "Play" : "Pause"}
            >
              {isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
              <span>{isPaused ? 'RESUME' : 'FREEZE'}</span>
            </button>

            <button
              onClick={resetStats}
              className={`p-1.5 rounded-lg border transition-all font-mono text-xs flex items-center gap-1.5 cursor-pointer ${theme === 'dark'
                ? 'border-zinc-800 text-zinc-400 hover:text-emerald-400 hover:border-emerald-900/50'
                : 'border-zinc-200 text-zinc-600 bg-zinc-50 hover:bg-zinc-100 hover:text-emerald-600'
                }`}
              title="Reset Stats"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>RESET</span>
            </button>

            <button
              onClick={() => setShowGroundTruth(!showGroundTruth)}
              className={`p-1.5 rounded-lg border transition-all font-mono text-xs flex items-center gap-1.5 cursor-pointer ${showGroundTruth
                ? (theme === 'dark' ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-400' : 'bg-emerald-50 border-emerald-500/40 text-emerald-600')
                : (theme === 'dark' ? 'border-zinc-800 text-zinc-400 hover:text-emerald-400' : 'border-zinc-200 text-zinc-600 bg-zinc-50 hover:bg-zinc-100 hover:text-emerald-600')
                }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>OVERLAY</span>
            </button>
          </div>
        </div>

        <div ref={containerRef} className={`w-full aspect-square relative overflow-hidden flex justify-center items-center rounded-xl cursor-crosshair border transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950 border-zinc-900' : 'bg-zinc-100 border-zinc-200'
          }`}>
          {displayMode === '3D' ? (
            <RadarScene3D
              targets={sceneTargets}
              sweepAngleRef={sweepAngleRef}
              selectedId={selectedTarget?.id ?? null}
              showOverlay={showGroundTruth}
              paused={isPaused}
              onSelect={setSelectedTarget}
              theme={theme}
              hardwareMode={hardwareMode}
            />
          ) : (
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className="block"
            />
          )}
          {!isPaused && (
            <div className={`absolute top-4 left-4 font-mono text-[10px] px-2 py-1 rounded border flex items-center gap-1.5 transition-colors duration-300 ${theme === 'dark'
              ? 'text-emerald-500/60 bg-zinc-950/80 border-emerald-900/30'
              : 'text-emerald-600 bg-white/90 border-emerald-200 shadow-sm'
              }`}>
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-ping"></span>
              SWEEP ACTIVE (75°/S)
            </div>
          )}
        </div>

        <div className={`flex gap-6 mt-3 text-[10px] font-mono transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
          }`}>
          <div className="flex items-center gap-1.5">
            <span className={`w-2.5 h-2.5 rounded-full border border-red-500 ${theme === 'dark' ? 'bg-red-950/30' : 'bg-red-50'
              }`}></span>
            <span>DRONE (LOW RCS)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2.5 h-2.5 rounded-sm border border-blue-500 ${theme === 'dark' ? 'bg-blue-950/30' : 'bg-blue-50'
              }`}></span>
            <span>AIRCRAFT (HIGH RCS)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2.5 h-2.5 rounded-full border border-amber-500 ${theme === 'dark' ? 'bg-amber-950/30' : 'bg-amber-50'
              }`}></span>
            <span>BIRD (MEDIUM)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className={`w-2.5 h-2.5 rounded-sm border border-purple-500 ${theme === 'dark' ? 'bg-purple-950/30' : 'bg-purple-50'
              }`}></span>
            <span>STEALTH UAV (ERRATIC)</span>
          </div>
        </div>
      </div>

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-5 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/70 border-emerald-950/50' : 'bg-white border-zinc-200 shadow-sm'
        }`}>

        <div>
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span className={`font-mono text-sm font-semibold tracking-wider transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                }`}>SNR Noise Stress-Test</span>
            </div>
            <span className={`font-mono text-sm font-bold ${snrDb > -14 ? 'text-emerald-400' : snrDb > -21 ? 'text-amber-400' : 'text-red-500'
              }`}>
              {snrDb} dB
            </span>
          </div>

          <input
            type="range"
            min="-25"
            max="-10"
            step="0.5"
            value={snrDb}
            onChange={(e) => setSnrDb(parseFloat(e.target.value))}
            className={`w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-emerald-500 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-800' : 'bg-zinc-200'
              }`}
          />

          <div className={`flex justify-between text-[9px] font-mono mt-1 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
            }`}>
            <span>-25dB (CRITICAL NOISE)</span>
            <span>-20dB (TRAINED LIMIT)</span>
            <span>-15dB (STANDARD)</span>
            <span>-10dB (CLEAN SIGNAL)</span>
          </div>

          <p className={`text-[11px] font-sans mt-2 italic transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
            }`}>
            {snrDb <= -21 ? (
              <span className="text-red-400/90 font-semibold flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5" /> Noise exceeds Pair 1 training threshold (-20dB). Classification error spikes.
              </span>
            ) : snrDb <= -15 ? (
              <span className="text-amber-400/90 font-semibold flex items-center gap-1">
                <Shield className="w-3.5 h-3.5" /> High noise degradation. Model boundary begins shifting to clutter classification.
              </span>
            ) : (
              <span className="text-emerald-400/90 font-semibold flex items-center gap-1">
                <CheckCircle className="w-3.5 h-3.5" /> Target features clean. Precision-recall metrics track standard benchmark validation.
              </span>
            )}
          </p>

          <div className="mt-3.5">
            <div className={`text-[9.5px] font-mono uppercase mb-2 tracking-wider transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
              }`}>Tactical Scenario Presets</div>
            <div className="flex flex-wrap gap-1.5">
              {[
                { name: 'Airport', db: -12.0, desc: 'Moderate standard SNR' },
                { name: 'Forest', db: -17.0, desc: 'High forest clutter' },
                { name: 'Border', db: -10.0, desc: 'Clean border radar signal' },
                { name: 'Urban', db: -14.5, desc: 'Reflective urban canopy' },
                { name: 'Night', db: -10.0, desc: 'Stealth night operations' },
                { name: 'Heavy Noise', db: -23.0, desc: 'Extreme noise jamming environment' }
              ].map(preset => (
                <button
                  key={preset.name}
                  onClick={() => setSnrDb(preset.db)}
                  className={`px-2 py-1 font-mono text-[9px] border rounded transition-all duration-300 cursor-pointer ${snrDb === preset.db
                    ? 'bg-emerald-950/20 border-emerald-500/40 text-emerald-400 font-semibold shadow-[0_0_8px_rgba(52,211,153,0.15)]'
                    : (theme === 'dark'
                      ? 'border-zinc-800 text-zinc-400 bg-zinc-900/10 hover:border-zinc-700 hover:text-zinc-300'
                      : 'border-zinc-200 text-zinc-600 bg-zinc-50 hover:bg-zinc-100 hover:text-zinc-900 shadow-sm')
                    }`}
                  title={`${preset.desc} (${preset.db} dB)`}
                >
                  {preset.name.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className={`border-t pt-4 transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900' : 'border-zinc-200'
          }`}>
          <div className="flex justify-between items-center mb-2">
            <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
              }`}>
              <Settings className="w-4 h-4 text-emerald-400" /> Model Engine Configuration
            </span>
            {isModelLoaded && inferenceMode === 'classical' && (
              <button
                onClick={unloadCustomModel}
                className="text-[10px] text-red-400 hover:text-red-300 font-mono underline cursor-pointer"
              >
                UNLOAD MODEL
              </button>
            )}
          </div>

          <div className={`grid grid-cols-3 gap-1 p-1 rounded-xl mb-3 border transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-950/40 border-zinc-900' : 'bg-zinc-100/60 border-zinc-200'
            }`}>
            <button
              onClick={() => { setInferenceMode('classical'); setUseEnsemble(false); }}
              className={`py-1 text-[11px] font-semibold font-mono rounded-lg transition-all cursor-pointer ${inferenceMode === 'classical'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-zinc-500 hover:text-zinc-400 border border-transparent'
                }`}
            >
              Classical ONNX
            </button>
            <button
              onClick={() => { setInferenceMode('quantum'); setUseEnsemble(false); }}
              className={`py-1 text-[11px] font-semibold font-mono rounded-lg transition-all cursor-pointer ${inferenceMode === 'quantum' && !useEnsemble
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-zinc-500 hover:text-zinc-400 border border-transparent'
                }`}
            >
              Pure Quantum
            </button>
            <button
              onClick={() => { setInferenceMode('quantum'); setUseEnsemble(true); }}
              className={`py-1 text-[11px] font-semibold font-mono rounded-lg transition-all cursor-pointer ${inferenceMode === 'quantum' && useEnsemble
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'text-zinc-500 hover:text-zinc-400 border border-transparent'
                }`}
            >
              Stacking Ensemble
            </button>
          </div>

          {inferenceMode === 'classical' ? (
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-4 text-center transition-all ${dragActive
                ? (theme === 'dark' ? 'border-emerald-500 bg-emerald-950/10' : 'border-emerald-500 bg-emerald-50/50')
                : isModelLoaded
                  ? (theme === 'dark' ? 'border-emerald-900/40 bg-emerald-950/5' : 'border-emerald-200 bg-emerald-50/20')
                  : (theme === 'dark' ? 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/10' : 'border-zinc-200 hover:border-zinc-300 bg-zinc-50')
                }`}
            >
              <div className="flex flex-col items-center gap-1">
                <Upload className={`w-6 h-6 ${isModelLoaded ? 'text-emerald-400' : 'text-zinc-500'}`} />
                <div className={`text-xs font-semibold mt-1 font-mono transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                  }`}>
                  {modelSourceName}
                </div>
                <p className={`text-[10px] font-sans max-w-[280px] transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
                  }`}>
                  {isModelLoaded
                    ? "ONNX model loaded successfully into WebAssembly runtime. Predictions run live."
                    : "Drag and drop Pair 1's exported `.onnx` model file to evaluate real-time inference, or use the default physics stub."
                  }
                </p>

                {!isModelLoaded && (
                  <label className={`mt-2 px-3 py-1 border rounded font-mono text-[11px] cursor-pointer transition-all ${theme === 'dark'
                    ? 'bg-zinc-800 hover:bg-zinc-700 border-zinc-700 text-zinc-300'
                    : 'bg-zinc-100 hover:bg-zinc-200 border-zinc-200 text-zinc-700'
                    }`}>
                    SELECT ONNX FILE
                    <input
                      type="file"
                      accept=".onnx"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                )}
              </div>
            </div>
          ) : (
            <div>
              {quantumServerStatus.online ? (
                <div className={`border rounded-xl p-4 transition-all ${theme === 'dark' ? 'border-emerald-900/40 bg-emerald-950/5' : 'border-emerald-200 bg-emerald-50/20'
                  }`}>
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-400 mt-0.5">
                      <Server className="w-4 h-4" />
                    </div>
                    <div className="flex-1 text-left">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-semibold text-emerald-400">QSVC Engine</span>
                        <span className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[9px] font-bold font-mono">
                          <span className="w-1 h-1 rounded-full bg-emerald-400 animate-ping"></span>
                          ONLINE
                        </span>
                      </div>
                      <p className={`text-[10px] mt-1 transition-colors duration-300 leading-snug ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
                        }`}>
                        Connected to local QSVC fidelity quantum kernel simulator backend.
                      </p>

                      {quantumServerStatus.metadata && (
                        <div className="grid grid-cols-2 gap-x-3 gap-y-2 mt-3 pt-3 border-t border-zinc-800/40 font-mono text-[9px]">
                          <div className="flex flex-col">
                            <span className="text-zinc-500 text-[8px] uppercase">Qubits / Map</span>
                            <span className={`font-semibold ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                              {quantumServerStatus.metadata.circuit?.qubits ?? 4} (ZZFeatureMap)
                            </span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-zinc-500 text-[8px] uppercase">Circuit Depth</span>
                            <span className={`font-semibold ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                              {quantumServerStatus.metadata.circuit?.circuit_depth ?? 24}
                            </span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-zinc-500 text-[8px] uppercase">Sim Backend</span>
                            <span className={`font-semibold ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                              Exact Statevector
                            </span>
                          </div>
                          <div className="flex flex-col">
                            <span className="text-zinc-500 text-[8px] uppercase">Training Set</span>
                            <span className={`font-semibold ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                              n={quantumServerStatus.metadata.circuit?.n_train ?? 800}
                            </span>
                          </div>
                        </div>
                      )}

                      <div className="mt-3 pt-3 border-t border-zinc-800/40">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[9px] uppercase font-mono font-bold text-cyan-400 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                            IBM Hardware Viability (ibm_fez)
                          </span>
                          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-bold">
                            r = 0.996 (Mitigated)
                          </span>
                        </div>
                        <div className={`p-2.5 rounded-lg border font-mono text-[9px] space-y-2 ${theme === 'dark' ? 'bg-cyan-950/20 border-cyan-900/40 text-cyan-100' : 'bg-cyan-50 border-cyan-200 text-cyan-900'}`}>
                          <div className="flex items-center justify-between text-[8px] uppercase text-cyan-400/80">
                            <span>Processor: 156-Qubit ibm_fez</span>
                            <span>Job: d9fdbrsjeosc73fjq5g0</span>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-center pt-1">
                            <div className="p-1.5 rounded bg-black/30 border border-cyan-500/20">
                              <div className="text-[8px] text-zinc-400">RAW MAE</div>
                              <div className="font-bold text-amber-400 text-[10px]">0.0256</div>
                              <div className="text-[7px] text-zinc-500">r = 0.9807</div>
                            </div>
                            <div className="p-1.5 rounded bg-black/30 border border-emerald-500/30">
                              <div className="text-[8px] text-emerald-400">MITIGATED MAE</div>
                              <div className="font-bold text-emerald-400 text-[10px]">0.0205</div>
                              <div className="text-[7px] text-emerald-300">r = 0.9963</div>
                            </div>
                          </div>
                          <p className="text-[8.5px] leading-tight text-cyan-200/90 italic pt-1 border-t border-cyan-900/40">
                            💡 XY4 Dynamical Decoupling + TREX Readout Mitigation boosts fidelity agreement to 99.6% on real quantum hardware.
                          </p>
                        </div>
                      </div>

                      <div className="mt-3 pt-3 border-t border-zinc-800/40">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-[9px] uppercase font-mono font-bold text-zinc-500">Inference Engine Mode</span>
                          <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 rounded ${useEnsemble ? 'bg-purple-500/10 text-purple-400 border border-purple-500/30' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                            }`}>
                            {useEnsemble ? 'Stacking Ensemble' : 'Pure QSVC'}
                          </span>
                        </div>
                        <div className={`grid grid-cols-2 gap-1 p-1 rounded-lg border text-[10px] font-mono ${theme === 'dark' ? 'bg-zinc-950/60 border-zinc-800' : 'bg-zinc-100 border-zinc-200'
                          }`}>
                          <button
                            type="button"
                            onClick={() => setUseEnsemble(false)}
                            className={`py-1 text-center rounded font-semibold transition-all cursor-pointer ${!useEnsemble
                              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-sm'
                              : 'text-zinc-500 hover:text-zinc-400 border border-transparent'
                              }`}
                          >
                            Pure QSVC
                          </button>
                          <button
                            type="button"
                            onClick={() => setUseEnsemble(true)}
                            className={`py-1 text-center rounded font-semibold transition-all cursor-pointer ${useEnsemble
                              ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30 shadow-sm'
                              : 'text-zinc-500 hover:text-zinc-400 border border-transparent'
                              }`}
                          >
                            Stacking Ensemble
                          </button>
                        </div>
                        {useEnsemble && (
                          <p className="mt-2 text-[9px] font-mono text-purple-300/80 bg-purple-950/20 border border-purple-900/30 rounded p-1.5 leading-snug">
                            Combines Quantum Kernel (QSVC) + Classical Kernel (RBF SVM) using Logistic Regression meta-model stacking.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

              ) : (
                <div className={`border rounded-xl p-4 text-center transition-all ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/10' : 'border-zinc-200 bg-zinc-50'
                  }`}>
                  <div className="flex flex-col items-center gap-2">
                    <WifiOff className="w-8 h-8 text-amber-500 animate-pulse" />
                    <div className={`text-xs font-semibold font-mono ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'}`}>
                      Quantum API Server Offline
                    </div>
                    <p className={`text-[10px] font-sans max-w-[280px] leading-relaxed transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-600'
                      }`}>
                      To use real quantum kernel predictions, start the API server on port 8081:
                    </p>
                    <div className="w-full bg-black/40 border border-zinc-900 p-2 rounded-lg font-mono text-[9px] text-zinc-400 select-all cursor-pointer text-left overflow-x-auto whitespace-nowrap">
                      .venv\Scripts\uvicorn api.quantum_server:app --port 8081
                    </div>

                    <button
                      onClick={checkQuantumServer}
                      disabled={quantumServerStatus.checking}
                      className={`mt-2 px-3 py-1.5 border rounded-lg font-mono text-[10px] font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${theme === 'dark'
                        ? 'bg-zinc-800 hover:bg-zinc-700 border-zinc-700 text-zinc-300 disabled:opacity-50'
                        : 'bg-zinc-100 hover:bg-zinc-200 border-zinc-200 text-zinc-700 disabled:opacity-50'
                        }`}
                    >
                      <RefreshCw className={`w-3 h-3 ${quantumServerStatus.checking ? 'animate-spin' : ''}`} />
                      {quantumServerStatus.checking ? 'CHECKING...' : 'RECHECK CONNECTION'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {!hideHardware && (
            <div className={`p-4 rounded-xl border mt-4 transition-colors duration-300 ${theme === 'dark' ? 'bg-zinc-900/40 border-zinc-800' : 'bg-white border-zinc-200 shadow-sm'
              }`}>
              <div className="flex items-center justify-between mb-3">
                <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 transition-colors duration-300 ${theme === 'dark' ? 'text-zinc-300' : 'text-zinc-700'
                  }`}>
                  <Usb className="w-4 h-4 text-cyan-400" /> Serial Hardware (COM Port)
                </span>
                <span className={`flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-bold font-mono border ${serialConnected
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : 'bg-zinc-800 text-zinc-400 border-zinc-700'
                  }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${serialConnected ? 'bg-emerald-400 animate-ping' : 'bg-zinc-500'}`}></span>
                  {serialConnected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
              </div>

              <p className={`text-[10px] font-sans leading-relaxed mb-3 ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-600'
                }`}>
                Connect physical radar sweep sensor via Web Serial API. Reads live hardware telemetry streams (<code className="text-cyan-400 font-mono">servo: 54</code>, <code className="text-emerald-400 font-mono">Distance: 0.93</code>).
              </p>

              {serialError && (
                <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-[10px] text-red-400 font-mono flex items-start gap-1.5 leading-relaxed">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <span>{serialError}</span>
                </div>
              )}

              <div className="flex items-center gap-2 mb-3">
                <select
                  value={baudRate}
                  onChange={(e) => setBaudRate(parseInt(e.target.value, 10))}
                  disabled={serialConnected}
                  className={`px-2 py-1.5 border rounded text-[10px] font-mono outline-none cursor-pointer ${theme === 'dark' ? 'bg-zinc-950 border-zinc-800 text-zinc-300' : 'bg-zinc-100 border-zinc-300 text-zinc-800'
                    }`}
                >
                  <option value={4800}>4800 Baud</option>
                  <option value={9600}>9600 Baud</option>
                  <option value={19200}>19200 Baud</option>
                  <option value={38400}>38400 Baud</option>
                  <option value={57600}>57600 Baud</option>
                  <option value={115200}>115200 Baud</option>
                  <option value={230400}>230400 Baud</option>
                </select>

                {!serialConnected ? (
                  <button
                    onClick={connectSerialPort}
                    className="flex-1 py-1.5 px-3 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 rounded text-xs font-mono font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer shadow-sm"
                  >
                    <Cable className="w-3.5 h-3.5" /> CONNECT COM PORT
                  </button>
                ) : (
                  <button
                    onClick={disconnectSerialPort}
                    className="flex-1 py-1.5 px-3 bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/40 rounded text-xs font-mono font-semibold flex items-center justify-center gap-1.5 transition-all cursor-pointer shadow-sm"
                  >
                    DISCONNECT
                  </button>
                )}
              </div>

              <div className="grid grid-cols-2 gap-2 mb-3 font-mono text-[10px]">
                <div className={`p-2 rounded border ${theme === 'dark' ? 'bg-zinc-950/60 border-zinc-800/80' : 'bg-zinc-50 border-zinc-200'
                  }`}>
                  <div className="text-[8px] uppercase text-zinc-500 flex items-center gap-1">
                    <Compass className="w-2.5 h-2.5 text-cyan-400" /> Servo Angle
                  </div>
                  <div className="text-xs font-bold text-cyan-400 mt-0.5">
                    {lastServoAngle !== null ? `${lastServoAngle}°` : '--°'}
                  </div>
                </div>
                <div className={`p-2 rounded border ${theme === 'dark' ? 'bg-zinc-950/60 border-zinc-800/80' : 'bg-zinc-50 border-zinc-200'
                  }`}>
                  <div className="text-[8px] uppercase text-zinc-500 flex items-center gap-1">
                    <Gauge className="w-2.5 h-2.5 text-emerald-400" /> Measured Distance
                  </div>
                  <div className="text-xs font-bold text-emerald-400 mt-0.5">
                    {lastDistance !== null ? `${lastDistance} m` : '-- m'}
                  </div>
                </div>
              </div>

              <div className={`rounded border p-2 mb-3 font-mono text-[9px] ${theme === 'dark' ? 'bg-black/60 border-zinc-900 text-zinc-400' : 'bg-zinc-900 text-zinc-300'
                }`}>
                <div className="flex items-center justify-between pb-1 mb-1 border-b border-zinc-800/60 text-[8px] text-zinc-500">
                  <span className="flex items-center gap-1"><Terminal className="w-2.5 h-2.5" /> RAW SERIAL DATA STREAM</span>
                  <span>{serialLogs.length} LINES</span>
                </div>
                <div className="h-16 overflow-y-auto space-y-0.5 scrollbar-thin">
                  {serialLogs.length > 0 ? (
                    serialLogs.map((log, idx) => (
                      <div key={idx} className="truncate">
                        <span className="text-zinc-600 select-none">&gt; </span>
                        <span className={log.includes('servo') ? 'text-cyan-300 font-semibold' : log.includes('Distance') ? 'text-emerald-300 font-semibold' : 'text-zinc-300'}>
                          {log}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="text-zinc-600 italic py-2 text-center">Waiting for incoming serial telemetry...</div>
                  )}
                </div>
              </div>

              <div className="pt-2 border-t border-zinc-800/40">
                <div className="text-[8px] uppercase font-mono font-bold text-zinc-500 mb-1">
                  Simulate Serial Stream (Hardware Input Test)
                </div>
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    placeholder="e.g. servo: 54  or  Distance: 0.93"
                    value={manualSerialInput}
                    onChange={(e) => setManualSerialInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && manualSerialInput.trim()) {
                        processSerialLine(manualSerialInput);
                        setManualSerialInput('');
                      }
                    }}
                    className={`flex-1 px-2 py-1 border rounded text-[10px] font-mono outline-none ${theme === 'dark' ? 'bg-zinc-950 border-zinc-800 text-zinc-300 focus:border-cyan-500/50' : 'bg-zinc-100 border-zinc-300 text-zinc-800 focus:border-cyan-500'
                      }`}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (manualSerialInput.trim()) {
                        processSerialLine(manualSerialInput);
                        setManualSerialInput('');
                      }
                    }}
                    className="px-2 py-1 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/30 rounded text-[10px] font-mono font-semibold transition-all cursor-pointer"
                  >
                    SEND
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}



const AIRSPACE_SCALE = 1 / 125;

function classColor(target: Target) {
  const detected = target.lastPrediction?.predictedClass ?? target.type;
  return detected === 'drone' ? '#fb4d5d'
    : detected === 'aircraft' ? '#54a5ff'
      : detected === 'bird' ? '#f7b84b'
        : detected === 'anomaly' ? '#b58bff'
          : '#a1a1aa';
}

function RadarScene3D({
  targets,
  sweepAngleRef,
  selectedId,
  showOverlay,
  paused,
  onSelect,
  theme,
  hardwareMode = false,
}: {
  targets: Target[];
  sweepAngleRef: React.MutableRefObject<number>;
  selectedId: number | null;
  showOverlay: boolean;
  paused: boolean;
  onSelect: (target: Target | null) => void;
  theme: 'dark' | 'light';
  hardwareMode?: boolean;
}) {
  const isDark = theme === 'dark';
  const bgColor = isDark ? '#040b0d' : '#f4f5f7';

  return (
    <Canvas
      dpr={[1, 1.5]}
      shadows={false}
      frameloop={paused ? 'demand' : 'always'}
      camera={{ position: [35, 28, 50], fov: 50, near: 0.1, far: 250 }}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance', stencil: false, preserveDrawingBuffer: false }}
      onPointerMissed={() => onSelect(null)}
      className="!absolute inset-0"
    >
      <color attach="background" args={[bgColor]} />
      <fog attach="fog" args={[bgColor, 50, 125]} />
      <ambientLight intensity={isDark ? 0.5 : 0.85} />
      <directionalLight position={[25, 45, 10]} intensity={isDark ? 1.7 : 1.3} color={isDark ? '#c8fff0' : '#ffffff'} />
      <pointLight position={[0, 12, 0]} intensity={isDark ? 18 : 8} distance={36} color={isDark ? '#26dca2' : '#10b981'} />

      <AirspaceFloor theme={theme} hardwareMode={hardwareMode} />
      <RadarTower theme={theme} />
      <RadarSweep sweepAngleRef={sweepAngleRef} theme={theme} />
      {targets.map((target) => (
        <AirTarget
          key={target.id}
          target={target}
          selected={target.id === selectedId}
          showOverlay={showOverlay}
          onSelect={onSelect}
        />
      ))}
      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={28}
        maxDistance={105}
        minPolarAngle={0.3}
        maxPolarAngle={1.42}
        target={[0, 7, 0]}
        enableDamping
        dampingFactor={0.08}
      />
    </Canvas>
  );
}

function AirspaceFloor({ theme, hardwareMode = false }: { theme: 'dark' | 'light'; hardwareMode?: boolean }) {
  const isDark = theme === 'dark';

  
  const floorGeo = useMemo(() => {
    if (hardwareMode) {
      
      return new THREE.CircleGeometry(40, 96, 0, Math.PI);
    }
    return new THREE.CircleGeometry(40, 96);
  }, [hardwareMode]);

  
  const makeRing = (radius: number) => {
    if (hardwareMode) {
      
      const segments = 64;
      const points: THREE.Vector3[] = [];
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI;
        points.push(new THREE.Vector3(Math.sin(theta) * radius, 0.025, Math.cos(theta) * radius));
      }
      const curve = new THREE.CatmullRomCurve3(points, false);
      return new THREE.TubeGeometry(curve, segments, 0.035, 4, false);
    }
    return new THREE.RingGeometry(radius - 0.035, radius + 0.035, 128);
  };

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow geometry={floorGeo}>
        <meshStandardMaterial color={isDark ? '#071514' : '#e2e8f0'} roughness={0.94} metalness={0.2} />
      </mesh>
      {!hardwareMode && (
        <gridHelper args={[80, 16, isDark ? '#176a59' : '#10b981', isDark ? '#0a2e28' : '#cbd5e1']} position={[0, 0.015, 0]}>
          <lineBasicMaterial transparent opacity={0.32} />
        </gridHelper>
      )}
      {[8, 16, 24, 32, 40].map((radius) =>
        hardwareMode ? (
          <mesh key={radius} geometry={makeRing(radius)}>
            <meshBasicMaterial color={isDark ? '#1bc998' : '#10b981'} transparent opacity={isDark ? 0.28 : 0.4} />
          </mesh>
        ) : (
          <mesh key={radius} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.025, 0]}>
            <ringGeometry args={[radius - 0.035, radius + 0.035, 128]} />
            <meshBasicMaterial color={isDark ? '#1bc998' : '#10b981'} transparent opacity={isDark ? 0.28 : 0.4} side={THREE.DoubleSide} />
          </mesh>
        )
      )}
      {(hardwareMode ? [0, 15, 30, 45, 60, 75, 90] : [0, 45, 90, 135]).map((angle) => (
        <mesh key={angle} rotation={[0, angle * Math.PI / 180, 0]} position={[0, 0.03, 0]}>
          <boxGeometry args={[0.045, 0.02, hardwareMode ? 40 : 80]} />
          <meshBasicMaterial color={isDark ? '#1bc998' : '#10b981'} transparent opacity={isDark ? 0.2 : 0.35} />
        </mesh>
      ))}
      {[8, 16, 24, 32, 40].map((radius) => (
        <Html key={`label-${radius}`} position={[0, 0.04, radius]} center>
          <span style={{ color: isDark ? '#1bc998' : '#10b981', fontSize: 7, fontFamily: 'monospace', opacity: 0.5 }}>
            {radius * 125}m
          </span>
        </Html>
      ))}
      {hardwareMode && (
        <mesh position={[0, 0.04, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <boxGeometry args={[80, 0.07, 0.02]} />
          <meshBasicMaterial color={isDark ? '#1bc998' : '#10b981'} transparent opacity={0.3} />
        </mesh>
      )}
    </group>
  );
}

function RadarTower({ theme }: { theme: 'dark' | 'light' }) {
  const dishRef = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (dishRef.current) dishRef.current.rotation.y = state.clock.elapsedTime * 1.1;
  });
  const isDark = theme === 'dark';
  return (
    <group position={[0, 0, 0]}>
      <mesh position={[0, 1.2, 0]}>
        <cylinderGeometry args={[0.34, 0.52, 2.4, 10]} />
        <meshStandardMaterial color={isDark ? "#42625d" : "#64748b"} metalness={0.8} roughness={0.28} />
      </mesh>
      <group ref={dishRef} position={[0, 2.45, 0]}>
        <mesh rotation={[0, 0, Math.PI / 2]}>
          <sphereGeometry args={[0.6, 18, 10, 0, Math.PI]} />
          <meshStandardMaterial color={isDark ? "#b4ded5" : "#cbd5e1"} metalness={0.9} roughness={0.18} emissive={isDark ? "#0f5d4d" : "#094f3e"} emissiveIntensity={isDark ? 0.7 : 0.4} />
        </mesh>
        <mesh position={[0, 0, 0.34]}>
          <sphereGeometry args={[0.12, 12, 12]} />
          <meshBasicMaterial color={isDark ? "#6fffd3" : "#10b981"} />
        </mesh>
      </group>
      <pointLight position={[0, 2.5, 0]} intensity={isDark ? 4 : 2} distance={10} color={isDark ? "#32e2aa" : "#10b981"} />
    </group>
  );
}

function RadarSweep({ sweepAngleRef, theme }: { sweepAngleRef: React.MutableRefObject<number>, theme: 'dark' | 'light' }) {
  const sweepRef = useRef<THREE.Group>(null);
  const beamMatRef = useRef<THREE.MeshBasicMaterial>(null);
  const isDark = theme === 'dark';
  const geometry = useMemo(() => {
    const segments = 32;
    const range = 40;
    const arc = Math.PI / 8;
    const positions = [0, 0.07, 0];
    const indices: number[] = [];
    for (let i = 0; i <= segments; i += 1) {
      const theta = -arc + (arc * i) / segments;
      positions.push(Math.sin(theta) * range, 0.07, Math.cos(theta) * range);
    }
    for (let i = 0; i < segments; i += 1) indices.push(0, i + 1, i + 2);
    const result = new THREE.BufferGeometry();
    result.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    result.setIndex(indices);
    return result;
  }, []);

  useEffect(() => () => geometry.dispose(), [geometry]);
  useFrame((state) => {
    if (sweepRef.current) sweepRef.current.rotation.y = sweepAngleRef.current * Math.PI / 180;
    if (beamMatRef.current) {
      const pulse = 0.12 + Math.sin(state.clock.elapsedTime * 6) * 0.04;
      beamMatRef.current.opacity = isDark ? pulse : pulse + 0.07;
    }
  });

  return (
    <group ref={sweepRef}>
      <mesh geometry={geometry}>
        <meshBasicMaterial ref={beamMatRef} color={isDark ? "#35f5b5" : "#10b981"} transparent blending={THREE.AdditiveBlending} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh geometry={geometry} scale={[1.15, 1, 1.15]}>
        <meshBasicMaterial color={isDark ? "#35f5b5" : "#10b981"} transparent opacity={0.04} blending={THREE.AdditiveBlending} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0.085, 20]}>
        <boxGeometry args={[0.08, 0.02, 40]} />
        <meshBasicMaterial color={isDark ? "#8dffe0" : "#6ee7b7"} transparent opacity={0.8} />
      </mesh>
    </group>
  );
}

function AirTarget({
  target,
  selected,
  showOverlay,
  onSelect,
}: {
  target: Target;
  selected: boolean;
  showOverlay: boolean;
  onSelect: (target: Target) => void;
}) {
  const anchorRef = useRef<THREE.Group>(null);
  const modelRef = useRef<THREE.Group>(null);
  const signalRef = useRef<THREE.MeshBasicMaterial>(null);

  useFrame((state) => {
    const anchor = anchorRef.current;
    if (!anchor) return;
    const altitude = Math.max(0.25, target.altitude * AIRSPACE_SCALE);
    anchor.position.set(target.x * AIRSPACE_SCALE, 0, target.y * AIRSPACE_SCALE);
    if (modelRef.current) {
      modelRef.current.position.y = altitude + (target.type === 'bird' ? Math.sin(state.clock.elapsedTime * 7 + target.id) * 0.1 : 0);
      modelRef.current.rotation.y = Math.atan2(target.vx, target.vy);
    }
    if (signalRef.current) signalRef.current.color.set(classColor(target));
  });

  const altitude = Math.max(0.25, target.altitude * AIRSPACE_SCALE);
  const color = classColor(target);
  const targetLabel = `${target.subtype} · ${Math.round(target.altitude)}m`;

  return (
    <group ref={anchorRef}>
      <TargetTrail target={target} />
      <AltitudeLine target={target} />
      <mesh position={[0, altitude / 2, 0]} scale={[1, altitude, 1]}>
        <cylinderGeometry args={[0.018, 0.018, 1, 6]} />
        <meshBasicMaterial color="#56e6be" transparent opacity={0.3} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.055, 0]}>
        <ringGeometry args={[0.18, 0.28, 24]} />
        <meshBasicMaterial ref={signalRef} color={color} transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
      {selected && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
          <ringGeometry args={[0.42, 0.5, 32]} />
          <meshBasicMaterial color="#6fffd3" transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      )}
      <group
        ref={modelRef}
        onPointerDown={(event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation();
          onSelect(target);
        }}
      >
        <TargetModel target={target} color={color} />
        {selected && showOverlay && (
          <Html center distanceFactor={11} style={{ pointerEvents: 'none' }}>
            <div className="min-w-max rounded border border-emerald-400/50 bg-zinc-950/90 px-2 py-1 font-mono text-[10px] leading-4 text-emerald-100 shadow-[0_0_18px_rgba(16,185,129,0.25)]">
              <div className="text-emerald-300">TRK-{target.id} · {targetLabel}</div>
              <div className="text-zinc-400">{target.lastPrediction ? `PRED ${target.lastPrediction.predictedClass.toUpperCase()}` : 'AWAITING SWEEP'}</div>
            </div>
          </Html>
        )}
      </group>
    </group>
  );
}

function TargetTrail({ target }: { target: Target }) {
  const lineRef = useRef<any>(null);
  const MAX_POINTS = 30;
  const initPos = useMemo(() => new Float32Array(MAX_POINTS * 3), []);

  useFrame(() => {
    const line = lineRef.current;
    if (!line) return;
    const history = target.history;
    const geo = line.geometry;
    const pos = geo.attributes.position.array as Float32Array;
    const count = Math.min(history.length, MAX_POINTS);
    const offset = history.length - count;
    for (let i = 0; i < count; i++) {
      pos[i * 3] = history[offset + i].x * AIRSPACE_SCALE;
      pos[i * 3 + 1] = 0.03;
      pos[i * 3 + 2] = history[offset + i].y * AIRSPACE_SCALE;
    }
    geo.attributes.position.needsUpdate = true;
    geo.setDrawRange(0, count);
  });

  return (
    <line ref={lineRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[initPos, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#56e6be" transparent opacity={0.1} />
    </line>
  );
}

function AltitudeLine({ target }: { target: Target }) {
  const lineRef = useRef<any>(null);
  const initPos = useMemo(() => new Float32Array([0, 0, 0, 0, 1, 0]), []);

  useFrame(() => {
    const line = lineRef.current;
    if (!line) return;
    const alt = Math.max(0.25, target.altitude * AIRSPACE_SCALE);
    const pos = line.geometry.attributes.position.array as Float32Array;
    pos[0] = 0; pos[1] = 0; pos[2] = 0;
    pos[3] = 0; pos[4] = alt; pos[5] = 0;
    line.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <line ref={lineRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[initPos, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#56e6be" transparent opacity={0.07} />
    </line>
  );
}

function TargetModel({ target, color }: { target: Target; color: string }) {
  const material = <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.28} metalness={0.55} roughness={0.3} />;
  if (target.subtype === 'Weather Balloon') {
    return <group scale={0.75}>
      <mesh>{<sphereGeometry args={[0.58, 16, 16]} />}{material}</mesh>
      <mesh position={[0, -0.65, 0]}>{<cylinderGeometry args={[0.012, 0.012, 0.75, 6]} />}{<meshBasicMaterial color="#c7d8d4" />}</mesh>
      <mesh position={[0, -1.05, 0]}>{<boxGeometry args={[0.28, 0.14, 0.2]} />}{<meshStandardMaterial color="#5b7370" />}</mesh>
    </group>;
  }
  if (target.type === 'bird') {
    return <group scale={target.subtype === 'Flock' ? 0.7 : 0.85}>
      <mesh rotation={[0, 0, Math.PI / 2]}>{<sphereGeometry args={[0.22, 12, 10]} />}{material}</mesh>
      <mesh position={[-0.32, 0, 0]} rotation={[0.15, 0, -0.28]}>{<coneGeometry args={[0.36, 0.95, 3]} />}{material}</mesh>
      <mesh position={[0.32, 0, 0]} rotation={[0.15, 0, 0.28]}>{<coneGeometry args={[0.36, 0.95, 3]} />}{material}</mesh>
      <mesh position={[0, 0, -0.35]} rotation={[Math.PI / 2, 0, 0]}>{<coneGeometry args={[0.12, 0.42, 3]} />}{material}</mesh>
    </group>;
  }
  if (target.type === 'anomaly') {
    const stealthGeo = useMemo(() => new THREE.OctahedronGeometry(0.55, 1), []);
    return <group scale={0.7} rotation={[0.35, 0.2, 0.1]}>
      <mesh geometry={stealthGeo}>{material}</mesh>
      <lineSegments>
        <edgesGeometry args={[stealthGeo]} />
        <lineBasicMaterial color="#e6d9ff" transparent opacity={0.55} />
      </lineSegments>
    </group>;
  }
  if (target.type === 'drone' && target.subtype !== 'Fixed-Wing UAV') {
    const compact = target.subtype === 'Nano Drone';
    return <group scale={compact ? 0.55 : 0.9}>
      <mesh>{<boxGeometry args={[0.46, 0.16, 0.62]} />}{material}</mesh>
      <mesh rotation={[0, Math.PI / 4, 0]}>{<boxGeometry args={[1.55, 0.055, 0.09]} />}{material}</mesh>
      <mesh rotation={[0, -Math.PI / 4, 0]}>{<boxGeometry args={[1.55, 0.055, 0.09]} />}{material}</mesh>
      {[[0.55, 0.55], [-0.55, 0.55], [0.55, -0.55], [-0.55, -0.55]].map(([x, z], index) => (
        <mesh key={index} position={[x, 0.06, z]} rotation={[Math.PI / 2, 0, 0]}>{<cylinderGeometry args={[0.2, 0.2, 0.035, 16]} />}{<meshBasicMaterial color="#d1fff1" transparent opacity={0.78} />}</mesh>
      ))}
      <mesh position={[0, -0.15, 0]}>{<sphereGeometry args={[0.14, 12, 8]} />}{<meshStandardMaterial color="#172329" metalness={0.8} />}</mesh>
    </group>;
  }
  
  const fighter = target.subtype === 'Fighter Jet';
  return <group scale={fighter ? 1.15 : target.subtype === 'Fixed-Wing UAV' ? 0.72 : 1}>
    <mesh rotation={[Math.PI / 2, 0, 0]}>{<cylinderGeometry args={[0.16, 0.22, 1.8, 12]} />}{material}</mesh>
    <mesh position={[0, 0, -0.95]} rotation={[Math.PI / 2, 0, 0]}>{<coneGeometry args={[0.22, 0.7, 12]} />}{material}</mesh>
    <mesh position={[0, 0, 0.05]}>{<boxGeometry args={[2.3, 0.055, 0.35]} />}{material}</mesh>
    <mesh position={[0, 0.18, 0.65]}>{<boxGeometry args={[0.72, 0.28, 0.08]} />}{material}</mesh>
    {target.subtype === 'Helicopter' && <>
      <mesh position={[0, 0.43, 0]}>{<boxGeometry args={[2.4, 0.025, 0.08]} />}{<meshBasicMaterial color="#d7fff3" />}</mesh>
      <mesh position={[0, 0.28, 0]}>{<cylinderGeometry args={[0.06, 0.06, 0.42, 8]} />}{material}</mesh>
    </>}
  </group>;
}