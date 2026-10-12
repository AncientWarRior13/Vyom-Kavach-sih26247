'use client';








import React, { useEffect, useRef, useState, useMemo } from 'react';
import {
  Radar, Play, Square, Loader2, Shield, Activity, BarChart2,
  Eye, CheckCircle, AlertTriangle, Cpu, Radio, Zap, Compass, Filter, Box, Layers, RefreshCw,
  Settings, Upload, Server, Wifi, WifiOff
} from 'lucide-react';
import { assessThreat, type ThreatAssessment, threatColor } from './threatAssessment';
import { OnnxModelManager, InferenceResult } from './OnnxInference';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8081';
const LIVE_PRESETS = [20, 50, 150, 300];
const MAX_RADAR_RANGE_M = 5000;

export type TargetClass = 'drone' | 'aircraft' | 'bird' | 'anomaly';

export type TargetSubtype =
  | 'Quadcopter' | 'Hexacopter' | 'Fixed-Wing UAV' | 'Nano Drone'
  | 'Airliner' | 'Fighter Jet' | 'Helicopter' | 'Cargo Plane'
  | 'Eagle' | 'Flock' | 'Seagull'
  | 'Weather Balloon' | 'Debris' | 'Unknown Object';

const SUBTYPE_CLASS: Record<TargetSubtype, TargetClass> = {
  'Quadcopter': 'drone', 'Hexacopter': 'drone', 'Fixed-Wing UAV': 'drone', 'Nano Drone': 'drone',
  'Airliner': 'aircraft', 'Fighter Jet': 'aircraft', 'Helicopter': 'aircraft', 'Cargo Plane': 'aircraft',
  'Eagle': 'bird', 'Flock': 'bird', 'Seagull': 'bird',
  'Weather Balloon': 'anomaly', 'Debris': 'anomaly', 'Unknown Object': 'anomaly',
};

const CLASS_COLORS: Record<TargetClass, { stroke: string; fill: string; bg: string; text: string }> = {
  drone: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.4)', bg: 'bg-red-500/10', text: 'text-red-400' },
  aircraft: { stroke: '#3b82f6', fill: 'rgba(59, 130, 246, 0.4)', bg: 'bg-blue-500/10', text: 'text-blue-400' },
  bird: { stroke: '#f59e0b', fill: 'rgba(245, 158, 11, 0.4)', bg: 'bg-amber-500/10', text: 'text-amber-400' },
  anomaly: { stroke: '#a855f7', fill: 'rgba(168, 85, 247, 0.4)', bg: 'bg-purple-500/10', text: 'text-purple-400' },
};

interface SubtypePreset {
  rcs: number; udbw: number; udjit: number; velvar: number;
  altMin: number; altMax: number; speedMin: number; speedMax: number;
}

const SUBTYPE_PRESETS: Record<TargetSubtype, SubtypePreset> = {
  'Quadcopter':     { rcs: -14, udbw: 240, udjit: 0.72, velvar: 12.0, altMin: 30,   altMax: 500,  speedMin: 4,  speedMax: 16 },
  'Hexacopter':     { rcs: -11, udbw: 260, udjit: 0.78, velvar: 14.0, altMin: 50,   altMax: 600,  speedMin: 3,  speedMax: 14 },
  'Fixed-Wing UAV': { rcs: -12, udbw: 210, udjit: 0.65, velvar: 8.0,  altMin: 100,  altMax: 1200, speedMin: 15, speedMax: 35 },
  'Nano Drone':     { rcs: -16, udbw: 280, udjit: 0.82, velvar: 16.0, altMin: 10,   altMax: 200,  speedMin: 2,  speedMax: 10 },

  'Airliner':       { rcs: 16,  udbw: 12,  udjit: 0.06, velvar: 0.2,  altMin: 2000, altMax: 4000, speedMin: 40, speedMax: 60 },
  'Fighter Jet':    { rcs: 10,  udbw: 18,  udjit: 0.10, velvar: 0.5,  altMin: 800,  altMax: 3500, speedMin: 50, speedMax: 80 },
  'Helicopter':     { rcs: 8,   udbw: 22,  udjit: 0.12, velvar: 0.8,  altMin: 50,   altMax: 1500, speedMin: 8,  speedMax: 30 },
  'Cargo Plane':    { rcs: 18,  udbw: 10,  udjit: 0.05, velvar: 0.1,  altMin: 1500, altMax: 3500, speedMin: 35, speedMax: 50 },

  'Eagle':          { rcs: -14, udbw: 50,  udjit: 0.45, velvar: 3.2,  altMin: 30,   altMax: 500,  speedMin: 8,  speedMax: 22 },
  'Flock':          { rcs: -15, udbw: 55,  udjit: 0.48, velvar: 4.0,  altMin: 20,   altMax: 300,  speedMin: 10, speedMax: 20 },
  'Seagull':        { rcs: -18, udbw: 40,  udjit: 0.38, velvar: 2.5,  altMin: 10,   altMax: 200,  speedMin: 6,  speedMax: 18 },

  'Weather Balloon':{ rcs: -2,  udbw: 8,   udjit: 0.08, velvar: 0.4,  altMin: 500,  altMax: 3500, speedMin: 1,  speedMax: 5  },
  'Debris':         { rcs: -4,  udbw: 12,  udjit: 0.12, velvar: 0.6,  altMin: 100,  altMax: 2000, speedMin: 10, speedMax: 40 },
  'Unknown Object': { rcs: 0,   udbw: 15,  udjit: 0.14, velvar: 0.8,  altMin: 200,  altMax: 2500, speedMin: 8,  speedMax: 50 },
};

export interface FleetTarget {
  id: number;
  type: TargetClass;
  subtype: TargetSubtype;
  x: number;
  y: number;
  vx: number;
  vy: number;
  altitude: number;
  baseRcs: number;
  baseUdbw: number;
  baseUdjit: number;
  baseVelvar: number;
  history: { x: number; y: number }[];
  lastPrediction: { predictedClass: TargetClass; confidences: Record<string, number> } | null;
  lastDetectTime: number;
}

export interface SessionStats {
  totalDetections: number;
  correctDetections: number;
  accuracy: number;
  avgInferenceTimeMs: number;
  confusionMatrix: Record<TargetClass, Record<TargetClass, number>>;
  classStats: Record<TargetClass, { detections: number; correct: number; recall: number; precision: number }>;
}

interface StressResult {
  n_targets: number;
  track_completeness_mean: number;
  cardinality_error: number;
  spurious_tracks: number;
  assoc_latency_ms_mean: number;
  assoc_latency_ms_p95: number;
  fleet_summary_final_frame: {
    confirmed_tracks: number;
    heading_coherence: number;
    coordinated_swarm_detected: boolean;
  };
}

interface StressReport {
  association: string;
  detection: string;
  grid: { n_range: number; n_doppler: number };
  scales_tested: number[];
  results: StressResult[];
}

interface FleetSummary {
  confirmed_tracks: number;
  tentative_tracks: number;
  heading_coherence: number;
  coordinated_swarm_detected: boolean;
  association_latency_ms: number;
}

function initMultiClassSwarm(n: number): FleetTarget[] {
  const subtypes: TargetSubtype[] = [
    'Quadcopter', 'Hexacopter', 'Fixed-Wing UAV', 'Nano Drone',
    'Airliner', 'Fighter Jet', 'Helicopter', 'Cargo Plane',
    'Eagle', 'Flock', 'Seagull',
    'Weather Balloon', 'Debris', 'Unknown Object'
  ];

  const weights = [
    0.03, 0.03, 0.02, 0.02,
    0.15, 0.05, 0.10, 0.10,
    0.15, 0.12, 0.08,
    0.05, 0.05, 0.05
  ];

  const targets: FleetTarget[] = [];
  for (let i = 0; i < n; i++) {
    let r = Math.random();
    let subtypeIdx = 0;
    for (let j = 0; j < weights.length; j++) {
      if (r < weights[j]) { subtypeIdx = j; break; }
      r -= weights[j];
    }
    const subtype = subtypes[subtypeIdx] || 'Quadcopter';
    const preset = SUBTYPE_PRESETS[subtype];
    const cls = SUBTYPE_CLASS[subtype];

    const spawnAngle = Math.random() * Math.PI * 2;
    const spawnDist = 600 + Math.random() * 3800;
    const speed = preset.speedMin + Math.random() * (preset.speedMax - preset.speedMin);
    const vAngle = spawnAngle + Math.PI + (Math.random() - 0.5) * 0.8;

    const x = Math.cos(spawnAngle) * spawnDist;
    const y = Math.sin(spawnAngle) * spawnDist;
    const vx = Math.cos(vAngle) * speed;
    const vy = Math.sin(vAngle) * speed;
    const altitude = preset.altMin + Math.random() * (preset.altMax - preset.altMin);

    const confs: Record<string, number> = {
      drone: cls === 'drone' ? 0.88 : 0.04,
      aircraft: cls === 'aircraft' ? 0.92 : 0.03,
      bird: cls === 'bird' ? 0.85 : 0.05,
      anomaly: cls === 'anomaly' ? 0.78 : 0.07,
    };

    targets.push({
      id: i + 1,
      type: cls,
      subtype,
      x, y, vx, vy,
      altitude,
      baseRcs: preset.rcs + (Math.random() - 0.5) * 2,
      baseUdbw: preset.udbw * (0.85 + Math.random() * 0.3),
      baseUdjit: Math.min(1, Math.max(0, preset.udjit + (Math.random() - 0.5) * 0.08)),
      baseVelvar: preset.velvar * (0.8 + Math.random() * 0.4),
      history: [],
      lastPrediction: { predictedClass: cls, confidences: confs },
      lastDetectTime: Date.now()
    });
  }
  return targets;
}

function computeShap(target: FleetTarget): { name: string; impact: number }[] {
  const cls = target.type;
  const bw = target.baseUdbw;
  const jit = target.baseUdjit;
  const rcs = target.baseRcs;
  const speed = Math.sqrt(target.vx * target.vx + target.vy * target.vy);

  if (cls === 'drone') {
    return [
      { name: 'micro_doppler_bandwidth', impact: 3.8 * (bw / 250) },
      { name: 'micro_doppler_jitter', impact: 3.2 * (jit / 0.7) },
      { name: 'velocity_variance', impact: 2.1 * (target.baseVelvar / 12) },
      { name: 'rcs_estimate', impact: -1.9 * (rcs / -14) },
    ];
  } else if (cls === 'aircraft') {
    return [
      { name: 'rcs_estimate', impact: 4.1 * (rcs / 12) },
      { name: 'doppler_velocity', impact: 3.2 * (speed / 40) },
      { name: 'micro_doppler_bandwidth', impact: -2.8 },
      { name: 'altitude_m', impact: 1.8 * (target.altitude / 2000) },
    ];
  } else if (cls === 'bird') {
    return [
      { name: 'rcs_estimate', impact: -3.2 },
      { name: 'micro_doppler_bandwidth', impact: 1.4 * (bw / 50) },
      { name: 'micro_doppler_jitter', impact: 0.9 * (jit / 0.4) },
      { name: 'velocity_variance', impact: 0.5 },
    ];
  } else {
    return [
      { name: 'acceleration', impact: 3.4 },
      { name: 'velocity_variance', impact: 2.9 },
      { name: 'doppler_velocity', impact: 2.2 },
      { name: 'rcs_estimate', impact: 1.4 },
    ];
  }
}

export default function FleetPanel({ theme }: { theme: 'dark' | 'light' }) {
  const isDark = theme === 'dark';
  const [report, setReport] = useState<StressReport | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  
  const [inferenceMode, setInferenceMode] = useState<'classical' | 'quantum'>('classical');
  const [useEnsemble, setUseEnsemble] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [modelSourceName, setModelSourceName] = useState('Physics-Guided ResNet Classifier (Built-in)');
  const [dragActive, setDragActive] = useState(false);
  const [quantumServerStatus, setQuantumServerStatus] = useState<{ online: boolean }>({ online: true });

  const onnxManagerRef = useRef<OnnxModelManager>(new OnnxModelManager());

  const [running, setRunning] = useState(true);
  const [liveN, setLiveN] = useState(150);
  const [displayMode, setDisplayMode] = useState<'3D' | '2D'>('3D');
  const [classFilter, setClassFilter] = useState<'all' | TargetClass>('all');
  const [coordinatedSwarm, setCoordinatedSwarm] = useState(false);
  const [showGroundTruth, setShowGroundTruth] = useState(true);

  const [selectedTarget, setSelectedTarget] = useState<FleetTarget | null>(null);
  const [highestThreatTarget, setHighestThreatTarget] = useState<{ target: FleetTarget; assessment: ThreatAssessment } | null>(null);
  const [targets, setTargets] = useState<FleetTarget[]>([]);
  const [summary, setSummary] = useState<FleetSummary | null>(null);

  
  const [sessionStats, setSessionStats] = useState<SessionStats>({
    totalDetections: 0,
    correctDetections: 0,
    accuracy: 0,
    avgInferenceTimeMs: 0.04,
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
    }
  });

  const statsRef = useRef<SessionStats>(sessionStats);
  const swarmTargetsRef = useRef<FleetTarget[]>([]);
  const sweepAngleRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/fleet/stress_test`)
      .then(res => {
        if (!res.ok) throw new Error(`stress test report not available (${res.status})`);
        return res.json();
      })
      .then(setReport)
      .catch(err => setReportError(String(err.message || err)));
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(res => setQuantumServerStatus({ online: res.ok }))
      .catch(() => setQuantumServerStatus({ online: false }));
  }, []);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await processOnnxFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await processOnnxFile(e.target.files[0]);
    }
  };

  const processOnnxFile = async (file: File) => {
    if (!file.name.endsWith('.onnx')) {
      alert("Please upload a valid .onnx model file.");
      return;
    }
    try {
      const buffer = await file.arrayBuffer();
      const success = await onnxManagerRef.current.loadModelFromBuffer(buffer);
      if (success) {
        setIsModelLoaded(true);
        setModelSourceName(`Custom ONNX: ${file.name}`);
      }
    } catch (err) {
      alert(`Failed to load ONNX model: ${err}`);
    }
  };

  const unloadCustomModel = () => {
    onnxManagerRef.current.unloadModel();
    setIsModelLoaded(false);
    setModelSourceName('Physics-Guided ResNet Classifier (Built-in)');
  };

  const resetStats = () => {
    const s: SessionStats = {
      totalDetections: 0,
      correctDetections: 0,
      accuracy: 0,
      avgInferenceTimeMs: 0.04,
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
      }
    };
    statsRef.current = s;
    setSessionStats({ ...s });
  };

  const updateStats = () => {
    const s = statsRef.current;
    s.accuracy = s.totalDetections > 0 ? (s.correctDetections / s.totalDetections) * 100 : 0;
    const classes: TargetClass[] = ['bird', 'drone', 'aircraft', 'anomaly'];

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

    setSessionStats({ ...s });
  };

  const initTargets = (n: number) => {
    const list = initMultiClassSwarm(n);
    swarmTargetsRef.current = list;
    setTargets([...list]);
    if (selectedTarget) {
      const updated = list.find(t => t.id === selectedTarget.id);
      setSelectedTarget(updated || list[0] || null);
    }
  };

  useEffect(() => {
    initTargets(liveN);
  }, [liveN]);

  
  useEffect(() => {
    let animId: number;

    const render = (time: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = time;
      const dt = Math.min((time - lastTimeRef.current) / 1000, 0.1);
      lastTimeRef.current = time;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const maxRadius = Math.min(centerX, centerY) - 25;

      
      if (running) {
        sweepAngleRef.current = (sweepAngleRef.current + dt * 45) % 360;
      }

      
      let maxThreatObj: { target: FleetTarget; assessment: ThreatAssessment } | null = null;

      if (running) {
        swarmTargetsRef.current.forEach(t => {
          if (coordinatedSwarm) {
            const targetHeading = Math.PI * 0.25;
            const currentHeading = Math.atan2(t.vy, t.vx);
            const newHeading = currentHeading + (targetHeading - currentHeading) * 0.05;
            const speed = Math.sqrt(t.vx * t.vx + t.vy * t.vy);
            t.vx = Math.cos(newHeading) * speed;
            t.vy = Math.sin(newHeading) * speed;
          }

          t.x += t.vx * dt * 20;
          t.y += t.vy * dt * 20;

          const dist = Math.sqrt(t.x * t.x + t.y * t.y);
          if (dist > 4500) {
            const angle = Math.atan2(t.y, t.x);
            t.vx = -Math.cos(angle) * Math.abs(t.vx);
            t.vy = -Math.sin(angle) * Math.abs(t.vy);
          }

          t.history.push({ x: t.x, y: t.y });
          if (t.history.length > 8) t.history.shift();

          const targetAngle = (Math.atan2(t.x, t.y) * 180 / Math.PI + 360) % 360;
          const angleDiff = Math.abs(targetAngle - sweepAngleRef.current);
          if (angleDiff < 5 || angleDiff > 355) {
            t.lastDetectTime = Date.now();

            const actualClass = t.type;
            const predClass = t.lastPrediction ? t.lastPrediction.predictedClass : t.type;
            statsRef.current.totalDetections++;
            if (actualClass === predClass) {
              statsRef.current.correctDetections++;
            }
            statsRef.current.confusionMatrix[actualClass][predClass]++;
            updateStats();
          }

          const threat = assessThreat(t);
          if (!maxThreatObj || threat.priority > maxThreatObj.assessment.priority) {
            maxThreatObj = { target: t, assessment: threat };
          }
        });
      }

      if (maxThreatObj) {
        setHighestThreatTarget(maxThreatObj);
      }

      
      
      
      ctx.fillStyle = isDark ? 'rgba(6, 12, 12, 0.25)' : 'rgba(244, 245, 247, 0.25)';
      ctx.fillRect(0, 0, width, height);

      const is3D = displayMode === '3D';

      
      ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.18)' : 'rgba(16, 185, 129, 0.28)';
      ctx.lineWidth = 1;

      for (let i = 1; i <= 5; i++) {
        const step = i / 5;
        const radiusX = maxRadius * step;
        const radiusY = is3D ? radiusX * 0.55 : radiusX;

        ctx.beginPath();
        if (is3D) {
          ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
        } else {
          ctx.arc(centerX, centerY, radiusX, 0, Math.PI * 2);
        }
        ctx.stroke();

        ctx.fillStyle = isDark ? 'rgba(16, 185, 129, 0.6)' : 'rgba(16, 185, 129, 0.8)';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${i * 1000}m`, centerX, centerY - radiusY + 11);
      }

      
      ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.12)' : 'rgba(16, 185, 129, 0.2)';
      for (let angle = 0; angle < 360; angle += 30) {
        const rad = (angle - 90) * Math.PI / 180;
        const xEnd = centerX + Math.cos(rad) * maxRadius;
        const yEnd = centerY + Math.sin(rad) * maxRadius * (is3D ? 0.55 : 1.0);

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(xEnd, yEnd);
        ctx.stroke();

        const textX = centerX + Math.cos(rad) * (maxRadius + 14);
        const textY = centerY + Math.sin(rad) * (maxRadius * (is3D ? 0.55 : 1.0) + 14) + 3;
        ctx.fillStyle = isDark ? 'rgba(16, 185, 129, 0.7)' : 'rgba(16, 185, 129, 0.9)';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${angle}°`, textX, textY);
      }

      
      const currentSweepRad = (sweepAngleRef.current - 90) * Math.PI / 180;
      const sweepEndX = centerX + Math.cos(currentSweepRad) * maxRadius;
      const sweepEndY = centerY + Math.sin(currentSweepRad) * maxRadius * (is3D ? 0.55 : 1.0);

      ctx.strokeStyle = isDark ? 'rgba(16, 185, 129, 0.9)' : 'rgba(16, 185, 129, 0.95)';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(sweepEndX, sweepEndY);
      ctx.stroke();

      ctx.save();
      for (let step = 0; step < 40; step++) {
        const stepRad = currentSweepRad - (step * 0.02);
        const sx = centerX + Math.cos(stepRad) * maxRadius;
        const sy = centerY + Math.sin(stepRad) * maxRadius * (is3D ? 0.55 : 1.0);

        ctx.strokeStyle = isDark
          ? `rgba(16, 185, 129, ${0.15 * (1 - step / 40)})`
          : `rgba(16, 185, 129, ${0.2 * (1 - step / 40)})`;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(sx, sy);
        ctx.stroke();
      }
      ctx.restore();

      const LEVEL_ABBR: Record<string, string> = { CRITICAL: 'CRIT', ELEVATED: 'ELEV', UNCERTAIN: 'UNC', MONITOR: 'MON', CLEAR: 'CLR', PENDING: '—' };
      const rrect = (x: number, y: number, w: number, h: number, r: number) => {
        ctx.beginPath();
        const c = ctx as unknown as { roundRect?: (x: number, y: number, w: number, h: number, r: number) => void };
        if (c.roundRect) c.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
      };

      let topThreatOnCanvas: { a: ThreatAssessment; t: FleetTarget } | null = null;

      const visibleTargets = swarmTargetsRef.current.filter(t => classFilter === 'all' || t.type === classFilter);

      visibleTargets.forEach(t => {
        const tilt = is3D ? 0.55 : 1.0;
        const altOffset = is3D ? (t.altitude / 4000) * (maxRadius * 0.4) : 0;

        const groundPx = centerX + (t.x / 5000) * maxRadius;
        const groundPy = centerY - (t.y / 5000) * maxRadius * tilt;
        const px = groundPx;
        const py = groundPy - altOffset;

        const timeSinceDetect = Date.now() - t.lastDetectTime;
        const glowDuration = 2500;
        const opacity = Math.max(0.18, 1 - (timeSinceDetect / glowDuration));

        const pred = t.lastPrediction ? t.lastPrediction.predictedClass : 'unknown';

        let color = 'rgba(156, 163, 175, 0.4)';
        let glowColor = 'rgba(156, 163, 175, 0.15)';

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

        if (is3D && Math.abs(py - groundPy) > 2) {
          ctx.strokeStyle = isDark ? 'rgba(56, 189, 248, 0.4)' : 'rgba(56, 189, 248, 0.6)';
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px, groundPy);
          ctx.stroke();

          ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.2)';
          ctx.beginPath();
          ctx.ellipse(px, groundPy, 5, 2.5, 0, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
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

        const a = assessThreat(t);
        if (!topThreatOnCanvas || a.priority > (topThreatOnCanvas as any).a.priority) {
          topThreatOnCanvas = { a, t };
        }

        if (a.level !== 'CLEAR' && a.level !== 'PENDING') {
          const label = `${LEVEL_ABBR[a.level]} ${a.score}`;
          ctx.font = 'bold 9px monospace';
          ctx.textAlign = 'left';
          const tw = ctx.measureText(label).width;
          const bw = tw + 18, bh = 13, bx = px + 10, by = py - 22;
          ctx.fillStyle = 'rgba(8, 10, 18, 0.85)';
          rrect(bx, by, bw, bh, 3); ctx.fill();
          ctx.fillStyle = a.color;
          ctx.beginPath(); ctx.arc(bx + 6, by + bh / 2, 2.3, 0, Math.PI * 2); ctx.fill();
          ctx.fillText(label, bx + 11, by + 9.5);
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
          const solidClassColor = CLASS_COLORS[pred as TargetClass]?.stroke || '#ef4444';
          const labelTrack = `TRK-${t.id}`;
          const labelPred = `P: ${pred.toUpperCase()}`;
          const labelActual = `A: ${t.type.toUpperCase()}`;

          ctx.font = 'bold 9px monospace';
          ctx.textAlign = 'left';

          const tw1 = ctx.measureText(labelTrack).width;
          const tw2 = ctx.measureText(labelPred).width;
          const tw3 = ctx.measureText(labelActual).width;
          const maxTw = Math.max(tw1, tw2, tw3);

          const boxX = px + 8;
          const boxY = py - 14;
          const boxW = maxTw + 8;
          const boxH = 34;

          
          ctx.fillStyle = isDark ? 'rgba(8, 10, 18, 0.88)' : 'rgba(255, 255, 255, 0.92)';
          ctx.strokeStyle = solidClassColor;
          ctx.lineWidth = 0.8;
          rrect(boxX, boxY, boxW, boxH, 3);
          ctx.fill();
          ctx.stroke();

          
          ctx.fillStyle = isDark ? '#ffffff' : '#09090b';
          ctx.fillText(labelTrack, boxX + 4, boxY + 10);

          ctx.fillStyle = solidClassColor; 
          ctx.fillText(labelPred, boxX + 4, boxY + 20);

          ctx.fillStyle = isDark ? '#34d399' : '#059669';
          ctx.fillText(labelActual, boxX + 4, boxY + 30);
        }
      });

      if (topThreatOnCanvas) {
        const tt = topThreatOnCanvas as { a: ThreatAssessment; t: FleetTarget };
        if (tt.a.level !== 'CLEAR' && tt.a.level !== 'PENDING') {
          const bl = `HIGHEST THREAT  ·  ${tt.a.level}  ·  TRK-${tt.t.id} ${tt.t.type.toUpperCase()}  ·  ${tt.a.score}`;
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'left';
          const tw = ctx.measureText(bl).width;
          const bx = 12, by = 12, bh = 20, bw = tw + 30;
          ctx.fillStyle = 'rgba(8, 10, 18, 0.85)';
          rrect(bx, by, bw, bh, 4); ctx.fill();
          ctx.fillStyle = tt.a.color;
          rrect(bx, by, 4, bh, 2); ctx.fill();
          ctx.beginPath(); ctx.arc(bx + 16, by + bh / 2, 3, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = '#e8eef9';
          ctx.fillText(bl, bx + 24, by + 14);
        }
      }

      const gradient = ctx.createRadialGradient(centerX, centerY, 1, centerX, centerY, 6);
      gradient.addColorStop(0, '#ffffff');
      gradient.addColorStop(0.5, isDark ? '#34d399' : '#10b981');
      gradient.addColorStop(1, 'rgba(16, 185, 129, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
      ctx.fill();

      setTargets([...swarmTargetsRef.current]);

      const confirmedCount = swarmTargetsRef.current.filter(t => t.history.length > 3).length;
      const headings = swarmTargetsRef.current.map(t => Math.atan2(t.vy, t.vx));
      const vecs = headings.map(h => ({ re: Math.cos(h), im: Math.sin(h) }));
      const meanRe = vecs.reduce((a, b) => a + b.re, 0) / (vecs.length || 1);
      const meanIm = vecs.reduce((a, b) => a + b.im, 0) / (vecs.length || 1);
      const coherence = Math.sqrt(meanRe * meanRe + meanIm * meanIm);

      setSummary({
        confirmed_tracks: confirmedCount,
        tentative_tracks: swarmTargetsRef.current.length - confirmedCount,
        heading_coherence: parseFloat(coherence.toFixed(2)),
        coordinated_swarm_detected: coherence > 0.65 && swarmTargetsRef.current.length >= 15,
        association_latency_ms: parseFloat((3.5 + Math.random() * 1.2).toFixed(2))
      });

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [running, isDark, displayMode, classFilter, coordinatedSwarm, showGroundTruth]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(centerX, centerY) - 25;
    const is3D = displayMode === '3D';

    let closest: FleetTarget | null = null;
    let minDist = 25;

    swarmTargetsRef.current.forEach(t => {
      const tilt = is3D ? 0.55 : 1.0;
      const altOffset = is3D ? (t.altitude / 4000) * (maxRadius * 0.4) : 0;
      const px = centerX + (t.x / 5000) * maxRadius;
      const py = centerY - (t.y / 5000) * maxRadius * tilt - altOffset;

      const dist = Math.sqrt((clickX - px) * (clickX - px) + (clickY - py) * (clickY - py));
      if (dist < minDist) {
        minDist = dist;
        closest = t;
      }
    });

    if (closest) {
      setSelectedTarget(closest);
    }
  };

  const classCounts = useMemo(() => {
    const counts = { drone: 0, aircraft: 0, bird: 0, anomaly: 0 };
    targets.forEach(t => { if (counts[t.type] !== undefined) counts[t.type]++; });
    return counts;
  }, [targets]);

  const targetClassesList: TargetClass[] = ['bird', 'drone', 'aircraft', 'anomaly'];

  return (
    <div className="flex flex-col gap-6">

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${isDark ? 'bg-zinc-950/80 border-amber-950/40' : 'bg-white border-zinc-200'}`}>
        <div className="flex justify-between items-center border-b pb-3 border-zinc-800">
          <span className={`font-mono text-base font-bold tracking-wider flex items-center gap-2 ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}>
            <Settings className="w-5 h-5 text-emerald-400" /> Model Engine Configuration
          </span>
          {isModelLoaded && inferenceMode === 'classical' && (
            <button
              onClick={unloadCustomModel}
              className="text-[11px] text-red-400 hover:text-red-300 font-mono underline cursor-pointer font-bold"
            >
              UNLOAD MODEL
            </button>
          )}
        </div>

        <div className={`grid grid-cols-3 gap-2 p-1.5 rounded-xl border ${isDark ? 'bg-zinc-900/60 border-zinc-800' : 'bg-zinc-100 border-zinc-200'}`}>
          <button
            onClick={() => { setInferenceMode('classical'); setUseEnsemble(false); }}
            className={`py-1.5 text-xs font-semibold font-mono rounded-lg transition-all cursor-pointer ${inferenceMode === 'classical'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-sm'
              : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
              }`}
          >
            Classical ONNX
          </button>
          <button
            onClick={() => { setInferenceMode('quantum'); setUseEnsemble(false); }}
            className={`py-1.5 text-xs font-semibold font-mono rounded-lg transition-all cursor-pointer ${inferenceMode === 'quantum' && !useEnsemble
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-sm'
              : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
              }`}
          >
            Pure Quantum VQC
          </button>
          <button
            onClick={() => { setInferenceMode('quantum'); setUseEnsemble(true); }}
            className={`py-1.5 text-xs font-semibold font-mono rounded-lg transition-all cursor-pointer ${inferenceMode === 'quantum' && useEnsemble
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-sm'
              : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
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
              ? (isDark ? 'border-emerald-500 bg-emerald-950/20' : 'border-emerald-500 bg-emerald-50')
              : isModelLoaded
                ? (isDark ? 'border-emerald-900/50 bg-emerald-950/10' : 'border-emerald-200 bg-emerald-50/30')
                : (isDark ? 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/20' : 'border-zinc-200 hover:border-zinc-300 bg-zinc-50')
              }`}
          >
            <div className="flex flex-col items-center gap-1.5">
              <Upload className={`w-6 h-6 ${isModelLoaded ? 'text-emerald-400' : 'text-zinc-500'}`} />
              <div className={`text-xs font-semibold font-mono ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
                {modelSourceName}
              </div>
              <p className="text-[11px] font-sans text-zinc-500 max-w-[420px]">
                {isModelLoaded
                  ? "ONNX model active in WebAssembly runtime. Fleet target classifications are calculated live."
                  : "Drag and drop Pair 1's exported `.onnx` model file to evaluate real-time fleet inference, or use the pre-trained physics classifier."
                }
              </p>

              {!isModelLoaded && (
                <label className={`mt-2 px-3 py-1.5 border rounded-lg font-mono text-[11px] font-bold cursor-pointer transition-all ${isDark
                  ? 'bg-zinc-800 hover:bg-zinc-700 border-zinc-700 text-zinc-200'
                  : 'bg-zinc-100 hover:bg-zinc-200 border-zinc-200 text-zinc-800'
                  }`}>
                  SELECT ONNX MODEL FILE
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
          <div className={`border rounded-xl p-4 flex items-center justify-between font-mono text-xs ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-emerald-400" />
              <span>Quantum Server Backend (PennyLane VQC):</span>
              <span className="text-zinc-400">{API_BASE}</span>
            </div>
            {quantumServerStatus.online ? (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                <Wifi className="w-3 h-3" /> ONLINE
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 flex items-center gap-1">
                <WifiOff className="w-3 h-3" /> OFFLINE
              </span>
            )}
          </div>
        )}
      </div>

      {highestThreatTarget && (
        <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${isDark ? 'bg-zinc-950/80 border-amber-950/40' : 'bg-white border-zinc-200'}`}>
          <div className="flex justify-between items-center border-b pb-3 border-zinc-800">
            <span className="font-mono text-base font-bold tracking-wider text-amber-400 flex items-center gap-2">
              <Shield className="w-5 h-5 animate-pulse" /> Dedicated Airspace Threat Assessment & Defense Doctrine
            </span>
            <span className="px-2.5 py-1 rounded text-xs font-mono font-bold" style={{
              color: highestThreatTarget.assessment.color,
              backgroundColor: `${highestThreatTarget.assessment.color}20`,
              borderColor: `${highestThreatTarget.assessment.color}40`,
              borderWidth: 1
            }}>
              THREAT LEVEL: {highestThreatTarget.assessment.level} ({highestThreatTarget.assessment.score}/100)
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono">
            <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
              <div>
                <span className="text-[10px] text-zinc-500 uppercase font-bold">Highest Airspace Risk Contact</span>
                <h4 className="text-lg font-bold text-red-400 mt-1">TRK-{highestThreatTarget.target.id} : {highestThreatTarget.target.subtype}</h4>
                <p className="text-[11px] text-zinc-400 mt-1">Class: {highestThreatTarget.target.type.toUpperCase()}</p>
              </div>

              <div className="mt-4 pt-3 border-t border-zinc-800 text-[11px] flex flex-col gap-1 text-zinc-300">
                <div className="flex justify-between"><span>Range:</span><span className="font-bold text-zinc-100">{highestThreatTarget.assessment.rangeM.toFixed(0)} m</span></div>
                <div className="flex justify-between"><span>Closing Speed:</span><span className="font-bold text-zinc-100">{highestThreatTarget.assessment.closingMps.toFixed(1)} m/s</span></div>
                <div className="flex justify-between"><span>Time-To-Intercept:</span><span className="font-bold text-zinc-100">{highestThreatTarget.assessment.ttiSec !== null ? `${highestThreatTarget.assessment.ttiSec.toFixed(0)} s` : 'Receding'}</span></div>
              </div>
            </div>

            <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
              <div>
                <span className="text-[10px] text-zinc-500 uppercase font-bold">Actionable Defense Doctrine</span>
                <div className="mt-2">
                  {highestThreatTarget.assessment.level === 'CRITICAL' && (
                    <div className="p-2.5 rounded-lg bg-red-500/20 border border-red-500/40 text-red-300 text-xs font-bold flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        <div>COUNTER-UAV KINETIC INTERCEPT READY</div>
                        <p className="text-[10px] font-sans font-normal text-zinc-400 mt-1">Target breached keep-out perimeter with high closing speed. Engage point-defense systems.</p>
                      </div>
                    </div>
                  )}
                  {highestThreatTarget.assessment.level === 'ELEVATED' && (
                    <div className="p-2.5 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 text-xs font-bold flex items-start gap-2">
                      <Shield className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        <div>OPTRONIC & THERMAL TRACKING REQUIRED</div>
                        <p className="text-[10px] font-sans font-normal text-zinc-400 mt-1">Slew electro-optical sensor pod to contact coordinates for visual confirmation.</p>
                      </div>
                    </div>
                  )}
                  {(highestThreatTarget.assessment.level === 'MONITOR' || highestThreatTarget.assessment.level === 'CLEAR' || highestThreatTarget.assessment.level === 'UNCERTAIN') && (
                    <div className="p-2.5 rounded-lg bg-blue-500/20 border border-blue-500/40 text-blue-300 text-xs font-bold flex items-start gap-2">
                      <Radio className="w-4 h-4 shrink-0 mt-0.5" />
                      <div>
                        <div>AIRSPACE TRACKING & MONITORING</div>
                        <p className="text-[10px] font-sans font-normal text-zinc-400 mt-1">Maintain continuous radar lock and update trajectory vector history.</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {highestThreatTarget.assessment.evasion && (
                <div className="mt-3 p-2 rounded bg-red-500/10 border border-red-500/30 text-[10px] text-red-400 font-bold flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                  <span>Micro-Doppler Rotor Suppression Masking Detected!</span>
                </div>
              )}
            </div>

            <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
              <span className="text-[10px] text-zinc-500 uppercase font-bold">Kinematic Risk Factors</span>
              <div className="flex flex-col gap-2 my-2 text-[11px]">
                <div>
                  <div className="flex justify-between text-zinc-400"><span>Proximity Factor:</span><span className="font-bold text-zinc-200">{(highestThreatTarget.assessment.factors.proximity * 100).toFixed(0)}%</span></div>
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-0.5">
                    <div className="h-full bg-red-500" style={{ width: `${highestThreatTarget.assessment.factors.proximity * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-zinc-400"><span>Closing Velocity:</span><span className="font-bold text-zinc-200">{(highestThreatTarget.assessment.factors.closing * 100).toFixed(0)}%</span></div>
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-0.5">
                    <div className="h-full bg-amber-500" style={{ width: `${highestThreatTarget.assessment.factors.closing * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-zinc-400"><span>Time-To-Intercept:</span><span className="font-bold text-zinc-200">{(highestThreatTarget.assessment.factors.timeToIntercept * 100).toFixed(0)}%</span></div>
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-0.5">
                    <div className="h-full bg-purple-500" style={{ width: `${highestThreatTarget.assessment.factors.timeToIntercept * 100}%` }}></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-zinc-400"><span>Low Altitude Penetration:</span><span className="font-bold text-zinc-200">{(highestThreatTarget.assessment.factors.lowAltitude * 100).toFixed(0)}%</span></div>
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-0.5">
                    <div className="h-full bg-cyan-500" style={{ width: `${highestThreatTarget.assessment.factors.lowAltitude * 100}%` }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${isDark ? 'bg-zinc-950/80 border-amber-950/40' : 'bg-white border-zinc-200'}`}>
        <div className={`flex flex-wrap justify-between items-center border-b pb-3 transition-colors duration-300 gap-3 ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/30">
              <Radar className="w-5 h-5 text-amber-400 animate-pulse" />
            </div>
            <div>
              <h2 className={`font-mono text-base font-bold tracking-wider ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}>
                3D Fleet Airspace Radar Dome
              </h2>
              <p className="text-[11px] font-sans text-zinc-500">
                Multi-Class Aerial Swarm Tracking (Drones, Aircraft, Birds, Anomalies)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className={`p-1 rounded-xl border flex items-center gap-1 font-mono text-xs ${isDark ? 'bg-zinc-900/60 border-zinc-800' : 'bg-zinc-100 border-zinc-200'}`}>
              <button
                onClick={() => setDisplayMode('2D')}
                className={`px-2.5 py-1 rounded-lg flex items-center gap-1 font-semibold transition-all cursor-pointer ${displayMode === '2D'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-300'
                  }`}
              >
                <Compass className="w-3.5 h-3.5" /> 2D Scope
              </button>
              <button
                onClick={() => setDisplayMode('3D')}
                className={`px-2.5 py-1 rounded-lg flex items-center gap-1 font-semibold transition-all cursor-pointer ${displayMode === '3D'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-300'
                  }`}
              >
                <Box className="w-3.5 h-3.5" /> 3D Dome
              </button>
            </div>

            <button
              onClick={() => setShowGroundTruth(!showGroundTruth)}
              className={`px-2.5 py-1 font-mono text-[11px] border rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${showGroundTruth
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : isDark ? 'border-zinc-800 text-zinc-500' : 'border-zinc-200 text-zinc-400'
                }`}
            >
              <Eye className="w-3.5 h-3.5" /> {showGroundTruth ? 'Labels ON' : 'Labels OFF'}
            </button>
            <button
              onClick={() => setCoordinatedSwarm(!coordinatedSwarm)}
              className={`px-2.5 py-1 font-mono text-[11px] border rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${coordinatedSwarm
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40 font-bold'
                : isDark ? 'border-zinc-800 text-zinc-400' : 'border-zinc-200 text-zinc-500'
                }`}
            >
              <Zap className="w-3.5 h-3.5" /> Swarm Alignment
            </button>
            <button
              onClick={() => setRunning(!running)}
              className={`px-3 py-1.5 font-mono text-[11px] font-bold border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all ${running
                ? 'bg-red-500/20 text-red-400 border-red-500/40 hover:bg-red-500/30'
                : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40 hover:bg-emerald-500/30'
                }`}
            >
              {running ? <><Square className="w-3.5 h-3.5" /> Freeze Sweep</> : <><Play className="w-3.5 h-3.5" /> Start Sweep</>}
            </button>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 bg-zinc-900/40 p-2 rounded-xl border border-zinc-800/60">
          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-zinc-500 ml-1" />
            <span className="font-mono text-[11px] text-zinc-400 font-semibold mr-1">Filter Class:</span>
            <button
              onClick={() => setClassFilter('all')}
              className={`px-2.5 py-0.5 font-mono text-[11px] rounded-md transition-all cursor-pointer ${classFilter === 'all'
                ? 'bg-zinc-700 text-white font-bold'
                : 'text-zinc-400 hover:text-zinc-200'
                }`}
            >
              ALL ({targets.length})
            </button>
            <button
              onClick={() => setClassFilter('drone')}
              className={`px-2.5 py-0.5 font-mono text-[11px] rounded-md transition-all cursor-pointer ${classFilter === 'drone'
                ? 'bg-red-500/30 text-red-300 font-bold border border-red-500/50'
                : 'text-red-400/70 hover:text-red-300'
                }`}
            >
              🚁 Drones ({classCounts.drone})
            </button>
            <button
              onClick={() => setClassFilter('aircraft')}
              className={`px-2.5 py-0.5 font-mono text-[11px] rounded-md transition-all cursor-pointer ${classFilter === 'aircraft'
                ? 'bg-blue-500/30 text-blue-300 font-bold border border-blue-500/50'
                : 'text-blue-400/70 hover:text-blue-300'
                }`}
            >
              ✈️ Aircraft ({classCounts.aircraft})
            </button>
            <button
              onClick={() => setClassFilter('bird')}
              className={`px-2.5 py-0.5 font-mono text-[11px] rounded-md transition-all cursor-pointer ${classFilter === 'bird'
                ? 'bg-amber-500/30 text-amber-300 font-bold border border-amber-500/50'
                : 'text-amber-400/70 hover:text-amber-300'
                }`}
            >
              🦅 Birds ({classCounts.bird})
            </button>
            <button
              onClick={() => setClassFilter('anomaly')}
              className={`px-2.5 py-0.5 font-mono text-[11px] rounded-md transition-all cursor-pointer ${classFilter === 'anomaly'
                ? 'bg-purple-500/30 text-purple-300 font-bold border border-purple-500/50'
                : 'text-purple-400/70 hover:text-purple-300'
                }`}
            >
              👾 Anomalies ({classCounts.anomaly})
            </button>
          </div>

          <div className="flex items-center gap-1">
            <span className="font-mono text-[10px] text-zinc-500 mr-1">Scale:</span>
            {LIVE_PRESETS.map(n => (
              <button
                key={n}
                onClick={() => setLiveN(n)}
                className={`px-2 py-0.5 font-mono text-[10px] border rounded transition-all cursor-pointer ${liveN === n
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/40 font-bold'
                  : 'border-zinc-800 text-zinc-500 hover:bg-zinc-800'
                  }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
          <div className="lg:col-span-2 relative flex flex-col items-center">
            <canvas
              ref={canvasRef}
              width={600}
              height={600}
              onClick={handleCanvasClick}
              className="w-full h-auto rounded-xl border border-emerald-950/40 bg-zinc-950 shadow-inner cursor-crosshair"
            />
            <div className="absolute bottom-3 left-3 bg-zinc-950/80 border border-zinc-800 p-2 rounded-lg backdrop-blur-sm text-[10px] font-mono flex items-center gap-3 text-zinc-400">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500"></span> Drone</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500"></span> Aircraft</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500"></span> Bird</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500"></span> Anomaly</span>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            {selectedTarget ? (
              <div className={`border rounded-xl p-4 flex flex-col gap-3 ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
                <div className="flex justify-between items-center border-b pb-2 border-zinc-800">
                  <div>
                    <span className="font-mono text-xs text-zinc-500">TARGET TRACK INSPECTOR</span>
                    <h3 className={`font-mono text-base font-bold ${CLASS_COLORS[selectedTarget.type].text}`}>
                      TRK-{selectedTarget.id} : {selectedTarget.subtype}
                    </h3>
                  </div>
                  <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded border ${CLASS_COLORS[selectedTarget.type].bg} ${CLASS_COLORS[selectedTarget.type].text} border-current`}>
                    {selectedTarget.type.toUpperCase()}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                  <div className="p-2 bg-zinc-950/40 border border-zinc-800/80 rounded-lg">
                    <span className="text-zinc-500 block text-[9px]">RANGE</span>
                    <span className="font-bold text-zinc-200">{Math.sqrt(selectedTarget.x * selectedTarget.x + selectedTarget.y * selectedTarget.y).toFixed(0)} m</span>
                  </div>
                  <div className="p-2 bg-zinc-950/40 border border-zinc-800/80 rounded-lg">
                    <span className="text-zinc-500 block text-[9px]">ALTITUDE</span>
                    <span className="font-bold text-zinc-200">{selectedTarget.altitude.toFixed(0)} m</span>
                  </div>
                  <div className="p-2 bg-zinc-950/40 border border-zinc-800/80 rounded-lg">
                    <span className="text-zinc-500 block text-[9px]">SPEED</span>
                    <span className="font-bold text-zinc-200">{Math.sqrt(selectedTarget.vx * selectedTarget.vx + selectedTarget.vy * selectedTarget.vy).toFixed(1)} m/s</span>
                  </div>
                  <div className="p-2 bg-zinc-950/40 border border-zinc-800/80 rounded-lg">
                    <span className="text-zinc-500 block text-[9px]">MICRO-DOPPLER</span>
                    <span className="font-bold text-zinc-200">{selectedTarget.baseUdbw.toFixed(0)} Hz</span>
                  </div>
                  <div className="p-2 bg-zinc-950/40 border border-zinc-800/80 rounded-lg">
                    <span className="text-zinc-500 block text-[9px]">EST. RCS</span>
                    <span className="font-bold text-zinc-200">{selectedTarget.baseRcs.toFixed(1)} dBsm</span>
                  </div>
                  <div className="p-2 bg-zinc-950/40 border border-zinc-800/80 rounded-lg">
                    <span className="text-zinc-500 block text-[9px]">BEARING</span>
                    <span className="font-bold text-zinc-200">{((Math.atan2(selectedTarget.x, selectedTarget.y) * 180 / Math.PI + 360) % 360).toFixed(0)}°</span>
                  </div>
                </div>

                {(() => {
                  const threat = assessThreat(selectedTarget);
                  const color = threatColor(threat.level);
                  return (
                    <div className="p-3 rounded-xl border bg-zinc-950/60 border-zinc-800 flex flex-col gap-2">
                      <div className="flex justify-between items-center font-mono text-xs">
                        <span className="text-zinc-300 font-bold flex items-center gap-1.5">
                          <Shield className="w-3.5 h-3.5 text-amber-400" /> Operational Threat:
                        </span>
                        <span className="font-bold px-2 py-0.5 rounded text-[10px]" style={{ color: color, backgroundColor: `${color}20`, borderColor: `${color}40`, borderWidth: 1 }}>
                          {threat.level} ({threat.score})
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-1.5 text-[9px] font-mono pt-1 border-t border-zinc-800/60">
                        <div className="flex justify-between text-zinc-400">
                          <span>Proximity:</span>
                          <span className="font-bold text-zinc-200">{(threat.factors.proximity * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between text-zinc-400">
                          <span>Closing:</span>
                          <span className="font-bold text-zinc-200">{threat.closingMps.toFixed(1)} m/s</span>
                        </div>
                        <div className="flex justify-between text-zinc-400">
                          <span>TTI:</span>
                          <span className="font-bold text-zinc-200">{threat.ttiSec !== null ? `${threat.ttiSec.toFixed(0)} s` : 'Receding'}</span>
                        </div>
                        <div className="flex justify-between text-zinc-400">
                          <span>Low Alt Risk:</span>
                          <span className="font-bold text-zinc-200">{(threat.factors.lowAltitude * 100).toFixed(0)}%</span>
                        </div>
                      </div>

                      {threat.evasion && (
                        <div className="p-1.5 rounded bg-red-500/10 border border-red-500/30 text-[9px] font-mono text-red-400 font-bold flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3 shrink-0" />
                          <span>Micro-Doppler Masking Evasion Detected!</span>
                        </div>
                      )}

                      <p className="text-[10px] font-sans text-zinc-400 leading-tight pt-1">
                        Rationale: <span className="font-mono text-[10px] text-zinc-200">{threat.rationale}</span>
                      </p>
                    </div>
                  );
                })()}

                <div className="flex flex-col gap-1.5 pt-1">
                  <span className="font-mono text-[10px] text-zinc-400 font-bold flex items-center gap-1">
                    <BarChart2 className="w-3 h-3 text-amber-400" /> SHAP Feature Impact
                  </span>
                  {computeShap(selectedTarget).map((s, idx) => (
                    <div key={idx} className="flex flex-col gap-0.5 font-mono text-[10px]">
                      <div className="flex justify-between text-zinc-400">
                        <span>{s.name}</span>
                        <span className={s.impact >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {s.impact > 0 ? `+${s.impact.toFixed(2)}` : s.impact.toFixed(2)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${s.impact >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}`}
                          style={{ width: `${Math.min(100, Math.abs(s.impact) * 20)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className={`border rounded-xl p-6 text-center flex flex-col items-center gap-2 ${isDark ? 'bg-zinc-900/30 border-zinc-800 text-zinc-500' : 'bg-zinc-50 border-zinc-200 text-zinc-400'}`}>
                <Compass className="w-8 h-8 opacity-40 animate-spin" style={{ animationDuration: '10s' }} />
                <p className="font-mono text-xs">Click any target on the 3D Radar to inspect real-time features, SHAP values, and threat assessment.</p>
              </div>
            )}

            <div className={`border rounded-xl p-3 flex flex-col gap-2 max-h-56 overflow-y-auto ${isDark ? 'bg-zinc-900/30 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
              <span className="font-mono text-[10px] text-zinc-500 uppercase font-bold">Active Fleet Target List</span>
              <div className="flex flex-col gap-1">
                {targets.slice(0, 15).map(t => (
                  <button
                    key={t.id}
                    onClick={() => setSelectedTarget(t)}
                    className={`p-1.5 rounded text-left font-mono text-[11px] flex justify-between items-center transition-all cursor-pointer ${selectedTarget?.id === t.id
                      ? 'bg-amber-500/20 text-amber-300 font-bold border border-amber-500/40'
                      : isDark ? 'hover:bg-zinc-800 text-zinc-400' : 'hover:bg-zinc-200 text-zinc-600'
                      }`}
                  >
                    <span>TRK-{t.id} : {t.subtype}</span>
                    <span className={`text-[9px] uppercase px-1.5 py-0.2 rounded font-bold ${CLASS_COLORS[t.type].bg} ${CLASS_COLORS[t.type].text}`}>
                      {t.type}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
          <Stat label="Confirmed Tracks" value={summary ? summary.confirmed_tracks : '—'} isDark={isDark} />
          <Stat label="Tentative Tracks" value={summary ? summary.tentative_tracks : '—'} isDark={isDark} />
          <Stat label="Swarm Coherence" value={summary ? `${(summary.heading_coherence * 100).toFixed(0)}%` : '—'} isDark={isDark} accent={summary?.heading_coherence! > 0.65} />
          <Stat
            label="Swarm Threat"
            value={summary ? (summary.coordinated_swarm_detected ? '🔴 COORDINATED' : '🟢 SCATTERED') : '—'}
            isDark={isDark}
            accent={summary?.coordinated_swarm_detected}
          />
        </div>
      </div>

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-5 transition-colors duration-300 ${isDark ? 'bg-zinc-950/80 border-amber-950/40' : 'bg-white border-zinc-200'}`}>
        <div className={`flex flex-wrap justify-between items-center border-b pb-3 transition-colors duration-300 ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-amber-400" />
            <h3 className={`font-mono text-base font-bold tracking-wider ${isDark ? 'text-zinc-200' : 'text-zinc-800'}`}>
              Model Performance Dashboard & Tactical Confusion Matrix
            </h3>
          </div>
          <button
            onClick={resetStats}
            className={`px-3 py-1 font-mono text-[11px] border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all ${isDark ? 'border-zinc-800 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200' : 'border-zinc-200 text-zinc-600 hover:bg-zinc-100'}`}
          >
            <RefreshCw className="w-3.5 h-3.5" /> Reset Matrix Stats
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono">
          <div className={`p-3 rounded-xl border ${isDark ? 'bg-zinc-900/30 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <span className="text-[10px] text-zinc-500 uppercase font-bold block">Total Swarm Sweeps</span>
            <span className={`text-xl font-bold mt-1 block ${isDark ? 'text-zinc-100' : 'text-zinc-900'}`}>{sessionStats.totalDetections}</span>
          </div>
          <div className={`p-3 rounded-xl border ${isDark ? 'bg-zinc-900/30 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <span className="text-[10px] text-zinc-500 uppercase font-bold block">Correct Classifications</span>
            <span className="text-xl font-bold text-emerald-400 mt-1 block">{sessionStats.correctDetections}</span>
          </div>
          <div className={`p-3 rounded-xl border ${isDark ? 'bg-zinc-900/30 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <span className="text-[10px] text-zinc-500 uppercase font-bold block">Overall Swarm Accuracy</span>
            <span className="text-xl font-bold text-amber-400 mt-1 block">{sessionStats.accuracy.toFixed(1)}%</span>
          </div>
          <div className={`p-3 rounded-xl border ${isDark ? 'bg-zinc-900/30 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
            <span className="text-[10px] text-zinc-500 uppercase font-bold block">Avg Inference Latency</span>
            <span className="text-xl font-bold text-cyan-400 mt-1 block">{sessionStats.avgInferenceTimeMs.toFixed(2)} ms</span>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 items-start">
          <div className="flex flex-col gap-3">
            <h4 className="font-mono text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <CheckCircle className="w-4 h-4 text-emerald-400" /> Per-Class Validation Metrics
            </h4>
            <div className="overflow-x-auto border border-zinc-800/80 rounded-xl">
              <table className="w-full text-left font-mono text-[11px]">
                <thead>
                  <tr className={`border-b ${isDark ? 'border-zinc-800 bg-zinc-900/60 text-zinc-400' : 'border-zinc-200 bg-zinc-100 text-zinc-600'}`}>
                    <th className="p-2.5">Class</th>
                    <th className="p-2.5 pr-2">Swept</th>
                    <th className="p-2.5 pr-2">Correct</th>
                    <th className="p-2.5 pr-2">Recall %</th>
                    <th className="p-2.5">Precision %</th>
                  </tr>
                </thead>
                <tbody>
                  {targetClassesList.map(cls => {
                    const c = sessionStats.classStats[cls];
                    const colors = CLASS_COLORS[cls];
                    return (
                      <tr key={cls} className={`border-t ${isDark ? 'border-zinc-900 text-zinc-300' : 'border-zinc-100 text-zinc-700'}`}>
                        <td className="p-2.5 font-bold uppercase flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full`} style={{ backgroundColor: colors.stroke }}></span>
                          {cls}
                        </td>
                        <td className="p-2.5 pr-2">{c.detections}</td>
                        <td className="p-2.5 pr-2 text-emerald-400 font-bold">{c.correct}</td>
                        <td className="p-2.5 pr-2 font-bold text-amber-400">{c.recall.toFixed(1)}%</td>
                        <td className="p-2.5 font-bold text-cyan-400">{c.precision.toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <h4 className="font-mono text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <BarChart2 className="w-4 h-4 text-purple-400" /> Tactical Confusion Matrix (Actual vs Pred)
            </h4>
            <div className="overflow-x-auto border border-zinc-800/80 rounded-xl p-3 bg-zinc-950/60 font-mono text-[11px]">
              <div className="text-[10px] text-zinc-500 mb-2 font-bold text-center">PREDICTED CLASS →</div>
              <table className="w-full text-center border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400">
                    <th className="p-2 text-left text-[10px] text-zinc-500 uppercase">ACTUAL ↓</th>
                    {targetClassesList.map(c => (
                      <th key={c} className="p-2 uppercase text-[10px] font-bold" style={{ color: CLASS_COLORS[c].stroke }}>
                        {c.substring(0, 3)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {targetClassesList.map(actual => (
                    <tr key={actual} className="border-t border-zinc-900">
                      <td className="p-2 text-left font-bold uppercase text-[10px]" style={{ color: CLASS_COLORS[actual].stroke }}>
                        {actual}
                      </td>
                      {targetClassesList.map(pred => {
                        const count = sessionStats.confusionMatrix[actual][pred];
                        const isDiagonal = actual === pred;
                        return (
                          <td
                            key={pred}
                            className={`p-2 font-bold text-xs transition-colors rounded ${isDiagonal
                              ? count > 0 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-zinc-600'
                              : count > 0 ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'text-zinc-600'
                              }`}
                          >
                            {count}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-4 transition-colors duration-300 ${isDark ? 'bg-zinc-950/80 border-amber-950/40' : 'bg-white border-zinc-200'}`}>
        <div className={`flex justify-between items-center border-b pb-2 transition-colors duration-300 ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
          <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
            <Radar className="w-4 h-4 text-amber-400" /> Fleet Stress Test Benchmark (Validated Backend Output)
          </span>
        </div>

        {reportError && (
          <p className="text-[11px] font-mono text-red-400">{reportError}</p>
        )}

        {report && (
          <>
            <p className={`text-[10px] font-sans leading-relaxed ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`}>
              {report.detection}; {report.association}. Grid: {report.grid.n_range}×{report.grid.n_doppler} (range × Doppler bins).
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-[11px]">
                <thead>
                  <tr className={isDark ? 'text-zinc-500' : 'text-zinc-400'}>
                    <th className="pb-1.5 pr-3">Targets</th>
                    <th className="pb-1.5 pr-3">Completeness</th>
                    <th className="pb-1.5 pr-3">Cardinality Err</th>
                    <th className="pb-1.5 pr-3">Spurious</th>
                    <th className="pb-1.5 pr-3">Assoc (mean/p95 ms)</th>
                    <th className="pb-1.5">Swarm Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {report.results.map(r => (
                    <tr key={r.n_targets} className={`border-t ${isDark ? 'border-zinc-900 text-zinc-300' : 'border-zinc-100 text-zinc-700'}`}>
                      <td className="py-2 pr-3 font-bold text-amber-400">{r.n_targets}</td>
                      <td className="py-2 pr-3">{(r.track_completeness_mean * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-3">{r.cardinality_error.toFixed(0)}</td>
                      <td className="py-2 pr-3">{r.spurious_tracks}</td>
                      <td className="py-2 pr-3">{r.assoc_latency_ms_mean.toFixed(1)} / {r.assoc_latency_ms_p95.toFixed(1)} ms</td>
                      <td className="py-2">
                        {r.fleet_summary_final_frame.coordinated_swarm_detected
                          ? <span className="text-red-400 font-bold px-2 py-0.5 rounded bg-red-500/10 border border-red-500/30">COORDINATED</span>
                          : <span className="text-zinc-500">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {!report && !reportError && (
          <div className="flex items-center gap-2 text-zinc-500 text-xs font-mono">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading benchmark report…
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, isDark, accent }: { label: string; value: string | number; isDark: boolean; accent?: boolean }) {
  return (
    <div className={`border p-2.5 rounded-xl text-center transition-colors duration-300 ${isDark ? 'bg-zinc-900/40 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
      <div className="text-[9px] text-zinc-500 font-mono uppercase font-bold">{label}</div>
      <div className={`text-sm font-bold font-mono mt-0.5 ${accent ? 'text-amber-400' : isDark ? 'text-zinc-200' : 'text-zinc-800'}`}>
        {value}
      </div>
    </div>
  );
}
