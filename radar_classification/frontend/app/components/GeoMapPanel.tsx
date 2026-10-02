'use client';




































import React, { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  MapPin, Play, Square, Search, Crosshair, X, LocateFixed, Building2,
  Pause, RefreshCw, Cpu, CheckCircle, AlertTriangle,
} from 'lucide-react';
import {
  SUBTYPE_PRESETS, SUBTYPE_CLASS, type TargetSubtype,
  computeFeatureVectors, type SessionStats,
} from './RadarSimulator';
import { OnnxModelManager, type InferenceResult } from './OnnxInference';

const EARTH_RADIUS_M = 6378137;
const ALL_SUBTYPES = Object.keys(SUBTYPE_PRESETS) as TargetSubtype[];
export const DEFAULT_BUILDING_HEIGHT_M = 12;


const SNR_PRESETS = [
  { name: 'Airport', db: -12.0, desc: 'Moderate standard SNR' },
  { name: 'Forest', db: -17.0, desc: 'High forest clutter' },
  { name: 'Border', db: -10.0, desc: 'Clean border radar signal' },
  { name: 'Urban', db: -14.5, desc: 'Reflective urban canopy' },
  { name: 'Night', db: -10.0, desc: 'Stealth night operations' },
  { name: 'Heavy Noise', db: -23.0, desc: 'Extreme noise jamming environment' },
];







function freshStats(): SessionStats {
  return {
    totalDetections: 0, correctDetections: 0, accuracy: 0,
    confusionMatrix: {
      bird: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
      drone: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
      aircraft: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
      anomaly: { bird: 0, drone: 0, aircraft: 0, anomaly: 0 },
    },
    classStats: {
      bird: { detections: 0, correct: 0, recall: 0, precision: 0 },
      drone: { detections: 0, correct: 0, recall: 0, precision: 0 },
      aircraft: { detections: 0, correct: 0, recall: 0, precision: 0 },
      anomaly: { detections: 0, correct: 0, recall: 0, precision: 0 },
    },
    avgInferenceTimeMs: 0,
  };
}

export interface GeoTarget {
  id: number;
  subtype: TargetSubtype;
  cls: 'drone' | 'aircraft' | 'bird' | 'anomaly';
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
  lastPrediction: InferenceResult | null;
}

export interface BuildingStats {
  count: number;
  tagged: number;
  untagged: number;
  maxHeight: number;
  avgHeight: number;
}

const CLASS_COLOR: Record<GeoTarget['cls'], string> = {
  drone: '#fb4d5d', aircraft: '#54a5ff', bird: '#f59e0b', anomaly: '#b58bff',
};

export function metersToLatLon(originLat: number, originLon: number, xEast: number, yNorth: number): [number, number] {
  const latRad = (originLat * Math.PI) / 180;
  const dLat = (yNorth / EARTH_RADIUS_M) * (180 / Math.PI);
  const dLon = (xEast / (EARTH_RADIUS_M * Math.cos(latRad))) * (180 / Math.PI);
  return [originLat + dLat, originLon + dLon];
}





function spawnRealisticTarget(id: number): GeoTarget {
  const subtype = ALL_SUBTYPES[Math.floor(Math.random() * ALL_SUBTYPES.length)];
  const preset = SUBTYPE_PRESETS[subtype];
  const cls = SUBTYPE_CLASS[subtype];
  const spawnAngle = Math.random() * Math.PI * 2;
  const spawnDist = 1500 + Math.random() * 3000;
  const speed = preset.speedMin + Math.random() * (preset.speedMax - preset.speedMin);
  const vAngle = spawnAngle + Math.PI + (Math.random() - 0.5) * 0.8;
  return {
    id, subtype, cls,
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
    lastPrediction: null,
  };
}

function spawnRealisticTargets(n: number): GeoTarget[] {
  return Array.from({ length: n }, (_, i) => spawnRealisticTarget(i));
}


const LeafletMap = dynamic(() => import('./LeafletMapInner'), {
  ssr: false,
  loading: () => <div className="w-full h-[420px] rounded-xl bg-zinc-900/40 animate-pulse" />,
});
const MapLibre3D = dynamic(() => import('./MapLibre3DInner'), {
  ssr: false,
  loading: () => <div className="w-full h-[420px] rounded-xl bg-zinc-900/40 animate-pulse" />,
});

interface GeocodeResult { display_name: string; lat: string; lon: string; }

export default function GeoMapPanel({ theme }: { theme: 'dark' | 'light' }) {
  const isDark = theme === 'dark';

  
  const [modelManager] = useState(() => new OnnxModelManager());
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loadedModelName, setLoadedModelName] = useState('');
  const [snrDb, setSnrDb] = useState(-10);
  const [inferenceMode, setInferenceMode] = useState<'classical' | 'quantum'>('classical');
  const [useEnsemble, setUseEnsemble] = useState(true);
  const [stats, setStats] = useState<SessionStats>(() => freshStats());
  
  
  
  
  const statsRef = useRef<SessionStats>(freshStats());
  const totalInferenceTimeRef = useRef(0);

  useEffect(() => {
    let loadedFrom = 'feat31_model.onnx';
    fetch('/models/feat31_model.onnx')
      .then(res => {
        if (res.ok) return res.arrayBuffer();
        loadedFrom = 'radarsense.onnx';
        return fetch('/models/radarsense.onnx').then(r => r.ok ? r.arrayBuffer() : null);
      })
      .then(async buffer => {
        if (!buffer) return;
        try {
          const success = await modelManager.loadModelFromBuffer(buffer);
          if (success) { setIsModelLoaded(true); setLoadedModelName(loadedFrom); }
        } catch (err) {
          console.warn('Geo map: auto-load ONNX model failed:', err);
        }
      })
      .catch(() => {});
  }, [modelManager]);

  const updateStats = () => {
    const s = statsRef.current;
    s.accuracy = s.totalDetections > 0 ? (s.correctDetections / s.totalDetections) * 100 : 0;
    s.avgInferenceTimeMs = s.totalDetections > 0 ? totalInferenceTimeRef.current / s.totalDetections : 0;
    const classes: GeoTarget['cls'][] = ['bird', 'drone', 'aircraft', 'anomaly'];
    classes.forEach(gt => {
      const actualCount = Object.values(s.confusionMatrix[gt]).reduce((a, b) => a + b, 0);
      let predictedCount = 0;
      classes.forEach(c => { predictedCount += s.confusionMatrix[c][gt]; });
      const correctCount = s.confusionMatrix[gt][gt];
      s.classStats[gt].detections = actualCount;
      s.classStats[gt].correct = correctCount;
      s.classStats[gt].recall = actualCount > 0 ? (correctCount / actualCount) * 100 : 0;
      s.classStats[gt].precision = predictedCount > 0 ? (correctCount / predictedCount) * 100 : 0;
    });
    setStats({ ...s, confusionMatrix: { ...s.confusionMatrix }, classStats: { ...s.classStats } });
  };

  const resetStats = () => {
    totalInferenceTimeRef.current = 0;
    statsRef.current = freshStats();
    setStats(statsRef.current);
  };

  const runInference = async (t: GeoTarget) => {
    
    
    
    const { featureVector11, featureVector31 } = computeFeatureVectors({ ...t, type: t.cls }, snrDbRef.current);
    try {
      const result = await modelManager.predict(
        featureVector31, featureVector11, t.cls, snrDbRef.current,
        inferenceModeRef.current === 'quantum', useEnsembleRef.current,
      );
      t.lastPrediction = result;
      if (!result.isStub) {
        const s = statsRef.current;
        s.totalDetections += 1;
        totalInferenceTimeRef.current += result.inferenceTime;
        if (result.predictedClass === t.cls) s.correctDetections += 1;
        s.confusionMatrix[t.cls][result.predictedClass] += 1;
        updateStats();
      }
      setTargets([...targetsRef.current]);
      setSelected(prev => prev && prev.id === t.id ? { ...t } : prev);
    } catch (err) {
      console.error('Geo map inference failed:', err);
    }
  };

  
  const [originLat, setOriginLat] = useState(0);
  const [originLon, setOriginLon] = useState(0);
  const [running, setRunning] = useState(false);
  const [paused, setPaused] = useState(false);
  const [targets, setTargets] = useState<GeoTarget[]>([]);
  const targetsRef = useRef<GeoTarget[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');

  const [selected, setSelected] = useState<GeoTarget | null>(null);
  const [pickMode, setPickMode] = useState(false);
  const [search, setSearch] = useState('');
  const [searchResults, setSearchResults] = useState<GeocodeResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [locating, setLocating] = useState(false);
  const [locateError, setLocateError] = useState<string | null>(null);
  const [buildingStats, setBuildingStats] = useState<BuildingStats | null>(null);

  
  
  const snrDbRef = useRef(snrDb);
  const inferenceModeRef = useRef(inferenceMode);
  const useEnsembleRef = useRef(useEnsemble);
  const pausedRef = useRef(paused);
  useEffect(() => {
    snrDbRef.current = snrDb;
    inferenceModeRef.current = inferenceMode;
    useEnsembleRef.current = useEnsemble;
    pausedRef.current = paused;
  });

  const spawnScene = (n: number) => {
    targetsRef.current = spawnRealisticTargets(n);
    setTargets([...targetsRef.current]);
  };

  const start = () => {
    spawnScene(8);
    setSelected(null);
    setPaused(false);
    setRunning(true);
    intervalRef.current = setInterval(() => {
      if (pausedRef.current) return;
      
      
      
      
      
      
      
      
      targetsRef.current.forEach(t => { t.x += t.vx; t.y += t.vy; });
      setTargets([...targetsRef.current]);
      setSelected(prev => prev ? targetsRef.current.find(t => t.id === prev.id) ?? null : null);
      
      
      for (const t of targetsRef.current) void runInference(t);
    }, 500);
  };

  const stop = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    setRunning(false);
    setPaused(false);
  };

  useEffect(() => () => { if (intervalRef.current) clearInterval(intervalRef.current); }, []);

  const runSearch = async () => {
    if (!search.trim()) return;
    setSearching(true);
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(search)}&limit=5`,
        { headers: { 'Accept': 'application/json' } },
      );
      const data = await res.json();
      setSearchResults(data);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  const jumpTo = (lat: number, lon: number) => {
    setOriginLat(lat);
    setOriginLon(lon);
    setSearchResults([]);
    setSearch('');
    setSelected(null);
    setBuildingStats(null);
    if (running) spawnScene(8);
  };

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setLocateError('Geolocation is not available in this browser.');
      return;
    }
    setLocating(true);
    setLocateError(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        jumpTo(pos.coords.latitude, pos.coords.longitude);
      },
      (err) => {
        setLocating(false);
        setLocateError(
          err.code === err.PERMISSION_DENIED
            ? 'Location permission denied — search a city or pick on the map instead.'
            : `Could not get your location: ${err.message}`
        );
      },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 },
    );
  };

  const markers = targets.map(t => {
    const [lat, lon] = metersToLatLon(originLat, originLon, t.x, t.y);
    return { ...t, lat, lon };
  });

  const range = selected ? Math.hypot(selected.x, selected.y) : 0;
  const bearing = selected ? ((Math.atan2(selected.x, selected.y) * 180 / Math.PI) + 360) % 360 : 0;
  const speed = selected ? Math.hypot(selected.vx, selected.vy) : 0;
  const [selLat, selLon] = selected ? metersToLatLon(originLat, originLon, selected.x, selected.y) : [0, 0];
  const clearanceAboveTallest = selected && buildingStats ? selected.altitude - buildingStats.maxHeight : null;
  const selMatch = selected?.lastPrediction ? selected.lastPrediction.predictedClass === selected.cls : null;

  return (
    <div className={`border rounded-2xl p-5 backdrop-blur-md shadow-2xl flex flex-col gap-3 transition-colors duration-300 ${isDark ? 'bg-zinc-950/70 border-sky-950/40' : 'bg-white border-zinc-200'}`}>
      <div className={`flex justify-between items-center border-b pb-2 ${isDark ? 'border-zinc-900' : 'border-zinc-200'}`}>
        <span className={`font-mono text-sm font-semibold tracking-wider flex items-center gap-2 ${isDark ? 'text-zinc-300' : 'text-zinc-700'}`}>
          <MapPin className="w-4 h-4 text-sky-400" /> Geo Overlay
        </span>
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-1.5 border px-2 py-1 rounded-lg font-mono text-[9px] ${isDark ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-300 bg-zinc-50'}`}>
            <Cpu className={`w-3 h-3 ${isModelLoaded ? 'text-emerald-400' : 'text-zinc-500'}`} />
            <span className={isDark ? 'text-zinc-400' : 'text-zinc-600'}>
              {isModelLoaded ? `ONNX (${loadedModelName})` : 'Demo stub'}
            </span>
          </div>
          <div className={`flex items-center border rounded-lg overflow-hidden font-mono text-[10px] ${isDark ? 'border-zinc-800' : 'border-zinc-300'}`}>
            <button
              onClick={() => setViewMode('2d')}
              className={`px-2 py-1 cursor-pointer transition-all ${viewMode === '2d' ? 'bg-sky-500/20 text-sky-400 font-bold' : isDark ? 'text-zinc-500 hover:bg-zinc-800/50' : 'text-zinc-500 hover:bg-zinc-100'}`}
            >2D</button>
            <button
              onClick={() => setViewMode('3d')}
              className={`px-2 py-1 cursor-pointer transition-all ${viewMode === '3d' ? 'bg-sky-500/20 text-sky-400 font-bold' : isDark ? 'text-zinc-500 hover:bg-zinc-800/50' : 'text-zinc-500 hover:bg-zinc-100'}`}
            >3D</button>
          </div>
          {running && (
            <button
              onClick={() => setPaused(p => !p)}
              title={paused ? 'Resume' : 'Freeze'}
              className={`px-2.5 py-1.5 font-mono text-[11px] font-bold border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all ${isDark ? 'border-zinc-800 text-zinc-300 hover:bg-zinc-800/50' : 'border-zinc-300 hover:bg-zinc-100'}`}
            >
              {paused ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
            </button>
          )}
          <button
            onClick={() => { resetStats(); }}
            title="Reset session stats"
            className={`px-2.5 py-1.5 font-mono text-[11px] font-bold border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all ${isDark ? 'border-zinc-800 text-zinc-300 hover:bg-zinc-800/50' : 'border-zinc-300 hover:bg-zinc-100'}`}
          >
            <RefreshCw className="w-3 h-3" />
          </button>
          <button
            onClick={running ? stop : start}
            className={`px-3 py-1.5 font-mono text-[11px] font-bold border rounded-lg flex items-center gap-1.5 cursor-pointer transition-all ${running
              ? 'bg-red-500/20 text-red-400 border-red-500/40 hover:bg-red-500/30'
              : 'bg-sky-500/20 text-sky-400 border-sky-500/40 hover:bg-sky-500/30'
              }`}
          >
            {running ? <><Square className="w-3 h-3" /> Stop</> : <><Play className="w-3 h-3" /> Start Demo</>}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[9px] font-mono text-zinc-500 uppercase">SNR</span>
        {SNR_PRESETS.map(preset => (
          <button
            key={preset.name}
            onClick={() => setSnrDb(preset.db)}
            title={`${preset.desc} (${preset.db} dB)`}
            className={`px-2 py-1 font-mono text-[9px] border rounded transition-all cursor-pointer ${snrDb === preset.db
              ? 'bg-sky-500/20 text-sky-400 border-sky-500/40 font-bold'
              : isDark ? 'border-zinc-800 text-zinc-400 hover:bg-zinc-800/50' : 'border-zinc-300 text-zinc-500 hover:bg-zinc-100'
              }`}
          >
            {preset.name.toUpperCase()}
          </button>
        ))}
        <div className={`ml-auto flex items-center border rounded-lg overflow-hidden font-mono text-[10px] ${isDark ? 'border-zinc-800' : 'border-zinc-300'}`}>
          <button
            onClick={() => setInferenceMode('classical')}
            className={`px-2 py-1 cursor-pointer transition-all ${inferenceMode === 'classical' ? 'bg-emerald-500/20 text-emerald-400 font-bold' : isDark ? 'text-zinc-500 hover:bg-zinc-800/50' : 'text-zinc-500 hover:bg-zinc-100'}`}
          >CLASSICAL</button>
          <button
            onClick={() => setInferenceMode('quantum')}
            className={`px-2 py-1 cursor-pointer transition-all ${inferenceMode === 'quantum' ? 'bg-violet-500/20 text-violet-400 font-bold' : isDark ? 'text-zinc-500 hover:bg-zinc-800/50' : 'text-zinc-500 hover:bg-zinc-100'}`}
          >QUANTUM</button>
        </div>
        {inferenceMode === 'quantum' && (
          <label className="flex items-center gap-1 text-[9px] font-mono text-zinc-500 cursor-pointer">
            <input type="checkbox" checked={useEnsemble} onChange={e => setUseEnsemble(e.target.checked)} />
            ENSEMBLE
          </label>
        )}
      </div>

      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <button
            onClick={useMyLocation}
            disabled={locating}
            title="Use your browser's current location as the defended-asset origin"
            className={`px-2.5 py-1.5 rounded-lg border font-mono text-[11px] flex items-center gap-1.5 cursor-pointer transition-all disabled:opacity-50 ${isDark ? 'border-emerald-800/40 text-emerald-400 bg-emerald-950/20 hover:bg-emerald-950/40' : 'border-emerald-200 text-emerald-600 bg-emerald-50 hover:bg-emerald-100'}`}
          >
            <LocateFixed className="w-3 h-3" /> {locating ? 'Locating…' : 'Use My Location'}
          </button>
          <div className="relative flex-1">
            <Search className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && runSearch()}
              placeholder="…or search a city (e.g. Bengaluru)"
              className={`w-full pl-7 pr-2 py-1.5 rounded-lg border font-mono text-[11px] ${isDark ? 'bg-zinc-900 border-zinc-800 text-zinc-200 placeholder:text-zinc-600' : 'bg-white border-zinc-300 placeholder:text-zinc-400'}`}
            />
          </div>
          <button
            onClick={runSearch}
            disabled={searching}
            className={`px-2.5 py-1.5 rounded-lg border font-mono text-[11px] cursor-pointer disabled:opacity-50 ${isDark ? 'border-zinc-800 text-zinc-300 hover:bg-zinc-800/50' : 'border-zinc-300 hover:bg-zinc-100'}`}
          >
            {searching ? '…' : 'Search'}
          </button>
          <button
            onClick={() => setPickMode(p => !p)}
            title="Click anywhere on the map to set the defended-asset origin there"
            className={`px-2.5 py-1.5 rounded-lg border font-mono text-[11px] flex items-center gap-1 cursor-pointer transition-all ${pickMode
              ? 'bg-sky-500/20 text-sky-400 border-sky-500/40'
              : isDark ? 'border-zinc-800 text-zinc-300 hover:bg-zinc-800/50' : 'border-zinc-300 hover:bg-zinc-100'
              }`}
          >
            <Crosshair className="w-3 h-3" /> {pickMode ? 'Click map…' : 'Pick on map'}
          </button>
        </div>
        {locateError && <p className="text-[10px] font-mono text-amber-500">{locateError}</p>}
        {searchResults.length > 0 && (
          <div className={`rounded-lg border overflow-hidden ${isDark ? 'border-zinc-800 bg-zinc-900' : 'border-zinc-300 bg-white'}`}>
            {searchResults.map((r, i) => (
              <button
                key={i}
                onClick={() => jumpTo(parseFloat(r.lat), parseFloat(r.lon))}
                className={`w-full text-left px-2.5 py-1.5 text-[10.5px] font-mono border-b last:border-b-0 cursor-pointer ${isDark ? 'border-zinc-800 text-zinc-300 hover:bg-zinc-800' : 'border-zinc-200 hover:bg-zinc-50'}`}
              >
                {r.display_name}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 text-[10px] font-mono">
        <label className="flex items-center gap-1.5">
          <span className="text-zinc-500">Defended asset lat</span>
          <input type="number" step="0.0001" value={originLat}
            onChange={e => setOriginLat(parseFloat(e.target.value) || 0)}
            className={`w-24 px-1.5 py-0.5 rounded border ${isDark ? 'bg-zinc-900 border-zinc-800 text-zinc-200' : 'bg-white border-zinc-300'}`} />
        </label>
        <label className="flex items-center gap-1.5">
          <span className="text-zinc-500">lon</span>
          <input type="number" step="0.0001" value={originLon}
            onChange={e => setOriginLon(parseFloat(e.target.value) || 0)}
            className={`w-24 px-1.5 py-0.5 rounded border ${isDark ? 'bg-zinc-900 border-zinc-800 text-zinc-200' : 'bg-white border-zinc-300'}`} />
        </label>
        <span className="text-zinc-600 italic">defaults to 0,0 (Null Island)</span>
      </div>

      {viewMode === '2d' ? (
        <LeafletMap
          originLat={originLat} originLon={originLon} markers={markers} classColor={CLASS_COLOR} isDark={isDark}
          pickMode={pickMode}
          onPick={(lat, lon) => { jumpTo(lat, lon); setPickMode(false); }}
          onMarkerClick={(id) => setSelected(targetsRef.current.find(t => t.id === id) ?? null)}
          selectedId={selected?.id ?? null}
        />
      ) : (
        <MapLibre3D
          originLat={originLat} originLon={originLon} markers={markers} classColor={CLASS_COLOR} isDark={isDark}
          pickMode={pickMode}
          onPick={(lat, lon) => { jumpTo(lat, lon); setPickMode(false); }}
          onMarkerClick={(id) => setSelected(targetsRef.current.find(t => t.id === id) ?? null)}
          selectedId={selected?.id ?? null}
          onBuildingStats={setBuildingStats}
        />
      )}

      {viewMode === '3d' && buildingStats && (
        <div className={`flex items-center gap-2 text-[10px] font-mono px-2.5 py-1.5 rounded-lg border ${isDark ? 'bg-zinc-900/40 border-zinc-800 text-zinc-400' : 'bg-zinc-50 border-zinc-200 text-zinc-600'}`}>
          <Building2 className="w-3 h-3 text-sky-400" />
          {buildingStats.count > 0
            ? <span>
                {buildingStats.count} OSM buildings in view — tallest {buildingStats.maxHeight.toFixed(0)}m,
                avg {buildingStats.avgHeight.toFixed(0)}m
                {buildingStats.untagged > 0 && (
                  <> ({buildingStats.tagged} with real height data, {buildingStats.untagged} drawn at an
                  assumed {DEFAULT_BUILDING_HEIGHT_M}m default)</>
                )}
              </span>
            : <span>No buildings in view at this location/zoom — zoom in or move to a denser area</span>}
        </div>
      )}

      {stats.totalDetections > 0 && (
        <div className={`grid grid-cols-4 gap-2 text-[10px] font-mono`}>
          <Stat label="Detections" value={String(stats.totalDetections)} isDark={isDark} />
          <Stat label="Accuracy" value={`${stats.accuracy.toFixed(1)}%`} isDark={isDark} />
          <Stat label="Drone Recall" value={`${stats.classStats.drone.recall.toFixed(0)}%`} isDark={isDark} />
          <Stat label="Avg Latency" value={`${stats.avgInferenceTimeMs.toFixed(2)}ms`} isDark={isDark} />
        </div>
      )}

      {selected && (
        <div className={`border rounded-xl p-3 flex flex-col gap-1.5 transition-colors duration-300 ${isDark ? 'bg-zinc-900/50 border-zinc-800' : 'bg-zinc-50 border-zinc-200'}`}>
          <div className="flex justify-between items-center mb-1">
            <span className="font-mono text-[11px] font-bold uppercase" style={{ color: CLASS_COLOR[selected.cls] }}>
              Track {selected.id} &middot; {selected.subtype} ({selected.cls})
            </span>
            <button onClick={() => setSelected(null)} className="cursor-pointer text-zinc-500 hover:text-zinc-300">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {selected.lastPrediction && (
            <div className="grid grid-cols-2 gap-2 mb-1">
              <div className={`border rounded-lg p-1.5 text-center ${isDark ? 'bg-zinc-900/40 border-zinc-900' : 'bg-white border-zinc-200'}`}>
                <div className="text-[8px] text-zinc-500 uppercase">Ground Truth</div>
                <div className="font-mono text-[11px] font-bold uppercase" style={{ color: CLASS_COLOR[selected.cls] }}>{selected.cls}</div>
              </div>
              <div className={`border rounded-lg p-1.5 text-center flex flex-col items-center ${isDark ? 'bg-zinc-900/40 border-zinc-900' : 'bg-white border-zinc-200'}`}>
                <div className="text-[8px] text-zinc-500 uppercase flex items-center gap-1">
                  Predicted {selMatch === true ? <CheckCircle className="w-2.5 h-2.5 text-emerald-400" /> : selMatch === false ? <AlertTriangle className="w-2.5 h-2.5 text-red-400" /> : null}
                </div>
                <div className="font-mono text-[11px] font-bold uppercase" style={{ color: CLASS_COLOR[selected.lastPrediction.predictedClass] }}>
                  {selected.lastPrediction.predictedClass}
                  <span className="text-zinc-500 font-normal ml-1 text-[9px]">
                    {(selected.lastPrediction.confidences[selected.lastPrediction.predictedClass] * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
            <Stat label="Latitude" value={selLat.toFixed(6)} isDark={isDark} />
            <Stat label="Longitude" value={selLon.toFixed(6)} isDark={isDark} />
            <Stat label="Elevation (AGL)" value={`${selected.altitude.toFixed(0)} m`} isDark={isDark} />
            <Stat label="Range" value={`${range.toFixed(0)} m`} isDark={isDark} />
            <Stat label="Bearing" value={`${bearing.toFixed(0)}°`} isDark={isDark} />
            <Stat label="Speed" value={`${speed.toFixed(1)} m/s`} isDark={isDark} />
            {clearanceAboveTallest !== null && (
              <Stat
                label="Vs. tallest building"
                value={`${clearanceAboveTallest >= 0 ? '+' : ''}${clearanceAboveTallest.toFixed(0)} m`}
                isDark={isDark}
              />
            )}
          </div>
        </div>
      )}

      <p className={`text-[10px] font-sans leading-relaxed ${isDark ? 'text-zinc-500' : 'text-zinc-600'}`}>
        Targets are spawned with RadarSimulator.tsx&apos;s actual SUBTYPE_PRESETS and classified
        through the same OnnxModelManager (real ONNX inference, classical/quantum/ensemble modes,
        computeFeatureVectors() for feature synthesis) &mdash; not a display-only mock. 2D uses free
        OpenStreetMap tiles (Leaflet); 3D uses MapLibre GL + OpenFreeMap&apos;s free vector tiles,
        with real building heights queried live from OpenStreetMap data (shown above) &mdash; both
        free, no API key. Nominatim geocoding for search, also free.
      </p>
    </div>
  );
}

function Stat({ label, value, isDark }: { label: string; value: string; isDark: boolean }) {
  return (
    <div className={`border rounded-lg px-2 py-1 text-center ${isDark ? 'bg-zinc-900/40 border-zinc-900' : 'bg-white border-zinc-200'}`}>
      <div className="text-[8px] text-zinc-500 uppercase">{label}</div>
      <div className={isDark ? 'text-zinc-200' : 'text-zinc-700'}>{value}</div>
    </div>
  );
}
