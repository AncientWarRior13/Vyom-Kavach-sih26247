

















export type ThreatLevel = 'CLEAR' | 'MONITOR' | 'ELEVATED' | 'CRITICAL' | 'UNCERTAIN' | 'PENDING';

export interface ThreatFactors {
  classSeverity: number;   
  proximity: number;       
  closing: number;         
  timeToIntercept: number; 
  lowAltitude: number;     
  confidence: number;      
}

export interface ThreatAssessment {
  level: ThreatLevel;
  score: number;           
  color: string;           
  rationale: string;       
  priority: number;        
  evasion: boolean;        
  uncertain: boolean;      
  factors: ThreatFactors;
  
  rangeM: number;
  closingMps: number;      
  ttiSec: number | null;   
}

export interface ThreatInput {
  x: number; y: number; vx: number; vy: number;
  altitude: number; baseUdbw: number;
  lastPrediction: { predictedClass: 'bird' | 'drone' | 'aircraft' | 'anomaly'; confidences: Record<string, number> } | null;
}

export interface ThreatConfig {
  keepOutRangeM: number;    
  watchRangeM: number;      
  closingRefMps: number;    
  ttiRefSec: number;        
  lowAltRefM: number;       
  mdSuppressionBwHz: number;
  classSeverity: Record<'drone' | 'anomaly' | 'aircraft' | 'bird', number>;
  weights: { proximity: number; closing: number; tti: number; altitude: number };
  levelThresholds: { critical: number; elevated: number; monitor: number };
  lowConfidenceDrone: number; 
}



export const DEFAULT_THREAT_CONFIG: ThreatConfig = {
  keepOutRangeM: 300,
  watchRangeM: 2000,
  closingRefMps: 20,
  ttiRefSec: 120,
  lowAltRefM: 500,
  mdSuppressionBwHz: 120,
  classSeverity: { drone: 1.0, anomaly: 0.5, aircraft: 0.35, bird: 0.05 },
  weights: { proximity: 0.32, closing: 0.30, tti: 0.23, altitude: 0.15 },
  levelThresholds: { critical: 70, elevated: 45, monitor: 22 },
  lowConfidenceDrone: 0.55,
};

const LEVEL_COLORS: Record<ThreatLevel, string> = {
  CRITICAL: '#fb4d5d',
  ELEVATED: '#f59e0b',
  UNCERTAIN: '#f7b84b',
  MONITOR: '#54a5ff',
  CLEAR: '#10b981',
  PENDING: '#a1a1aa',
};

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export function threatColor(level: ThreatLevel): string {
  return LEVEL_COLORS[level] ?? LEVEL_COLORS.PENDING;
}

export function assessThreat(t: ThreatInput, cfg: ThreatConfig = DEFAULT_THREAT_CONFIG): ThreatAssessment {
  const rangeM = Math.hypot(t.x, t.y);
  
  const closingMps = rangeM > 1e-6 ? -(t.x * t.vx + t.y * t.vy) / rangeM : 0;
  const ttiSec = closingMps > 0.1 ? rangeM / closingMps : null;

  
  if (!t.lastPrediction) {
    return {
      level: 'PENDING', score: 0, color: LEVEL_COLORS.PENDING,
      rationale: 'awaiting sweep', priority: -1, evasion: false, uncertain: false,
      factors: { classSeverity: 0, proximity: 0, closing: 0, timeToIntercept: 0, lowAltitude: 0, confidence: 0 },
      rangeM, closingMps, ttiSec,
    };
  }

  const cls = t.lastPrediction.predictedClass;
  const confidence = clamp01(t.lastPrediction.confidences?.[cls] ?? 0);
  const classSeverity = cfg.classSeverity[cls] ?? 0;

  
  const proximity = rangeM <= cfg.keepOutRangeM ? 1
    : clamp01((cfg.watchRangeM - rangeM) / (cfg.watchRangeM - cfg.keepOutRangeM));
  const closing = clamp01(closingMps / cfg.closingRefMps);
  const timeToIntercept = ttiSec === null ? 0 : clamp01((cfg.ttiRefSec - ttiSec) / cfg.ttiRefSec);
  const lowAltitude = clamp01((cfg.lowAltRefM - t.altitude) / cfg.lowAltRefM);

  const kinematic =
    cfg.weights.proximity * proximity +
    cfg.weights.closing * closing +
    cfg.weights.tti * timeToIntercept +
    cfg.weights.altitude * lowAltitude;

  
  let raw = classSeverity * (0.30 + 0.70 * kinematic);

  
  
  
  const evasion = cls === 'drone' && t.baseUdbw < cfg.mdSuppressionBwHz;
  if (evasion) raw = Math.min(1, raw + 0.15);

  
  
  const confBlend = 0.5 + 0.5 * confidence;
  let score = Math.round(100 * clamp01(raw) * confBlend);

  
  
  const uncertain = cls === 'drone' && confidence < cfg.lowConfidenceDrone;
  if (uncertain) score = Math.max(score, cfg.levelThresholds.elevated);

  let level: ThreatLevel;
  if (uncertain && score < cfg.levelThresholds.critical) level = 'UNCERTAIN';
  else if (score >= cfg.levelThresholds.critical) level = 'CRITICAL';
  else if (score >= cfg.levelThresholds.elevated) level = 'ELEVATED';
  else if (score >= cfg.levelThresholds.monitor) level = 'MONITOR';
  else level = 'CLEAR';

  
  const priority = level === 'CRITICAL' ? 300 + score
    : level === 'UNCERTAIN' ? 200 + score
      : score;

  return {
    level, score, color: LEVEL_COLORS[level],
    rationale: buildRationale(cls, rangeM, closingMps, ttiSec, t.altitude, evasion, uncertain),
    priority, evasion, uncertain,
    factors: { classSeverity, proximity, closing, timeToIntercept, lowAltitude, confidence },
    rangeM, closingMps, ttiSec,
  };
}

function buildRationale(
  cls: string, rangeM: number, closingMps: number, ttiSec: number | null,
  altitude: number, evasion: boolean, uncertain: boolean,
): string {
  const parts: string[] = [cls.toUpperCase(), `${Math.round(rangeM / 5) * 5} m`];
  if (closingMps > 0.1) {
    parts.push(`closing ${closingMps.toFixed(1)} m/s`);
    if (ttiSec !== null && ttiSec < 600) parts.push(`TTI ${Math.round(ttiSec)} s`);
  } else {
    parts.push('receding');
  }
  if (altitude < 120) parts.push(`low-alt ${Math.round(altitude)} m`);
  let s = parts.join(' · ');
  if (evasion) s += ' · µD suppressed → evasion suspected';
  if (uncertain) s += ' · UNCERTAIN — investigate';
  return s;
}
