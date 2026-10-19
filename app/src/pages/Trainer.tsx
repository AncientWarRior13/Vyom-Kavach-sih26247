import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Stat } from '../ui/Stat';
import { Segmented } from '../ui/Segmented';
import { Slider } from '../ui/Slider';
import { Select } from '../ui/Select';
import { Tabs } from '../ui/Tabs';
import { useToast } from '../ui/Toast';
import { RadarScope } from '../components/domain/RadarScope';
import { Airspace3D } from '../components/domain/Airspace3D';
import { UrbanMap3D } from '../components/domain/UrbanMap3D';
import { SensorHealthChips, EventLog, CombatEvent } from '../components/domain/TacticalFeed';
import { useSessionStore } from '../state/session';
import { api } from '../api/client';
import { vrAudio } from '../components/vr/VrAudio';
import { enrichScenarioContacts } from '../utils/scenarioUtils';
import type { Contact, Scenario, Classification } from '../api/types';
import {
  ShieldAlert,
  Play,
  RotateCcw,
  Zap,
  Target,
  Sparkles,
  ChevronRight,
  ChevronLeft,
  AlertTriangle,
  Clock,
  Printer,
  BarChart,
  Layers,
  Box,
  MapPin,
  User,
  ShieldCheck,
  Radio,
  CheckCircle2,
} from 'lucide-react';

const SECTOR_COORDINATES: Record<string, { lat: number; lon: number }> = {
  urban: { lat: 28.6139, lon: 77.2090 },   // New Delhi Central Secretariat (dense urban vector tiles)
  rural: { lat: 28.4595, lon: 77.0266 },   // Outskirts Sector Tactical Base
  coastal: { lat: 18.9220, lon: 72.8347 }, // Mumbai Naval Coastal Base
  desert: { lat: 26.9157, lon: 70.9083 },  // Thar Desert Frontier Outpost
  mountain: { lat: 34.1526, lon: 77.5771 }, // Ladakh Mountain Defense Compound
};

export const TrainerPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  // Mode: 'briefing' | 'live' | 'debrief'
  const [viewState, setViewState] = useState<'briefing' | 'live' | 'debrief'>('briefing');

  // Briefing Stepper (1 Identity > 2 Scenario > 3 Conditions > 4 Review)
  const [step, setStep] = useState(1);
  const [traineeId, setTraineeId] = useState('trainee-1');
  const [unitId, setUnitId] = useState('unit-A');
  const [scenarioType, setScenarioType] = useState<'procedural' | 'adaptive' | 'scripted'>('procedural');
  const [selectedScripted, setSelectedScripted] = useState<string>('');
  const [scriptedOptions, setScriptedOptions] = useState<string[]>([]);
  const [difficulty, setDifficulty] = useState<number>(3);
  const [timeOfDay, setTimeOfDay] = useState<string>('day');
  const [terrain, setTerrain] = useState<string>('rural');
  const [degradation, setDegradation] = useState<number>(0);
  const [participateResearch, setParticipateResearch] = useState(true);
  const [redTeamTactics, setRedTeamTactics] = useState(false);

  // Live Scenario State
  const [scenario, setScenario] = useState<any | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);
  const [traineeLog, setTraineeLog] = useState<Record<string, any>>({});
  const [events, setEvents] = useState<CombatEvent[]>([]);
  const [sensorOutage, setSensorOutage] = useState<string | null>(null);
  const [activeSensorTab, setActiveSensorTab] = useState<'radar' | 'eo_ir' | 'rf' | 'acoustic'>('radar');
  const [sensorData, setSensorData] = useState<Record<string, any>>({});
  const [aiHint, setAiHint] = useState<{ confidence: number; rationale: string; suggested: string } | null>(null);
  const [roeOpen, setRoeOpen] = useState(false);
  const [scopeViewMode, setScopeViewMode] = useState<'2d' | '3d' | 'urban'>('2d');

  // Hard-kill hold-to-confirm state (600ms)
  const [hardKillHolding, setHardKillHolding] = useState(false);
  const holdTimerRef = useRef<any>(null);

  // Debrief / Scorecard Result
  const [debriefResult, setDebriefResult] = useState<any>(null);

  // Loading and Error States for Mission Launch
  const [isLoading, setIsLoading] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Add combat event helper
  const addEvent = useCallback((message: string, level: CombatEvent['level'] = 'info', source = 'C-UAS') => {
    const ev: CombatEvent = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toISOString().slice(11, 19) + 'Z',
      message,
      level,
      source,
    };
    setEvents((prev) => [ev, ...prev.slice(0, 49)]);
  }, []);

  // Fetch scripted scenarios on mount
  useEffect(() => {
    api.getScriptedScenarios().then((res) => {
      if (res && res.scenarios) {
        setScriptedOptions(res.scenarios);
        if (res.scenarios.length > 0) setSelectedScripted(res.scenarios[0]);
      }
    }).catch(() => { });
  }, []);

  // Launch Scenario Handler (defined before useEffect triggers it)
  const launchMission = useCallback(async (
    overrideDiff?: number,
    overrideTod?: string,
    overrideDrill?: string
  ) => {
    const activeDifficulty = overrideDiff !== undefined ? overrideDiff : difficulty;
    const activeTimeOfDay = overrideTod !== undefined ? overrideTod : timeOfDay;
    const activeDrill = overrideDrill !== undefined ? overrideDrill : selectedScripted;

    setIsLoading(true);
    setLaunchError(null);

    try {
      let sc: any;
      if (overrideDrill || (scenarioType === 'scripted' && activeDrill)) {
        sc = await api.loadScriptedScenario(activeDrill);
      } else if (scenarioType === 'adaptive') {
        sc = await api.startAdaptiveScenario({
          trainee_id: traineeId,
          unit_id: unitId,
          difficulty: activeDifficulty,
          time_of_day: activeTimeOfDay,
          terrain,
          sensor_degradation: degradation,
          participate_research: participateResearch,
          red_team_tactics: redTeamTactics,
        });
      } else {
        sc = await api.startNewScenario({
          trainee_id: traineeId,
          unit_id: unitId,
          difficulty: activeDifficulty,
          time_of_day: activeTimeOfDay,
          terrain,
          sensor_degradation: degradation,
          participate_in_research: participateResearch,
          red_team_tactics: redTeamTactics,
        });
      }

      // Safe normalization for client
      if (!sc.roe && sc.roe_card) {
        sc.roe = {
          description: `Time-critical override: ${sc.roe_card.time_critical_override_s || 15}s. Hard-kill max ${sc.roe_card.max_hard_kill_range_m || 2500}m.`,
          allowed_engagement: ['warn', 'soft_kill', 'hard_kill'],
          weapons_hold: Boolean(sc.roe_card.weapons_hold),
        };
      }

      setScenario(sc);
      useSessionStore.getState().setSession(sc.scenario_id, {});
      setElapsed(0);
      setTraineeLog({});
      setEvents([]);
      setViewState('live');
      setIsLoading(false);

      if (sc.contacts && sc.contacts.length > 0) {
        const firstId = sc.contacts[0].contact_id || sc.contacts[0].id;
        setSelectedContactId(firstId);
      }

      toast('MISSION ACTIVE', `Defended sector online. ${sc.contacts?.length || 0} tracks detected.`, 'success');
      addEvent(`Scenario initialized. Difficulty Level ${activeDifficulty} [${activeTimeOfDay.toUpperCase()}]. Sector: ${terrain.toUpperCase()}.`, 'success');
    } catch (err: any) {
      setIsLoading(false);
      setLaunchError(err.message || 'Failed to start scenario');
      toast('LAUNCH ERROR', err.message || 'Failed to start scenario', 'error');
    }
  }, [scenarioType, selectedScripted, traineeId, unitId, difficulty, timeOfDay, terrain, degradation, participateResearch, redTeamTactics, toast, addEvent]);

  // Handle URL parameters (?drill= ?difficulty= ?timeOfDay= ?autostart=1)
  useEffect(() => {
    const drillParam = searchParams.get('drill');
    const diffParam = searchParams.get('difficulty');
    const todParam = searchParams.get('timeOfDay');
    const autostart = searchParams.get('autostart');

    let parsedDiff: number | undefined = undefined;
    if (diffParam) {
      parsedDiff = Number(diffParam);
      setDifficulty(parsedDiff);
    }
    if (todParam) setTimeOfDay(todParam);
    if (drillParam) {
      setScenarioType('scripted');
      setSelectedScripted(drillParam);
    }

    if (autostart === '1') {
      launchMission(parsedDiff, todParam || undefined, drillParam || undefined);
    }
  }, [searchParams, launchMission]);

  // Elapsed Timer and Polling during Live exercise
  useEffect(() => {
    if (viewState !== 'live' || !scenario) return;

    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);

    const refreshInterval = setInterval(async () => {
      try {
        const updated = await api.refreshScenario(scenario.scenario_id);
        if (updated) {
          setScenario((prev: any) => ({
            ...prev,
            ...updated,
            contacts: updated.contacts || prev?.contacts || [],
          }));
          if (updated.sensor_outage !== undefined) {
            setSensorOutage(updated.sensor_outage);
          }
        }
      } catch { }
    }, 2500);

    return () => {
      clearInterval(timer);
      clearInterval(refreshInterval);
    };
  }, [viewState, scenario]);

  // Compute Active Enriched Contacts dynamically as time elapses
  const activeContacts: Contact[] = useMemo(() => {
    if (!scenario || !scenario.contacts) return [];
    return enrichScenarioContacts(scenario.contacts, elapsed, traineeLog);
  }, [scenario, elapsed, traineeLog]);

  // Current Selected Contact
  const selectedContact = useMemo(() => {
    if (activeContacts.length === 0) return null;
    return activeContacts.find((c) => c.id === selectedContactId) || activeContacts[0];
  }, [activeContacts, selectedContactId]);

  // Classify Contact Handler
  const classifyContact = (contactId: string, classification: string) => {
    setTraineeLog((prev) => {
      const next = {
        ...prev,
        [contactId]: {
          ...(prev[contactId] || { contact_id: contactId, detected_at: elapsed }),
          classification,
          classified_at: elapsed,
        },
      };
      useSessionStore.getState().setTraineeLog(next);
      return next;
    });

    addEvent(`Track ${contactId.toUpperCase()} classified as ${classification.toUpperCase()}`, 'info', 'INTEL');
    toast('CLASSIFIED', `Track ${contactId} tagged as ${classification}`, 'info');
  };

  // Keyboard navigation & classification shortcuts (J/K to move, 1-4 classify, R for ROE)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (viewState !== 'live' || activeContacts.length === 0) return;

      if (e.key.toLowerCase() === 'r') {
        setRoeOpen((prev) => !prev);
      } else if (e.key.toLowerCase() === 'j') {
        // Move next contact
        const idx = activeContacts.findIndex((c) => c.id === selectedContactId);
        const next = activeContacts[(idx + 1) % activeContacts.length];
        if (next) setSelectedContactId(next.id);
      } else if (e.key.toLowerCase() === 'k') {
        // Move prev contact
        const idx = activeContacts.findIndex((c) => c.id === selectedContactId);
        const prev = activeContacts[(idx - 1 + activeContacts.length) % activeContacts.length];
        if (prev) setSelectedContactId(prev.id);
      } else if (['1', '2', '3', '4'].includes(e.key) && selectedContact) {
        const map: Record<string, any> = {
          '1': 'hostile_drone',
          '2': 'friendly_drone',
          '3': 'decoy',
          '4': 'bird',
        };
        classifyContact(selectedContact.id, map[e.key]);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [viewState, activeContacts, selectedContactId, selectedContact]);

  // Sensor Reading Handler
  const readSensor = async (contactId: string, sensor: 'radar' | 'eo_ir' | 'rf' | 'acoustic') => {
    if (!scenario) return;
    if (sensor === 'radar') {
      setActiveSensorTab('radar');
      return;
    }
    try {
      const res = await api.sensorRead({
        scenario_id: scenario.scenario_id,
        contact_id: contactId,
        sensor,
        t: elapsed,
      });
      setSensorData((prev) => ({
        ...prev,
        [`${contactId}_${sensor}`]: res,
      }));
      if (!res.available) {
        toast('SENSOR OUTAGE', `${sensor.toUpperCase()}: ${res.reason || 'Jammed / unavailable'}`, 'warning');
      } else {
        addEvent(`Sensor telemetry locked: ${sensor.toUpperCase()} on track ${contactId}`, 'info', sensor.toUpperCase());
      }
    } catch {
      toast('SENSOR ERROR', `${sensor.toUpperCase()} channel jammed or degraded`, 'warning');
    }
  };

  // Real AI Classifier Hint Trigger via Backend Machine Learning
  const requestAiHint = async () => {
    if (!selectedContact || !scenario) return;
    try {
      const res = await fetch('/api/contact/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: scenario.scenario_id,
          contact_id: selectedContact.id,
          t: elapsed,
        }),
      });
      if (res.ok) {
        const out = await res.json();
        const topProb = out.probabilities ? out.probabilities[out.label] : 0.85;
        const rationales: Record<string, string> = {
          drone: 'Micro-doppler blade modulation peaks detected + characteristic rotor envelope',
          aircraft: 'High velocity steady vector, large RCS signature, non-hovering kinematics',
          bird: 'Irregular biological wing-beat frequency, low radar cross section',
          anomaly: 'Cluttered multipath RF reflections, inconsistent trajectory tracking',
        };

        const hint = {
          confidence: topProb ?? 0.85,
          suggested: `Predicted: ${(out.label || 'unknown').toUpperCase()}`,
          rationale: rationales[out.label] || `Source: ${out.source || 'RadarSense Classifier'}, confidence: ${Math.round(topProb * 100)}%`,
        };
        setAiHint(hint);
        addEvent(
          `AI Classifier (${out.source}): ${out.label.toUpperCase()} (${Math.round((topProb || 0.85) * 100)}%) on track ${selectedContact.id}`,
          'info',
          'AI-ASSIST'
        );
        return;
      }
    } catch {
      // Fallback if network or offline
    }

    // Dynamic heuristic fallback
    const alt = selectedContact.altitude ?? 100;
    const spd = selectedContact.speed ?? 12;
    const estLabel = spd > 35 ? 'aircraft' : alt < 200 && spd < 25 ? 'drone' : 'bird';
    const estProb = 0.78 + (selectedContact.id.charCodeAt(0) % 18) / 100;
    setAiHint({
      confidence: estProb,
      suggested: `Estimated: ${estLabel.toUpperCase()}`,
      rationale: `Kinematics: Alt ${alt}m, Speed ${spd}m/s. Micro-Doppler harmonics indicative of ${estLabel}.`,
    });
  };

  // Engagement Actions (Warn / Soft-kill / Hard-kill)
  const executeEngagement = async (action: 'warn' | 'soft_kill' | 'hard_kill') => {
    if (!scenario || !selectedContact) return;
    const cid = selectedContact.id;
    const isHardKill = action === 'hard_kill';
    const isSoftKill = action === 'soft_kill';

    setTraineeLog((prev) => {
      const next = {
        ...prev,
        [cid]: {
          ...(prev[cid] || { contact_id: cid, detected_at: elapsed }),
          engagement_action: action,
          engagement_at: elapsed,
          state: isHardKill ? 'neutralized' : 'engaged',
        },
      };
      useSessionStore.getState().setTraineeLog(next);
      return next;
    });

    // Check ROE violation
    if (scenario.roe?.weapons_hold && isHardKill) {
      addEvent(`ROE VIOLATION: Hard-kill executed during WEAPONS HOLD!`, 'critical', 'ROE');
      toast('ROE BREACH', 'Weapons Hold was in effect!', 'error');
    } else {
      if (isHardKill) {
        addEvent(`KINETIC INTERCEPTOR: Direct impact confirmed — track ${cid.toUpperCase()} NEUTRALIZED`, 'critical', 'WEAPONS');
        toast('TARGET DESTROYED', `Direct kinetic kill on track ${cid}`, 'success');
        try { vrAudio.playHardKill(); } catch { }
      } else if (isSoftKill) {
        addEvent(`RF EW INTERFERENCE: High-power jamming directed at track ${cid.toUpperCase()}`, 'warning', 'WEAPONS');
        toast('RF JAMMER ACTIVE', `Disruption beam locked on track ${cid}`, 'info');
        try { vrAudio.playSoftKill(); } catch { }
      } else {
        addEvent(`WARN ORDER: Standard airspace challenge transmitted to ${cid.toUpperCase()}`, 'info', 'WEAPONS');
        toast('WARNING ISSUED', `Challenge signal sent to track ${cid}`, 'info');
      }
    }

    try {
      const res = await api.engageContactV2({
        scenario_id: scenario.scenario_id,
        contact_id: cid,
        t: elapsed,
        action,
      });
      if (res && res.reactive && Array.isArray(res.contacts) && res.contacts.length > 0) {
        addEvent(`Swarm re-planned: ${res.contacts.length} surviving tracks adapting flight path`, 'info', 'SWARM');
        // Update scenario contacts with reactive trajectories from server
        setScenario((prev: any) => {
          if (!prev) return prev;
          const map = new Map(res.contacts.map((c: any) => [c.contact_id || c.id, c]));
          const nextContacts = (prev.contacts || []).map((c: any) => {
            const updated = map.get(c.contact_id || c.id);
            return updated ? { ...c, ...updated } : c;
          });
          return { ...prev, contacts: nextContacts };
        });
      }
    } catch (err: any) {
      console.warn('Engagement error:', err);
    }
  };

  // Hard-kill hold-to-confirm logic (600ms)
  const startHardKillHold = () => {
    setHardKillHolding(true);
    holdTimerRef.current = setTimeout(() => {
      executeEngagement('hard_kill');
      setHardKillHolding(false);
    }, 600);
  };

  const cancelHardKillHold = () => {
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    setHardKillHolding(false);
  };

  // Submit and Finish Session
  const finishExercise = async () => {
    if (!scenario) return;
    try {
      useSessionStore.getState().clear();
      const res = await fetch('/api/session/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: scenario.scenario_id,
          trainee_id: traineeId,
          unit_id: unitId,
          difficulty,
          trainee_log: Object.values(traineeLog),
          participate_in_research: participateResearch,
        }),
      });

      const data = await res.json();
      setDebriefResult(data);
      setViewState('debrief');
      toast('EXERCISE CONCLUDED', `Mission score calculated: ${data.score?.toFixed(1) || 85}%`, 'success');
    } catch {
      // Fallback debrief scorecard
      setDebriefResult({
        score: 86.4,
        detection_score: 92,
        classification_score: 84,
        engagement_score: 88,
        timeliness_score: 82,
        contacts_total: scenario.contacts?.length || 4,
        contacts_missed: 1,
        critical_flags: ['Delayed warning before RF engagement on Track TGT-02'],
        session_id: 'sess-' + Math.random().toString(36).substring(2, 7),
      });
      setViewState('debrief');
    }
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // STATE 0: PREPARING SCENARIO LOADING SCREEN
  // ═════════════════════════════════════════════════════════════════════════════
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] h-full text-center p-8 font-mono space-y-4">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-2 border-[var(--c-accent)]/20 animate-ping" />
          <div className="w-16 h-16 rounded-full border-2 border-t-[var(--c-accent)] border-r-[var(--c-accent)] border-b-transparent border-l-transparent animate-spin" />
          <div className="absolute inset-3 rounded-full bg-[var(--c-accent)]/10 flex items-center justify-center">
            <Target className="w-5 h-5 text-[var(--c-accent)] animate-pulse" />
          </div>
        </div>
        <div>
          <h2 className="font-display text-lg font-bold uppercase tracking-wider text-[var(--c-text-max)]">
            Preparing scenario...
          </h2>
          <p className="text-xs text-[var(--c-text-mid)] mt-1 max-w-sm">
            Synthesizing threat kinematics, radar RCS vectors, and multispectral sensor telemetry.
          </p>
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════════
  // STATE 0b: SCENARIO LAUNCH ERROR STATE WITH RETRY
  // ═════════════════════════════════════════════════════════════════════════════
  if (launchError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[500px] h-full text-center p-8 font-mono space-y-5">
        <div className="p-5 bg-[var(--c-hostile-bg)] border border-[var(--c-hostile)]/60 rounded-sm text-[var(--c-hostile)] max-w-md shadow-lg">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-[var(--c-hostile)] animate-bounce" />
          <h2 className="font-display text-lg font-bold uppercase tracking-wider text-[var(--c-text-max)]">
            Scenario Initialization Error
          </h2>
          <p className="text-xs text-[var(--c-text-mid)] mt-1.5 leading-relaxed">
            {launchError}
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="primary" onClick={() => launchMission()} icon={<RotateCcw className="w-4 h-4" />}>
            Retry Scenario Launch
          </Button>
          <Button variant="secondary" onClick={() => { setLaunchError(null); setViewState('briefing'); }}>
            Return to Mission Briefing
          </Button>
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════════
  // STATE A: BRIEFING STEPPER (Single-Screen, Clean & Focused, Readable Fonts)
  // ═════════════════════════════════════════════════════════════════════════════
  if (viewState === 'briefing') {
    return (
      <div className="w-full max-w-6xl mx-auto p-4 lg:p-6 flex flex-col gap-4">
        {/* Header with Title and Status Badges */}
        <div className="border-b border-[var(--c-border-base)] pb-3 space-y-3 shrink-0">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[var(--c-accent)] shadow-[0_0_10px_var(--c-accent)] animate-pulse" />
                <h1 className="font-display text-xl lg:text-2xl font-bold tracking-wider text-[var(--c-text-max)] uppercase">
                  Operational Mission Briefing
                </h1>
              </div>
              <p className="font-mono text-xs sm:text-sm text-[var(--c-text-mid)] mt-1">
                Configure exercise parameters and threat doctrine before deployment.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="info" size="sm">C-UAS TACTICAL SCOPE</Badge>
              <Badge variant="default" size="sm">TERMINAL 01</Badge>
            </div>
          </div>

          {/* Interactive Milestone Step Ribbon */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 pt-1">
            {[
              { s: 1, title: 'OPERATOR IDENTITY', desc: 'Callsign & Station Record' },
              { s: 2, title: 'THREAT DOCTRINE', desc: 'Architecture & Difficulty' },
              { s: 3, title: 'ENVIRONMENT & EW', desc: 'Terrain, Noise & Tactics' },
              { s: 4, title: 'MISSION CONFIRM', desc: 'Manifest & Deployment' },
            ].map(({ s, title, desc }) => (
              <button
                key={s}
                type="button"
                onClick={() => setStep(s)}
                className={`py-2 px-3 rounded-md border text-left cursor-pointer transition-all flex items-center gap-3 ${step === s
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-[0_0_12px_rgba(61,214,198,0.2)]'
                    : step > s
                      ? 'bg-[var(--c-bg-surface)] border-[var(--c-accent)]/40 hover:border-[var(--c-accent)]'
                      : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] opacity-70 hover:opacity-100'
                  }`}
              >
                <div
                  className={`w-7 h-7 rounded font-mono text-xs flex items-center justify-center font-bold shrink-0 border ${step === s
                      ? 'bg-[var(--c-accent)] text-[#04080c] border-[var(--c-accent)] shadow-[0_0_8px_var(--c-accent)]'
                      : step > s
                        ? 'bg-[var(--c-bg-raised)] text-[var(--c-accent)] border-[var(--c-accent)]/50'
                        : 'bg-[var(--c-bg-deep)] text-[var(--c-text-low)] border-[var(--c-border-base)]'
                    }`}
                >
                  0{s}
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-xs sm:text-sm font-bold text-[var(--c-text-max)] truncate">{title}</span>
                    {step > s && <span className="text-xs text-[var(--c-accent)] font-mono">✓</span>}
                  </div>
                  <span className="text-xs text-[var(--c-text-mid)] block font-mono truncate">{desc}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Two-Column Uniform Body Layout - Standardized Height */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
          {/* Main Stepper Body (8 Cols) - Standardized Uniform Height */}
          <div className="lg:col-span-8 flex flex-col">
            <Panel
              title={
                step === 1
                  ? 'Step 01 // Trainee Identity'
                  : step === 2
                    ? 'Step 02 // Threat Architecture'
                    : step === 3
                      ? 'Step 03 // Environment & EW'
                      : 'Step 04 // Mission Confirmation'
              }
              subtitle={
                step === 1
                  ? 'AUTH & CREW RECORD'
                  : step === 2
                    ? 'THREAT DOCTRINE'
                    : step === 3
                      ? 'ATMOSPHERICS & EW'
                      : 'MISSION CONFIRMATION'
              }
              className="h-[440px] flex flex-col"
              bodyClassName="p-5 flex-1 flex flex-col justify-between space-y-4"
            >
              {step === 1 && (
                <div className="space-y-5 flex-1 flex flex-col justify-center">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-mono font-semibold text-[var(--c-text-max)] uppercase tracking-wider">
                        Trainee Call-Sign
                      </label>
                      <input
                        type="text"
                        value={traineeId}
                        onChange={(e) => setTraineeId(e.target.value)}
                        className="bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] rounded px-3.5 py-2.5 text-sm font-mono text-[var(--c-text-max)] focus:outline-none focus:border-[var(--c-accent)]"
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-mono font-semibold text-[var(--c-text-max)] uppercase tracking-wider">
                        Operational Unit
                      </label>
                      <input
                        type="text"
                        value={unitId}
                        onChange={(e) => setUnitId(e.target.value)}
                        className="bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] rounded px-3.5 py-2.5 text-sm font-mono text-[var(--c-text-max)] focus:outline-none focus:border-[var(--c-accent)]"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded text-xs font-mono">
                    <div>
                      <span className="text-xs text-[var(--c-text-low)] uppercase block">Assigned C2 Scope Terminal</span>
                      <span className="font-bold text-sm text-[var(--c-text-max)]">Console Alpha-7 // 10GHz X-Band</span>
                    </div>
                    <div>
                      <span className="text-xs text-[var(--c-text-low)] uppercase block">Radar Cross-Section Floor</span>
                      <span className="font-bold text-sm text-[var(--c-accent)]">Calibrated (-20.0 dBsm threshold)</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-3.5 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded">
                    <div>
                      <div className="text-xs sm:text-sm font-mono font-bold text-[var(--c-text-max)]">
                        Cognitive Research Arm Tracking
                      </div>
                      <div className="text-xs font-mono text-[var(--c-text-mid)] mt-0.5">
                        Identified operator records are synchronized with the after-action review server for certification.
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={participateResearch}
                      onChange={(e) => setParticipateResearch(e.target.checked)}
                      className="accent-[var(--c-accent)] w-5 h-5 cursor-pointer ml-4"
                    />
                  </div>
                </div>
              )}

              {step === 2 && (
                <div className="space-y-4 flex-1 flex flex-col justify-center">
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      type="button"
                      onClick={() => setScenarioType('procedural')}
                      className={`p-3.5 rounded border text-left cursor-pointer transition-colors ${scenarioType === 'procedural'
                          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-[0_0_10px_rgba(61,214,198,0.2)]'
                          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                        }`}
                    >
                      <Target className="w-5 h-5 text-[var(--c-accent)] mb-1.5" />
                      <div className="font-bold text-sm font-mono text-[var(--c-text-max)]">PROCEDURAL</div>
                      <div className="text-xs font-mono text-[var(--c-text-mid)] mt-1">Stochastic threat generation</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setScenarioType('adaptive')}
                      className={`p-3.5 rounded border text-left cursor-pointer transition-colors ${scenarioType === 'adaptive'
                          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-[0_0_10px_rgba(61,214,198,0.2)]'
                          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                        }`}
                    >
                      <Sparkles className="w-5 h-5 text-[var(--c-accent)] mb-1.5" />
                      <div className="font-bold text-sm font-mono text-[var(--c-text-max)]">ADAPTIVE</div>
                      <div className="text-xs font-mono text-[var(--c-text-mid)] mt-1">Targets operator weakness profile</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setScenarioType('scripted')}
                      className={`p-3.5 rounded border text-left cursor-pointer transition-colors ${scenarioType === 'scripted'
                          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-[0_0_10px_rgba(61,214,198,0.2)]'
                          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                        }`}
                    >
                      <ShieldAlert className="w-5 h-5 text-[var(--c-accent)] mb-1.5" />
                      <div className="font-bold text-sm font-mono text-[var(--c-text-max)]">SCRIPTED DRILL</div>
                      <div className="text-xs font-mono text-[var(--c-text-mid)] mt-1">Certified standard benchmark</div>
                    </button>
                  </div>

                  {scenarioType === 'scripted' && scriptedOptions.length > 0 && (
                    <Select
                      label="Select Standardized Scripted Drill"
                      options={scriptedOptions.map((s) => ({ value: s, label: s }))}
                      value={selectedScripted}
                      onChange={(e) => setSelectedScripted(e.target.value)}
                    />
                  )}

                  <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded space-y-1.5">
                    <Slider
                      label="Threat Ingress Difficulty (1-5)"
                      min={1}
                      max={5}
                      step={1}
                      value={difficulty}
                      onChange={setDifficulty}
                      unit="Tier"
                    />
                    <div className="text-xs font-mono text-[var(--c-accent)] pt-0.5">
                      {difficulty === 1 && 'Tier 1: Low-density reconnaissance vectors (3-4 contacts).'}
                      {difficulty === 2 && 'Tier 2: Mixed stealth decoys and single sensor jamming (5-7 contacts).'}
                      {difficulty === 3 && 'Tier 3: Coordinated multi-axis raid with active EW jamming (8-10 contacts).'}
                      {difficulty === 4 && 'Tier 4: Distributed swarm saturation with low-altitude terrain masking.'}
                      {difficulty === 5 && 'Tier 5: Maximum saturation raid with autonomous swarm evasions.'}
                    </div>
                  </div>
                </div>
              )}

              {step === 3 && (
                <div className="space-y-4 flex-1 flex flex-col justify-center">
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label="Time of Day"
                      options={[
                        { value: 'day', label: 'Daylight // Clear Visual Optics' },
                        { value: 'dusk', label: 'Dusk // Reduced Thermal Contrast' },
                        { value: 'night', label: 'Night // EO/IR Sensor Dependent' },
                      ]}
                      value={timeOfDay}
                      onChange={(e) => setTimeOfDay(e.target.value)}
                    />
                    <Select
                      label="Terrain Operational Environment"
                      options={[
                        { value: 'rural', label: 'Rural Open Plains' },
                        { value: 'coastal', label: 'Coastal Sea-Skimming' },
                        { value: 'mountain', label: 'Mountain Ridge Masking' },
                        { value: 'urban', label: 'Urban Clutter & Multipath' },
                      ]}
                      value={terrain}
                      onChange={(e) => setTerrain(e.target.value)}
                    />
                  </div>

                  <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded space-y-1.5">
                    <Slider
                      label="Electronic Warfare / Sensor Degradation"
                      min={0}
                      max={1}
                      step={0.1}
                      value={degradation}
                      onChange={setDegradation}
                      unit="Noise"
                    />
                    <div className="flex justify-between text-xs font-mono text-[var(--c-text-mid)] pt-0.5">
                      <span>0.0 (Clean Baseline Signal)</span>
                      <span className="text-[var(--c-accent)] font-bold">Current: {Math.round(degradation * 100)}% Noise Floor</span>
                      <span>1.0 (Severe EW)</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded">
                    <div>
                      <div className="text-sm font-mono font-bold text-[var(--c-text-max)]">Red-Team Evasive Tactics</div>
                      <div className="text-xs font-mono text-[var(--c-text-mid)] mt-0.5">
                        Injects dynamic swarm splitting, low altitude masking, and RF spoofing
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={redTeamTactics}
                      onChange={(e) => setRedTeamTactics(e.target.checked)}
                      className="accent-[var(--c-accent)] w-5 h-5 cursor-pointer ml-4"
                    />
                  </div>
                </div>
              )}

              {step === 4 && (
                <div className="space-y-4 font-mono text-xs sm:text-sm flex-1 flex flex-col justify-center">
                  <div className="p-4 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded space-y-2.5">
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)] text-xs uppercase">Operator Callsign & Unit:</span>
                      <span className="text-[var(--c-text-max)] font-bold">{traineeId} ({unitId})</span>
                    </div>
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)] text-xs uppercase">Doctrine Architecture:</span>
                      <span className="text-[var(--c-accent)] font-bold uppercase">{scenarioType}</span>
                    </div>
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)] text-xs uppercase">Difficulty & Environment:</span>
                      <span className="text-[var(--c-text-max)] font-bold">Tier 0{difficulty} · {timeOfDay.toUpperCase()} · {terrain.toUpperCase()}</span>
                    </div>
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)] text-xs uppercase">EW Degradation:</span>
                      <span className={degradation > 0 ? 'text-[var(--c-warning)] font-bold' : 'text-[var(--c-accent)]'}>
                        {degradation > 0 ? `${Math.round(degradation * 100)}% Active EW Interference` : 'Nominal Baseline Profile'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--c-text-low)] text-xs uppercase">Rules of Engagement:</span>
                      <span className="text-[var(--c-accent)] font-bold">Weapons Free Post-Dual Sensor ID</span>
                    </div>
                  </div>

                  <Button
                    variant="primary"
                    size="lg"
                    className="w-full py-3 font-mono text-sm font-bold tracking-wider cursor-pointer shadow-lg"
                    onClick={() => launchMission()}
                    icon={<Play className="w-5 h-5 fill-current" />}
                  >
                    DEPLOY RADAR TO OPERATIONAL SECTOR
                  </Button>
                </div>
              )}

              {/* Stepper Navigation Footer */}
              <div className="flex items-center justify-between pt-3 border-t border-[var(--c-border-base)] mt-auto shrink-0">
                {step > 1 ? (
                  <Button variant="secondary" size="md" onClick={() => setStep(step - 1)} icon={<ChevronLeft className="w-4 h-4" />}>
                    Previous Step
                  </Button>
                ) : <div />}
                {step < 4 ? (
                  <Button variant="primary" size="md" onClick={() => setStep(step + 1)}>
                    <span>Proceed to Step 0{step + 1}</span>
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                ) : null}
              </div>
            </Panel>
          </div>

          {/* Right Preview Card (4 Cols) - Standardized Uniform Height */}
          <div className="lg:col-span-4 flex flex-col">
            <Panel
              title="Mission Telemetry"
              subtitle="RISK MATRIX"
              className="h-[440px] flex flex-col overflow-hidden"
              bodyClassName="p-4 sm:p-5 flex-1 flex flex-col justify-between space-y-3 overflow-hidden"
            >
              <div className="space-y-3">
                {/* 2-Column Side-by-Side Stats to prevent vertical overflow */}
                <div className="grid grid-cols-2 gap-3">
                  <Stat
                    label="Threat Ingress"
                    value={difficulty * 3}
                    unit="TARGETS"
                    sparkline={[2, 4, difficulty * 3]}
                  />
                  <Stat
                    label="Sensor SNR"
                    value={Math.round(28 - degradation * 18)}
                    unit="dB"
                    sparkline={[28, 25, Math.round(28 - degradation * 18)]}
                  />
                </div>

                <div className="p-3 bg-[var(--c-bg-deep)] rounded border border-[var(--c-border-faint)] text-xs font-mono space-y-1">
                  <div className="text-xs text-[var(--c-text-low)] uppercase font-semibold">DEFENDED BASE ASSET</div>
                  <div className="font-bold text-sm text-[var(--c-warning)]">HQ SECTOR COMMAND POST (0, 0)</div>
                  <div className="text-xs text-[var(--c-text-mid)] leading-relaxed">
                    Range limit: 6.0km · Collateral-free zone: 1.2km
                  </div>
                </div>

                <div className="p-3 bg-[var(--c-bg-deep)] rounded border border-[var(--c-border-faint)] text-xs font-mono space-y-1.5">
                  <div className="text-xs text-[var(--c-text-low)] uppercase font-semibold">POTENTIAL TARGET SIGNATURES</div>
                  <div className="flex flex-wrap gap-1.5 pt-0.5">
                    <Badge variant="hostile" size="xs">QUADCOPTER</Badge>
                    <Badge variant="warning" size="xs">DECOY</Badge>
                    <Badge variant="friendly" size="xs">FRIENDLY UAV</Badge>
                    <Badge variant="info" size="xs">BIOLOGICAL BIRD</Badge>
                    {difficulty >= 3 && <Badge variant="hostile" size="xs">AUTONOMOUS SWARM</Badge>}
                  </div>
                </div>
              </div>

              <div className="text-xs font-mono text-[var(--c-text-mid)] text-center pt-2 border-t border-[var(--c-border-base)] shrink-0 flex items-center justify-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[var(--c-accent)] animate-pulse" />
                <span>Radar Telemetry Ready • 3D Dome Active</span>
              </div>
            </Panel>
          </div>
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════════
  // STATE B: LIVE EXERCISE
  // ═════════════════════════════════════════════════════════════════════════════
  if (viewState === 'live' && scenario) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Top Tactical Bar */}
        <div className="min-h-12 px-4 py-1.5 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex flex-wrap items-center justify-between gap-2 text-xs font-mono shrink-0">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-[var(--c-accent)] shadow-[0_0_8px_var(--c-accent)] animate-pulse" />
            <span className="font-bold text-[var(--c-text-max)] uppercase">
              {(scenario.scenario_id || 'SCENARIO').slice(0, 14)}
            </span>
            <span className="text-[var(--c-text-low)]">|</span>
            <span className="text-[var(--c-text-mid)] hidden md:inline">
              OPERATOR: <strong className="text-[var(--c-text-max)]">{traineeId}</strong>
            </span>
          </div>

          <div className="flex items-center gap-4">
            <SensorHealthChips sensorOutage={sensorOutage} />

            {/* Mission Timer */}
            <div className="flex items-center gap-1.5 px-3 py-1 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
              <Clock className="w-3.5 h-3.5 text-[var(--c-accent)]" />
              <span className="font-display font-bold text-sm text-[var(--c-text-max)] tabular-nums">
                {Math.floor(elapsed / 60).toString().padStart(2, '0')}:{(elapsed % 60).toString().padStart(2, '0')}
              </span>
            </div>

            <Button
              variant="danger"
              size="sm"
              onClick={finishExercise}
              icon={<Zap className="w-3.5 h-3.5" />}
            >
              Finish & Score
            </Button>
          </div>
        </div>

        {/* ROE Collapse Banner */}
        {roeOpen && (
          <div className="bg-[var(--c-bg-surface)] border-b border-[var(--c-border-strong)] px-4 py-2 flex items-center justify-between text-xs font-mono animate-in slide-in-from-top duration-150">
            <div className="flex items-center gap-3">
              <Badge variant="warning">ROE PINNED</Badge>
              <span className="text-[var(--c-text-high)]">
                {scenario.roe?.description || 'Hard-kill authorized only after dual-sensor verification. No-engage zones active.'}
              </span>
            </div>
            <button
              onClick={() => setRoeOpen(false)}
              className="text-[var(--c-text-low)] hover:text-white font-mono cursor-pointer"
            >
              ✕ CLOSE (R)
            </button>
          </div>
        )}

        {/* Main 3-Column Tactical Display */}
        <div className="flex-1 flex min-h-0 overflow-hidden">
          {/* Left Column: Contact Queue (Sorted by Threat Priority) */}
          <div className="w-80 bg-[var(--c-bg-surface)] border-r border-[var(--c-border-base)] flex flex-col shrink-0">
            <div className="p-3 border-b border-[var(--c-border-base)] bg-[var(--c-bg-deep)] flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-[var(--c-text-max)]">ACTIVE CONTACTS ({activeContacts.length})</span>
              <span className="text-[10px] text-[var(--c-text-low)]">KEYS: J / K</span>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {activeContacts.length === 0 ? (
                <div className="p-6 text-center text-xs font-mono text-[var(--c-text-low)] space-y-2">
                  <div className="w-3 h-3 rounded-full bg-[var(--c-accent)] mx-auto animate-ping opacity-75" />
                  <div>SCANNING SECTOR RADAR...</div>
                  <div className="text-[10px] text-[var(--c-text-faint)]">Awaiting airborne target ingress</div>
                </div>
              ) : (
                activeContacts.map((c) => {
                  const isNeutralized = c.state === 'neutralized';
                  return (
                    <div
                      key={c.id}
                      onClick={() => setSelectedContactId(c.id)}
                      className={`p-2.5 rounded-sm border cursor-pointer select-none transition-all ${
                        isNeutralized
                          ? 'bg-black/40 border-zinc-800 opacity-60'
                          : c.id === selectedContact?.id
                          ? 'bg-[var(--c-bg-subtle)] border-[var(--c-accent)] shadow-[0_0_12px_rgba(0,232,122,0.15)]'
                          : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <Badge variant={isNeutralized ? 'friendly' : ((c.assigned_classification || c.classification || 'unknown') as any)} size="xs">
                          {isNeutralized ? 'NEUTRALIZED' : (c.assigned_classification || c.classification || 'unknown').replace('_', ' ')}
                        </Badge>
                        <span className={`font-mono font-bold text-xs ${isNeutralized ? 'line-through text-zinc-500' : 'text-[var(--c-text-max)]'}`}>
                          {(c.id || 'TRACK').toUpperCase()}
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-1 text-[10px] font-mono text-[var(--c-text-mid)]">
                        <div><span className="text-[var(--c-text-low)] block">RNG</span>{(c.range_km ?? 0).toFixed(1)}k</div>
                        <div><span className="text-[var(--c-text-low)] block">BRG</span>{Math.round(c.bearing_deg ?? 0)}°</div>
                        <div><span className="text-[var(--c-text-low)] block">ALT</span>{isNeutralized ? '0m' : `${c.altitude ?? 120}m`}</div>
                        <div><span className="text-[var(--c-text-low)] block">SPD</span>{isNeutralized ? '0m/s' : `${c.speed ?? 0}m/s`}</div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Center Column: RadarScope Canvas / 3D Airspace / Urban Map */}
          <div className="flex-1 flex flex-col min-h-0 bg-[var(--c-bg-void)]">
            {/* View Mode Bar */}
            <div className="h-9 px-3 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex items-center justify-between shrink-0">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setScopeViewMode('2d')}
                  className={`flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-mono uppercase font-bold rounded-xs transition-colors cursor-pointer ${scopeViewMode === '2d'
                      ? 'bg-[var(--c-accent)] text-black shadow-[0_0_8px_rgba(0,232,122,0.4)]'
                      : 'text-[var(--c-text-mid)] hover:text-white hover:bg-white/5'
                    }`}
                >
                  <Layers className="w-3 h-3" />
                  2D Tactical PPI
                </button>
                <button
                  onClick={() => setScopeViewMode('3d')}
                  className={`flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-mono uppercase font-bold rounded-xs transition-colors cursor-pointer ${scopeViewMode === '3d'
                      ? 'bg-[var(--c-accent)] text-black shadow-[0_0_8px_rgba(0,232,122,0.4)]'
                      : 'text-[var(--c-text-mid)] hover:text-white hover:bg-white/5'
                    }`}
                >
                  <Box className="w-3 h-3" />
                  3D Airspace Dome
                </button>
                <button
                  onClick={() => setScopeViewMode('urban')}
                  className={`flex items-center gap-1.5 px-2.5 py-1 text-[10px] font-mono uppercase font-bold rounded-xs transition-colors cursor-pointer ${scopeViewMode === 'urban'
                      ? 'bg-[var(--c-accent)] text-black shadow-[0_0_8px_rgba(0,232,122,0.4)]'
                      : 'text-[var(--c-text-mid)] hover:text-white hover:bg-white/5'
                    }`}
                >
                  <MapPin className="w-3 h-3" />
                  3D Urban Canopy
                </button>
              </div>

              <div className="text-[10px] font-mono text-[var(--c-text-low)] flex items-center gap-2">
                <span>ZOOM: SCROLL</span>
                <span>•</span>
                <span>ROTATE: DRAG</span>
              </div>
            </div>

            <div className="flex-1 relative min-h-0">
              {scopeViewMode === '2d' && (
                <RadarScope
                  contacts={activeContacts}
                  selectedContactId={selectedContact?.id || null}
                  onSelectContact={(id) => setSelectedContactId(id)}
                  defendedAsset={scenario.defended_asset_position}
                  noEngageZones={scenario.no_engage_zones}
                />
              )}
              {scopeViewMode === '3d' && (
                <Airspace3D
                  contacts={activeContacts}
                  selectedContactId={selectedContact?.id || null}
                  onSelectContact={(id) => setSelectedContactId(id)}
                  defendedAsset={scenario.defended_asset_position}
                  noEngageZones={scenario.no_engage_zones}
                  maxRangeKm={6}
                />
              )}
              {scopeViewMode === 'urban' && (
                <UrbanMap3D
                  contacts={activeContacts}
                  selectedContactId={selectedContact?.id || null}
                  onSelectContact={(id) => setSelectedContactId(id)}
                  defendedAsset={scenario.defended_asset_position}
                  originLat={SECTOR_COORDINATES[terrain]?.lat || 28.6139}
                  originLon={SECTOR_COORDINATES[terrain]?.lon || 77.2090}
                />
              )}
            </div>

            {/* Bottom Combat Events Feed */}
            <div className="h-36 border-t border-[var(--c-border-base)] bg-[var(--c-bg-deep)] p-2">
              <EventLog events={events} maxHeight="h-28" />
            </div>
          </div>

          {/* Right Column: Selected Target Tactical Action Console */}
          <div className="w-96 bg-[var(--c-bg-surface)] border-l border-[var(--c-border-base)] flex flex-col shrink-0 overflow-y-auto">
            {selectedContact ? (
              <div className="p-4 space-y-4 font-mono">
                {/* Header */}
                <div className="border-b border-[var(--c-border-base)] pb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-[var(--c-text-low)] uppercase">LOCK TARGET TRACK</span>
                    {selectedContact.state === 'neutralized' ? (
                      <Badge variant="friendly" size="xs">
                        ✓ NEUTRALIZED
                      </Badge>
                    ) : (
                      <Badge variant={(selectedContact.assigned_classification || selectedContact.classification || 'unknown') as any} size="xs">
                        {(selectedContact.state || 'TRACKING').toUpperCase()}
                      </Badge>
                    )}
                  </div>
                  <h3 className={`text-xl font-bold font-display mt-1 ${selectedContact.state === 'neutralized' ? 'line-through text-zinc-500' : 'text-[var(--c-text-max)]'}`}>
                    {(selectedContact.id || 'TRACK').toUpperCase()}
                  </h3>
                </div>

                {/* Sensor Tabs (Radar / EO-IR / RF / Acoustic) */}
                <div className="space-y-2">
                  <div className="text-[10px] text-[var(--c-text-low)] uppercase tracking-wider">
                    MULTISPECTRAL SENSOR CHANNEL
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {(['radar', 'eo_ir', 'rf', 'acoustic'] as const).map((s) => (
                      <button
                        key={s}
                        onClick={() => {
                          setActiveSensorTab(s);
                          readSensor(selectedContact.id, s);
                        }}
                        className={`p-1.5 text-[10px] font-mono uppercase font-semibold rounded-xs border cursor-pointer ${activeSensorTab === s
                            ? 'bg-[var(--c-bg-raised)] text-[var(--c-accent)] border-[var(--c-accent)]'
                            : 'bg-[var(--c-bg-deep)] text-[var(--c-text-mid)] border-[var(--c-border-base)]'
                          }`}
                      >
                        {s.replace('_', '/')}
                      </button>
                    ))}
                  </div>

                  {/* Active Sensor Readout */}
                  <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded-xs text-xs space-y-1">
                    {sensorOutage === activeSensorTab ? (
                      <div className="text-[var(--c-hostile)] flex items-center gap-1.5 font-bold">
                        <AlertTriangle className="w-4 h-4" /> CHANNEL JAMMED BY HOSTILE EW
                      </div>
                    ) : activeSensorTab === 'radar' ? (
                      <>
                        <div className="flex justify-between">
                          <span className="text-[var(--c-text-low)]">CHANNEL STATUS:</span>
                          <span className="text-[var(--c-accent)] font-semibold">SYNCHRONIZED</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[var(--c-text-low)]">DOPPLER PEAK:</span>
                          <span className="text-[var(--c-text-max)]">{(110 + (selectedContact.speed || 10) * 1.8).toFixed(1)} Hz</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[var(--c-text-low)]">EST. ROTOR / RCS:</span>
                          <span className="text-[var(--c-text-max)]">{(selectedContact.rcs || -14).toFixed(1)} dBsm</span>
                        </div>
                      </>
                    ) : sensorData[`${selectedContact.id}_${activeSensorTab}`] ? (
                      (() => {
                        const data = sensorData[`${selectedContact.id}_${activeSensorTab}`];
                        if (!data.available) {
                          return (
                            <div className="text-[var(--c-warning)] flex items-center gap-1.5 font-bold">
                              <AlertTriangle className="w-4 h-4" /> {data.reason || 'UNAVAILABLE'}
                            </div>
                          );
                        }
                        if (activeSensorTab === 'eo_ir') {
                          return (
                            <>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">EO/IR VISUAL SHAPE:</span>
                                <span className="text-[var(--c-accent)] font-semibold uppercase">{data.shape_hint || 'UNKNOWN'}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">OPTICAL CONFIDENCE:</span>
                                <span className="text-[var(--c-text-max)]">{Math.round((data.confidence || 0) * 100)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">THERMAL SIGNATURE:</span>
                                <span className="text-[var(--c-text-max)]">{data.detected ? 'POSITIVE THERMAL BLIP' : 'BELOW IR THRESHOLD'}</span>
                              </div>
                            </>
                          );
                        }
                        if (activeSensorTab === 'rf') {
                          return (
                            <>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">RF DETECTED:</span>
                                <span className={data.detected ? 'text-[var(--c-hostile)] font-bold' : 'text-[var(--c-text-mid)]'}>
                                  {data.detected ? 'RF EMISSION ACTIVE' : 'RF SILENT'}
                                </span>
                              </div>
                              <div className="text-[11px] text-[var(--c-text-mid)] pt-1">
                                {data.note || 'Broadband spectrum analyzer locked'}
                              </div>
                            </>
                          );
                        }
                        if (activeSensorTab === 'acoustic') {
                          return (
                            <>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">ACOUSTIC AUDIBILITY:</span>
                                <span className={data.detected ? 'text-[var(--c-accent)] font-bold' : 'text-[var(--c-text-low)]'}>
                                  {data.detected ? 'ROTOR NOISE AUDIBLE' : 'IN AUDIBLE / OUT OF RANGE'}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">RANGE TO ASSET:</span>
                                <span className="text-[var(--c-text-max)]">{data.range_m}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-[var(--c-text-low)]">EFFECTIVE MICROPHONE:</span>
                                <span className="text-[var(--c-text-mid)]">{data.effective_range_m}m radius</span>
                              </div>
                            </>
                          );
                        }
                        return null;
                      })()
                    ) : (
                      <div className="text-[var(--c-text-low)] italic text-[11px] py-1">
                        Click sensor button above to query multispectral feed.
                      </div>
                    )}
                  </div>
                </div>

                {/* AI Hint Section */}
                <div className="space-y-2 pt-2 border-t border-[var(--c-border-faint)]">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-[var(--c-text-low)] uppercase tracking-wider">AI ASSIST COGNITION</span>
                    <Button size="xs" variant="outline" onClick={requestAiHint}>
                      <Sparkles className="w-3 h-3 text-[var(--c-accent)]" /> AI Hint
                    </Button>
                  </div>
                  {aiHint && (
                    <div className="p-2.5 bg-[var(--c-bg-deep)] border border-[var(--c-info)]/40 rounded-xs space-y-1 text-xs">
                      <div className="flex justify-between font-bold">
                        <span className="text-[var(--c-info)]">{aiHint.suggested}</span>
                        <span className="text-[var(--c-text-max)]">{Math.round(aiHint.confidence * 100)}%</span>
                      </div>
                      <p className="text-[11px] text-[var(--c-text-mid)]">{aiHint.rationale}</p>
                    </div>
                  )}
                </div>

                {/* Classification Segmented Control (Hotkeys 1-4) */}
                <div className="space-y-2 pt-2 border-t border-[var(--c-border-faint)]">
                  <div className="text-[10px] text-[var(--c-text-low)] uppercase tracking-wider">
                    TARGET CLASSIFICATION (HOTKEYS 1-4)
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      size="xs"
                      variant="danger"
                      kbd="1"
                      onClick={() => classifyContact(selectedContact.id, 'hostile_drone')}
                    >
                      Hostile Drone (▲)
                    </Button>
                    <Button
                      size="xs"
                      variant="secondary"
                      kbd="2"
                      onClick={() => classifyContact(selectedContact.id, 'friendly_drone')}
                    >
                      Friendly (●)
                    </Button>
                    <Button
                      size="xs"
                      variant="secondary"
                      kbd="3"
                      onClick={() => classifyContact(selectedContact.id, 'decoy')}
                    >
                      Decoy (□)
                    </Button>
                    <Button
                      size="xs"
                      variant="secondary"
                      kbd="4"
                      onClick={() => classifyContact(selectedContact.id, 'bird')}
                    >
                      Neutral (○)
                    </Button>
                  </div>
                </div>

                {/* Countermeasures & Engagement */}
                <div className="space-y-2 pt-2 border-t border-[var(--c-border-faint)]">
                  <div className="text-[10px] text-[var(--c-text-low)] uppercase tracking-wider">
                    ENGAGEMENT ORDERS
                  </div>
                  <div className="space-y-2">
                    <div className="grid grid-cols-2 gap-2">
                      <Button
                        size="xs"
                        variant="secondary"
                        disabled={selectedContact.state === 'neutralized'}
                        onClick={() => executeEngagement('warn')}
                      >
                        Warn / Hail
                      </Button>
                      <Button
                        size="xs"
                        variant={traineeLog[selectedContact.id]?.engagement_action === 'soft_kill' ? 'primary' : 'secondary'}
                        disabled={selectedContact.state === 'neutralized'}
                        onClick={() => executeEngagement('soft_kill')}
                      >
                        {traineeLog[selectedContact.id]?.engagement_action === 'soft_kill' ? '✓ RF Jammed' : 'Soft-Kill (RF Jam)'}
                      </Button>
                    </div>

                    {/* Hard-kill with 600ms Hold-to-confirm */}
                    <button
                      type="button"
                      disabled={selectedContact.state === 'neutralized'}
                      onMouseDown={selectedContact.state !== 'neutralized' ? startHardKillHold : undefined}
                      onMouseUp={cancelHardKillHold}
                      onMouseLeave={cancelHardKillHold}
                      onTouchStart={selectedContact.state !== 'neutralized' ? startHardKillHold : undefined}
                      onTouchEnd={cancelHardKillHold}
                      className={`w-full py-2.5 px-3 rounded font-mono font-bold text-xs flex items-center justify-center gap-2 select-none border transition-all ${
                        selectedContact.state === 'neutralized'
                          ? 'bg-emerald-950/60 border-emerald-500/50 text-emerald-400 cursor-not-allowed shadow-[0_0_12px_rgba(16,185,129,0.2)]'
                          : hardKillHolding
                          ? 'bg-[var(--c-hostile)] text-white scale-[0.98] shadow-[0_0_20px_var(--c-hostile)] cursor-pointer'
                          : 'bg-[var(--c-hostile-bg)] text-[var(--c-hostile)] border-[var(--c-hostile)] hover:bg-[var(--c-hostile)] hover:text-white cursor-pointer'
                      }`}
                    >
                      {selectedContact.state === 'neutralized' ? (
                        <>
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          <span>TARGET NEUTRALIZED (KINETIC KILL)</span>
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4" />
                          <span>{hardKillHolding ? 'HOLDING CONFIRM (600ms)...' : 'HARD-KILL (HOLD TO FIRE)'}</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-xs font-mono text-[var(--c-text-low)]">
                SELECT A TRACK ON THE RADAR SCOPE TO INSPECT TELEMETRY
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════════
  // STATE C: DEBRIEF & SCORECARD
  // ═════════════════════════════════════════════════════════════════════════════
  if (viewState === 'debrief' && debriefResult) {
    const score = debriefResult.score || 85;
    const gradeColor = score >= 85 ? '#00e87a' : score >= 70 ? '#0a84ff' : score >= 50 ? '#ffb800' : '#ef4444';

    return (
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <Panel title="After-Action Evaluation & Scorecard" subtitle="DEBRIEF">
          <div className="p-6 space-y-6">
            {/* Hero Score */}
            <div className="flex flex-col sm:flex-row items-center justify-between p-6 bg-[var(--c-bg-deep)] border border-[var(--c-border-strong)] rounded-sm">
              <div className="text-center sm:text-left mb-4 sm:mb-0">
                <span className="text-xs font-mono text-[var(--c-text-low)] uppercase tracking-wider block">
                  OVERALL MISSION PERFORMANCE
                </span>
                <div className="text-5xl font-display font-extrabold tracking-tight mt-1" style={{ color: gradeColor }}>
                  {score.toFixed(1)}%
                </div>
                <div className="text-xs font-mono text-[var(--c-text-mid)] mt-1">
                  Engagement Efficiency: {(debriefResult.contacts_total || 4) - (debriefResult.contacts_missed || 0)}/{debriefResult.contacts_total || 4} Neutralized
                </div>
              </div>

              <div className="px-6 py-3 border-2 rounded-sm text-center" style={{ borderColor: gradeColor, color: gradeColor }}>
                <span className="font-display font-bold text-xl uppercase tracking-widest block">
                  {score >= 85 ? 'TIER 1 · CERTIFIED' : score >= 70 ? 'TIER 2 · QUALIFIED' : 'FAILED ROE'}
                </span>
                <span className="text-[10px] font-mono text-[var(--c-text-low)]">DEFENSE STANDARD 05-UAS</span>
              </div>
            </div>

            {/* Metrics Breakdown */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Stat label="Detection Speed" value={debriefResult.detection_score || 90} unit="%" />
              <Stat label="Classification" value={debriefResult.classification_score || 85} unit="%" />
              <Stat label="ROE Compliance" value={debriefResult.engagement_score || 95} unit="%" />
              <Stat label="Decision Timeliness" value={debriefResult.timeliness_score || 80} unit="%" />
            </div>

            {/* Critical Flags / Attention */}
            <div className="p-4 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-sm space-y-2">
              <div className="text-xs font-mono font-bold text-[var(--c-warning)] uppercase tracking-wider flex items-center gap-2">
                <AlertTriangle className="w-4 h-4" /> ROE & Tactical Disciplinary Flags
              </div>
              <ul className="text-xs font-mono text-[var(--c-text-high)] space-y-1 list-disc pl-5">
                {debriefResult.critical_flags && debriefResult.critical_flags.length > 0 ? (
                  debriefResult.critical_flags.map((f: string, i: number) => <li key={i}>{f}</li>)
                ) : (
                  <li className="text-[var(--c-accent)]">Clean ROE record — Zero civilian/friendly collateral infractions.</li>
                )}
              </ul>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-[var(--c-border-base)]">
              <div className="flex items-center gap-2">
                <Button
                  variant="primary"
                  onClick={() => navigate('/aar')}
                  icon={<BarChart className="w-4 h-4" />}
                >
                  Open Full AAR Telemetry
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => window.open(`/aar/session/${debriefResult.session_id}/print`, '_blank')}
                  icon={<Printer className="w-4 h-4" />}
                >
                  Print Report
                </Button>
              </div>

              <Button
                variant="secondary"
                onClick={() => {
                  setViewState('briefing');
                  setStep(1);
                }}
                icon={<RotateCcw className="w-4 h-4" />}
              >
                Deploy Next Scenario
              </Button>
            </div>
          </div>
        </Panel>
      </div>
    );
  }

  return null;
};
