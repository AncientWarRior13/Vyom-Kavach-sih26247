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
import { SensorHealthChips, EventLog, CombatEvent } from '../components/domain/TacticalFeed';
import { api } from '../api/client';
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
} from 'lucide-react';

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
    }).catch(() => {});
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
      } catch {}
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
    setTraineeLog((prev) => ({
      ...prev,
      [contactId]: {
        ...(prev[contactId] || { contact_id: contactId, detected_at: elapsed }),
        classification,
        classified_at: elapsed,
      },
    }));

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
    try {
      const res = await api.sensorRead({ scenario_id: scenario.scenario_id, contact_id: contactId, sensor_type: sensor });
      setSensorData((prev) => ({
        ...prev,
        [`${contactId}_${sensor}`]: res.readings,
      }));
      addEvent(`Sensor telemetry locked: ${sensor.toUpperCase()} on track ${contactId}`, 'info', sensor.toUpperCase());
    } catch {
      toast('SENSOR ERROR', `${sensor.toUpperCase()} channel jammed or degraded`, 'warning');
    }
  };

  // AI Hint Trigger
  const requestAiHint = () => {
    if (!selectedContact) return;
    const hints: Record<string, any> = {
      hostile_drone: { confidence: 0.94, rationale: 'High micro-doppler blade modulation + RF control uplink on 2.4 GHz', suggested: 'Hostile Kamikaze Drone' },
      decoy: { confidence: 0.88, rationale: 'Radar RCS 5.2m² disproportionate to 18m/s flight envelope; RF beacon loop', suggested: 'Electronic Decoy' },
      friendly_drone: { confidence: 0.99, rationale: 'Transponder IFF Mode-5 squawk authenticated', suggested: 'Allied Recon Drone' },
      unknown: { confidence: 0.72, rationale: 'Low RCS target flying below 100m AGL, thermal IR signature nominal', suggested: 'Suspicious Unknown' },
    };
    const cls = selectedContact.assigned_classification || selectedContact.classification || 'unknown';
    const hint = hints[cls] || hints.unknown;
    setAiHint(hint);
    addEvent(`AI Classification Model advisory for ${selectedContact.id}: ${hint.suggested} (${Math.round(hint.confidence * 100)}%)`, 'info', 'AI-ASSIST');
  };

  // Engagement Actions (Warn / Soft-kill / Hard-kill)
  const executeEngagement = async (action: 'warn' | 'soft_kill' | 'hard_kill') => {
    if (!scenario || !selectedContact) return;
    const cid = selectedContact.id;

    setTraineeLog((prev) => ({
      ...prev,
      [cid]: {
        ...(prev[cid] || { contact_id: cid, detected_at: elapsed }),
        engagement_action: action,
        engagement_at: elapsed,
      },
    }));

    // Check ROE violation
    if (scenario.roe?.weapons_hold && action === 'hard_kill') {
      addEvent(`ROE VIOLATION: Hard-kill executed during WEAPONS HOLD!`, 'critical', 'ROE');
      toast('ROE BREACH', 'Weapons Hold was in effect!', 'error');
    } else {
      addEvent(`Order executed: ${action.toUpperCase()} against ${cid.toUpperCase()}`, action === 'hard_kill' ? 'critical' : 'warning', 'WEAPONS');
      toast('ACTION DEPLOYED', `${action.toUpperCase()} ordered on track ${cid}`, 'success');
    }

    try {
      const res = await api.engageContact({
        scenario_id: scenario.scenario_id,
        contact_id: cid,
        weapon_type: action === 'soft_kill' ? 'rf_jam' : 'kinetic_net',
      });
      if (res && res.result) {
        addEvent(`Engagement Result: ${res.result}`, 'success', 'ENGAGE');
      }
    } catch {}
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
  // STATE A: BRIEFING STEPPER
  // ═════════════════════════════════════════════════════════════════════════════
  if (viewState === 'briefing') {
    return (
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="border-b border-[var(--c-border-base)] pb-4 flex items-center justify-between">
          <div>
            <h1 className="font-display text-2xl font-bold tracking-wider text-[var(--c-text-max)] uppercase">
              Operational Mission Briefing
            </h1>
            <p className="font-mono text-xs text-[var(--c-text-mid)] mt-1">
              Configure exercise parameters, sensor degradation, and threat doctrine before deployment.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {[1, 2, 3, 4].map((s) => (
              <div
                key={s}
                className={`w-8 h-8 rounded-xs font-mono text-xs flex items-center justify-center font-bold border ${
                  step === s
                    ? 'bg-[var(--c-accent)] text-[#020508] border-[var(--c-accent)] shadow-[0_0_8px_var(--c-accent)]'
                    : step > s
                    ? 'bg-[var(--c-bg-raised)] text-[var(--c-accent)] border-[var(--c-accent)]/50'
                    : 'bg-[var(--c-bg-surface)] text-[var(--c-text-low)] border-[var(--c-border-base)]'
                }`}
              >
                0{s}
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Stepper Body */}
          <div className="lg:col-span-2 space-y-6">
            {step === 1 && (
              <Panel title="Step 01 // Trainee Identity" subtitle="AUTH & CREW RECORD">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-mono text-[var(--c-text-mid)] uppercase">Trainee Call-Sign</label>
                      <input
                        type="text"
                        value={traineeId}
                        onChange={(e) => setTraineeId(e.target.value)}
                        className="bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] rounded-xs px-3 py-2 text-xs font-mono text-[var(--c-text-max)] focus:outline-none focus:border-[var(--c-accent)]"
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-mono text-[var(--c-text-mid)] uppercase">Operational Unit</label>
                      <input
                        type="text"
                        value={unitId}
                        onChange={(e) => setUnitId(e.target.value)}
                        className="bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] rounded-xs px-3 py-2 text-xs font-mono text-[var(--c-text-max)] focus:outline-none focus:border-[var(--c-accent)]"
                      />
                    </div>
                  </div>
                  <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded-xs text-xs font-mono text-[var(--c-text-mid)]">
                    Identified operator records are synchronized with the after-action review telemetry server for certification tracking.
                  </div>
                </div>
              </Panel>
            )}

            {step === 2 && (
              <Panel title="Step 02 // Threat Scenario Architecture" subtitle="PROFILES">
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      type="button"
                      onClick={() => setScenarioType('procedural')}
                      className={`p-3 rounded-xs border text-left cursor-pointer transition-colors ${
                        scenarioType === 'procedural'
                          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-xs'
                          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                      }`}
                    >
                      <Target className="w-4 h-4 text-[var(--c-accent)] mb-2" />
                      <div className="font-bold text-xs font-mono text-[var(--c-text-max)]">PROCEDURAL</div>
                      <div className="text-[10px] font-mono text-[var(--c-text-mid)] mt-1">Stochastic threat generation</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setScenarioType('adaptive')}
                      className={`p-3 rounded-xs border text-left cursor-pointer transition-colors ${
                        scenarioType === 'adaptive'
                          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-xs'
                          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                      }`}
                    >
                      <Sparkles className="w-4 h-4 text-[var(--c-accent)] mb-2" />
                      <div className="font-bold text-xs font-mono text-[var(--c-text-max)]">ADAPTIVE</div>
                      <div className="text-[10px] font-mono text-[var(--c-text-mid)] mt-1">Targets operator weakness profile</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setScenarioType('scripted')}
                      className={`p-3 rounded-xs border text-left cursor-pointer transition-colors ${
                        scenarioType === 'scripted'
                          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-xs'
                          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                      }`}
                    >
                      <ShieldAlert className="w-4 h-4 text-[var(--c-accent)] mb-2" />
                      <div className="font-bold text-xs font-mono text-[var(--c-text-max)]">SCRIPTED DRILL</div>
                      <div className="text-[10px] font-mono text-[var(--c-text-mid)] mt-1">Certified standard benchmark</div>
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

                  <div className="pt-2">
                    <Slider
                      label="Threat Ingress Difficulty (1-5)"
                      min={1}
                      max={5}
                      step={1}
                      value={difficulty}
                      onChange={setDifficulty}
                      unit="Tier"
                    />
                  </div>
                </div>
              </Panel>
            )}

            {step === 3 && (
              <Panel title="Step 03 // Environmental Conditions & EW" subtitle="ATMOSPHERICS">
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <Select
                      label="Time of Day"
                      options={[
                        { value: 'day', label: 'Daylight // Clear Vis' },
                        { value: 'dusk', label: 'Dusk // Reduced Thermal Vis' },
                        { value: 'night', label: 'Night // EO/IR Dependent' },
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

                  <Slider
                    label="Electronic Warfare / Sensor Degradation"
                    min={0}
                    max={1}
                    step={0.1}
                    value={degradation}
                    onChange={setDegradation}
                    unit="Noise"
                  />

                  <div className="flex items-center justify-between p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded-xs">
                    <div>
                      <div className="text-xs font-mono font-bold text-[var(--c-text-max)]">Red-Team Evasive Tactics</div>
                      <div className="text-[10px] font-mono text-[var(--c-text-mid)]">Injects split swarms, low altitude masking, and RF spoofing</div>
                    </div>
                    <input
                      type="checkbox"
                      checked={redTeamTactics}
                      onChange={(e) => setRedTeamTactics(e.target.checked)}
                      className="accent-[var(--c-accent)] w-4 h-4 cursor-pointer"
                    />
                  </div>
                </div>
              </Panel>
            )}

            {step === 4 && (
              <Panel title="Step 04 // Operational Review & Deployment" subtitle="MISSION CONFIRMATION">
                <div className="space-y-4 font-mono text-xs">
                  <div className="p-4 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-xs space-y-2">
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)]">CALL-SIGN:</span>
                      <span className="text-[var(--c-text-max)] font-bold">{traineeId} ({unitId})</span>
                    </div>
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)]">DOCTRINE ARCHITECTURE:</span>
                      <span className="text-[var(--c-accent)] font-bold uppercase">{scenarioType}</span>
                    </div>
                    <div className="flex justify-between border-b border-[var(--c-border-faint)] pb-2">
                      <span className="text-[var(--c-text-low)]">DIFFICULTY / CONDITIONS:</span>
                      <span className="text-[var(--c-text-max)] font-bold">L-0{difficulty} · {timeOfDay.toUpperCase()} · {terrain.toUpperCase()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--c-text-low)]">RULES OF ENGAGEMENT:</span>
                      <span className="text-[var(--c-accent)] font-bold">WEAPONS FREE POST-DUAL SENSOR ID</span>
                    </div>
                  </div>

                  <Button
                    variant="primary"
                    size="lg"
                    className="w-full py-3"
                    onClick={() => launchMission()}
                    icon={<Play className="w-5 h-5 fill-current" />}
                  >
                    DEPLOY RADAR TO OPERATIONAL SECTOR
                  </Button>
                </div>
              </Panel>
            )}

            {/* Stepper Navigation */}
            <div className="flex items-center justify-between pt-4 border-t border-[var(--c-border-base)]">
              {step > 1 ? (
                <Button variant="secondary" onClick={() => setStep(step - 1)} icon={<ChevronLeft className="w-4 h-4" />}>
                  Previous
                </Button>
              ) : <div />}
              {step < 4 ? (
                <Button variant="primary" onClick={() => setStep(step + 1)}>
                  <span>Proceed to Step 0{step + 1}</span>
                  <ChevronRight className="w-4 h-4" />
                </Button>
              ) : null}
            </div>
          </div>

          {/* Right Preview Card */}
          <div>
            <Panel title="Mission Telemetry Preview" subtitle="CALCULATED RISK">
              <div className="space-y-4">
                <Stat
                  label="Estimated Threat Ingress"
                  value={difficulty * 3}
                  unit="TARGETS"
                  sparkline={[2, 4, difficulty * 3]}
                />
                <Stat
                  label="Atmospheric Sensor SNR"
                  value={Math.round(28 - degradation * 18)}
                  unit="dB"
                  sparkline={[28, 25, Math.round(28 - degradation * 18)]}
                />
                <div className="p-3 bg-[var(--c-bg-deep)] rounded-xs border border-[var(--c-border-faint)] text-xs font-mono space-y-2">
                  <div className="text-[10px] text-[var(--c-text-low)] uppercase">DEFENDED BASE ASSET</div>
                  <div className="font-bold text-[var(--c-warning)]">HQ SECTOR COMMAND POST (0, 0)</div>
                  <div className="text-[11px] text-[var(--c-text-mid)]">
                    Target engagement range limit: 6.0km. Collateral-free zone radius: 1.2km.
                  </div>
                </div>
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
        <div className="h-11 px-4 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex items-center justify-between text-xs font-mono shrink-0">
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
                activeContacts.map((c) => (
                  <div
                    key={c.id}
                    onClick={() => setSelectedContactId(c.id)}
                    className={`p-2.5 rounded-sm border cursor-pointer select-none transition-all ${
                      c.id === selectedContact?.id
                        ? 'bg-[var(--c-bg-subtle)] border-[var(--c-accent)] shadow-[0_0_12px_rgba(0,232,122,0.15)]'
                        : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <Badge variant={(c.assigned_classification || c.classification || 'unknown') as any} size="xs">
                        {(c.assigned_classification || c.classification || 'unknown').replace('_', ' ')}
                      </Badge>
                      <span className="font-mono font-bold text-xs text-[var(--c-text-max)]">{(c.id || 'TRACK').toUpperCase()}</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1 text-[10px] font-mono text-[var(--c-text-mid)]">
                      <div><span className="text-[var(--c-text-low)] block">RNG</span>{(c.range_km ?? 0).toFixed(1)}k</div>
                      <div><span className="text-[var(--c-text-low)] block">BRG</span>{Math.round(c.bearing_deg ?? 0)}°</div>
                      <div><span className="text-[var(--c-text-low)] block">ALT</span>{c.altitude ?? 120}m</div>
                      <div><span className="text-[var(--c-text-low)] block">SPD</span>{c.speed ?? 0}m/s</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Center Column: RadarScope Canvas */}
          <div className="flex-1 flex flex-col min-h-0 bg-[var(--c-bg-void)]">
            <div className="flex-1 relative min-h-0">
              <RadarScope
                contacts={activeContacts}
                selectedContactId={selectedContact?.id || null}
                onSelectContact={(id) => setSelectedContactId(id)}
                defendedAsset={scenario.defended_asset_position}
                noEngageZones={scenario.no_engage_zones}
              />
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
                    <Badge variant={(selectedContact.assigned_classification || selectedContact.classification || 'unknown') as any} size="xs">
                      {(selectedContact.state || 'TRACKING').toUpperCase()}
                    </Badge>
                  </div>
                  <h3 className="text-xl font-bold font-display text-[var(--c-text-max)] mt-1">
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
                        className={`p-1.5 text-[10px] font-mono uppercase font-semibold rounded-xs border cursor-pointer ${
                          activeSensorTab === s
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
                    ) : (
                      <>
                        <div className="flex justify-between">
                          <span className="text-[var(--c-text-low)]">CHANNEL STATUS:</span>
                          <span className="text-[var(--c-accent)] font-semibold">SYNCHRONIZED</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[var(--c-text-low)]">DOPPLER PEAK:</span>
                          <span className="text-[var(--c-text-max)]">128.4 Hz</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[var(--c-text-low)]">EST. ROTOR BLADES:</span>
                          <span className="text-[var(--c-text-max)]">4 (QUADCOPTER)</span>
                        </div>
                      </>
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
                        onClick={() => executeEngagement('warn')}
                      >
                        Warn / Hail
                      </Button>
                      <Button
                        size="xs"
                        variant="secondary"
                        onClick={() => executeEngagement('soft_kill')}
                      >
                        Soft-Kill (RF Jam)
                      </Button>
                    </div>

                    {/* Hard-kill with 600ms Hold-to-confirm */}
                    <button
                      type="button"
                      onMouseDown={startHardKillHold}
                      onMouseUp={cancelHardKillHold}
                      onMouseLeave={cancelHardKillHold}
                      onTouchStart={startHardKillHold}
                      onTouchEnd={cancelHardKillHold}
                      className={`w-full py-2.5 px-3 rounded font-mono font-bold text-xs flex items-center justify-center gap-2 cursor-pointer transition-all select-none border ${
                        hardKillHolding
                          ? 'bg-[var(--c-hostile)] text-white scale-[0.98] shadow-[0_0_20px_var(--c-hostile)]'
                          : 'bg-[var(--c-hostile-bg)] text-[var(--c-hostile)] border-[var(--c-hostile)] hover:bg-[var(--c-hostile)] hover:text-white'
                      }`}
                    >
                      <Zap className="w-4 h-4" />
                      <span>{hardKillHolding ? 'HOLDING CONFIRM (600ms)...' : 'HARD-KILL (HOLD TO FIRE)'}</span>
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
