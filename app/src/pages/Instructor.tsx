import React, { useState, useEffect } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Select } from '../ui/Select';
import { Slider } from '../ui/Slider';
import { useToast } from '../ui/Toast';
import { RadarScope } from '../components/domain/RadarScope';
import { EventLog, CombatEvent } from '../components/domain/TacticalFeed';
import {
  Radio,
  Zap,
  AlertTriangle,
  Play,
  RotateCcw,
  Sparkles,
  Flame,
  Shield,
  Eye,
  Activity,
} from 'lucide-react';

export const InstructorPage: React.FC = () => {
  const { toast } = useToast();

  const [activeScenarios, setActiveScenarios] = useState<any[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('');
  const [mirroredScenario, setMirroredScenario] = useState<any>(null);
  const [revealGroundTruth, setRevealGroundTruth] = useState(false);

  // Injection Toolbox
  const [injectType, setInjectType] = useState('hostile_drone');
  const [injectTime, setInjectTime] = useState(0);
  const [jamSensor, setJamSensor] = useState<'eo_ir' | 'rf' | 'acoustic'>('rf');
  const [events, setEvents] = useState<CombatEvent[]>([]);

  // Fetch active scenarios polling every 2s
  const fetchScenarios = async () => {
    try {
      const res = await fetch('/api/instructor/active_scenarios').then(r => r.json());
      setActiveScenarios(res || []);
      if (!selectedScenarioId && res && res.length > 0) {
        setSelectedScenarioId(res[0].scenario_id);
      }
    } catch {}
  };

  useEffect(() => {
    fetchScenarios();
    const interval = setInterval(fetchScenarios, 2000);
    return () => clearInterval(interval);
  }, []);

  // Poll mirrored scenario state
  useEffect(() => {
    if (!selectedScenarioId) return;

    const fetchMirrored = async () => {
      try {
        const sc = await fetch(`/api/scenario/${encodeURIComponent(selectedScenarioId)}/refresh`).then(r => r.json());
        setMirroredScenario(sc);
      } catch {}
    };

    fetchMirrored();
    const interval = setInterval(fetchMirrored, 2000);
    return () => clearInterval(interval);
  }, [selectedScenarioId]);

  const addLog = (message: string, level: CombatEvent['level'] = 'info', source = 'INSTRUCTOR') => {
    const ev: CombatEvent = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: new Date().toISOString().slice(11, 19) + 'Z',
      message,
      level,
      source,
    };
    setEvents((prev) => [ev, ...prev]);
  };

  // Inject Threat Handler
  const handleInject = async () => {
    if (!selectedScenarioId) {
      toast('SELECT SCENARIO', 'Choose an active trainee session first', 'warning');
      return;
    }
    try {
      const res = await fetch('/api/instructor/inject_threat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: selectedScenarioId,
          contact_type: injectType,
          t: injectTime,
        }),
      });
      const data = await res.json();
      if (data.injected) {
        toast('INJECTION SUCCESS', `Injected ${injectType} at t=${injectTime}s`, 'success');
        addLog(`Injected target ${data.contact?.contact_id || 'UAS'} (${injectType}) into session ${selectedScenarioId.slice(0, 8)}`, 'warning');
      } else {
        toast('INJECTION FAILED', 'Scenario may have concluded', 'error');
      }
    } catch (err: any) {
      toast('ERROR', err.message, 'error');
    }
  };

  // Jam Sensor Handler
  const handleJam = async (sensorToJam: string | null) => {
    if (!selectedScenarioId) return;
    try {
      const res = await fetch('/api/instructor/jam_sensor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: selectedScenarioId,
          sensor: sensorToJam,
        }),
      });
      const data = await res.json();
      if (sensorToJam) {
        toast('JAM ACTIVATED', `Sensor outage set: ${sensorToJam.toUpperCase()}`, 'warning');
        addLog(`Electronic Jamming activated on ${sensorToJam.toUpperCase()}`, 'critical', 'EW-JAM');
      } else {
        toast('JAM CLEARED', 'Sensors restored to nominal operation', 'success');
        addLog(`Electronic Jamming cleared on session ${selectedScenarioId.slice(0, 8)}`, 'success', 'EW-JAM');
      }
    } catch (err: any) {
      toast('ERROR', err.message, 'error');
    }
  };

  // Stress Test Macros
  const runMacro = async (macroName: string) => {
    if (!selectedScenarioId) {
      toast('SELECT SCENARIO', 'Choose an active trainee session first', 'warning');
      return;
    }
    if (macroName === 'swarm_rf_jam') {
      await handleJam('rf');
      await fetch('/api/instructor/inject_threat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: selectedScenarioId, contact_type: 'hostile_drone', t: injectTime }),
      });
      await fetch('/api/instructor/inject_threat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: selectedScenarioId, contact_type: 'decoy', t: injectTime + 1 }),
      });
      toast('MACRO FIRED', 'Swarm Ingress + RF Jam deployed in sequence', 'success');
      addLog(`Macro 'Swarm Ingress + RF Jam' executed against session`, 'critical', 'MACRO');
    } else if (macroName === 'sensor_blackout') {
      await handleJam('eo_ir');
      toast('MACRO FIRED', 'Optical / IR Blackout injected', 'warning');
      addLog(`Macro 'Sensor Blackout' triggered on EO/IR channel`, 'warning', 'MACRO');
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-wider text-[var(--c-text-max)] uppercase">
            Instructor Control & Injection Room
          </h1>
          <p className="font-mono text-xs text-[var(--c-text-mid)] mt-1">
            Real-time trainee telemetry surveillance, threat vector insertion, and EW jamming control.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="warning" pulse>INSTRUCTOR PRIVILEGES ACTIVE</Badge>
        </div>
      </div>

      {/* Main 3-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[720px]">
        {/* Left: Active Scenario Cards (3 Cols) */}
        <div className="lg:col-span-3 flex flex-col h-full overflow-hidden">
          <Panel
            title="Live Sessions"
            subtitle={`${activeScenarios.length} ACTIVE`}
            className="h-full flex flex-col"
          >
            <div className="p-2 space-y-2 overflow-y-auto flex-1">
              {activeScenarios.length === 0 ? (
                <div className="p-6 text-center text-xs font-mono text-[var(--c-text-low)]">
                  NO ACTIVE TRAINEE SESSIONS
                  <div className="mt-2 text-[10px] text-[var(--c-text-mid)]">
                    Launch a drill from /trainer to mirror here.
                  </div>
                </div>
              ) : (
                activeScenarios.map((s) => (
                  <div
                    key={s.scenario_id}
                    onClick={() => setSelectedScenarioId(s.scenario_id)}
                    className={`p-3 rounded-sm border cursor-pointer select-none transition-all font-mono text-xs space-y-1.5 ${
                      s.scenario_id === selectedScenarioId
                        ? 'bg-[var(--c-bg-subtle)] border-[var(--c-accent)] shadow-[0_0_12px_rgba(0,232,122,0.15)]'
                        : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)]'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-[var(--c-text-max)]">
                        {s.scenario_id.slice(0, 10).toUpperCase()}
                      </span>
                      <span className="w-2 h-2 rounded-full bg-[var(--c-accent)] shadow-[0_0_6px_var(--c-accent)] animate-pulse" />
                    </div>
                    <div className="text-[11px] text-[var(--c-text-mid)]">
                      Contacts: <span className="text-[var(--c-text-max)] font-bold">{s.contact_count}</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-[var(--c-text-low)]">
                      <span>ELAPSED: {s.duration_s || 0}s</span>
                      <Badge variant={s.sensor_outage ? 'critical' : 'default'} size="xs">
                        {s.sensor_outage ? `JAM: ${s.sensor_outage}` : 'NOMINAL'}
                      </Badge>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Panel>
        </div>

        {/* Center: Mirrored RadarScope (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col h-full overflow-hidden">
          <Panel
            title="Trainee Radar Scope Mirror"
            subtitle={selectedScenarioId ? selectedScenarioId.slice(0, 10).toUpperCase() : 'NO LOCK'}
            actions={
              <button
                type="button"
                onClick={() => setRevealGroundTruth(!revealGroundTruth)}
                className={`text-[10px] font-mono px-2 py-0.5 rounded-xs border cursor-pointer ${
                  revealGroundTruth
                    ? 'bg-[var(--c-accent-bg)] text-[var(--c-accent)] border-[var(--c-accent)]'
                    : 'bg-[var(--c-bg-raised)] text-[var(--c-text-low)] border-[var(--c-border-base)]'
                }`}
              >
                <Eye className="w-3 h-3 inline mr-1" />
                {revealGroundTruth ? 'GROUND TRUTH ON' : 'TRAINEE VIEW'}
              </button>
            }
            className="h-full flex flex-col"
          >
            <div className="flex-1 min-h-0 relative">
              <RadarScope
                contacts={mirroredScenario?.contacts || []}
              />
            </div>
          </Panel>
        </div>

        {/* Right: Injection Toolbox & Macros (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col h-full space-y-4 overflow-y-auto">
          {/* Threat Injection Panel */}
          <Panel title="Threat Vector Injector" subtitle="CONTROL">
            <div className="space-y-4 font-mono text-xs">
              <Select
                label="Target Classification Type"
                value={injectType}
                onChange={(e) => setInjectType(e.target.value)}
                options={[
                  { value: 'hostile_drone', label: 'Hostile Kamikaze Drone (▲)' },
                  { value: 'decoy', label: 'Electronic Decoy Transponder (□)' },
                  { value: 'friendly_drone', label: 'Friendly Commercial Drone (●)' },
                  { value: 'bird', label: 'Bird / Environmental Radar False Alarm (○)' },
                ]}
              />

              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[var(--c-text-low)]">DELAY / INJECTION TIME:</span>
                  <span className="text-[var(--c-accent)] font-bold tabular-nums">{injectTime}s</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={60}
                  value={injectTime}
                  onChange={(e) => setInjectTime(Number(e.target.value))}
                  className="accent-[var(--c-accent)]"
                />
              </div>

              <Button
                variant="danger"
                className="w-full"
                onClick={handleInject}
                icon={<Flame className="w-4 h-4" />}
              >
                Inject Target into Trainee Sector
              </Button>
            </div>
          </Panel>

          {/* EW Jamming Controls */}
          <Panel title="Electronic Warfare Jammer" subtitle="DEGRADATION">
            <div className="space-y-3 font-mono text-xs">
              <div className="grid grid-cols-3 gap-2">
                {(['rf', 'eo_ir', 'acoustic'] as const).map((s) => (
                  <Button
                    key={s}
                    size="xs"
                    variant={jamSensor === s ? 'danger' : 'secondary'}
                    onClick={() => {
                      setJamSensor(s);
                      handleJam(s);
                    }}
                  >
                    Jam {s.replace('_', '/').toUpperCase()}
                  </Button>
                ))}
              </div>

              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={() => handleJam(null)}
              >
                Clear Jamming (Restore Nominal)
              </Button>
            </div>
          </Panel>

          {/* Tactical Macro Presets */}
          <Panel title="Tactical Stress Macros" subtitle="ONE-CLICK">
            <div className="space-y-2 font-mono text-xs">
              <button
                type="button"
                onClick={() => runMacro('swarm_rf_jam')}
                className="w-full p-2.5 rounded-xs bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] hover:border-[var(--c-hostile)] text-left cursor-pointer transition-colors"
              >
                <div className="font-bold text-[var(--c-hostile)] flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5" /> SWARM DIVERGENCE + RF JAM
                </div>
                <div className="text-[10px] text-[var(--c-text-low)] mt-0.5">
                  Injects hostile + decoy simultaneous pair and disables RF spectrum.
                </div>
              </button>

              <button
                type="button"
                onClick={() => runMacro('sensor_blackout')}
                className="w-full p-2.5 rounded-xs bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] hover:border-[var(--c-warning)] text-left cursor-pointer transition-colors"
              >
                <div className="font-bold text-[var(--c-warning)] flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" /> OPTICAL SENSOR BLACKOUT
                </div>
                <div className="text-[10px] text-[var(--c-text-low)] mt-0.5">
                  Jams EO/IR thermal channel to force acoustic & radar reliance.
                </div>
              </button>
            </div>
          </Panel>
        </div>
      </div>

      {/* Bottom Events Feed */}
      <Panel title="Instructor Injection & EW Transmission Log" className="h-44">
        <EventLog events={events} maxHeight="h-32" />
      </Panel>
    </div>
  );
};
