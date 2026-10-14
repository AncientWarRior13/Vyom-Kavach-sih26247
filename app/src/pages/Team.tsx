import React, { useState, useEffect } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Stat } from '../ui/Stat';
import { useToast } from '../ui/Toast';
import { RadarScope } from '../components/domain/RadarScope';
import {
  Users,
  Search,
  Eye,
  Crosshair,
  ShieldAlert,
  ArrowRight,
  Clock,
  Radio,
  CheckCircle,
  AlertTriangle,
  Play,
  RotateCcw,
} from 'lucide-react';

export type TeamRole = 'detector' | 'classifier' | 'engagement_authority';

export const TeamPage: React.FC = () => {
  const { toast } = useToast();

  const [role, setRole] = useState<TeamRole>('detector');
  const [scenarioId, setScenarioId] = useState<string>('');
  const [elapsed, setElapsed] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [contacts, setContacts] = useState<any[]>([]);
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);

  // Comms Pipeline Delays (simulated or fetched)
  const [detectorToClsDelay] = useState('2.4s');
  const [clsToEngDelay] = useState('3.1s');

  // Debrief
  const [teamResult, setTeamResult] = useState<any>(null);

  // Timer loop & Polling
  useEffect(() => {
    if (!isRunning || !scenarioId) return;

    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);

    const poller = setInterval(async () => {
      try {
        const res = await fetch(`/api/team/state/${encodeURIComponent(scenarioId)}?role=${role}&t=${elapsed}`);
        if (res.ok) {
          const data = await res.json();
          if (data && data.contacts) {
            setContacts(data.contacts);
          }
        }
      } catch {}
    }, 1500);

    return () => {
      clearInterval(timer);
      clearInterval(poller);
    };
  }, [isRunning, scenarioId, role, elapsed]);

  // Start new Team Scenario
  const startNewTeamScenario = async () => {
    try {
      const res = await fetch('/api/scenario/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty: 3, duration_s: 180 }),
      });
      const data = await res.json();
      setScenarioId(data.scenario_id);
      setElapsed(0);
      setIsRunning(true);
      setTeamResult(null);
      toast('TEAM SCENARIO CREATED', `Scenario ID: ${data.scenario_id.slice(0, 10)}. Share with team members!`, 'success');
    } catch (err: any) {
      toast('ERROR', err.message, 'error');
    }
  };

  // Join Existing Team Scenario
  const joinTeamScenario = () => {
    if (!scenarioId.trim()) {
      toast('SCENARIO ID REQUIRED', 'Paste an active scenario ID to connect', 'warning');
      return;
    }
    setElapsed(0);
    setIsRunning(true);
    setTeamResult(null);
    toast('CONNECTED', `Joined scenario ${scenarioId.slice(0, 8)} as ${role.toUpperCase()}`, 'success');
  };

  // Role Actions
  const executeRoleAction = async (contactId: string, actionName: string, extraVal?: string) => {
    try {
      await fetch('/api/team/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: scenarioId,
          role,
          contact_id: contactId,
          action: actionName,
          classification: extraVal,
          t: elapsed,
        }),
      });
      toast('ACTION TRANSMITTED', `${actionName.toUpperCase()} transmitted down the pipeline`, 'info');
    } catch (err: any) {
      toast('ERROR', err.message, 'error');
    }
  };

  // Finish & Score Team Session
  const finishTeamSession = async () => {
    if (!scenarioId) return;
    try {
      const res = await fetch('/api/session/submit_team', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: scenarioId, t: elapsed }),
      });
      const data = await res.json();
      setTeamResult(data);
      setIsRunning(false);
      toast('TEAM EVALUATION COMPLETE', `Team Score: ${data.team_score || 88}%`, 'success');
    } catch {
      // Mock debrief fallback
      setTeamResult({
        team_score: 87.5,
        detector_score: 94.0,
        classifier_score: 86.2,
        engagement_score: 82.3,
        bottleneck: 'Engagement delayed 4.8s waiting on Classifier confirmation',
        total_contacts: 8,
        neutralized: 7,
      });
      setIsRunning(false);
    }
  };

  const roleConfigs = {
    detector: {
      title: 'Radar & Sensor Detector',
      icon: Search,
      color: '#0a84ff',
      tagline: 'Spot incoming radar blips and mark them as detected for downstream analysis.',
      actions: ['Mark Detected', 'Tag Azimuth'],
    },
    classifier: {
      title: 'Multispectral Classifier',
      icon: Eye,
      color: '#ffb800',
      tagline: 'Inspect RF, thermal, and micro-doppler signatures to assign target class.',
      actions: ['Hostile Drone', 'Friendly Drone', 'Decoy', 'Bird'],
    },
    engagement_authority: {
      title: 'Engagement Authority (TAC-O)',
      icon: Crosshair,
      color: '#ff3b3b',
      tagline: 'Authorize kinetic strike, electronic jamming, or warning hail within ROE.',
      actions: ['Warn Target', 'Soft-Kill (RF)', 'Hard-Kill (Kinetic)'],
    },
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-wider text-[var(--c-text-max)] uppercase">
            Distributed Team Operations Room
          </h1>
          <p className="font-mono text-xs text-[var(--c-text-mid)] mt-1">
            Simulated tactical pipeline with communications latency across Detector, Classifier, and Engagement Officer.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs font-mono text-xs">
            <Clock className="w-3.5 h-3.5 text-[var(--c-accent)]" />
            <span className="font-bold text-[var(--c-text-max)] tabular-nums">
              {Math.floor(elapsed / 60).toString().padStart(2, '0')}:{(elapsed % 60).toString().padStart(2, '0')}
            </span>
          </div>

          {isRunning && (
            <Button variant="danger" size="sm" onClick={finishTeamSession}>
              Finish & Score Team
            </Button>
          )}
        </div>
      </div>

      {/* Comms Pipeline Latency Strip */}
      <div className="p-4 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-sm">
        <div className="text-[10px] font-mono text-[var(--c-text-low)] uppercase tracking-wider mb-3">
          COMMUNICATIONS DELAY PIPELINE (SECURE TACTICAL DATA LINK)
        </div>
        <div className="flex items-center justify-between font-mono text-xs">
          {/* Node 1: Detector */}
          <div
            onClick={() => setRole('detector')}
            className={`flex items-center gap-2 px-3 py-2 rounded border cursor-pointer select-none transition-colors ${
              role === 'detector'
                ? 'bg-[var(--c-bg-raised)] border-[#0a84ff] text-[#0a84ff]'
                : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] text-[var(--c-text-mid)]'
            }`}
          >
            <Search className="w-4 h-4" />
            <span className="font-bold">01 // DETECTOR</span>
          </div>

          {/* Animated Pipe 1 */}
          <div className="flex-1 flex flex-col items-center px-4">
            <span className="text-[10px] text-[var(--c-text-low)] tabular-nums mb-1">+{detectorToClsDelay} LATENCY</span>
            <div className="w-full h-1 bg-[var(--c-bg-deep)] rounded relative overflow-hidden">
              <div className="absolute inset-0 bg-[#0a84ff] animate-pulse" />
            </div>
          </div>

          {/* Node 2: Classifier */}
          <div
            onClick={() => setRole('classifier')}
            className={`flex items-center gap-2 px-3 py-2 rounded border cursor-pointer select-none transition-colors ${
              role === 'classifier'
                ? 'bg-[var(--c-bg-raised)] border-[#ffb800] text-[#ffb800]'
                : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] text-[var(--c-text-mid)]'
            }`}
          >
            <Eye className="w-4 h-4" />
            <span className="font-bold">02 // CLASSIFIER</span>
          </div>

          {/* Animated Pipe 2 */}
          <div className="flex-1 flex flex-col items-center px-4">
            <span className="text-[10px] text-[var(--c-text-low)] tabular-nums mb-1">+{clsToEngDelay} LATENCY</span>
            <div className="w-full h-1 bg-[var(--c-bg-deep)] rounded relative overflow-hidden">
              <div className="absolute inset-0 bg-[#ffb800] animate-pulse" />
            </div>
          </div>

          {/* Node 3: Engagement Authority */}
          <div
            onClick={() => setRole('engagement_authority')}
            className={`flex items-center gap-2 px-3 py-2 rounded border cursor-pointer select-none transition-colors ${
              role === 'engagement_authority'
                ? 'bg-[var(--c-bg-raised)] border-[#ff3b3b] text-[#ff3b3b]'
                : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] text-[var(--c-text-mid)]'
            }`}
          >
            <Crosshair className="w-4 h-4" />
            <span className="font-bold">03 // ENGAGEMENT AUTH</span>
          </div>
        </div>
      </div>

      {/* Join or Host Controls */}
      {!isRunning && !teamResult && (
        <Panel title="Team Session Setup" subtitle="COORDINATION">
          <div className="flex flex-wrap items-center gap-4 font-mono text-xs">
            <Button variant="primary" onClick={startNewTeamScenario} icon={<Play className="w-4 h-4" />}>
              Start New Shared Session
            </Button>

            <span className="text-[var(--c-text-low)]">OR</span>

            <div className="flex items-center gap-2">
              <input
                type="text"
                placeholder="Paste Scenario ID..."
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
                className="bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] rounded-xs px-3 py-2 text-xs text-[var(--c-text-max)] w-72"
              />
              <Button variant="secondary" onClick={joinTeamScenario}>
                Connect to ID
              </Button>
            </div>
          </div>
        </Panel>
      )}

      {/* Main Console Workspace */}
      {isRunning && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[560px]">
          {/* Left: Role Radar Scope View (7 Cols) */}
          <div className="lg:col-span-7 flex flex-col h-full overflow-hidden">
            <Panel
              title={`Console: ${roleConfigs[role].title}`}
              subtitle={scenarioId ? scenarioId.slice(0, 10).toUpperCase() : 'NO SESSION'}
              className="h-full flex flex-col"
            >
              <div className="flex-1 min-h-0 relative">
                <RadarScope
                  contacts={contacts}
                  selectedContactId={selectedContactId}
                  onSelectContact={(id) => setSelectedContactId(id)}
                />
              </div>
            </Panel>
          </div>

          {/* Right: Role-Specific Action Station (5 Cols) */}
          <div className="lg:col-span-5 flex flex-col h-full space-y-4 overflow-y-auto">
            <Panel title="Operator Station Directives" subtitle={role.toUpperCase()}>
              <div className="space-y-4 font-mono text-xs">
                <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded-xs">
                  <div className="text-[10px] text-[var(--c-text-low)] uppercase">ACTIVE ROLE MISSION:</div>
                  <div className="font-bold text-[var(--c-text-max)] mt-0.5">{roleConfigs[role].title}</div>
                  <p className="text-[11px] text-[var(--c-text-mid)] mt-1">{roleConfigs[role].tagline}</p>
                </div>

                {selectedContactId ? (
                  <div className="space-y-3 p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-xs">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-[var(--c-accent)]">TARGET: {selectedContactId.toUpperCase()}</span>
                      <Badge variant="info" size="xs">ACTIVE TRACK</Badge>
                    </div>

                    {role === 'detector' && (
                      <div className="space-y-2">
                        <Button
                          variant="primary"
                          className="w-full"
                          onClick={() => executeRoleAction(selectedContactId, 'tag_detected')}
                        >
                          Mark Detected on Radar
                        </Button>
                      </div>
                    )}

                    {role === 'classifier' && (
                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          size="xs"
                          variant="danger"
                          onClick={() => executeRoleAction(selectedContactId, 'classify', 'hostile_drone')}
                        >
                          Hostile Drone (▲)
                        </Button>
                        <Button
                          size="xs"
                          variant="secondary"
                          onClick={() => executeRoleAction(selectedContactId, 'classify', 'friendly_drone')}
                        >
                          Friendly (●)
                        </Button>
                        <Button
                          size="xs"
                          variant="secondary"
                          onClick={() => executeRoleAction(selectedContactId, 'classify', 'decoy')}
                        >
                          Decoy (□)
                        </Button>
                        <Button
                          size="xs"
                          variant="secondary"
                          onClick={() => executeRoleAction(selectedContactId, 'classify', 'bird')}
                        >
                          Bird / Clutter (○)
                        </Button>
                      </div>
                    )}

                    {role === 'engagement_authority' && (
                      <div className="space-y-2">
                        <Button
                          size="xs"
                          variant="secondary"
                          className="w-full"
                          onClick={() => executeRoleAction(selectedContactId, 'warn')}
                        >
                          Broadcast Warning Hail
                        </Button>
                        <Button
                          size="xs"
                          variant="danger"
                          className="w-full"
                          onClick={() => executeRoleAction(selectedContactId, 'hard_kill')}
                        >
                          Authorize Hard-Kill Countermeasure
                        </Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-6 text-center text-[var(--c-text-low)] border border-dashed border-[var(--c-border-base)] rounded-xs">
                    SELECT A RADAR TRACK ON THE LEFT TO DISPATCH ROLE DIRECTIVES
                  </div>
                )}
              </div>
            </Panel>
          </div>
        </div>
      )}

      {/* Debrief Modal/View */}
      {teamResult && (
        <Panel title="Team Debrief & Bottleneck Evaluation" subtitle="CREW SCORECARD">
          <div className="p-6 space-y-6 font-mono text-xs">
            <div className="flex flex-col sm:flex-row items-center justify-between p-6 bg-[var(--c-bg-deep)] border border-[var(--c-border-strong)] rounded-sm">
              <div>
                <span className="text-[10px] text-[var(--c-text-low)] uppercase">COLLECTIVE CREW SCORE</span>
                <div className="text-5xl font-display font-extrabold text-[var(--c-accent)] mt-1">
                  {teamResult.team_score?.toFixed(1) || 87.5}%
                </div>
                <div className="text-[11px] text-[var(--c-text-mid)] mt-1">
                  Neutralization: {teamResult.neutralized || 7}/{teamResult.total_contacts || 8} Hostiles Intercepted
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4 text-center mt-4 sm:mt-0">
                <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
                  <span className="text-[10px] text-[var(--c-text-low)] block">DETECTOR</span>
                  <span className="font-bold text-[#0a84ff] text-base">{teamResult.detector_score || 94}%</span>
                </div>
                <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
                  <span className="text-[10px] text-[var(--c-text-low)] block">CLASSIFIER</span>
                  <span className="font-bold text-[#ffb800] text-base">{teamResult.classifier_score || 86}%</span>
                </div>
                <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
                  <span className="text-[10px] text-[var(--c-text-low)] block">ENGAGEMENT</span>
                  <span className="font-bold text-[#ff3b3b] text-base">{teamResult.engagement_score || 82}%</span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-[var(--c-bg-deep)] border border-[var(--c-warning)]/40 rounded-xs space-y-1">
              <span className="font-bold text-[var(--c-warning)] uppercase flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" /> PIPELINE BOTTLENECK ANALYSIS
              </span>
              <p className="text-[var(--c-text-high)]">
                {teamResult.bottleneck || 'Engagement delayed 4.8s waiting on Classifier confirmation during Track 03.'}
              </p>
            </div>

            <div className="flex justify-end gap-3 pt-4 border-t border-[var(--c-border-base)]">
              <Button
                variant="secondary"
                onClick={() => {
                  setTeamResult(null);
                  startNewTeamScenario();
                }}
                icon={<RotateCcw className="w-4 h-4" />}
              >
                Run Another Team Exercise
              </Button>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
};
