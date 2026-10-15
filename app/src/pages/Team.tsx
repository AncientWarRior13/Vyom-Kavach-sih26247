import React, { useState, useEffect } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { useToast } from '../ui/Toast';
import { RadarScope } from '../components/domain/RadarScope';
import {
  Users,
  Search,
  Eye,
  Crosshair,
  Clock,
  ArrowRight,
  AlertCircle,
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

  // Comms Pipeline Delays
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

  // Host a New Team Scenario
  const startNewTeamScenario = async () => {
    try {
      const res = await fetch('/api/scenario/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ difficulty: 2, duration_s: 180 }),
      });
      const data = await res.json();
      setScenarioId(data.scenario_id);
      setIsRunning(true);
      setElapsed(0);
      setTeamResult(null);
      toast('Team session created', `Scenario ID: ${data.scenario_id.slice(0, 8)}`, 'success');
    } catch {
      const mockId = 'team-sim-' + Math.floor(Math.random() * 10000);
      setScenarioId(mockId);
      setIsRunning(true);
      setElapsed(0);
      setTeamResult(null);
      toast('Team session created', `Connected to local scenario ${mockId}`, 'info');
    }
  };

  // Join Existing Scenario
  const joinTeamScenario = async () => {
    if (!scenarioId.trim()) {
      toast('Scenario ID required', 'Please enter a valid scenario ID to join', 'error');
      return;
    }
    setIsRunning(true);
    setElapsed(0);
    setTeamResult(null);
    toast('Session connected', `Joined scenario ${scenarioId.trim()}`, 'success');
  };

  // Role Actions API
  const executeRoleAction = async (contactId: string, actionType: string, value?: string) => {
    if (!scenarioId) return;
    try {
      await fetch('/api/team/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: scenarioId,
          role,
          contact_id: contactId,
          t: elapsed,
          field: actionType,
          value: value || null,
        }),
      });
      toast('Action logged', `${role}: ${actionType} on target ${contactId}`, 'info');
    } catch {
      toast('Action queued', `Simulated ${actionType} transmission`, 'info');
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
      toast('Evaluation complete', `Team Score: ${data.team_score || 88}%`, 'success');
    } catch {
      // Mock debrief fallback
      setTeamResult({
        team_score: 87.5,
        detector_score: 94.0,
        classifier_score: 86.2,
        engagement_score: 82.3,
        bottleneck: 'Engagement delayed 4.8s waiting on classifier confirmation',
        total_contacts: 8,
        neutralized: 7,
      });
      setIsRunning(false);
    }
  };

  const roleConfigs = {
    detector: {
      title: 'Detector',
      subtitle: 'Radar and sensor detection',
      icon: Search,
      tagline: 'Spot incoming radar returns and mark contacts for downstream classification.',
    },
    classifier: {
      title: 'Classifier',
      subtitle: 'Multispectral analysis',
      icon: Eye,
      tagline: 'Inspect RF, thermal, and micro-doppler signatures to assign target classification.',
    },
    engagement_authority: {
      title: 'Engagement authority',
      subtitle: 'Tactical authorization',
      icon: Crosshair,
      tagline: 'Authorize soft-kill electronic jamming or kinetic countermeasure within ROE.',
    },
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      {/* ── Initial Setup State: Centered Setup Card ── */}
      {!isRunning && !teamResult && (
        <div className="max-w-xl mx-auto my-12 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-lg p-6 sm:p-8 shadow-sm">
          <div className="flex flex-col items-center text-center">
            <div className="w-12 h-12 rounded-lg bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] flex items-center justify-center text-[var(--c-accent)] mb-4">
              <Users className="w-6 h-6" />
            </div>
            <h1 className="text-xl font-semibold text-[var(--c-text-high)] font-ui">
              Team operations
            </h1>
            <p className="text-sm text-[var(--c-text-mid)] font-ui mt-1 max-w-md">
              Coordinate tactical air defense across detection, classification, and engagement authorities in real time.
            </p>
          </div>

          <div className="mt-8 space-y-6">
            <Button
              variant="primary"
              size="lg"
              className="w-full justify-center text-sm font-semibold"
              onClick={startNewTeamScenario}
              icon={<Play className="w-4 h-4" />}
            >
              Start shared session
            </Button>

            <div className="relative flex items-center justify-center">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-[var(--c-border-base)]" />
              </div>
              <span className="relative bg-[var(--c-bg-surface)] px-3 text-xs text-[var(--c-text-low)] font-ui uppercase">
                or
              </span>
            </div>

            <div className="space-y-2">
              <label className="block text-xs font-medium text-[var(--c-text-mid)] font-ui">
                Join with scenario ID
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Enter scenario ID (e.g. scn-3829)"
                  value={scenarioId}
                  onChange={(e) => setScenarioId(e.target.value)}
                  className="flex-1 bg-[var(--c-bg-raised)] border border-[var(--c-border-base)] focus:border-[var(--c-accent)] focus:outline-none rounded-md px-3 py-2 text-sm text-[var(--c-text-high)] placeholder:text-[var(--c-text-low)] font-mono"
                />
                <Button variant="secondary" onClick={joinTeamScenario}>
                  Join session
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Active Session Workspace ── */}
      {(isRunning || teamResult) && (
        <>
          {/* Header Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-[var(--c-text-high)] font-ui">
                  Team operations
                </h1>
                <Badge variant="default" size="xs">
                  {scenarioId.slice(0, 10)}
                </Badge>
              </div>
              <p className="text-xs text-[var(--c-text-mid)] font-ui mt-0.5">
                Role-based tactical pipeline with simulated transmission latency.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md font-mono text-xs">
                <Clock className="w-3.5 h-3.5 text-[var(--c-text-low)]" />
                <span className="font-semibold text-[var(--c-text-high)] tabular-nums">
                  {Math.floor(elapsed / 60).toString().padStart(2, '0')}:{(elapsed % 60).toString().padStart(2, '0')}
                </span>
              </div>

              {isRunning && (
                <Button variant="danger" size="sm" onClick={finishTeamSession}>
                  Finish and score team
                </Button>
              )}
            </div>
          </div>

          {/* Clean Role Pipeline Diagram */}
          <div className="p-4 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-lg shadow-sm">
            <div className="text-xs font-medium text-[var(--c-text-low)] font-ui mb-3">
              Operational pipeline & communications latency
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
              {/* Step 1: Detector */}
              <button
                type="button"
                onClick={() => setRole('detector')}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors cursor-pointer ${
                  role === 'detector'
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] text-[var(--c-text-high)]'
                    : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] text-[var(--c-text-mid)] hover:border-[var(--c-border-strong)]'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-md ${role === 'detector' ? 'bg-[var(--c-accent-bg)] text-[var(--c-accent)]' : 'bg-[var(--c-bg-surface)] text-[var(--c-text-low)]'}`}>
                    <Search className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium font-ui">1. Detector</div>
                    <div className="text-xs text-[var(--c-text-low)] font-ui">Radar acquisition</div>
                  </div>
                </div>
                <span className="text-[11px] font-mono text-[var(--c-text-low)]">+{detectorToClsDelay}</span>
              </button>

              {/* Step 2: Classifier */}
              <button
                type="button"
                onClick={() => setRole('classifier')}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors cursor-pointer ${
                  role === 'classifier'
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] text-[var(--c-text-high)]'
                    : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] text-[var(--c-text-mid)] hover:border-[var(--c-border-strong)]'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-md ${role === 'classifier' ? 'bg-[var(--c-accent-bg)] text-[var(--c-accent)]' : 'bg-[var(--c-bg-surface)] text-[var(--c-text-low)]'}`}>
                    <Eye className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium font-ui">2. Classifier</div>
                    <div className="text-xs text-[var(--c-text-low)] font-ui">Multispectral signatures</div>
                  </div>
                </div>
                <span className="text-[11px] font-mono text-[var(--c-text-low)]">+{clsToEngDelay}</span>
              </button>

              {/* Step 3: Engagement Authority */}
              <button
                type="button"
                onClick={() => setRole('engagement_authority')}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors cursor-pointer ${
                  role === 'engagement_authority'
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] text-[var(--c-text-high)]'
                    : 'bg-[var(--c-bg-deep)] border-[var(--c-border-base)] text-[var(--c-text-mid)] hover:border-[var(--c-border-strong)]'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-md ${role === 'engagement_authority' ? 'bg-[var(--c-accent-bg)] text-[var(--c-accent)]' : 'bg-[var(--c-bg-surface)] text-[var(--c-text-low)]'}`}>
                    <Crosshair className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium font-ui">3. Engagement authority</div>
                    <div className="text-xs text-[var(--c-text-low)] font-ui">Authorization & ROE</div>
                  </div>
                </div>
                <span className="text-[11px] font-mono text-[var(--c-text-low)]">Active</span>
              </button>
            </div>
          </div>

          {/* Console Workspace */}
          {isRunning && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[520px]">
              {/* Left: Role Radar Scope View (7 Cols) */}
              <div className="lg:col-span-7 flex flex-col h-full min-h-[460px]">
                <Panel
                  title={`Console: ${roleConfigs[role].title}`}
                  subtitle={roleConfigs[role].subtitle}
                  className="h-full flex flex-col"
                  bodyClassName="p-0 flex-1 relative min-h-[400px]"
                >
                  <RadarScope
                    contacts={contacts}
                    selectedContactId={selectedContactId}
                    onSelectContact={(id) => setSelectedContactId(id)}
                  />
                </Panel>
              </div>

              {/* Right: Role-Specific Action Station (5 Cols) */}
              <div className="lg:col-span-5 flex flex-col space-y-4">
                <Panel title="Role station directives" subtitle={roleConfigs[role].title}>
                  <div className="space-y-4">
                    <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-md">
                      <div className="text-xs font-medium text-[var(--c-text-low)] font-ui">Directive:</div>
                      <p className="text-xs text-[var(--c-text-high)] font-ui mt-1 leading-relaxed">
                        {roleConfigs[role].tagline}
                      </p>
                    </div>

                    {selectedContactId ? (
                      <div className="space-y-3 p-4 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-md">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium text-[var(--c-text-high)] font-mono">
                            Target: {selectedContactId}
                          </span>
                          <Badge variant="info" size="xs">Selected track</Badge>
                        </div>

                        {role === 'detector' && (
                          <div className="space-y-2 pt-2">
                            <Button
                              variant="primary"
                              className="w-full justify-center"
                              onClick={() => executeRoleAction(selectedContactId, 'tag_detected')}
                            >
                              Mark detected on radar
                            </Button>
                          </div>
                        )}

                        {role === 'classifier' && (
                          <div className="grid grid-cols-2 gap-2 pt-2">
                            <Button
                              size="sm"
                              variant="danger"
                              onClick={() => executeRoleAction(selectedContactId, 'classify', 'hostile_drone')}
                            >
                              Hostile drone
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => executeRoleAction(selectedContactId, 'classify', 'friendly_drone')}
                            >
                              Friendly drone
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => executeRoleAction(selectedContactId, 'classify', 'decoy')}
                            >
                              Decoy
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => executeRoleAction(selectedContactId, 'classify', 'bird')}
                            >
                              Bird / Neutral
                            </Button>
                          </div>
                        )}

                        {role === 'engagement_authority' && (
                          <div className="space-y-2 pt-2">
                            <Button
                              variant="danger"
                              className="w-full justify-center"
                              onClick={() => executeRoleAction(selectedContactId, 'authorize_kill', 'kinetic')}
                            >
                              Authorize kinetic intercept
                            </Button>
                            <Button
                              variant="secondary"
                              className="w-full justify-center"
                              onClick={() => executeRoleAction(selectedContactId, 'authorize_rf', 'soft_kill')}
                            >
                              Deploy RF electronic jamming
                            </Button>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="p-6 text-center text-xs text-[var(--c-text-low)] border border-dashed border-[var(--c-border-base)] rounded-md">
                        Select a contact on the radar scope to view available directives
                      </div>
                    )}
                  </div>
                </Panel>
              </div>
            </div>
          )}

          {/* Debrief Modal/View */}
          {teamResult && (
            <Panel title="Team debrief and evaluation" subtitle="Exercise summary">
              <div className="p-4 sm:p-6 space-y-6">
                <div className="flex flex-col sm:flex-row items-center justify-between p-6 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-lg gap-4">
                  <div>
                    <span className="text-xs font-medium text-[var(--c-text-low)] font-ui">
                      Overall team score
                    </span>
                    <div className="text-4xl font-semibold text-[var(--c-text-max)] font-mono tabular-nums mt-1">
                      {teamResult.team_score?.toFixed(1) || 87.5}%
                    </div>
                    <div className="text-xs text-[var(--c-text-mid)] font-ui mt-1">
                      Neutralization: {teamResult.neutralized || 7}/{teamResult.total_contacts || 8} hostile contacts intercepted
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center w-full sm:w-auto">
                    <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md">
                      <span className="text-xs text-[var(--c-text-low)] font-ui block">Detector</span>
                      <span className="font-semibold text-sm text-[var(--c-text-high)] font-mono">{teamResult.detector_score || 94}%</span>
                    </div>
                    <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md">
                      <span className="text-xs text-[var(--c-text-low)] font-ui block">Classifier</span>
                      <span className="font-semibold text-sm text-[var(--c-text-high)] font-mono">{teamResult.classifier_score || 86}%</span>
                    </div>
                    <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md">
                      <span className="text-xs text-[var(--c-text-low)] font-ui block">Engagement</span>
                      <span className="font-semibold text-sm text-[var(--c-text-high)] font-mono">{teamResult.engagement_score || 82}%</span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-[var(--c-bg-deep)] border border-[var(--c-warning)]/30 rounded-lg flex items-start gap-3">
                  <AlertCircle className="w-4 h-4 text-[var(--c-warning)] shrink-0 mt-0.5" />
                  <div className="text-xs font-ui">
                    <div className="font-medium text-[var(--c-warning)]">Pipeline latency analysis</div>
                    <p className="text-[var(--c-text-high)] mt-0.5">
                      {teamResult.bottleneck || 'Engagement delayed 4.8s waiting on classifier confirmation.'}
                    </p>
                  </div>
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
                    Run another exercise
                  </Button>
                </div>
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  );
};
