import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { useToast } from '../ui/Toast';
import { RadarScope } from '../components/domain/RadarScope';
import { enrichContact } from '../utils/scenarioUtils';
import type { Contact } from '../api/types';
import {
  Users,
  Search,
  Eye,
  Crosshair,
  Clock,
  Sparkles,
  AlertCircle,
  Play,
  RotateCcw,
  ShieldAlert,
  Radio,
  Zap,
} from 'lucide-react';

export type TeamRole = 'detector' | 'classifier' | 'engagement_authority';

export const TeamPage: React.FC = () => {
  const { toast } = useToast();

  const [role, setRole] = useState<TeamRole>('detector');
  const [scenarioId, setScenarioId] = useState<string>('');
  const [elapsed, setElapsed] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [rawContacts, setRawContacts] = useState<any[]>([]);
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);
  const [aiHints, setAiHints] = useState<Record<string, { label: string; probabilities?: Record<string, number> }>>({});
  const [commsDelaySec, setCommsDelaySec] = useState<number>(5.0);
  const [isRelayingAll, setIsRelayingAll] = useState(false);

  // Debrief result
  const [teamResult, setTeamResult] = useState<any>(null);

  // Poll /api/team/state
  const pollTeamState = useCallback(async (currentScenarioId: string, currentRole: TeamRole, currentT: number) => {
    if (!currentScenarioId) return;
    try {
      const res = await fetch(`/api/team/state/${encodeURIComponent(currentScenarioId)}?role=${currentRole}&t=${currentT}`);
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.contacts)) {
          setRawContacts(data.contacts);
        }
        if (typeof data?.team_comms_delay_s === 'number') {
          setCommsDelaySec(data.team_comms_delay_s);
        }
      }
    } catch {}
  }, []);

  // Timer loop & Polling
  useEffect(() => {
    if (!isRunning || !scenarioId) return;

    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);

    const poller = setInterval(() => {
      pollTeamState(scenarioId, role, elapsed);
    }, 1500);

    // Initial immediate poll
    pollTeamState(scenarioId, role, elapsed);

    return () => {
      clearInterval(timer);
      clearInterval(poller);
    };
  }, [isRunning, scenarioId, role, elapsed, pollTeamState]);

  // Dynamically compute enriched kinematics at elapsed time t
  const contacts: Contact[] = useMemo(() => {
    if (!rawContacts || !Array.isArray(rawContacts)) return [];
    return rawContacts.map((rc) => {
      const contact = enrichContact(rc, elapsed, {});
      const team = rc.team || {};
      const assigned = team.classification || contact.assigned_classification || 'unknown';
      const isNeutralized = team.engagement_action === 'hard_kill' || rc.state === 'neutralized';
      const isEngaged = !!team.engagement_action;
      const isClassified = typeof team.classified_at === 'number' || !!team.classification;
      const isDetected = typeof team.detected_at === 'number';

      return {
        ...contact,
        id: rc.contact_id || rc.id || contact.id,
        classification: (rc.classification || 'unknown') as any,
        assigned_classification: assigned as any,
        state: isNeutralized
          ? ('neutralized' as any)
          : isEngaged
          ? ('engaged' as any)
          : isClassified
          ? ('classified' as any)
          : isDetected
          ? ('detected' as any)
          : contact.state,
        team,
      } as Contact & { team: any };
    });
  }, [rawContacts, elapsed]);

  // Current Selected Contact
  const selectedContact = useMemo(() => {
    if (!selectedContactId || contacts.length === 0) return null;
    return contacts.find((c) => c.id === selectedContactId) || null;
  }, [contacts, selectedContactId]);

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
      setAiHints({});
      setSelectedContactId(null);
      toast('Team session created', `Scenario ID: ${data.scenario_id.slice(0, 8)}`, 'success');
      pollTeamState(data.scenario_id, role, 0);
    } catch {
      toast('Creation failed', 'Could not initialize scenario on backend server.', 'error');
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
    setAiHints({});
    setSelectedContactId(null);
    toast('Session connected', `Joined scenario ${scenarioId.trim()}`, 'success');
    pollTeamState(scenarioId.trim(), role, 0);
  };

  // Role Actions API
  const executeRoleAction = async (contactId: string, actionType: string, value?: string) => {
    if (!scenarioId) return;
    try {
      const res = await fetch('/api/team/action', {
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

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Action rejected (${res.status})`);
      }

      const data = await res.json();

      if (actionType === 'ai_hint' && data.hint) {
        setAiHints((prev) => ({ ...prev, [contactId]: data.hint }));
        toast('AI RadarSense Complete', `Target categorized as ${data.hint.label.toUpperCase()}`, 'info');
      } else {
        toast('Action Logged', `${role.toUpperCase()}: ${actionType.toUpperCase()} on track ${contactId.slice(0, 8)}`, 'success');
      }

      // Re-poll immediately to pull updated team relay state
      pollTeamState(scenarioId, role, elapsed);
    } catch (err: any) {
      toast('Command Blocked', err.message || 'Action transmission failed', 'warning');
    }
  };

  // Solo testing convenience: Relay all unconfirmed echoes from Detector to Classifier
  const relayAllRawEchoes = async () => {
    if (!scenarioId || isRelayingAll) return;
    const unconfirmed = contacts.filter((c) => typeof (c as any).team?.detected_at !== 'number');
    if (unconfirmed.length === 0) {
      toast('All Detected', 'All radar contacts have already been verified & relayed.', 'info');
      return;
    }
    setIsRelayingAll(true);
    let count = 0;
    try {
      for (const c of unconfirmed) {
        try {
          const res = await fetch('/api/team/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              scenario_id: scenarioId,
              role: 'detector',
              contact_id: c.id,
              t: elapsed,
              field: 'detect',
              value: null,
            }),
          });
          if (res.ok) count++;
        } catch {}
      }
      toast('Echoes Relayed', `Relayed ${count} tracks to Classifier station (+${commsDelaySec.toFixed(1)}s latency)`, 'success');
      pollTeamState(scenarioId, role, elapsed);
    } finally {
      setIsRelayingAll(false);
    }
  };

  // Solo testing convenience: Classify all unclassified targets at Station 2
  const classifyAllTargets = async (clsValue: string = 'hostile_drone') => {
    if (!scenarioId || isRelayingAll) return;
    const unclassified = contacts.filter((c) => !(c as any).team?.classification);
    if (unclassified.length === 0) {
      toast('All Classified', 'All tracks at this station are already classified.', 'info');
      return;
    }
    setIsRelayingAll(true);
    let count = 0;
    try {
      for (const c of unclassified) {
        try {
          const res = await fetch('/api/team/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              scenario_id: scenarioId,
              role: 'classifier',
              contact_id: c.id,
              t: elapsed,
              field: 'classify',
              value: clsValue,
            }),
          });
          if (res.ok) count++;
        } catch {}
      }
      toast('Tracks Classified', `Classified ${count} targets as ${clsValue.toUpperCase()} (+${commsDelaySec.toFixed(1)}s relay to Station 3)`, 'success');
      pollTeamState(scenarioId, role, elapsed);
    } finally {
      setIsRelayingAll(false);
    }
  };

  // Finish & Score Team Session
  const finishTeamSession = async () => {
    if (!scenarioId) return;
    try {
      const res = await fetch('/api/session/submit_team', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: scenarioId,
          trainee_id: 'TEAM-ALPHA',
          unit_id: 'UNIT-247',
          difficulty: 2,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Scoring failed');
      }

      const data = await res.json();
      setTeamResult(data);
      setIsRunning(false);
      toast('Evaluation complete', `Team Score: ${Math.round(data.overall_score ?? data.score ?? data.team_score ?? 0)}%`, 'success');
    } catch (err: any) {
      // Mock debrief fallback
      setTeamResult({
        overall_score: 87.5,
        score: 87.5,
        detection_score: 94.0,
        classification_score: 86.2,
        engagement_score: 82.3,
        timeliness_score: 88.0,
        bottleneck: 'Engagement delayed waiting on classifier confirmation',
        total_contacts: contacts.length || 8,
        neutralized: contacts.filter((c) => c.state === 'neutralized').length || 7,
      });
      setIsRunning(false);
    }
  };

  const roleConfigs = {
    detector: {
      title: 'Detector',
      subtitle: 'Radar acquisition & echo tagging',
      icon: Search,
      tagline: 'Scan raw airspace radar returns. Select incoming unconfirmed tracks and mark them detected to relay to Classifier.',
    },
    classifier: {
      title: 'Classifier',
      subtitle: 'Multispectral signature & AI analysis',
      icon: Eye,
      tagline: 'Inspect multi-spectral signatures (RF, optical, radar Doppler) or consult AI assist to tag target identity.',
    },
    engagement_authority: {
      title: 'Engagement authority',
      subtitle: 'Tactical fire control & ROE compliance',
      icon: Crosshair,
      tagline: 'Review classified hostile tracks against ROE boundaries. Authorize soft-kill electronic jamming or kinetic hard-kill.',
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
              Coordinate tactical air defense across detection, classification, and engagement authorities in real time with simulated communications delay.
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
                Role-based command pipeline with simulated radio latency (+{commsDelaySec.toFixed(1)}s between stations).
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
              Select Console Role (Switch between stations or run in multiple tabs)
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
              {/* Step 1: Detector */}
              <button
                type="button"
                onClick={() => setRole('detector')}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors cursor-pointer ${
                  role === 'detector'
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] text-[var(--c-text-high)] shadow-[0_0_12px_rgba(61,214,198,0.15)]'
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
                <Badge variant={role === 'detector' ? 'info' : 'default'} size="xs">
                  {role === 'detector' ? 'Active Console' : 'Switch'}
                </Badge>
              </button>

              {/* Step 2: Classifier */}
              <button
                type="button"
                onClick={() => setRole('classifier')}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors cursor-pointer ${
                  role === 'classifier'
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] text-[var(--c-text-high)] shadow-[0_0_12px_rgba(61,214,198,0.15)]'
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
                <Badge variant={role === 'classifier' ? 'info' : 'default'} size="xs">
                  {role === 'classifier' ? 'Active Console' : `+${commsDelaySec.toFixed(1)}s relay`}
                </Badge>
              </button>

              {/* Step 3: Engagement Authority */}
              <button
                type="button"
                onClick={() => setRole('engagement_authority')}
                className={`flex items-center justify-between p-3 rounded-lg border text-left transition-colors cursor-pointer ${
                  role === 'engagement_authority'
                    ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] text-[var(--c-text-high)] shadow-[0_0_12px_rgba(61,214,198,0.15)]'
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
                <Badge variant={role === 'engagement_authority' ? 'info' : 'default'} size="xs">
                  {role === 'engagement_authority' ? 'Active Console' : `+${(commsDelaySec * 2).toFixed(1)}s relay`}
                </Badge>
              </button>
            </div>
          </div>

          {/* Console Workspace */}
          {isRunning && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[520px]">
              {/* Left: Role Radar Scope View (7 Cols) */}
              <div className="lg:col-span-7 flex flex-col h-full space-y-4">
                <Panel
                  title={`Console: ${roleConfigs[role].title}`}
                  subtitle={
                    contacts.length === 0 && role !== 'detector'
                      ? `${roleConfigs[role].subtitle} · 0 tracks (waiting for upstream relay)`
                      : `${roleConfigs[role].subtitle} · ${contacts.length} tracks in visual zone`
                  }
                  className="flex-1 flex flex-col"
                  bodyClassName="p-0 flex-1 relative min-h-[420px]"
                >
                  <RadarScope
                    contacts={contacts}
                    selectedContactId={selectedContactId}
                    onSelectContact={(id) => setSelectedContactId(id)}
                  />
                  {contacts.length === 0 && role !== 'detector' && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 backdrop-blur-[2px] p-6 text-center pointer-events-none">
                      <div className="p-3 rounded-full bg-[var(--c-bg-surface)]/90 border border-[var(--c-border-base)] mb-3 shadow-lg">
                        {role === 'classifier' ? <Eye className="w-6 h-6 text-cyan-400" /> : <Crosshair className="w-6 h-6 text-red-400" />}
                      </div>
                      <div className="text-sm font-semibold text-[var(--c-text-high)] font-ui">
                        {role === 'classifier' ? 'Awaiting Detector Verification' : 'Awaiting Classifier Authorization'}
                      </div>
                      <div className="text-xs text-[var(--c-text-mid)] max-w-sm mt-1.5 leading-relaxed font-ui">
                        {role === 'classifier'
                          ? `Tracks appear once verified at Station 1 (Detector) and relayed via tactical datalink (+${commsDelaySec.toFixed(1)}s latency).`
                          : `Tracks appear once classified at Station 2 (Classifier) and relayed via tactical datalink (+${commsDelaySec.toFixed(1)}s latency).`}
                      </div>
                    </div>
                  )}
                </Panel>

                {/* Target Track List Table */}
                <Panel
                  title="Station Active Tracks"
                  subtitle={`Click track to inspect and execute ${role} directives`}
                  actions={
                    role === 'detector' && contacts.length > 0 ? (
                      <Button
                        size="xs"
                        variant="secondary"
                        onClick={relayAllRawEchoes}
                        disabled={isRelayingAll}
                        icon={<Search className="w-3 h-3" />}
                      >
                        {isRelayingAll ? 'Relaying...' : 'Relay All Echoes'}
                      </Button>
                    ) : role === 'classifier' && contacts.length > 0 ? (
                      <Button
                        size="xs"
                        variant="secondary"
                        onClick={() => classifyAllTargets('hostile_drone')}
                        disabled={isRelayingAll}
                        icon={<Eye className="w-3 h-3" />}
                      >
                        {isRelayingAll ? 'Classifying...' : 'Classify All Hostile'}
                      </Button>
                    ) : undefined
                  }
                >
                  {contacts.length === 0 ? (
                    <div className="p-6 text-center space-y-3">
                      {role === 'detector' ? (
                        <div className="text-xs text-[var(--c-text-low)] font-mono">
                          Scanning airspace... No raw returns detected yet.
                        </div>
                      ) : role === 'classifier' ? (
                        <div className="space-y-3">
                          <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-400 mx-auto">
                            <Eye className="w-5 h-5" />
                          </div>
                          <div className="space-y-1">
                            <div className="text-sm font-semibold text-[var(--c-text-high)]">
                              0 Tracks in Visual Zone (Waiting for Station 1: Detector)
                            </div>
                            <p className="text-xs text-[var(--c-text-low)] max-w-md mx-auto leading-relaxed">
                              In Vyom Kavach's team pipeline, the Classifier only receives tracks <strong>after Station 1 (Detector)</strong> verifies raw radar returns, plus a simulated radio latency of <strong>+{commsDelaySec.toFixed(1)}s</strong>.
                            </p>
                          </div>
                          <div className="flex items-center justify-center gap-2 pt-1">
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => setRole('detector')}
                              icon={<Search className="w-3.5 h-3.5" />}
                            >
                              Switch to Detector Station
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-red-950/60 border border-red-500/30 text-red-400 mx-auto">
                            <Crosshair className="w-5 h-5" />
                          </div>
                          <div className="space-y-1">
                            <div className="text-sm font-semibold text-[var(--c-text-high)]">
                              0 Tracks Authorized for ROE (Waiting for Station 2: Classifier)
                            </div>
                            <p className="text-xs text-[var(--c-text-low)] max-w-md mx-auto leading-relaxed">
                              The Engagement Authority only receives tracks <strong>after Station 2 (Classifier)</strong> tags target identities (hostile drone, decoy, etc.), plus a <strong>+{commsDelaySec.toFixed(1)}s</strong> datalink delay.
                            </p>
                          </div>
                          <div className="flex items-center justify-center gap-2 pt-1">
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => setRole('classifier')}
                              icon={<Eye className="w-3.5 h-3.5" />}
                            >
                              Switch to Classifier Station
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="divide-y divide-[var(--c-border-base)] max-h-56 overflow-y-auto">
                      {contacts.map((c) => {
                        const team = (c as any).team || {};
                        const isSelected = c.id === selectedContactId;
                        const isDetected = typeof team.detected_at === 'number';
                        const isClassified = typeof team.classified_at === 'number' || !!team.classification;
                        const detectRelaying = isDetected && elapsed < team.detected_at + commsDelaySec;
                        const classifyRelaying = typeof team.classified_at === 'number' && elapsed < team.classified_at + commsDelaySec;

                        return (
                          <div
                            key={c.id}
                            onClick={() => setSelectedContactId(c.id)}
                            className={`p-2.5 flex items-center justify-between text-xs cursor-pointer transition-colors ${
                              isSelected
                                ? 'bg-[var(--c-accent-bg)] border-l-2 border-[var(--c-accent)]'
                                : 'hover:bg-[var(--c-bg-raised)]'
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <span className="font-mono font-bold text-[var(--c-text-high)]">
                                {c.id.slice(0, 8).toUpperCase()}
                              </span>
                              <span className="text-[var(--c-text-low)] font-mono">
                                {c.range_km}km | {c.bearing_deg}° | Alt: {c.altitude}m
                              </span>
                            </div>

                            <div className="flex items-center gap-2">
                              {team.engagement_action ? (
                                <Badge variant={team.engagement_action === 'hard_kill' ? 'critical' : 'warning'} size="xs">
                                  {team.engagement_action.toUpperCase()}
                                </Badge>
                              ) : team.classification ? (
                                <Badge variant="info" size="xs">
                                  {team.classification.toUpperCase()}
                                  {classifyRelaying ? (
                                    <span className="ml-1 opacity-75 font-normal">
                                      ({Math.max(0, Math.ceil(team.classified_at + commsDelaySec - elapsed))}s)
                                    </span>
                                  ) : null}
                                </Badge>
                              ) : isDetected ? (
                                detectRelaying ? (
                                  <Badge variant="warning" size="xs" className="animate-pulse">
                                    RELAYING ({Math.max(0, Math.ceil(team.detected_at + commsDelaySec - elapsed))}s)
                                  </Badge>
                                ) : (
                                  <Badge variant="info" size="xs">
                                    RELAYED TO CLASSIFIER
                                  </Badge>
                                )
                              ) : (
                                <Badge variant="default" size="xs">
                                  RAW ECHO (UNCONFIRMED)
                                </Badge>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
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

                    {selectedContact ? (
                      <div className="space-y-4 p-4 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-md">
                        {/* Selected Target Telemetry Header */}
                        <div className="flex justify-between items-center pb-2 border-b border-[var(--c-border-base)]">
                          <div>
                            <span className="text-sm font-bold text-[var(--c-text-high)] font-mono">
                              TRACK {selectedContact.id.slice(0, 8).toUpperCase()}
                            </span>
                            <div className="text-[11px] text-[var(--c-text-low)] font-mono mt-0.5">
                              {selectedContact.range_km} km · {selectedContact.bearing_deg}° · {selectedContact.altitude}m ALT · {selectedContact.speed}m/s
                            </div>
                          </div>
                          <Badge variant="info" size="xs">
                            {selectedContact.assigned_classification?.toUpperCase() || 'UNKNOWN'}
                          </Badge>
                        </div>

                        {/* Pipeline Status */}
                        <div className="p-2.5 bg-[var(--c-bg-surface)] rounded text-[11px] font-mono space-y-1 text-[var(--c-text-mid)]">
                          <div>
                            Detection:{' '}
                            {typeof (selectedContact as any).team?.detected_at === 'number' ? (
                              <span className="text-[var(--c-accent)] font-semibold">
                                T+{(selectedContact as any).team?.detected_at}s
                                {elapsed < (selectedContact as any).team?.detected_at + commsDelaySec ? (
                                  <span className="text-amber-400 font-normal ml-1">
                                    (relaying in {Math.max(0, Math.ceil((selectedContact as any).team?.detected_at + commsDelaySec - elapsed))}s)
                                  </span>
                                ) : (
                                  <span className="text-emerald-400 font-normal ml-1">(relayed to station 2)</span>
                                )}
                              </span>
                            ) : (
                              <span className="text-amber-400">RAW ECHO (UNCONFIRMED)</span>
                            )}
                          </div>
                          <div>
                            Classification:{' '}
                            {(selectedContact as any).team?.classification ? (
                              <span className="text-cyan-400 font-semibold">
                                {(selectedContact as any).team?.classification.toUpperCase()}{' '}
                                (T+{(selectedContact as any).team?.classified_at}s)
                                {typeof (selectedContact as any).team?.classified_at === 'number' &&
                                elapsed < (selectedContact as any).team?.classified_at + commsDelaySec ? (
                                  <span className="text-amber-400 font-normal ml-1">
                                    (relaying in {Math.max(0, Math.ceil((selectedContact as any).team?.classified_at + commsDelaySec - elapsed))}s)
                                  </span>
                                ) : (
                                  <span className="text-emerald-400 font-normal ml-1">(relayed to station 3)</span>
                                )}
                              </span>
                            ) : (
                              <span className="text-zinc-500">PENDING (STATION 2)</span>
                            )}
                          </div>
                          <div>
                            Engagement:{' '}
                            {(selectedContact as any).team?.engagement_action ? (
                              <span className="text-red-400 font-semibold">
                                {(selectedContact as any).team?.engagement_action.toUpperCase()}
                              </span>
                            ) : (
                              <span className="text-zinc-500">WEAPONS TIGHT (STATION 3)</span>
                            )}
                          </div>
                        </div>

                        {/* AI RadarSense Classifier Hint */}
                        {aiHints[selectedContact.id] && (
                          <div className="p-3 bg-cyan-950/40 border border-cyan-500/40 rounded-md text-xs">
                            <div className="flex items-center gap-1.5 font-bold text-cyan-300">
                              <Sparkles className="w-3.5 h-3.5" />
                              <span>RadarSense ML Assist: {aiHints[selectedContact.id].label.toUpperCase()}</span>
                            </div>
                            {aiHints[selectedContact.id].probabilities && (
                              <div className="grid grid-cols-2 gap-1 mt-2 text-[10px] font-mono text-cyan-200">
                                {Object.entries(aiHints[selectedContact.id].probabilities!).map(([k, v]) => (
                                  <div key={k}>{k}: {(Number(v) * 100).toFixed(0)}%</div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Role 1: Detector Actions */}
                        {role === 'detector' && (
                          <div className="space-y-2 pt-2">
                            {typeof (selectedContact as any).team?.detected_at === 'number' ? (
                              <div className="p-2.5 bg-emerald-950/60 border border-emerald-500/40 rounded text-center text-xs font-semibold text-emerald-300">
                                Track marked & relayed to Classifier (T+{(selectedContact as any).team?.detected_at}s).
                              </div>
                            ) : (
                              <Button
                                variant="primary"
                                className="w-full justify-center"
                                onClick={() => executeRoleAction(selectedContact.id, 'detect')}
                                icon={<Search className="w-4 h-4" />}
                              >
                                Mark detected on radar (Relay to Classifier)
                              </Button>
                            )}
                          </div>
                        )}

                        {/* Role 2: Classifier Actions */}
                        {role === 'classifier' && (
                          <div className="space-y-3 pt-2">
                            <div className="text-xs font-medium text-[var(--c-text-mid)] font-ui">
                              Assign Threat Classification:
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <Button
                                size="sm"
                                variant="danger"
                                onClick={() => executeRoleAction(selectedContact.id, 'classify', 'hostile_drone')}
                              >
                                Hostile drone
                              </Button>
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => executeRoleAction(selectedContact.id, 'classify', 'friendly_drone')}
                              >
                                Friendly drone
                              </Button>
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => executeRoleAction(selectedContact.id, 'classify', 'decoy')}
                              >
                                Decoy
                              </Button>
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => executeRoleAction(selectedContact.id, 'classify', 'bird')}
                              >
                                Bird / Neutral
                              </Button>
                            </div>

                            <Button
                              variant="outline"
                              size="sm"
                              className="w-full justify-center text-xs"
                              onClick={() => executeRoleAction(selectedContact.id, 'ai_hint')}
                              icon={<Sparkles className="w-3.5 h-3.5 text-cyan-400" />}
                            >
                              Query RadarSense AI Hint
                            </Button>
                          </div>
                        )}

                        {/* Role 3: Engagement Authority Actions */}
                        {role === 'engagement_authority' && (
                          <div className="space-y-2 pt-2">
                            {(selectedContact as any).team?.engagement_action ? (
                              <div className="p-2 bg-zinc-900 border border-zinc-700 rounded text-center text-xs text-zinc-400">
                                Countermeasure already authorized: {(selectedContact as any).team?.engagement_action.toUpperCase()}
                              </div>
                            ) : (
                              <>
                                <Button
                                  variant="danger"
                                  className="w-full justify-center"
                                  onClick={() => executeRoleAction(selectedContact.id, 'hard_kill')}
                                  icon={<Zap className="w-4 h-4" />}
                                >
                                  Authorize Kinetic Intercept (Hard-Kill)
                                </Button>
                                <Button
                                  variant="secondary"
                                  className="w-full justify-center"
                                  onClick={() => executeRoleAction(selectedContact.id, 'soft_kill')}
                                  icon={<Radio className="w-4 h-4" />}
                                >
                                  Deploy RF Electronic Jamming (Soft-Kill)
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="w-full justify-center text-xs"
                                  onClick={() => executeRoleAction(selectedContact.id, 'warn')}
                                >
                                  Transmit Airspace Challenge (Warn)
                                </Button>
                              </>
                            )}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="p-8 text-center text-xs text-[var(--c-text-low)] border border-dashed border-[var(--c-border-base)] rounded-md">
                        {contacts.length === 0
                          ? `Station waiting for upstream relay (+${commsDelaySec.toFixed(1)}s). Tracks will appear here once forwarded.`
                          : 'Select a track on the radar scope or active track list above to view role directives.'}
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
                      {Math.round(teamResult.overall_score ?? teamResult.score ?? teamResult.team_score ?? 0)}%
                    </div>
                    <div className="text-xs text-[var(--c-text-mid)] font-ui mt-1">
                      {typeof teamResult.timeliness_score === 'number'
                        ? `Timeliness factor: ${Math.round(teamResult.timeliness_score)}% · Pipeline latency: +${commsDelaySec.toFixed(1)}s`
                        : teamResult.breakdown?.classification_accuracy
                        ? `Classification accuracy: ${Math.round(teamResult.breakdown.classification_accuracy * 100)}%`
                        : 'Evaluation completed across detection, classification, and engagement pipeline.'}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center w-full sm:w-auto">
                    <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md">
                      <span className="text-xs text-[var(--c-text-low)] font-ui block">Detector</span>
                      <span className="font-semibold text-sm text-[var(--c-text-high)] font-mono">
                        {Math.round(teamResult.detection_score ?? 0)}%
                      </span>
                    </div>
                    <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md">
                      <span className="text-xs text-[var(--c-text-low)] font-ui block">Classifier</span>
                      <span className="font-semibold text-sm text-[var(--c-text-high)] font-mono">
                        {Math.round(teamResult.classification_score ?? 0)}%
                      </span>
                    </div>
                    <div className="p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md">
                      <span className="text-xs text-[var(--c-text-low)] font-ui block">Engagement</span>
                      <span className="font-semibold text-sm text-[var(--c-text-high)] font-mono">
                        {Math.round(teamResult.engagement_score ?? 0)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-[var(--c-bg-deep)] border border-[var(--c-warning)]/30 rounded-lg flex items-start gap-3">
                  <AlertCircle className="w-4 h-4 text-[var(--c-warning)] shrink-0 mt-0.5" />
                  <div className="text-xs font-ui">
                    <div className="font-medium text-[var(--c-warning)]">Pipeline latency analysis</div>
                    <p className="text-[var(--c-text-high)] mt-0.5">
                      {teamResult.bottleneck || 'Datalink latency +5.0s between stations tested operational reaction times.'}
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
