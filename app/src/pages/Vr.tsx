import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useVrSession } from '../components/vr/useVrSession';
import { VrCanvas } from '../components/vr/VrCanvas';
import { ErrorBoundary } from '../ui/ErrorBoundary';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import {
  Glasses,
  Play,
  RotateCcw,
  AlertTriangle,
  Radio,
} from 'lucide-react';

export const VrPage: React.FC = () => {
  const navigate = useNavigate();
  const {
    scenarioId,
    setScenarioId,
    available,
    status,
    error,
    scenario,
    contacts,
    elapsed,
    getElapsed,
  } = useVrSession();

  const formatElapsed = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${String(mins).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms}`;
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'live':
        return <Badge variant="advisory" size="sm">LIVE TELEMETRY</Badge>;
      case 'loading':
        return <Badge variant="info" size="sm">SYNCING...</Badge>;
      case 'empty':
        return <Badge variant="neutral" size="sm">NO ACTIVE SCENARIO</Badge>;
      case 'error':
        return <Badge variant="critical" size="sm">DISCONNECTED</Badge>;
      default:
        return null;
    }
  };

  return (
    <ErrorBoundary fallbackTitle="VR Threat Dome Runtime Error">
      <div className="flex flex-col h-full bg-[var(--c-bg-void)] text-[var(--c-text-high)] font-mono select-none overflow-hidden">
        {/* Tactical Sub-Header Toolbar */}
        <div className="h-12 px-4 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex items-center justify-between shrink-0 z-30">
          <div className="flex items-center gap-3">
            <Glasses className="w-4 h-4 text-[var(--c-accent)]" />
            <div className="flex items-center gap-2">
              <span className="font-bold text-xs tracking-wider text-[var(--c-text-max)]">
                VR THREAT DOME
              </span>
              <span className="text-[10px] text-[var(--c-text-low)] hidden sm:inline">
                [360° SPATIAL TWIN · 1 UNIT = 100M]
              </span>
            </div>
            {getStatusBadge()}
          </div>

          {/* Scenario Selector & Controls */}
          <div className="flex items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-[var(--c-text-low)]">SCENARIO:</span>
              <select
                value={scenarioId || ''}
                onChange={(e) => setScenarioId(e.target.value || null)}
                className="bg-[var(--c-bg-surface)] text-[var(--c-text-high)] border border-[var(--c-border-base)] px-2 py-0.5 rounded text-xs focus:outline-none focus:border-[var(--c-accent)] cursor-pointer"
              >
                {available.length === 0 && !scenarioId && (
                  <option value="">(No active scenarios found)</option>
                )}
                {scenarioId && !available.some((a) => a.scenario_id === scenarioId) && (
                  <option value={scenarioId}>{scenarioId.slice(0, 10)}...</option>
                )}
                {available.map((sc) => (
                  <option key={sc.scenario_id} value={sc.scenario_id}>
                    {sc.scenario_id.slice(0, 8)}... ({sc.contact_count} tracks, {sc.params?.terrain || 'rural'})
                  </option>
                ))}
              </select>
            </div>

            <div className="hidden md:flex items-center gap-1.5 px-2 py-0.5 bg-[var(--c-bg-surface)] border border-[var(--c-border-faint)] rounded text-xs">
              <span className="text-[10px] text-[var(--c-text-low)]">ELAPSED:</span>
              <span className="font-bold text-[var(--c-accent)] tabular-nums">
                {formatElapsed(elapsed)}
              </span>
            </div>

            <Button
              size="xs"
              variant="outline"
              icon={<RotateCcw className="w-3 h-3" />}
              onClick={() => navigate('/trainer?autostart=1')}
            >
              Start In Trainer
            </Button>
          </div>
        </div>

        {/* Viewport Area */}
        <div className="flex-1 relative overflow-hidden">
          {status === 'error' && (
            <div className="absolute top-4 left-4 right-4 z-40 p-3 bg-red-950/90 border border-red-500 rounded flex items-center justify-between text-xs text-red-200">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                <span>{error || 'Unable to stream scenario telemetry from backend.'}</span>
              </div>
              <Button
                size="xs"
                variant="outline"
                onClick={() => setScenarioId(scenarioId)}
              >
                Retry Stream
              </Button>
            </div>
          )}

          {status === 'empty' ? (
            /* Honest Empty State: Never render fake or simulated targets */
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto p-8 rounded-lg border border-dashed border-[var(--c-border-base)] bg-[var(--c-bg-deep)]/60 my-auto">
              <div className="w-14 h-14 rounded-full bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] flex items-center justify-center mb-4 text-[var(--c-text-mid)] shadow-lg">
                <Radio className="w-6 h-6 animate-pulse text-[var(--c-accent)]" />
              </div>
              <h3 className="text-sm font-bold text-[var(--c-text-max)] uppercase tracking-wider mb-2">
                No active scenario
              </h3>
              <p className="text-xs text-[var(--c-text-mid)] mb-6 leading-relaxed">
                Start one in Trainer to stream live multi-spectral radar contacts into the 360° threat dome.
              </p>
              <Button
                variant="primary"
                size="md"
                icon={<Play className="w-4 h-4" />}
                onClick={() => navigate('/trainer?autostart=1')}
              >
                Start one in Trainer
              </Button>
            </div>
          ) : (
            /* 360° Spatial Threat Dome 3D Scene */
            <VrCanvas
              scenario={scenario}
              contacts={contacts}
              elapsed={elapsed}
              getElapsed={getElapsed}
            />
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
};
