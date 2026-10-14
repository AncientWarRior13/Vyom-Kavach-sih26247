import React from 'react';

export interface SensorStatus {
  type: 'radar' | 'eo_ir' | 'rf' | 'acoustic';
  label: string;
  online: boolean;
  jammed?: boolean;
  snr?: string;
  latencyMs?: number;
}

export const SensorHealthChips: React.FC<{
  sensors?: SensorStatus[];
  sensorOutage?: string | null;
  className?: string;
}> = ({
  sensorOutage,
  className = '',
}) => {
  const defaultSensors: SensorStatus[] = [
    { type: 'radar', label: 'RADAR X-BAND', online: sensorOutage !== 'radar', jammed: sensorOutage === 'radar', snr: '24dB' },
    { type: 'eo_ir', label: 'EO/IR MWIR', online: sensorOutage !== 'eo_ir', jammed: sensorOutage === 'eo_ir', snr: '98%' },
    { type: 'rf', label: 'RF SPECTRUM', online: sensorOutage !== 'rf', jammed: sensorOutage === 'rf', snr: '-78dBm' },
    { type: 'acoustic', label: 'ACOUSTIC ARRAY', online: sensorOutage !== 'acoustic', jammed: sensorOutage === 'acoustic', snr: '18dB' },
  ];

  return (
    <div className={`flex flex-wrap items-center gap-2 ${className}`}>
      {defaultSensors.map((s) => {
        const isDegraded = s.jammed || !s.online;
        return (
          <div
            key={s.type}
            className={`flex items-center gap-2 px-2.5 py-1 rounded-sm border text-[11px] font-mono transition-colors ${
              isDegraded
                ? 'bg-[var(--c-hostile-bg)] border-[var(--c-hostile)]/60 text-[var(--c-hostile)]'
                : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] text-[var(--c-text-high)]'
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isDegraded
                  ? 'bg-[var(--c-hostile)] shadow-[0_0_6px_var(--c-hostile)] animate-pulse'
                  : 'bg-[var(--c-accent)] shadow-[0_0_6px_var(--c-accent)]'
              }`}
            />
            <span className="font-semibold">{s.label}</span>
            <span className="text-[10px] text-[var(--c-text-low)] border-l border-[var(--c-border-faint)] pl-1.5">
              {isDegraded ? 'JAMMED / OFFLINE' : s.snr}
            </span>
          </div>
        );
      })}
    </div>
  );
};

export interface CombatEvent {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'critical' | 'success';
  message: string;
  source?: string;
}

export const EventLog: React.FC<{
  events: CombatEvent[];
  maxHeight?: string;
  className?: string;
}> = ({ events, maxHeight = 'h-48', className = '' }) => {
  const levelColors = {
    info: 'text-[var(--c-info)] border-[var(--c-info)]',
    warning: 'text-[var(--c-warning)] border-[var(--c-warning)]',
    critical: 'text-[var(--c-critical)] border-[var(--c-critical)]',
    success: 'text-[var(--c-accent)] border-[var(--c-accent)]',
  };

  return (
    <div className={`overflow-y-auto space-y-1.5 font-mono text-xs pr-1 ${maxHeight} ${className}`}>
      {events.length === 0 ? (
        <div className="text-[var(--c-text-low)] text-center py-4">NO TELEMETRY EVENTS</div>
      ) : (
        events.map((ev) => (
          <div
            key={ev.id}
            className="flex items-start gap-2 py-1 px-1.5 rounded-xs bg-[var(--c-bg-deep)]/40 border-l-2 hover:bg-[var(--c-bg-overlay)] transition-colors"
            style={{ borderLeftColor: `var(--c-${ev.level === 'success' ? 'accent' : ev.level})` }}
          >
            <span className="text-[10px] text-[var(--c-text-low)] shrink-0 tabular-nums">
              {ev.timestamp}
            </span>
            {ev.source && (
              <span className="text-[10px] px-1 bg-[var(--c-bg-raised)] rounded text-[var(--c-text-mid)] font-semibold uppercase">
                {ev.source}
              </span>
            )}
            <span className="text-[var(--c-text-high)] flex-1">{ev.message}</span>
          </div>
        ))
      )}
    </div>
  );
};

export const ServerStatusLED: React.FC<{
  status?: string;
  latencyMs?: number;
  className?: string;
}> = ({ status = 'ONLINE', latencyMs = 24, className = '' }) => {
  const isHealthy = status === 'ok' || status === 'ONLINE';

  return (
    <div className={`inline-flex items-center gap-2 px-2 py-1 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs font-mono text-xs ${className}`}>
      <span
        className={`w-2 h-2 rounded-full ${
          isHealthy
            ? 'bg-[var(--c-accent)] shadow-[0_0_8px_var(--c-accent)]'
            : 'bg-[var(--c-critical)] shadow-[0_0_8px_var(--c-critical)] animate-pulse'
        }`}
      />
      <span className="text-[var(--c-text-high)] font-semibold uppercase">{status}</span>
      <span className="text-[10px] text-[var(--c-text-low)] tabular-nums">{latencyMs}ms</span>
    </div>
  );
};
