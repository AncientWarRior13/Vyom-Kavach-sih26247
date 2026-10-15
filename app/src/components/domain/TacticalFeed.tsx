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
    { type: 'radar', label: 'Radar X-band', online: sensorOutage !== 'radar', jammed: sensorOutage === 'radar', snr: '24 dB' },
    { type: 'eo_ir', label: 'EO/IR MWIR', online: sensorOutage !== 'eo_ir', jammed: sensorOutage === 'eo_ir', snr: '98%' },
    { type: 'rf', label: 'RF spectrum', online: sensorOutage !== 'rf', jammed: sensorOutage === 'rf', snr: '-78 dBm' },
    { type: 'acoustic', label: 'Acoustic array', online: sensorOutage !== 'acoustic', jammed: sensorOutage === 'acoustic', snr: '18 dB' },
  ];

  return (
    <div className={`flex flex-wrap items-center gap-2 ${className}`}>
      {defaultSensors.map((s) => {
        const isDegraded = s.jammed || !s.online;
        return (
          <div
            key={s.type}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md border text-xs font-ui transition-colors ${
              isDegraded
                ? 'bg-[var(--c-hostile-bg)] border-[var(--c-hostile)]/30 text-[var(--c-hostile)]'
                : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] text-[var(--c-text-high)]'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                isDegraded
                  ? 'bg-[var(--c-hostile)]'
                  : 'bg-[var(--c-advisory)]'
              }`}
            />
            <span className="font-medium">{s.label}</span>
            <span className="text-[11px] font-mono text-[var(--c-text-low)] border-l border-[var(--c-border-base)] pl-2">
              {isDegraded ? 'Offline' : s.snr}
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
  return (
    <div className={`overflow-y-auto space-y-1.5 text-xs font-ui pr-1 ${maxHeight} ${className}`}>
      {events.length === 0 ? (
        <div className="text-[var(--c-text-low)] text-center py-4">No events logged</div>
      ) : (
        events.map((ev) => (
          <div
            key={ev.id}
            className="flex items-start gap-2 py-1.5 px-2 rounded-md bg-[var(--c-bg-deep)]/60 border-l-2 hover:bg-[var(--c-bg-raised)] transition-colors"
            style={{ borderLeftColor: `var(--c-${ev.level === 'success' ? 'advisory' : ev.level === 'critical' ? 'hostile' : ev.level})` }}
          >
            <span className="text-[11px] font-mono text-[var(--c-text-low)] shrink-0 tabular-nums">
              {ev.timestamp}
            </span>
            {ev.source && (
              <span className="text-[10px] font-mono px-1.5 py-0.2 bg-[var(--c-bg-raised)] rounded text-[var(--c-text-mid)] font-medium">
                {ev.source}
              </span>
            )}
            <span className="text-[var(--c-text-high)] flex-1 leading-normal">{ev.message}</span>
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
    <div className={`inline-flex items-center gap-2 px-2.5 py-1 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-md font-ui text-xs ${className}`}>
      <span
        className={`w-2 h-2 rounded-full shrink-0 ${
          isHealthy
            ? 'bg-[var(--c-advisory)]'
            : 'bg-[var(--c-critical)]'
        }`}
      />
      <span className="text-[var(--c-text-high)] font-medium">{status === 'ONLINE' ? 'Online' : status}</span>
      <span className="text-xs font-mono text-[var(--c-text-low)] tabular-nums">{latencyMs}ms</span>
    </div>
  );
};
