import React from 'react';

export interface StatProps {
  label: string;
  value: string | number;
  unit?: string;
  delta?: {
    value: string | number;
    trend: 'up' | 'down' | 'neutral';
  };
  sparkline?: number[];
  color?: string;
  className?: string;
}

export const Stat: React.FC<StatProps> = ({
  label,
  value,
  unit,
  delta,
  sparkline,
  color = 'var(--c-accent)',
  className = '',
}) => {
  // Generate sparkline SVG path if provided
  let sparklinePath = '';
  if (sparkline && sparkline.length > 1) {
    const min = Math.min(...sparkline);
    const max = Math.max(...sparkline);
    const range = max - min || 1;
    const w = 60;
    const h = 20;
    const step = w / (sparkline.length - 1);
    
    sparklinePath = sparkline
      .map((val, idx) => {
        const x = idx * step;
        const y = h - ((val - min) / range) * (h - 4) - 2;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  const deltaColors = {
    up: 'text-[var(--c-accent)]',
    down: 'text-[var(--c-hostile)]',
    neutral: 'text-[var(--c-text-mid)]',
  };

  return (
    <div
      className={`p-3 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-sm flex flex-col justify-between ${className}`}
    >
      <div className="flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-[var(--c-text-mid)]">
        <span>{label}</span>
        {delta && (
          <span className={`text-[10px] font-semibold flex items-center gap-0.5 ${deltaColors[delta.trend]}`}>
            {delta.trend === 'up' && '▲'}
            {delta.trend === 'down' && '▼'}
            {delta.value}
          </span>
        )}
      </div>

      <div className="flex items-baseline justify-between mt-2">
        <div className="flex items-baseline gap-1">
          <span className="font-display text-2xl font-bold tracking-tight text-[var(--c-text-max)] tabular-nums">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-mono text-[var(--c-text-low)]">{unit}</span>
          )}
        </div>

        {sparklinePath && (
          <svg className="w-16 h-5 overflow-visible" viewBox="0 0 60 20">
            <path
              d={sparklinePath}
              fill="none"
              stroke={color}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </div>
    </div>
  );
};
