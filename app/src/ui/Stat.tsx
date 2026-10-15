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
    up: 'text-[var(--c-advisory)]',
    down: 'text-[var(--c-critical)]',
    neutral: 'text-[var(--c-text-mid)]',
  };

  return (
    <div
      className={`p-4 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-lg shadow-sm flex flex-col justify-between ${className}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-[var(--c-text-mid)] font-ui">{label}</span>
        {delta && (
          <span className={`text-xs font-medium flex items-center gap-1 font-mono tabular-nums ${deltaColors[delta.trend]}`}>
            {delta.trend === 'up' && '↑'}
            {delta.trend === 'down' && '↓'}
            {delta.value}
          </span>
        )}
      </div>

      <div className="flex items-baseline justify-between mt-2">
        <div className="flex items-baseline gap-1.5">
          <span className="font-mono text-2xl font-semibold tracking-tight text-[var(--c-text-max)] tabular-nums">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-ui text-[var(--c-text-low)]">{unit}</span>
          )}
        </div>

        {sparklinePath && (
          <svg className="w-16 h-5 overflow-visible shrink-0 ml-2" viewBox="0 0 60 20">
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
