import React from 'react';

export const Skeleton: React.FC<{ className?: string }> = ({ className = 'h-4 w-full' }) => {
  return (
    <div
      className={`animate-pulse bg-[var(--c-bg-raised)]/70 rounded-xs border border-[var(--c-border-faint)] ${className}`}
    />
  );
};

export const EmptyState: React.FC<{
  title: string;
  description?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}> = ({ title, description, icon, action, className = '' }) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-8 text-center bg-[var(--c-bg-deep)]/40 border border-dashed border-[var(--c-border-base)] rounded-sm ${className}`}
    >
      {icon && <div className="text-3xl text-[var(--c-text-low)] mb-2">{icon}</div>}
      <h4 className="font-display font-semibold text-xs tracking-wider uppercase text-[var(--c-text-max)] mb-1">
        {title}
      </h4>
      {description && (
        <p className="text-xs font-mono text-[var(--c-text-mid)] max-w-sm mb-4">
          {description}
        </p>
      )}
      {action}
    </div>
  );
};

export const ProgressRing: React.FC<{
  value: number; // 0 to 100
  size?: number;
  strokeWidth?: number;
  color?: string;
  label?: string;
}> = ({
  value,
  size = 64,
  strokeWidth = 5,
  color = 'var(--c-accent)',
  label,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const clamped = Math.min(100, Math.max(0, value));
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="var(--c-border-base)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          fill="none"
          className="transition-all duration-300"
        />
      </svg>
      <div className="absolute text-center">
        <span className="font-mono text-xs font-bold text-[var(--c-text-max)] tabular-nums">
          {Math.round(clamped)}%
        </span>
        {label && <div className="text-[9px] font-mono text-[var(--c-text-low)]">{label}</div>}
      </div>
    </div>
  );
};

export interface TimelineItem {
  id: string;
  time: string;
  title: string;
  description?: string;
  status?: 'info' | 'warning' | 'critical' | 'success';
}

export const Timeline: React.FC<{ items: TimelineItem[]; className?: string }> = ({
  items,
  className = '',
}) => {
  const dotColors = {
    info: 'bg-[var(--c-info)] shadow-[0_0_6px_var(--c-info)]',
    warning: 'bg-[var(--c-warning)] shadow-[0_0_6px_var(--c-warning)]',
    critical: 'bg-[var(--c-critical)] shadow-[0_0_6px_var(--c-critical)]',
    success: 'bg-[var(--c-accent)] shadow-[0_0_6px_var(--c-accent)]',
  };

  return (
    <div className={`relative pl-4 space-y-4 border-l border-[var(--c-border-base)] ${className}`}>
      {items.map((item) => (
        <div key={item.id} className="relative group">
          <span
            className={`absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full border-2 border-[var(--c-bg-void)] ${
              dotColors[item.status || 'info']
            }`}
          />
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="font-semibold text-[var(--c-text-max)] uppercase tracking-wider">
              {item.title}
            </span>
            <span className="text-[var(--c-text-low)] tabular-nums">{item.time}</span>
          </div>
          {item.description && (
            <p className="text-xs font-mono text-[var(--c-text-mid)] mt-0.5">
              {item.description}
            </p>
          )}
        </div>
      ))}
    </div>
  );
};
