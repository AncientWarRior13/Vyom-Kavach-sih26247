import React from 'react';

export interface PanelProps {
  title?: React.ReactNode;
  subtitle?: string;
  statusLed?: 'green' | 'amber' | 'red' | 'blue' | 'none';
  actions?: React.ReactNode;
  cornerTicks?: boolean; // kept for interface compatibility, ignored
  className?: string;
  children: React.ReactNode;
  bodyClassName?: string;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  statusLed = 'none',
  actions,
  className = '',
  bodyClassName = 'p-4',
  children,
}) => {
  const ledColors = {
    green: 'bg-[var(--c-accent)]',
    amber: 'bg-[var(--c-warning)]',
    red: 'bg-[var(--c-critical)]',
    blue: 'bg-[var(--c-info)]',
    none: '',
  };

  return (
    <div
      className={`bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-lg shadow-sm flex flex-col ${className}`}
    >
      {/* Header */}
      {title && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--c-border-base)]">
          <div className="flex items-center gap-2 min-w-0">
            {statusLed !== 'none' && (
              <span className={`w-2 h-2 rounded-full shrink-0 ${ledColors[statusLed]}`} />
            )}
            <h3 className="text-sm font-semibold text-[var(--c-text-high)] font-ui truncate">
              {title}
            </h3>
            {subtitle && (
              <span className="text-xs text-[var(--c-text-low)] font-ui truncate">
                {subtitle}
              </span>
            )}
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0 ml-3">{actions}</div>}
        </div>
      )}

      {/* Body */}
      <div className={`flex-1 ${bodyClassName}`}>{children}</div>
    </div>
  );
};
