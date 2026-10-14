import React from 'react';

export interface PanelProps {
  title?: React.ReactNode;
  subtitle?: string;
  statusLed?: 'green' | 'amber' | 'red' | 'blue' | 'none';
  actions?: React.ReactNode;
  cornerTicks?: boolean;
  className?: string;
  children: React.ReactNode;
  bodyClassName?: string;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  statusLed = 'none',
  actions,
  cornerTicks = true,
  className = '',
  bodyClassName = 'p-3',
  children,
}) => {
  const ledColors = {
    green: 'bg-[var(--c-accent)] shadow-[0_0_8px_var(--c-accent)]',
    amber: 'bg-[var(--c-warning)] shadow-[0_0_8px_var(--c-warning)]',
    red: 'bg-[var(--c-critical)] shadow-[0_0_8px_var(--c-critical)]',
    blue: 'bg-[var(--c-info)] shadow-[0_0_8px_var(--c-info)]',
    none: '',
  };

  return (
    <div
      className={`relative bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-sm overflow-hidden flex flex-col ${className}`}
    >
      {/* Corner Ticks (tactical aesthetic) */}
      {cornerTicks && (
        <>
          <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-[var(--c-accent)] pointer-events-none z-10" />
          <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-[var(--c-accent)] pointer-events-none z-10" />
          <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-[var(--c-border-strong)] pointer-events-none z-10" />
          <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-[var(--c-border-strong)] pointer-events-none z-10" />
        </>
      )}

      {/* Header */}
      {title && (
        <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--c-border-base)] bg-[var(--c-bg-deep)] select-none">
          <div className="flex items-center gap-2">
            {statusLed !== 'none' && (
              <span className={`w-2 h-2 rounded-full ${ledColors[statusLed]}`} />
            )}
            <h3 className="font-display font-semibold tracking-wider uppercase text-xs text-[var(--c-text-max)]">
              {title}
            </h3>
            {subtitle && (
              <span className="text-[10px] font-mono text-[var(--c-text-low)] tracking-normal">
                [{subtitle}]
              </span>
            )}
          </div>
          {actions && <div className="flex items-center gap-1.5">{actions}</div>}
        </div>
      )}

      {/* Body */}
      <div className={`flex-1 overflow-auto ${bodyClassName}`}>{children}</div>
    </div>
  );
};
