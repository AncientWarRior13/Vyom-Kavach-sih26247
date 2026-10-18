import React from 'react';

export interface TabItem {
  id: string;
  label: React.ReactNode;
  icon?: React.ReactNode;
  badge?: string | number;
}

export interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onChange,
  className = '',
}) => {
  return (
    <div className={`flex border-b border-[var(--c-border-base)] gap-1 px-1 ${className}`}>
      {tabs.map((tab) => {
        const active = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2 px-3.5 py-2.5 text-sm font-ui font-medium transition-colors relative select-none cursor-pointer ${active
                ? 'text-[var(--c-text-max)] font-semibold'
                : 'text-[var(--c-text-mid)] hover:text-[var(--c-text-high)] hover:bg-[var(--c-bg-raised)]/40 rounded-t-md'
              }`}
          >
            {tab.icon && <span className="shrink-0">{tab.icon}</span>}
            <span>{tab.label}</span>
            {tab.badge !== undefined && (
              <span
                className={`px-1.5 py-0.5 rounded text-xs font-mono ${active
                    ? 'bg-[var(--c-accent-bg)] text-[var(--c-accent)] font-semibold'
                    : 'bg-[var(--c-bg-raised)] text-[var(--c-text-low)]'
                  }`}
              >
                {tab.badge}
              </span>
            )}
            {active && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--c-accent)]" />
            )}
          </button>
        );
      })}
    </div>
  );
};
