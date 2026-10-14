import React from 'react';

export interface SegmentedOption<T extends string = string> {
  value: T;
  label: React.ReactNode;
  icon?: React.ReactNode;
}

export interface SegmentedProps<T extends string = string> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  size?: 'xs' | 'sm' | 'md';
  className?: string;
}

export function Segmented<T extends string = string>({
  options,
  value,
  onChange,
  size = 'sm',
  className = '',
}: SegmentedProps<T>) {
  const sizeClasses = {
    xs: 'p-0.5 text-xs',
    sm: 'p-1 text-xs',
    md: 'p-1 text-sm',
  }[size];

  const itemPadding = {
    xs: 'px-2 py-0.5',
    sm: 'px-2.5 py-1',
    md: 'px-3.5 py-1.5',
  }[size];

  return (
    <div
      className={`inline-flex items-center bg-[var(--c-bg-void)] border border-[var(--c-border-base)] rounded-sm ${sizeClasses} ${className}`}
    >
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`flex items-center gap-1.5 font-mono font-medium rounded-xs transition-all duration-150 select-none cursor-pointer ${itemPadding} ${
              active
                ? 'bg-[var(--c-bg-raised)] text-[var(--c-accent)] shadow-[0_1px_3px_rgba(0,0,0,0.5)] border border-[var(--c-border-strong)]'
                : 'text-[var(--c-text-mid)] hover:text-[var(--c-text-max)] border border-transparent'
            }`}
          >
            {opt.icon && <span className="text-[11px]">{opt.icon}</span>}
            <span>{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
