import React from 'react';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: SelectOption[];
}

export const Select: React.FC<SelectProps> = ({
  label,
  options,
  className = '',
  id,
  ...props
}) => {
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label htmlFor={id} className="text-xs font-mono uppercase tracking-wider text-[var(--c-text-mid)]">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          id={id}
          className={`w-full bg-[var(--c-bg-raised)] text-[var(--c-text-max)] border border-[var(--c-border-base)] rounded-sm px-3 py-1.5 text-xs font-mono appearance-none focus:outline-none focus:border-[var(--c-accent)] transition-colors cursor-pointer ${className}`}
          {...props}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value} className="bg-[var(--c-bg-surface)] text-[var(--c-text-high)]">
              {opt.label}
            </option>
          ))}
        </select>
        <div className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--c-text-low)] text-[10px]">
          ▼
        </div>
      </div>
    </div>
  );
};
