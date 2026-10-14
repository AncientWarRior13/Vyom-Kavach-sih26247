import React from 'react';

export interface SliderProps {
  label?: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (val: number) => void;
  unit?: string;
  className?: string;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  min,
  max,
  step = 1,
  value,
  onChange,
  unit = '',
  className = '',
}) => {
  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      {(label || unit) && (
        <div className="flex items-center justify-between text-xs font-mono">
          {label && <span className="text-[var(--c-text-mid)] uppercase tracking-wider">{label}</span>}
          <span className="text-[var(--c-accent)] font-semibold tabular-nums">
            {value} {unit}
          </span>
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-[var(--c-bg-void)] border border-[var(--c-border-base)] rounded-lg appearance-none cursor-pointer accent-[var(--c-accent)] focus:outline-none"
      />
      <div className="flex justify-between text-[10px] font-mono text-[var(--c-text-low)]">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
};
