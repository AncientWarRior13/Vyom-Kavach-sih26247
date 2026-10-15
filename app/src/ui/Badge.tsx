import React from 'react';

export type BadgeVariant =
  | 'hostile'
  | 'unknown'
  | 'friendly'
  | 'decoy'
  | 'neutral'
  | 'info'
  | 'advisory'
  | 'warning'
  | 'critical'
  | 'default';

export interface BadgeProps {
  variant?: BadgeVariant;
  size?: 'xs' | 'sm' | 'md';
  pulse?: boolean;
  children: React.ReactNode;
  className?: string;
}

const glyphs: Record<string, string> = {
  hostile: '▲',
  unknown: '◆',
  friendly: '●',
  decoy: '□',
  neutral: '○',
  info: 'ℹ',
  advisory: '✦',
  warning: '▲',
  critical: '✖',
  default: '■',
};

export const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  size = 'sm',
  pulse = false,
  children,
  className = '',
}) => {
  const sizeClasses = {
    xs: 'px-1.5 py-0.5 text-[11px] gap-1 font-ui',
    sm: 'px-2 py-0.5 text-xs gap-1.5 font-ui',
    md: 'px-2.5 py-1 text-xs gap-2 font-ui',
  }[size];

  const variantStyles: Record<BadgeVariant, string> = {
    hostile: 'bg-[var(--c-hostile-bg)] text-[var(--c-hostile)] border-[var(--c-hostile)]/30',
    unknown: 'bg-[var(--c-unknown-bg)] text-[var(--c-unknown)] border-[var(--c-unknown)]/30',
    friendly: 'bg-[var(--c-friendly-bg)] text-[var(--c-friendly)] border-[var(--c-friendly)]/30',
    decoy: 'bg-[var(--c-decoy-bg)] text-[var(--c-decoy)] border-[var(--c-decoy)]/30',
    neutral: 'bg-[var(--c-neutral-bg)] text-[var(--c-neutral)] border-[var(--c-neutral)]/30',
    info: 'bg-[var(--c-info)]/10 text-[var(--c-info)] border-[var(--c-info)]/25',
    advisory: 'bg-[var(--c-advisory)]/10 text-[var(--c-advisory)] border-[var(--c-advisory)]/25',
    warning: 'bg-[var(--c-warning)]/10 text-[var(--c-warning)] border-[var(--c-warning)]/25',
    critical: 'bg-[var(--c-critical)]/10 text-[var(--c-critical)] border-[var(--c-critical)]/25',
    default: 'bg-[var(--c-bg-raised)] text-[var(--c-text-mid)] border-[var(--c-border-base)]',
  };

  const glyph = glyphs[variant] || '■';

  return (
    <span
      className={`inline-flex items-center border rounded-md font-medium select-none ${sizeClasses} ${variantStyles[variant]} ${className}`}
    >
      <span className={`text-[9px] shrink-0 ${pulse ? 'animate-pulse' : ''}`}>{glyph}</span>
      <span>{children}</span>
    </span>
  );
};
