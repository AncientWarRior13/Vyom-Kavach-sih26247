import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  kbd?: string;
  icon?: React.ReactNode;
  loading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'secondary',
  size = 'md',
  kbd,
  icon,
  loading = false,
  children,
  className = '',
  disabled,
  ...props
}) => {
  const base =
    'inline-flex items-center justify-center font-ui font-medium transition-colors duration-150 rounded-md cursor-pointer select-none active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none relative shadow-sm';

  const sizeClasses = {
    xs: 'px-2 py-1 text-xs gap-1.5 min-h-[26px]',
    sm: 'px-3 py-1.5 text-xs gap-2 min-h-[30px]',
    md: 'px-3.5 py-2 text-sm gap-2 min-h-[36px]',
    lg: 'px-5 py-2.5 text-base gap-2.5 min-h-[42px]',
  }[size];

  const variantClasses = {
    primary:
      'bg-[var(--c-accent)] text-[#0B0E13] font-semibold hover:bg-[var(--c-accent-dim)] border border-[var(--c-accent)]',
    secondary:
      'bg-[var(--c-bg-raised)] text-[var(--c-text-high)] border border-[var(--c-border-base)] hover:bg-[var(--c-bg-overlay)] hover:border-[var(--c-border-strong)] hover:text-[var(--c-text-max)]',
    danger:
      'bg-[var(--c-hostile-bg)] text-[var(--c-hostile)] border border-[var(--c-hostile)]/25 hover:bg-[var(--c-hostile)] hover:text-white',
    ghost:
      'bg-transparent text-[var(--c-text-mid)] hover:text-[var(--c-text-high)] hover:bg-[var(--c-bg-raised)] border border-transparent shadow-none',
    outline:
      'bg-transparent text-[var(--c-accent)] border border-[var(--c-accent)]/50 hover:bg-[var(--c-accent-bg)] hover:border-[var(--c-accent)]',
  }[variant];

  return (
    <button
      className={`${base} ${sizeClasses} ${variantClasses} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin shrink-0" />
      ) : (
        icon && <span className="shrink-0">{icon}</span>
      )}
      {children && <span>{children}</span>}
      {kbd && (
        <kbd className="ml-1.5 px-1.5 py-0.5 bg-[var(--c-bg-surface)] text-[var(--c-text-low)] border border-[var(--c-border-base)] rounded text-[10px] uppercase font-mono tracking-wider">
          {kbd}
        </kbd>
      )}
    </button>
  );
};
