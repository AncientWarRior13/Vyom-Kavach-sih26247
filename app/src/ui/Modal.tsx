import React, { useEffect } from 'react';

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  children: React.ReactNode;
  actions?: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl';
}

export const Modal: React.FC<ModalProps> = ({
  open,
  onClose,
  title,
  children,
  actions,
  maxWidth = 'md',
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const maxWidthClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-xl',
    xl: 'max-w-3xl',
  }[maxWidth];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/75 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      {/* Modal Dialog */}
      <div
        className={`relative w-full ${maxWidthClasses} bg-[var(--c-bg-surface)] border border-[var(--c-border-strong)] rounded-sm shadow-[0_16px_48px_rgba(0,0,0,0.8)] z-10 flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150`}
      >
        {/* Tactical Corner Accents */}
        <div className="absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 border-[var(--c-accent)] pointer-events-none" />
        <div className="absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 border-[var(--c-accent)] pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 border-[var(--c-border-strong)] pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 border-[var(--c-border-strong)] pointer-events-none" />

        {/* Title Bar */}
        {title && (
          <div className="flex items-center justify-between px-4 py-3 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)]">
            <h3 className="font-display font-semibold text-sm tracking-wider uppercase text-[var(--c-text-max)]">
              {title}
            </h3>
            <button
              onClick={onClose}
              className="text-[var(--c-text-low)] hover:text-[var(--c-text-max)] font-mono text-sm cursor-pointer"
            >
              ✕
            </button>
          </div>
        )}

        {/* Content */}
        <div className="p-4 text-xs font-mono text-[var(--c-text-high)] overflow-y-auto max-h-[75vh]">
          {children}
        </div>

        {/* Footer Actions */}
        {actions && (
          <div className="flex items-center justify-end gap-2 px-4 py-3 bg-[var(--c-bg-deep)] border-t border-[var(--c-border-base)]">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
};
