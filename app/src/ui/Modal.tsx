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
  // ESC key handler
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
    lg: 'max-w-lg',
    xl: 'max-w-2xl',
  }[maxWidth];

  return (
    <div className="fixed inset-0 z-[var(--z-modal)] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-150"
        onClick={onClose}
      />

      {/* Modal Dialog */}
      <div
        className={`relative w-full ${maxWidthClasses} bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-lg shadow-lg z-10 flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150`}
      >
        {/* Title Bar */}
        {title && (
          <div className="flex items-center justify-between px-5 py-3.5 bg-[var(--c-bg-surface)] border-b border-[var(--c-border-base)]">
            <h3 className="font-semibold text-sm text-[var(--c-text-high)] font-ui">
              {title}
            </h3>
            <button
              onClick={onClose}
              className="text-[var(--c-text-low)] hover:text-[var(--c-text-high)] p-1 rounded-md hover:bg-[var(--c-bg-raised)] cursor-pointer text-xs"
            >
              ✕
            </button>
          </div>
        )}

        {/* Content */}
        <div className="p-5 text-sm font-ui text-[var(--c-text-high)] overflow-y-auto max-h-[75vh]">
          {children}
        </div>

        {/* Footer Actions */}
        {actions && (
          <div className="flex items-center justify-end gap-2.5 px-5 py-3.5 bg-[var(--c-bg-deep)]/50 border-t border-[var(--c-border-base)]">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
};
