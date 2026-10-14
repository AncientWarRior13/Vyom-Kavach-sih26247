import React, { useEffect } from 'react';

export interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  position?: 'left' | 'right';
  width?: string;
  children: React.ReactNode;
}

export const Drawer: React.FC<DrawerProps> = ({
  open,
  onClose,
  title,
  position = 'right',
  width = 'w-96',
  children,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity"
        onClick={onClose}
      />

      {/* Drawer Panel */}
      <div
        className={`fixed top-0 bottom-0 ${
          position === 'right' ? 'right-0 border-l' : 'left-0 border-r'
        } ${width} bg-[var(--c-bg-surface)] border-[var(--c-border-strong)] shadow-2xl z-10 flex flex-col`}
      >
        {title && (
          <div className="flex items-center justify-between px-4 py-3 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)]">
            <h3 className="font-display font-semibold text-xs tracking-wider uppercase text-[var(--c-text-max)]">
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
        <div className="flex-1 p-4 overflow-y-auto text-xs font-mono">{children}</div>
      </div>
    </div>
  );
};
