import React, { useEffect } from 'react';

export interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title?: React.ReactNode;
  children: React.ReactNode;
  position?: 'left' | 'right';
  width?: string;
}

export const Drawer: React.FC<DrawerProps> = ({
  open,
  onClose,
  title,
  children,
  position = 'right',
  width = 'w-[440px]',
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

  return (
    <div className="fixed inset-0 z-[var(--z-modal)] flex">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-xs transition-opacity animate-in fade-in duration-150"
        onClick={onClose}
      />

      {/* Drawer Panel */}
      <div
        className={`fixed top-0 bottom-0 ${
          position === 'right' ? 'right-0 border-l' : 'left-0 border-r'
        } ${width} bg-[var(--c-bg-surface)] border-[var(--c-border-base)] shadow-xl z-10 flex flex-col`}
      >
        {title && (
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--c-border-base)]">
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
        <div className="flex-1 p-5 overflow-y-auto text-sm font-ui">{children}</div>
      </div>
    </div>
  );
};
