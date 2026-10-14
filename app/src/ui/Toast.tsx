import React, { createContext, useContext, useState, useCallback } from 'react';

export type ToastType = 'info' | 'success' | 'warning' | 'error';

export interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
}

interface ToastContextValue {
  toast: (title: string, message?: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const toast = useCallback((title: string, message?: string, type: ToastType = 'info') => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, title, message, type }]);

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const borderColors: Record<ToastType, string> = {
    info: 'border-[var(--c-info)] text-[var(--c-info)]',
    success: 'border-[var(--c-accent)] text-[var(--c-accent)]',
    warning: 'border-[var(--c-warning)] text-[var(--c-warning)]',
    error: 'border-[var(--c-critical)] text-[var(--c-critical)]',
  };

  const icons: Record<ToastType, string> = {
    info: 'ℹ',
    success: '✔',
    warning: '▲',
    error: '✖',
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto bg-[var(--c-bg-surface)] border-l-4 ${borderColors[t.type]} border-t border-r border-b border-[var(--c-border-strong)] p-3 rounded-sm shadow-xl flex items-start gap-2.5 animate-in slide-in-from-right duration-200`}
          >
            <span className="text-sm font-bold shrink-0">{icons[t.type]}</span>
            <div className="flex-1">
              <div className="text-xs font-mono font-bold uppercase tracking-wider text-[var(--c-text-max)]">
                {t.title}
              </div>
              {t.message && (
                <div className="text-[11px] font-mono text-[var(--c-text-mid)] mt-0.5">
                  {t.message}
                </div>
              )}
            </div>
            <button
              onClick={() => removeToast(t.id)}
              className="text-[var(--c-text-low)] hover:text-[var(--c-text-max)] font-mono text-xs cursor-pointer"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}
