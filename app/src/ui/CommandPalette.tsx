import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export interface CommandItem {
  id: string;
  title: string;
  category: string;
  shortcut?: string;
  action: () => void;
}

export const CommandPalette: React.FC<{
  open: boolean;
  onClose: () => void;
}> = ({ open, onClose }) => {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const commands: CommandItem[] = [
    { id: 'trainer', title: 'Tactical Scope & Trainer', category: 'Navigation', shortcut: 'G T', action: () => navigate('/trainer') },
    { id: 'instructor', title: 'Instructor Console & Threat Injector', category: 'Navigation', shortcut: 'G I', action: () => navigate('/instructor') },
    { id: 'team', title: 'Team Multi-Operator Stations', category: 'Navigation', shortcut: 'G M', action: () => navigate('/team') },
    { id: 'aar', title: 'After-Action Review & Telemetry', category: 'Navigation', shortcut: 'G A', action: () => navigate('/aar') },
    { id: 'vr', title: 'VR Tactical Threat Dome', category: 'Navigation', shortcut: 'G V', action: () => navigate('/vr') },
    { id: 'swarm', title: 'Swarm Dynamics & Dispersion', category: 'Navigation', shortcut: 'G S', action: () => navigate('/swarm') },
    { id: 'styleguide', title: 'Design System & Component Styleguide', category: 'System', shortcut: 'G D', action: () => navigate('/styleguide') },
    { id: 'refresh', title: 'Force Telemetry Sensor Refresh', category: 'Actions', shortcut: 'R', action: () => window.location.reload() },
  ];

  const filtered = commands.filter((c) =>
    c.title.toLowerCase().includes(query.toLowerCase()) ||
    c.category.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    if (open) setQuery('');
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-24 p-4">
      <div className="fixed inset-0 bg-black/75 backdrop-blur-xs" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-[var(--c-bg-surface)] border border-[var(--c-border-strong)] rounded-sm shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-100 z-10">
        <div className="flex items-center px-3 py-2 border-b border-[var(--c-border-base)] bg-[var(--c-bg-deep)]">
          <span className="text-[var(--c-accent)] font-mono text-sm mr-2">⌘</span>
          <input
            autoFocus
            type="text"
            placeholder="Type tactical command or navigation destination..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-transparent text-xs font-mono text-[var(--c-text-max)] placeholder-[var(--c-text-low)] focus:outline-none"
          />
          <kbd className="px-1.5 py-0.5 bg-[var(--c-bg-raised)] text-[10px] font-mono border border-[var(--c-border-faint)] rounded text-[var(--c-text-low)]">
            ESC
          </kbd>
        </div>

        <div className="max-h-80 overflow-y-auto p-2">
          {filtered.length === 0 ? (
            <div className="p-4 text-center text-xs font-mono text-[var(--c-text-low)]">
              NO MATCHING COMMANDS
            </div>
          ) : (
            filtered.map((cmd) => (
              <button
                key={cmd.id}
                onClick={() => {
                  cmd.action();
                  onClose();
                }}
                className="w-full flex items-center justify-between p-2 rounded-xs text-xs font-mono text-left hover:bg-[var(--c-bg-raised)] hover:text-[var(--c-text-max)] transition-colors group cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[var(--c-text-low)] uppercase font-semibold">
                    [{cmd.category}]
                  </span>
                  <span className="text-[var(--c-text-high)] group-hover:text-[var(--c-accent)]">
                    {cmd.title}
                  </span>
                </div>
                {cmd.shortcut && (
                  <kbd className="px-1.5 py-0.5 bg-[var(--c-bg-deep)] text-[10px] font-mono border border-[var(--c-border-faint)] rounded text-[var(--c-text-low)]">
                    {cmd.shortcut}
                  </kbd>
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export const ShortcutCheatSheet: React.FC<{
  open: boolean;
  onClose: () => void;
}> = ({ open, onClose }) => {
  if (!open) return null;

  const shortcuts = [
    { key: 'Ctrl + K', desc: 'Open Command Palette' },
    { key: '?', desc: 'Show Tactical Shortcut Sheet' },
    { key: '1 - 4', desc: 'Classify Contact: Hostile / Friendly / Unknown / Decoy' },
    { key: 'R', desc: 'RF Sensor Scan' },
    { key: 'E', desc: 'EO/IR Thermal Scan' },
    { key: 'A', desc: 'Acoustic Signature Scan' },
    { key: 'Space', desc: 'Engage Selected Hostile (Kinetic/RF/Laser)' },
    { key: 'Tab', desc: 'Cycle Radar Contacts' },
    { key: 'Esc', desc: 'Deselect Contact / Close Modal' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="fixed inset-0 bg-black/75 backdrop-blur-xs" onClick={onClose} />
      <div className="relative w-full max-w-md bg-[var(--c-bg-surface)] border border-[var(--c-border-strong)] rounded-sm shadow-2xl p-4 z-10">
        <div className="flex items-center justify-between border-b border-[var(--c-border-base)] pb-2 mb-3">
          <h3 className="font-display font-semibold text-xs tracking-wider uppercase text-[var(--c-text-max)]">
            TACTICAL KEYBOARD CONTROLS
          </h3>
          <button onClick={onClose} className="text-[var(--c-text-low)] hover:text-white font-mono text-sm cursor-pointer">
            ✕
          </button>
        </div>
        <div className="space-y-2 text-xs font-mono">
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center justify-between py-1 border-b border-[var(--c-border-faint)]">
              <span className="text-[var(--c-text-high)]">{s.desc}</span>
              <kbd className="px-2 py-0.5 bg-[var(--c-bg-deep)] border border-[var(--c-border-strong)] rounded text-[11px] text-[var(--c-accent)] font-semibold">
                {s.key}
              </kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
