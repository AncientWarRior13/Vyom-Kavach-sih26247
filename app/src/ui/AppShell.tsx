import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Crosshair,
  Glasses,
  GraduationCap,
  Users,
  BarChart3,
  Boxes,
  Palette,
  ChevronLeft,
  ChevronRight,
  Command,
  HelpCircle,
  Eye,
} from 'lucide-react';
import { CommandPalette, ShortcutCheatSheet } from './CommandPalette';
import { useModelStatus } from '../api/hooks';

export const AppShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [colorBlindMode, setColorBlindMode] = useState(false);
  const [timeStr, setTimeStr] = useState('');
  const location = useLocation();

  const { data: modelStatus } = useModelStatus();

  // Zulu Clock
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toUTCString().slice(17, 25) + 'Z');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  // Global keybindings (Ctrl+K, ?)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((prev) => !prev);
      } else if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        setShortcutsOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const navItems = [
    { to: '/trainer', label: 'Trainer Scope', icon: Crosshair },
    { to: '/vr', label: 'VR Dome', icon: Glasses },
    { to: '/instructor', label: 'Instructor', icon: GraduationCap },
    { to: '/team', label: 'Team Roles', icon: Users },
    { to: '/aar', label: 'AAR Telemetry', icon: BarChart3 },
    { to: '/swarm', label: 'Swarm Sim', icon: Boxes },
    { to: '/styleguide', label: 'Styleguide', icon: Palette },
  ];

  const toggleColorBlind = () => {
    setColorBlindMode((prev) => {
      const next = !prev;
      if (next) {
        document.documentElement.classList.add('high-contrast');
      } else {
        document.documentElement.classList.remove('high-contrast');
      }
      return next;
    });
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--c-bg-void)] text-[var(--c-text-high)] font-ui select-none">
      {/* ── Top Bar ── */}
      <header className="fixed top-0 left-0 right-0 h-[var(--topbar-h)] z-40 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex items-center justify-between px-4">
        {/* Left: Branding & Breadcrumb */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-[var(--c-accent)] rounded-xs shadow-[0_0_8px_var(--c-accent)]" />
            <span className="font-display font-bold text-sm tracking-wider text-[var(--c-text-max)]">
              PS-247
            </span>
          </div>
          <span className="text-[var(--c-border-strong)]">|</span>
          <span className="font-display text-xs text-[var(--c-text-mid)] tracking-widest uppercase hidden md:inline">
            C-UAS TACTICAL THREAT TRAINER
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-xs bg-[var(--c-bg-raised)] text-[var(--c-text-low)] border border-[var(--c-border-faint)]">
            {location.pathname.replace('/', '').toUpperCase() || 'HOME'}
          </span>
        </div>

        {/* Right: Operational Status, Zulu Clock, Actions */}
        <div className="flex items-center gap-4 text-xs font-mono">
          {/* DEFCON Readiness */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
            <span className="text-[10px] text-[var(--c-text-low)]">STATUS:</span>
            <span className="text-[var(--c-accent)] font-semibold tracking-wider">DEFCON-3 ACTIVE</span>
          </div>

          {/* Model Status LED */}
          <div className="flex items-center gap-1.5 px-2 py-1 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                modelStatus?.status === 'ok' || modelStatus?.weights_loaded !== false
                  ? 'bg-[var(--c-accent)] shadow-[0_0_8px_var(--c-accent)]'
                  : 'bg-[var(--c-warning)] shadow-[0_0_8px_var(--c-warning)] animate-pulse'
              }`}
            />
            <span className="text-[10px] text-[var(--c-text-mid)] hidden lg:inline">
              AI ENGINE: {modelStatus?.status || 'ONLINE'}
            </span>
          </div>

          {/* Zulu Time */}
          <div className="font-bold text-[var(--c-text-max)] tabular-nums tracking-widest hidden md:inline">
            {timeStr}
          </div>

          {/* Tools & Palette */}
          <div className="flex items-center gap-1">
            <button
              onClick={toggleColorBlind}
              title="Toggle High-Contrast / Color-Blind Safe Mode"
              className={`p-1.5 rounded-xs border transition-colors cursor-pointer ${
                colorBlindMode
                  ? 'bg-[var(--c-accent-bg)] text-[var(--c-accent)] border-[var(--c-accent)]'
                  : 'text-[var(--c-text-low)] hover:text-[var(--c-text-max)] border-transparent hover:border-[var(--c-border-base)]'
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={() => setPaletteOpen(true)}
              className="flex items-center gap-1.5 px-2 py-1 bg-[var(--c-bg-raised)] hover:bg-[var(--c-bg-overlay)] border border-[var(--c-border-base)] rounded-xs text-[var(--c-text-mid)] hover:text-[var(--c-text-max)] cursor-pointer"
            >
              <Command className="w-3 h-3 text-[var(--c-accent)]" />
              <span className="text-[10px] hidden sm:inline">Ctrl+K</span>
            </button>

            <button
              onClick={() => setShortcutsOpen(true)}
              title="Keyboard Shortcuts (?)"
              className="p-1.5 text-[var(--c-text-low)] hover:text-[var(--c-text-max)] border border-transparent hover:border-[var(--c-border-base)] rounded-xs cursor-pointer"
            >
              <HelpCircle className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Layout (Side Nav + Content) ── */}
      <div className="flex flex-1 pt-[var(--topbar-h)] overflow-hidden">
        {/* Collapsible Side Rail */}
        <aside
          className={`shrink-0 bg-[var(--c-bg-surface)] border-r border-[var(--c-border-base)] transition-all duration-200 flex flex-col justify-between ${
            collapsed ? 'w-14' : 'w-52'
          }`}
        >
          {/* Navigation Links */}
          <nav className="p-2 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-xs font-mono text-xs transition-colors group cursor-pointer ${
                      isActive
                        ? 'bg-[var(--c-bg-overlay)] text-[var(--c-accent)] border-l-2 border-l-[var(--c-accent)] font-semibold shadow-xs'
                        : 'text-[var(--c-text-mid)] hover:bg-[var(--c-bg-raised)] hover:text-[var(--c-text-max)]'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </NavLink>
              );
            })}
          </nav>

          {/* Rail Footer & Collapse Toggle */}
          <div className="p-2 border-t border-[var(--c-border-base)] bg-[var(--c-bg-deep)]">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="w-full flex items-center justify-center p-1.5 text-[var(--c-text-low)] hover:text-[var(--c-text-max)] rounded-xs hover:bg-[var(--c-bg-raised)] cursor-pointer"
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <div className="flex items-center justify-between w-full px-2 text-[10px] font-mono">
                  <span>COLLAPSE RAIL</span>
                  <ChevronLeft className="w-3.5 h-3.5" />
                </div>
              )}
            </button>
          </div>
        </aside>

        {/* Viewport Content */}
        <main className="flex-1 overflow-auto bg-[var(--c-bg-void)]">
          {children}
        </main>
      </div>

      {/* Modals & Overlays */}
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      <ShortcutCheatSheet open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </div>
  );
};
