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
} from 'lucide-react';
import { CommandPalette, ShortcutCheatSheet } from './CommandPalette';
import { useModelStatus } from '../api/hooks';
import { VyomKavachLogo } from './VyomKavachLogo';

export const AppShell: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const location = useLocation();

  const { data: modelStatus } = useModelStatus();

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

  const navGroups = [
    {
      title: 'Training',
      items: [
        { to: '/trainer', label: 'Trainer Scope', icon: Crosshair },
        { to: '/vr', label: 'VR Dome', icon: Glasses },
        { to: '/team', label: 'Team Roles', icon: Users },
      ],
    },
    {
      title: 'Monitoring',
      items: [
        { to: '/aar', label: 'AAR Telemetry', icon: BarChart3 },
        { to: '/swarm', label: 'Swarm Sim', icon: Boxes },
        { to: '/instructor', label: 'Instructor', icon: GraduationCap },
      ],
    },
    {
      title: 'Tools',
      items: [
        { to: '/styleguide', label: 'Styleguide', icon: Palette },
      ],
    },
  ];

  // Route label lookup for top breadcrumb
  const routeTitles: Record<string, string> = {
    '/trainer': 'Trainer Scope',
    '/vr': 'VR Dome',
    '/team': 'Team Roles',
    '/aar': 'AAR Telemetry',
    '/swarm': 'Swarm Simulation',
    '/instructor': 'Instructor Console',
    '/styleguide': 'Design System',
  };

  const currentTitle = routeTitles[location.pathname] || 'Operations';

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[var(--c-bg-void)] text-[var(--c-text-high)] font-ui select-none">
      {/* ── Top Bar (48px) ── */}
      <header className="h-[48px] shrink-0 z-40 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex items-center justify-between px-4">
        {/* Left: Branding & Breadcrumb */}
        <div className="flex items-center gap-3">
          <NavLink to="/team" className="flex items-center gap-2 text-inherit no-underline">
            <VyomKavachLogo className="w-5 h-5 drop-shadow-[0_0_6px_rgba(0,229,255,0.4)] shrink-0" />
            <span className="font-logo font-bold text-sm tracking-wide text-[var(--c-text-max)]">
              Vyom Kavach
            </span>
          </NavLink>
          <span className="text-[var(--c-text-low)] text-xs">/</span>
          <span className="text-xs font-medium text-[var(--c-text-mid)] font-ui">
            {currentTitle}
          </span>
        </div>

        {/* Right: Single Status Dot + Keyboard Shortcut */}
        <div className="flex items-center gap-3 text-xs font-ui">
          {/* Operational Status Dot */}
          <div className="flex items-center gap-1.5 text-xs text-[var(--c-text-mid)]">
            <span
              className={`w-2 h-2 rounded-full ${modelStatus?.status === 'error'
                  ? 'bg-[var(--c-critical)]'
                  : 'bg-[var(--c-advisory)]'
                }`}
            />
            <span className="text-xs text-[var(--c-text-low)] hidden sm:inline">
              {modelStatus?.status === 'error' ? 'Service degraded' : 'Operational'}
            </span>
          </div>

          {/* Quick Find (Ctrl+K) */}
          <button
            onClick={() => setPaletteOpen(true)}
            className="flex items-center gap-2 px-2.5 py-1 bg-[var(--c-bg-raised)] hover:bg-[var(--c-bg-overlay)] border border-[var(--c-border-base)] rounded-md text-xs text-[var(--c-text-mid)] hover:text-[var(--c-text-max)] transition-colors cursor-pointer"
          >
            <Command className="w-3.5 h-3.5 text-[var(--c-text-low)]" />
            <span className="text-xs hidden sm:inline">Search</span>
            <kbd className="text-[10px] font-mono px-1 py-0.2 bg-[var(--c-bg-surface)] text-[var(--c-text-low)] border border-[var(--c-border-faint)] rounded">
              Ctrl+K
            </kbd>
          </button>

          {/* Shortcut Help (?) */}
          <button
            onClick={() => setShortcutsOpen(true)}
            title="Keyboard shortcuts (?)"
            className="p-1 text-[var(--c-text-low)] hover:text-[var(--c-text-max)] rounded-md hover:bg-[var(--c-bg-raised)] transition-colors cursor-pointer"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* ── Main Layout (Side Nav + Content) ── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Collapsible Side Rail (220px) */}
        <aside
          className={`shrink-0 bg-[var(--c-bg-surface)] border-r border-[var(--c-border-base)] transition-all duration-150 flex flex-col justify-between min-h-0 ${collapsed ? 'w-14' : 'w-[220px]'
            }`}
        >
          {/* Navigation Links Grouped */}
          <nav className="p-2 space-y-4 overflow-y-auto flex-1 min-h-0">
            {navGroups.map((group) => (
              <div key={group.title} className="space-y-1">
                {!collapsed && (
                  <div className="px-3 pt-2 pb-1 text-[11px] font-medium text-[var(--c-text-low)] uppercase tracking-wider">
                    {group.title}
                  </div>
                )}
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      title={collapsed ? item.label : undefined}
                      className={({ isActive }) =>
                        `flex items-center gap-3 px-3 py-2 rounded-md font-ui text-sm transition-colors cursor-pointer ${isActive
                          ? 'bg-[var(--c-bg-raised)] text-[var(--c-accent)] font-medium'
                          : 'text-[var(--c-text-mid)] hover:bg-[var(--c-bg-raised)]/60 hover:text-[var(--c-text-high)]'
                        } ${collapsed ? 'justify-center px-0' : ''}`
                      }
                    >
                      <Icon className="w-4 h-4 shrink-0" />
                      {!collapsed && <span className="truncate">{item.label}</span>}
                    </NavLink>
                  );
                })}
              </div>
            ))}
          </nav>

          {/* Rail Footer & Collapse Toggle */}
          <div className="p-2 border-t border-[var(--c-border-base)] shrink-0">
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="w-full flex items-center justify-center p-2 text-xs text-[var(--c-text-low)] hover:text-[var(--c-text-high)] rounded-md hover:bg-[var(--c-bg-raised)] transition-colors cursor-pointer"
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <div className="flex items-center justify-between w-full px-2 text-xs font-ui">
                  <span>Collapse sidebar</span>
                  <ChevronLeft className="w-3.5 h-3.5" />
                </div>
              )}
            </button>
          </div>
        </aside>

        {/* Viewport Content */}
        <main
          className={`flex-1 flex flex-col min-h-0 bg-[var(--c-bg-void)] min-w-0 ${location.pathname === '/trainer' ? 'overflow-hidden' : 'overflow-y-auto overflow-x-hidden'
            }`}
        >
          {children}
        </main>
      </div>

      {/* Modals & Overlays */}
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      <ShortcutCheatSheet open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </div>
  );
};
