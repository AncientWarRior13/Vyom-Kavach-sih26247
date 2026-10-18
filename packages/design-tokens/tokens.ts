/** Vyom Kavach Design Tokens — TypeScript export */

export const colors = {
  bg: {
    void: '#020508',
    deep: '#040c14',
    surface: '#08121e',
    raised: '#0d1b2c',
    overlay: '#112236',
    subtle: '#1a3050',
  },
  border: {
    faint: 'rgba(30, 60, 100, 0.45)',
    base: 'rgba(30, 60, 100, 0.75)',
    strong: 'rgba(50, 100, 160, 0.90)',
  },
  text: {
    max: '#f0f8ff',
    high: '#c8deef',
    mid: '#8aaabf',
    low: '#4e718a',
    ghost: '#2a4258',
  },
  contact: {
    hostile: '#ff3b3b',
    hostileDim: '#cc2828',
    hostileBg: 'rgba(255, 59, 59, 0.10)',
    unknown: '#ffb800',
    unknownBg: 'rgba(255, 184, 0, 0.10)',
    friendly: '#0a84ff',
    friendlyBg: 'rgba(10, 132, 255, 0.10)',
    decoy: '#bf5af2',
    decoyBg: 'rgba(191, 90, 242, 0.10)',
    neutral: '#5a9e8a',
    neutralBg: 'rgba(90, 158, 138, 0.10)',
  },
  accent: '#00e87a',
  accentDim: '#00bb60',
  severity: {
    info: '#38bdf8',
    advisory: '#a3e635',
    warning: '#f59e0b',
    critical: '#ef4444',
  },
} as const;

export const fonts = {
  display: "'Chakra Petch', system-ui, sans-serif",
  ui: "'Geist', system-ui, sans-serif",
  mono: "'Geist Mono', 'JetBrains Mono', monospace",
} as const;

export const spacing = {
  1: '4px', 2: '8px', 3: '12px', 4: '16px',
  5: '20px', 6: '24px', 8: '32px', 10: '40px', 12: '48px',
} as const;

export const radius = {
  xs: '3px', sm: '5px', md: '8px', lg: '12px', xl: '18px', full: '9999px',
} as const;

export const duration = {
  fast: '80ms', base: '160ms', slow: '280ms', appear: '220ms',
} as const;

/** Contact classification → visual style mapping */
export const contactStyle = (cls: string) => {
  switch (cls) {
    case 'hostile_drone':
    case 'hostile_swarm_member': return { color: colors.contact.hostile, bg: colors.contact.hostileBg, glyph: 'triangle' };
    case 'friendly_drone': return { color: colors.contact.friendly, bg: colors.contact.friendlyBg, glyph: 'circle' };
    case 'decoy': return { color: colors.contact.decoy, bg: colors.contact.decoyBg, glyph: 'square' };
    case 'bird': return { color: colors.contact.neutral, bg: colors.contact.neutralBg, glyph: 'circle' };
    default: return { color: colors.contact.unknown, bg: colors.contact.unknownBg, glyph: 'diamond' };
  }
};

/** Severity → color */
export const severityColor = (s: 'info' | 'advisory' | 'warning' | 'critical') =>
  colors.severity[s] ?? colors.severity.info;
