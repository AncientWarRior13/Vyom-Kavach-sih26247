# PS-247 Drone Threat Trainer — UI/UX Redesign Changelog

## [2.0.0] - Tactical UI/UX Redesign & React Unified Migration

### Architecture & Design System
- **`packages/design-tokens/`**:
  - `tokens.css`: Created unified tactical token system with surface layers (`bg-void`, `bg-deep`, `bg-surface`, `bg-raised`, `bg-overlay`), border strengths, text hierarchy, semantic colors (`hostile`, `unknown`, `friendly`, `decoy`, `neutral`, `accent`), severity scale, 4px spacing, radii, motion easings, and high-contrast / color-blind safe mode media queries.
  - `tokens.ts`: Exported TypeScript tokens and helper mapping functions (`contactStyle(cls)`, `severityColor(s)`).
  - `package.json`: Export map configuration for `@ps247/design-tokens`.
- **`DESIGN.md`**: Created tactical design system documentation, principles, rules of engagement, and guide for adding new screens.

### New React Application (`app/`)
- Scaffolded Vite 6 + React 19 + TypeScript + Tailwind 4 + React Router 6 + TanStack Query 5 + Lucide React.
- **Component Suite (`app/src/ui/`)**:
  - `AppShell`: Top bar with live Zulu clock, DEFCON-3 indicator, AI model LED, quick commands, accessibility toggle, and collapsible side rail.
  - `Panel`: Tactical console panel with corner ticks, title bar, status LED, and action slots.
  - `Stat`: Numeral display + delta trend + inline SVG sparkline.
  - `Badge`: MIL-STD geometric shape encoding (Hostile = `▲`, Unknown = `◆`, Friendly = `●`, Decoy = `□`, Neutral = `○`).
  - `Button`: Primary (accent glow), Secondary, Danger, Ghost, Outline, with `<kbd>` shortcut badge and loading spinner.
  - `Segmented`, `Slider`, `Select`, `Tabs`, `Table`: Dense tactical form & navigation components with keyboard focus rings.
  - `Modal`, `Drawer`, `Toast`, `Tooltip`: Tactical overlays with backdrop blur and escape handling.
  - `CommandPalette` (`Ctrl+K`) & `ShortcutCheatSheet` (`?`).
- **Domain Components (`app/src/components/domain/`)**:
  - `RadarScope`: 2D Canvas radar engine with phosphor decay sweep, concentric range rings in nautical miles/km, 30° radial bearing lines, contact shape glyphs, velocity vectors, history breadcrumbs, zoom/pan controls, and click-to-select target locking.
  - `ContactRow`: Eliminates button clutter by showing actions only when selected.
  - `SensorHealthChips`: Status chips for Radar X-band, EO/IR, RF, and Acoustic channels with jamming/outage indication.
  - `EventLog`: Chronological tactical events feed with severity coloring.
  - `ServerStatusLED`: Heartbeat pulse connected to backend status.
- **Pages**:
  - `/trainer` (`Trainer.tsx`): 3-state workflow: Briefing Stepper > Live Exercise Console with RadarScope & J/K navigation > Debrief Evaluation Scorecard. Supports `?drill=`, `?difficulty=`, `?timeOfDay=`, `?autostart=1`.
  - `/aar` (`Aar.tsx`): Overview KPI row, multi-session trend charts, missed-threat polar heatmap, error taxonomy bars, certification audit, and synchronized session replay drawer.
  - `/instructor` (`Instructor.tsx`): Real-time mirrored trainee radar scope, threat injection toolbox, electronic jamming controls, and one-click stress test macros.
  - `/team` (`Team.tsx`): Distributed crew operations with simulated comms delay pipeline across Detector, Classifier, and Engagement Authority.
  - `/vr` (`Vr.tsx`): 3D spatial threat dome with WebXR headset integration and desktop orbit/look controls.
  - `/swarm` (`Swarm.tsx`): ARES-X autonomous multi-UAV mesh network routing, dynamic relay re-allocation, and node inspector.
  - `/styleguide` (`Styleguide.tsx`): Interactive living styleguide showing all tokens, components, and radar engine in all states.

### Landing Page (`landing/`)
- Rebuilt `landing/src/App.tsx` as a scroll-driven cinematic drone flight through four biomes (Urban > Coastal > Mountain > Rural).
- Implemented `landing/src/world/Drone.tsx`: Procedural quadcopter with 4 diagonal arms, spinning rotor blades, blinking port/starboard navigation LEDs, gimbal optical pod, and third-person chase camera.
- Connected tactical HUD frame with 8 progress rail ticks, flight readouts (ALT, SPD, HDG, BIOME, TIME), and mission briefing launcher.

### Backend & Build Pipeline
- `server.py`: Serves `landing/dist` at `/` and `app/dist` for `/trainer`, `/aar`, `/instructor`, `/team`, `/vr`, `/swarm`, and `/styleguide` with static asset mounting and SPA fallback.
- `start.bat` & `start.sh`: Automatically build `landing/` and `app/` if `dist/` is missing before launching uvicorn.
