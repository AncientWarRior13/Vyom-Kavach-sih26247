# Vyom Kavach — Design System & Architecture

This document establishes the UI/UX architecture, visual grammar, token hierarchy, and guidelines for adding new tactical screens to the VyomKavach platform.

---

## 1. Core Principles

1. **Dark Tactical Operations Center (TOC) Aesthetic**
   - High information density without visual clutter.
   - Contrast ratio WCAG AA compliant on all text and critical indicators.
   - Void/Deep background layering (`#020508` void, `#08121e` surface, `#0d1b2c` raised).

2. **Never Encode State by Color Alone**
   - In tactical C-UAS environments, target classification must never rely solely on color.
   - Always pair color with standardized MIL-STD geometric shapes and glyphs:
     - **Hostile**: Red + Upward Triangle (`▲`)
     - **Unknown**: Amber + Diamond (`◆`)
     - **Friendly**: Blue + Circle (`●`)
     - **Decoy**: Violet + Hollow Square (`□`)
     - **Neutral / Environmental**: Muted Teal + Ring (`○`)

3. **Strict Information Hierarchy & Progressive Disclosure**
   - Do NOT repeat 7+ action buttons on every card.
   - Only the **SELECTED** contact displays detailed sensor scans and engagement commands.
   - Unselected rows display clear identification, range, bearing, altitude, and speed.

4. **Zero Hardcoded Styles**
   - Always consume tokens from `@vyomkavach/design-tokens` (`tokens.css` or `tokens.ts`).

---

## 2. Token Architecture (`packages/design-tokens/`)

| Token Layer | Variable | Tailwind Class | Value / Role |
|---|---|---|---|
| Background Void | `--c-bg-void` | `bg-[var(--c-bg-void)]` | `#020508` (Deepest canvas) |
| Background Deep | `--c-bg-deep` | `bg-[var(--c-bg-deep)]` | `#040c14` (Header/Rails) |
| Surface | `--c-bg-surface` | `bg-[var(--c-bg-surface)]` | `#08121e` (Panels/Cards) |
| Raised | `--c-bg-raised` | `bg-[var(--c-bg-raised)]` | `#0d1b2c` (Buttons/Inputs) |
| Overlay | `--c-bg-overlay` | `bg-[var(--c-bg-overlay)]` | `#112236` (Hover/Selection) |
| Accent | `--c-accent` | `text-[var(--c-accent)]` | `#00e87a` (Radar phosphor green) |
| Hostile | `--c-hostile` | `text-[var(--c-hostile)]` | `#ff3b3b` (Direct threat) |
| Unknown | `--c-unknown` | `text-[var(--c-unknown)]` | `#ffb800` (Unclassified contact) |
| Friendly | `--c-friendly` | `text-[var(--c-friendly)]` | `#0a84ff` (Allied asset) |
| Decoy | `--c-decoy` | `text-[var(--c-decoy)]` | `#bf5af2` (Electronic countermeasure) |

### Typography

- **Headings & Badges**: `'Chakra Petch', system-ui, sans-serif` (`font-display`)
- **UI Chrome & Body**: `'Geist', system-ui, sans-serif` (`font-ui`)
- **Telemetry & Numbers**: `'Geist Mono', monospace` with `tabular-nums` (`font-mono`)

### Spacing Scale

Base 4px grid:
- `4px` (`--sp-1`), `8px` (`--sp-2`), `12px` (`--sp-3`), `16px` (`--sp-4`), `20px` (`--sp-5`), `24px` (`--sp-6`), `32px` (`--sp-8`).

---

## 3. UI Component Suite (`app/src/ui/`)

- **`AppShell`**: Global top bar (DEFCON status, Zulu clock, AI model LED, quick commands, accessibility toggle) + collapsible side rail.
- **`Panel`**: Tactical container with corner ticks, title bar, action slots, and optional status LED.
- **`Stat`**: Metric numeral + delta trend + inline SVG sparkline.
- **`Button`**: Tactical button variants (`primary`, `secondary`, `danger`, `outline`, `ghost`) with `<kbd>` shortcut badge.
- **`Badge`**: Shape-encoded classification and severity chips.
- **`Segmented`**: Multi-state pill switcher.
- **`Slider`**: Numeric range slider with live HUD readout.
- **`Select`**: Tactical dropdown with custom chevron and dark theme.
- **`Tabs`**: Tab strip with active glowing neon bar.
- **`Table`**: Dense data table with sticky header and column sorting.
- **`Modal` & `Drawer`**: Dialogs with backdrop blur and Esc handling.
- **`Toast`**: Combat event alert system (`useToast`).
- **`CommandPalette`**: Global `Ctrl+K` navigation and tactical command palette.
- **`ShortcutCheatSheet`**: Tactical hotkey cheat sheet triggered by `?`.

---

## 4. Domain Components (`app/src/components/domain/`)

- **`RadarScope`**: 2D HTML5 Canvas radar engine with phosphor decay sweep, range rings in nautical miles/km, 30° radial bearing lines, contact shape glyphs, velocity vectors, history breadcrumbs, zoom/pan, and contact click selection.
- **`ContactRow`**: Compact contact card with progressive action disclosure.
- **`SensorHealthChips`**: Real-time status for Radar, EO/IR, RF, and Acoustic channels.
- **`EventLog`**: Real-time scrolling combat log.
- **`ServerStatusLED`**: Heartbeat pulse connected to `/api/model/status`.

---

## 5. How to Add a New Screen

1. Create a page component in `app/src/pages/MyScreen.tsx`.
2. Wrap content inside standard `Panel` and `Stat` components.
3. Add the route in `app/src/App.tsx`:
   ```tsx
   <Route path="/myscreen" element={<MyScreen />} />
   ```
4. Add the navigation item in `app/src/ui/AppShell.tsx`:
   ```tsx
   { to: '/myscreen', label: 'My Screen', icon: MyIcon }
   ```
5. Add the FastAPI route in `server.py`:
   ```python
   @app.get("/myscreen")
   def myscreen_page():
       return serve_app_or_fallback(...)
   ```
6. Rebuild using `npm run build` inside `app/`.
