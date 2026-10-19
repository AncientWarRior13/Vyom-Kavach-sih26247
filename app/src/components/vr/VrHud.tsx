import React, { useState, useRef, useEffect } from 'react';
import type { Contact } from '../../api/types';
import { Button } from '../../ui/Button';
import { Badge } from '../../ui/Badge';
import { vrAudio } from './VrAudio';
import {
  Glasses,
  Smartphone,
  Monitor,
  Maximize2,
  Minimize2,
  Compass,
  Volume2,
  VolumeX,
  AlertTriangle,
  ShieldAlert,
  Crosshair,
  Radio,
  Zap,
} from 'lucide-react';

export interface VrHudProps {
  scenario: any;
  contacts: Contact[];
  selectedContact: Contact | null;
  elapsed: number;
  viewMode: 'desktop' | 'splitscreen' | 'immersive';
  webXrSupported: boolean;
  webXrReason?: string;
  hasGyro: boolean;
  useGyro: boolean;
  dwellProgress: number; // 0 to 1 for gaze lock
  isMuted: boolean;
  onToggleMute: () => void;
  onSelectMode: (mode: 'desktop' | 'splitscreen' | 'immersive') => void;
  onToggleFullscreen: () => void;
  isFullscreen: boolean;
  onToggleGyro: () => void;
  onRecenterGyro: () => void;
  onSelectContact: (id: string | null) => void;
  onExecuteEngagement: (action: 'warn' | 'soft_kill' | 'hard_kill') => void;
}

export const VrHud: React.FC<VrHudProps> = ({
  scenario,
  contacts,
  selectedContact,
  elapsed,
  viewMode,
  webXrSupported,
  webXrReason,
  hasGyro,
  useGyro,
  dwellProgress,
  isMuted,
  onToggleMute,
  onSelectMode,
  onToggleFullscreen,
  isFullscreen,
  onToggleGyro,
  onRecenterGyro,
  onSelectContact,
  onExecuteEngagement,
}) => {
  // Hard-kill hold-to-confirm timer state (600ms)
  const [holdingHardKill, setHoldingHardKill] = useState(false);
  const [holdProgress, setHoldProgress] = useState(0); // 0 to 1
  const holdStartTimeRef = useRef<number | null>(null);
  const holdAnimRef = useRef<number | null>(null);

  const startHardKillHold = () => {
    if (isHardKillBlocked) return;
    setHoldingHardKill(true);
    holdStartTimeRef.current = performance.now();

    const checkHold = () => {
      if (!holdStartTimeRef.current) return;
      const elapsedMs = performance.now() - holdStartTimeRef.current;
      const prog = Math.min(1, elapsedMs / 600);
      setHoldProgress(prog);

      if (prog >= 1) {
        setHoldingHardKill(false);
        setHoldProgress(0);
        holdStartTimeRef.current = null;
        onExecuteEngagement('hard_kill');
      } else {
        holdAnimRef.current = requestAnimationFrame(checkHold);
      }
    };

    holdAnimRef.current = requestAnimationFrame(checkHold);
  };

  const cancelHardKillHold = () => {
    setHoldingHardKill(false);
    setHoldProgress(0);
    if (holdStartTimeRef.current) {
      holdStartTimeRef.current = null;
    }
    if (holdAnimRef.current) {
      cancelAnimationFrame(holdAnimRef.current);
      holdAnimRef.current = null;
    }
  };

  // Check if selected target violates No-Engage Zone or Max Range
  const roe = scenario?.roe_card || {};
  const maxHardKillRangeM = Math.max(2500, roe.max_hard_kill_range_m || 2500);
  const maxSoftKillRangeM = Math.max(3500, roe.max_soft_kill_range_m || 3500);
  const isWeaponsHold = Boolean(roe.weapons_hold);

  let targetRangeM = 0;
  let isInsideNoEngageZone = false;
  let violatingZoneName = '';

  if (selectedContact) {
    targetRangeM = Math.round(Math.hypot(selectedContact.x, selectedContact.y));
    const zones = scenario?.no_engage_zones || [];
    for (const z of zones) {
      const zx = z.center?.[0] || 0;
      const zy = z.center?.[1] || 0;
      const zr = z.radius_m || 300;
      const distToZoneCenter = Math.hypot(selectedContact.x - zx, selectedContact.y - zy);
      if (distToZoneCenter <= zr) {
        isInsideNoEngageZone = true;
        violatingZoneName = z.name || 'CIVILIAN AIRWAY / DEFENDED ASSET';
        break;
      }
    }
  }

  const isHardKillBeyondRange = targetRangeM > maxHardKillRangeM;
  const isSoftKillBeyondRange = targetRangeM > maxSoftKillRangeM;
  // Weapons Hold and civilian No-Engage Zones strictly block weapons. Long-range shows an advisory but allows firing.
  const isHardKillBlocked = isWeaponsHold || isInsideNoEngageZone;

  return (
    <div className="absolute inset-0 pointer-events-none font-mono flex flex-col justify-between p-4 z-20">
      {/* ── Top Bar Overlay: Status, Badges & Mode Switcher ── */}
      <div className="flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-3">
          {/* Live Scenario Badge */}
          <div className="pointer-events-auto px-3 py-1.5 bg-black/80 backdrop-blur-md border border-[var(--c-border-base)] rounded text-xs flex items-center gap-2 text-white shadow-xl">
            <span className="w-2 h-2 rounded-full bg-[var(--c-accent)] animate-pulse" />
            <span className="font-bold text-[var(--c-accent)] tracking-wider">
              LIVE: {scenario?.scenario_id ? scenario.scenario_id.slice(0, 8).toUpperCase() : 'SEARCHING'}
            </span>
            <span className="text-[10px] text-[var(--c-text-mid)]">|</span>
            <span className="text-[10px] text-white">TRACKS: <strong className="text-[var(--c-accent)]">{contacts.length}</strong></span>
            <span className="text-[10px] px-1.5 py-0.5 bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 font-bold rounded">ALT ×3</span>
          </div>

          {/* Sensor Outage Banner */}
          {scenario?.sensor_outage && (
            <div className="px-3 py-1.5 bg-amber-950/85 border border-amber-500/60 rounded text-xs font-bold text-amber-200 flex items-center gap-1.5 animate-pulse shadow-xl">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>SENSOR OUTAGE: {String(scenario.sensor_outage).toUpperCase()} JAMMED</span>
            </div>
          )}

          {/* ROE Weapons Hold Banner */}
          {isWeaponsHold && (
            <div className="px-3 py-1.5 bg-red-950/90 border border-red-500 rounded text-xs font-bold text-red-200 flex items-center gap-1.5 shadow-xl">
              <ShieldAlert className="w-3.5 h-3.5 text-red-400" />
              <span>ROE: WEAPONS HOLD ACTIVE</span>
            </div>
          )}
        </div>

        {/* Mode Switcher & Tools */}
        <div className="pointer-events-auto flex items-center gap-2 bg-black/80 backdrop-blur-md p-1.5 border border-[var(--c-border-base)] rounded shadow-xl">
          {/* Desktop Mode */}
          <button
            type="button"
            onClick={() => onSelectMode('desktop')}
            className={`px-2.5 py-1 rounded text-xs font-bold flex items-center gap-1.5 cursor-pointer transition-colors ${
              viewMode === 'desktop'
                ? 'bg-[var(--c-accent)] text-black'
                : 'text-[var(--c-text-mid)] hover:text-white hover:bg-[var(--c-bg-raised)]'
            }`}
          >
            <Monitor className="w-3.5 h-3.5" />
            <span>Desktop</span>
          </button>

          {/* Split-Screen (Cardboard VR) */}
          <button
            type="button"
            onClick={() => onSelectMode('splitscreen')}
            className={`px-2.5 py-1 rounded text-xs font-bold flex items-center gap-1.5 cursor-pointer transition-colors ${
              viewMode === 'splitscreen'
                ? 'bg-[var(--c-accent)] text-black'
                : 'text-[var(--c-text-mid)] hover:text-white hover:bg-[var(--c-bg-raised)]'
            }`}
          >
            <Smartphone className="w-3.5 h-3.5" />
            <span>Split-Screen</span>
          </button>

          {/* Immersive WebXR Headset */}
          <button
            type="button"
            disabled={!webXrSupported}
            onClick={() => onSelectMode('immersive')}
            title={webXrSupported ? 'Launch Immersive WebXR' : webXrReason || 'No WebXR headset detected'}
            className={`px-2.5 py-1 rounded text-xs font-bold flex items-center gap-1.5 transition-colors ${
              viewMode === 'immersive'
                ? 'bg-[var(--c-accent)] text-black'
                : webXrSupported
                ? 'text-[var(--c-text-mid)] hover:text-white hover:bg-[var(--c-bg-raised)] cursor-pointer'
                : 'text-zinc-600 bg-zinc-900/50 cursor-not-allowed'
            }`}
          >
            <Glasses className="w-3.5 h-3.5" />
            <span>Immersive VR</span>
          </button>

          <div className="w-[1px] h-4 bg-white/20 mx-1" />

          {/* Gyro Look Toggle (for Mobile) */}
          {hasGyro && (
            <>
              <button
                type="button"
                onClick={onToggleGyro}
                title={useGyro ? 'Gyro Look Active' : 'Enable Gyroscope Look'}
                className={`p-1.5 rounded cursor-pointer ${
                  useGyro ? 'text-[var(--c-accent)] bg-emerald-950/50' : 'text-[var(--c-text-mid)] hover:text-white'
                }`}
              >
                <Compass className="w-4 h-4" />
              </button>
              {useGyro && (
                <button
                  type="button"
                  onClick={onRecenterGyro}
                  className="px-1.5 py-0.5 text-[10px] text-white bg-zinc-800 hover:bg-zinc-700 rounded cursor-pointer"
                >
                  Recenter
                </button>
              )}
            </>
          )}

          {/* Fullscreen Toggle */}
          <button
            type="button"
            onClick={onToggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
            className="p-1.5 text-[var(--c-text-mid)] hover:text-white rounded cursor-pointer"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>

          {/* Mute Audio Toggle */}
          <button
            type="button"
            onClick={onToggleMute}
            title={isMuted ? 'Unmute Tactical Audio' : 'Mute Audio'}
            className="p-1.5 text-[var(--c-text-mid)] hover:text-white rounded cursor-pointer"
          >
            {isMuted ? <VolumeX className="w-4 h-4 text-amber-400" /> : <Volume2 className="w-4 h-4 text-[var(--c-accent)]" />}
          </button>
        </div>
      </div>

      {/* ── Center Gaze Dwell Reticle for Split-Screen ── */}
      {viewMode === 'splitscreen' && (
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
          {/* Left Eye Reticle */}
          <div className="w-1/2 flex items-center justify-center">
            <div className="relative w-8 h-8 flex items-center justify-center">
              <svg className="w-8 h-8 -rotate-90">
                <circle cx="16" cy="16" r="12" stroke="rgba(255,255,255,0.2)" strokeWidth="2" fill="none" />
                {dwellProgress > 0 && (
                  <circle
                    cx="16"
                    cy="16"
                    r="12"
                    stroke="#00e87a"
                    strokeWidth="3"
                    fill="none"
                    strokeDasharray={2 * Math.PI * 12}
                    strokeDashoffset={2 * Math.PI * 12 * (1 - dwellProgress)}
                    className="transition-all duration-75"
                  />
                )}
              </svg>
              <div className="absolute w-1.5 h-1.5 rounded-full bg-white" />
            </div>
          </div>

          {/* Center Divider Line */}
          <div className="w-[2px] h-full bg-white/20" />

          {/* Right Eye Reticle */}
          <div className="w-1/2 flex items-center justify-center">
            <div className="relative w-8 h-8 flex items-center justify-center">
              <svg className="w-8 h-8 -rotate-90">
                <circle cx="16" cy="16" r="12" stroke="rgba(255,255,255,0.2)" strokeWidth="2" fill="none" />
                {dwellProgress > 0 && (
                  <circle
                    cx="16"
                    cy="16"
                    r="12"
                    stroke="#00e87a"
                    strokeWidth="3"
                    fill="none"
                    strokeDasharray={2 * Math.PI * 12}
                    strokeDashoffset={2 * Math.PI * 12 * (1 - dwellProgress)}
                    className="transition-all duration-75"
                  />
                )}
              </svg>
              <div className="absolute w-1.5 h-1.5 rounded-full bg-white" />
            </div>
          </div>
        </div>
      )}

      {/* ── Bottom Floating Console: Target Lock & Engagement ── */}
      <div className="flex items-end justify-between gap-4 shrink-0">
        {/* Navigation & Shortcuts Tooltip */}
        <div className="px-3 py-2 bg-black/80 backdrop-blur-md border border-[var(--c-border-faint)] rounded text-[10px] text-[var(--c-text-mid)] space-y-1 shadow-lg">
          <div className="font-bold text-white uppercase tracking-wider">360° SPATIAL AIR DEFENSE</div>
          <div>DESKTOP: Drag to orbit · Scroll to zoom FOV (40°-90°)</div>
          <div>KEYBOARD: 1–9 / Tab to cycle targets · Esc deselect</div>
        </div>

        {/* Selected Target Tactical Action Console */}
        {selectedContact ? (
          <div className="pointer-events-auto p-4 bg-[var(--c-bg-surface)]/95 backdrop-blur-md border border-[var(--c-border-strong)] rounded-lg shadow-2xl space-y-3 w-96 text-xs text-white">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-[var(--c-border-base)] pb-2">
              <div className="flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-[var(--c-accent)]" />
                <span className="font-bold text-sm text-[var(--c-text-max)]">
                  LOCK: {selectedContact.id.slice(0, 8).toUpperCase()}
                </span>
              </div>
              <Badge variant="hostile" size="xs">
                {((selectedContact as any).assigned || selectedContact.assigned_classification || 'unknown').toUpperCase()}
              </Badge>
            </div>

            {/* Target Telemetry Metrics */}
            <div className="grid grid-cols-3 gap-2 text-[11px] bg-[var(--c-bg-deep)] p-2.5 rounded border border-[var(--c-border-faint)]">
              <div>
                <span className="text-[10px] text-[var(--c-text-low)] block">RANGE</span>
                <strong className="text-white text-xs">{targetRangeM.toLocaleString()} m</strong>
              </div>
              <div>
                <span className="text-[10px] text-[var(--c-text-low)] block">BEARING</span>
                <strong className="text-white text-xs">{selectedContact.bearing_deg.toFixed(1)}°</strong>
              </div>
              <div>
                <span className="text-[10px] text-[var(--c-text-low)] block">ALTITUDE</span>
                <strong className="text-white text-xs">{selectedContact.altitude} m</strong>
              </div>
            </div>

            {/* Engagement Restriction Warnings */}
            {isInsideNoEngageZone && (
              <div className="p-2 bg-red-950/90 border border-red-500 rounded text-[10px] font-bold text-red-200 flex items-center gap-1.5">
                <ShieldAlert className="w-4 h-4 text-red-400 shrink-0" />
                <span>HARD-KILL PROHIBITED: INSIDE {violatingZoneName.toUpperCase()}</span>
              </div>
            )}

            {!isInsideNoEngageZone && isHardKillBeyondRange && (
              <div className="p-2 bg-amber-950/80 border border-amber-500/60 rounded text-[10px] font-bold text-amber-200 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                <span>LONG-RANGE INTERCEPTION: Target at {targetRangeM}m (nominal limit: {maxHardKillRangeM}m)</span>
              </div>
            )}

            {isSoftKillBeyondRange && (
              <div className="p-1.5 bg-zinc-900 border border-zinc-700 rounded text-[9px] text-zinc-400">
                Notice: Target is beyond standard soft-kill boundary ({maxSoftKillRangeM}m).
              </div>
            )}

            {/* Engagement Action Buttons */}
            <div className="grid grid-cols-3 gap-2 pt-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onExecuteEngagement('warn')}
              >
                Warn / Hail
              </Button>

              <Button
                size="sm"
                variant="secondary"
                onClick={() => onExecuteEngagement('soft_kill')}
              >
                RF Soft-Kill
              </Button>

              {/* Hard-kill with 600ms Hold-to-Confirm Progress */}
              <div className="relative">
                <button
                  type="button"
                  disabled={isHardKillBlocked}
                  onMouseDown={startHardKillHold}
                  onMouseUp={cancelHardKillHold}
                  onMouseLeave={cancelHardKillHold}
                  onTouchStart={startHardKillHold}
                  onTouchEnd={cancelHardKillHold}
                  className={`w-full h-full py-1.5 px-2 rounded text-xs font-bold transition-all relative overflow-hidden select-none flex items-center justify-center gap-1.5 ${
                    isHardKillBlocked
                      ? 'bg-zinc-800 text-zinc-500 border border-zinc-700 cursor-not-allowed opacity-60'
                      : holdingHardKill
                      ? 'bg-red-700 text-white cursor-pointer shadow-[0_0_15px_#ef4444]'
                      : 'bg-red-600 hover:bg-red-500 text-white cursor-pointer shadow-lg'
                  }`}
                >
                  {/* Progress Fill Background */}
                  {holdingHardKill && (
                    <div
                      className="absolute inset-0 bg-red-950/80 transition-all duration-75"
                      style={{ width: `${Math.round(holdProgress * 100)}%` }}
                    />
                  )}
                  <Zap className="w-3.5 h-3.5 relative z-10" />
                  <span className="relative z-10">
                    {holdingHardKill ? `HOLD (${Math.round((1 - holdProgress) * 600)}ms)` : 'HARD-KILL'}
                  </span>
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3 bg-black/80 backdrop-blur-md border border-[var(--c-border-faint)] rounded text-xs text-[var(--c-text-mid)]">
            Click any target or press <strong className="text-white">1–9 / Tab</strong> to acquire target lock.
          </div>
        )}
      </div>
    </div>
  );
};
