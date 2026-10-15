import React, { useState, useEffect, useRef } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { useToast } from '../ui/Toast';
import {
  Glasses,
  Maximize2,
  Minimize2,
  Compass,
  Crosshair,
  ShieldAlert,
  Play,
  RotateCcw,
  Zap,
} from 'lucide-react';

export const VrPage: React.FC = () => {
  const { toast } = useToast();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [inVr, setInVr] = useState(false);
  const [webXrSupported, setWebXrSupported] = useState(false);
  const [activeContact, setActiveContact] = useState<string | null>('TGT-01');
  const [yaw, setYaw] = useState(0);
  const [pitch, setPitch] = useState(15);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0, yaw: 0, pitch: 0 });

  // Check WebXR support
  useEffect(() => {
    if (typeof navigator !== 'undefined' && (navigator as any).xr) {
      (navigator as any).xr.isSessionSupported('immersive-vr').then((supported: boolean) => {
        setWebXrSupported(supported);
      }).catch(() => setWebXrSupported(false));
    }
  }, []);

  // 3D Spatial Threat Dome Canvas Simulation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let t = 0;

    const render = () => {
      t += 0.015;
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;

      // 1. Background Void & Star/Grid Dome
      ctx.fillStyle = '#020508';
      ctx.fillRect(0, 0, w, h);

      // 2. Horizon Line and Ground Grid
      const horizonY = cy + (pitch * 4);
      ctx.strokeStyle = 'rgba(30, 60, 100, 0.4)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, horizonY);
      ctx.lineTo(w, horizonY);
      ctx.stroke();

      // Perspective ground grid lines
      for (let x = -400; x <= 400; x += 80) {
        ctx.beginPath();
        ctx.moveTo(cx + x * 0.2 + (yaw * 2), horizonY);
        ctx.lineTo(cx + x * 2.5 + (yaw * 6), h);
        ctx.strokeStyle = 'rgba(0, 232, 122, 0.12)';
        ctx.stroke();
      }

      // 3. 3D Spatial Hologram Rings (Arm's Reach Dome)
      const domeRadius = Math.min(w, h) * 0.42;
      ctx.beginPath();
      ctx.ellipse(cx, horizonY + 60, domeRadius, domeRadius * 0.35, 0, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0, 232, 122, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.beginPath();
      ctx.ellipse(cx, horizonY + 60, domeRadius * 0.65, domeRadius * 0.22, 0, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0, 232, 122, 0.2)';
      ctx.stroke();

      // Rotating 3D sweep vector
      const sweepRad = t * 2;
      ctx.beginPath();
      ctx.moveTo(cx, horizonY + 60);
      ctx.lineTo(
        cx + Math.cos(sweepRad) * domeRadius,
        horizonY + 60 + Math.sin(sweepRad) * (domeRadius * 0.35)
      );
      ctx.strokeStyle = 'rgba(0, 232, 122, 0.7)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // 4. Simulated 3D Targets with Billboard Facing HUD Tags
      const targets = [
        { id: 'TGT-01', dist: 3.8, az: 35 + yaw, elev: 120, cls: 'hostile_drone' },
        { id: 'TGT-02', dist: 5.1, az: -40 + yaw, elev: 80, cls: 'decoy' },
        { id: 'TGT-03', dist: 2.4, az: 110 + yaw, elev: 240, cls: 'friendly_drone' },
      ];

      targets.forEach((tgt) => {
        const rad = (tgt.az * Math.PI) / 180;
        const screenX = cx + Math.sin(rad) * (domeRadius * 0.9);
        const screenY = horizonY - (tgt.elev * 0.5) + (pitch * 2);

        const isSelected = activeContact === tgt.id;
        const col = tgt.cls === 'hostile_drone' ? '#ff3b3b' : tgt.cls === 'friendly_drone' ? '#0a84ff' : '#bf5af2';

        // 3D Threat Glyph
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(screenX, screenY, isSelected ? 8 : 5, 0, Math.PI * 2);
        ctx.fill();

        // Pulsing Selection Reticle
        if (isSelected) {
          const p = 14 + Math.sin(t * 8) * 3;
          ctx.strokeStyle = '#00e87a';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(screenX, screenY, p, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Billboard Tag
        ctx.fillStyle = 'rgba(8, 18, 30, 0.85)';
        ctx.fillRect(screenX + 10, screenY - 14, 85, 28);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1;
        ctx.strokeRect(screenX + 10, screenY - 14, 85, 28);

        ctx.fillStyle = '#f0f8ff';
        ctx.font = 'bold 10px "Geist Mono", monospace';
        ctx.fillText(tgt.id, screenX + 14, screenY - 2);
        ctx.fillStyle = 'rgba(176, 200, 224, 0.8)';
        ctx.font = '9px "Geist Mono", monospace';
        ctx.fillText(`${tgt.dist}km | ${tgt.elev}m`, screenX + 14, screenY + 9);
      });

      // 5. Head-Tracking HUD Reticle at Center
      ctx.strokeStyle = 'rgba(0, 232, 122, 0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 24, cy); ctx.lineTo(cx - 8, cy);
      ctx.moveTo(cx + 8, cy); ctx.lineTo(cx + 24, cy);
      ctx.moveTo(cx, cy - 24); ctx.lineTo(cx, cy - 8);
      ctx.moveTo(cx, cy + 8); ctx.lineTo(cx, cy + 24);
      ctx.stroke();

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [yaw, pitch, activeContact]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      const c = canvasRef.current;
      if (!c) return;
      c.width = c.parentElement?.clientWidth || window.innerWidth;
      c.height = c.parentElement?.clientHeight || 600;
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Mouse Drag Look / Pan for Desktop Fallback
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY, yaw, pitch };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    setYaw(dragStartRef.current.yaw + dx * 0.25);
    setPitch(Math.max(-40, Math.min(50, dragStartRef.current.pitch + dy * 0.25)));
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="flex flex-col h-full bg-[var(--c-bg-void)] select-none">
      {/* Top Controls */}
      <div className="h-12 px-4 bg-[var(--c-bg-deep)] border-b border-[var(--c-border-base)] flex items-center justify-between text-xs font-mono shrink-0">
        <div className="flex items-center gap-3">
          <Glasses className="w-4 h-4 text-[var(--c-accent)]" />
          <span className="font-bold text-[var(--c-text-max)]">VR THREAT DOME (360° WEBXR)</span>
          <span className="text-[10px] text-[var(--c-text-low)]">[DESKTOP ORBIT ACTIVE: DRAG TO LOOK]</span>
        </div>

        <div className="flex items-center gap-2">
          {webXrSupported ? (
            <Button
              variant="primary"
              size="sm"
              icon={<Glasses className="w-4 h-4" />}
              onClick={() => toast('ENTER VR', 'Launching WebXR session in connected headset...', 'success')}
            >
              Enter Immersive VR
            </Button>
          ) : (
            <div className="flex items-center gap-2 text-[10px] text-[var(--c-text-mid)] px-2 py-1 bg-[var(--c-bg-surface)] border border-[var(--c-border-base)] rounded-xs">
              <span>HEADSET SIMULATOR ACTIVE</span>
            </div>
          )}
        </div>
      </div>

      {/* 3D Threat Dome Canvas */}
      <div
        className="flex-1 relative overflow-hidden cursor-move"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <canvas ref={canvasRef} className="w-full h-full block" />

        {/* Floating Gaze-Follow Wrist HUD */}
        <div className="absolute bottom-6 left-6 p-4 bg-[var(--c-bg-surface)]/90 backdrop-blur-md border border-[var(--c-border-strong)] rounded-sm shadow-2xl font-mono text-xs space-y-3 w-80 pointer-events-auto">
          <div className="flex items-center justify-between border-b border-[var(--c-border-base)] pb-2">
            <span className="text-[10px] text-[var(--c-text-low)] uppercase">VIRTUAL WRIST CONTROLLER</span>
            <Badge variant="warning" size="xs">LOCKED</Badge>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-[var(--c-text-low)]">AZIMUTH / ELEV:</span>
              <span className="text-[var(--c-accent)] font-bold">{Math.round(yaw)}° / {Math.round(pitch)}°</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--c-text-low)]">TARGET LOCK:</span>
              <span className="text-[var(--c-text-max)] font-bold">{activeContact}</span>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[var(--c-border-faint)]">
            <Button size="xs" variant="secondary" onClick={() => setActiveContact('TGT-01')}>Target 01</Button>
            <Button size="xs" variant="secondary" onClick={() => setActiveContact('TGT-02')}>Target 02</Button>
            <Button size="xs" variant="danger" onClick={() => toast('VR ENGAGE', 'Trigger pressed: Kinetic countermeasure deployed', 'error')}>
              Trigger Fire
            </Button>
            <Button size="xs" variant="outline" onClick={() => { setYaw(0); setPitch(15); }}>
              Reset Horizon
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
