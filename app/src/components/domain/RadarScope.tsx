import React, { useRef, useEffect, useState, useCallback } from 'react';
import type { Contact } from '../../api/types';

export interface RadarScopeProps {
  contacts: Contact[];
  selectedContactId?: string | null;
  onSelectContact?: (contactId: string) => void;
  maxRangeKm?: number;
  sweepSpeedRpm?: number;
  className?: string;
  showTrails?: boolean;
  defendedAsset?: [number, number];
  noEngageZones?: Array<{ name?: string; center: [number, number]; radius_m: number }>;
}

export const RadarScope: React.FC<RadarScopeProps> = ({
  contacts,
  selectedContactId,
  onSelectContact,
  maxRangeKm = 6,
  sweepSpeedRpm = 15,
  className = '',
  showTrails = true,
  defendedAsset,
  noEngageZones,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // View state: zoom & pan
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  // Sweep angle in radians
  const sweepAngleRef = useRef(0);
  const lastTimeRef = useRef<number>(performance.now());

  // Render loop
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + pan.x;
    const centerY = height / 2 + pan.y;
    const radius = (Math.min(width, height) / 2 - 30) * zoom;
    const safeMaxRangeKm = maxRangeKm > 0 ? maxRangeKm : 6;

    // 1. Clear with deep void tactical background
    ctx.fillStyle = '#0B0E13';
    ctx.fillRect(0, 0, width, height);

    // 2. Tactical grid rings
    const ringCount = 4;
    ctx.lineWidth = 1;
    for (let i = 1; i <= ringCount; i++) {
      const r = (radius / ringCount) * i;
      ctx.beginPath();
      ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
      ctx.strokeStyle = i === ringCount ? 'rgba(61, 214, 198, 0.35)' : 'rgba(255, 255, 255, 0.06)';
      ctx.stroke();

      // Range labels (km)
      const rangeLabel = `${((safeMaxRangeKm / ringCount) * i).toFixed(1)} km`;
      ctx.fillStyle = 'rgba(154, 164, 178, 0.8)';
      ctx.font = '10px "Geist Mono", monospace';
      ctx.fillText(rangeLabel, centerX + 5, centerY - r + 12);
    }

    // 3. Radial Bearing Lines (every 30 deg)
    for (let deg = 0; deg < 360; deg += 30) {
      const rad = (deg - 90) * (Math.PI / 180);
      const isCardinal = deg % 90 === 0;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + Math.cos(rad) * radius, centerY + Math.sin(rad) * radius);
      ctx.strokeStyle = isCardinal ? 'rgba(61, 214, 198, 0.25)' : 'rgba(255, 255, 255, 0.04)';
      ctx.stroke();

      // Bearing text on outer ring
      const labelDist = radius + 15;
      const labelX = centerX + Math.cos(rad) * labelDist;
      const labelY = centerY + Math.sin(rad) * labelDist;
      ctx.fillStyle = isCardinal ? '#3DD6C6' : 'rgba(154, 164, 178, 0.6)';
      ctx.font = '10px "Geist Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const degText = deg.toString().padStart(3, '0') + '°';
      ctx.fillText(degText, labelX, labelY);
    }

    // 4. Defended Asset Marker
    if (defendedAsset) {
      const [ax, ay] = defendedAsset;
      const aDist = (Math.hypot(ax, ay) / 1000 / safeMaxRangeKm) * radius;
      const aAngle = ((Math.atan2(ax, -ay) * 180 / Math.PI) - 90) * (Math.PI / 180);
      const aPx = centerX + Math.cos(aAngle) * aDist;
      const aPy = centerY + Math.sin(aAngle) * aDist;

      ctx.fillStyle = 'rgba(245, 158, 11, 0.20)';
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(aPx - 8, aPy - 8, 16, 16);
      ctx.fillRect(aPx - 8, aPy - 8, 16, 16);

      ctx.fillStyle = '#F59E0B';
      ctx.font = '500 10px "Geist", system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Defended asset', aPx + 12, aPy + 3);
    }

    // 5. No-Engage Zones
    if (noEngageZones && noEngageZones.length > 0) {
      noEngageZones.forEach((zone) => {
        const [zx, zy] = zone.center;
        const zDist = (Math.hypot(zx, zy) / 1000 / safeMaxRangeKm) * radius;
        const zAngle = ((Math.atan2(zx, -zy) * 180 / Math.PI) - 90) * (Math.PI / 180);
        const zPx = centerX + Math.cos(zAngle) * zDist;
        const zPy = centerY + Math.sin(zAngle) * zDist;
        const zRadius = ((zone.radius_m / 1000) / safeMaxRangeKm) * radius;

        ctx.save();
        ctx.beginPath();
        ctx.arc(zPx, zPy, Math.max(4, zRadius), 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
        ctx.font = '500 10px "Geist", system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(zone.name || 'No-engage zone', zPx, zPy - zRadius - 5);
      });
    }

    // 6. Sweep Beam with Phosphor Decay
    const sweepAngle = sweepAngleRef.current;
    const sweepGradient = ctx.createConicGradient(sweepAngle - Math.PI / 2, centerX, centerY);
    sweepGradient.addColorStop(0, 'rgba(61, 214, 198, 0.20)');
    sweepGradient.addColorStop(0.08, 'rgba(61, 214, 198, 0.03)');
    sweepGradient.addColorStop(0.18, 'rgba(61, 214, 198, 0)');
    sweepGradient.addColorStop(1, 'rgba(61, 214, 198, 0)');

    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = sweepGradient;
    ctx.fill();
    ctx.restore();

    // Leading sweep line
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(centerX + Math.cos(sweepAngle) * radius, centerY + Math.sin(sweepAngle) * radius);
    ctx.strokeStyle = 'rgba(61, 214, 198, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Center home station crosshair
    ctx.beginPath();
    ctx.arc(centerX, centerY, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#3DD6C6';
    ctx.fill();
    ctx.strokeStyle = '#3DD6C6';
    ctx.strokeRect(centerX - 6, centerY - 6, 12, 12);

    // 7. Contacts Rendering
    (contacts || []).forEach((contact) => {
      // Calculate contact screen position relative to center safely
      const rKm = typeof contact.range_km === 'number' && !isNaN(contact.range_km)
        ? contact.range_km
        : (Math.hypot(contact.x ?? 0, contact.y ?? 0) / 1000);

      const bDeg = typeof contact.bearing_deg === 'number' && !isNaN(contact.bearing_deg)
        ? contact.bearing_deg
        : (((Math.atan2(contact.x ?? 0, -(contact.y ?? 0)) * 180 / Math.PI) + 360) % 360);

      const contactDist = Math.max(0, (rKm / safeMaxRangeKm)) * radius;
      const angleRad = (bDeg - 90) * (Math.PI / 180);
      const px = centerX + Math.cos(angleRad) * contactDist;
      const py = centerY + Math.sin(angleRad) * contactDist;

      const contactId = contact.id || (contact as any).contact_id || 'TRACK';
      const isSelected = contactId === selectedContactId;
      const cls = contact.assigned_classification || contact.classification || 'unknown';

      // Pick tactical color & glyph shape
      let color = '#ffb800'; // unknown amber
      let shape = 'diamond';
      if (cls === 'hostile_drone' || cls === 'hostile_swarm_member') {
        color = '#ff3b3b'; // hostile red
        shape = 'triangle';
      } else if (cls === 'friendly_drone') {
        color = '#0a84ff'; // friendly blue
        shape = 'circle';
      } else if (cls === 'decoy') {
        color = '#bf5af2'; // decoy purple
        shape = 'square';
      } else if (cls === 'bird') {
        color = '#5a9e8a'; // neutral
        shape = 'circle';
      }

      // Breadcrumb History Trails
      if (showTrails && contact.history && contact.history.length > 0) {
        ctx.beginPath();
        contact.history.forEach((pt, idx) => {
          const histDist = (Math.hypot(pt.x, pt.y) / 1000 / safeMaxRangeKm) * radius;
          const histAngle = ((Math.atan2(pt.x, -pt.y) * 180 / Math.PI) - 90) * (Math.PI / 180);
          const hx = centerX + Math.cos(histAngle) * histDist;
          const hy = centerY + Math.sin(histAngle) * histDist;
          if (idx === 0) ctx.moveTo(hx, hy);
          else ctx.lineTo(hx, hy);
        });
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Neutralized Target: Render red destruction marker and strike cross
      if (contact.state === 'neutralized') {
        ctx.beginPath();
        ctx.arc(px, py, 8 * zoom, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(px - 5, py - 5);
        ctx.lineTo(px + 5, py + 5);
        ctx.moveTo(px + 5, py - 5);
        ctx.lineTo(px - 5, py + 5);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 8px "Geist Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('KILL', px, py + 14);

        if (isSelected) {
          ctx.beginPath();
          ctx.arc(px, py, 14 * zoom, 0, Math.PI * 2);
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        return;
      }

      // Velocity Vector Line
      const speed = contact.speed ?? 20;
      if (speed > 0) {
        const headingRad = ((contact.heading ?? 0) - 90) * (Math.PI / 180);
        const vectorLen = Math.min(30, (speed / 20) * 15 * zoom);
        const vx = px + Math.cos(headingRad) * vectorLen;
        const vy = py + Math.sin(headingRad) * vectorLen;

        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(vx, vy);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Draw Tactical Shape Glyph (Hostile=triangle, Unknown=diamond, Friendly=circle, Decoy=square)
      const size = 6 * zoom;
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;

      if (shape === 'triangle') {
        ctx.beginPath();
        ctx.moveTo(px, py - size);
        ctx.lineTo(px - size, py + size);
        ctx.lineTo(px + size, py + size);
        ctx.closePath();
        ctx.fill();
      } else if (shape === 'diamond') {
        ctx.beginPath();
        ctx.moveTo(px, py - size);
        ctx.lineTo(px + size, py);
        ctx.lineTo(px, py + size);
        ctx.lineTo(px - size, py);
        ctx.closePath();
        ctx.fill();
      } else if (shape === 'circle') {
        ctx.beginPath();
        ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.strokeRect(px - size, py - size, size * 2, size * 2);
      }

      // Selected Ring & Pulsing Target Lock Indicator
      if (isSelected) {
        const pulse = 14 + Math.sin(performance.now() / 150) * 3;
        ctx.beginPath();
        ctx.arc(px, py, pulse * zoom, 0, Math.PI * 2);
        ctx.strokeStyle = '#00e87a';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Target ID tag
        ctx.fillStyle = '#f0f8ff';
        ctx.font = 'bold 10px "Geist Mono", monospace';
        ctx.textAlign = 'left';
        ctx.fillText(contactId.toUpperCase(), px + size + 8, py - 4);
        ctx.fillStyle = 'rgba(138, 170, 191, 0.9)';
        ctx.font = '9px "Geist Mono", monospace';
        ctx.fillText(`${contact.altitude ?? 120}m | ${speed}m/s`, px + size + 8, py + 8);
      }
    });
  }, [contacts, selectedContactId, maxRangeKm, zoom, pan, showTrails, defendedAsset, noEngageZones]);

  // Animation Loop
  useEffect(() => {
    let animId: number;
    const animate = (time: number) => {
      const dt = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      const radPerSec = (sweepSpeedRpm * 2 * Math.PI) / 60;
      sweepAngleRef.current = (sweepAngleRef.current + radPerSec * dt) % (2 * Math.PI);

      draw();
      animId = requestAnimationFrame(animate);
    };

    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, [draw, sweepSpeedRpm]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Zoom interaction
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.max(0.4, Math.min(4, z * zoomFactor)));
  };

  // Pan interaction
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStartRef.current.x,
      y: e.clientY - dragStartRef.current.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Click on contact to select
  const handleClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !onSelectContact || !contacts) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2 + pan.x;
    const centerY = height / 2 + pan.y;
    const radius = (Math.min(width, height) / 2 - 30) * zoom;
    const safeMaxRangeKm = maxRangeKm > 0 ? maxRangeKm : 6;

    for (const c of contacts) {
      const rKm = typeof c.range_km === 'number' && !isNaN(c.range_km)
        ? c.range_km
        : (Math.hypot(c.x ?? 0, c.y ?? 0) / 1000);

      const bDeg = typeof c.bearing_deg === 'number' && !isNaN(c.bearing_deg)
        ? c.bearing_deg
        : (((Math.atan2(c.x ?? 0, -(c.y ?? 0)) * 180 / Math.PI) + 360) % 360);

      const contactDist = (rKm / safeMaxRangeKm) * radius;
      const angleRad = (bDeg - 90) * (Math.PI / 180);
      const px = centerX + Math.cos(angleRad) * contactDist;
      const py = centerY + Math.sin(angleRad) * contactDist;

      const dist = Math.hypot(clickX - px, clickY - py);
      if (dist < 18) {
        onSelectContact(c.id || (c as any).contact_id);
        return;
      }
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full overflow-hidden bg-[var(--c-bg-void)] select-none ${className}`}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
    >
      <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />

      {/* Scope Controls Overlay */}
      <div className="absolute bottom-3 right-3 flex items-center gap-1 bg-[var(--c-bg-surface)]/90 backdrop-blur-xs border border-[var(--c-border-base)] p-1 rounded-md text-xs font-ui shadow-sm">
        <button
          onClick={() => setZoom((z) => Math.min(4, z * 1.25))}
          className="px-2 py-0.5 hover:bg-[var(--c-bg-raised)] text-[var(--c-text-high)] rounded cursor-pointer font-mono"
        >
          +
        </button>
        <span className="text-[11px] text-[var(--c-text-mid)] font-mono tabular-nums px-1">
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={() => setZoom((z) => Math.max(0.4, z / 1.25))}
          className="px-2 py-0.5 hover:bg-[var(--c-bg-raised)] text-[var(--c-text-high)] rounded cursor-pointer font-mono"
        >
          -
        </button>
        <button
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
          className="px-2 py-0.5 text-xs hover:bg-[var(--c-bg-raised)] text-[var(--c-accent)] rounded cursor-pointer border-l border-[var(--c-border-base)] ml-1"
        >
          Reset
        </button>
      </div>
    </div>
  );
};
