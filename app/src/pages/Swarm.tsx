import React, { useState, useEffect, useRef } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Stat } from '../ui/Stat';
import { Select } from '../ui/Select';
import {
  Boxes,
  Play,
  Pause,
  RotateCcw,
  Battery,
  Wifi,
  Radio,
  RadioTower,
  Shield,
  Activity,
} from 'lucide-react';

interface UavNode {
  id: string;
  role: 'relay' | 'scout' | 'interceptor' | 'gateway';
  normX: number; // 0 to 1 normalized coordinates for resolution-independent rendering
  normY: number;
  battery: number;
  pdr: number;
  neighbors: string[];
}

export const SwarmPage: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [scenario, setScenario] = useState('baseline_earthquake_01');
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedUavId, setSelectedUavId] = useState<string>('UAV-01');

  // 6 UAV Nodes in normalized coordinate space
  const [nodes] = useState<UavNode[]>([
    { id: 'UAV-01', role: 'gateway', normX: 0.25, normY: 0.35, battery: 94, pdr: 99.2, neighbors: ['UAV-02', 'UAV-03'] },
    { id: 'UAV-02', role: 'relay', normX: 0.50, normY: 0.25, battery: 88, pdr: 96.5, neighbors: ['UAV-01', 'UAV-04', 'UAV-05'] },
    { id: 'UAV-03', role: 'scout', normX: 0.22, normY: 0.65, battery: 82, pdr: 94.8, neighbors: ['UAV-01', 'UAV-05'] },
    { id: 'UAV-04', role: 'interceptor', normX: 0.78, normY: 0.38, battery: 76, pdr: 92.1, neighbors: ['UAV-02', 'UAV-06'] },
    { id: 'UAV-05', role: 'relay', normX: 0.48, normY: 0.72, battery: 85, pdr: 95.0, neighbors: ['UAV-02', 'UAV-03', 'UAV-06'] },
    { id: 'UAV-06', role: 'scout', normX: 0.75, normY: 0.68, battery: 71, pdr: 91.4, neighbors: ['UAV-04', 'UAV-05'] },
  ]);

  // Resolution-independent Canvas Setup with ResizeObserver and devicePixelRatio
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    };

    updateSize();

    const observer = new ResizeObserver(() => {
      updateSize();
    });
    observer.observe(container);

    return () => observer.disconnect();
  }, []);

  // Canvas Simulation Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;

      ctx.save();
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = '#0B0E13';
      ctx.fillRect(0, 0, w, h);

      // Subtle Grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      const step = 48;
      for (let x = 0; x < w; x += step) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += step) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // Convert nodes to pixel coords
      const pixelNodes = nodes.map((n) => ({
        ...n,
        px: n.normX * w,
        py: n.normY * h,
      }));

      // Draw mesh links
      pixelNodes.forEach((n1) => {
        n1.neighbors.forEach((nid) => {
          const n2 = pixelNodes.find((x) => x.id === nid);
          if (n2 && n1.id < n2.id) {
            ctx.beginPath();
            ctx.moveTo(n1.px, n1.py);
            ctx.lineTo(n2.px, n2.py);
            ctx.strokeStyle = 'rgba(61, 214, 198, 0.25)';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Midpoint signal indicator
            const midX = (n1.px + n2.px) / 2;
            const midY = (n1.py + n2.py) / 2;
            ctx.fillStyle = 'rgba(61, 214, 198, 0.4)';
            ctx.beginPath();
            ctx.arc(midX, midY, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      });

      // Role color palette (calm enterprise)
      const roleColors: Record<string, string> = {
        gateway: '#3DD6C6',
        relay: '#3B82F6',
        interceptor: '#EF4444',
        scout: '#F59E0B',
      };

      // Draw UAV nodes
      pixelNodes.forEach((uav) => {
        const isSelected = uav.id === selectedUavId;
        const color = roleColors[uav.role] || '#9AA4B2';

        // Selection ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(uav.px, uav.py, 16, 0, Math.PI * 2);
          ctx.strokeStyle = 'var(--c-accent)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Node circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(uav.px, uav.py, 6, 0, Math.PI * 2);
        ctx.fill();

        // Node label
        ctx.fillStyle = isSelected ? '#FFFFFF' : '#E6EAF0';
        ctx.font = '600 12px "Geist", system-ui, sans-serif';
        ctx.fillText(uav.id, uav.px + 12, uav.py - 3);

        ctx.fillStyle = '#6B7585';
        ctx.font = '500 11px "Geist", system-ui, sans-serif';
        ctx.fillText(`${uav.role} · ${uav.battery}%`, uav.px + 12, uav.py + 11);
      });

      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [nodes, selectedUavId]);

  const selectedNode = nodes.find((n) => n.id === selectedUavId) || nodes[0];

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <h1 className="text-lg font-semibold text-[var(--c-text-high)] font-ui">
            Swarm dynamics and mesh network
          </h1>
          <p className="text-xs text-[var(--c-text-mid)] font-ui mt-0.5">
            Autonomous ad-hoc communication routing, spatial dispersion, and relay re-allocation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Select
            label="Scenario"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            options={[
              { value: 'baseline_earthquake_01', label: 'Baseline earthquake' },
              { value: 'multi_failure_stress', label: 'Multi-node failure stress' },
              { value: 'comms_outage_region', label: 'Comms outage region' },
              { value: 'cascading_emergency_priorities', label: 'Cascading emergency' },
            ]}
          />
        </div>
      </div>

      {/* KPI Stats Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="Mesh packet delivery" value="98.4" unit="%" sparkline={[94, 96, 97, 98.4]} />
        <Stat label="Relay availability" value="100" unit="%" sparkline={[100, 100, 100, 100]} />
        <Stat label="Routing latency" value="14.2" unit="ms" sparkline={[18, 16, 15, 14.2]} />
        <Stat label="Collision margin" value="100" unit="%" sparkline={[100, 100, 100, 100]} />
      </div>

      {/* Main Workspace (Canvas + Inspector) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[500px]">
        {/* Left: 2D Tactical Swarm Mesh Panel (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col h-full min-h-[460px]">
          <Panel
            title="Mesh topology and spatial dispersion"
            subtitle={`${nodes.length} active nodes`}
            className="h-full flex flex-col"
            bodyClassName="p-0 flex-1 relative min-h-[420px] overflow-hidden"
          >
            <div ref={containerRef} className="absolute inset-0 w-full h-full">
              <canvas
                ref={canvasRef}
                className="w-full h-full block cursor-pointer"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const cx = e.clientX - rect.left;
                  const cy = e.clientY - rect.top;
                  nodes.forEach((n) => {
                    const nx = n.normX * rect.width;
                    const ny = n.normY * rect.height;
                    if (Math.hypot(nx - cx, ny - cy) < 26) {
                      setSelectedUavId(n.id);
                    }
                  });
                }}
              />
            </div>
          </Panel>
        </div>

        {/* Right: UAV Node Inspector (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col">
          <Panel
            title="Node inspector"
            subtitle={selectedNode.id}
            bodyClassName="p-5 space-y-5"
          >
            <div className="flex items-center justify-between pb-4 border-b border-[var(--c-border-base)]">
              <div>
                <span className="text-xs text-[var(--c-text-low)] font-ui block">Call sign</span>
                <div className="text-lg font-semibold text-[var(--c-text-high)] font-mono mt-0.5">
                  {selectedNode.id}
                </div>
              </div>
              <Badge
                variant={
                  selectedNode.role === 'interceptor'
                    ? 'hostile'
                    : selectedNode.role === 'gateway'
                    ? 'info'
                    : selectedNode.role === 'relay'
                    ? 'friendly'
                    : 'warning'
                }
                size="sm"
              >
                {selectedNode.role}
              </Badge>
            </div>

            {/* Clear Label/Value Rows */}
            <div className="space-y-3 text-xs font-ui">
              <div className="flex items-center justify-between">
                <span className="text-[var(--c-text-mid)]">Battery level</span>
                <span className="font-mono tabular-nums font-semibold text-[var(--c-text-high)]">
                  {selectedNode.battery}%
                </span>
              </div>
              <div className="h-2 bg-[var(--c-bg-raised)] rounded-full overflow-hidden">
                <div
                  className="h-full bg-[var(--c-accent)] rounded-full"
                  style={{ width: `${selectedNode.battery}%` }}
                />
              </div>

              <div className="pt-2 border-t border-[var(--c-border-base)] space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[var(--c-text-mid)]">Packet delivery ratio</span>
                  <span className="font-mono tabular-nums font-semibold text-[var(--c-text-high)]">
                    {selectedNode.pdr}%
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[var(--c-text-mid)]">Connected mesh peers</span>
                  <span className="font-mono text-[var(--c-text-high)]">
                    {selectedNode.neighbors.join(', ')}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-[var(--c-text-mid)]">Normalized position</span>
                  <span className="font-mono tabular-nums text-[var(--c-text-low)]">
                    {(selectedNode.normX * 100).toFixed(0)}%, {(selectedNode.normY * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="p-3.5 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-md space-y-1 mt-4">
                <span className="text-[11px] font-medium text-[var(--c-text-low)] block">
                  Autonomous directive
                </span>
                <span className="text-xs text-[var(--c-text-high)] font-medium leading-relaxed block">
                  Distributed ad-hoc coverage (Sector B perimeter)
                </span>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
};
