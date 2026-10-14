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
  Zap,
  Activity,
  ShieldAlert,
  Battery,
  Wifi,
  Radio,
  Layers,
} from 'lucide-react';

interface UavNode {
  id: string;
  role: 'relay' | 'scout' | 'interceptor' | 'gateway';
  x: number;
  y: number;
  vx: number;
  vy: number;
  battery: number;
  pdr: number;
  neighbors: string[];
}

export const SwarmPage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [scenario, setScenario] = useState('baseline_earthquake_01');
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedUavId, setSelectedUavId] = useState<string>('UAV-01');
  const [timelineSec, setTimelineSec] = useState(42);

  // Simulated UAV Nodes
  const [nodes, setNodes] = useState<UavNode[]>([
    { id: 'UAV-01', role: 'gateway', x: 200, y: 180, vx: 0.5, vy: -0.2, battery: 94, pdr: 99.2, neighbors: ['UAV-02', 'UAV-03'] },
    { id: 'UAV-02', role: 'relay', x: 310, y: 150, vx: 0.2, vy: 0.4, battery: 88, pdr: 96.5, neighbors: ['UAV-01', 'UAV-04'] },
    { id: 'UAV-03', role: 'scout', x: 180, y: 310, vx: -0.3, vy: 0.2, battery: 82, pdr: 94.8, neighbors: ['UAV-01', 'UAV-05'] },
    { id: 'UAV-04', role: 'interceptor', x: 440, y: 200, vx: 0.1, vy: -0.5, battery: 76, pdr: 92.1, neighbors: ['UAV-02'] },
    { id: 'UAV-05', role: 'relay', x: 290, y: 340, vx: -0.2, vy: -0.3, battery: 85, pdr: 95.0, neighbors: ['UAV-03', 'UAV-06'] },
    { id: 'UAV-06', role: 'scout', x: 420, y: 330, vx: 0.4, vy: 0.1, battery: 71, pdr: 91.4, neighbors: ['UAV-05'] },
  ]);

  // Canvas Simulation Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = '#020508';
      ctx.fillRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = 'rgba(30, 60, 100, 0.35)';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }

      // Draw mesh comms links
      nodes.forEach((n1) => {
        n1.neighbors.forEach((nid) => {
          const n2 = nodes.find((x) => x.id === nid);
          if (n2) {
            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.strokeStyle = 'rgba(0, 232, 122, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        });
      });

      // Draw UAV nodes
      nodes.forEach((uav) => {
        const isSelected = uav.id === selectedUavId;
        const color = uav.role === 'gateway' ? '#00e87a' : uav.role === 'relay' ? '#0a84ff' : uav.role === 'interceptor' ? '#ff3b3b' : '#ffb800';

        // Selection ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(uav.x, uav.y, 16, 0, Math.PI * 2);
          ctx.strokeStyle = '#00e87a';
          ctx.lineWidth = 2;
          ctx.setLineDash([3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // UAV Glyph
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(uav.x, uav.y, 7, 0, Math.PI * 2);
        ctx.fill();

        // Label
        ctx.fillStyle = '#f0f8ff';
        ctx.font = 'bold 10px "Geist Mono", monospace';
        ctx.fillText(uav.id, uav.x + 12, uav.y - 4);
        ctx.fillStyle = 'rgba(138, 170, 191, 0.8)';
        ctx.font = '9px "Geist Mono", monospace';
        ctx.fillText(`${uav.role.toUpperCase()} | ${uav.battery}%`, uav.x + 12, uav.y + 8);
      });

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [nodes, selectedUavId]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      const c = canvasRef.current;
      if (!c) return;
      c.width = c.parentElement?.clientWidth || 600;
      c.height = c.parentElement?.clientHeight || 450;
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const selectedNode = nodes.find((n) => n.id === selectedUavId) || nodes[0];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Top Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-wider text-[var(--c-text-max)] uppercase">
            ARES-X Swarm Dynamics & Dispersion
          </h1>
          <p className="font-mono text-xs text-[var(--c-text-mid)] mt-1">
            Autonomous multi-UAV mesh network routing, dynamic relay re-allocation, and collective coverage.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Select
            label="Tactical Scenario"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            options={[
              { value: 'baseline_earthquake_01', label: 'baseline_earthquake_01' },
              { value: 'multi_failure_stress', label: 'multi_failure_stress' },
              { value: 'comms_outage_region', label: 'comms_outage_region' },
              { value: 'cascading_emergency_priorities', label: 'cascading_emergency' },
            ]}
          />
        </div>
      </div>

      {/* KPI Stat Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="Mesh PDR" value="98.4" unit="%" delta={{ value: '+1.1%', trend: 'up' }} sparkline={[92, 95, 98.4]} />
        <Stat label="Relay Availability" value="100" unit="%" delta={{ value: '0.0%', trend: 'neutral' }} sparkline={[100, 100, 100]} />
        <Stat label="Mean Latency" value="14.2" unit="MS" delta={{ value: '-2ms', trend: 'up' }} sparkline={[18, 16, 14.2]} />
        <Stat label="Zero-Collision Rate" value="100" unit="%" delta={{ value: 'CLEAN', trend: 'up' }} sparkline={[100, 100, 100]} />
      </div>

      {/* Main Workspace (Canvas + Inspector) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[520px]">
        {/* Left: 2D Tactical Swarm Mesh (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col h-full overflow-hidden">
          <Panel
            title="Mesh Topology & Spatial Dispersion"
            subtitle={`${nodes.length} AUTONOMOUS NODES`}
            className="h-full flex flex-col"
          >
            <div className="flex-1 min-h-0 relative">
              <canvas
                ref={canvasRef}
                className="w-full h-full block cursor-pointer"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const cx = e.clientX - rect.left;
                  const cy = e.clientY - rect.top;
                  nodes.forEach((n) => {
                    if (Math.hypot(n.x - cx, n.y - cy) < 22) setSelectedUavId(n.id);
                  });
                }}
              />
            </div>
          </Panel>
        </div>

        {/* Right: UAV Node Inspector (4 Cols) */}
        <div className="lg:col-span-4 flex flex-col h-full space-y-4 overflow-y-auto">
          <Panel title="UAV Node Telemetry Inspector" subtitle={selectedNode.id}>
            <div className="space-y-4 font-mono text-xs">
              <div className="flex items-center justify-between border-b border-[var(--c-border-base)] pb-3">
                <div>
                  <span className="text-[10px] text-[var(--c-text-low)] uppercase">NODE CALLSIGN:</span>
                  <div className="text-xl font-bold text-[var(--c-accent)]">{selectedNode.id}</div>
                </div>
                <Badge variant={selectedNode.role === 'interceptor' ? 'critical' : selectedNode.role === 'relay' ? 'friendly' : 'advisory'} size="sm">
                  {selectedNode.role.toUpperCase()}
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-[var(--c-text-low)] flex items-center gap-1.5">
                    <Battery className="w-3.5 h-3.5 text-[var(--c-accent)]" /> BATTERY LEVEL:
                  </span>
                  <span className="font-bold text-[var(--c-text-max)] tabular-nums">{selectedNode.battery}%</span>
                </div>
                <div className="h-1.5 bg-[var(--c-bg-raised)] rounded-full overflow-hidden">
                  <div className="h-full bg-[var(--c-accent)]" style={{ width: `${selectedNode.battery}%` }} />
                </div>
              </div>

              <div className="space-y-2 pt-2 border-t border-[var(--c-border-faint)]">
                <div className="flex justify-between">
                  <span className="text-[var(--c-text-low)]">PACKET DELIVERY RATIO:</span>
                  <span className="font-bold text-[var(--c-accent)] tabular-nums">{selectedNode.pdr}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--c-text-low)]">MESH NEIGHBORS:</span>
                  <span className="text-[var(--c-text-max)]">{selectedNode.neighbors.join(', ')}</span>
                </div>
              </div>

              <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded-xs space-y-1">
                <span className="text-[10px] text-[var(--c-text-low)] uppercase block">AUTONOMOUS TASK DIRECTIVE:</span>
                <span className="text-[var(--c-text-high)] font-semibold">DISTRIBUTED AD-HOC COVERAGE (ZONE B)</span>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
};
