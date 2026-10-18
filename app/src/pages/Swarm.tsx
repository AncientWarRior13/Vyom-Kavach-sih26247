import React, { useState, useEffect, useRef, useMemo } from 'react';
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
  AlertTriangle,
  Zap,
} from 'lucide-react';

export type SwarmFormation = 'orbit' | 'flock' | 'wedge' | 'grid';

export interface DynamicUavNode {
  id: string;
  role: 'gateway' | 'relay' | 'scout' | 'interceptor';
  x: number; // pixel coords or normalized
  y: number;
  vx: number;
  vy: number;
  heading: number;
  battery: number;
  pdr: number;
  isJammed: boolean;
  baseRadius?: number;
  orbitAngle?: number;
  orbitSpeed?: number;
}

interface DataPacket {
  id: number;
  sourceId: string;
  targetId: string;
  progress: number; // 0 to 1
  speed: number;
}

export const SwarmPage: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [formation, setFormation] = useState<SwarmFormation>('orbit');
  const [nodeCount, setNodeCount] = useState<number>(10);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [selectedUavId, setSelectedUavId] = useState<string>('UAV-01');
  const [jammedNodes, setJammedNodes] = useState<Set<string>>(new Set());

  // Dynamic simulation nodes ref for high-fps canvas updates
  const simNodesRef = useRef<DynamicUavNode[]>([]);
  const packetsRef = useRef<DataPacket[]>([]);
  const packetIdCounter = useRef(0);
  const animFrameRef = useRef<number | null>(null);

  // Initialize or reconfigure nodes based on count and formation
  const initNodes = (count: number, form: SwarmFormation) => {
    const newNodes: DynamicUavNode[] = [];
    const roles: Array<'gateway' | 'relay' | 'scout' | 'interceptor'> = [
      'gateway',
      'relay',
      'relay',
      'scout',
      'scout',
      'interceptor',
      'interceptor',
      'relay',
      'scout',
      'interceptor',
      'relay',
      'scout',
    ];

    for (let i = 0; i < count; i++) {
      const id = `UAV-${String(i + 1).padStart(2, '0')}`;
      const role = i === 0 ? 'gateway' : roles[i % roles.length];
      const orbitRadius = i === 0 ? 0 : 0.15 + (i % 3) * 0.12;
      const orbitAngle = (i / (count - 1 || 1)) * Math.PI * 2;
      const orbitSpeed = (0.008 + (i % 3) * 0.004) * (i % 2 === 0 ? 1 : -1);

      newNodes.push({
        id,
        role,
        x: 0.5 + (i === 0 ? 0 : Math.cos(orbitAngle) * orbitRadius),
        y: 0.5 + (i === 0 ? 0 : Math.sin(orbitAngle) * orbitRadius),
        vx: (Math.random() - 0.5) * 0.002,
        vy: (Math.random() - 0.5) * 0.002,
        heading: Math.random() * 360,
        battery: Math.round(75 + Math.random() * 24),
        pdr: Number((95 + Math.random() * 4.9).toFixed(1)),
        isJammed: false,
        baseRadius: orbitRadius,
        orbitAngle,
        orbitSpeed,
      });
    }

    simNodesRef.current = newNodes;
    packetsRef.current = [];
    if (!newNodes.some((n) => n.id === selectedUavId)) {
      setSelectedUavId(newNodes[0].id);
    }
  };

  // Re-init on nodeCount changes
  useEffect(() => {
    initNodes(nodeCount, formation);
  }, [nodeCount]);

  // Handle ResizeObserver for canvas resolution
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
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Main Dynamic Physics and Animation Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastTime = performance.now();

    const loop = (currentTime: number) => {
      const dt = Math.min((currentTime - lastTime) / 1000, 0.1);
      lastTime = currentTime;

      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      if (w === 0 || h === 0) {
        animFrameRef.current = requestAnimationFrame(loop);
        return;
      }

      const nodes = simNodesRef.current;

      // Update positions if playing
      if (isPlaying) {
        nodes.forEach((node, idx) => {
          if (node.isJammed) return;

          if (formation === 'orbit') {
            if (node.role === 'gateway') {
              node.x = 0.5;
              node.y = 0.5;
              node.heading += 10 * dt;
            } else {
              node.orbitAngle = (node.orbitAngle || 0) + (node.orbitSpeed || 0.01);
              const r = node.baseRadius || 0.25;
              const targetX = 0.5 + Math.cos(node.orbitAngle) * r;
              const targetY = 0.5 + Math.sin(node.orbitAngle) * (r * 0.85);
              node.x += (targetX - node.x) * 0.08;
              node.y += (targetY - node.y) * 0.08;
              node.heading = (node.orbitAngle * 180) / Math.PI + 90;
            }
          } else if (formation === 'flock') {
            // Boids: Cohesion + Separation + Center gravity
            let steerX = (0.5 - node.x) * 0.0005;
            let steerY = (0.5 - node.y) * 0.0005;

            nodes.forEach((other, oIdx) => {
              if (idx === oIdx) return;
              const dx = (node.x - other.x) * w;
              const dy = (node.y - other.y) * h;
              const dist = Math.hypot(dx, dy);
              if (dist > 0 && dist < 60) {
                // Separation
                steerX += (dx / dist) * 0.00015;
                steerY += (dy / dist) * 0.00015;
              }
            });

            node.vx += steerX;
            node.vy += steerY;
            const speed = Math.hypot(node.vx, node.vy);
            const maxSpd = 0.0015;
            if (speed > maxSpd) {
              node.vx = (node.vx / speed) * maxSpd;
              node.vy = (node.vy / speed) * maxSpd;
            }
            node.x += node.vx;
            node.y += node.vy;
            node.x = Math.max(0.08, Math.min(0.92, node.x));
            node.y = Math.max(0.08, Math.min(0.92, node.y));
            node.heading = (Math.atan2(node.vy, node.vx) * 180) / Math.PI + 90;
          } else if (formation === 'wedge') {
            // Tactical V-formation escorting gateway
            const leadX = 0.5;
            const leadY = 0.35 + Math.sin(currentTime * 0.001) * 0.08;
            if (node.role === 'gateway') {
              node.x = leadX;
              node.y = leadY;
            } else {
              const side = idx % 2 === 0 ? 1 : -1;
              const tier = Math.ceil(idx / 2);
              const targetX = leadX + side * tier * 0.07;
              const targetY = leadY + tier * 0.06;
              node.x += (targetX - node.x) * 0.06;
              node.y += (targetY - node.y) * 0.06;
            }
            node.heading = 0;
          } else if (formation === 'grid') {
            // Dispersed patrol grid
            const cols = 4;
            const row = Math.floor(idx / cols);
            const col = idx % cols;
            const targetX = 0.2 + (col / (cols - 1)) * 0.6;
            const targetY = 0.22 + (row / Math.ceil(nodes.length / cols)) * 0.56;
            node.x += (targetX - node.x) * 0.05;
            node.y += (targetY - node.y) * 0.05;
            node.heading = (currentTime * 0.05) % 360;
          }

          // Slow battery drain
          if (Math.random() < 0.005 && node.battery > 15) {
            node.battery -= 0.1;
          }
        });

        // Spawn traveling data packets between close mesh peers
        if (Math.random() < 0.2 && packetsRef.current.length < 25) {
          const activeNodes = nodes.filter((n) => !n.isJammed);
          if (activeNodes.length >= 2) {
            const src = activeNodes[Math.floor(Math.random() * activeNodes.length)];
            // Find neighbor within range
            const neighbors = activeNodes.filter((n) => {
              if (n.id === src.id) return false;
              const dist = Math.hypot((src.x - n.x) * w, (src.y - n.y) * h);
              return dist < 180;
            });
            if (neighbors.length > 0) {
              const tgt = neighbors[Math.floor(Math.random() * neighbors.length)];
              packetsRef.current.push({
                id: packetIdCounter.current++,
                sourceId: src.id,
                targetId: tgt.id,
                progress: 0,
                speed: 0.02 + Math.random() * 0.02,
              });
            }
          }
        }

        // Update packets
        packetsRef.current.forEach((pkt) => {
          pkt.progress += pkt.speed;
        });
        packetsRef.current = packetsRef.current.filter((pkt) => pkt.progress < 1);
      }

      // Render Tactical Canvas
      ctx.save();
      ctx.scale(dpr, dpr);

      // Deep cyber background
      ctx.fillStyle = '#060A10';
      ctx.fillRect(0, 0, w, h);

      // Radial Range Rings from Center
      const cx = w * 0.5;
      const cy = h * 0.5;
      const maxR = Math.min(w, h) * 0.45;

      ctx.strokeStyle = 'rgba(61, 214, 198, 0.06)';
      ctx.lineWidth = 1;
      for (let r = maxR * 0.25; r <= maxR; r += maxR * 0.25) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Crosshairs
      ctx.beginPath();
      ctx.moveTo(cx - maxR, cy);
      ctx.lineTo(cx + maxR, cy);
      ctx.moveTo(cx, cy - maxR);
      ctx.lineTo(cx, cy + maxR);
      ctx.stroke();

      // Tactical Grid Lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
      const step = 40;
      for (let x = 0; x < w; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Calculate Pixel Positions & Dynamic Mesh Links
      const maxCommDistance = 190;
      const pixelNodes = nodes.map((n) => ({
        ...n,
        px: n.x * w,
        py: n.y * h,
      }));

      // Render Active Dynamic Mesh Links
      for (let i = 0; i < pixelNodes.length; i++) {
        for (let j = i + 1; j < pixelNodes.length; j++) {
          const n1 = pixelNodes[i];
          const n2 = pixelNodes[j];
          if (n1.isJammed || n2.isJammed) continue;

          const dist = Math.hypot(n1.px - n2.px, n1.py - n2.py);
          if (dist < maxCommDistance) {
            const alpha = (1 - dist / maxCommDistance) * 0.45;
            ctx.beginPath();
            ctx.moveTo(n1.px, n1.py);
            ctx.lineTo(n2.px, n2.py);
            ctx.strokeStyle = `rgba(61, 214, 198, ${alpha})`;
            ctx.lineWidth = Math.max(1, 2.2 * (1 - dist / maxCommDistance));
            ctx.stroke();
          }
        }
      }

      // Render Traveling Data Packets
      packetsRef.current.forEach((pkt) => {
        const s = pixelNodes.find((n) => n.id === pkt.sourceId);
        const t = pixelNodes.find((n) => n.id === pkt.targetId);
        if (!s || !t) return;

        const px = s.px + (t.px - s.px) * pkt.progress;
        const py = s.py + (t.py - s.py) * pkt.progress;

        // Glowing packet dot
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#3DD6C6';
        ctx.shadowColor = '#3DD6C6';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Role Colors
      const roleColors: Record<string, string> = {
        gateway: '#3DD6C6',
        relay: '#3B82F6',
        interceptor: '#EF4444',
        scout: '#F59E0B',
      };

      // Draw Drone Nodes
      pixelNodes.forEach((uav) => {
        const isSelected = uav.id === selectedUavId;
        const baseColor = uav.isJammed ? '#EF4444' : roleColors[uav.role] || '#9AA4B2';

        // Outer Coverage / Sensor Footprint Circle
        ctx.beginPath();
        ctx.arc(uav.px, uav.py, 38, 0, Math.PI * 2);
        ctx.fillStyle = isSelected
          ? 'rgba(61, 214, 198, 0.08)'
          : uav.isJammed
          ? 'rgba(239, 68, 68, 0.04)'
          : 'rgba(61, 214, 198, 0.02)';
        ctx.fill();

        // EW Jamming Field Ripple
        if (uav.isJammed) {
          const jamPhase = ((currentTime * 0.05) % 25);
          ctx.beginPath();
          ctx.arc(uav.px, uav.py, 14 + jamPhase, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(239, 68, 68, ${Math.max(0, 1 - jamPhase / 25) * 0.8})`;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Selection Target Reticle
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(uav.px, uav.py, 20, 0, Math.PI * 2);
          ctx.strokeStyle = '#3DD6C6';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Corner reticles
          const cr = 24;
          ctx.strokeStyle = '#3DD6C6';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(uav.px - cr, uav.py - 12);
          ctx.lineTo(uav.px - cr, uav.py - cr);
          ctx.lineTo(uav.px - 12, uav.py - cr);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(uav.px + cr, uav.py - 12);
          ctx.lineTo(uav.px + cr, uav.py - cr);
          ctx.lineTo(uav.px + 12, uav.py - cr);
          ctx.stroke();
        }

        // Heading Vector Pointer
        const rad = (uav.heading * Math.PI) / 180;
        const hLen = 14;
        ctx.beginPath();
        ctx.moveTo(uav.px, uav.py);
        ctx.lineTo(uav.px + Math.sin(rad) * hLen, uav.py - Math.cos(rad) * hLen);
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Node Central Core
        ctx.beginPath();
        ctx.arc(uav.px, uav.py, uav.role === 'gateway' ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = baseColor;
        ctx.shadowColor = baseColor;
        ctx.shadowBlur = isSelected ? 12 : 6;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Node ID and Telemetry Label
        ctx.font = '600 11px "Geist", monospace, system-ui';
        ctx.fillStyle = isSelected ? '#FFFFFF' : '#D1D5DB';
        ctx.fillText(uav.id, uav.px + 12, uav.py - 4);

        ctx.font = '500 9px "Geist", monospace, system-ui';
        ctx.fillStyle = uav.isJammed ? '#EF4444' : '#9CA3AF';
        ctx.fillText(
          uav.isJammed ? 'JAMMED / SILENT' : `${uav.role.toUpperCase()} • ${Math.round(uav.battery)}%`,
          uav.px + 12,
          uav.py + 8
        );
      });

      ctx.restore();
      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, formation, selectedUavId]);

  // Live Mesh Autonomy Event Log
  const [meshEvents, setMeshEvents] = useState<Array<{ id: string; time: string; text: string; type: 'alert' | 'heal' | 'info' }>>([
    { id: '1', time: '00:01', text: 'Ad-hoc 915MHz token ring established across all active UAV nodes.', type: 'info' },
    { id: '2', time: '00:03', text: 'Dynamic mesh routes converged (latency: 11.4ms, packet delivery: 98.8%).', type: 'heal' },
  ]);

  const addMeshEvent = (text: string, type: 'alert' | 'heal' | 'info') => {
    const time = new Date().toISOString().slice(14, 19);
    setMeshEvents((prev) => [{ id: Math.random().toString(36).substring(2, 9), time, text, type }, ...prev.slice(0, 19)]);
  };

  // Selected Node computation - REACTIVE to jammedNodes set
  const selectedNode = useMemo(() => {
    const node = simNodesRef.current.find((n) => n.id === selectedUavId) || simNodesRef.current[0] || null;
    if (!node) return null;
    return {
      ...node,
      isJammed: jammedNodes.has(node.id),
    };
  }, [selectedUavId, nodeCount, jammedNodes]);

  const isJammed = Boolean(selectedNode?.isJammed);

  // Dynamic Neighbors for Selected Node
  const selectedNeighbors = useMemo(() => {
    if (!selectedNode || isJammed) return [];
    const container = containerRef.current;
    const w = container ? container.clientWidth : 800;
    const h = container ? container.clientHeight : 500;
    return simNodesRef.current
      .filter((other) => {
        if (other.id === selectedNode.id || jammedNodes.has(other.id)) return false;
        const dist = Math.hypot((selectedNode.x - other.x) * w, (selectedNode.y - other.y) * h);
        return dist < 190;
      })
      .map((n) => n.id);
  }, [selectedNode, isPlaying, isJammed, jammedNodes]);

  // Toggle Jamming on Selected Node with reactive state and event log
  const toggleJamSelected = () => {
    if (!selectedNode) return;
    const willJam = !jammedNodes.has(selectedNode.id);

    simNodesRef.current = simNodesRef.current.map((n) => {
      if (n.id === selectedNode.id) {
        return { ...n, isJammed: willJam };
      }
      return n;
    });

    setJammedNodes((prev) => {
      const next = new Set(prev);
      if (willJam) next.add(selectedNode.id);
      else next.delete(selectedNode.id);
      return next;
    });

    if (willJam) {
      addMeshEvent(`EW JAMMING DETECTED: ${selectedNode.id} signal jammed. Comm link severed.`, 'alert');
      addMeshEvent(`MESH SELF-HEALED: Ad-hoc traffic re-routed around ${selectedNode.id} in 0.4s.`, 'heal');
    } else {
      addMeshEvent(`COMM RESTORED: ${selectedNode.id} re-established peer links with mesh network.`, 'info');
    }
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Boxes className="w-5 h-5 text-[var(--c-accent)]" />
            <h1 className="text-xl font-bold text-[var(--c-text-high)] font-ui">
              Autonomous Swarm Dynamics & Mesh Network
            </h1>
          </div>
          <p className="text-xs text-[var(--c-text-mid)] font-ui mt-1">
            Real-time boids dispersion physics, ad-hoc radio propagation, dynamic link healing, and multi-agent coordination.
          </p>
        </div>

        {/* Formation & Swarm Configuration */}
        <div className="flex items-center gap-3">
          <Select
            label="Tactical Formation"
            value={formation}
            onChange={(e) => setFormation(e.target.value as SwarmFormation)}
            options={[
              { value: 'orbit', label: 'Perimeter Defense Orbit' },
              { value: 'flock', label: 'Autonomous Boids Flocking' },
              { value: 'wedge', label: 'Tactical Intercept Wedge' },
              { value: 'grid', label: 'Dispersed Sensor Grid' },
            ]}
          />

          <Select
            label="Swarm Scale"
            value={String(nodeCount)}
            onChange={(e) => setNodeCount(Number(e.target.value))}
            options={[
              { value: '6', label: '6 Nodes (Tactical Cell)' },
              { value: '10', label: '10 Nodes (Patrol Flight)' },
              { value: '16', label: '16 Nodes (Full Swarm)' },
            ]}
          />

          <div className="pt-5">
            <Button
              variant={isPlaying ? 'secondary' : 'primary'}
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? <Pause className="w-4 h-4 mr-1" /> : <Play className="w-4 h-4 mr-1" />}
              {isPlaying ? 'Pause' : 'Resume'}
            </Button>
          </div>
        </div>
      </div>

      {/* KPI Stats Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat
          label="Active Mesh Nodes"
          value={String(simNodesRef.current.filter((n) => !n.isJammed).length)}
          unit={`/ ${nodeCount}`}
          sparkline={[nodeCount - 1, nodeCount, nodeCount, simNodesRef.current.filter((n) => !n.isJammed).length]}
        />
        <Stat
          label="Packet Delivery Ratio"
          value={simNodesRef.current.filter((n) => !n.isJammed).length > 2 ? '98.8' : '42.1'}
          unit="%"
          sparkline={[94, 96, 97.5, 98.8]}
        />
        <Stat label="Mesh Routing Latency" value="11.4" unit="ms" sparkline={[18, 15, 12, 11.4]} />
        <Stat
          label="Self-Healing Recovery"
          value="< 0.8"
          unit="sec"
          sparkline={[1.2, 1.0, 0.9, 0.8]}
        />
      </div>

      {/* Main Workspace (Canvas + Inspector) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[520px]">
        {/* Left: 2D Tactical Swarm Mesh Panel (8 Cols) */}
        <div className="lg:col-span-8 flex flex-col h-full min-h-[480px]">
          <Panel
            title="Real-Time Mesh Topology & Kinetic Dispersion"
            subtitle={`${simNodesRef.current.filter((n) => !n.isJammed).length} online • Click node to track telemetry`}
            className="h-full flex flex-col"
            bodyClassName="p-0 flex-1 relative min-h-[440px] overflow-hidden"
          >
            <div ref={containerRef} className="absolute inset-0 w-full h-full">
              <canvas
                ref={canvasRef}
                className="w-full h-full block cursor-crosshair"
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const cx = e.clientX - rect.left;
                  const cy = e.clientY - rect.top;
                  simNodesRef.current.forEach((n) => {
                    const nx = n.x * rect.width;
                    const ny = n.y * rect.height;
                    if (Math.hypot(nx - cx, ny - cy) < 28) {
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
            title="Node Telemetry & Comms"
            subtitle={selectedNode ? selectedNode.id : 'No Node Selected'}
            bodyClassName="p-5 space-y-5"
          >
            {selectedNode ? (
              <>
                <div className="flex items-center justify-between pb-4 border-b border-[var(--c-border-base)]">
                  <div>
                    <span className="text-xs text-[var(--c-text-low)] font-ui block">TACTICAL CALLSIGN</span>
                    <div className="text-xl font-bold text-[var(--c-text-high)] font-mono mt-0.5">
                      {selectedNode.id}
                    </div>
                  </div>
                  <Badge
                    variant={
                      isJammed
                        ? 'hostile'
                        : selectedNode.role === 'gateway'
                        ? 'info'
                        : selectedNode.role === 'interceptor'
                        ? 'warning'
                        : 'friendly'
                    }
                    size="sm"
                  >
                    {isJammed ? 'EW JAMMED / OFF-GRID' : selectedNode.role.toUpperCase()}
                  </Badge>
                </div>

                {/* Clear Label/Value Rows */}
                <div className="space-y-3 text-xs font-ui">
                  <div className="flex items-center justify-between">
                    <span className="text-[var(--c-text-mid)]">Battery Reserve</span>
                    <span className="font-mono tabular-nums font-semibold text-[var(--c-text-high)]">
                      {Math.round(selectedNode.battery)}%
                    </span>
                  </div>
                  <div className="h-2 bg-[var(--c-bg-raised)] rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        selectedNode.battery < 25 ? 'bg-[var(--c-hostile)]' : 'bg-[var(--c-accent)]'
                      }`}
                      style={{ width: `${selectedNode.battery}%` }}
                    />
                  </div>

                  <div className="pt-2 border-t border-[var(--c-border-base)] space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[var(--c-text-mid)]">Dynamic Mesh Peers</span>
                      <span className={`font-mono font-semibold ${isJammed ? 'text-[var(--c-hostile)]' : 'text-[var(--c-accent)]'}`}>
                        {isJammed ? 'DISCONNECTED (LINK SEVERED)' : selectedNeighbors.length > 0 ? selectedNeighbors.join(', ') : 'ISOLATED'}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-[var(--c-text-mid)]">Packet Delivery Rate</span>
                      <span className={`font-mono tabular-nums font-semibold ${isJammed ? 'text-[var(--c-hostile)]' : 'text-[var(--c-text-high)]'}`}>
                        {isJammed ? '0.0% (SIGNAL BLOCKED)' : `${selectedNode.pdr}%`}
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-[var(--c-text-mid)]">Heading & Bearing</span>
                      <span className="font-mono tabular-nums text-[var(--c-text-high)]">
                        {Math.round(selectedNode.heading % 360)}° Vector
                      </span>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-[var(--c-text-mid)]">RF Transmit Power</span>
                      <span className={`font-mono tabular-nums font-semibold ${isJammed ? 'text-[var(--c-hostile)]' : 'text-[var(--c-text-mid)]'}`}>
                        {isJammed ? '0 dBm (NOISE FLOOR EXCEEDED)' : '+20 dBm (LoRa / 915MHz)'}
                      </span>
                    </div>
                  </div>

                  {/* Node Jamming Simulation Action Button */}
                  <div className="pt-3">
                    <Button
                      variant={isJammed ? 'primary' : 'danger'}
                      size="sm"
                      className="w-full justify-center font-mono text-xs cursor-pointer shadow-md"
                      onClick={toggleJamSelected}
                    >
                      {isJammed ? (
                        <>
                          <RotateCcw className="w-4 h-4 mr-1.5 text-[var(--c-accent)]" />
                          Clear EW Jamming & Reconnect Node
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4 mr-1.5 text-white" />
                          Simulate EW Jamming on Node
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Outcome and Self-Healing Directive Display */}
                  {isJammed ? (
                    <div className="p-3 bg-[var(--c-hostile)]/10 border border-[var(--c-hostile)]/40 rounded-xs space-y-2 mt-3 font-mono">
                      <div className="flex items-center gap-1.5 text-[var(--c-hostile)] font-bold text-[11px]">
                        <AlertTriangle className="w-4 h-4 shrink-0 text-[var(--c-hostile)]" />
                        EW INTERFERENCE ACTIVE
                      </div>
                      <p className="text-[11px] text-[var(--c-text-mid)] font-ui leading-relaxed">
                        Hostile jamming has flooded the 915MHz channel. Node has lost mesh sync and locked into emergency stationary hover.
                      </p>
                      <div className="p-2.5 bg-[var(--c-bg-deep)] border border-[var(--c-accent)]/30 rounded-xs space-y-1">
                        <div className="text-[10px] text-[var(--c-accent)] font-semibold flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--c-accent)] animate-pulse" />
                          AD-HOC MESH SELF-HEALED (0.4s)
                        </div>
                        <p className="text-[10px] text-[var(--c-text-low)] font-ui">
                          Surrounding peers dynamically re-allocated routing tables. Packets bypass {selectedNode.id} without network collapse.
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-md space-y-1 mt-3">
                      <span className="text-[11px] font-semibold text-[var(--c-text-low)] block">
                        AUTONOMOUS DIRECTIVE
                      </span>
                      <span className="text-xs text-[var(--c-text-high)] font-medium leading-relaxed block">
                        {formation === 'orbit' && 'Perimeter asset ring patrol with active token-ring telemetry.'}
                        {formation === 'flock' && 'Boids flocking with collision avoidance & consensus centroid.'}
                        {formation === 'wedge' && 'Forward tactical intercept vector escorting the primary gateway.'}
                        {formation === 'grid' && 'Dispersed sensor area sweeping with dynamic neighbor handover.'}
                      </span>
                    </div>
                  )}
                </div>
              </>
            ) : null}
          </Panel>
        </div>
      </div>

      {/* Live Mesh Network Autonomy & Electronic Warfare Event Log */}
      <Panel
        title="Mesh Autonomy & Self-Healing Network Log"
        subtitle="Real-time ad-hoc topology events, EW impact tracking, and packet rerouting"
        bodyClassName="p-3 bg-[var(--c-bg-deep)]"
      >
        <div className="space-y-1.5 font-mono text-xs max-h-36 overflow-y-auto">
          {meshEvents.map((ev) => (
            <div key={ev.id} className="flex items-start gap-2 py-0.5 border-b border-[var(--c-border-faint)] last:border-0">
              <span className="text-[var(--c-text-low)] text-[10px] shrink-0 font-semibold">{ev.time}</span>
              <span
                className={`text-[11px] ${
                  ev.type === 'alert'
                    ? 'text-[var(--c-hostile)] font-semibold'
                    : ev.type === 'heal'
                    ? 'text-[var(--c-accent)] font-semibold'
                    : 'text-[var(--c-text-mid)]'
                }`}
              >
                {ev.text}
              </span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
};
