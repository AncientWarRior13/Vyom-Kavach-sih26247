import React, { useState } from 'react';
import { Panel } from '../ui/Panel';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Stat } from '../ui/Stat';
import { Segmented } from '../ui/Segmented';
import { Slider } from '../ui/Slider';
import { Select } from '../ui/Select';
import { Tabs } from '../ui/Tabs';
import { Table, Column } from '../ui/Table';
import { Modal } from '../ui/Modal';
import { Drawer } from '../ui/Drawer';
import { useToast } from '../ui/Toast';
import { Tooltip } from '../ui/Tooltip';
import { Skeleton, EmptyState, ProgressRing, Timeline } from '../ui/Feedback';
import { RadarScope } from '../components/domain/RadarScope';
import { ContactRow } from '../components/domain/ContactRow';
import { SensorHealthChips, EventLog, ServerStatusLED } from '../components/domain/TacticalFeed';
import type { Contact } from '../api/types';

export const StyleguidePage: React.FC = () => {
  const { toast } = useToast();

  const [activeTab, setActiveTab] = useState('components');
  const [segValue, setSegValue] = useState('2d');
  const [sliderVal, setSliderVal] = useState(75);
  const [selectVal, setSelectVal] = useState('hostile');
  const [modalOpen, setModalOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedContactId, setSelectedContactId] = useState<string | null>('tgt-01');

  // Mock Contacts for Radar and List
  const mockContacts: Contact[] = [
    {
      id: 'tgt-01',
      x: 2.4,
      y: 4.1,
      range_km: 4.8,
      bearing_deg: 32,
      altitude: 180,
      speed: 28,
      heading: 140,
      classification: 'hostile_drone',
      state: 'tracking',
      history: [
        { x: 1.8, y: 3.2, t: 1 },
        { x: 2.1, y: 3.7, t: 2 },
        { x: 2.4, y: 4.1, t: 3 },
      ],
    },
    {
      id: 'tgt-02',
      x: -3.5,
      y: 2.1,
      range_km: 4.1,
      bearing_deg: 300,
      altitude: 90,
      speed: 34,
      heading: 95,
      classification: 'unknown',
      state: 'detected',
      history: [
        { x: -4.0, y: 2.0, t: 1 },
        { x: -3.5, y: 2.1, t: 2 },
      ],
    },
    {
      id: 'tgt-03',
      x: 1.2,
      y: -5.0,
      range_km: 5.1,
      bearing_deg: 165,
      altitude: 350,
      speed: 15,
      heading: 340,
      classification: 'friendly_drone',
      state: 'classified',
    },
    {
      id: 'tgt-04',
      x: -4.8,
      y: -3.2,
      range_km: 5.8,
      bearing_deg: 235,
      altitude: 210,
      speed: 42,
      heading: 45,
      classification: 'decoy',
      state: 'detected',
    },
  ];

  const tableColumns: Column<Contact>[] = [
    { key: 'id', header: 'TRACK ID', sortable: true, render: (c) => <span className="font-bold text-[var(--c-accent)]">{c.id.toUpperCase()}</span> },
    {
      key: 'classification',
      header: 'TYPE',
      sortable: true,
      render: (c) => (
        <Badge variant={c.classification as any} size="xs">
          {c.classification?.replace('_', ' ')}
        </Badge>
      ),
    },
    { key: 'range_km', header: 'RANGE', sortable: true, align: 'right', render: (c) => `${c.range_km.toFixed(1)} km` },
    { key: 'bearing_deg', header: 'BEARING', sortable: true, align: 'right', render: (c) => `${Math.round(c.bearing_deg)}°` },
    { key: 'altitude', header: 'ALT', sortable: true, align: 'right', render: (c) => `${c.altitude} m` },
    { key: 'speed', header: 'SPD', sortable: true, align: 'right', render: (c) => `${c.speed} m/s` },
  ];

  const mockEvents = [
    { id: '1', timestamp: '04:12:01Z', level: 'warning' as const, source: 'RADAR', message: 'High-speed micro-UAS signature detected bearing 032° range 4.8km.' },
    { id: '2', timestamp: '04:12:05Z', level: 'info' as const, source: 'RF', message: 'Frequency sweep intercepted 2.448 GHz hopping protocol FHSS.' },
    { id: '3', timestamp: '04:12:12Z', level: 'critical' as const, source: 'OPTIC', message: 'Visual confirmed weaponized quadcopter carrying shaped charge.' },
    { id: '4', timestamp: '04:12:18Z', level: 'success' as const, source: 'C-UAS', message: 'Kinetic intercept net deployed. Target neutralized.' },
  ];

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-wider text-[var(--c-text-max)] uppercase">
            Tactical Operations Design System
          </h1>
          <p className="font-mono text-xs text-[var(--c-text-mid)] mt-1">
            PS-247 Token Architecture, Component Suite, and Radar Engine
          </p>
        </div>
        <div className="flex items-center gap-2">
          <ServerStatusLED status="ONLINE" latencyMs={18} />
          <Button
            variant="primary"
            size="sm"
            onClick={() => toast('TELEMETRY SYNC', 'System tokens verified successfully', 'success')}
          >
            Trigger Toast
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs
        activeTab={activeTab}
        onChange={setActiveTab}
        tabs={[
          { id: 'components', label: 'UI COMPONENTS', badge: '18' },
          { id: 'domain', label: 'RADAR & DOMAIN', badge: 'LIVE' },
          { id: 'tokens', label: 'DESIGN TOKENS & COLOR' },
        ]}
      />

      {/* Tab 1: Components */}
      {activeTab === 'components' && (
        <div className="space-y-6">
          {/* Row 1: Buttons & Badges */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Panel title="Button Variants & Keyboard Hints">
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  <Button variant="primary" kbd="SPC">Primary Action</Button>
                  <Button variant="secondary" kbd="TAB">Secondary</Button>
                  <Button variant="danger" kbd="DEL">Danger Strike</Button>
                  <Button variant="outline">Outline</Button>
                  <Button variant="ghost">Ghost</Button>
                </div>
                <div className="flex items-center gap-2 pt-2 border-t border-[var(--c-border-faint)]">
                  <Button size="xs" variant="secondary">Extra Small</Button>
                  <Button size="sm" variant="secondary">Small</Button>
                  <Button size="md" variant="secondary">Medium</Button>
                  <Button size="lg" variant="primary">Large Launcher</Button>
                </div>
                <div className="flex items-center gap-2 pt-2 border-t border-[var(--c-border-faint)]">
                  <Button variant="primary" loading>Acquiring Target...</Button>
                  <Button variant="secondary" disabled>Disabled State</Button>
                </div>
              </div>
            </Panel>

            <Panel title="Contact Badges (Shape-Encoded)">
              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="hostile" pulse>Hostile Drone (▲)</Badge>
                  <Badge variant="unknown">Unknown (◆)</Badge>
                  <Badge variant="friendly">Friendly (●)</Badge>
                  <Badge variant="decoy">Decoy (□)</Badge>
                  <Badge variant="neutral">Neutral (○)</Badge>
                </div>
                <div className="flex flex-wrap gap-2 pt-2 border-t border-[var(--c-border-faint)]">
                  <Badge variant="info">Info</Badge>
                  <Badge variant="advisory">Advisory</Badge>
                  <Badge variant="warning">Warning</Badge>
                  <Badge variant="critical">Critical</Badge>
                </div>
                <p className="text-[11px] font-mono text-[var(--c-text-low)] pt-1">
                  Note: Never encode state by color alone. Every badge pairs color with standard MIL-STD geometric shapes.
                </p>
              </div>
            </Panel>
          </div>

          {/* Row 2: Stats & Progress */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Stat
              label="Active Tracks"
              value="14"
              unit="TARGETS"
              delta={{ value: '+3', trend: 'up' }}
              sparkline={[4, 6, 8, 7, 9, 12, 14]}
            />
            <Stat
              label="Threat Intercept Rate"
              value="96.4"
              unit="%"
              delta={{ value: '+1.2%', trend: 'up' }}
              sparkline={[88, 90, 92, 91, 95, 96.4]}
            />
            <Stat
              label="Radar Noise Floor"
              value="-92"
              unit="dBm"
              delta={{ value: '0.0', trend: 'neutral' }}
              sparkline={[-92, -91, -93, -92, -92]}
            />
            <Stat
              label="Mean Classification Time"
              value="1.84"
              unit="SEC"
              delta={{ value: '-0.3s', trend: 'up' }}
              sparkline={[2.8, 2.4, 2.1, 1.95, 1.84]}
            />
          </div>

          {/* Row 3: Controls (Segmented, Slider, Select, Tooltips) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Panel title="Segmented Switch">
              <div className="space-y-4">
                <Segmented
                  value={segValue}
                  onChange={setSegValue}
                  options={[
                    { value: '2d', label: '2D SCOPE' },
                    { value: '3d', label: '3D TERRAIN' },
                    { value: 'vr', label: 'VR DOME' },
                  ]}
                />
                <div className="text-xs font-mono text-[var(--c-text-mid)]">
                  Active Display Mode: <span className="text-[var(--c-accent)] font-bold">{segValue.toUpperCase()}</span>
                </div>
              </div>
            </Panel>

            <Panel title="Range Slider & Inputs">
              <Slider
                label="Radar Gain Sensitivity"
                min={0}
                max={100}
                value={sliderVal}
                onChange={setSliderVal}
                unit="dB"
              />
            </Panel>

            <Panel title="Tactical Select & Tooltip">
              <div className="space-y-4">
                <Select
                  label="Simulated Threat Profile"
                  value={selectVal}
                  onChange={(e) => setSelectVal(e.target.value)}
                  options={[
                    { value: 'hostile', label: 'Shahed-136 Kamikaze Profile' },
                    { value: 'swarm', label: 'Distributed 8-Drone Swarm' },
                    { value: 'recon', label: 'FPV Low-Altitude Reconnaissance' },
                    { value: 'friendly', label: 'Civilian Commercial Delivery Drone' },
                  ]}
                />
                <div className="pt-2">
                  <Tooltip content="Sensor Outage Injection overrides active antenna telemetry">
                    <Button size="xs" variant="secondary">Hover for Tactical Tooltip</Button>
                  </Tooltip>
                </div>
              </div>
            </Panel>
          </div>

          {/* Row 4: Modals, Drawers & Overlays */}
          <Panel title="Modal Dialogs & Slide-In Drawers">
            <div className="flex flex-wrap gap-3">
              <Button variant="secondary" onClick={() => setModalOpen(true)}>
                Open Mission Briefing Modal
              </Button>
              <Button variant="secondary" onClick={() => setDrawerOpen(true)}>
                Open Telemetry Diagnostics Drawer
              </Button>
            </div>
          </Panel>

          {/* Row 5: Dense Data Table */}
          <Panel title="Tactical Contact Table (Sticky Header & Sortable)">
            <Table
              columns={tableColumns}
              data={mockContacts}
              keyField="id"
              selectedKey={selectedContactId || undefined}
              onRowClick={(c) => setSelectedContactId(c.id)}
            />
          </Panel>
        </div>
      )}

      {/* Tab 2: Domain Components (Radar & Contact List) */}
      {activeTab === 'domain' && (
        <div className="space-y-6">
          <SensorHealthChips sensorOutage={null} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[580px]">
            {/* Radar Scope */}
            <div className="lg:col-span-2 h-full border border-[var(--c-border-base)] rounded-sm overflow-hidden flex flex-col">
              <div className="bg-[var(--c-bg-deep)] px-3 py-2 border-b border-[var(--c-border-base)] flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-[var(--c-accent)] tracking-wider">
                  RADAR SCANNER // 10KM AIR-DEFENSE SECTOR
                </span>
                <span className="text-[var(--c-text-low)]">SWEEP: 15 RPM</span>
              </div>
              <div className="flex-1 min-h-0">
                <RadarScope
                  contacts={mockContacts}
                  selectedContactId={selectedContactId}
                  onSelectContact={(id) => setSelectedContactId(id)}
                />
              </div>
            </div>

            {/* Contacts & Event Feed */}
            <div className="flex flex-col gap-4 h-full">
              <Panel
                title="Tactical Tracks"
                subtitle={`${mockContacts.length} TRACKS`}
                className="flex-1 min-h-0 overflow-hidden"
              >
                <div className="space-y-2 overflow-y-auto max-h-[260px] pr-1">
                  {mockContacts.map((c) => (
                    <ContactRow
                      key={c.id}
                      contact={c}
                      isSelected={c.id === selectedContactId}
                      onSelect={(id) => setSelectedContactId(id)}
                      onClassify={(id, classification) =>
                        toast('CLASSIFICATION', `Track ${id} classified as ${classification}`, 'info')
                      }
                      onSensorRead={(id, sensor) =>
                        toast('SENSOR PING', `Requested ${sensor} scan on track ${id}`, 'info')
                      }
                      onEngage={(id) =>
                        toast('ENGAGEMENT ORDER', `Kinetic strike ordered on ${id}`, 'error')
                      }
                    />
                  ))}
                </div>
              </Panel>

              <Panel title="Combat Events Feed" className="h-56">
                <EventLog events={mockEvents} maxHeight="h-44" />
              </Panel>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Design Tokens */}
      {activeTab === 'tokens' && (
        <div className="space-y-6">
          <Panel title="Tactical Color Palette">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-xs font-mono">
              <div className="p-3 rounded-xs border border-[var(--c-border-base)] bg-[var(--c-bg-void)]">
                <div className="h-8 rounded bg-[var(--c-bg-void)] border border-[var(--c-border-strong)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-semibold">bg-void</span>
                <span className="text-[var(--c-text-low)] text-[10px]">#020508</span>
              </div>
              <div className="p-3 rounded-xs border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-bg-surface)] border border-[var(--c-border-strong)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-semibold">bg-surface</span>
                <span className="text-[var(--c-text-low)] text-[10px]">#08121e</span>
              </div>
              <div className="p-3 rounded-xs border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-accent)] mb-2 shadow-[0_0_8px_var(--c-accent)]" />
                <span className="text-[var(--c-text-high)] block font-semibold">accent</span>
                <span className="text-[var(--c-text-low)] text-[10px]">#00e87a</span>
              </div>
              <div className="p-3 rounded-xs border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-hostile)] mb-2 shadow-[0_0_8px_var(--c-hostile)]" />
                <span className="text-[var(--c-text-high)] block font-semibold">hostile</span>
                <span className="text-[var(--c-text-low)] text-[10px]">#ff3b3b</span>
              </div>
              <div className="p-3 rounded-xs border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-unknown)] mb-2 shadow-[0_0_8px_var(--c-unknown)]" />
                <span className="text-[var(--c-text-high)] block font-semibold">unknown</span>
                <span className="text-[var(--c-text-low)] text-[10px]">#ffb800</span>
              </div>
              <div className="p-3 rounded-xs border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-friendly)] mb-2 shadow-[0_0_8px_var(--c-friendly)]" />
                <span className="text-[var(--c-text-high)] block font-semibold">friendly</span>
                <span className="text-[var(--c-text-low)] text-[10px]">#0a84ff</span>
              </div>
            </div>
          </Panel>

          <Panel title="Typography System">
            <div className="space-y-4 font-mono text-xs">
              <div>
                <span className="text-[10px] text-[var(--c-text-low)] uppercase block mb-1">
                  Display Header (Chakra Petch)
                </span>
                <h2 className="font-display text-xl font-bold tracking-wider text-[var(--c-text-max)]">
                  COUNTER-UNMANNED AIRCRAFT SYSTEM (C-UAS) READINESS
                </h2>
              </div>
              <div className="border-t border-[var(--c-border-faint)] pt-3">
                <span className="text-[10px] text-[var(--c-text-low)] uppercase block mb-1">
                  UI Chrome & Body (Geist)
                </span>
                <p className="font-ui text-sm text-[var(--c-text-high)] max-w-xl">
                  Operators must identify and classify radar tracks within engagement windows. Rules of engagement dictate zero collateral damage.
                </p>
              </div>
              <div className="border-t border-[var(--c-border-faint)] pt-3">
                <span className="text-[10px] text-[var(--c-text-low)] uppercase block mb-1">
                  Data & Telemetry Readouts (Geist Mono with Tabular Nums)
                </span>
                <div className="text-base text-[var(--c-accent)] tabular-nums tracking-widest font-bold">
                  BRG 032.4° // RNG 04.82 KM // ALT 0180 M // VEL +028.5 M/S
                </div>
              </div>
            </div>
          </Panel>
        </div>
      )}

      {/* Modals & Drawers */}
      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title="Tactical Mission Briefing: Operation SkyShield"
        actions={
          <>
            <Button variant="ghost" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button variant="primary" onClick={() => setModalOpen(false)}>Acknowledge ROE</Button>
          </>
        }
      >
        <div className="space-y-3">
          <p>
            Hostile micro-drones have been reported operating in the perimeter sector. Rules of Engagement (ROE) mandate that weapons hold is in effect until target classification is confirmed by at least two sensor channels (Radar + RF or EO/IR).
          </p>
          <div className="p-2.5 bg-[var(--c-bg-deep)] border border-[var(--c-border-faint)] rounded-xs">
            <span className="text-[var(--c-accent)] font-semibold block mb-1">PRIMARY DIRECTIVE</span>
            Identify all incoming contacts. Distinguish decoys and friendly delivery platforms before engaging hostile kamikaze swarm elements.
          </div>
        </div>
      </Modal>

      <Drawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        title="Telemetry Diagnostics"
      >
        <div className="space-y-4">
          <Stat label="Antenna Temp" value="38.4" unit="°C" />
          <Stat label="Azimuth Motor Latency" value="4.2" unit="MS" />
          <div className="pt-2">
            <span className="text-[10px] text-[var(--c-text-low)] uppercase block mb-2">Sensor Subsystems</span>
            <SensorHealthChips />
          </div>
        </div>
      </Drawer>
    </div>
  );
};
