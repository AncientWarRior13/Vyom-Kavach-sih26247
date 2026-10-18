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
import { RadarScope } from '../components/domain/RadarScope';
import { ContactRow } from '../components/domain/ContactRow';
import { SensorHealthChips, EventLog } from '../components/domain/TacticalFeed';
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
      heading: 215,
      classification: 'hostile_drone',
      state: 'classified',
      history: [
        { x: 2.8, y: 4.8, t: 1 },
        { x: 2.6, y: 4.4, t: 2 },
      ],
    },
    {
      id: 'tgt-02',
      x: -3.1,
      y: 2.2,
      range_km: 3.8,
      bearing_deg: 305,
      altitude: 95,
      speed: 34,
      heading: 120,
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
    { key: 'id', header: 'Track ID', sortable: true, render: (c) => <span className="font-mono text-xs">{c.id}</span> },
    {
      key: 'classification',
      header: 'Classification',
      render: (c) => {
        const v =
          c.classification === 'hostile_drone'
            ? 'hostile'
            : c.classification === 'friendly_drone'
            ? 'friendly'
            : c.classification === 'decoy'
            ? 'decoy'
            : 'unknown';
        return <Badge variant={v} size="xs">{(c.classification || 'unknown').replace('_', ' ')}</Badge>;
      },
    },
    { key: 'range_km', header: 'Range', sortable: true, align: 'right', render: (c) => <span className="font-mono text-xs">{c.range_km.toFixed(1)} km</span> },
    { key: 'bearing_deg', header: 'Bearing', sortable: true, align: 'right', render: (c) => <span className="font-mono text-xs">{c.bearing_deg}°</span> },
    { key: 'altitude', header: 'Altitude', sortable: true, align: 'right', render: (c) => <span className="font-mono text-xs">{c.altitude} m</span> },
    { key: 'speed', header: 'Speed', sortable: true, align: 'right', render: (c) => <span className="font-mono text-xs">{c.speed} m/s</span> },
  ];

  const mockEvents = [
    { id: '1', timestamp: '04:12:02Z', level: 'warning' as const, source: 'Radar', message: 'New high-speed contact detected in sector 4.' },
    { id: '2', timestamp: '04:12:08Z', level: 'info' as const, source: 'Optics', message: 'Optical channel locked on track tgt-01.' },
    { id: '3', timestamp: '04:12:14Z', level: 'critical' as const, source: 'Classifier', message: 'RF profile matches Shahed-136 delta-wing.' },
    { id: '4', timestamp: '04:12:18Z', level: 'success' as const, source: 'C-UAS', message: 'Kinetic intercept net deployed. Target neutralized.' },
  ];

  return (
    <div className="p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--c-border-base)] pb-4">
        <div>
          <h1 className="text-lg font-semibold text-[var(--c-text-high)] font-ui">
            Design system
          </h1>
          <p className="text-xs text-[var(--c-text-mid)] font-ui mt-0.5">
            Calm enterprise operations token architecture, UI components, and domain widgets.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => toast('Notification', 'Design tokens verified and active', 'success')}
          >
            Trigger toast
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs
        activeTab={activeTab}
        onChange={setActiveTab}
        tabs={[
          { id: 'components', label: 'UI components' },
          { id: 'domain', label: 'Domain widgets' },
          { id: 'tokens', label: 'Design tokens' },
        ]}
      />

      {/* ── Tab 1: Components ── */}
      {activeTab === 'components' && (
        <div className="space-y-6">
          {/* Row 1: Buttons & Badges */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Panel title="Button variants and keyboard hints">
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <Button variant="primary" kbd="SPC">Primary</Button>
                  <Button variant="secondary" kbd="TAB">Secondary</Button>
                  <Button variant="danger" kbd="DEL">Danger</Button>
                  <Button variant="outline">Outline</Button>
                  <Button variant="ghost">Ghost</Button>
                </div>
                <div className="flex items-center gap-2 pt-3 border-t border-[var(--c-border-base)]">
                  <Button size="xs" variant="secondary">Extra small</Button>
                  <Button size="sm" variant="secondary">Small</Button>
                  <Button size="md" variant="secondary">Medium</Button>
                  <Button size="lg" variant="primary">Large action</Button>
                </div>
                <div className="flex items-center gap-2 pt-3 border-t border-[var(--c-border-base)]">
                  <Button variant="primary" loading>Acquiring target...</Button>
                  <Button variant="secondary" disabled>Disabled state</Button>
                </div>
              </div>
            </Panel>

            <Panel title="Status badges with geometric shapes">
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="hostile">Hostile drone</Badge>
                  <Badge variant="unknown">Unknown contact</Badge>
                  <Badge variant="friendly">Friendly drone</Badge>
                  <Badge variant="decoy">Decoy track</Badge>
                  <Badge variant="neutral">Neutral bird</Badge>
                </div>
                <div className="flex flex-wrap gap-2 pt-3 border-t border-[var(--c-border-base)]">
                  <Badge variant="info">Information</Badge>
                  <Badge variant="advisory">Advisory</Badge>
                  <Badge variant="warning">Warning</Badge>
                  <Badge variant="critical">Critical</Badge>
                </div>
                <p className="text-xs text-[var(--c-text-low)] font-ui pt-1 leading-relaxed">
                  Every status badge pairs a color with a standard geometric shape (▲, ◆, ●, □, ○) so status is never conveyed by color alone.
                </p>
              </div>
            </Panel>
          </div>

          {/* Row 2: Stats */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Stat
              label="Active tracks"
              value="14"
              unit="targets"
              sparkline={[4, 6, 8, 7, 9, 12, 14]}
            />
            <Stat
              label="Threat intercept rate"
              value="96.4"
              unit="%"
              sparkline={[88, 90, 92, 91, 95, 96.4]}
            />
            <Stat
              label="Radar noise floor"
              value="-92"
              unit="dBm"
              sparkline={[-92, -91, -93, -92, -92]}
            />
            <Stat
              label="Mean classification time"
              value="1.84"
              unit="sec"
              sparkline={[2.8, 2.4, 2.1, 1.95, 1.84]}
            />
          </div>

          {/* Row 3: Controls */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Panel title="Segmented switch">
              <div className="space-y-3">
                <Segmented
                  value={segValue}
                  onChange={setSegValue}
                  options={[
                    { value: '2d', label: '2D scope' },
                    { value: '3d', label: '3D terrain' },
                    { value: 'vr', label: 'VR dome' },
                  ]}
                />
                <div className="text-xs text-[var(--c-text-mid)] font-ui">
                  Active mode: <span className="text-[var(--c-text-high)] font-medium">{segValue}</span>
                </div>
              </div>
            </Panel>

            <Panel title="Range slider">
              <Slider
                label="Radar gain sensitivity"
                min={0}
                max={100}
                value={sliderVal}
                onChange={setSliderVal}
                unit="dB"
              />
            </Panel>

            <Panel title="Select dropdown and tooltip">
              <div className="space-y-4">
                <Select
                  label="Simulated threat profile"
                  value={selectVal}
                  onChange={(e) => setSelectVal(e.target.value)}
                  options={[
                    { value: 'hostile', label: 'Shahed-136 profile' },
                    { value: 'swarm', label: 'Autonomous swarm (6 nodes)' },
                    { value: 'recon', label: 'FPV low-altitude scout' },
                    { value: 'friendly', label: 'Commercial delivery drone' },
                  ]}
                />
                <div className="pt-1">
                  <Tooltip content="Overrides active antenna telemetry readings">
                    <Button size="xs" variant="secondary">Hover for tooltip</Button>
                  </Tooltip>
                </div>
              </div>
            </Panel>
          </div>

          {/* Row 4: Modals & Drawers */}
          <Panel title="Modals and slide-in drawers">
            <div className="flex flex-wrap gap-3">
              <Button variant="secondary" onClick={() => setModalOpen(true)}>
                Open briefing modal
              </Button>
              <Button variant="secondary" onClick={() => setDrawerOpen(true)}>
                Open diagnostics drawer
              </Button>
            </div>
          </Panel>

          {/* Row 5: Data Table */}
          <Panel title="Contact table" subtitle="Sortable with sticky header">
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

      {/* ── Tab 2: Domain Components ── */}
      {activeTab === 'domain' && (
        <div className="space-y-6">
          <SensorHealthChips sensorOutage={null} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[540px]">
            {/* Radar Scope */}
            <div className="lg:col-span-2 h-full border border-[var(--c-border-base)] rounded-lg overflow-hidden flex flex-col bg-[var(--c-bg-surface)]">
              <div className="bg-[var(--c-bg-deep)] px-4 py-3 border-b border-[var(--c-border-base)] flex items-center justify-between text-xs font-ui">
                <span className="font-semibold text-[var(--c-text-high)]">
                  Radar scanner (10 km defense sector)
                </span>
                <span className="text-[var(--c-text-low)] font-mono">15 RPM</span>
              </div>
              <div className="flex-1 min-h-[400px]">
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
                title="Active tracks"
                subtitle={`${mockContacts.length} contacts`}
                className="flex-1 min-h-0 overflow-hidden"
              >
                <div className="space-y-2 overflow-y-auto max-h-[250px] pr-1">
                  {mockContacts.map((c) => (
                    <ContactRow
                      key={c.id}
                      contact={c}
                      isSelected={c.id === selectedContactId}
                      onSelect={(id) => setSelectedContactId(id)}
                      onClassify={(id, classification) =>
                        toast('Classification', `Track ${id} marked as ${classification}`, 'info')
                      }
                      onSensorRead={(id, sensor) =>
                        toast('Sensor read', `Requested ${sensor} scan on track ${id}`, 'info')
                      }
                      onEngage={(id) =>
                        toast('Engagement', `Kinetic strike ordered on ${id}`, 'error')
                      }
                    />
                  ))}
                </div>
              </Panel>

              <Panel title="Event feed" className="h-56">
                <EventLog events={mockEvents} maxHeight="h-40" />
              </Panel>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 3: Design Tokens ── */}
      {activeTab === 'tokens' && (
        <div className="space-y-6">
          <Panel title="Surface and accent palette">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-xs font-ui">
              <div className="p-3 rounded-lg border border-[var(--c-border-base)] bg-[var(--c-bg-void)]">
                <div className="h-8 rounded bg-[var(--c-bg-void)] border border-[var(--c-border-strong)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-medium">bg-void</span>
                <span className="text-[var(--c-text-low)] font-mono text-[11px]">#0B0E13</span>
              </div>
              <div className="p-3 rounded-lg border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-bg-surface)] border border-[var(--c-border-strong)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-medium">bg-surface</span>
                <span className="text-[var(--c-text-low)] font-mono text-[11px]">#11151C</span>
              </div>
              <div className="p-3 rounded-lg border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-bg-raised)] border border-[var(--c-border-strong)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-medium">bg-raised</span>
                <span className="text-[var(--c-text-low)] font-mono text-[11px]">#181D26</span>
              </div>
              <div className="p-3 rounded-lg border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-accent)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-medium">accent</span>
                <span className="text-[var(--c-text-low)] font-mono text-[11px]">#3DD6C6</span>
              </div>
              <div className="p-3 rounded-lg border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-hostile)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-medium">status-critical</span>
                <span className="text-[var(--c-text-low)] font-mono text-[11px]">#EF4444</span>
              </div>
              <div className="p-3 rounded-lg border border-[var(--c-border-base)] bg-[var(--c-bg-surface)]">
                <div className="h-8 rounded bg-[var(--c-advisory)] mb-2" />
                <span className="text-[var(--c-text-high)] block font-medium">status-success</span>
                <span className="text-[var(--c-text-low)] font-mono text-[11px]">#10B981</span>
              </div>
            </div>
          </Panel>

          <Panel title="Typography hierarchy">
            <div className="space-y-4 text-xs font-ui">
              <div>
                <span className="text-xs text-[var(--c-text-low)] block mb-1">
                  Logo font (Chakra Petch)
                </span>
                <div className="font-logo text-xl font-bold text-[var(--c-text-max)]">
                  Vyom Kavach C-UAS OPERATIONS
                </div>
              </div>
              <div className="border-t border-[var(--c-border-base)] pt-3">
                <span className="text-xs text-[var(--c-text-low)] block mb-1">
                  Primary UI & Headings (Geist)
                </span>
                <h3 className="text-base font-semibold text-[var(--c-text-high)]">
                  Tactical mission operations and airspace defense
                </h3>
                <p className="text-sm text-[var(--c-text-mid)] mt-1 max-w-xl leading-relaxed">
                  Operators evaluate sensor telemetry, categorize incoming threats, and authorize soft-kill or kinetic countermeasures within rules of engagement.
                </p>
              </div>
              <div className="border-t border-[var(--c-border-base)] pt-3">
                <span className="text-xs text-[var(--c-text-low)] block mb-1">
                  Numeric & Telemetry Readouts (Geist Mono tabular numbers)
                </span>
                <div className="text-sm font-mono tabular-nums text-[var(--c-text-high)]">
                  Bearing: 032.4° · Range: 4.82 km · Altitude: 180 m · Velocity: 28.5 m/s
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
        title="Mission briefing: Operation SkyShield"
        actions={
          <>
            <Button variant="ghost" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button variant="primary" onClick={() => setModalOpen(false)}>Acknowledge ROE</Button>
          </>
        }
      >
        <div className="space-y-3 text-xs font-ui text-[var(--c-text-high)] leading-relaxed">
          <p>
            Hostile micro-drones have been detected operating near the defended perimeter. Rules of Engagement mandate weapons hold until contact classification is verified across multiple sensor channels.
          </p>
          <div className="p-3 bg-[var(--c-bg-deep)] border border-[var(--c-border-base)] rounded-md">
            <span className="font-semibold text-[var(--c-text-high)] block mb-1">Primary directive</span>
            Identify all incoming contacts. Distinguish decoys and friendly delivery platforms before engaging hostile kamikaze swarm elements.
          </div>
        </div>
      </Modal>

      <Drawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        title="Telemetry diagnostics"
      >
        <div className="space-y-4 text-xs font-ui">
          <Stat label="Antenna temperature" value="38.4" unit="°C" />
          <Stat label="Azimuth motor latency" value="4.2" unit="ms" />
          <div className="pt-2">
            <span className="text-xs font-medium text-[var(--c-text-low)] block mb-2">Sensor subsystems</span>
            <SensorHealthChips />
          </div>
        </div>
      </Drawer>
    </div>
  );
};
