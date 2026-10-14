import React from 'react';
import type { Contact } from '../../api/types';
import { Badge } from '../../ui/Badge';
import { Button } from '../../ui/Button';

export interface ContactRowProps {
  contact: Contact;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onClassify?: (id: string, classification: string) => void;
  onSensorRead?: (id: string, sensorType: string) => void;
  onEngage?: (id: string) => void;
}

export const ContactRow: React.FC<ContactRowProps> = ({
  contact,
  isSelected,
  onSelect,
  onClassify,
  onSensorRead,
  onEngage,
}) => {
  const cls = (contact.assigned_classification || contact.classification || 'unknown') as any;

  const stateBadges: Record<string, { label: string; variant: any }> = {
    undetected: { label: 'UNDETECTED', variant: 'default' },
    detected: { label: 'DETECTED', variant: 'info' },
    tracking: { label: 'TRACKING', variant: 'warning' },
    classified: { label: 'CLASSIFIED', variant: 'advisory' },
    engaged: { label: 'ENGAGING', variant: 'critical' },
    neutralized: { label: 'NEUTRALIZED', variant: 'friendly' },
    escaped: { label: 'ESCAPED', variant: 'default' },
  };

  const stateInfo = stateBadges[contact.state] || { label: contact.state.toUpperCase(), variant: 'default' };

  return (
    <div
      onClick={() => onSelect(contact.id)}
      className={`border rounded-sm transition-all duration-150 p-2.5 flex flex-col gap-2 cursor-pointer select-none ${
        isSelected
          ? 'bg-[var(--c-bg-subtle)] border-[var(--c-accent)] shadow-[0_0_12px_rgba(0,232,122,0.15)]'
          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)] hover:bg-[var(--c-bg-overlay)]/60'
      }`}
    >
      {/* Contact Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant={cls} size="xs" pulse={contact.state === 'tracking'}>
            {cls.replace('_', ' ')}
          </Badge>
          <span className="font-mono font-bold text-xs text-[var(--c-text-max)]">
            {contact.id.toUpperCase()}
          </span>
        </div>
        <Badge variant={stateInfo.variant} size="xs">
          {stateInfo.label}
        </Badge>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-4 gap-1.5 text-[10px] font-mono bg-[var(--c-bg-deep)]/50 p-1.5 rounded-xs border border-[var(--c-border-faint)]">
        <div>
          <span className="text-[var(--c-text-low)] block">RNG</span>
          <span className="text-[var(--c-text-high)] font-semibold tabular-nums">
            {contact.range_km.toFixed(1)} km
          </span>
        </div>
        <div>
          <span className="text-[var(--c-text-low)] block">BRG</span>
          <span className="text-[var(--c-text-high)] font-semibold tabular-nums">
            {Math.round(contact.bearing_deg)}°
          </span>
        </div>
        <div>
          <span className="text-[var(--c-text-low)] block">ALT</span>
          <span className="text-[var(--c-text-high)] font-semibold tabular-nums">
            {contact.altitude} m
          </span>
        </div>
        <div>
          <span className="text-[var(--c-text-low)] block">SPD</span>
          <span className="text-[var(--c-text-high)] font-semibold tabular-nums">
            {contact.speed} m/s
          </span>
        </div>
      </div>

      {/* Actions (ONLY rendered when selected to prevent visual clutter) */}
      {isSelected && (
        <div className="pt-2 border-t border-[var(--c-border-base)] flex flex-col gap-2 animate-in fade-in duration-150">
          {/* Quick Sensor Scans */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] font-mono text-[var(--c-text-low)] uppercase tracking-wider">
              SCANS:
            </span>
            <Button
              size="xs"
              variant="secondary"
              kbd="R"
              onClick={(e) => {
                e.stopPropagation();
                onSensorRead && onSensorRead(contact.id, 'rf');
              }}
            >
              RF
            </Button>
            <Button
              size="xs"
              variant="secondary"
              kbd="E"
              onClick={(e) => {
                e.stopPropagation();
                onSensorRead && onSensorRead(contact.id, 'eo_ir');
              }}
            >
              EO/IR
            </Button>
            <Button
              size="xs"
              variant="secondary"
              kbd="A"
              onClick={(e) => {
                e.stopPropagation();
                onSensorRead && onSensorRead(contact.id, 'acoustic');
              }}
            >
              ACOUSTIC
            </Button>
          </div>

          {/* Quick Classification & Engage */}
          <div className="flex items-center justify-between gap-1.5 pt-1">
            <div className="flex items-center gap-1">
              <Button
                size="xs"
                variant="danger"
                kbd="1"
                onClick={(e) => {
                  e.stopPropagation();
                  onClassify && onClassify(contact.id, 'hostile_drone');
                }}
              >
                HOSTILE
              </Button>
              <Button
                size="xs"
                variant="secondary"
                kbd="2"
                onClick={(e) => {
                  e.stopPropagation();
                  onClassify && onClassify(contact.id, 'friendly_drone');
                }}
              >
                FRIENDLY
              </Button>
              <Button
                size="xs"
                variant="secondary"
                kbd="3"
                onClick={(e) => {
                  e.stopPropagation();
                  onClassify && onClassify(contact.id, 'decoy');
                }}
              >
                DECOY
              </Button>
            </div>

            {onEngage && (
              <Button
                size="xs"
                variant="danger"
                kbd="SPC"
                onClick={(e) => {
                  e.stopPropagation();
                  onEngage(contact.id);
                }}
              >
                ENGAGE
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
