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
    undetected: { label: 'Undetected', variant: 'default' },
    detected: { label: 'Detected', variant: 'info' },
    tracking: { label: 'Tracking', variant: 'warning' },
    classified: { label: 'Classified', variant: 'advisory' },
    engaged: { label: 'Engaging', variant: 'critical' },
    neutralized: { label: 'Neutralized', variant: 'friendly' },
    escaped: { label: 'Escaped', variant: 'default' },
  };

  const stateInfo = stateBadges[contact.state] || { label: contact.state, variant: 'default' };

  return (
    <div
      onClick={() => onSelect(contact.id)}
      className={`border rounded-md transition-colors p-3 flex flex-col gap-2.5 cursor-pointer select-none ${
        isSelected
          ? 'bg-[var(--c-bg-raised)] border-[var(--c-accent)] shadow-sm'
          : 'bg-[var(--c-bg-surface)] border-[var(--c-border-base)] hover:border-[var(--c-border-strong)] hover:bg-[var(--c-bg-overlay)]/40'
      }`}
    >
      {/* Contact Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant={cls} size="xs" pulse={contact.state === 'tracking'}>
            {cls.replace('_', ' ')}
          </Badge>
          <span className="font-mono font-medium text-xs text-[var(--c-text-high)]">
            {contact.id}
          </span>
        </div>
        <Badge variant={stateInfo.variant} size="xs">
          {stateInfo.label}
        </Badge>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-4 gap-2 text-xs font-ui bg-[var(--c-bg-deep)]/70 p-2 rounded-md border border-[var(--c-border-base)]">
        <div>
          <span className="text-[11px] text-[var(--c-text-low)] block">Range</span>
          <span className="text-[var(--c-text-high)] font-mono font-medium tabular-nums text-xs">
            {contact.range_km.toFixed(1)} km
          </span>
        </div>
        <div>
          <span className="text-[11px] text-[var(--c-text-low)] block">Bearing</span>
          <span className="text-[var(--c-text-high)] font-mono font-medium tabular-nums text-xs">
            {Math.round(contact.bearing_deg)}°
          </span>
        </div>
        <div>
          <span className="text-[11px] text-[var(--c-text-low)] block">Altitude</span>
          <span className="text-[var(--c-text-high)] font-mono font-medium tabular-nums text-xs">
            {contact.altitude} m
          </span>
        </div>
        <div>
          <span className="text-[11px] text-[var(--c-text-low)] block">Speed</span>
          <span className="text-[var(--c-text-high)] font-mono font-medium tabular-nums text-xs">
            {contact.speed} m/s
          </span>
        </div>
      </div>

      {/* Actions (Rendered when selected) */}
      {isSelected && (
        <div className="pt-2 border-t border-[var(--c-border-base)] flex flex-col gap-2">
          {/* Quick Sensor Scans */}
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-ui text-[var(--c-text-low)] font-medium">
              Scans:
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
              Acoustic
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
                Hostile
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
                Friendly
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
                Decoy
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
                Engage
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
