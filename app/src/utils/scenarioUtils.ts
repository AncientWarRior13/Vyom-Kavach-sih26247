import type { Contact, Scenario, Classification, ContactState } from '../api/types';

export interface RawWaypoint {
  t: number;
  x: number;
  y: number;
  z?: number;
}

export interface RawContact {
  contact_id?: string;
  id?: string;
  spawn_t?: number;
  despawn_t?: number;
  path?: RawWaypoint[];
  classification?: Classification;
  assigned_classification?: Classification;
  state?: ContactState;
  [key: string]: any;
}

/**
 * Interpolates a contact's position at elapsed time `t`
 */
export function posAt(contact: RawContact, t: number): { x: number; y: number; z: number } {
  const pts = contact.path;
  if (!pts || pts.length === 0) {
    return {
      x: contact.x ?? 0,
      y: contact.y ?? 0,
      z: contact.altitude ?? contact.z ?? 120,
    };
  }

  if (t <= pts[0].t) {
    return { x: pts[0].x, y: pts[0].y, z: pts[0].z ?? 120 };
  }

  if (t >= pts[pts.length - 1].t) {
    const last = pts[pts.length - 1];
    return { x: last.x, y: last.y, z: last.z ?? 120 };
  }

  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (a.t <= t && t <= b.t) {
      const dt = b.t - a.t;
      const f = dt <= 0 ? 0 : (t - a.t) / dt;
      return {
        x: a.x + (b.x - a.x) * f,
        y: a.y + (b.y - a.y) * f,
        z: (a.z ?? 120) + ((b.z ?? 120) - (a.z ?? 120)) * f,
      };
    }
  }

  const last = pts[pts.length - 1];
  return { x: last.x, y: last.y, z: last.z ?? 120 };
}

/**
 * Calculates current heading in degrees (0 = North, 90 = East, 180 = South, 270 = West)
 */
export function calculateHeading(contact: RawContact, t: number): number {
  const pts = contact.path;
  if (!pts || pts.length < 2) return 0;

  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (a.t <= t && t <= b.t) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      return ((Math.atan2(dx, -dy) * 180 / Math.PI) + 360) % 360;
    }
  }

  const p1 = pts[pts.length - 2];
  const p2 = pts[pts.length - 1];
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return ((Math.atan2(dx, -dy) * 180 / Math.PI) + 360) % 360;
}

/**
 * Calculates current speed in m/s
 */
export function calculateSpeed(contact: RawContact, t: number): number {
  const pts = contact.path;
  if (!pts || pts.length < 2) return 24;

  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    if (a.t <= t && t <= b.t) {
      const dt = b.t - a.t;
      if (dt <= 0) return 24;
      const dist = Math.hypot(b.x - a.x, b.y - a.y);
      return Math.round(dist / dt);
    }
  }

  const p1 = pts[pts.length - 2];
  const p2 = pts[pts.length - 1];
  const dt = p2.t - p1.t;
  if (dt <= 0) return 24;
  return Math.round(Math.hypot(p2.x - p1.x, p2.y - p1.y) / dt);
}

/**
 * Enriches a contact with computed kinematics, range, bearing, and history trail
 */
export function enrichContact(
  raw: RawContact,
  t: number,
  traineeLog: Record<string, any> = {}
): Contact {
  const id = raw.contact_id || raw.id || 'TRACK';
  const pos = posAt(raw, t);
  const range_m = Math.hypot(pos.x, pos.y);
  const range_km = Number((range_m / 1000).toFixed(2));
  const bearing_deg = Number((((Math.atan2(pos.x, -pos.y) * 180 / Math.PI) + 360) % 360).toFixed(1));
  const heading = calculateHeading(raw, t);
  const entry = traineeLog[id];
  const isNeutralized =
    entry?.state === 'neutralized' ||
    entry?.state === 'destroyed' ||
    entry?.engagement_action === 'hard_kill' ||
    raw.state === 'neutralized' ||
    raw.state === 'destroyed' ||
    raw.engagement_action === 'hard_kill' ||
    (raw.despawn_t !== undefined && raw.despawn_t > 0 && t >= raw.despawn_t);

  const speed = isNeutralized ? 0 : calculateSpeed(raw, t);
  const altitude = isNeutralized ? 0 : Math.round(pos.z ?? 120);

  const assigned_classification = (entry?.classification || raw.assigned_classification || 'unknown') as Classification;
  const classification = (raw.classification || 'unknown') as Classification;

  let state: ContactState = 'undetected';
  if (isNeutralized) state = 'neutralized';
  else if (entry?.engagement_action) state = 'engaged';
  else if (entry?.classification) state = 'classified';
  else if (entry?.detected_at !== undefined) state = 'detected';
  else if (raw.state) state = raw.state;

  // Build breadcrumb trail from past waypoints + current position
  const history: Array<{ x: number; y: number; t: number }> = [];
  if (raw.path) {
    for (const wp of raw.path) {
      if (wp.t <= t) {
        history.push({ x: wp.x, y: wp.y, t: wp.t });
      }
    }
  }
  history.push({ x: pos.x, y: pos.y, t });

  return {
    id,
    x: pos.x,
    y: pos.y,
    altitude,
    heading,
    speed,
    range_km,
    bearing_deg,
    classification,
    assigned_classification,
    state,
    engagement_action: isNeutralized ? 'hard_kill' : (entry?.engagement_action || raw.engagement_action),
    despawn_t: raw.despawn_t,
    history,
    rcs: raw.rcs,
    threat_level: raw.threat_level,
    sensor_readings: raw.sensor_readings,
  };
}

/**
 * Normalizes all contacts in a scenario for the current simulation time `t`
 */
export function enrichScenarioContacts(
  rawContacts: RawContact[] | undefined,
  t: number,
  traineeLog: Record<string, any> = {}
): Contact[] {
  if (!rawContacts || !Array.isArray(rawContacts)) return [];
  return rawContacts.map((c) => enrichContact(c, t, traineeLog));
}
