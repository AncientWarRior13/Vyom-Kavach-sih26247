export type Classification = 
  | 'hostile_drone'
  | 'hostile_swarm_member'
  | 'friendly_drone'
  | 'decoy'
  | 'bird'
  | 'unknown';

export type ContactState = 
  | 'undetected'
  | 'detected'
  | 'tracking'
  | 'classified'
  | 'engaged'
  | 'neutralized'
  | 'destroyed'
  | 'escaped';

export interface Contact {
  id: string;
  x: number;
  y: number;
  altitude: number;
  heading: number; // degrees
  speed: number;   // m/s
  range_km: number;
  bearing_deg: number;
  rcs?: number;
  threat_level?: number;
  classification?: Classification;
  assigned_classification?: Classification;
  state: ContactState;
  engagement_action?: string;
  despawn_t?: number;
  history?: Array<{ x: number; y: number; t: number }>;
  sensor_readings?: {
    radar?: { rcs: number; doppler_hz: number; snr_db: number };
    eo_ir?: { thermal_signature: string; visual_id?: string; confidence: number };
    rf?: { frequency_ghz: number; protocol?: string; signal_strength_dbm: number };
    acoustic?: { blade_count?: number; rpm?: number; dba: number };
  };
}

export interface Scenario {
  scenario_id: string;
  difficulty: number;
  time_of_day: string;
  terrain: string;
  sensor_outage?: string | null;
  roe: {
    description: string;
    allowed_engagement: string[];
    weapons_hold: boolean;
  };
  contacts: Contact[];
  started_at: number;
  time_limit_sec?: number;
  degrade_sensors?: boolean;
}

export interface ModelStatus {
  status: string;
  weights_loaded?: boolean;
  model_name?: string;
  device?: string;
  latency_ms?: number;
  accuracy?: number;
  last_updated?: string;
}

export interface ClassifyRequest {
  scenario_id: string;
  contact_id: string;
  classification: Classification;
  justification?: string;
}

export interface SensorReadRequest {
  scenario_id: string;
  contact_id: string;
  sensor: 'radar' | 'eo_ir' | 'rf' | 'acoustic' | string;
  t: number;
}

export interface EngageRequest {
  scenario_id: string;
  contact_id: string;
  weapon_type: 'rf_jam' | 'kinetic_net' | 'high_power_laser' | 'emp';
}

export interface EngageRequestV2 {
  scenario_id: string;
  contact_id: string;
  t: number;
  action?: 'hard_kill' | 'warn' | 'soft_kill';
}

export interface EngageResponseV2 {
  reactive: boolean;
  contacts: any[];
}

export interface SubmitSessionRequest {
  scenario_id: string;
  trainee_id: string;
  actions: any[];
  duration_seconds: number;
}
