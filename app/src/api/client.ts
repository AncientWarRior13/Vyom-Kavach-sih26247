import type {
  Scenario,
  ModelStatus,
  ClassifyRequest,
  SensorReadRequest,
  EngageRequest,
  SubmitSessionRequest,
} from './types';

const BASE_URL = '';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${url}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!res.ok) {
    let errMsg = `Request failed: ${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data && data.detail) errMsg = data.detail;
    } catch {}
    throw new Error(errMsg);
  }
  return res.json();
}

export const api = {
  getModelStatus: () => fetchJson<ModelStatus>('/api/model/status'),
  getScriptedScenarios: () => fetchJson<{ scenarios: string[] }>('/api/scenario/scripted'),
  
  startNewScenario: (data: any) =>
    fetchJson<Scenario>('/api/scenario/new', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  startAdaptiveScenario: (data: any) =>
    fetchJson<Scenario>('/api/scenario/adaptive', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  loadScriptedScenario: (name: string) =>
    fetchJson<Scenario>(`/api/scenario/scripted/${encodeURIComponent(name)}`, {
      method: 'POST',
    }),

  refreshScenario: (scenarioId: string) =>
    fetchJson<Scenario>(`/api/scenario/${encodeURIComponent(scenarioId)}/refresh`),

  classifyContact: (req: ClassifyRequest) =>
    fetchJson<{ success: boolean; contact: any }>('/api/contact/classify', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  sensorRead: (req: SensorReadRequest) =>
    fetchJson<any>('/api/contact/sensor_read', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  engageContact: (req: EngageRequest) =>
    fetchJson<{ success: boolean; result: string }>('/api/contact/engage', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  engageContactV2: (req: {
    scenario_id: string;
    contact_id: string;
    t: number;
    action?: 'hard_kill' | 'warn' | 'soft_kill';
  }) =>
    fetchJson<{ reactive: boolean; contacts: any[] }>('/api/contact/engage', {
      method: 'POST',
      body: JSON.stringify({
        scenario_id: req.scenario_id,
        contact_id: req.contact_id,
        t: req.t,
        action: req.action || 'hard_kill',
      }),
    }),

  submitSession: (req: SubmitSessionRequest) =>
    fetchJson<{ session_id: string; score: number; passed: boolean }>('/api/session/submit', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  getResearchAssignment: (traineeId: string) =>
    fetchJson<any>(`/api/research/assignment/${encodeURIComponent(traineeId)}`),

  instructorInjectThreat: (data: any) =>
    fetchJson<{ injected: boolean; contact: any }>('/api/instructor/inject_threat', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  instructorJamSensor: (data: { scenario_id: string; sensor: string | null }) =>
    fetchJson<{ sensor_outage: string | null }>('/api/instructor/jam_sensor', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
};
