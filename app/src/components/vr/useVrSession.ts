import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useSessionStore } from '../../state/session';
import { enrichScenarioContacts } from '../../utils/scenarioUtils';
import type { Contact, Scenario, Classification } from '../../api/types';

export type VrSessionStatus = 'loading' | 'live' | 'empty' | 'error';

export interface ActiveScenarioMeta {
  scenario_id: string;
  duration_s: number;
  contact_count: number;
  sensor_outage?: string | null;
  params: any;
}

export interface UseVrSessionReturn {
  scenarioId: string | null;
  setScenarioId: (id: string | null) => void;
  available: ActiveScenarioMeta[];
  status: VrSessionStatus;
  error: string | null;
  scenario: any | null;
  contacts: Contact[];
  elapsed: number;
  getElapsed: () => number;
}

const STORAGE_KEY = 'vk:scenario_id';
const POLL_INTERVAL_MS = 2500;

export function useVrSession(): UseVrSessionReturn {
  const [searchParams, setSearchParams] = useSearchParams();
  const storeScenarioId = useSessionStore((s) => s.scenarioId);
  const traineeLog = useSessionStore((s) => s.traineeLog);
  const setStoreSession = useSessionStore((s) => s.setSession);

  const [available, setAvailable] = useState<ActiveScenarioMeta[]>([]);
  const [scenarioId, setScenarioIdState] = useState<string | null>(null);
  const [scenario, setScenario] = useState<any | null>(null);
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [status, setStatus] = useState<VrSessionStatus>('loading');
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number>(0);

  const scenarioRef = useRef<any | null>(null);
  scenarioRef.current = scenario;

  const localStartEpochRef = useRef<number>(Date.now());
  const consecutiveFailuresRef = useRef<number>(0);
  const teamClassificationsRef = useRef<Record<string, string>>({});

  // High-precision elapsed time getter for 60fps+ WebGL/XR render loops
  const getElapsed = useCallback((): number => {
    const epoch = scenarioRef.current?.server_start_epoch_ms ?? localStartEpochRef.current;
    if (!epoch) return 0;
    const duration = scenarioRef.current?.duration_s || 300;
    const raw = Math.max(0, (Date.now() - epoch) / 1000);
    return raw > duration ? (raw % duration) : raw;
  }, []);

  // Update scenarioId with storage mirror and state update
  const setScenarioId = useCallback((id: string | null) => {
    setScenarioIdState(id);
    setStoreSession(id);
    if (typeof window !== 'undefined') {
      if (id) {
        sessionStorage.setItem(STORAGE_KEY, id);
      } else {
        sessionStorage.removeItem(STORAGE_KEY);
      }
    }
  }, [setStoreSession]);

  // Fetch active scenarios from instructor endpoint
  const fetchAvailable = useCallback(async (): Promise<ActiveScenarioMeta[]> => {
    try {
      const res = await fetch('/api/instructor/active_scenarios');
      if (!res.ok) return [];
      const list = await res.json();
      if (Array.isArray(list)) {
        setAvailable(list);
        return list;
      }
    } catch {
      // transient failure
    }
    return [];
  }, []);

  // Resolve scenario ID in strict order:
  // 1. ?scenario= query param
  // 2. zustand store
  // 3. sessionStorage key 'vk:scenario_id'
  // 4. newest entry of GET /api/instructor/active_scenarios
  const resolveScenarioId = useCallback(async (): Promise<string | null> => {
    const queryParam = searchParams.get('scenario');
    if (queryParam) return queryParam;

    if (storeScenarioId) return storeScenarioId;

    if (typeof window !== 'undefined') {
      const stored = sessionStorage.getItem(STORAGE_KEY);
      if (stored) return stored;
    }

    const activeList = await fetchAvailable();
    if (activeList.length > 0) {
      // insertion order, last = newest
      return activeList[activeList.length - 1].scenario_id;
    }

    return null;
  }, [searchParams, storeScenarioId, fetchAvailable]);

  // Initial scenario ID resolution
  useEffect(() => {
    let mounted = true;
    resolveScenarioId().then((resolved) => {
      if (!mounted) return;
      if (resolved) {
        setScenarioIdState(resolved);
      } else {
        setStatus('empty');
      }
    });
    return () => {
      mounted = false;
    };
  }, [resolveScenarioId]);

  // Query team state to pick up detector/classifier team activity if present
  const fetchTeamClassification = useCallback(async (scId: string, currentT: number) => {
    try {
      const res = await fetch(`/api/team/state/${encodeURIComponent(scId)}?role=detector&t=${currentT.toFixed(1)}`);
      if (res.ok) {
        const data = await res.json();
        const map: Record<string, string> = {};
        if (Array.isArray(data.contacts)) {
          for (const item of data.contacts) {
            const cid = item.contact_id || item.id;
            const cls = item.team?.classification;
            if (cid && cls) {
              map[cid] = cls;
            }
          }
        }
        teamClassificationsRef.current = map;
      }
    } catch {
      // Silently keep previous or empty team state on network glitch
    }
  }, []);

  // Poll /api/scenario/{id}/refresh every 2.5s
  useEffect(() => {
    if (!scenarioId) return;

    let isCancelled = false;

    const pollScenario = async () => {
      try {
        const res = await fetch(`/api/scenario/${encodeURIComponent(scenarioId)}/refresh`);

        if (isCancelled) return;

        if (res.status === 404) {
          // Scenario expired or finished on server: clear and re-resolve
          setScenarioId(null);
          const nextId = await resolveScenarioId();
          if (isCancelled) return;
          if (nextId && nextId !== scenarioId) {
            setScenarioId(nextId);
          } else {
            setScenario(null);
            setStatus('empty');
          }
          consecutiveFailuresRef.current = 0;
          return;
        }

        if (!res.ok) {
          throw new Error(`Refresh failed: HTTP ${res.status}`);
        }

        const data = await res.json();
        if (isCancelled) return;

        consecutiveFailuresRef.current = 0;
        setScenario(data);
        setStatus('live');
        setError(null);

        // Also query team state for detector/classifier overlays
        const t = (Date.now() - (data.server_start_epoch_ms || localStartEpochRef.current)) / 1000;
        fetchTeamClassification(scenarioId, t);
      } catch (err: any) {
        if (isCancelled) return;
        consecutiveFailuresRef.current += 1;
        // Keep last good data on transient errors, set 'error' only after 3 consecutive failures
        if (consecutiveFailuresRef.current >= 3) {
          setStatus('error');
          setError(err.message || 'Scenario communication failure (3 retries failed)');
        }
      }
    };

    // Initial immediate poll on id change
    pollScenario();

    const interval = setInterval(pollScenario, POLL_INTERVAL_MS);
    return () => {
      isCancelled = true;
      clearInterval(interval);
    };
  }, [scenarioId, setScenarioId, resolveScenarioId, fetchTeamClassification]);

  // Periodically refresh available scenarios list every 5s so dropdown stays current
  useEffect(() => {
    const interval = setInterval(fetchAvailable, 5000);
    return () => clearInterval(interval);
  }, [fetchAvailable]);

  // Compute Enriched Contacts and update elapsed time
  // Runs via animation frame or 100ms interval for React UI consumption
  useEffect(() => {
    if (!scenario) {
      setContacts([]);
      return;
    }

    let animId: number;

    const updateTick = () => {
      const t = getElapsed();
      setElapsed(t);

      if (scenario.contacts && Array.isArray(scenario.contacts)) {
        const enriched = enrichScenarioContacts(scenario.contacts, t, traineeLog);

        // Keep all scenario tracks matching Trainer.tsx scopes (do not drop tracks before spawn_t)
        // Only exclude targets marked destroyed or hard-killed
        const visible = enriched.filter((c) => {
          const entry = traineeLog[c.id];
          if (entry?.state === 'destroyed' || (c as any).engagement_action === 'hard_kill') {
            return false;
          }
          return true;
        });

        // Overlay classification:
        // Priority 1: team[contact_id].classification
        // Priority 2: store's traineeLog[contact_id].classification
        // Priority 3: 'unknown'
        // Never use any true-class field
        const processed = visible.map((c) => {
          const teamCls = teamClassificationsRef.current[c.id];
          const localCls = traineeLog[c.id]?.classification;
          const assigned = teamCls || localCls || 'unknown';

          return {
            ...c,
            assigned_classification: assigned as Classification,
            assigned, // convenient accessor
          };
        });

        setContacts(processed);
      } else {
        setContacts([]);
      }

      animId = requestAnimationFrame(updateTick);
    };

    animId = requestAnimationFrame(updateTick);
    return () => cancelAnimationFrame(animId);
  }, [scenario, traineeLog, getElapsed]);

  return {
    scenarioId,
    setScenarioId,
    available,
    status,
    error,
    scenario,
    contacts,
    elapsed,
    getElapsed,
  };
}
