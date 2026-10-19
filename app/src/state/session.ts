import { create } from 'zustand';

export interface SessionState {
  scenarioId: string | null;
  traineeLog: Record<string, any>;
  setSession: (scenarioId: string | null, traineeLog?: Record<string, any>) => void;
  setTraineeLog: (
    logOrUpdater:
      | Record<string, any>
      | ((prev: Record<string, any>) => Record<string, any>)
  ) => void;
  clear: () => void;
}

const STORAGE_KEY = 'vk:scenario_id';

export const useSessionStore = create<SessionState>((set) => ({
  scenarioId: typeof window !== 'undefined' ? sessionStorage.getItem(STORAGE_KEY) : null,
  traineeLog: {},
  setSession: (scenarioId: string | null, traineeLog: Record<string, any> = {}) => {
    if (typeof window !== 'undefined') {
      if (scenarioId) {
        sessionStorage.setItem(STORAGE_KEY, scenarioId);
      } else {
        sessionStorage.removeItem(STORAGE_KEY);
      }
    }
    set({
      scenarioId,
      traineeLog,
    });
  },
  setTraineeLog: (logOrUpdater) => {
    set((state) => ({
      traineeLog:
        typeof logOrUpdater === 'function'
          ? logOrUpdater(state.traineeLog)
          : logOrUpdater,
    }));
  },
  clear: () => {
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem(STORAGE_KEY);
    }
    set({
      scenarioId: null,
      traineeLog: {},
    });
  },
}));
