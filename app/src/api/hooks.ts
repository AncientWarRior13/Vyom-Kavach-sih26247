import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type { ClassifyRequest, SensorReadRequest, EngageRequest, SubmitSessionRequest } from './types';

export function useModelStatus() {
  return useQuery({
    queryKey: ['modelStatus'],
    queryFn: api.getModelStatus,
    refetchInterval: 10000,
  });
}

export function useScriptedScenarios() {
  return useQuery({
    queryKey: ['scriptedScenarios'],
    queryFn: api.getScriptedScenarios,
    staleTime: 60000,
  });
}

export function useRefreshScenario(scenarioId: string | null, enabled: boolean = true) {
  return useQuery({
    queryKey: ['scenario', scenarioId],
    queryFn: () => (scenarioId ? api.refreshScenario(scenarioId) : Promise.reject('No ID')),
    enabled: Boolean(scenarioId) && enabled,
    refetchInterval: 2500,
  });
}

export function useClassifyContact() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: ClassifyRequest) => api.classifyContact(req),
    onSuccess: (_, vars) => {
      queryClient.invalidateQueries({ queryKey: ['scenario', vars.scenario_id] });
    },
  });
}

export function useSensorRead() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: SensorReadRequest) => api.sensorRead(req),
    onSuccess: (_, vars) => {
      queryClient.invalidateQueries({ queryKey: ['scenario', vars.scenario_id] });
    },
  });
}

export function useEngageContact() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: EngageRequest) => api.engageContact(req),
    onSuccess: (_, vars) => {
      queryClient.invalidateQueries({ queryKey: ['scenario', vars.scenario_id] });
    },
  });
}

export function useSubmitSession() {
  return useMutation({
    mutationFn: (req: SubmitSessionRequest) => api.submitSession(req),
  });
}
