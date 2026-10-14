import { create } from 'zustand'

export interface AppState {
  activeSection: number
  tagged: Record<string, string>
  gatesPassed: [boolean, boolean, boolean]
  engagedCount: number
  channel: number // 0: radar, 1: eo/ir, 2: rf, 3: acoustic
  loaded: boolean

  setActiveSection: (index: number) => void
  setTagged: (id: string, classification: string) => void
  setGatePassed: (gateIndex: number, passed: boolean) => void
  incrementEngaged: () => void
  setChannel: (channel: number) => void
  setLoaded: (loaded: boolean) => void
  resetSwarm: () => void
}

export const useStore = create<AppState>((set) => ({
  activeSection: 0,
  tagged: {},
  gatesPassed: [false, false, false],
  engagedCount: 0,
  channel: 0,
  loaded: false,

  setActiveSection: (activeSection) => set({ activeSection }),
  setTagged: (id, classification) =>
    set((state) => ({ tagged: { ...state.tagged, [id]: classification } })),
  setGatePassed: (gateIndex, passed) =>
    set((state) => {
      const next = [...state.gatesPassed] as [boolean, boolean, boolean]
      next[gateIndex] = passed
      return { gatesPassed: next }
    }),
  incrementEngaged: () => set((state) => ({ engagedCount: state.engagedCount + 1 })),
  setChannel: (channel) => set({ channel: ((channel % 4) + 4) % 4 }),
  setLoaded: (loaded) => set({ loaded }),
  resetSwarm: () => set({ engagedCount: 0 }),
}))
