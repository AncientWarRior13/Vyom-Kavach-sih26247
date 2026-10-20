export interface Section {
  id: string
  eyebrow: string
  title: string
  side: 'left' | 'right' | 'center'
}

export const SECTION_COUNT = 8

export const SECTIONS: Section[] = [
  {
    id: 'hero',
    eyebrow: 'DRONE & SWARM THREAT TRAINING',
    title: 'Train the call before it counts.',
    side: 'left',
  },
  {
    id: 'detect',
    eyebrow: '01 · DETECT',
    title: 'Spot it before it spots you.',
    side: 'right',
  },
  {
    id: 'respond',
    eyebrow: '02 · RESPOND',
    title: 'Right response. Right time. Right place.',
    side: 'left',
  },
  {
    id: 'swarm',
    eyebrow: '03 · SWARM',
    title: 'Take one down. The rest react.',
    side: 'right',
  },
  {
    id: 'sensors',
    eyebrow: '04 · SENSORS',
    title: 'Every sensor tells a different truth.',
    side: 'left',
  },
  {
    id: 'review',
    eyebrow: '05 · REVIEW',
    title: 'Every decision, scored and replayable.',
    side: 'right',
  },
  {
    id: 'modes',
    eyebrow: '06 · MODES',
    title: 'One exercise, 3+ ways in.',
    side: 'left',
  },
  {
    id: 'land',
    eyebrow: 'READY WHEN YOU ARE',
    title: 'Begin the mission.',
    side: 'center',
  },
]

export function sectionIndexAt(p: number): number {
  return Math.min(7, Math.max(0, Math.floor(p * 8)))
}

export function localProgress(p: number, i: number): number {
  return Math.min(1, Math.max(0, (p - i / 8) * 8))
}
