import { sampleTimeOfDay } from './timeOfDay'

export type Biome = 'urban' | 'coastal' | 'mountain' | 'rural'

export function biomeAt(z: number): Biome {
  if (z >= -400) return 'urban'
  if (z >= -600) return 'coastal'
  if (z >= -1000) return 'mountain'
  return 'rural'
}

export function effectiveDegradation(p: number): number {
  const clamped = Math.max(0, Math.min(1, p))
  const tod = sampleTimeOfDay(clamped)

  // Continuous time penalty from starOpacity / sunIntensity
  const nightFactor = Math.max(0, Math.min(1, tod.starOpacity))
  const duskFactor = Math.max(0, Math.min(1, 1 - tod.sunIntensity / 2.2))
  const timePenalty = nightFactor * 0.35 + (1 - nightFactor) * duskFactor * 0.15

  // Terrain penalty based on current world Z along flight path
  const worldZ = clamped * -1600
  const biome = biomeAt(worldZ)

  let terrainPenalty = 0.0
  if (biome === 'urban') terrainPenalty = 0.25
  else if (biome === 'mountain') terrainPenalty = 0.15
  else if (biome === 'coastal') terrainPenalty = 0.05
  else terrainPenalty = 0.0

  return Math.min(1.0, Math.max(0.0, 0.6 * (timePenalty + terrainPenalty)))
}
