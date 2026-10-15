export const WORLD_LENGTH = 1600

export const BIOME_RANGES = {
  urban: [0, -400] as [number, number],
  coastal: [-400, -600] as [number, number],
  mountain: [-600, -1000] as [number, number],
  rural: [-1000, -1600] as [number, number],
}

export function pathX(z: number): number {
  return Math.sin(z * 0.004) * 40 + Math.sin(z * 0.011) * 12
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

export const URBAN_MARGIN = 24
export const BASE_CLEARANCE = 18
export const MAX_DESCENT_SLOPE = 0.12

export function buildingMargin(z: number): number {
  return URBAN_MARGIN * smoothstep(-560, -300, z)
}
