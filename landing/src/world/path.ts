import * as THREE from 'three'
import { heightAt } from './terrainHeight'
import {
  pathX,
  WORLD_LENGTH,
  BIOME_RANGES,
  BASE_CLEARANCE,
  buildingMargin,
} from './pathCore'

export { pathX, WORLD_LENGTH, BIOME_RANGES }

const NUM_SAMPLES = 400
const curvePoints: THREE.Vector3[] = []

for (let i = 0; i <= NUM_SAMPLES; i++) {
  const t = i / NUM_SAMPLES
  const z = -t * WORLD_LENGTH
  const x = pathX(z)
  const groundY = heightAt(x, z)
  const clearance = buildingMargin(z) + BASE_CLEARANCE
  const y = groundY + clearance

  curvePoints.push(new THREE.Vector3(x, y, z))
}

export const flightCurve = new THREE.CatmullRomCurve3(curvePoints, false, 'catmullrom', 0.5)

// Assertion: verify curve never drops below heightAt + 8
for (let i = 0; i <= 200; i++) {
  const t = i / 200
  const pt = flightCurve.getPointAt(t)
  const gy = heightAt(pt.x, pt.z)
  if (pt.y < gy + 8) {
    console.warn(`[Path] Clearance warning at t=${t.toFixed(3)}: y=${pt.y.toFixed(1)}, ground=${gy.toFixed(1)}`)
  }
}

export function getFlightPoint(t: number): THREE.Vector3 {
  const clamped = Math.max(0, Math.min(1, t))
  return flightCurve.getPointAt(clamped)
}

export function getFlightTangent(t: number): THREE.Vector3 {
  const clamped = Math.max(0, Math.min(1, t))
  return flightCurve.getTangentAt(clamped)
}
