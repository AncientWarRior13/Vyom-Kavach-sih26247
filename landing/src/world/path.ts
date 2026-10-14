import * as THREE from 'three'
import { heightAt } from './terrainHeight'
import { pathX, smoothstep, WORLD_LENGTH, BIOME_RANGES } from './pathCore'

export { pathX, WORLD_LENGTH, BIOME_RANGES }

function getBiomeClearance(z: number): number {
  const cUrban = 34
  const cCoastal = 18
  const cMountain = 26
  const cRural = 22

  if (z >= -370) return cUrban
  if (z >= -430) {
    const t = smoothstep(-430, -370, z)
    return cCoastal + (cUrban - cCoastal) * t
  }
  if (z >= -570) return cCoastal
  if (z >= -630) {
    const t = smoothstep(-630, -570, z)
    return cMountain + (cCoastal - cMountain) * t
  }
  if (z >= -970) return cMountain
  if (z >= -1030) {
    const t = smoothstep(-1030, -970, z)
    return cRural + (cMountain - cRural) * t
  }
  return cRural
}

const NUM_SAMPLES = 400
const curvePoints: THREE.Vector3[] = []

for (let i = 0; i <= NUM_SAMPLES; i++) {
  const t = i / NUM_SAMPLES
  const z = -t * WORLD_LENGTH
  const x = pathX(z)
  const groundY = heightAt(x, z)
  const clearance = getBiomeClearance(z)
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
