import { createNoise2D } from 'simplex-noise'
import { pathX, smoothstep } from './pathCore'

function alea(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

const noise2D = createNoise2D(alea(1337))

function fbm(x: number, z: number): number {
  let total = 0
  let amplitude = 1
  let frequency = 0.005
  let maxValue = 0

  for (let i = 0; i < 4; i++) {
    total += noise2D(x * frequency, z * frequency) * amplitude
    maxValue += amplitude
    amplitude *= 0.5
    frequency *= 2
  }

  return total / maxValue
}

function getBiomeAmplitude(z: number): number {
  const ampUrban = 6
  const ampCoastal = 3
  const ampMountain = 95
  const ampRural = 18

  if (z >= -370) return ampUrban
  if (z >= -430) {
    const t = smoothstep(-430, -370, z)
    return ampCoastal + (ampUrban - ampCoastal) * t
  }
  if (z >= -570) return ampCoastal
  if (z >= -630) {
    const t = smoothstep(-630, -570, z)
    return ampMountain + (ampCoastal - ampMountain) * t
  }
  if (z >= -970) return ampMountain
  if (z >= -1030) {
    const t = smoothstep(-1030, -970, z)
    return ampRural + (ampMountain - ampRural) * t
  }
  return ampRural
}

export function heightAt(x: number, z: number): number {
  const px = pathX(z)
  const distFromValley = Math.abs(x - px)
  const valleyMask = smoothstep(18, 90, distFromValley)
  const baseAmp = getBiomeAmplitude(z)
  const rawNoise = fbm(x, z)
  const normalizedNoise = Math.max(0, rawNoise * 0.5 + 0.5)

  let height = normalizedNoise * baseAmp * valleyMask

  // Mountain peak boost
  if (z < -600 && z > -1000 && distFromValley > 100) {
    const peakFactor = smoothstep(100, 160, distFromValley)
    height = Math.max(height, 95 * peakFactor * normalizedNoise)
  }

  // Coastal sea level drop
  if (z <= -400 && z >= -600) {
    const seaDropDist = x - (px + 70)
    if (seaDropDist > 0) {
      const dropFactor = smoothstep(0, 40, seaDropDist)
      height = height * (1 - dropFactor) - dropFactor * 4.0
    }
  }

  return height
}
