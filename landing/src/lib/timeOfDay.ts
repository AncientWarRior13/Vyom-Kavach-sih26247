import * as THREE from 'three'

export interface TimeOfDayFrame {
  p: number
  skyTop: THREE.Color
  skyBottom: THREE.Color
  fog: THREE.Color
  sunColor: THREE.Color
  sunIntensity: number
  hemiIntensity: number
  starOpacity: number
  windowGlow: number
  bloom: number
}

interface RawFrame {
  p: number
  skyTop: string
  skyBottom: string
  sunColor: string
  sunIntensity: number
  hemiIntensity: number
  starOpacity: number
  windowGlow: number
  bloom: number
}

const RAW_KEYFRAMES: RawFrame[] = [
  {
    p: 0.0,
    skyTop: '#3f86c9',
    skyBottom: '#cfe6f5',
    sunColor: '#fff3dc',
    sunIntensity: 2.2,
    hemiIntensity: 1.0,
    starOpacity: 0.0,
    windowGlow: 0.0,
    bloom: 0.15,
  },
  {
    p: 0.22,
    skyTop: '#4a78b0',
    skyBottom: '#f2d3a0',
    sunColor: '#ffd9a0',
    sunIntensity: 1.8,
    hemiIntensity: 0.8,
    starOpacity: 0.0,
    windowGlow: 0.0,
    bloom: 0.2,
  },
  {
    p: 0.38,
    skyTop: '#3a2f5c',
    skyBottom: '#f08a4b',
    sunColor: '#ff9a5a',
    sunIntensity: 1.0,
    hemiIntensity: 0.5,
    starOpacity: 0.1,
    windowGlow: 0.3,
    bloom: 0.3,
  },
  {
    p: 0.55,
    skyTop: '#14173a',
    skyBottom: '#6a3a52',
    sunColor: '#7a5a9a',
    sunIntensity: 0.4,
    hemiIntensity: 0.3,
    starOpacity: 0.6,
    windowGlow: 0.7,
    bloom: 0.35,
  },
  {
    p: 0.7,
    skyTop: '#080d1a',
    skyBottom: '#182436',
    sunColor: '#88a4d4',
    sunIntensity: 0.85,
    hemiIntensity: 0.65,
    starOpacity: 0.9,
    windowGlow: 1.0,
    bloom: 0.35,
  },
  {
    p: 0.88,
    skyTop: '#162244',
    skyBottom: '#48446c',
    sunColor: '#8a8ab8',
    sunIntensity: 1.1,
    hemiIntensity: 0.7,
    starOpacity: 0.4,
    windowGlow: 0.8,
    bloom: 0.3,
  },
  {
    p: 1.0,
    skyTop: '#4a6fa5',
    skyBottom: '#ffbf8a',
    sunColor: '#ffcf9a',
    sunIntensity: 1.6,
    hemiIntensity: 0.8,
    starOpacity: 0.0,
    windowGlow: 0.1,
    bloom: 0.25,
  },
]

export const KEYFRAMES: TimeOfDayFrame[] = RAW_KEYFRAMES.map((k) => {
  const skyBottom = new THREE.Color(k.skyBottom)
  return {
    p: k.p,
    skyTop: new THREE.Color(k.skyTop),
    skyBottom,
    fog: skyBottom.clone(),
    sunColor: new THREE.Color(k.sunColor),
    sunIntensity: k.sunIntensity,
    hemiIntensity: k.hemiIntensity,
    starOpacity: k.starOpacity,
    windowGlow: k.windowGlow,
    bloom: k.bloom,
  }
})

// Reusable cached result to avoid per-frame allocations
const resultFrame: TimeOfDayFrame = {
  p: 0,
  skyTop: new THREE.Color(),
  skyBottom: new THREE.Color(),
  fog: new THREE.Color(),
  sunColor: new THREE.Color(),
  sunIntensity: 1,
  hemiIntensity: 1,
  starOpacity: 0,
  windowGlow: 0,
  bloom: 0.2,
}

export function sampleTimeOfDay(p: number): TimeOfDayFrame {
  const clamped = Math.max(0, Math.min(1, p))

  let i = 0
  while (i < KEYFRAMES.length - 2 && KEYFRAMES[i + 1].p < clamped) {
    i++
  }

  const k1 = KEYFRAMES[i]
  const k2 = KEYFRAMES[i + 1]
  const span = Math.max(0.0001, k2.p - k1.p)
  const factor = Math.max(0, Math.min(1, (clamped - k1.p) / span))

  resultFrame.p = clamped
  resultFrame.skyTop.copy(k1.skyTop).lerp(k2.skyTop, factor)
  resultFrame.skyBottom.copy(k1.skyBottom).lerp(k2.skyBottom, factor)
  resultFrame.fog.copy(k1.fog).lerp(k2.fog, factor)
  resultFrame.sunColor.copy(k1.sunColor).lerp(k2.sunColor, factor)
  resultFrame.sunIntensity = THREE.MathUtils.lerp(k1.sunIntensity, k2.sunIntensity, factor)
  resultFrame.hemiIntensity = THREE.MathUtils.lerp(k1.hemiIntensity, k2.hemiIntensity, factor)
  resultFrame.starOpacity = THREE.MathUtils.lerp(k1.starOpacity, k2.starOpacity, factor)
  resultFrame.windowGlow = THREE.MathUtils.lerp(k1.windowGlow, k2.windowGlow, factor)
  resultFrame.bloom = THREE.MathUtils.lerp(k1.bloom, k2.bloom, factor)

  return resultFrame
}
