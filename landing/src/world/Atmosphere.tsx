import { useRef, useMemo, useState } from 'react'
import * as THREE from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import { PerformanceMonitor } from '@react-three/drei'
import { sampleTimeOfDay } from '../lib/timeOfDay'
import { flight } from '../lib/flightState'

const SKY_VERT = `
  varying vec3 vWorldPosition;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`
const SKY_FRAG = `
  uniform vec3 uSkyTop;
  uniform vec3 uSkyBot;
  varying vec3 vWorldPosition;
  void main() {
    float t = clamp(normalize(vWorldPosition).y * 0.5 + 0.35, 0.0, 1.0);
    gl_FragColor = vec4(mix(uSkyBot, uSkyTop, smoothstep(0.0, 1.0, t)), 1.0);
  }
`

function Stars() {
  const pointsRef = useRef<THREE.Points>(null)

  const { geometry, material } = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions: number[] = []
    const R = 800
    for (let i = 0; i < 900; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      positions.push(
        R * Math.sin(phi) * Math.cos(theta),
        Math.abs(R * Math.cos(phi)) + 15,
        R * Math.sin(phi) * Math.sin(theta)
      )
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const mat = new THREE.PointsMaterial({
      color: '#B0D8FF',
      size: 1.6,
      sizeAttenuation: false,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    })
    return { geometry: geo, material: mat }
  }, [])

  useFrame(() => {
    if (!material) return
    const tod = sampleTimeOfDay(flight.damped)
    material.opacity = tod.starOpacity
  })

  return <points ref={pointsRef} geometry={geometry} material={material} frustumCulled={false} />
}

export default function Atmosphere() {
  const sunRef = useRef<THREE.DirectionalLight>(null)
  const moonRef = useRef<THREE.DirectionalLight>(null)
  const hemiRef = useRef<THREE.HemisphereLight>(null)
  const skyMatRef = useRef<THREE.ShaderMaterial | null>(null)
  const { scene } = useThree()
  const fogRef = useRef<THREE.Fog | null>(null)
  const [hasBloom, setHasBloom] = useState(true)

  const skyMat = useMemo(() => {
    const mat = new THREE.ShaderMaterial({
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      uniforms: {
        uSkyTop: { value: new THREE.Color('#3f86c9') },
        uSkyBot: { value: new THREE.Color('#cfe6f5') },
      },
      side: THREE.BackSide,
      depthWrite: false,
    })
    skyMatRef.current = mat
    return mat
  }, [])

  useFrame(() => {
    const tod = sampleTimeOfDay(flight.damped)

    if (skyMatRef.current) {
      skyMatRef.current.uniforms.uSkyTop.value.copy(tod.skyTop)
      skyMatRef.current.uniforms.uSkyBot.value.copy(tod.skyBottom)
    }

    if (sunRef.current) {
      sunRef.current.color.copy(tod.sunColor)
      sunRef.current.intensity = tod.sunIntensity
    }

    if (moonRef.current) {
      // Moon directional light illuminates terrain at dusk/night
      moonRef.current.intensity = Math.max(0.2, tod.starOpacity * 2.4)
    }

    if (hemiRef.current) {
      // Raised ambient floor so terrain always stays readable
      hemiRef.current.intensity = Math.max(0.5, tod.hemiIntensity)
    }

    // Cut fog density by 70%, fading only the far third of the view (500 to 1600 units)
    if (!fogRef.current) {
      const linearFog = new THREE.Fog(tod.skyBottom.getHex(), 500, 1600)
      fogRef.current = linearFog
      scene.fog = linearFog
    } else {
      fogRef.current.color.copy(tod.skyBottom)
    }
  })

  return (
    <>
      {/* Big inverted sky sphere (radius 900) */}
      <mesh material={skyMat} frustumCulled={false}>
        <sphereGeometry args={[900, 32, 16]} />
      </mesh>

      {/* Stars */}
      <Stars />

      {/* Sun directional light */}
      <directionalLight
        ref={sunRef}
        color="#fff3dc"
        intensity={2.2}
        position={[-150, 250, -100]}
      />

      {/* Night Moon directional light */}
      <directionalLight
        ref={moonRef}
        color="#c8dcfa"
        intensity={0}
        position={[140, 220, 80]}
      />

      {/* Hemisphere light */}
      <hemisphereLight
        ref={hemiRef}
        args={['#3f86c9', '#1a2420', 1.0]}
      />

      {/* Performance monitor & post-processing */}
      <PerformanceMonitor
        onDecline={() => setHasBloom(false)}
        onIncline={() => setHasBloom(true)}
      >
        <EffectComposer multisampling={0}>
          {hasBloom ? (
            <Bloom
              luminanceThreshold={0.7}
              luminanceSmoothing={0.3}
              intensity={0.4}
              mipmapBlur
            />
          ) : (
            <></>
          )}
          <Vignette darkness={0.4} offset={0.3} />
        </EffectComposer>
      </PerformanceMonitor>
    </>
  )
}
