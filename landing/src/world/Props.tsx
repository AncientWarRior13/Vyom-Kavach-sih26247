import { useMemo, useRef, useLayoutEffect } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { heightAt } from './terrainHeight'
import { pathX } from './pathCore'
import { flight } from '../lib/flightState'
import { sampleTimeOfDay } from '../lib/timeOfDay'

function mkRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

// Neutral enterprise facade palette (albedo 0.35 - 0.60)
const FACADE_PALETTE = [
  '#8b9299', // concrete light
  '#747c85', // concrete mid
  '#a29f95', // stone warm
  '#5f7387', // glass blue-grey
  '#b4b9be', // pale limestone
  '#3e4b5a', // dark architectural glass
]

// Counts for instanced meshes
const MAX_SLAB_BLDG = 180
const MAX_SETBACK_LOWER = 120
const MAX_SETBACK_UPPER = 120
const MAX_PODIUM_BASE = 80
const MAX_PODIUM_TOWER = 80

const MAX_ROOF_PROPS = 160
const MAX_ROOF_ANTENNAS = 60
const MAX_STREET_LIGHTS = 70
const MAX_CARS = 45

const MAX_CONTAINERS = 90
const MAX_CRANES = 6
const MAX_FARMHOUSES = 24
const MAX_TREES = 600
const MAX_ROCKS = 160
const MAX_PYLONS = 20

export default function Props() {
  // InstancedMesh references
  const slabRef = useRef<THREE.InstancedMesh>(null)
  const setbackLowerRef = useRef<THREE.InstancedMesh>(null)
  const setbackUpperRef = useRef<THREE.InstancedMesh>(null)
  const podiumBaseRef = useRef<THREE.InstancedMesh>(null)
  const podiumTowerRef = useRef<THREE.InstancedMesh>(null)

  const roofPropRef = useRef<THREE.InstancedMesh>(null)
  const antennaRef = useRef<THREE.InstancedMesh>(null)
  const beaconRef = useRef<THREE.InstancedMesh>(null)
  const streetLightRef = useRef<THREE.InstancedMesh>(null)
  const carRef = useRef<THREE.InstancedMesh>(null)

  const containerRef = useRef<THREE.InstancedMesh>(null)
  const craneRef = useRef<THREE.InstancedMesh>(null)
  const farmhouseRef = useRef<THREE.InstancedMesh>(null)
  const treeRef = useRef<THREE.InstancedMesh>(null)
  const rockRef = useRef<THREE.InstancedMesh>(null)
  const pylonRef = useRef<THREE.InstancedMesh>(null)
  const pylonLightRef = useRef<THREE.InstancedMesh>(null)

  const snowParticlesRef = useRef<THREE.Points>(null)

  // Uniform container for window glow and time-of-day
  const customUniforms = useMemo(
    () => ({
      uWindowGlow: { value: 0.0 },
    }),
    []
  )

  // Shared Procedural Facade Standard Material
  const facadeMaterial = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      color: '#ffffff',
      roughness: 0.85,
      metalness: 0.05,
    })

    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uWindowGlow = customUniforms.uWindowGlow

      // Vertex shader modifications
      shader.vertexShader = `
        varying vec3 vCustomWorldPos;
        varying vec3 vCustomWorldNorm;
        ${shader.vertexShader}
      `
      shader.vertexShader = shader.vertexShader.replace(
        '#include <worldpos_vertex>',
        `
        #include <worldpos_vertex>
        vec4 customWPos = vec4( transformed, 1.0 );
        #ifdef USE_INSTANCING
          customWPos = instanceMatrix * customWPos;
        #endif
        customWPos = modelMatrix * customWPos;
        vCustomWorldPos = customWPos.xyz;

        vec3 customWNorm = normal;
        #ifdef USE_INSTANCING
          customWNorm = mat3(instanceMatrix) * customWNorm;
        #endif
        customWNorm = normalize(mat3(modelMatrix) * customWNorm);
        vCustomWorldNorm = customWNorm;
        `
      )

      // Fragment shader modifications
      shader.fragmentShader = `
        uniform float uWindowGlow;
        varying vec3 vCustomWorldPos;
        varying vec3 vCustomWorldNorm;
        ${shader.fragmentShader}
      `
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <map_fragment>',
        `
        #include <map_fragment>

        // Only render facade windows on vertical walls
        bool isWall = abs(vCustomWorldNorm.y) < 0.5;
        if (isWall) {
          // World-space cells: floor height 3.2, bay width 2.4
          float floorIdx = floor(vCustomWorldPos.y / 3.2);
          float floorFrac = fract(vCustomWorldPos.y / 3.2);

          float horizCoord = abs(vCustomWorldNorm.x) > 0.5 ? vCustomWorldPos.z : vCustomWorldPos.x;
          float bayIdx = floor(horizCoord / 2.4);
          float bayFrac = fract(horizCoord / 2.4);

          // Window pane occupies 55% of cell width and 58% of cell height
          bool isWindow = (floorFrac > 0.22 && floorFrac < 0.80) && (bayFrac > 0.22 && bayFrac < 0.77);

          if (isWindow) {
            // Day glass: darker and shinier
            diffuseColor.rgb *= 0.55;

            // Lit window hash
            float winHash = fract(sin(dot(vec2(bayIdx, floorIdx), vec2(12.9898, 78.233))) * 43758.5453);
            if (winHash < 0.45 && uWindowGlow > 0.01) {
              vec3 litColor = winHash < 0.30
                ? vec3(1.0, 0.85, 0.63) // warm amber
                : (winHash < 0.40 ? vec3(1.0, 0.93, 0.80) : vec3(0.81, 0.90, 1.0)); // warm white / cool blue
              // Glow only at dusk/night
              totalEmissiveRadiance += litColor * (uWindowGlow * 0.9);
            }
          }

          // Fake AO: darken facades near ground (world y: 0-10) by up to 35%
          float groundAo = mix(0.65, 1.0, clamp(vCustomWorldPos.y / 10.0, 0.0, 1.0));
          diffuseColor.rgb *= groundAo;

          // Subtle vertical gradient
          diffuseColor.rgb *= (0.94 + 0.10 * clamp(vCustomWorldPos.y / 60.0, 0.0, 1.0));
        } else {
          // Roof top: slightly darker concrete
          diffuseColor.rgb *= 0.88;
        }
        `
      )
    }

    return mat
  }, [customUniforms])

  // Simple materials for props
  const propMats = useMemo(() => {
    return {
      roof: new THREE.MeshStandardMaterial({ color: '#2d3748', roughness: 0.8 }),
      antenna: new THREE.MeshStandardMaterial({ color: '#1a202c', roughness: 0.5, metalness: 0.8 }),
      beacon: new THREE.MeshBasicMaterial({ color: '#ff3b3b' }),
      streetLightPole: new THREE.MeshStandardMaterial({ color: '#4a5568', roughness: 0.6, metalness: 0.8 }),
      car: new THREE.MeshStandardMaterial({ color: '#2b3648', roughness: 0.4, metalness: 0.6 }),
      container: new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.2 }),
      crane: new THREE.MeshStandardMaterial({ color: '#d97706', roughness: 0.5, metalness: 0.6 }),
      farmhouse: new THREE.MeshStandardMaterial({ color: '#8a4b38', roughness: 0.85 }),
      tree: new THREE.MeshStandardMaterial({ color: '#1d3a24', roughness: 0.9 }),
      rock: new THREE.MeshStandardMaterial({ color: '#4b5563', roughness: 0.95, flatShading: true }),
      pylon: new THREE.MeshStandardMaterial({ color: '#374151', roughness: 0.6, metalness: 0.8 }),
    }
  }, [])

  // Geometries
  const unitBoxGeo = useMemo(() => new THREE.BoxGeometry(1, 1, 1), [])
  const cylinderGeo = useMemo(() => new THREE.CylinderGeometry(0.5, 0.5, 1, 12), [])
  const coneGeo = useMemo(() => new THREE.ConeGeometry(1, 1, 6), [])
  const rockGeo = useMemo(() => new THREE.DodecahedronGeometry(1, 0), [])
  const sphereGeo = useMemo(() => new THREE.SphereGeometry(0.3, 6, 6), [])

  // Snow / mist particle system for mountains
  const snowGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const pos: number[] = []
    const rng = mkRng(555)
    for (let i = 0; i < 400; i++) {
      pos.push(
        (rng() - 0.5) * 160,
        35 + rng() * 45,
        -600 - rng() * 400
      )
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    return geo
  }, [])

  const snowMat = useMemo(() => {
    return new THREE.PointsMaterial({
      color: '#e2f0fc',
      size: 1.8,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
    })
  }, [])

  // Placement Layout
  useLayoutEffect(() => {
    const dummy = new THREE.Object3D()
    const colBuf = new THREE.Color()

    // ─────────────────────────────────────────────────────────────────────────
    // 1. URBAN BUILDINGS: Aligned Grid (0 to -380 Z)
    // ─────────────────────────────────────────────────────────────────────────
    const rng = mkRng(777)
    let slabCount = 0
    let setbackCount = 0
    let podiumCount = 0
    let roofPropCount = 0
    let antennaCount = 0
    let streetLightCount = 0
    let carCount = 0

    // Grid: blocks every ~40 units in Z, 14-unit streets, 20-unit corridor around flight path
    for (let zBlock = 0; zBlock > -380; zBlock -= 38) {
      for (let xBlock = -110; xBlock <= 110; xBlock += 32) {
        // Path corridor check
        const px = pathX(zBlock)
        if (Math.abs(xBlock - px) < 18) continue // 20-unit empty flight corridor

        // Street grid alignment (rotation is strictly 0 or Math.PI / 2)
        const rotY = rng() > 0.5 ? 0 : Math.PI / 2

        // Building dimensions
        const distFromCenter = Math.abs(zBlock - -200) / 200
        const distFromPath = Math.abs(xBlock - px)

        // Height distribution: tall (45-80) 35+ units from path, low (12-30) near path, falloff at edges
        const baseH = distFromPath > 35 ? (45 + rng() * 32) : (14 + rng() * 16)
        const h = Math.max(10, baseH * (1 - distFromCenter * 0.4))
        const w = 12 + rng() * 10
        const d = 12 + rng() * 10

        const groundY = heightAt(xBlock, zBlock)

        // Choose Archetype
        const archetype = rng()
        const colHex = FACADE_PALETTE[Math.floor(rng() * FACADE_PALETTE.length)]
        colBuf.set(colHex)

        if (archetype < 0.45 && slabCount < MAX_SLAB_BLDG) {
          // Archetype A: Slab Tower
          dummy.position.set(xBlock, groundY + h / 2, zBlock)
          dummy.scale.set(w, h, d)
          dummy.rotation.set(0, rotY, 0)
          dummy.updateMatrix()
          if (slabRef.current) {
            slabRef.current.setMatrixAt(slabCount, dummy.matrix)
            slabRef.current.setColorAt(slabCount, colBuf)
          }
          slabCount++
        } else if (archetype < 0.80 && setbackCount < MAX_SETBACK_LOWER) {
          // Archetype B: Setback Tower (Lower 60% + Upper 40%)
          const lowerH = h * 0.6
          dummy.position.set(xBlock, groundY + lowerH / 2, zBlock)
          dummy.scale.set(w, lowerH, d)
          dummy.rotation.set(0, rotY, 0)
          dummy.updateMatrix()
          if (setbackLowerRef.current) {
            setbackLowerRef.current.setMatrixAt(setbackCount, dummy.matrix)
            setbackLowerRef.current.setColorAt(setbackCount, colBuf)
          }

          const upperH = h * 0.4
          const upperScale = 0.68
          dummy.position.set(xBlock, groundY + lowerH + upperH / 2, zBlock)
          dummy.scale.set(w * upperScale, upperH, d * upperScale)
          dummy.rotation.set(0, rotY, 0)
          dummy.updateMatrix()
          if (setbackUpperRef.current) {
            setbackUpperRef.current.setMatrixAt(setbackCount, dummy.matrix)
            setbackUpperRef.current.setColorAt(setbackCount, colBuf)
          }
          setbackCount++
        } else if (podiumCount < MAX_PODIUM_BASE) {
          // Archetype C: Podium + Tower
          const podH = 8
          dummy.position.set(xBlock, groundY + podH / 2, zBlock)
          dummy.scale.set(w * 1.35, podH, d * 1.35)
          dummy.rotation.set(0, rotY, 0)
          dummy.updateMatrix()
          if (podiumBaseRef.current) {
            podiumBaseRef.current.setMatrixAt(podiumCount, dummy.matrix)
            podiumBaseRef.current.setColorAt(podiumCount, colBuf)
          }

          const towH = h
          dummy.position.set(xBlock, groundY + podH + towH / 2, zBlock)
          dummy.scale.set(w * 0.75, towH, d * 0.75)
          dummy.rotation.set(0, rotY, 0)
          dummy.updateMatrix()
          if (podiumTowerRef.current) {
            podiumTowerRef.current.setMatrixAt(podiumCount, dummy.matrix)
            podiumTowerRef.current.setColorAt(podiumCount, colBuf)
          }
          podiumCount++
        }

        // Roof Details on ~50% of buildings
        if (rng() > 0.45 && roofPropCount < MAX_ROOF_PROPS) {
          const roofY = groundY + h
          dummy.position.set(xBlock + (rng() - 0.5) * 4, roofY + 1.2, zBlock + (rng() - 0.5) * 4)
          dummy.scale.set(2.4, 2.0, 2.4)
          dummy.rotation.set(0, 0, 0)
          dummy.updateMatrix()
          if (roofPropRef.current) roofPropRef.current.setMatrixAt(roofPropCount, dummy.matrix)
          roofPropCount++

          // Antenna with blinking beacon atop tall buildings
          if (h > 40 && antennaCount < MAX_ROOF_ANTENNAS) {
            dummy.position.set(xBlock, roofY + 5, zBlock)
            dummy.scale.set(0.3, 10, 0.3)
            dummy.updateMatrix()
            if (antennaRef.current) antennaRef.current.setMatrixAt(antennaCount, dummy.matrix)

            dummy.position.set(xBlock, roofY + 10.2, zBlock)
            dummy.scale.set(1, 1, 1)
            dummy.updateMatrix()
            if (beaconRef.current) beaconRef.current.setMatrixAt(antennaCount, dummy.matrix)
            antennaCount++
          }
        }

        // Street light & low poly car in the adjacent street
        if (streetLightCount < MAX_STREET_LIGHTS && rng() > 0.6) {
          const sx = xBlock + (xBlock > px ? -12 : 12)
          const sy = heightAt(sx, zBlock)
          dummy.position.set(sx, sy + 3.5, zBlock)
          dummy.scale.set(0.2, 7, 0.2)
          dummy.rotation.set(0, 0, 0)
          dummy.updateMatrix()
          if (streetLightRef.current) streetLightRef.current.setMatrixAt(streetLightCount, dummy.matrix)
          streetLightCount++
        }

        if (carCount < MAX_CARS && rng() > 0.7) {
          const cx = xBlock + (xBlock > px ? -14 : 14)
          const cy = heightAt(cx, zBlock)
          dummy.position.set(cx, cy + 0.6, zBlock + (rng() - 0.5) * 10)
          dummy.scale.set(1.8, 1.1, 3.8)
          dummy.rotation.set(0, rotY, 0)
          dummy.updateMatrix()
          if (carRef.current) carRef.current.setMatrixAt(carCount, dummy.matrix)
          carCount++
        }
      }
    }

    // Flush urban instance updates
    if (slabRef.current) {
      slabRef.current.count = slabCount
      slabRef.current.instanceMatrix.needsUpdate = true
      if (slabRef.current.instanceColor) slabRef.current.instanceColor.needsUpdate = true
    }
    if (setbackLowerRef.current) {
      setbackLowerRef.current.count = setbackCount
      setbackLowerRef.current.instanceMatrix.needsUpdate = true
      if (setbackLowerRef.current.instanceColor) setbackLowerRef.current.instanceColor.needsUpdate = true
    }
    if (setbackUpperRef.current) {
      setbackUpperRef.current.count = setbackCount
      setbackUpperRef.current.instanceMatrix.needsUpdate = true
      if (setbackUpperRef.current.instanceColor) setbackUpperRef.current.instanceColor.needsUpdate = true
    }
    if (podiumBaseRef.current) {
      podiumBaseRef.current.count = podiumCount
      podiumBaseRef.current.instanceMatrix.needsUpdate = true
      if (podiumBaseRef.current.instanceColor) podiumBaseRef.current.instanceColor.needsUpdate = true
    }
    if (podiumTowerRef.current) {
      podiumTowerRef.current.count = podiumCount
      podiumTowerRef.current.instanceMatrix.needsUpdate = true
      if (podiumTowerRef.current.instanceColor) podiumTowerRef.current.instanceColor.needsUpdate = true
    }
    if (roofPropRef.current) {
      roofPropRef.current.count = roofPropCount
      roofPropRef.current.instanceMatrix.needsUpdate = true
    }
    if (antennaRef.current) {
      antennaRef.current.count = antennaCount
      antennaRef.current.instanceMatrix.needsUpdate = true
    }
    if (beaconRef.current) {
      beaconRef.current.count = antennaCount
      beaconRef.current.instanceMatrix.needsUpdate = true
    }
    if (streetLightRef.current) {
      streetLightRef.current.count = streetLightCount
      streetLightRef.current.instanceMatrix.needsUpdate = true
    }
    if (carRef.current) {
      carRef.current.count = carCount
      carRef.current.instanceMatrix.needsUpdate = true
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. COASTAL PORT: Shipping Containers, Pier & Cranes (-400 to -600 Z)
    // ─────────────────────────────────────────────────────────────────────────
    if (containerRef.current && craneRef.current) {
      const containerColors = ['#1e40af', '#b91c1c', '#047857', '#d97706', '#4b5563']
      let cCount = 0
      for (let i = 0; i < MAX_CONTAINERS; i++) {
        const cz = -420 - rng() * 140
        const side = i % 2 === 0 ? 1 : -1
        const cx = pathX(cz) + side * (26 + rng() * 55)
        const cy = heightAt(cx, cz)
        if (cy > 0.5) {
          const stack = Math.floor(rng() * 3) // 0, 1, or 2 stacked
          dummy.position.set(cx, cy + 1.2 + stack * 2.4, cz)
          dummy.scale.set(2.4, 2.4, 6.0)
          dummy.rotation.set(0, rng() > 0.5 ? 0 : Math.PI / 2, 0)
          dummy.updateMatrix()
          containerRef.current.setMatrixAt(cCount, dummy.matrix)

          colBuf.set(containerColors[Math.floor(rng() * containerColors.length)])
          containerRef.current.setColorAt(cCount, colBuf)
          cCount++
        }
      }
      containerRef.current.count = cCount
      containerRef.current.instanceMatrix.needsUpdate = true
      if (containerRef.current.instanceColor) containerRef.current.instanceColor.needsUpdate = true

      // Port harbor cranes
      for (let i = 0; i < MAX_CRANES; i++) {
        const crz = -440 - i * 26
        const crx = pathX(crz) + (i % 2 === 0 ? 45 : -45)
        const cry = heightAt(crx, crz)
        dummy.position.set(crx, cry + 14, crz)
        dummy.scale.set(4, 28, 4)
        dummy.rotation.set(0, i % 2 === 0 ? -0.4 : 0.4, 0)
        dummy.updateMatrix()
        craneRef.current.setMatrixAt(i, dummy.matrix)
      }
      craneRef.current.count = MAX_CRANES
      craneRef.current.instanceMatrix.needsUpdate = true
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. MOUNTAIN ZONE: Slate Rocks & Pylons (-600 to -1000 Z)
    // ─────────────────────────────────────────────────────────────────────────
    if (rockRef.current) {
      for (let i = 0; i < MAX_ROCKS; i++) {
        const rz = -600 - rng() * 380
        const rx = pathX(rz) + (rng() > 0.5 ? 1 : -1) * (20 + rng() * 110)
        const ry = heightAt(rx, rz)
        const rScale = 2.4 + rng() * 5.5
        dummy.position.set(rx, ry + rScale * 0.35, rz)
        dummy.scale.set(rScale, rScale * (0.8 + rng() * 0.5), rScale)
        dummy.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI)
        dummy.updateMatrix()
        rockRef.current.setMatrixAt(i, dummy.matrix)
      }
      rockRef.current.count = MAX_ROCKS
      rockRef.current.instanceMatrix.needsUpdate = true
    }

    if (pylonRef.current && pylonLightRef.current) {
      for (let i = 0; i < MAX_PYLONS; i++) {
        const pz = -620 - (i / (MAX_PYLONS - 1)) * 360
        const side = i % 2 === 0 ? 1 : -1
        const px = pathX(pz) + side * (55 + rng() * 35)
        const py = heightAt(px, pz)
        const ph = 26 + rng() * 12

        dummy.position.set(px, py + ph / 2, pz)
        dummy.scale.set(1.4, ph, 1.4)
        dummy.rotation.set(0, 0, 0)
        dummy.updateMatrix()
        pylonRef.current.setMatrixAt(i, dummy.matrix)

        dummy.position.set(px, py + ph + 0.4, pz)
        dummy.scale.set(1, 1, 1)
        dummy.updateMatrix()
        pylonLightRef.current.setMatrixAt(i, dummy.matrix)
      }
      pylonRef.current.count = MAX_PYLONS
      pylonRef.current.instanceMatrix.needsUpdate = true
      pylonLightRef.current.count = MAX_PYLONS
      pylonLightRef.current.instanceMatrix.needsUpdate = true
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. RURAL ZONE: Trees & Farmhouses (-1000 to -1600 Z)
    // ─────────────────────────────────────────────────────────────────────────
    if (treeRef.current) {
      let tCount = 0
      while (tCount < MAX_TREES) {
        const isRural = rng() > 0.25
        const tz = isRural ? -1000 - rng() * 580 : -380 - rng() * 600
        const tx = pathX(tz) + (rng() > 0.5 ? 1 : -1) * (18 + rng() * 115)
        const ty = heightAt(tx, tz)
        if (ty > 1.2) {
          const th = 4.5 + rng() * 6.0
          const tr = 1.3 + rng() * 1.5
          dummy.position.set(tx, ty + th / 2, tz)
          dummy.scale.set(tr, th, tr)
          dummy.rotation.set(0, rng() * Math.PI * 2, 0)
          dummy.updateMatrix()
          treeRef.current.setMatrixAt(tCount, dummy.matrix)
          tCount++
        }
      }
      treeRef.current.count = tCount
      treeRef.current.instanceMatrix.needsUpdate = true
    }

    if (farmhouseRef.current) {
      for (let i = 0; i < MAX_FARMHOUSES; i++) {
        const fz = -1020 - rng() * 540
        const fx = pathX(fz) + (rng() > 0.5 ? 1 : -1) * (35 + rng() * 70)
        const fy = heightAt(fx, fz)
        dummy.position.set(fx, fy + 3.0, fz)
        dummy.scale.set(8 + rng() * 4, 6, 12 + rng() * 4)
        dummy.rotation.set(0, rng() * Math.PI, 0)
        dummy.updateMatrix()
        farmhouseRef.current.setMatrixAt(i, dummy.matrix)
      }
      farmhouseRef.current.count = MAX_FARMHOUSES
      farmhouseRef.current.instanceMatrix.needsUpdate = true
    }
  }, [])

  // Animate dynamic lighting, beacon blinks, and snow particles in useFrame
  useFrame(({ clock }) => {
    const tod = sampleTimeOfDay(flight.damped)

    // Update window glow uniform (strictly 0.0 in day mode)
    customUniforms.uWindowGlow.value = tod.windowGlow

    // Blinking pylon beacons
    const blink = Math.sin(clock.getElapsedTime() * 4.0) > 0 ? 1.0 : 0.08
    propMats.beacon.opacity = blink

    // Drift snow particles along mountain pass
    if (snowParticlesRef.current) {
      const positions = snowParticlesRef.current.geometry.attributes.position as THREE.BufferAttribute
      const count = positions.count
      for (let i = 0; i < count; i++) {
        let y = positions.getY(i) - 0.12
        if (y < 25) y = 75
        positions.setY(i, y)
      }
      positions.needsUpdate = true
    }
  })

  return (
    <group>
      {/* ── Urban Building Archetypes ── */}
      <instancedMesh
        ref={slabRef}
        args={[unitBoxGeo, facadeMaterial, MAX_SLAB_BLDG]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={setbackLowerRef}
        args={[unitBoxGeo, facadeMaterial, MAX_SETBACK_LOWER]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={setbackUpperRef}
        args={[unitBoxGeo, facadeMaterial, MAX_SETBACK_UPPER]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={podiumBaseRef}
        args={[unitBoxGeo, facadeMaterial, MAX_PODIUM_BASE]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={podiumTowerRef}
        args={[unitBoxGeo, facadeMaterial, MAX_PODIUM_TOWER]}
        frustumCulled={false}
      />

      {/* ── Roof Props & Lighting ── */}
      <instancedMesh
        ref={roofPropRef}
        args={[unitBoxGeo, propMats.roof, MAX_ROOF_PROPS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={antennaRef}
        args={[cylinderGeo, propMats.antenna, MAX_ROOF_ANTENNAS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={beaconRef}
        args={[sphereGeo, propMats.beacon, MAX_ROOF_ANTENNAS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={streetLightRef}
        args={[cylinderGeo, propMats.streetLightPole, MAX_STREET_LIGHTS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={carRef}
        args={[unitBoxGeo, propMats.car, MAX_CARS]}
        frustumCulled={false}
      />

      {/* ── Coastal Port Props ── */}
      <instancedMesh
        ref={containerRef}
        args={[unitBoxGeo, propMats.container, MAX_CONTAINERS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={craneRef}
        args={[unitBoxGeo, propMats.crane, MAX_CRANES]}
        frustumCulled={false}
      />

      {/* ── Mountain Rocks & Pylons ── */}
      <instancedMesh
        ref={rockRef}
        args={[rockGeo, propMats.rock, MAX_ROCKS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={pylonRef}
        args={[unitBoxGeo, propMats.pylon, MAX_PYLONS]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={pylonLightRef}
        args={[sphereGeo, propMats.beacon, MAX_PYLONS]}
        frustumCulled={false}
      />

      {/* ── Rural Farmhouses & Trees ── */}
      <instancedMesh
        ref={farmhouseRef}
        args={[unitBoxGeo, propMats.farmhouse, MAX_FARMHOUSES]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={treeRef}
        args={[coneGeo, propMats.tree, MAX_TREES]}
        frustumCulled={false}
      />

      {/* ── Mountain Snow / Mist Particles ── */}
      <points ref={snowParticlesRef} geometry={snowGeo} material={snowMat} frustumCulled={false} />
    </group>
  )
}
