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

const BLDG_COUNT = 320
const ROOF_COUNT = 160
const TREE_COUNT = 550
const ROCK_COUNT = 140
const PYLON_COUNT = 20

export default function Props() {
  const bldgRef = useRef<THREE.InstancedMesh>(null)
  const roofRef = useRef<THREE.InstancedMesh>(null)
  const treeRef = useRef<THREE.InstancedMesh>(null)
  const rockRef = useRef<THREE.InstancedMesh>(null)
  const pylonRef = useRef<THREE.InstancedMesh>(null)
  const lightRef = useRef<THREE.InstancedMesh>(null)
  const bldgMatRef = useRef<THREE.MeshStandardMaterial | null>(null)
  const lightMatRef = useRef<THREE.MeshBasicMaterial | null>(null)

  // Window emissive canvas texture
  const windowTex = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 64
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#05070a'
    ctx.fillRect(0, 0, 64, 64)
    const rng = mkRng(42)
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 6; c++) {
        if (rng() > 0.4) {
          ctx.fillStyle = `rgba(255, 186, 50, ${0.7 + rng() * 0.3})`
          ctx.fillRect(c * 10 + 2, r * 8 + 2, 6, 4)
        }
      }
    }
    const tex = new THREE.CanvasTexture(canvas)
    tex.wrapS = THREE.RepeatWrapping
    tex.wrapT = THREE.RepeatWrapping
    tex.repeat.set(1, 2)
    return tex
  }, [])

  const bldgGeo = useMemo(() => new THREE.BoxGeometry(1, 1, 1), [])
  const bldgMat = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({
      color: '#1a2634',
      emissive: '#ffbe40',
      emissiveMap: windowTex,
      emissiveIntensity: 0.4,
      roughness: 0.6,
      metalness: 0.4,
    })
    bldgMatRef.current = mat
    return mat
  }, [windowTex])

  // Roof Equipment / Parapets
  const roofGeo = useMemo(() => new THREE.BoxGeometry(1, 1, 1), [])
  const roofMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#0e1822',
        roughness: 0.7,
        metalness: 0.5,
      }),
    []
  )

  const treeGeo = useMemo(() => new THREE.ConeGeometry(1, 1, 5), [])
  const treeMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#1a3622',
        roughness: 0.9,
      }),
    []
  )

  // Mountain Rocks / Boulders
  const rockGeo = useMemo(() => new THREE.DodecahedronGeometry(1, 0), [])
  const rockMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#424a52',
        roughness: 0.95,
        metalness: 0.1,
        flatShading: true,
      }),
    []
  )

  const pylonGeo = useMemo(() => new THREE.BoxGeometry(0.5, 1, 0.5), [])
  const pylonMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: '#263442',
        roughness: 0.6,
      }),
    []
  )

  const lightGeo = useMemo(() => new THREE.SphereGeometry(0.35, 6, 6), [])
  const lightMat = useMemo(() => {
    const mat = new THREE.MeshBasicMaterial({
      color: '#ff2a2a',
      toneMapped: false,
    })
    lightMatRef.current = mat
    return mat
  }, [])

  useLayoutEffect(() => {
    const dummy = new THREE.Object3D()
    const colorBuf = new THREE.Color()

    // 1. Buildings: urban range (0 to -400)
    if (bldgRef.current && roofRef.current) {
      const rng = mkRng(101)
      let roofIdx = 0
      for (let i = 0; i < BLDG_COUNT; i++) {
        const z = -rng() * 390
        const side = rng() > 0.5 ? 1 : -1
        const dist = 16 + rng() * 85
        const px = pathX(z)
        const x = px + side * dist

        const w = 4 + rng() * 7
        const d = 4 + rng() * 7
        const distFromCenter = Math.abs(z - -200) / 200
        const h = 8 + (1 - distFromCenter * 0.5) * (16 + rng() * 38)

        const y = heightAt(x, z)
        dummy.position.set(x, y + h / 2, z)
        dummy.scale.set(w, h, d)
        dummy.rotation.set(0, rng() * Math.PI, 0)
        dummy.updateMatrix()
        bldgRef.current.setMatrixAt(i, dummy.matrix)

        // Varied facade shades
        const cVal = 0.12 + rng() * 0.1
        colorBuf.setRGB(cVal, cVal * 1.25, cVal * 1.45)
        bldgRef.current.setColorAt(i, colorBuf)

        // Roof details for taller buildings
        if (h > 20 && roofIdx < ROOF_COUNT) {
          const rw = w * 0.4
          const rd = d * 0.4
          const rh = 1.2 + rng() * 2.2
          dummy.position.set(x, y + h + rh / 2, z)
          dummy.scale.set(rw, rh, rd)
          dummy.updateMatrix()
          roofRef.current.setMatrixAt(roofIdx, dummy.matrix)
          roofIdx++
        }
      }
      bldgRef.current.instanceMatrix.needsUpdate = true
      if (bldgRef.current.instanceColor) bldgRef.current.instanceColor.needsUpdate = true
      roofRef.current.instanceMatrix.needsUpdate = true
    }

    // 2. Trees: rural (-1000 to -1600) and coastal/mountain margins
    if (treeRef.current) {
      const rng = mkRng(202)
      let placed = 0
      while (placed < TREE_COUNT) {
        const isRural = rng() > 0.3
        const z = isRural ? -1000 - rng() * 580 : -380 - rng() * 580
        const px = pathX(z)
        const side = rng() > 0.5 ? 1 : -1
        const dist = 18 + rng() * 120
        const x = px + side * dist

        const y = heightAt(x, z)
        if (y > 1.2) {
          const h = 4.5 + rng() * 6.5
          const r = 1.3 + rng() * 1.6
          dummy.position.set(x, y + h / 2, z)
          dummy.scale.set(r, h, r)
          dummy.rotation.set(0, rng() * Math.PI * 2, 0)
          dummy.updateMatrix()
          treeRef.current.setMatrixAt(placed, dummy.matrix)

          const gVal = 0.12 + rng() * 0.1
          colorBuf.setRGB(gVal * 0.6, gVal * 1.6, gVal * 0.8)
          treeRef.current.setColorAt(placed, colorBuf)
          placed++
        }
      }
      treeRef.current.instanceMatrix.needsUpdate = true
      if (treeRef.current.instanceColor) treeRef.current.instanceColor.needsUpdate = true
    }

    // 3. Mountain Rocks / Boulders: (-600 to -1000)
    if (rockRef.current) {
      const rng = mkRng(404)
      for (let i = 0; i < ROCK_COUNT; i++) {
        const z = -600 - rng() * 390
        const px = pathX(z)
        const side = rng() > 0.5 ? 1 : -1
        const dist = 24 + rng() * 110
        const x = px + side * dist
        const y = heightAt(x, z)

        const rScale = 2.5 + rng() * 5.0
        dummy.position.set(x, y + rScale * 0.35, z)
        dummy.scale.set(rScale, rScale * (0.8 + rng() * 0.4), rScale)
        dummy.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI)
        dummy.updateMatrix()
        rockRef.current.setMatrixAt(i, dummy.matrix)

        // Slate/rock tones
        const c = 0.22 + rng() * 0.15
        colorBuf.setRGB(c, c * 0.95, c * 0.9)
        rockRef.current.setColorAt(i, colorBuf)
      }
      rockRef.current.instanceMatrix.needsUpdate = true
      if (rockRef.current.instanceColor) rockRef.current.instanceColor.needsUpdate = true
    }

    // 4. Pylons / radio masts on mountain ridge (-620 to -980)
    if (pylonRef.current && lightRef.current) {
      const rng = mkRng(303)
      for (let i = 0; i < PYLON_COUNT; i++) {
        const z = -620 - (i / (PYLON_COUNT - 1)) * 360
        const side = i % 2 === 0 ? 1 : -1
        const dist = 65 + rng() * 45
        const px = pathX(z)
        const x = px + side * dist
        const y = heightAt(x, z)
        const h = 24 + rng() * 14

        dummy.position.set(x, y + h / 2, z)
        dummy.scale.set(1.2, h, 1.2)
        dummy.rotation.set(0, 0, 0)
        dummy.updateMatrix()
        pylonRef.current.setMatrixAt(i, dummy.matrix)

        // Blinking beacon atop pylon
        dummy.position.set(x, y + h + 0.4, z)
        dummy.scale.set(1, 1, 1)
        dummy.updateMatrix()
        lightRef.current.setMatrixAt(i, dummy.matrix)
      }
      pylonRef.current.instanceMatrix.needsUpdate = true
      lightRef.current.instanceMatrix.needsUpdate = true
    }
  }, [])

  useFrame(({ clock }) => {
    const tod = sampleTimeOfDay(flight.damped)
    if (bldgMatRef.current) {
      bldgMatRef.current.emissiveIntensity = 0.4 + tod.windowGlow * 1.6
    }
    // Blink red pylon warning lights
    if (lightMatRef.current) {
      const blink = Math.sin(clock.getElapsedTime() * 4.0) > 0 ? 1.0 : 0.1
      lightMatRef.current.opacity = blink
    }
  })

  return (
    <group>
      <instancedMesh
        ref={bldgRef}
        args={[bldgGeo, bldgMat, BLDG_COUNT]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={roofRef}
        args={[roofGeo, roofMat, ROOF_COUNT]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={treeRef}
        args={[treeGeo, treeMat, TREE_COUNT]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={rockRef}
        args={[rockGeo, rockMat, ROCK_COUNT]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={pylonRef}
        args={[pylonGeo, pylonMat, PYLON_COUNT]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={lightRef}
        args={[lightGeo, lightMat, PYLON_COUNT]}
        frustumCulled={false}
      />
    </group>
  )
}
