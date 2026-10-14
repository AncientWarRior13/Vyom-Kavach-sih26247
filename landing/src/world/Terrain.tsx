import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { heightAt } from './terrainHeight'

const W = 520
const L = 1800
const SEG_W = 130
const SEG_L = 450
const CENTRE_Z = -800

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

function fieldHash(bx: number, bz: number): number {
  const n = Math.sin(bx * 12.9898 + bz * 78.233) * 43758.5453
  return n - Math.floor(n)
}

export default function Terrain() {
  const seaMeshRef = useRef<THREE.Mesh>(null)

  const { geometry, material } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(W, L, SEG_W, SEG_L)
    geo.rotateX(-Math.PI / 2)
    geo.translate(0, 0, CENTRE_Z)

    const pos = geo.attributes.position
    const colors: number[] = []
    const colBuf = new THREE.Color()

    const colUrban = new THREE.Color('#1a2420')
    const colSand = new THREE.Color('#bfa97a')
    const colCoastGreen = new THREE.Color('#223824')
    const colMountain = new THREE.Color('#3a3f44')
    const colSnow = new THREE.Color('#e8eef2')
    const colFieldA = new THREE.Color('#3d5a34')
    const colFieldB = new THREE.Color('#6b7a3a')

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i)
      const z = pos.getZ(i)
      const y = heightAt(x, z)
      pos.setY(i, y)

      // Biome-based vertex colors
      if (z >= -400) {
        // Urban
        colBuf.copy(colUrban)
      } else if (z >= -600) {
        // Coastal: sand near sea level, fading to green inland
        const seaDist = smoothstep(-2, 12, y)
        colBuf.copy(colSand).lerp(colCoastGreen, seaDist)
      } else if (z >= -1000) {
        // Mountain: dark rock with snow above y > 62
        const snowBlend = smoothstep(58, 72, y)
        colBuf.copy(colMountain).lerp(colSnow, snowBlend)
      } else {
        // Rural: patchwork of field bands in blocks of ~40 units
        const bx = Math.floor(x / 40)
        const bz = Math.floor(z / 40)
        const h = fieldHash(bx, bz)
        colBuf.copy(h > 0.5 ? colFieldA : colFieldB)
      }

      colors.push(colBuf.r, colBuf.g, colBuf.b)
    }

    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    geo.computeVertexNormals()

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      flatShading: true,
      roughness: 1,
      metalness: 0,
    })

    return { geometry: geo, material: mat }
  }, [])

  const { seaGeo, seaMat } = useMemo(() => {
    const geo = new THREE.PlaneGeometry(W, 220, 64, 32)
    geo.rotateX(-Math.PI / 2)
    const mat = new THREE.MeshStandardMaterial({
      color: '#0d2b3d',
      transparent: true,
      opacity: 0.92,
      roughness: 0.2,
      metalness: 0.3,
    })
    return { seaGeo: geo, seaMat: mat }
  }, [])

  useFrame(({ clock }) => {
    if (!seaMeshRef.current) return
    const time = clock.getElapsedTime()
    const geo = seaMeshRef.current.geometry as THREE.BufferGeometry
    if (!geo || !geo.attributes.position) return
    const pos = geo.attributes.position
    for (let i = 0; i < pos.count; i++) {
      const u = pos.getX(i)
      const v = pos.getY(i)
      const wave = Math.sin(u * 0.05 + time * 1.5) * Math.cos(v * 0.05 + time) * 0.35
      pos.setZ(i, wave)
    }
    pos.needsUpdate = true
  })

  return (
    <group>
      {/* Terrain mesh */}
      <mesh geometry={geometry} material={material} frustumCulled={false} receiveShadow />

      {/* Sea plane at y = 0 over coastal range (-400 to -600 -> center z = -500) */}
      <mesh
        ref={seaMeshRef}
        geometry={seaGeo}
        material={seaMat}
        position={[0, 0, -500]}
        frustumCulled={false}
      />
    </group>
  )
}
