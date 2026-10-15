import React, { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { flight } from '../lib/flightState'
import { flightCurve } from './path'
import { heightAt } from './terrainHeight'
import {
  pathX,
  URBAN_MARGIN,
  BASE_CLEARANCE,
  MAX_DESCENT_SLOPE,
  buildingMargin,
} from './pathCore'
import { SECTIONS } from '../lib/sections'

interface DroneProps {
  onTelemetryUpdate?: (data: {
    fps: number
    delta: number
    deltaSpike: boolean
    pos: THREE.Vector3
    rotDelta: number
  }) => void
}

export default function Drone({ onTelemetryUpdate }: DroneProps) {
  // Master group (position & orientation via quaternion slerp)
  const masterGroupRef = useRef<THREE.Group>(null!)
  // Hover bob child group (receives hover bob only, camera never sees it)
  const hoverGroupRef = useRef<THREE.Group>(null!)
  // 4 horizontal rotor groups
  const rotorsRef = useRef<THREE.Group[]>([])

  // Persistent math state to avoid allocations in useFrame
  const initAlt = heightAt(0, 0) + URBAN_MARGIN + BASE_CLEARANCE
  const dronePos = useRef(new THREE.Vector3(0, initAlt, 0))
  const smoothedAlt = useRef(initAlt)
  const lastZ = useRef<number | null>(null)

  const droneQuat = useRef(new THREE.Quaternion())
  const prevQuat = useRef(new THREE.Quaternion())
  const targetQuat = useRef(new THREE.Quaternion())
  const targetEuler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'))
  const filteredRoll = useRef(0)
  const filteredLateral = useRef(0)
  const camPos = useRef(new THREE.Vector3(0, initAlt + 4.2, 14.8))
  const lookTarget = useRef(new THREE.Vector3())
  const lastTimeRef = useRef(performance.now())
  const fpsRef = useRef(60)

  // Pre-created geometries and materials for optimal performance
  const materials = useMemo(() => {
    return {
      fuselage: new THREE.MeshStandardMaterial({
        color: '#0d1824',
        roughness: 0.25,
        metalness: 0.85,
      }),
      fuselageTrim: new THREE.MeshStandardMaterial({
        color: '#1a3050',
        emissive: '#3dd6c6',
        emissiveIntensity: 0.35,
        roughness: 0.3,
        metalness: 0.7,
      }),
      arm: new THREE.MeshStandardMaterial({
        color: '#070f17',
        roughness: 0.4,
        metalness: 0.9,
      }),
      motor: new THREE.MeshStandardMaterial({
        color: '#1b2a38',
        roughness: 0.2,
        metalness: 0.9,
      }),
      rotorBlade: new THREE.MeshStandardMaterial({
        color: '#2a445e',
        roughness: 0.3,
        metalness: 0.4,
        transparent: true,
        opacity: 0.85,
      }),
      rotorDisc: new THREE.MeshBasicMaterial({
        color: '#3dd6c6',
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
      }),
      cameraPod: new THREE.MeshStandardMaterial({
        color: '#060a0e',
        roughness: 0.2,
        metalness: 0.95,
      }),
      cameraLens: new THREE.MeshStandardMaterial({
        color: '#3dd6c6',
        emissive: '#3dd6c6',
        emissiveIntensity: 0.9,
        roughness: 0.1,
      }),
      navRed: new THREE.MeshBasicMaterial({ color: '#ff3b3b' }),
      navGreen: new THREE.MeshBasicMaterial({ color: '#10b981' }),
      strobeWhite: new THREE.MeshBasicMaterial({ color: '#ffffff' }),
    }
  }, [])

  useFrame((state, delta) => {
    const clampedDelta = Math.min(delta, 0.05)
    const now = performance.now()
    const frameTime = (now - lastTimeRef.current) / 1000
    lastTimeRef.current = now
    fpsRef.current = Math.round(1 / Math.max(0.001, frameTime))
    const deltaSpike = clampedDelta > 0.033

    // 1. Frame-rate independent progress damping: k = 1 - Math.exp(-rate * delta)
    const dampRate = 4.5
    const kProgress = 1 - Math.exp(-dampRate * clampedDelta)
    flight.damped += (flight.progress - flight.damped) * kProgress
    const p = Math.max(0.0001, Math.min(0.9999, flight.damped))

    // 2. Flight curve sampling: Position & Tangent
    const curvePt = flightCurve.getPointAt(p)
    const tangent = flightCurve.getTangentAt(p).normalize()

    // 3. Smoothed Altitude: Look ahead 40 units to clear buildings and ridge lines
    let maxGroundAhead = 0
    for (let offset = 0; offset <= 40; offset += 10) {
      const aheadZ = curvePt.z - offset
      const aheadX = pathX(aheadZ)
      const ground = heightAt(aheadX, aheadZ) + buildingMargin(aheadZ)
      maxGroundAhead = Math.max(maxGroundAhead, ground)
    }
    const targetAlt = maxGroundAhead + BASE_CLEARANCE

    // Distance-based descent limit (dz = abs(curvePt.z - lastZ))
    const isFirstFrame = lastZ.current === null
    const dz = isFirstFrame ? 0 : Math.abs(curvePt.z - lastZ.current!)
    lastZ.current = curvePt.z

    if (isFirstFrame) {
      smoothedAlt.current = targetAlt
    } else {
      const kAlt = 1 - Math.exp(-3.5 * clampedDelta)
      let next = smoothedAlt.current + (targetAlt - smoothedAlt.current) * kAlt
      // Limit descent slope to MAX_DESCENT_SLOPE * dz; climbing stays unlimited
      next = Math.max(next, smoothedAlt.current - MAX_DESCENT_SLOPE * dz)
      // Safety floor: at least heightAt(curvePt.x, curvePt.z) + 8
      smoothedAlt.current = Math.max(next, heightAt(curvePt.x, curvePt.z) + 8)
    }

    // Update master position
    dronePos.current.set(curvePt.x, smoothedAlt.current, curvePt.z)
    if (masterGroupRef.current) {
      masterGroupRef.current.position.copy(dronePos.current)
    }

    // 4. Stable Orientation via Quaternion Slerp (Nose points -Z)
    // Yaw from tangent (direction of travel)
    const yaw = Math.atan2(tangent.x, -tangent.z)

    // Pitch from tangent vertical angle, clamped to +/-6 degrees (+/-0.1047 rad)
    const horizDist = Math.hypot(tangent.x, tangent.z)
    const rawPitch = Math.atan2(tangent.y, Math.max(0.001, horizDist))
    const pitch = THREE.MathUtils.clamp(rawPitch, -0.1047, 0.1047)

    // Bank (Roll) from low-passed lateral curvature, clamped to +/-15 degrees (+/-0.2618 rad)
    const pNext = Math.min(1.0, p + 0.012)
    const tangentNext = flightCurve.getTangentAt(pNext).normalize()
    const yawNext = Math.atan2(tangentNext.x, -tangentNext.z)
    let deltaYaw = yawNext - yaw
    if (deltaYaw > Math.PI) deltaYaw -= Math.PI * 2
    if (deltaYaw < -Math.PI) deltaYaw += Math.PI * 2
    const lateralCurvature = deltaYaw / 0.012
    const targetRoll = THREE.MathUtils.clamp(-lateralCurvature * 0.2, -0.2618, 0.2618)
    const kRoll = 1 - Math.exp(-4.0 * clampedDelta)
    filteredRoll.current += (targetRoll - filteredRoll.current) * kRoll

    // Construct target quaternion and slerp smoothly
    targetEuler.current.set(pitch, yaw, filteredRoll.current, 'YXZ')
    targetQuat.current.setFromEuler(targetEuler.current)
    if (isFirstFrame) {
      droneQuat.current.copy(targetQuat.current)
    } else {
      const kQuat = 1 - Math.exp(-7.0 * clampedDelta)
      droneQuat.current.slerp(targetQuat.current, kQuat)
    }

    if (masterGroupRef.current) {
      masterGroupRef.current.quaternion.copy(droneQuat.current)
    }

    // Compute rotation delta for telemetry jitter check
    const rotDelta = prevQuat.current.angleTo(droneQuat.current)
    prevQuat.current.copy(droneQuat.current)

    // 5. Hover bob applied ONLY to child group (max 0.14 units, never on camera!)
    if (hoverGroupRef.current) {
      hoverGroupRef.current.position.y = Math.sin(state.clock.elapsedTime * Math.PI) * 0.14
    }

    // 6. Horizontal Rotors Spin around local Y only
    const flightSpeed = Math.abs(flight.speed)
    const spinSpeed = THREE.MathUtils.clamp(25 + flightSpeed * 35, 20, 80) * clampedDelta
    rotorsRef.current.forEach((rotor, idx) => {
      if (rotor) {
        const dir = idx % 2 === 0 ? 1 : -1
        rotor.rotation.y += dir * spinSpeed
      }
    })

    // 7. Framing & Camera Offset (Step 3a):
    // Drone sits in lower-center (~62% down the viewport).
    // Lateral shift: when text is on 'left', drone shifts right (+15% viewport width);
    // when text is on 'right', drone shifts left (-15% viewport width);
    // when 'center', drone is centered.
    const secIdx = Math.min(7, Math.floor(p * 8))
    const curSide = SECTIONS[secIdx]?.side || 'center'
    // Target lateral shift: -1.0 for drone left (text right), +1.0 for drone right (text left)
    const targetShiftFactor = curSide === 'left' ? 1.0 : curSide === 'right' ? -1.0 : 0.0
    const targetLateralOffset = targetShiftFactor * 2.8

    if (isFirstFrame) {
      filteredLateral.current = targetLateralOffset
    } else {
      const kLateral = 1 - Math.exp(-3.5 * clampedDelta)
      filteredLateral.current += (targetLateralOffset - filteredLateral.current) * kLateral
    }

    // Camera chase geometry:
    // camDist = 14.8, camHeight = 4.2
    // Normal vector perpendicular to heading in horizontal plane: (cos(yaw), -sin(yaw))
    const rightX = Math.cos(yaw)
    const rightZ = -Math.sin(yaw)

    const camDist = 14.8
    const camHeight = 4.2
    const camOffsetX = -Math.sin(yaw) * camDist + rightX * (filteredLateral.current * 0.4)
    const camOffsetZ = Math.cos(yaw) * camDist + rightZ * (filteredLateral.current * 0.4)
    const targetCamPos = new THREE.Vector3(
      dronePos.current.x + camOffsetX,
      dronePos.current.y + camHeight,
      dronePos.current.z + camOffsetZ
    )

    if (isFirstFrame) {
      camPos.current.copy(targetCamPos)
    } else {
      const kCam = 1 - Math.exp(-6.0 * clampedDelta)
      camPos.current.lerp(targetCamPos, kCam)
    }
    state.camera.position.copy(camPos.current)

    // Camera looks ahead at lookTarget, with vertical offset +2.1 (framing drone at ~62% viewport)
    // and lateral offset opposite to drone shift to frame the drone toward the free side.
    const lookAheadP = Math.min(1.0, p + 0.024)
    const lookAheadPt = flightCurve.getPointAt(lookAheadP)
    lookTarget.current.set(
      lookAheadPt.x - rightX * (filteredLateral.current * 0.6),
      dronePos.current.y + 2.1,
      lookAheadPt.z - rightZ * (filteredLateral.current * 0.6)
    )
    state.camera.lookAt(lookTarget.current)

    // Update global flight state vectors
    flight.droneWorld.copy(dronePos.current)
    flight.droneForward.set(-Math.sin(yaw), Math.sin(pitch), -Math.cos(yaw)).normalize()

    // Telemetry callback
    if (onTelemetryUpdate) {
      onTelemetryUpdate({
        fps: fpsRef.current,
        delta: Math.round(clampedDelta * 1000),
        deltaSpike,
        pos: dronePos.current.clone(),
        rotDelta: Number((rotDelta * (180 / Math.PI)).toFixed(3)),
      })
    }
  })

  return (
    <group ref={masterGroupRef}>
      {/* ── Hover Bob Child Group (Camera is unaffected by this bob) ── */}
      <group ref={hoverGroupRef}>
        {/* ── 1. Fuselage Body (Aerodynamic Carbon Chassis, Nose points -Z) ── */}
        <group>
          {/* Main Chassis Box */}
          <mesh castShadow receiveShadow material={materials.fuselage}>
            <boxGeometry args={[1.0, 0.36, 1.9]} />
          </mesh>

          {/* Top Avionics Canopy */}
          <mesh position={[0, 0.22, -0.1]} material={materials.fuselageTrim}>
            <boxGeometry args={[0.7, 0.22, 1.2]} />
          </mesh>

          {/* Glowing Tactical Status Bar */}
          <mesh position={[0, 0.34, -0.2]}>
            <boxGeometry args={[0.3, 0.04, 0.8]} />
            <meshBasicMaterial color="#3dd6c6" />
          </mesh>

          {/* Under-Nose Gimbal Optical Pod (Under Nose at -Z) */}
          <group position={[0, -0.24, -0.75]}>
            <mesh material={materials.cameraPod}>
              <sphereGeometry args={[0.26, 16, 16]} />
            </mesh>
            <mesh position={[0, 0, -0.2]} material={materials.cameraLens}>
              <cylinderGeometry args={[0.12, 0.12, 0.12, 16]} />
            </mesh>
          </group>
        </group>

        {/* ── 2. Four Diagonal Arms (Attached to body at 45 degrees) ── */}
        {[
          { id: 'FL', x: -0.9, z: -0.9, rot: Math.PI / 4, isLeft: true },
          { id: 'FR', x: 0.9, z: -0.9, rot: -Math.PI / 4, isLeft: false },
          { id: 'RL', x: -0.9, z: 0.9, rot: -Math.PI / 4, isLeft: true },
          { id: 'RR', x: 0.9, z: 0.9, rot: Math.PI / 4, isLeft: false },
        ].map((arm, i) => (
          <group key={arm.id}>
            {/* Arm Strut at 45 deg */}
            <mesh
              position={[arm.x * 0.5, 0.04, arm.z * 0.5]}
              rotation={[0, arm.rot, 0]}
              material={materials.arm}
            >
              <boxGeometry args={[0.14, 0.09, 1.4]} />
            </mesh>

            {/* Motor Pod at Arm Tip */}
            <mesh position={[arm.x, 0.14, arm.z]} material={materials.motor}>
              <cylinderGeometry args={[0.22, 0.22, 0.32, 16]} />
            </mesh>

            {/* Navigation LED: Red Port (Left), Green Starboard (Right) */}
            <mesh
              position={[arm.x * 1.08, -0.06, arm.z * 1.08]}
              material={arm.isLeft ? materials.navRed : materials.navGreen}
            >
              <sphereGeometry args={[0.07, 8, 8]} />
            </mesh>

            {/* ── 3. Horizontal Rotor Blades lying in XZ plane, spinning around local Y ── */}
            <group
              ref={(el) => {
                if (el) rotorsRef.current[i] = el
              }}
              position={[arm.x, 0.34, arm.z]}
            >
              {/* Twin Propeller Blades (Horizontal) */}
              <mesh material={materials.rotorBlade}>
                <boxGeometry args={[1.7, 0.02, 0.12]} />
              </mesh>
              <mesh rotation={[0, Math.PI / 2, 0]} material={materials.rotorBlade}>
                <boxGeometry args={[1.7, 0.02, 0.12]} />
              </mesh>

              {/* Horizontal Spinning Rotor Disc Halo (Rotated -PI/2 on X so it lies horizontal) */}
              <mesh rotation={[-Math.PI / 2, 0, 0]} material={materials.rotorDisc}>
                <ringGeometry args={[0.3, 0.88, 24]} />
              </mesh>
            </group>
          </group>
        ))}

        {/* White Tail Strobe Beacon */}
        <mesh position={[0, 0.18, 0.95]} material={materials.strobeWhite}>
          <sphereGeometry args={[0.08, 8, 8]} />
        </mesh>

        {/* ── 4. Subtle Downward Tactical Under-Glow Light ── */}
        <pointLight color="#3dd6c6" intensity={1.8} distance={8} position={[0, -0.5, 0]} />

        {/* ── 5. Dedicated Drone Rim Light (Ensures high contrast in all biomes) ── */}
        <directionalLight color="#e2f0fd" intensity={2.2} position={[0, 8, 10]} />
      </group>
    </group>
  )
}
