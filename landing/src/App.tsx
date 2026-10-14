import { useState, useEffect, useRef, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import Lenis from 'lenis'
import Atmosphere from './world/Atmosphere'
import Terrain from './world/Terrain'
import Props from './world/Props'
import Drone from './world/Drone'
import { flight } from './lib/flightState'
import { SECTIONS } from './lib/sections'
import { BIOME_RANGES, WORLD_LENGTH } from './world/pathCore'

export default function App() {
  const [scrollProgress, setScrollProgress] = useState(0)
  const [activeSection, setActiveSection] = useState(0)
  const [timeMode, setTimeMode] = useState<'day' | 'dusk' | 'night'>('dusk')
  const [difficulty, setDifficulty] = useState(3)
  const [selectedDrill, setSelectedDrill] = useState('')
  const [scriptedDrills, setScriptedDrills] = useState<string[]>([])
  const [serverOnline, setServerOnline] = useState(true)
  const [hasScrolled, setHasScrolled] = useState(false)
  const [showDevTelemetry, setShowDevTelemetry] = useState(true)

  // Dev Telemetry Data
  const [telemetry, setTelemetry] = useState<{
    fps: number
    delta: number
    deltaSpike: boolean
    pos: THREE.Vector3
    rotDelta: number
  }>({
    fps: 60,
    delta: 16,
    deltaSpike: false,
    pos: new THREE.Vector3(),
    rotDelta: 0,
  })

  const lenisRef = useRef<Lenis | null>(null)

  // Fetch scripted scenarios & health
  useEffect(() => {
    fetch('/api/scenario/scripted')
      .then((r) => r.json())
      .then((data) => {
        if (data && data.scenarios) {
          setScriptedDrills(data.scenarios)
          if (data.scenarios.length > 0) setSelectedDrill(data.scenarios[0])
        }
      })
      .catch(() => {})

    fetch('/api/model/status')
      .then((r) => setServerOnline(r.ok))
      .catch(() => setServerOnline(false))
  }, [])

  // Initialize Lenis smooth scroll and connect to flightState
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    })
    lenisRef.current = lenis

    let rafId: number
    const onRaf = (time: number) => {
      lenis.raf(time)
      rafId = requestAnimationFrame(onRaf)
    }
    rafId = requestAnimationFrame(onRaf)

    const onScroll = () => {
      const totalH = document.documentElement.scrollHeight - window.innerHeight
      if (totalH <= 0) return
      const p = Math.max(0, Math.min(1, window.scrollY / totalH))

      // Direct write to flightState so useFrame reads it without React lag
      flight.progress = p
      const deltaP = Math.abs(p - flight.damped)
      flight.speed = Math.min(3.5, deltaP * 40)

      setScrollProgress(p)
      if (p > 0.015) setHasScrolled(true)

      const secIdx = Math.min(7, Math.floor(p * 8))
      setActiveSection(secIdx)
    }

    lenis.on('scroll', onScroll)

    return () => {
      cancelAnimationFrame(rafId)
      lenis.destroy()
    }
  }, [])

  // Click on a dot to scroll to section
  const scrollToSection = (idx: number) => {
    const totalH = document.documentElement.scrollHeight - window.innerHeight
    const targetY = (idx / 7) * totalH
    if (lenisRef.current) {
      lenisRef.current.scrollTo(targetY, { duration: 1.4 })
    } else {
      window.scrollTo({ top: targetY, behavior: 'smooth' })
    }
  }

  // Determine current biome based on drone Z
  const zPos = -scrollProgress * WORLD_LENGTH
  let currentBiome = 'URBAN SECTOR'
  if (zPos < BIOME_RANGES.rural[0]) currentBiome = 'RURAL FARMLAND'
  else if (zPos < BIOME_RANGES.mountain[0]) currentBiome = 'MOUNTAIN PASS'
  else if (zPos < BIOME_RANGES.coastal[0]) currentBiome = 'COASTAL DEFENSE'

  const launchCustom = () => {
    window.location.href = `/trainer?difficulty=${difficulty}&timeOfDay=${timeMode}&autostart=1`
  }

  const launchScripted = () => {
    if (selectedDrill) {
      window.location.href = `/trainer?drill=${encodeURIComponent(selectedDrill)}&autostart=1`
    }
  }

  // Handle telemetry update from Drone
  const handleTelemetry = useCallback((data: {
    fps: number
    delta: number
    deltaSpike: boolean
    pos: THREE.Vector3
    rotDelta: number
  }) => {
    setTelemetry(data)
  }, [])

  return (
    <div className="relative w-full bg-[#020508] text-[#c8deef] select-none font-sans">
      {/* ── Fixed 3D Canvas Scene ── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <Canvas
          camera={{ position: [0, 20, 30], fov: 50, near: 0.5, far: 1800 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          dpr={[1, 1.5]}
        >
          <Drone onTelemetryUpdate={handleTelemetry} />
          <Atmosphere />
          <Terrain />
          <Props />
        </Canvas>
      </div>

      {/* ── Fixed Tactical HUD Header (Top) ── */}
      <header className="fixed top-0 left-0 right-0 z-30 px-6 py-4 flex items-center justify-between pointer-events-auto bg-gradient-to-b from-[#020508]/80 to-transparent">
        <div className="flex items-center gap-3">
          <span className="w-2.5 h-2.5 bg-[#00e87a] rounded-xs shadow-[0_0_8px_#00e87a]" />
          <span className="font-['Chakra_Petch'] font-bold text-base tracking-widest text-[#f0f8ff]">
            PS-247 C-UAS
          </span>
          <span className="text-xs font-mono text-[#8aaabf] hidden sm:inline">
            // TACTICAL DRONE THREAT TRAINER
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          {/* Server Status LED */}
          <div className="flex items-center gap-2 px-2.5 py-1 bg-[#05070a]/80 border border-[#1e3c64] rounded-xs backdrop-blur-xs">
            <span className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-[#00e87a] shadow-[0_0_6px_#00e87a]' : 'bg-[#ff3b3b]'}`} />
            <span className="text-[#8aaabf] hidden md:inline">SYSTEM: {serverOnline ? 'ONLINE' : 'OFFLINE'}</span>
          </div>

          {/* Dev Telemetry Toggle */}
          <button
            onClick={() => setShowDevTelemetry((v) => !v)}
            className="px-2 py-1 bg-[#05070a]/80 border border-[#1e3c64] hover:border-[#00e87a] text-[11px] text-[#8aaabf] rounded-xs transition-colors cursor-pointer"
          >
            {showDevTelemetry ? 'HIDE METRICS' : 'DEV METRICS'}
          </button>

          {/* Direct App Link */}
          <a
            href="/trainer"
            className="px-3.5 py-1.5 bg-[#00e87a] text-[#020508] font-bold rounded-xs shadow-[0_0_14px_rgba(0,232,122,0.35)] hover:bg-[#00c968] transition-colors uppercase tracking-wider"
          >
            Launch Trainer →
          </a>
        </div>
      </header>

      {/* ── Fixed Full-Screen CSS Grid Overlay [180px] [content] [200px] ── */}
      <div className="fixed inset-0 z-20 pointer-events-none grid grid-cols-1 md:grid-cols-[180px_1fr_200px] items-center px-4 md:px-8">
        {/* Left Column: Collapsed 8-Dot Progress Rail */}
        <div className="hidden md:flex flex-col gap-3 pointer-events-auto p-4 bg-[rgba(5,7,10,0.72)] border border-[#1e3c64] rounded-xs backdrop-blur-md shadow-xl w-[170px]">
          <div className="text-[10px] font-mono uppercase tracking-widest text-[#4e718a] mb-1">
            FLIGHT SECTORS
          </div>
          <div className="flex flex-col gap-2.5">
            {SECTIONS.map((sec, idx) => {
              const isActive = activeSection === idx
              return (
                <button
                  key={sec.id}
                  onClick={() => scrollToSection(idx)}
                  className={`flex items-center gap-2.5 text-left transition-all cursor-pointer group select-none ${
                    isActive ? 'text-[#00e87a]' : 'text-[#4e718a] hover:text-[#c8deef]'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full transition-all ${
                      isActive
                        ? 'bg-[#00e87a] scale-125 shadow-[0_0_8px_#00e87a]'
                        : 'bg-[#1e3c64] group-hover:bg-[#4e718a]'
                    }`}
                  />
                  {isActive && (
                    <span className="font-mono text-xs font-bold uppercase tracking-wider text-[#00e87a] truncate animate-in fade-in duration-200">
                      {sec.eyebrow.replace(/^0\d · /, '')}
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>

        {/* Center Column: ONE Active Story Section at a Time */}
        <div className="w-full flex items-center justify-center relative min-h-[420px] pointer-events-none">
          {SECTIONS.map((sec, idx) => {
            // Local section progress (0 to 1 over its 1/8 span)
            const secSpan = 1 / 8
            const secStart = idx * secSpan
            const localT = (scrollProgress - secStart) / secSpan

            // Visibility window: visible only when localT is between -0.05 and 1.05
            const isVisible = localT >= -0.05 && localT <= 1.05

            if (!isVisible) return null

            // Fade in over first 20%, hold, fade out over last 20%
            let opacity = 0
            let translateY = 0
            if (localT < 0.2) {
              const f = Math.max(0, localT / 0.2)
              opacity = f
              translateY = (1 - f) * 16
            } else if (localT <= 0.8) {
              opacity = 1
              translateY = 0
            } else {
              const f = Math.max(0, (1 - localT) / 0.2)
              opacity = f
              translateY = -(1 - f) * 16
            }

            const alignClass =
              sec.side === 'center'
                ? 'justify-self-center text-center'
                : sec.side === 'right'
                ? 'justify-self-end text-left'
                : 'justify-self-start text-left'

            return (
              <div
                key={sec.id}
                style={{
                  opacity,
                  transform: `translateY(${translateY}px)`,
                  transition: 'opacity 0.15s ease-out, transform 0.15s ease-out',
                }}
                className={`absolute w-full max-w-[560px] p-6 bg-[rgba(5,7,10,0.78)] backdrop-blur-md rounded-xs border border-[#1e3c64] shadow-2xl space-y-4 pointer-events-auto ${alignClass}`}
              >
                {/* Eyebrow: at least 12px mono with letter-spacing */}
                <div className="font-mono text-xs font-bold uppercase tracking-widest text-[#00e87a]">
                  {sec.eyebrow}
                </div>

                {/* Headline: clamp(32px, 4.5vw, 64px), text-wrap: balance */}
                <h2 className="font-['Chakra_Petch'] text-[clamp(28px,4.2vw,54px)] font-extrabold uppercase tracking-wide text-[#f0f8ff] leading-[1.1] [text-wrap:balance]">
                  {sec.title}
                </h2>

                {/* Body Text: at least 16px with line-height 1.5, text contrast > 7:1 */}
                {sec.id === 'hero' && (
                  <p className="text-base leading-relaxed text-[#c8deef]">
                    Next-generation counter-UAS training simulator. Practice high-pressure radar classification, sensor cross-cueing, and graduated countermeasures before live operational deployments.
                  </p>
                )}

                {sec.id === 'detect' && (
                  <p className="text-base leading-relaxed text-[#c8deef]">
                    Multispectral radar sweeps hostile air corridors. Cross-reference micro-doppler blade modulations with transponder telemetry to isolate hostile kamikaze incursions from friendly recon.
                  </p>
                )}

                {sec.id === 'respond' && (
                  <p className="text-base leading-relaxed text-[#c8deef]">
                    Dynamic Rules of Engagement govern every strike. Enforce weapons-hold discipline, calculate collateral hazard radii, and deploy RF electronic disruption or kinetic interceptors.
                  </p>
                )}

                {sec.id === 'swarm' && (
                  <p className="text-base leading-relaxed text-[#c8deef]">
                    Hostile autonomous swarms adapt in real time. Severing leader nodes triggers divergence and saturation tactics that demand rapid multi-target prioritization.
                  </p>
                )}

                {sec.id === 'sensors' && (
                  <p className="text-base leading-relaxed text-[#c8deef]">
                    Adversarial electronic warfare actively jams radar and thermal cameras. Cross-validate tracks using acoustic microphone arrays and frequency-hopping RF scanners.
                  </p>
                )}

                {sec.id === 'review' && (
                  <p className="text-base leading-relaxed text-[#c8deef]">
                    Complete after-action debriefing. Replay full 3D engagement flights, diagnose classification vulnerabilities with polar heatmaps, and verify cryptographic audit chains.
                  </p>
                )}

                {sec.id === 'modes' && (
                  <div className="grid grid-cols-3 gap-3 pt-2">
                    <a href="/trainer" className="p-3 bg-[#08121e] border border-[#1e3c64] rounded hover:border-[#00e87a] transition-colors block text-center">
                      <div className="font-mono text-xs font-bold text-[#f0f8ff]">2D SCOPE</div>
                      <div className="text-[11px] text-[#8aaabf] mt-0.5">Desktop C-UAS</div>
                    </a>
                    <a href="/vr" className="p-3 bg-[#08121e] border border-[#1e3c64] rounded hover:border-[#00e87a] transition-colors block text-center">
                      <div className="font-mono text-xs font-bold text-[#f0f8ff]">VR DOME</div>
                      <div className="text-[11px] text-[#8aaabf] mt-0.5">Spatial WebXR</div>
                    </a>
                    <a href="/instructor" className="p-3 bg-[#08121e] border border-[#1e3c64] rounded hover:border-[#00e87a] transition-colors block text-center">
                      <div className="font-mono text-xs font-bold text-[#f0f8ff]">COMMAND</div>
                      <div className="text-[11px] text-[#8aaabf] mt-0.5">Threat Control</div>
                    </a>
                  </div>
                )}

                {sec.id === 'land' && (
                  <div className="space-y-4 pt-2">
                    <div className="space-y-2 text-left font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-[#8aaabf]">THREAT INGRESS LEVEL:</span>
                        <span className="text-[#00e87a] font-bold">TIER 0{difficulty}</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={5}
                        value={difficulty}
                        onChange={(e) => setDifficulty(Number(e.target.value))}
                        className="w-full accent-[#00e87a] cursor-pointer"
                      />
                    </div>

                    <div className="flex flex-col sm:flex-row gap-3">
                      <button
                        onClick={launchCustom}
                        className="flex-1 py-3 px-4 bg-[#00e87a] text-[#020508] font-mono font-bold text-xs rounded uppercase tracking-wider hover:bg-[#00c968] transition-colors shadow-lg cursor-pointer"
                      >
                        Start Training Drill →
                      </button>
                      <button
                        onClick={launchScripted}
                        className="py-3 px-4 bg-[#08121e] border border-[#1e3c64] text-[#f0f8ff] font-mono text-xs rounded hover:border-[#00e87a] transition-colors cursor-pointer"
                      >
                        Launch Scripted Drill
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Right Column: HUD Telemetry Panel (Bottom Right) */}
        <div className="hidden md:flex flex-col gap-2 pointer-events-auto p-4 bg-[rgba(5,7,10,0.72)] border border-[#1e3c64] rounded-xs backdrop-blur-md shadow-xl text-xs font-mono w-[190px] justify-self-end">
          <div className="text-[10px] uppercase tracking-widest text-[#4e718a] mb-1">
            SECTOR TELEMETRY
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-[#4e718a]">BIOME:</span>
              <span className="text-[#00e87a] font-bold text-right">{currentBiome}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#4e718a]">RANGE:</span>
              <span className="tabular-nums font-bold text-[#f0f8ff]">{Math.round(scrollProgress * 1600)}M</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#4e718a]">AIRSPEED:</span>
              <span className="tabular-nums font-bold text-[#f0f8ff]">{(20 + flight.speed * 14).toFixed(1)} M/S</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Dev-Only Stability & Jitter Metrics Overlay (Top Left) ── */}
      {showDevTelemetry && (
        <div className="fixed top-16 left-6 z-30 p-3 bg-[#020508]/85 border border-[#1e3c64] rounded-xs font-mono text-[11px] text-[#8aaabf] pointer-events-auto shadow-2xl space-y-1">
          <div className="text-[10px] font-bold text-[#00e87a] uppercase tracking-wider mb-1 flex items-center justify-between">
            <span>FLIGHT STABILITY METRICS</span>
            <span className={`w-2 h-2 rounded-full ${telemetry.deltaSpike ? 'bg-[#ff3b3b]' : 'bg-[#00e87a]'}`} />
          </div>
          <div>FPS: <strong className={telemetry.fps < 45 ? 'text-[#ffb800]' : 'text-[#00e87a]'}>{telemetry.fps}</strong></div>
          <div>DELTA: <span className="tabular-nums">{telemetry.delta}ms</span> {telemetry.deltaSpike && <span className="text-[#ff3b3b] font-bold">(SPIKE)</span>}</div>
          <div>ROT-DELTA: <span className="tabular-nums">{telemetry.rotDelta}°/frame</span></div>
          <div>DRONE-Z: <span className="tabular-nums">{Math.round(telemetry.pos.z)}</span></div>
        </div>
      )}

      {/* ── Scroll Guide Hint (Bottom Center) ── */}
      {!hasScrolled && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 pointer-events-none font-mono text-xs text-[#00e87a] tracking-widest uppercase animate-bounce text-center">
          ▼ SCROLL DOWN TO PILOT QUADCOPTER THROUGH ALL 8 BIOMES
        </div>
      )}

      {/* ── 800vh Virtual Scroll Track for Lenis Smooth Scrolling ── */}
      <div className="relative w-full h-[800vh] pointer-events-none" />
    </div>
  )
}
