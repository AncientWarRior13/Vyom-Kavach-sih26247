import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
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
import { VyomKavachLogo } from './VyomKavachLogo'

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

  // Debug flag (?debug=1 in query param)
  const isDebug = useMemo(() => {
    if (typeof window === 'undefined') return false
    return new URLSearchParams(window.location.search).get('debug') === '1'
  }, [])

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

    // Support testing query parameters (?p=0.25, ?time=night, ?section=2)
    const params = new URLSearchParams(window.location.search)
    const pParam = params.get('p')
    const timeParam = params.get('time')
    const secParam = params.get('section')
    if (timeParam === 'day' || timeParam === 'dusk' || timeParam === 'night') {
      setTimeMode(timeParam)
    }
    if (secParam !== null) {
      const sVal = Math.max(0, Math.min(7, parseInt(secParam, 10)))
      const pVal = sVal === 0 ? 0 : sVal === 7 ? 0.98 : (sVal + 0.5) / 8
      flight.progress = pVal
      flight.damped = pVal
      setScrollProgress(pVal)
      setActiveSection(sVal)
    } else if (pParam) {
      const pVal = Math.max(0, Math.min(1, parseFloat(pParam)))
      flight.progress = pVal
      flight.damped = pVal
      setScrollProgress(pVal)
      const secIdx = Math.min(7, Math.floor(pVal * 8))
      setActiveSection(secIdx)
    }
    ;(window as any).__setSection = (idx: number) => {
      const sVal = Math.max(0, Math.min(7, idx))
      const pVal = sVal === 0 ? 0 : sVal === 7 ? 0.98 : (sVal + 0.5) / 8
      flight.progress = pVal
      flight.damped = pVal
      setScrollProgress(pVal)
      setActiveSection(sVal)
      const totalH = document.documentElement.scrollHeight - window.innerHeight
      if (lenisRef.current && totalH > 0) {
        lenisRef.current.scrollTo(pVal * totalH, { immediate: true })
      }
    }
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
    const targetY = idx === 0 ? 0 : idx === 7 ? totalH : ((idx + 0.5) / 8) * totalH
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
    } else {
      window.location.href = `/trainer?autostart=1`
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
    <div className="relative w-full bg-[#0b0e13] text-[#e6eaf0] select-none font-sans">
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

      {/* ── Fixed Tactical Top Bar (56px height, px-6, z-40) ── */}
      <header
        id="app-header"
        className="fixed top-0 left-0 right-0 z-40 h-[var(--header-h)] px-6 flex items-center justify-between pointer-events-auto bg-[#0b0e13]/85 border-b border-white/[0.08] backdrop-blur-md"
      >
        <div className="flex items-center gap-3">
          <VyomKavachLogo className="w-5 h-5 drop-shadow-[0_0_8px_rgba(0,229,255,0.45)] shrink-0" />
          <span className="font-['Chakra_Petch'] font-semibold text-sm tracking-wider text-[#f8fafc]">
            Vyom Kavach C-UAS
          </span>
          <span className="text-xs font-mono text-[#6b7585] hidden sm:inline">
            Tactical Vyom Kavach trainer
          </span>
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          {/* Server Status Dot */}
          <div className="flex items-center gap-2 text-xs text-[#9aa4b2]">
            <span className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-[#10b981]' : 'bg-[#ef4444]'}`} />
            <span className="hidden md:inline">{serverOnline ? 'Operational' : 'Offline'}</span>
          </div>

          {/* Dev Telemetry Toggle (?debug=1 only) */}
          {isDebug && (
            <button
              onClick={() => setShowDevTelemetry((v) => !v)}
              className="px-2 py-1 bg-[#181d26] border border-white/10 hover:border-[#3dd6c6] text-[11px] text-[#9aa4b2] rounded transition-colors cursor-pointer"
            >
              {showDevTelemetry ? 'Hide metrics' : 'Metrics'}
            </button>
          )}

          {/* Direct App Link */}
          <a
            href="/trainer"
            className="px-3.5 py-1.5 bg-[#3dd6c6] text-[#0b0e13] font-semibold text-xs rounded hover:bg-[#2bbfa0] transition-colors"
          >
            Launch trainer →
          </a>
        </div>
      </header>

      {/* ── Left Progress Rail: Sectors Box (Top-Left, ~24px below nav, ~24px from left, w-[220px], z-20) ── */}
      <div
        id="sectors-box"
        className="fixed top-[calc(var(--header-h)+24px)] left-6 z-20 flex flex-col gap-2 p-3 bg-[rgba(8,11,16,0.88)] border border-white/[0.08] rounded-lg backdrop-blur-md shadow-xl w-[220px] max-[900px]:top-[calc(var(--header-h)+16px)] max-[900px]:left-4 max-[900px]:right-4 max-[900px]:w-[calc(100%-32px)] pointer-events-auto"
      >
        <div className="text-[10px] font-mono uppercase tracking-wider text-[#6b7585] mb-0.5">
          Sectors
        </div>
        <div className="flex flex-col max-[900px]:flex-row max-[900px]:flex-wrap gap-2 max-[900px]:gap-3">
          {SECTIONS.map((sec, idx) => {
            const isActive = activeSection === idx
            return (
              <button
                key={sec.id}
                onClick={() => scrollToSection(idx)}
                className={`flex items-center gap-2.5 text-left transition-all cursor-pointer group select-none pointer-events-auto ${
                  isActive ? 'text-[#3dd6c6]' : 'text-[#6b7585] hover:text-[#e6eaf0]'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full transition-all shrink-0 ${
                    isActive
                      ? 'bg-[#3dd6c6] scale-125'
                      : 'bg-white/20 group-hover:bg-white/40'
                  }`}
                />
                {isActive && (
                  <span className="font-mono text-xs font-medium text-[#3dd6c6] leading-tight whitespace-normal max-w-[160px]">
                    {sec.eyebrow.replace(/^0\d · /, '')}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Section Panels Overlay Wrapper (Pointer-events: none, z-30) ── */}
      <div className="section-overlay-container fixed inset-0 pointer-events-none z-30">
        {SECTIONS.map((sec, idx) => {
          const secSpan = 1 / 8
          const secStart = idx * secSpan
          const localT = (scrollProgress - secStart) / secSpan

          const isVisible = localT >= -0.05 && localT <= 1.05
          if (!isVisible) return null

          // Fade in over first 20%, hold, fade out over last 20%
          let opacity = 0
          let translateY = 0
          if (idx === 0) {
            if (localT <= 0.6) {
              opacity = 1
              translateY = 0
            } else {
              const f = Math.max(0, (1 - localT) / 0.4)
              opacity = f
              translateY = -(1 - f) * 12
            }
          } else if (idx === 7) {
            if (localT < 0.3) {
              opacity = Math.max(0, localT / 0.3)
              translateY = (1 - opacity) * 12
            } else {
              opacity = 1
              translateY = 0
            }
          } else {
            if (localT < 0.2) {
              const f = Math.max(0, localT / 0.2)
              opacity = f
              translateY = (1 - f) * 12
            } else if (localT <= 0.8) {
              opacity = 1
              translateY = 0
            } else {
              const f = Math.max(0, (1 - localT) / 0.2)
              opacity = f
              translateY = -(1 - f) * 12
            }
          }

          // Positioning:
          // Left-side (including Hero): bottom-left (24px left, 72px bottom - shifted up 48px), max-w-[360px]
          // Right-side: right-6, bottom-[calc(var(--telemetry-h)+96px)] - shifted up 48px, max-w-[360px]
          // Center-side: bottom-[calc(var(--telemetry-h)+36px)], horizontally centered
          // Under 900px: stacked at bottom, full width with 16px margins
          let posClass = ''
          if (sec.side === 'left') {
            posClass = 'section-panel-left fixed bottom-[72px] left-6 max-w-[360px] max-[900px]:bottom-[64px] max-[900px]:left-4 max-[900px]:right-4 max-[900px]:w-[calc(100%-32px)] max-[900px]:max-w-[calc(100%-32px)]'
          } else if (sec.side === 'right') {
            posClass = 'section-panel-right fixed bottom-[calc(var(--telemetry-h)+96px)] right-6 max-w-[360px] max-[900px]:bottom-[calc(var(--telemetry-h)+72px)] max-[900px]:left-4 max-[900px]:right-4 max-[900px]:w-[calc(100%-32px)] max-[900px]:max-w-[calc(100%-32px)]'
          } else {
            posClass = 'section-panel-center fixed bottom-[calc(var(--telemetry-h)+36px)] left-1/2 max-w-[460px] max-[900px]:bottom-[calc(var(--telemetry-h)+24px)] max-[900px]:left-4 max-[900px]:right-4 max-[900px]:translate-x-0 max-[900px]:w-[calc(100%-32px)] max-[900px]:max-w-[calc(100%-32px)]'
          }

          const isActive = activeSection === idx

          return (
            <div
              key={sec.id}
              id={`section-panel-${sec.id}`}
              style={{
                opacity,
                transform: `translateY(${translateY}px)${sec.side === 'center' ? ' translateX(-50%)' : ''}`,
                transition: 'opacity 0.15s ease-out, transform 0.15s ease-out',
              }}
              className={`${posClass} ${
                isActive ? 'active-section-panel' : ''
              } section-panel w-full p-5 sm:p-6 bg-[rgba(8,11,16,0.88)] border border-white/[0.08] rounded-lg backdrop-blur-md shadow-2xl pointer-events-auto z-30`}
            >
              {/* Eyebrow: 12px mono in calm teal */}
              <div className="font-mono text-xs font-semibold uppercase tracking-wider text-[#3dd6c6] mb-2 pointer-events-none">
                {sec.eyebrow}
              </div>

              {/* Headline: clamp(28px, 3vw, 40px), text-wrap: balance, line-height 1.1 */}
              <h2 className="section-headline text-[clamp(28px,3vw,40px)] font-bold text-[#f8fafc] leading-[1.1] [text-wrap:balance] mb-3 pointer-events-none">
                {sec.title}
              </h2>

              {/* Body Text: 15-16px, line-height 1.5, in #C6CED8 */}
              {sec.id === 'hero' && (
                <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                  Next-generation counter-UAS training simulator. Practice high-pressure radar classification, sensor cross-cueing, and graduated countermeasures before live operational deployments.
                </p>
              )}

              {sec.id === 'detect' && (
                <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                  Multispectral radar sweeps hostile air corridors. Cross-reference micro-doppler blade modulations with transponder telemetry to isolate hostile incursions from friendly reconnaissance.
                </p>
              )}

              {sec.id === 'respond' && (
                <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                  Dynamic Rules of Engagement govern every strike. Enforce weapons-hold discipline, calculate collateral hazard radii, and deploy RF electronic disruption or kinetic interceptors.
                </p>
              )}

              {sec.id === 'swarm' && (
                <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                  Autonomous swarms adapt in real time. Severing leader nodes triggers divergence and saturation tactics that demand rapid multi-target prioritization.
                </p>
              )}

              {sec.id === 'sensors' && (
                <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                  Adversarial electronic warfare actively jams radar and thermal cameras. Cross-validate tracks using acoustic microphone arrays and frequency-hopping RF scanners.
                </p>
              )}

              {sec.id === 'review' && (
                <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                  Complete after-action debriefing. Replay full 3D engagement flights, diagnose classification vulnerabilities with polar heatmaps, and verify cryptographic audit chains.
                </p>
              )}

              {sec.id === 'modes' && (
                <div className="space-y-2.5 pt-2">
                  <div className="grid grid-cols-3 gap-2.5">
                    <a
                      href="/trainer"
                      className="p-3 bg-[#11151c] border border-white/[0.08] rounded-md hover:border-[#3dd6c6] hover:bg-[#151a23] transition-all block text-center pointer-events-auto group"
                    >
                      <div className="font-mono text-xs font-semibold text-[#f8fafc] group-hover:text-[#3dd6c6] transition-colors">
                        2D scope
                      </div>
                      <div className="text-[11px] text-[#9aa4b2] mt-0.5">Desktop C-UAS</div>
                    </a>
                    <a
                      href="/vr"
                      className="p-3 bg-[#11151c] border border-white/[0.08] rounded-md hover:border-[#3dd6c6] hover:bg-[#151a23] transition-all block text-center pointer-events-auto group"
                    >
                      <div className="font-mono text-xs font-semibold text-[#f8fafc] group-hover:text-[#3dd6c6] transition-colors">
                        VR dome
                      </div>
                      <div className="text-[11px] text-[#9aa4b2] mt-0.5">Spatial WebXR</div>
                    </a>
                    <a
                      href="/instructor"
                      className="p-3 bg-[#11151c] border border-white/[0.08] rounded-md hover:border-[#3dd6c6] hover:bg-[#151a23] transition-all block text-center pointer-events-auto group"
                    >
                      <div className="font-mono text-xs font-semibold text-[#f8fafc] group-hover:text-[#3dd6c6] transition-colors">
                        Command
                      </div>
                      <div className="text-[11px] text-[#9aa4b2] mt-0.5">Threat control</div>
                    </a>
                  </div>

                  <div className="flex items-center justify-between px-0.5 text-xs font-mono border-t border-white/[0.06] pt-2">
                    <span className="text-[#3dd6c6] font-semibold text-[11px] flex items-center gap-1.5 shrink-0">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#3dd6c6] animate-pulse" />
                      +3 more:
                    </span>
                    <div className="flex items-center gap-1.5 text-[11px]">
                      <a
                        href="/team"
                        title="Multi-operator distributed crew C2"
                        className="px-2 py-0.5 bg-[#11151c] border border-white/[0.08] hover:border-[#3dd6c6] hover:text-[#3dd6c6] text-[#9aa4b2] rounded transition-all pointer-events-auto whitespace-nowrap"
                      >
                        Team C2
                      </a>
                      <a
                        href="/swarm"
                        title="Swarm behavior & saturation tactics sandbox"
                        className="px-2 py-0.5 bg-[#11151c] border border-white/[0.08] hover:border-[#3dd6c6] hover:text-[#3dd6c6] text-[#9aa4b2] rounded transition-all pointer-events-auto whitespace-nowrap"
                      >
                        Swarm Lab
                      </a>
                      <a
                        href="/aar"
                        title="After-action review & forensic playback"
                        className="px-2 py-0.5 bg-[#11151c] border border-white/[0.08] hover:border-[#3dd6c6] hover:text-[#3dd6c6] text-[#9aa4b2] rounded transition-all pointer-events-auto whitespace-nowrap"
                      >
                        AAR Replay
                      </a>
                    </div>
                  </div>
                </div>
              )}

              {sec.id === 'land' && (
                <div className="space-y-3.5 pt-1">
                  <p className="section-body text-[15px] sm:text-base leading-[1.5] text-[#C6CED8] pointer-events-none max-h-[720px]:text-[14px]">
                    Configure scenario parameters to launch a live air defense drill or load a validated scripted exercise.
                  </p>

                  {/* Difficulty slider */}
                  <div className="space-y-1 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-[#9aa4b2]">Threat ingress difficulty:</span>
                      <span className="text-[#3dd6c6] font-semibold">Tier 0{difficulty}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={5}
                      value={difficulty}
                      onChange={(e) => setDifficulty(Number(e.target.value))}
                      className="w-full accent-[#3dd6c6] cursor-pointer pointer-events-auto"
                    />
                  </div>

                  {/* Time of Day selection */}
                  <div className="flex items-center gap-2 text-xs font-mono">
                    <span className="text-[#9aa4b2]">Time of day:</span>
                    <div className="flex gap-1.5">
                      {(['day', 'dusk', 'night'] as const).map((t) => (
                        <button
                          key={t}
                          onClick={() => setTimeMode(t)}
                          className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer pointer-events-auto ${
                            timeMode === t
                              ? 'bg-[#3dd6c6] text-[#0b0e13] font-semibold'
                              : 'bg-[#181d26] text-[#9aa4b2] hover:text-[#e6eaf0]'
                          }`}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Launch Action Buttons */}
                  <div className="flex flex-col sm:flex-row gap-3 pt-1">
                    <button
                      onClick={launchCustom}
                      className="flex-1 py-2.5 px-4 bg-[#3dd6c6] text-[#0b0e13] font-semibold text-xs rounded-md hover:bg-[#2bbfa0] transition-colors shadow-lg cursor-pointer text-center pointer-events-auto"
                    >
                      Start training drill →
                    </button>
                    <button
                      onClick={launchScripted}
                      className="py-2.5 px-4 bg-[#181d26] border border-white/[0.08] text-[#e6eaf0] text-xs font-semibold rounded-md hover:border-[#3dd6c6] transition-colors cursor-pointer text-center pointer-events-auto"
                    >
                      Load scripted drill
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* ── Right HUD Telemetry Panel (Bottom Right, z-20) ── */}
      <div
        id="telemetry-card"
        className="fixed right-[var(--gutter)] bottom-[var(--gutter)] z-20 hidden md:flex flex-col justify-between p-3.5 bg-[rgba(8,11,16,0.88)] border border-white/[0.08] rounded-lg backdrop-blur-md shadow-xl text-xs font-mono w-[var(--telemetry-w)] h-[var(--telemetry-h)] pointer-events-auto"
      >
        <div className="text-[10px] uppercase tracking-wider text-[#6b7585] mb-0.5">
          Sector telemetry
        </div>
        <div className="space-y-1.5 text-xs">
          <div className="flex justify-between">
            <span className="text-[#9aa4b2]">Biome:</span>
            <span className="text-[#3dd6c6] font-semibold text-right">{currentBiome}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[#9aa4b2]">Range:</span>
            <span className="tabular-nums font-semibold text-[#f8fafc]">{Math.round(scrollProgress * 1600)}m</span>
          </div>
          <div className="flex justify-between">
            <span className="text-[#9aa4b2]">Airspeed:</span>
            <span className="tabular-nums font-semibold text-[#f8fafc]">{(20 + flight.speed * 14).toFixed(1)} m/s</span>
          </div>
        </div>
      </div>

      {/* ── Flight Stability Metrics Overlay (?debug=1 Only) ── */}
      {isDebug && showDevTelemetry && (
        <div className="fixed bottom-6 left-6 z-30 p-3 bg-[#0b0e13]/90 border border-white/[0.08] rounded-lg font-mono text-[11px] text-[#9aa4b2] pointer-events-auto shadow-2xl space-y-1">
          <div className="text-[10px] font-bold text-[#3dd6c6] uppercase tracking-wider mb-1 flex items-center justify-between">
            <span>Flight stability</span>
            <span className={`w-2 h-2 rounded-full ${telemetry.deltaSpike ? 'bg-[#ef4444]' : 'bg-[#10b981]'}`} />
          </div>
          <div>FPS: <strong className={telemetry.fps < 45 ? 'text-[#f59e0b]' : 'text-[#10b981]'}>{telemetry.fps}</strong></div>
          <div>DELTA: <span className="tabular-nums">{telemetry.delta}ms</span> {telemetry.deltaSpike && <span className="text-[#ef4444] font-bold">(SPIKE)</span>}</div>
          <div>ROT-DELTA: <span className="tabular-nums">{telemetry.rotDelta}°/frame</span></div>
          <div>DRONE-Z: <span className="tabular-nums">{Math.round(telemetry.pos.z)}</span></div>
        </div>
      )}

      {/* ── Scroll Guide Hint (Bottom Center) ── */}
      {!hasScrolled && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 pointer-events-none font-mono text-xs text-[#3dd6c6] tracking-wider uppercase animate-bounce text-center">
          ▼ Scroll down to fly quadcopter through sectors
        </div>
      )}

      {/* ── 800vh Virtual Scroll Track for Lenis Smooth Scrolling ── */}
      <div className="relative w-full h-[800vh] pointer-events-none" />
    </div>
  )
}
