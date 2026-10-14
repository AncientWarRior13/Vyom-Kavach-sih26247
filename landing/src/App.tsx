import { useState, useEffect, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import Atmosphere from './world/Atmosphere'
import Terrain from './world/Terrain'
import Props from './world/Props'
import { flight } from './lib/flightState'

function RotatingCamera({ mode }: { mode: 'day' | 'dusk' | 'night' }) {
  const targetP = mode === 'day' ? 0.05 : mode === 'dusk' ? 0.38 : 0.75
  const angleRef = useRef(0)

  useFrame((state, delta) => {
    // Smoothly animate flight progress to target time-of-day
    flight.damped += (targetP - flight.damped) * Math.min(1, delta * 3)
    flight.progress = flight.damped

    // Subtle gentle camera orbit around city
    angleRef.current += delta * 0.06
    const r = 70
    const x = Math.sin(angleRef.current) * r
    const z = Math.cos(angleRef.current) * r
    const y = 35 + Math.sin(angleRef.current * 0.5) * 8

    state.camera.position.set(x, y, z)
    state.camera.lookAt(0, 10, 0)
  })

  return null
}

export default function App() {
  const [timeMode, setTimeMode] = useState<'day' | 'dusk' | 'night'>('dusk')
  const [selectedDrill, setSelectedDrill] = useState('dusk_rural_swarm_raid')
  const [difficulty, setDifficulty] = useState(3)
  const [serverOnline, setServerOnline] = useState(true)

  useEffect(() => {
    // Initial flight state for dusk mode
    flight.damped = 0.38
    flight.progress = 0.38

    // Check backend health
    fetch('/api/model/status')
      .then((r) => setServerOnline(r.ok))
      .catch(() => setServerOnline(false))
  }, [])

  const handleLaunchPreset = (drill: string) => {
    window.location.href = `/trainer?drill=${encodeURIComponent(drill)}`
  }

  const handleLaunchCustom = () => {
    window.location.href = `/trainer?difficulty=${difficulty}&timeOfDay=${timeMode}&autostart=1`
  }

  return (
    <div style={{ position: 'relative', width: '100%', minHeight: '100vh', background: '#05070A', color: '#D7DDE3', fontFamily: 'var(--font-ui)', overflowX: 'hidden' }}>
      {/* 3D Background Canvas */}
      <div style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}>
        <Canvas
          camera={{ position: [0, 40, 70], fov: 50, near: 0.5, far: 1600 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          dpr={[1, 1.5]}
        >
          <RotatingCamera mode={timeMode} />
          <Atmosphere />
          <Terrain />
          <Props />
        </Canvas>
      </div>

      {/* Subtle Tactical Grid & Radial Vignette Overlay */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 1,
          pointerEvents: 'none',
          background: 'radial-gradient(ellipse at 50% 15%, rgba(5,7,10,0.55) 0%, rgba(5,7,10,0.85) 65%, rgba(5,7,10,0.98) 100%)',
          backgroundImage: 'linear-gradient(to right, rgba(22, 34, 43, 0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(22, 34, 43, 0.15) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      {/* Main Interactive UI Layer */}
      <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        
        {/* Top Tactical Navigation Header */}
        <header
          style={{
            position: 'sticky',
            top: 0,
            backdropFilter: 'blur(16px)',
            background: 'rgba(7, 11, 16, 0.82)',
            borderBottom: '1px solid rgba(22, 34, 43, 0.8)',
            padding: '14px 28px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 16,
            flexWrap: 'wrap',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            {/* Tactical Radar Hexagon Icon */}
            <div style={{
              width: 38,
              height: 38,
              borderRadius: 8,
              background: 'linear-gradient(135deg, rgba(77, 255, 168, 0.15), rgba(77, 216, 255, 0.25))',
              border: '1px solid rgba(77, 255, 168, 0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4DFFA8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 2a10 10 0 0 1 10 10"/>
                <path d="m12 12 5-5"/>
                <circle cx="12" cy="12" r="3"/>
              </svg>
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '17px', letterSpacing: '0.04em', color: '#FFF' }}>
                DRONE THREAT TRAINER
              </div>
              <div style={{ fontSize: '11px', color: '#6B7A86', letterSpacing: '0.08em', textTransform: 'uppercase', fontFamily: 'var(--font-data)' }}>
                DEFENSE C-UAS SIMULATOR // PS-247
              </div>
            </div>
          </div>

          {/* Quick Nav Links */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <a href="/trainer" style={{ color: '#D7DDE3', textDecoration: 'none', padding: '6px 12px', fontSize: '13px', borderRadius: 6, transition: 'all 0.2s', border: '1px solid transparent' }} onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(77,255,168,0.3)'} onMouseLeave={e => e.currentTarget.style.borderColor = 'transparent'}>
              2D Scope
            </a>
            <a href="/vr" style={{ color: '#D7DDE3', textDecoration: 'none', padding: '6px 12px', fontSize: '13px', borderRadius: 6, transition: 'all 0.2s', border: '1px solid transparent' }} onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(77,216,255,0.3)'} onMouseLeave={e => e.currentTarget.style.borderColor = 'transparent'}>
              3D VR Dome
            </a>
            <a href="/instructor" style={{ color: '#D7DDE3', textDecoration: 'none', padding: '6px 12px', fontSize: '13px', borderRadius: 6, transition: 'all 0.2s', border: '1px solid transparent' }} onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(255,176,32,0.3)'} onMouseLeave={e => e.currentTarget.style.borderColor = 'transparent'}>
              Instructor
            </a>
            <a href="/team" style={{ color: '#D7DDE3', textDecoration: 'none', padding: '6px 12px', fontSize: '13px', borderRadius: 6, transition: 'all 0.2s', border: '1px solid transparent' }} onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(180,140,255,0.3)'} onMouseLeave={e => e.currentTarget.style.borderColor = 'transparent'}>
              Team Roles
            </a>
            <a href="/aar" style={{ color: '#D7DDE3', textDecoration: 'none', padding: '6px 12px', fontSize: '13px', borderRadius: 6, transition: 'all 0.2s', border: '1px solid transparent' }} onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(77,255,168,0.3)'} onMouseLeave={e => e.currentTarget.style.borderColor = 'transparent'}>
              AAR Analytics
            </a>
          </nav>

          {/* Action Header Button & Health Pill */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', borderRadius: 20, background: 'rgba(11, 16, 21, 0.9)', border: '1px solid #16222B', fontSize: '11px', fontFamily: 'var(--font-data)' }}>
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: serverOnline ? '#4DFFA8' : '#FF5A5A', boxShadow: serverOnline ? '0 0 8px #4DFFA8' : 'none' }}></span>
              <span style={{ color: serverOnline ? '#4DFFA8' : '#FF5A5A' }}>{serverOnline ? 'SERVER ONLINE' : 'DISCONNECTED'}</span>
            </div>
            <a
              href="/trainer?autostart=1"
              style={{
                textDecoration: 'none',
                background: 'linear-gradient(135deg, #4DFFA8, #36c782)',
                color: '#05070A',
                fontWeight: 700,
                fontSize: '13px',
                padding: '8px 18px',
                borderRadius: 6,
                letterSpacing: '0.02em',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                boxShadow: '0 0 20px rgba(77, 255, 168, 0.25)',
                transition: 'all 0.2s',
              }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-1px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'none'}
            >
              LAUNCH SIMULATOR &rarr;
            </a>
          </div>
        </header>

        {/* Hero Section */}
        <div style={{ maxWidth: 1200, width: '100%', margin: '0 auto', padding: '50px 24px 30px', display: 'flex', flexDirection: 'column', gap: 24 }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 24 }}>
            <div style={{ flex: '1 1 600px', maxWidth: 760 }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '6px 14px',
                borderRadius: 20,
                background: 'rgba(77, 255, 168, 0.08)',
                border: '1px solid rgba(77, 255, 168, 0.25)',
                color: '#4DFFA8',
                fontSize: '12px',
                fontFamily: 'var(--font-data)',
                letterSpacing: '0.06em',
                marginBottom: 16,
              }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#4DFFA8' }}></span>
                AUTONOMOUS SWARM & MULTI-SENSOR C-UAS SIMULATOR
              </div>

              <h1 style={{
                fontFamily: 'var(--font-display)',
                fontSize: 'clamp(32px, 5vw, 56px)',
                lineHeight: 1.1,
                fontWeight: 700,
                color: '#FFFFFF',
                letterSpacing: '-0.02em',
                marginBottom: 16,
              }}>
                Train The Critical Air Defense Call <span style={{ color: '#4DFFA8' }}>Before It Counts.</span>
              </h1>

              <p style={{
                fontSize: '17px',
                lineHeight: 1.6,
                color: '#8A99A6',
                maxWidth: 680,
                marginBottom: 28,
              }}>
                A desktop- and VR-capable drone & swarm threat recognition training simulator.
                Features real <strong style={{ color: '#D7DDE3' }}>77GHz FMCW CA-CFAR radar DSP</strong>, multi-sensor interrogation (EO/IR, RF, Acoustic),
                autonomous swarm divergence, and cryptographic After-Action Review audit chains.
              </p>

              {/* Primary Launch Action Buttons */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
                <a
                  href="/trainer?autostart=1"
                  style={{
                    textDecoration: 'none',
                    background: '#4DFFA8',
                    color: '#05070A',
                    fontFamily: 'var(--font-display)',
                    fontWeight: 700,
                    fontSize: '15px',
                    padding: '12px 26px',
                    borderRadius: 8,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 8,
                    boxShadow: '0 0 25px rgba(77, 255, 168, 0.35)',
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                  onMouseLeave={e => e.currentTarget.style.transform = 'none'}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  ENTER 2D SCOPE
                </a>

                <a
                  href="/vr"
                  style={{
                    textDecoration: 'none',
                    background: 'rgba(77, 216, 255, 0.12)',
                    color: '#4DD8FF',
                    border: '1px solid rgba(77, 216, 255, 0.35)',
                    fontFamily: 'var(--font-display)',
                    fontWeight: 600,
                    fontSize: '15px',
                    padding: '12px 22px',
                    borderRadius: 8,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 8,
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'rgba(77, 216, 255, 0.2)'; e.currentTarget.style.transform = 'translateY(-2px)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'rgba(77, 216, 255, 0.12)'; e.currentTarget.style.transform = 'none' }}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
                  LAUNCH 3D VR DOME
                </a>

                <a
                  href="/aar"
                  style={{
                    textDecoration: 'none',
                    background: 'rgba(11, 16, 21, 0.85)',
                    color: '#D7DDE3',
                    border: '1px solid #16222B',
                    fontFamily: 'var(--font-display)',
                    fontWeight: 600,
                    fontSize: '15px',
                    padding: '12px 20px',
                    borderRadius: 8,
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 8,
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = '#4DFFA8'; e.currentTarget.style.transform = 'translateY(-2px)' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = '#16222B'; e.currentTarget.style.transform = 'none' }}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                  AAR DASHBOARD
                </a>
              </div>
            </div>

            {/* Quick Deploy Mission Card */}
            <div style={{
              flex: '1 1 340px',
              maxWidth: 380,
              background: 'rgba(11, 16, 21, 0.88)',
              border: '1px solid #16222B',
              borderRadius: 12,
              padding: 22,
              backdropFilter: 'blur(20px)',
              boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <span style={{ fontSize: '11px', fontFamily: 'var(--font-data)', color: '#4DFFA8', letterSpacing: '0.08em' }}>MISSION DEPLOYMENT</span>
                <span style={{ fontSize: '11px', fontFamily: 'var(--font-data)', color: '#6B7A86' }}>INSTANT LAUNCH</span>
              </div>

              {/* Scripted Drills Selector */}
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', fontSize: '12px', color: '#8A99A6', marginBottom: 6 }}>Select Scripted Drill</label>
                <select
                  value={selectedDrill}
                  onChange={e => setSelectedDrill(e.target.value)}
                  style={{
                    width: '100%',
                    background: '#070B10',
                    border: '1px solid #16222B',
                    color: '#FFF',
                    padding: '8px 10px',
                    borderRadius: 6,
                    fontSize: '13px',
                    outline: 'none',
                  }}
                >
                  <option value="dusk_rural_swarm_raid">Dusk Rural Swarm Raid (12 drones + reactive relay)</option>
                  <option value="night_urban_single_drone_incursion">Night Urban Incursion (RF-silent glide)</option>
                  <option value="day_coastal_mixed_decoy_drill">Day Coastal Mixed Decoy (Radar decoys + birds)</option>
                </select>
              </div>

              {/* Difficulty Slider */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#8A99A6', marginBottom: 6 }}>
                  <span>Custom Difficulty Tier</span>
                  <span style={{ color: '#4DFFA8', fontFamily: 'var(--font-data)' }}>TIER {difficulty}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={difficulty}
                  onChange={e => setDifficulty(parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: '#4DFFA8' }}
                />
              </div>

              {/* Environment Lighting Switcher */}
              <div style={{ marginBottom: 20 }}>
                <label style={{ display: 'block', fontSize: '12px', color: '#8A99A6', marginBottom: 6 }}>3D Lighting Environment</label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                  {(['day', 'dusk', 'night'] as const).map(mode => (
                    <button
                      key={mode}
                      onClick={() => setTimeMode(mode)}
                      style={{
                        background: timeMode === mode ? 'rgba(77, 255, 168, 0.15)' : '#070B10',
                        color: timeMode === mode ? '#4DFFA8' : '#8A99A6',
                        border: `1px solid ${timeMode === mode ? '#4DFFA8' : '#16222B'}`,
                        padding: '6px 8px',
                        borderRadius: 6,
                        fontSize: '11px',
                        fontFamily: 'var(--font-data)',
                        cursor: 'pointer',
                        textTransform: 'uppercase',
                        transition: 'all 0.2s',
                      }}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {/* Deploy Buttons */}
              <button
                onClick={() => handleLaunchPreset(selectedDrill)}
                style={{
                  width: '100%',
                  background: 'linear-gradient(135deg, #4DFFA8, #3fb950)',
                  color: '#05070A',
                  border: 'none',
                  padding: '10px 14px',
                  borderRadius: 6,
                  fontFamily: 'var(--font-display)',
                  fontWeight: 700,
                  fontSize: '13px',
                  cursor: 'pointer',
                  marginBottom: 8,
                  letterSpacing: '0.04em',
                }}
              >
                DEPLOY SELECTED DRILL &rarr;
              </button>

              <button
                onClick={handleLaunchCustom}
                style={{
                  width: '100%',
                  background: 'transparent',
                  color: '#8A99A6',
                  border: '1px solid #16222B',
                  padding: '8px 14px',
                  borderRadius: 6,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                Launch Procedural Scenario
              </button>
            </div>
          </div>

          {/* Core Capabilities Architecture Grid */}
          <div style={{ marginTop: 40 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
              <span style={{ width: 8, height: 8, background: '#4DFFA8' }}></span>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '20px', letterSpacing: '0.04em', color: '#FFF' }}>
                TACTICAL C-UAS SIMULATION CAPABILITIES
              </h2>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
              
              {/* Card 1: Multi-Sensor Fusion */}
              <div style={{ background: 'rgba(11, 16, 21, 0.75)', border: '1px solid #16222B', borderRadius: 10, padding: 20, backdropFilter: 'blur(12px)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ color: '#4DFFA8' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                  </div>
                  <h3 style={{ fontSize: '16px', color: '#FFF' }}>Multi-Sensor Fusion</h3>
                </div>
                <p style={{ fontSize: '13px', color: '#8A99A6', lineHeight: 1.5, marginBottom: 14 }}>
                  Cross-verify contacts via <strong>EO/IR thermal shape outlines</strong>, <strong>RF emission intercepts</strong>, and <strong>acoustic blade-vibration signatures</strong>. Weather, time-of-day, and urban clutter degrade signals realistically.
                </p>
                <a href="/trainer" style={{ fontSize: '12px', color: '#4DFFA8', textDecoration: 'none', fontWeight: 600 }}>Test in 2D Scope &rarr;</a>
              </div>

              {/* Card 2: RadarSense AI Engine */}
              <div style={{ background: 'rgba(11, 16, 21, 0.75)', border: '1px solid #16222B', borderRadius: 10, padding: 20, backdropFilter: 'blur(12px)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ color: '#4DD8FF' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                  </div>
                  <h3 style={{ fontSize: '16px', color: '#FFF' }}>RadarSense CA-CFAR DSP</h3>
                </div>
                <p style={{ fontSize: '13px', color: '#8A99A6', lineHeight: 1.5, marginBottom: 14 }}>
                  No hardcoded lookup tables. Contacts synthesize <strong>77GHz FMCW radar I/Q data</strong> through a range-Doppler FFT and CA-CFAR detector into a real <strong>ONNX CatBoost classifier</strong> for live AI guidance hints.
                </p>
                <a href="/trainer" style={{ fontSize: '12px', color: '#4DD8FF', textDecoration: 'none', fontWeight: 600 }}>Inspect Model Status &rarr;</a>
              </div>

              {/* Card 3: Autonomous Swarm Divergence */}
              <div style={{ background: 'rgba(11, 16, 21, 0.75)', border: '1px solid #16222B', borderRadius: 10, padding: 20, backdropFilter: 'blur(12px)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ color: '#FFB020' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><circle cx="19" cy="5" r="2"/><circle cx="5" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/></svg>
                  </div>
                  <h3 style={{ fontSize: '16px', color: '#FFF' }}>Reactive Swarm AI</h3>
                </div>
                <p style={{ fontSize: '13px', color: '#8A99A6', lineHeight: 1.5, marginBottom: 14 }}>
                  Multi-UAV swarms driven by <strong>ares-x kinematics</strong> dynamically adapt. Kinetic neutralization of a swarm leader causes surviving drones to immediately break formation, re-route, and maintain distributed relay.
                </p>
                <a href="/trainer" style={{ fontSize: '12px', color: '#FFB020', textDecoration: 'none', fontWeight: 600 }}>Load Swarm Drill &rarr;</a>
              </div>

              {/* Card 4: Graduated ROE Ladder */}
              <div style={{ background: 'rgba(11, 16, 21, 0.75)', border: '1px solid #16222B', borderRadius: 10, padding: 20, backdropFilter: 'blur(12px)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ color: '#FF5A5A' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                  </div>
                  <h3 style={{ fontSize: '16px', color: '#FFF' }}>Graduated ROE Ladder</h3>
                </div>
                <p style={{ fontSize: '13px', color: '#8A99A6', lineHeight: 1.5, marginBottom: 14 }}>
                  Trainees are evaluated against military Rules of Engagement: <strong>Warn</strong> &rarr; <strong>Soft-Kill (RF jamming/spoofing)</strong> &rarr; <strong>Hard-Kill (kinetic)</strong> &rarr; <strong>Monitor</strong> or <strong>Ignore</strong>, with defended zones and civilian safety boundaries.
                </p>
                <a href="/trainer" style={{ fontSize: '12px', color: '#FF5A5A', textDecoration: 'none', fontWeight: 600 }}>View ROE Rules &rarr;</a>
              </div>

              {/* Card 5: Instructor Event Injection */}
              <div style={{ background: 'rgba(11, 16, 21, 0.75)', border: '1px solid #16222B', borderRadius: 10, padding: 20, backdropFilter: 'blur(12px)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ color: '#B48CFF' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
                  </div>
                  <h3 style={{ fontSize: '16px', color: '#FFF' }}>Instructor Live Console</h3>
                </div>
                <p style={{ fontSize: '13px', color: '#8A99A6', lineHeight: 1.5, marginBottom: 14 }}>
                  Instructors can monitor ongoing drills and execute <strong>live threat injections</strong> (decoys, hostile drones, bird clutter) and <strong>sensor jamming outages</strong> in real-time without interrupting the trainee's stream.
                </p>
                <a href="/instructor" style={{ fontSize: '12px', color: '#B48CFF', textDecoration: 'none', fontWeight: 600 }}>Open Console &rarr;</a>
              </div>

              {/* Card 6: Cryptographic AAR Ledger */}
              <div style={{ background: 'rgba(11, 16, 21, 0.75)', border: '1px solid #16222B', borderRadius: 10, padding: 20, backdropFilter: 'blur(12px)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ color: '#4DFFA8' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                  </div>
                  <h3 style={{ fontSize: '16px', color: '#FFF' }}>SHA-256 Audit Chain</h3>
                </div>
                <p style={{ fontSize: '13px', color: '#8A99A6', lineHeight: 1.5, marginBottom: 14 }}>
                  Every completed drill is cryptographically hashed into an auditable ledger. Includes automated <strong>error taxonomies</strong>, <strong>missed-contact spatial heatmaps</strong>, and <strong>unit readiness certifications</strong>.
                </p>
                <a href="/aar" style={{ fontSize: '12px', color: '#4DFFA8', textDecoration: 'none', fontWeight: 600 }}>Verify Audit Chain &rarr;</a>
              </div>

            </div>
          </div>

          {/* Bottom Tactical Diagnostics Bar */}
          <div style={{
            marginTop: 40,
            marginBottom: 30,
            padding: '14px 24px',
            background: 'rgba(7, 11, 16, 0.9)',
            border: '1px solid #16222B',
            borderRadius: 8,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 12,
            fontFamily: 'var(--font-data)',
            fontSize: '12px',
            color: '#6B7A86',
          }}>
            <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
              <span>RADAR SWEEP: <strong style={{ color: '#4DFFA8' }}>360° FMCW</strong></span>
              <span>DSP: <strong style={{ color: '#4DD8FF' }}>CA-CFAR ACTIVE</strong></span>
              <span>CLASSIFIER: <strong style={{ color: '#FFF' }}>ONNX CATBOOST</strong></span>
              <span>SWARM LOGIC: <strong style={{ color: '#FFB020' }}>ARES-X RELAY</strong></span>
              <span>AUDIT: <strong style={{ color: '#4DFFA8' }}>SHA-256 SEALED</strong></span>
            </div>
            <div style={{ color: '#8A99A6' }}>
              PORT: <strong>8090</strong> // FASTAPI SERVER
            </div>
          </div>

        </div>

      </div>
    </div>
  )
}
