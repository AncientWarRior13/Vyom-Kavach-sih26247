'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import AlertCenterPanel from '../components/AlertCenterPanel';
import {
  Activity, Compass, Sun, Moon, Usb, Radar, ShieldAlert, MapPin, BarChart2,
} from 'lucide-react';

export default function AlertsPage() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  return (
    <div className={`min-h-screen font-sans flex flex-col transition-colors duration-300 selection:bg-red-500/30 selection:text-red-300 ${theme === 'dark' ? 'bg-[#07090b] text-zinc-100' : 'bg-zinc-50 text-zinc-900'
      }`}>

      <header className={`border-b backdrop-blur-md sticky top-0 z-50 py-3.5 px-6 shadow-lg transition-colors duration-300 ${theme === 'dark' ? 'border-zinc-900 bg-zinc-950/50' : 'border-zinc-200 bg-white/80'
        }`}>
        <div className="max-w-[1600px] mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-red-600 to-red-400 flex items-center justify-center shadow-[0_0_15px_rgba(239,68,68,0.3)] animate-pulse">
              <ShieldAlert className="w-5 h-5 text-zinc-950" />
            </div>
            <div>
              <h1 className={`text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r flex items-center gap-2 ${theme === 'dark' ? 'from-zinc-50 via-zinc-100 to-zinc-300' : 'from-zinc-900 via-zinc-800 to-zinc-700'
                }`}>
                RadarSense <span className={`font-mono text-xs border px-1.5 py-0.5 rounded ${theme === 'dark' ? 'border-red-800/40 bg-red-950/20 text-red-400' : 'border-red-200 bg-red-50 text-red-600'
                  }`}>ALERTS</span>
              </h1>
              <p className={`text-[10px] font-mono uppercase tracking-wider ${theme === 'dark' ? 'text-zinc-400' : 'text-zinc-500'
                }`}>Critical-Threat Alert Dispatch Center</p>
            </div>

            <nav className={`flex items-center gap-1 border p-1 rounded-xl font-mono text-xs ml-4 ${theme === 'dark' ? 'border-zinc-800 bg-zinc-900/40' : 'border-zinc-200 bg-white'}`}>
              <Link href="/" className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50">
                <Activity className="w-3.5 h-3.5" /> Simulation Radar
              </Link>
              <Link href="/hard" className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50">
                <Usb className="w-3.5 h-3.5" /> Hardware Radar
              </Link>
              <Link href="/fleet" className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50">
                <Radar className="w-3.5 h-3.5" /> Fleet
              </Link>
              <Link href="/alerts" className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all bg-red-500/20 text-red-400 font-bold border border-red-500/30 shadow-sm">
                <ShieldAlert className="w-3.5 h-3.5" /> Alerts
              </Link>
              <Link href="/map" className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50">
                <MapPin className="w-3.5 h-3.5" /> Map
              </Link>
              <Link href="/benchmark" className="px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50">
                <BarChart2 className="w-3.5 h-3.5" /> Benchmark
              </Link>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
              className={`p-2 rounded-xl border transition-all duration-300 cursor-pointer ${theme === 'dark'
                ? 'border-zinc-800 bg-zinc-900/40 text-amber-400 hover:bg-zinc-800 hover:text-amber-300'
                : 'border-zinc-200 bg-white text-indigo-600 hover:bg-zinc-100 hover:text-indigo-500 shadow-sm'
                }`}
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
            <div className="hidden lg:flex flex-col text-right font-mono">
              <span className={`text-[10px] uppercase ${theme === 'dark' ? 'text-zinc-500' : 'text-zinc-400'}`}>Quant-A-Thon&apos;26 Finals</span>
              <span className="text-xs font-semibold text-red-400">api/alerting.py</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full max-w-[900px] mx-auto p-6">
        <AlertCenterPanel theme={theme} />
      </main>
    </div>
  );
}
