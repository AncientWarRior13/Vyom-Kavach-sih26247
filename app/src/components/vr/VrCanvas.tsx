import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { VrScene, VrViewMode } from './VrScene';
import { VrHud } from './VrHud';
import { vrAudio } from './VrAudio';
import { api } from '../../api/client';
import { useSessionStore } from '../../state/session';
import { useToast } from '../../ui/Toast';
import type { Contact } from '../../api/types';

export interface VrCanvasProps {
  scenario: any;
  contacts: Contact[];
  elapsed: number;
  getElapsed: () => number;
  className?: string;
}

export const VrCanvas: React.FC<VrCanvasProps> = ({
  scenario,
  contacts,
  elapsed,
  getElapsed,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<VrScene | null>(null);
  const { toast } = useToast();

  const setStoreTraineeLog = useSessionStore((s) => s.setTraineeLog);

  // Mode and View State
  const [viewMode, setViewMode] = useState<VrViewMode>('desktop');
  const [webXrSupported, setWebXrSupported] = useState(false);
  const [webXrReason, setWebXrReason] = useState<string>('Detecting WebXR support...');
  const [hasGyro, setHasGyro] = useState(false);
  const [useGyro, setUseGyro] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedContactId, setSelectedContactId] = useState<string | null>(null);
  const [dwellProgress, setDwellProgress] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  // Check WebXR hardware support
  useEffect(() => {
    if (typeof navigator !== 'undefined' && (navigator as any).xr) {
      (navigator as any).xr
        .isSessionSupported('immersive-vr')
        .then((supported: boolean) => {
          setWebXrSupported(supported);
          if (!supported) {
            setWebXrReason('No compatible WebXR headset or VR runtime detected on this device.');
          }
        })
        .catch((err: any) => {
          setWebXrSupported(false);
          setWebXrReason(err.message || 'WebXR initialization error.');
        });
    } else {
      setWebXrSupported(false);
      setWebXrReason('Browser does not support WebXR Device API (requires secure origin/HTTPS or localhost).');
    }

    if (typeof window !== 'undefined' && 'DeviceOrientationEvent' in window) {
      setHasGyro(true);
    }
  }, []);

  // Initialize VrScene
  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new VrScene(containerRef.current, {
      onSelectContact: (id) => setSelectedContactId(id),
      onLockProgress: (prog) => setDwellProgress(prog),
    });

    sceneRef.current = scene;

    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Push updated telemetry data and scenario into VrScene
  useEffect(() => {
    if (!sceneRef.current) return;
    sceneRef.current.setData({
      contacts,
      scenario,
      selectedContactId,
    });
  }, [contacts, scenario, selectedContactId]);

  // Selected contact object
  const selectedContact = useMemo(() => {
    if (!selectedContactId || contacts.length === 0) return null;
    return contacts.find((c) => c.id === selectedContactId) || null;
  }, [contacts, selectedContactId]);

  // Mode Switch Handlers
  const handleSelectMode = async (mode: VrViewMode) => {
    if (!sceneRef.current) return;

    if (mode === 'immersive') {
      const res = await sceneRef.current.enterImmersiveVr();
      if (!res.success) {
        toast('WEBXR NOTICE', res.reason || 'Could not start immersive VR.', 'warning');
      } else {
        setViewMode('immersive');
      }
    } else {
      if (viewMode === 'immersive') {
        sceneRef.current.exitVr();
      }
      sceneRef.current.setViewMode(mode);
      setViewMode(mode);
    }
  };

  // Fullscreen Management
  const handleToggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  // Gyroscope Permission on Mobile / iOS
  const handleToggleGyro = async () => {
    if (typeof (DeviceOrientationEvent as any)?.requestPermission === 'function') {
      try {
        const permission = await (DeviceOrientationEvent as any).requestPermission();
        if (permission === 'granted') {
          const next = !useGyro;
          setUseGyro(next);
          sceneRef.current?.setUseGyro(next);
        }
      } catch {
        toast('GYRO ERROR', 'Device orientation permission was denied.', 'warning');
      }
    } else {
      const next = !useGyro;
      setUseGyro(next);
      sceneRef.current?.setUseGyro(next);
    }
  };

  const handleRecenterGyro = () => {
    sceneRef.current?.recenterGyro();
    toast('ORIENTATION CALIBRATED', 'Horizon and heading zeroed to current viewpoint.', 'info');
  };

  // Audio Toggle
  const handleToggleMute = () => {
    const nextMuted = vrAudio.toggleMute();
    setIsMuted(nextMuted);
  };

  // Real Engagement Execution per Server Contract
  const handleExecuteEngagement = async (action: 'warn' | 'soft_kill' | 'hard_kill') => {
    if (!selectedContact || !scenario) return;

    const cid = selectedContact.id;
    const t = getElapsed();

    if (action === 'hard_kill') {
      // 1. Visual & Audio Effect
      sceneRef.current?.triggerHardKillEffect(cid);
      setSelectedContactId(null);

      // 2. Synchronize local traineeLog in store
      setStoreTraineeLog((prev) => ({
        ...prev,
        [cid]: {
          ...(prev[cid] || { contact_id: cid, detected_at: t }),
          engagement_action: 'hard_kill',
          engagement_at: t,
          state: 'neutralized',
        },
      }));

      // 3. Server call to POST /api/contact/engage with real contract { scenario_id, contact_id, t, action }
      try {
        const res = await api.engageContactV2({
          scenario_id: scenario.scenario_id,
          contact_id: cid,
          t,
          action: 'hard_kill',
        });

        if (res.reactive) {
          toast('HARD-KILL CONFIRMED', `Swarm track ${cid.slice(0, 8)} intercepted! Swarm routes diverging.`, 'success');
        } else {
          toast('HARD-KILL CONFIRMED', `Kinetic countermeasure deployed against track ${cid.slice(0, 8)}.`, 'success');
        }
      } catch (err: any) {
        toast('ENGAGEMENT ERROR', err.message || 'Kinetic deployment failed.', 'error');
      }
    } else if (action === 'soft_kill') {
      // Soft-kill: Directed RF Jamming cone
      sceneRef.current?.triggerSoftKillEffect(cid);

      setStoreTraineeLog((prev) => ({
        ...prev,
        [cid]: {
          ...(prev[cid] || { contact_id: cid, detected_at: t }),
          engagement_action: 'soft_kill',
          engagement_at: t,
        },
      }));

      toast('RF JAM ACTIVE', `Multi-band directional jamming deployed against track ${cid.slice(0, 8)}.`, 'warning');
    } else {
      // Warn / Hail
      vrAudio.playWarning();

      setStoreTraineeLog((prev) => ({
        ...prev,
        [cid]: {
          ...(prev[cid] || { contact_id: cid, detected_at: t }),
          warn_at: t,
        },
      }));

      toast('WARNING TRANSMITTED', `Advisory broadcast on emergency defense frequencies to track ${cid.slice(0, 8)}.`, 'info');
    }
  };

  return (
    <div className={`relative w-full h-full overflow-hidden bg-[#020508] select-none ${className}`}>
      {/* 3D WebGL Canvas Container */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* 2D / Stereo HUD Overlay */}
      <VrHud
        scenario={scenario}
        contacts={contacts}
        selectedContact={selectedContact}
        elapsed={elapsed}
        viewMode={viewMode}
        webXrSupported={webXrSupported}
        webXrReason={webXrReason}
        hasGyro={hasGyro}
        useGyro={useGyro}
        dwellProgress={dwellProgress}
        isMuted={isMuted}
        onToggleMute={handleToggleMute}
        onSelectMode={handleSelectMode}
        onToggleFullscreen={handleToggleFullscreen}
        isFullscreen={isFullscreen}
        onToggleGyro={handleToggleGyro}
        onRecenterGyro={handleRecenterGyro}
        onSelectContact={(id) => {
          setSelectedContactId(id);
          sceneRef.current?.selectContact(id);
        }}
        onExecuteEngagement={handleExecuteEngagement}
      />
    </div>
  );
};
