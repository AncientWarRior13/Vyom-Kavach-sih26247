import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { Contact } from '../../api/types';

export interface UrbanMap3DProps {
  contacts: Contact[];
  selectedContactId?: string | null;
  onSelectContact?: (contactId: string) => void;
  originLat?: number;
  originLon?: number;
  defendedAsset?: [number, number];
  className?: string;
}

const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty';
const BUILDING_LAYER_ID = 'building-3d';
const FALLBACK_BUILDING_LAYER_ID = 'drone-3d-buildings';

const CLASS_COLORS: Record<string, string> = {
  hostile_drone: '#FF3B3B',
  hostile_swarm_member: '#FF3B3B',
  drone: '#FF3B3B',
  quadcopter: '#FF3B3B',
  fixed_wing: '#FF6B00',
  decoy: '#BF5AF2',
  friendly_drone: '#00E87A',
  commercial: '#0A84FF',
  bird: '#3DD6C6',
  clutter: '#BF5AF2',
  unknown: '#FFB800',
};

// Set local worker script URL for MapLibre GL
if (typeof window !== 'undefined') {
  maplibregl.setWorkerUrl('/maplibre-gl-worker.mjs');
}

export const UrbanMap3D: React.FC<UrbanMap3DProps> = ({
  contacts,
  selectedContactId,
  onSelectContact,
  originLat = 28.6139, // Default tactical coordinates (e.g. urban base)
  originLon = 77.2090,
  defendedAsset,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<Map<string, maplibregl.Marker>>(new Map());
  const originMarkerRef = useRef<maplibregl.Marker | null>(null);
  const [buildingStats, setBuildingStats] = useState({ count: 0, avgHeight: 18, maxHeight: 52 });
  const [mapError, setMapError] = useState<string | null>(null);

  // Reset view to overview of all contacts
  const resetView = () => {
    if (!mapRef.current) return;
    mapRef.current.flyTo({
      center: [originLon, originLat],
      zoom: 14.8,
      pitch: 58,
      bearing: -17,
      speed: 1.2,
      curve: 1.1,
      essential: true,
    });
  };

  // Initialize MapLibre 3D
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    try {
      const map = new maplibregl.Map({
        container: containerRef.current,
        style: STYLE_URL,
        center: [originLon, originLat],
        zoom: 14.8, // 14.8 shows full 3-4km tactical airspace while retaining 3D building extrusions
        pitch: 58,  // Tactical 58-degree tilt for depth
        bearing: -17,
        attributionControl: false,
      });

      map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');

      map.on('error', (e) => {
        const msg = e.error?.message || '';
        if (msg.includes('WebGL') || msg.includes('Failed to initialize')) {
          setMapError(msg);
        }
      });

      // ResizeObserver to ensure canvas renders when parent container dimensions change
      const ro = new ResizeObserver(() => {
        map.resize();
      });
      ro.observe(containerRef.current);

      map.on('load', () => {
        map.resize();

        // 3D Building Extrusion Layer from OpenMapTiles vector data
        const heightExpr: maplibregl.ExpressionSpecification = ['coalesce', ['get', 'render_height'], 18];
        const minHeightExpr: maplibregl.ExpressionSpecification = ['coalesce', ['get', 'render_min_height'], 0];
        const colorExpr: maplibregl.ExpressionSpecification = [
          'interpolate',
          ['linear'],
          heightExpr,
          0,
          '#1e293b',
          25,
          '#334155',
          60,
          '#475569',
          120,
          '#64748b',
        ];

        if (map.getLayer(BUILDING_LAYER_ID)) {
          map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-height', heightExpr);
          map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-base', minHeightExpr);
          map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-color', colorExpr);
          map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-opacity', 0.92);
        } else if (map.getSource('openmaptiles')) {
          map.addLayer({
            id: FALLBACK_BUILDING_LAYER_ID,
            source: 'openmaptiles',
            'source-layer': 'building',
            type: 'fill-extrusion',
            minzoom: 13,
            paint: {
              'fill-extrusion-color': colorExpr,
              'fill-extrusion-height': heightExpr,
              'fill-extrusion-base': minHeightExpr,
              'fill-extrusion-opacity': 0.92,
            },
          });
        }
      });

      map.on('idle', () => {
        // Query visible extruded buildings
        const features = map.queryRenderedFeatures(undefined, {
          layers: [BUILDING_LAYER_ID, FALLBACK_BUILDING_LAYER_ID].filter((l) => map.getLayer(l)),
        });
        if (features.length > 0) {
          const heights: number[] = [];
          features.forEach((f) => {
            const h = Number(f.properties?.render_height);
            if (Number.isFinite(h) && h > 0) heights.push(h);
          });
          if (heights.length > 0) {
            setBuildingStats({
              count: features.length,
              avgHeight: Math.round(heights.reduce((a, b) => a + b, 0) / heights.length),
              maxHeight: Math.round(Math.max(...heights)),
            });
          }
        }
      });

      mapRef.current = map;
      return () => {
        ro.disconnect();
        map.remove();
        mapRef.current = null;
      };
    } catch (err: any) {
      setMapError(err.message || 'WebGL not supported');
    }
  }, [originLat, originLon]);

  // Defended Base Asset Pin (Central Command Post)
  useEffect(() => {
    if (!mapRef.current) return;
    if (!originMarkerRef.current) {
      const el = document.createElement('div');
      el.className = 'w-6 h-6 rounded-xs border-2 border-white bg-emerald-500 rotate-45 shadow-[0_0_16px_#10b981] flex items-center justify-center pointer-events-none select-none';
      el.innerHTML = '<div class="w-2 h-2 rounded-full bg-white"></div>';
      originMarkerRef.current = new maplibregl.Marker({ element: el })
        .setLngLat([originLon, originLat])
        .addTo(mapRef.current);
    } else {
      originMarkerRef.current.setLngLat([originLon, originLat]);
    }
  }, [originLat, originLon]);

  // Sync Contacts to Glowing Radar Markers
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const seen = new Set<string>();

    contacts.forEach((c) => {
      seen.add(c.id);
      const isSelected = c.id === selectedContactId;
      const cls = (c.assigned_classification || c.classification || 'unknown').toLowerCase();
      const color = CLASS_COLORS[cls] || CLASS_COLORS.unknown;

      // Project range and bearing to ground coordinates over the urban canopy
      const DISTANCE_SCALE = 0.35;
      const rMeters = (c.range_km ?? (Math.hypot(c.x ?? 0, c.y ?? 0) / 1000)) * 1000 * DISTANCE_SCALE;
      const bDeg = c.bearing_deg ?? (((Math.atan2(c.x ?? 0, -(c.y ?? 0)) * 180 / Math.PI) + 360) % 360);
      const bRad = (bDeg * Math.PI) / 180;

      const dEast = rMeters * Math.sin(bRad);
      const dNorth = rMeters * Math.cos(bRad);

      const lat = originLat + dNorth / 111111;
      const lon = originLon + dEast / (111111 * Math.cos((originLat * Math.PI) / 180));

      let marker = markersRef.current.get(c.id);
      if (!marker) {
        const el = document.createElement('div');
        el.className = 'tactical-drone-marker group relative cursor-pointer select-none';
        el.style.width = '28px';
        el.style.height = '28px';
        el.style.display = 'flex';
        el.style.alignItems = 'center';
        el.style.justifyContent = 'center';
        el.style.pointerEvents = 'auto';

        el.innerHTML = `
          <div class="glow-aura absolute rounded-full pointer-events-none transition-all duration-300" style="
            inset: -8px;
            background: radial-gradient(circle, ${color} 0%, rgba(0,0,0,0) 70%);
            opacity: 0.85;
          "></div>
          <div class="glow-ring absolute rounded-full border pointer-events-none animate-ping opacity-60" style="
            inset: -4px;
            border-color: ${color};
          "></div>
          <div class="marker-core relative w-4 h-4 rounded-full flex items-center justify-center transition-all duration-150" style="
            background-color: ${color};
            box-shadow: 0 0 10px ${color}, 0 0 20px ${color}, inset 0 0 3px #ffffff;
            border: 2px solid #ffffff;
          ">
            <span class="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_4px_#fff]"></span>
          </div>
          <div class="marker-tag absolute left-1/2 -bottom-6 -translate-x-1/2 whitespace-nowrap font-mono text-[9px] font-bold px-1.5 py-0.5 rounded shadow-lg pointer-events-none flex items-center gap-1 transition-all duration-150" style="
            background: rgba(5, 10, 16, 0.94);
            border: 1px solid ${isSelected ? '#00e87a' : 'rgba(255,255,255,0.22)'};
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(0,0,0,0.8);
          ">
            <span class="tag-dot" style="color: ${color};">●</span>
            <span class="tag-id">${c.id.slice(0, 8).toUpperCase()}</span>
            <span class="tag-alt" style="color: rgba(255,255,255,0.6); font-weight: normal;">${c.altitude ?? 120}m</span>
          </div>
        `;

        el.addEventListener('click', (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          onSelectContact?.(c.id);
        });

        marker = new maplibregl.Marker({ element: el }).setLngLat([lon, lat]).addTo(map);
        markersRef.current.set(c.id, marker);
      } else {
        marker.setLngLat([lon, lat]);
      }

      // Update styling and selection ring
      const markerEl = marker.getElement();
      const coreEl = markerEl.querySelector('.marker-core') as HTMLElement;
      const auraEl = markerEl.querySelector('.glow-aura') as HTMLElement;
      const ringEl = markerEl.querySelector('.glow-ring') as HTMLElement;
      const tagEl = markerEl.querySelector('.marker-tag') as HTMLElement;
      const tagDot = markerEl.querySelector('.tag-dot') as HTMLElement;
      const tagAlt = markerEl.querySelector('.tag-alt') as HTMLElement;

      if (coreEl) {
        coreEl.style.backgroundColor = color;
        coreEl.style.boxShadow = isSelected
          ? `0 0 0 3px #00e87a, 0 0 16px ${color}, 0 0 30px ${color}`
          : `0 0 10px ${color}, 0 0 20px ${color}, inset 0 0 3px #ffffff`;
        coreEl.style.transform = isSelected ? 'scale(1.4)' : 'scale(1)';

        if (auraEl) {
          auraEl.style.background = `radial-gradient(circle, ${color} 0%, rgba(0,0,0,0) 70%)`;
          auraEl.style.inset = isSelected ? '-14px' : '-8px';
          auraEl.style.opacity = isSelected ? '1' : '0.85';
        }
        if (ringEl) {
          ringEl.style.borderColor = color;
        }
        if (tagEl) {
          tagEl.style.borderColor = isSelected ? '#00e87a' : 'rgba(255,255,255,0.22)';
          tagEl.style.boxShadow = isSelected ? '0 0 12px rgba(0,232,122,0.4)' : '0 2px 8px rgba(0,0,0,0.8)';
          if (tagDot) tagDot.style.color = color;
          if (tagAlt) tagAlt.textContent = `${c.altitude ?? 120}m`;
        }
        markerEl.style.zIndex = isSelected ? '50' : '10';
      }
    });

    // Remove obsolete markers
    markersRef.current.forEach((marker, id) => {
      if (!seen.has(id)) {
        marker.remove();
        markersRef.current.delete(id);
      }
    });
  }, [contacts, selectedContactId, originLat, originLon, onSelectContact]);

  // Smoothly pan camera towards selected contact without losing full context
  const lastFlownTargetIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!mapRef.current || !selectedContactId) return;

    if (lastFlownTargetIdRef.current === selectedContactId) return;
    lastFlownTargetIdRef.current = selectedContactId;

    const target = contacts.find((c) => c.id === selectedContactId);
    if (!target) return;

    const DISTANCE_SCALE = 0.35;
    const rMeters = (target.range_km ?? (Math.hypot(target.x ?? 0, target.y ?? 0) / 1000)) * 1000 * DISTANCE_SCALE;
    const bDeg = target.bearing_deg ?? (((Math.atan2(target.x ?? 0, -(target.y ?? 0)) * 180 / Math.PI) + 360) % 360);
    const bRad = (bDeg * Math.PI) / 180;
    const dEast = rMeters * Math.sin(bRad);
    const dNorth = rMeters * Math.cos(bRad);

    const lat = originLat + dNorth / 111111;
    const lon = originLon + dEast / (111111 * Math.cos((originLat * Math.PI) / 180));

    mapRef.current.flyTo({
      center: [lon, lat],
      zoom: 15.5, // gentle tactical zoom, keeps full urban context & nearby contacts visible
      pitch: 60,
      bearing: target.bearing_deg ?? -17,
      speed: 1.2,
      curve: 1.1,
      essential: true,
    });
  }, [selectedContactId, contacts, originLat, originLon]);

  return (
    <div className={`relative w-full h-full overflow-hidden ${className}`}>
      <div ref={containerRef} className="w-full h-full" />

      {/* Urban Canopy Telemetry & Controls Box */}
      <div className="absolute top-3 left-3 pointer-events-auto font-mono text-[10px] space-y-1.5 bg-black/85 backdrop-blur-md p-2.5 rounded border border-white/10 text-[var(--c-text-mid)] z-10 shadow-xl">
        <div className="text-[var(--c-accent)] font-bold uppercase tracking-wider flex items-center justify-between gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--c-accent)] animate-pulse" />
            URBAN 3D CANOPY & TERRAIN MAPPING
          </div>
          <button
            type="button"
            onClick={resetView}
            className="px-2 py-0.5 bg-[var(--c-bg-raised)] hover:bg-[var(--c-accent)] hover:text-black border border-white/20 text-white rounded cursor-pointer transition-colors text-[9px] font-bold uppercase"
          >
            Reset View
          </button>
        </div>
        <div className="flex gap-3 text-white">
          <span>CANOPY BUILDINGS: <strong className="text-[var(--c-accent)]">{buildingStats.count}</strong></span>
          <span>AVG ROOF: <strong>{buildingStats.avgHeight}m</strong></span>
          <span>ACTIVE CONTACTS: <strong className="text-[var(--c-accent)]">{contacts.length}</strong></span>
        </div>
        <div className="text-[var(--c-text-faint)]">OSM Vector Extrusions · Tilt: 58° · Real-time 3D Radar Blips</div>
      </div>

      {mapError && (
        <div className="absolute bottom-3 left-3 bg-red-950/80 border border-red-500/50 p-2 text-xs font-mono text-red-300 rounded-xs">
          Map rendering notice: {mapError}
        </div>
      )}
    </div>
  );
};
