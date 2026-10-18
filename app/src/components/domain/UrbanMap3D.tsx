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
  drone: '#EF4444',
  quadcopter: '#EF4444',
  fixed_wing: '#F97316',
  commercial: '#3B82F6',
  bird: '#EAB308',
  clutter: '#8B5CF6',
  unknown: '#9CA3AF',
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

  // Initialize MapLibre 3D
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    try {
      const map = new maplibregl.Map({
        container: containerRef.current,
        style: STYLE_URL,
        center: [originLon, originLat],
        zoom: 16.5,
        pitch: 60, // 60-degree tactical forward tilt
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

  // Defended Base Asset Pin
  useEffect(() => {
    if (!mapRef.current) return;
    if (!originMarkerRef.current) {
      const el = document.createElement('div');
      el.className = 'w-5 h-5 rounded-xs border-2 border-white bg-emerald-500 rotate-45 shadow-[0_0_12px_#10b981] flex items-center justify-center';
      originMarkerRef.current = new maplibregl.Marker({ element: el })
        .setLngLat([originLon, originLat])
        .addTo(mapRef.current);
    } else {
      originMarkerRef.current.setLngLat([originLon, originLat]);
    }
  }, [originLat, originLon]);

  // Sync Contacts to 3D Map Markers
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const seen = new Set<string>();

    contacts.forEach((c) => {
      seen.add(c.id);
      const isSelected = c.id === selectedContactId;
      const cls = (c.assigned_classification || c.classification || 'unknown').toLowerCase();
      const color = CLASS_COLORS[cls] || CLASS_COLORS.unknown;

      // Use true high-precision Cartesian coordinates (c.x = Easting, c.y = Northing in meters)
      // Scaled by 0.5 so targets sit over the 3D urban canopy while preserving true formation spacing
      const DISTANCE_SCALE = 0.5;
      const dEast = (c.x ?? 0) * DISTANCE_SCALE;
      const dNorth = (c.y ?? 0) * DISTANCE_SCALE;

      // Approx lat/lon delta (1 deg lat ~= 111,111m)
      const lat = originLat + dNorth / 111111;
      const lon = originLon + dEast / (111111 * Math.cos((originLat * Math.PI) / 180));

      let marker = markersRef.current.get(c.id);
      if (!marker) {
        const el = document.createElement('div');
        el.className = 'tactical-drone-marker group relative cursor-pointer select-none';
        el.style.width = '32px';
        el.style.height = '32px';
        el.style.display = 'flex';
        el.style.alignItems = 'center';
        el.style.justifyContent = 'center';
        el.style.pointerEvents = 'auto';

        el.innerHTML = `
          <div class="glow-ring absolute inset-0 rounded-full animate-ping opacity-75" style="background-color: ${color};"></div>
          <div class="marker-core relative w-5 h-5 rounded-full border-2 border-white shadow-[0_0_14px_currentColor] flex items-center justify-center font-mono text-[9px] font-bold text-white transition-transform duration-150" style="background-color: ${color}; color: ${color};">
            <span class="w-1.5 h-1.5 rounded-full bg-white"></span>
          </div>
          <div class="marker-label absolute bottom-7 left-1/2 -translate-x-1/2 whitespace-nowrap bg-black/95 font-mono text-[10px] px-2 py-0.5 rounded border border-white/20 text-white shadow-xl pointer-events-none transition-all duration-150 ${
            isSelected ? 'opacity-100 scale-100 z-50' : 'opacity-0 scale-90 group-hover:opacity-100 group-hover:scale-100 z-20'
          }">
            <span style="color: ${color}; font-weight: bold;">${c.id.toUpperCase()}</span> · ${c.altitude ?? 120}m
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
      const ringEl = markerEl.querySelector('.glow-ring') as HTMLElement;
      const labelEl = markerEl.querySelector('.marker-label') as HTMLElement;

      if (coreEl) {
        coreEl.style.backgroundColor = color;
        coreEl.style.color = color;
        if (ringEl) ringEl.style.backgroundColor = color;

        if (isSelected) {
          coreEl.style.transform = 'scale(1.5)';
          coreEl.style.boxShadow = `0 0 0 3px #00e87a, 0 0 20px ${color}`;
          markerEl.style.zIndex = '50';
          if (labelEl) {
            labelEl.style.opacity = '1';
            labelEl.style.transform = 'translate(-50%, 0) scale(1.05)';
            labelEl.style.borderColor = '#00e87a';
          }
        } else {
          coreEl.style.transform = 'scale(1)';
          coreEl.style.boxShadow = `0 0 14px ${color}`;
          markerEl.style.zIndex = '10';
          if (labelEl) {
            labelEl.style.opacity = '';
            labelEl.style.transform = 'translate(-50%, 0) scale(1)';
            labelEl.style.borderColor = 'rgba(255, 255, 255, 0.2)';
          }
        }
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

  // Smoothly Fly To the selected contact ONLY ONCE when a new target is clicked
  const lastFlownTargetIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!mapRef.current || !selectedContactId) return;

    // If we already flew to this target, do NOT snap camera again on tick updates
    if (lastFlownTargetIdRef.current === selectedContactId) return;
    lastFlownTargetIdRef.current = selectedContactId;

    const target = contacts.find((c) => c.id === selectedContactId);
    if (!target) return;

    const DISTANCE_SCALE = 0.35;
    const rangeM = (target.range_km ?? 0) * 1000 * DISTANCE_SCALE;
    const bearingRad = ((target.bearing_deg ?? 0) * Math.PI) / 180;
    const dNorth = Math.cos(bearingRad) * rangeM;
    const dEast = Math.sin(bearingRad) * rangeM;

    const lat = originLat + dNorth / 111111;
    const lon = originLon + dEast / (111111 * Math.cos((originLat * Math.PI) / 180));

    mapRef.current.flyTo({
      center: [lon, lat],
      zoom: 17.5,
      pitch: 65,
      bearing: target.bearing_deg ?? -17,
      speed: 1.4,
      curve: 1.2,
      essential: true,
    });
  }, [selectedContactId]);

  return (
    <div className={`relative w-full h-full overflow-hidden ${className}`}>
      <div ref={containerRef} className="w-full h-full" />

      {/* Urban Canopy Telemetry Box */}
      <div className="absolute top-3 left-3 pointer-events-none font-mono text-[10px] space-y-1 bg-black/75 backdrop-blur-md p-2.5 rounded-xs border border-white/10 text-[var(--c-text-mid)] z-10">
        <div className="text-[var(--c-accent)] font-bold uppercase tracking-wider flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--c-accent)] animate-pulse" />
          URBAN 3D CANOPY & TERRAIN MAPPING
        </div>
        <div className="flex gap-3 text-white">
          <span>CANOPY BUILDINGS: <strong className="text-[var(--c-accent)]">{buildingStats.count}</strong></span>
          <span>AVG ROOF: <strong>{buildingStats.avgHeight}m</strong></span>
          <span>MAX HEIGHT: <strong>{buildingStats.maxHeight}m</strong></span>
        </div>
        <div className="text-[var(--c-text-faint)]">OSM Vector Extrusions · Tilt: 60° · Free Global Tiles</div>
      </div>

      {mapError && (
        <div className="absolute bottom-3 left-3 bg-red-950/80 border border-red-500/50 p-2 text-xs font-mono text-red-300 rounded-xs">
          Map rendering notice: {mapError}
        </div>
      )}
    </div>
  );
};
