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

const DARK_STYLE_URL = 'https://tiles.openfreemap.org/styles/dark';
const COMPOUND_SOURCE_ID = 'tactical-compound';
const COMPOUND_LAYER_ID = 'tactical-compound-extrusions';
const OSM_3D_BUILDING_LAYER_ID = 'drone-3d-buildings';

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
  try {
    maplibregl.setWorkerUrl('/maplibre-gl-worker.mjs');
  } catch (err) {
    console.warn('MapLibre worker registration notice:', err);
  }
}

// Helper to create rectangular polygon in geo coordinates
function makeRectPolygon(
  centerLat: number,
  centerLon: number,
  offsetEastM: number,
  offsetNorthM: number,
  widthEastM: number,
  depthNorthM: number
): [number, number][] {
  const mToLat = 1 / 111111;
  const mToLon = 1 / (111111 * Math.cos((centerLat * Math.PI) / 180));
  const cLat = centerLat + offsetNorthM * mToLat;
  const cLon = centerLon + offsetEastM * mToLon;
  const halfW = (widthEastM / 2) * mToLon;
  const halfD = (depthNorthM / 2) * mToLat;

  return [
    [cLon - halfW, cLat - halfD],
    [cLon + halfW, cLat - halfD],
    [cLon + halfW, cLat + halfD],
    [cLon - halfW, cLat + halfD],
    [cLon - halfW, cLat - halfD],
  ];
}

// Helper to create regular polygon (e.g. octagonal tower)
function makeCirclePolygon(
  centerLat: number,
  centerLon: number,
  offsetEastM: number,
  offsetNorthM: number,
  radiusM: number,
  sides = 8
): [number, number][] {
  const mToLat = 1 / 111111;
  const mToLon = 1 / (111111 * Math.cos((centerLat * Math.PI) / 180));
  const cLat = centerLat + offsetNorthM * mToLat;
  const cLon = centerLon + offsetEastM * mToLon;

  const points: [number, number][] = [];
  for (let i = 0; i <= sides; i++) {
    const angle = (i * 2 * Math.PI) / sides;
    points.push([
      cLon + radiusM * Math.cos(angle) * mToLon,
      cLat + radiusM * Math.sin(angle) * mToLat,
    ]);
  }
  return points;
}

interface CompoundFeature {
  type: 'Feature';
  properties: {
    name: string;
    height: number;
    min_height: number;
    color: string;
  };
  geometry: {
    type: 'Polygon';
    coordinates: [number, number][][];
  };
}

interface CompoundFeatureCollection {
  type: 'FeatureCollection';
  features: CompoundFeature[];
}

// Generates tactical military installation compound structures around the defense origin
function generateTacticalCompoundGeoJSON(centerLat: number, centerLon: number): CompoundFeatureCollection {
  const features: CompoundFeature[] = [];

  // 1. C2 Tactical Command Bunker
  features.push({
    type: 'Feature',
    properties: {
      name: 'C2 TACTICAL OPERATIONS BUNKER',
      height: 24,
      min_height: 0,
      color: '#1e3a5f',
    },
    geometry: {
      type: 'Polygon',
      coordinates: [makeRectPolygon(centerLat, centerLon, 0, 0, 44, 34)],
    },
  });

  // 2. Primary 3D Air Surveillance Radar Mast (Octagonal Tower)
  features.push({
    type: 'Feature',
    properties: {
      name: '3D RADAR MAST [TOWER-01]',
      height: 68,
      min_height: 0,
      color: '#0f766e',
    },
    geometry: {
      type: 'Polygon',
      coordinates: [makeCirclePolygon(centerLat, centerLon, -65, 45, 9, 8)],
    },
  });

  // 3. Electronic Warfare & RF Intercept Tower
  features.push({
    type: 'Feature',
    properties: {
      name: 'EW / RF INTERCEPT TOWER [MAST-02]',
      height: 52,
      min_height: 0,
      color: '#4338ca',
    },
    geometry: {
      type: 'Polygon',
      coordinates: [makeCirclePolygon(centerLat, centerLon, 65, 40, 8, 8)],
    },
  });

  // 4. Drone Interceptor Hangars (Alpha, Bravo, Charlie, Delta)
  const hangars = [
    { east: -65, north: -55, name: 'INTERCEPTOR HANGAR-A' },
    { east: -22, north: -60, name: 'INTERCEPTOR HANGAR-B' },
    { east: 22, north: -60, name: 'INTERCEPTOR HANGAR-C' },
    { east: 65, north: -55, name: 'INTERCEPTOR HANGAR-D' },
  ];
  hangars.forEach((h) => {
    features.push({
      type: 'Feature',
      properties: {
        name: h.name,
        height: 16,
        min_height: 0,
        color: '#1e293b',
      },
      geometry: {
        type: 'Polygon',
        coordinates: [makeRectPolygon(centerLat, centerLon, h.east, h.north, 34, 22)],
      },
    });
  });

  // 5. CWIS & Directed Energy Emplacements (4 perimeter corners)
  const emplacements = [
    { east: -110, north: 85, name: 'CWIS TURRET NW' },
    { east: 110, north: 85, name: 'LASER EFFECTOR NE' },
    { east: -110, north: -85, name: 'CWIS TURRET SW' },
    { east: 110, north: -85, name: 'DEW EFFECTOR SE' },
  ];
  emplacements.forEach((e) => {
    features.push({
      type: 'Feature',
      properties: {
        name: e.name,
        height: 14,
        min_height: 0,
        color: '#334155',
      },
      geometry: {
        type: 'Polygon',
        coordinates: [makeRectPolygon(centerLat, centerLon, e.east, e.north, 18, 18)],
      },
    });
  });

  // 6. Perimeter Sentry Towers (4 cardinal posts)
  const sentryPosts = [
    { east: 0, north: 130, name: 'PERIMETER SENTRY POST NORTH' },
    { east: 0, north: -125, name: 'PERIMETER SENTRY POST SOUTH' },
    { east: -145, north: 0, name: 'PERIMETER SENTRY POST WEST' },
    { east: 145, north: 0, name: 'PERIMETER SENTRY POST EAST' },
  ];
  sentryPosts.forEach((p) => {
    features.push({
      type: 'Feature',
      properties: {
        name: p.name,
        height: 28,
        min_height: 0,
        color: '#475569',
      },
      geometry: {
        type: 'Polygon',
        coordinates: [makeCirclePolygon(centerLat, centerLon, p.east, p.north, 7, 8)],
      },
    });
  });

  // 7. Tactical Satcom & Power Sub-station
  features.push({
    type: 'Feature',
    properties: {
      name: 'SATCOM DISH PLATFORM',
      height: 12,
      min_height: 0,
      color: '#0284c7',
    },
    geometry: {
      type: 'Polygon',
      coordinates: [makeRectPolygon(centerLat, centerLon, 70, 0, 22, 22)],
    },
  });
  features.push({
    type: 'Feature',
    properties: {
      name: 'POWER & GENERATOR BANK',
      height: 9,
      min_height: 0,
      color: '#1f2937',
    },
    geometry: {
      type: 'Polygon',
      coordinates: [makeRectPolygon(centerLat, centerLon, -75, 0, 26, 18)],
    },
  });

  // 8. Perimeter Blast Defenses (Berms / Barriers)
  const blastWalls = [
    { east: 0, north: 105, w: 180, d: 5 },
    { east: 0, north: -105, w: 180, d: 5 },
    { east: -125, north: 0, w: 5, d: 160 },
    { east: 125, north: 0, w: 5, d: 160 },
  ];
  blastWalls.forEach((bw, i) => {
    features.push({
      type: 'Feature',
      properties: {
        name: `DEFENSIVE BLAST PERIMETER 0${i + 1}`,
        height: 7,
        min_height: 0,
        color: '#18202f',
      },
      geometry: {
        type: 'Polygon',
        coordinates: [makeRectPolygon(centerLat, centerLon, bw.east, bw.north, bw.w, bw.d)],
      },
    });
  });

  return {
    type: 'FeatureCollection',
    features,
  };
}

export const UrbanMap3D: React.FC<UrbanMap3DProps> = ({
  contacts,
  selectedContactId,
  onSelectContact,
  originLat = 28.6139, // Default tactical coordinates
  originLon = 77.2090,
  defendedAsset,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markersRef = useRef<Map<string, maplibregl.Marker>>(new Map());
  const originMarkerRef = useRef<maplibregl.Marker | null>(null);
  const [buildingStats, setBuildingStats] = useState({ count: 21, avgHeight: 24, maxHeight: 68 });
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
        style: DARK_STYLE_URL,
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

        // 1. Procedural Tactical Compound Extrusion Layer (guaranteed in rural, desert, or urban sectors)
        try {
          const compoundGeo = generateTacticalCompoundGeoJSON(originLat, originLon);
          map.addSource(COMPOUND_SOURCE_ID, {
            type: 'geojson',
            data: compoundGeo,
          });

          map.addLayer({
            id: COMPOUND_LAYER_ID,
            type: 'fill-extrusion',
            source: COMPOUND_SOURCE_ID,
            paint: {
              'fill-extrusion-color': ['get', 'color'],
              'fill-extrusion-height': ['get', 'height'],
              'fill-extrusion-base': ['get', 'min_height'],
              'fill-extrusion-opacity': 0.95,
            },
          });
        } catch (err) {
          console.warn('Could not add compound layer:', err);
        }

        // 2. OpenStreetMap 3D Building Extrusions (for urban sectors)
        try {
          if (map.getSource('openmaptiles')) {
            if (map.getLayer('building')) {
              map.setLayoutProperty('building', 'visibility', 'none');
            }
            map.addLayer({
              id: OSM_3D_BUILDING_LAYER_ID,
              source: 'openmaptiles',
              'source-layer': 'building',
              type: 'fill-extrusion',
              minzoom: 13,
              paint: {
                'fill-extrusion-color': [
                  'interpolate',
                  ['linear'],
                  ['coalesce', ['get', 'render_height'], 18],
                  0, '#0f172a',
                  25, '#1e293b',
                  60, '#334155',
                  120, '#475569',
                ],
                'fill-extrusion-height': ['coalesce', ['get', 'render_height'], 18],
                'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
                'fill-extrusion-opacity': 0.88,
              },
            });
          }
        } catch (err) {
          console.warn('Could not add OSM building extrusions:', err);
        }
      });

      map.on('idle', () => {
        const osmFeatures = map.queryRenderedFeatures(undefined, {
          layers: [OSM_3D_BUILDING_LAYER_ID].filter((l) => map.getLayer(l)),
        });

        const compoundCount = 21;
        const heights: number[] = [68, 52, 28, 24, 16, 14, 12, 9, 7];
        osmFeatures.forEach((f) => {
          const h = Number(f.properties?.render_height);
          if (Number.isFinite(h) && h > 0) heights.push(h);
        });

        setBuildingStats({
          count: compoundCount + osmFeatures.length,
          avgHeight: Math.round(heights.reduce((a, b) => a + b, 0) / heights.length),
          maxHeight: Math.round(Math.max(...heights)),
        });
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

  const lastDeconflictedCoordsRef = useRef<Map<string, { lat: number; lon: number }>>(new Map());

  // Sync Contacts to Glowing Radar Markers with Cluster De-confliction
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const seen = new Set<string>();

    // 1. Calculate base projected ground coordinates for each contact
    const DISTANCE_SCALE = 0.35;
    const projected = contacts.map((c) => {
      const rMeters = (c.range_km ?? (Math.hypot(c.x ?? 0, c.y ?? 0) / 1000)) * 1000 * DISTANCE_SCALE;
      const bDeg = c.bearing_deg ?? (((Math.atan2(c.x ?? 0, -(c.y ?? 0)) * 180 / Math.PI) + 360) % 360);
      const bRad = (bDeg * Math.PI) / 180;

      const dEast = rMeters * Math.sin(bRad);
      const dNorth = rMeters * Math.cos(bRad);
      return { c, dEast, dNorth };
    });

    // 2. Identify spatial clusters of contacts that would interfere/overlap (within 45m of each other)
    const CLUSTER_DIST_M = 45;
    const clusters: Array<typeof projected> = [];

    projected.forEach((item) => {
      const foundCluster = clusters.find((cl) =>
        cl.some((other) => Math.hypot(item.dEast - other.dEast, item.dNorth - other.dNorth) < CLUSTER_DIST_M)
      );
      if (foundCluster) {
        foundCluster.push(item);
      } else {
        clusters.push([item]);
      }
    });

    // 3. Compute de-conflicted positions with graceful tactical spacing
    const coordsMap = new Map<string, { lat: number; lon: number; tagPosClass: string }>();

    clusters.forEach((cl) => {
      if (cl.length === 1) {
        const item = cl[0];
        const lat = originLat + item.dNorth / 111111;
        const lon = originLon + item.dEast / (111111 * Math.cos((originLat * Math.PI) / 180));
        coordsMap.set(item.c.id, {
          lat,
          lon,
          tagPosClass: 'left-1/2 -bottom-6 -translate-x-1/2',
        });
      } else {
        // Disperse clustered contacts with clean tactical spacing (32m radius: ~45px separation on map)
        const N = cl.length;
        const SPREAD_RADIUS_M = Math.min(36, 24 + N * 3);
        cl.forEach((item, idx) => {
          const angle = (idx * 2 * Math.PI) / N + Math.PI / 4;
          const offEast = SPREAD_RADIUS_M * Math.cos(angle);
          const offNorth = SPREAD_RADIUS_M * Math.sin(angle);

          const lat = originLat + (item.dNorth + offNorth) / 111111;
          const lon = originLon + (item.dEast + offEast) / (111111 * Math.cos((originLat * Math.PI) / 180));

          // Position tags outward from cluster center so tags don't overlap either
          let tagPosClass = 'left-1/2 -bottom-6 -translate-x-1/2';
          if (offNorth > 6) {
            tagPosClass = 'left-1/2 -top-6 -translate-x-1/2';
          } else if (offNorth < -6) {
            tagPosClass = 'left-1/2 -bottom-6 -translate-x-1/2';
          } else if (offEast > 0) {
            tagPosClass = 'left-8 top-1/2 -translate-y-1/2';
          } else {
            tagPosClass = 'right-8 top-1/2 -translate-y-1/2';
          }

          coordsMap.set(item.c.id, { lat, lon, tagPosClass });
        });
      }
    });

    lastDeconflictedCoordsRef.current = coordsMap;

    contacts.forEach((c) => {
      seen.add(c.id);
      const isSelected = c.id === selectedContactId;
      const cls = (c.assigned_classification || c.classification || 'unknown').toLowerCase();
      const color = CLASS_COLORS[cls] || CLASS_COLORS.unknown;

      const coord = coordsMap.get(c.id) || {
        lat: originLat,
        lon: originLon,
        tagPosClass: 'left-1/2 -bottom-6 -translate-x-1/2',
      };
      const lat = coord.lat;
      const lon = coord.lon;

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
          <div class="marker-tag absolute ${coord.tagPosClass} whitespace-nowrap font-mono text-[9px] font-bold px-1.5 py-0.5 rounded shadow-lg pointer-events-none flex items-center gap-1 transition-all duration-150" style="
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
        if (c.state === 'neutralized') {
          coreEl.style.backgroundColor = '#ef4444';
          coreEl.style.boxShadow = isSelected
            ? '0 0 0 3px #ef4444, 0 0 16px #ef4444'
            : '0 0 10px #ef4444, inset 0 0 3px #ffffff';
          coreEl.style.transform = isSelected ? 'scale(1.3)' : 'scale(1)';
          coreEl.innerHTML = '<span style="font-size:12px;color:white;font-weight:bold;display:flex;align-items:center;justify-content:center;height:100%;">✕</span>';

          if (auraEl) {
            auraEl.style.background = 'radial-gradient(circle, rgba(239,68,68,0.65) 0%, rgba(0,0,0,0) 70%)';
            auraEl.style.inset = '-6px';
            auraEl.style.opacity = '0.75';
          }
          if (ringEl) {
            ringEl.style.borderColor = '#ef4444';
          }
          if (tagEl) {
            tagEl.className = `marker-tag absolute ${coord.tagPosClass} whitespace-nowrap font-mono text-[9px] font-bold px-1.5 py-0.5 rounded shadow-lg pointer-events-none flex items-center gap-1 bg-red-950/80 border border-red-500/60`;
            tagEl.innerHTML = `<span class="tag-id line-through text-red-300">${c.id.slice(0, 8).toUpperCase()}</span><span class="tag-alt text-red-400 font-bold"> [DESTROYED]</span>`;
          }
        } else {
          coreEl.style.backgroundColor = color;
          coreEl.style.boxShadow = isSelected
            ? `0 0 0 3px #00e87a, 0 0 16px ${color}, 0 0 30px ${color}`
            : `0 0 10px ${color}, 0 0 20px ${color}, inset 0 0 3px #ffffff`;
          coreEl.style.transform = isSelected ? 'scale(1.4)' : 'scale(1)';
          coreEl.innerHTML = '';

          if (auraEl) {
            auraEl.style.background = `radial-gradient(circle, ${color} 0%, rgba(0,0,0,0) 70%)`;
            auraEl.style.inset = isSelected ? '-14px' : '-8px';
            auraEl.style.opacity = isSelected ? '1' : '0.85';
          }
          if (ringEl) {
            ringEl.style.borderColor = color;
          }
          if (tagEl) {
            tagEl.className = `marker-tag absolute ${coord.tagPosClass} whitespace-nowrap font-mono text-[9px] font-bold px-1.5 py-0.5 rounded shadow-lg pointer-events-none flex items-center gap-1 transition-all duration-150`;
            tagEl.style.borderColor = isSelected ? '#00e87a' : 'rgba(255,255,255,0.22)';
            tagEl.style.boxShadow = isSelected ? '0 0 12px rgba(0,232,122,0.4)' : '0 2px 8px rgba(0,0,0,0.8)';
            if (tagDot) tagDot.style.color = color;
            if (tagAlt) tagAlt.textContent = `${c.altitude ?? 120}m`;
          }
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

    const coord = lastDeconflictedCoordsRef.current.get(target.id);
    let lat: number, lon: number;

    if (coord) {
      lat = coord.lat;
      lon = coord.lon;
    } else {
      const DISTANCE_SCALE = 0.35;
      const rMeters = (target.range_km ?? (Math.hypot(target.x ?? 0, target.y ?? 0) / 1000)) * 1000 * DISTANCE_SCALE;
      const bDeg = target.bearing_deg ?? (((Math.atan2(target.x ?? 0, -(target.y ?? 0)) * 180 / Math.PI) + 360) % 360);
      const bRad = (bDeg * Math.PI) / 180;
      const dEast = rMeters * Math.sin(bRad);
      const dNorth = rMeters * Math.cos(bRad);
      lat = originLat + dNorth / 111111;
      lon = originLon + dEast / (111111 * Math.cos((originLat * Math.PI) / 180));
    }

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
    <div className={`relative w-full h-full overflow-hidden ${className} bg-[#070B12]`}>
      <div ref={containerRef} className="w-full h-full bg-[#070B12]" />

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
        <div className="text-[var(--c-text-faint)]">Compound & OSM 3D Extrusions · Tilt: 58° · Real-time 3D Radar Blips</div>
      </div>

      {mapError && (
        <div className="absolute bottom-3 left-3 bg-red-950/80 border border-red-500/50 p-2 text-xs font-mono text-red-300 rounded-xs">
          Map rendering notice: {mapError}
        </div>
      )}
    </div>
  );
};
