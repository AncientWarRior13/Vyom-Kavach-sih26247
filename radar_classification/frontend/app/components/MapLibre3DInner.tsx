'use client';












import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { type GeoTarget, type BuildingStats, DEFAULT_BUILDING_HEIGHT_M } from './GeoMapPanel';

const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty';














maplibregl.setWorkerUrl('/maplibre-gl-worker.mjs');

const BUILDING_LAYER_ID = 'building-3d';                       
const FALLBACK_BUILDING_LAYER_ID = 'radarsense-3d-buildings';  
const BUILDING_LAYER_CANDIDATES = [BUILDING_LAYER_ID, FALLBACK_BUILDING_LAYER_ID];

interface MarkerPoint extends GeoTarget {
  lat: number;
  lon: number;
}



function hasWebGL(): boolean {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch {
    return false;
  }
}






function queryBuildingStats(map: maplibregl.Map): BuildingStats {
  for (const layerId of BUILDING_LAYER_CANDIDATES) {
    if (!map.getLayer(layerId)) continue;
    const features = map.queryRenderedFeatures(undefined, { layers: [layerId] });
    if (features.length === 0) continue;
    const heights: number[] = [];
    let untagged = 0;
    for (const f of features) {
      const h = Number(f.properties?.render_height);
      if (Number.isFinite(h) && h > 0) heights.push(h);
      else untagged += 1;
    }
    if (heights.length === 0 && untagged === 0) continue;
    return {
      count: features.length,
      tagged: heights.length,
      untagged,
      maxHeight: heights.length ? Math.max(...heights) : DEFAULT_BUILDING_HEIGHT_M,
      avgHeight: heights.length ? heights.reduce((a, b) => a + b, 0) / heights.length : DEFAULT_BUILDING_HEIGHT_M,
    };
  }
  return { count: 0, tagged: 0, untagged: 0, maxHeight: 0, avgHeight: 0 };
}

export default function MapLibre3DInner({
  originLat, originLon, markers, classColor, pickMode, onPick, onMarkerClick, selectedId, onBuildingStats,
}: {
  originLat: number;
  originLon: number;
  markers: MarkerPoint[];
  classColor: Record<GeoTarget['cls'], string>;
  isDark: boolean;
  pickMode: boolean;
  onPick: (lat: number, lon: number) => void;
  onMarkerClick: (id: number) => void;
  selectedId: number | null;
  onBuildingStats?: (stats: BuildingStats) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markerElsRef = useRef<Map<number, maplibregl.Marker>>(new Map());
  const originMarkerRef = useRef<maplibregl.Marker | null>(null);
  const pickModeRef = useRef(pickMode);
  const onPickRef = useRef(onPick);
  const onBuildingStatsRef = useRef(onBuildingStats);
  const onMapErrorRef = useRef<((msg: string) => void) | undefined>(undefined);
  
  
  
  
  const [webglMissing] = useState(() => !hasWebGL());
  const [mapError, setMapError] = useState<string | null>(null);
  
  
  
  
  
  
  useEffect(() => {
    pickModeRef.current = pickMode;
    onPickRef.current = onPick;
    onBuildingStatsRef.current = onBuildingStats;
    onMapErrorRef.current = setMapError;
  });

  useEffect(() => {
    if (!containerRef.current || mapRef.current || webglMissing) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: STYLE_URL,
      center: [originLon, originLat],
      
      
      
      
      
      
      zoom: 17,
      pitch: 60,
      bearing: -17,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-left');
    map.on('error', (e: { error?: { message?: string } }) => {
      onMapErrorRef.current?.(e?.error?.message || 'Map failed to load a resource.');
    });
    
    
    
    
    
    
    
    const ro = new ResizeObserver(() => map.resize());
    ro.observe(containerRef.current);
    map.once('load', () => map.resize());
    map.on('load', () => {
      
      
      
      
      
      
      
      const HEIGHT_EXPR: maplibregl.ExpressionSpecification =
        ['coalesce', ['get', 'render_height'], DEFAULT_BUILDING_HEIGHT_M];
      const BASE_EXPR: maplibregl.ExpressionSpecification =
        ['coalesce', ['get', 'render_min_height'], 0];
      const COLOR_EXPR: maplibregl.ExpressionSpecification = [
        'interpolate', ['linear'], HEIGHT_EXPR,
        0, '#55646f',
        30, '#7f929f',
        90, '#a9bdc9',
        200, '#d6e4ec',
      ];
      if (map.getLayer(BUILDING_LAYER_ID)) {
        map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-height', HEIGHT_EXPR);
        map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-base', BASE_EXPR);
        map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-color', COLOR_EXPR);
        map.setPaintProperty(BUILDING_LAYER_ID, 'fill-extrusion-opacity', 0.92);
      } else if (map.getSource('openmaptiles')) {
        
        map.addLayer({
          id: FALLBACK_BUILDING_LAYER_ID,
          source: 'openmaptiles',
          'source-layer': 'building',
          type: 'fill-extrusion',
          minzoom: 14,
          paint: {
            'fill-extrusion-color': COLOR_EXPR,
            'fill-extrusion-height': HEIGHT_EXPR,
            'fill-extrusion-base': BASE_EXPR,
            'fill-extrusion-opacity': 0.92,
          },
        });
      }
    });
    map.on('click', (e: maplibregl.MapMouseEvent) => {
      if (pickModeRef.current) onPickRef.current(e.lngLat.lat, e.lngLat.lng);
    });
    
    
    
    map.on('idle', () => {
      onBuildingStatsRef.current?.(queryBuildingStats(map));
    });
    mapRef.current = map;
    return () => { ro.disconnect(); map.remove(); mapRef.current = null; };
    
  }, []);

  
  useEffect(() => {
    mapRef.current?.flyTo({ center: [originLon, originLat], essential: true });
  }, [originLat, originLon]);

  
  useEffect(() => {
    if (!mapRef.current) return;
    if (!originMarkerRef.current) {
      const el = document.createElement('div');
      el.style.cssText = 'width:16px;height:16px;background:#10b981;border:2px solid white;transform:rotate(45deg);box-shadow:0 0 8px #10b981';
      originMarkerRef.current = new maplibregl.Marker({ element: el })
        .setLngLat([originLon, originLat]).addTo(mapRef.current);
    } else {
      originMarkerRef.current.setLngLat([originLon, originLat]);
    }
  }, [originLat, originLon]);

  
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    const seen = new Set<number>();
    for (const m of markers) {
      seen.add(m.id);
      const selected = m.id === selectedId;
      const size = selected ? 18 : Math.max(10, Math.min(20, 10 + m.altitude / 200));
      let marker = markerElsRef.current.get(m.id);
      if (!marker) {
        const el = document.createElement('div');
        el.style.cursor = 'pointer';
        el.addEventListener('click', (ev) => { ev.stopPropagation(); onMarkerClick(m.id); });
        marker = new maplibregl.Marker({ element: el }).setLngLat([m.lon, m.lat]).addTo(map);
        markerElsRef.current.set(m.id, marker);
      } else {
        marker.setLngLat([m.lon, m.lat]);
      }
      
      
      
      
      
      
      const el = marker.getElement();
      const color = classColor[m.cls];
      el.style.cursor = 'pointer';
      el.style.width = `${size}px`;
      el.style.height = `${size}px`;
      el.style.borderRadius = '50%';
      el.style.background = color;
      el.style.border = '2px solid rgba(255,255,255,0.9)';
      el.style.boxShadow = selected ? `0 0 0 3px rgba(255,255,255,0.9), 0 0 12px ${color}` : `0 0 6px ${color}`;
      el.title = `Track ${m.id} · ${m.cls} · ${m.altitude.toFixed(0)}m AGL`;
    }
    for (const [id, marker] of markerElsRef.current.entries()) {
      if (!seen.has(id)) { marker.remove(); markerElsRef.current.delete(id); }
    }
  }, [markers, selectedId, classColor, onMarkerClick]);

  if (webglMissing) {
    return (
      <div className="w-full h-[420px] rounded-xl border border-amber-800/40 bg-amber-950/10 flex items-center justify-center p-6">
        <p className="font-mono text-[11px] text-amber-400 text-center max-w-sm leading-relaxed">
          3D view needs WebGL, which this browser/session doesn&apos;t provide.
          The 2D map works without it — switch back to 2D.
        </p>
      </div>
    );
  }

  return (
    <div className={`w-full h-[420px] rounded-xl overflow-hidden border border-zinc-800/60 relative ${pickMode ? 'cursor-crosshair' : ''}`}>
      <div ref={containerRef} className="w-full h-full" />
      {mapError && (
        <div className="absolute bottom-2 left-2 right-2 px-2 py-1 rounded bg-red-950/80 border border-red-800/50">
          <p className="font-mono text-[10px] text-red-300 truncate">Map error: {mapError}</p>
        </div>
      )}
    </div>
  );
}
