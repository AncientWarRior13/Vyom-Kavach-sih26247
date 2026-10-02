'use client';






import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { GeoTarget } from './GeoMapPanel';

interface MarkerPoint extends GeoTarget {
  lat: number;
  lon: number;
}

function dotIcon(color: string, selected: boolean) {
  const ring = selected ? `0 0 0 3px rgba(255,255,255,0.9), 0 0 10px ${color}` : `0 0 6px ${color}`;
  const size = selected ? 16 : 12;
  return L.divIcon({
    className: '',
    html: `<div style="width:${size}px;height:${size}px;border-radius:50%;background:${color};border:2px solid rgba(255,255,255,0.85);box-shadow:${ring}"></div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function originIcon() {
  return L.divIcon({
    className: '',
    html: `<div style="width:14px;height:14px;border-radius:3px;background:#10b981;border:2px solid white;transform:rotate(45deg)"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
}

function Recenter({ lat, lon }: { lat: number; lon: number }) {
  const map = useMap();
  useEffect(() => { map.setView([lat, lon], map.getZoom()); }, [lat, lon, map]);
  return null;
}

function ClickHandler({ pickMode, onPick }: { pickMode: boolean; onPick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      if (pickMode) onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function LeafletMapInner({
  originLat, originLon, markers, classColor, isDark, pickMode, onPick, onMarkerClick, selectedId,
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
}) {
  return (
    <div className={`w-full h-[420px] rounded-xl overflow-hidden border border-zinc-800/60 ${pickMode ? 'cursor-crosshair' : ''}`}>
      <MapContainer center={[originLat, originLon]} zoom={12} scrollWheelZoom={true}
        style={{ width: '100%', height: '100%', background: isDark ? '#0a0f10' : '#e5e7eb' }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Recenter lat={originLat} lon={originLon} />
        <ClickHandler pickMode={pickMode} onPick={onPick} />
        <Marker position={[originLat, originLon]} icon={originIcon()}>
          <Popup>Defended asset (origin)</Popup>
        </Marker>
        {markers.map(m => (
          <Marker
            key={m.id}
            position={[m.lat, m.lon]}
            icon={dotIcon(classColor[m.cls], m.id === selectedId)}
            eventHandlers={{ click: () => onMarkerClick(m.id) }}
          >
            <Popup>
              Track {m.id} &middot; {m.subtype} ({m.cls})<br />
              {m.lat.toFixed(6)}, {m.lon.toFixed(6)}<br />
              elevation {m.altitude.toFixed(0)} m AGL
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
