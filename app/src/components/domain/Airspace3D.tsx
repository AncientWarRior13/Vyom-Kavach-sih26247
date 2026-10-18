import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import type { Contact } from '../../api/types';

export interface Airspace3DProps {
  contacts: Contact[];
  selectedContactId?: string | null;
  onSelectContact?: (contactId: string) => void;
  maxRangeKm?: number;
  sweepSpeedRpm?: number;
  defendedAsset?: [number, number];
  noEngageZones?: Array<{ name?: string; center: [number, number]; radius_m: number }>;
  className?: string;
  theme?: 'dark' | 'light';
}

const CLASS_COLORS: Record<string, string> = {
  drone: '#EF4444',      // Hostile red
  quadcopter: '#EF4444',
  fixed_wing: '#F97316', // Orange
  commercial: '#3B82F6', // Blue / civil
  bird: '#EAB308',       // Yellow fauna
  clutter: '#8B5CF6',    // Purple anomaly/clutter
  unknown: '#9CA3AF',    // Gray unknown
};

export const Airspace3D: React.FC<Airspace3DProps> = ({
  contacts,
  selectedContactId,
  onSelectContact,
  maxRangeKm = 6,
  sweepSpeedRpm = 15,
  defendedAsset,
  noEngageZones,
  className = '',
  theme = 'dark',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const reqIdRef = useRef<number | null>(null);

  // Interaction controls
  const isDraggingRef = useRef(false);
  const prevMouseRef = useRef({ x: 0, y: 0 });
  const sphericalRef = useRef({ radius: 75, theta: 0.8, phi: 1.05 }); // Orbit camera coords
  const targetLookAt = useRef(new THREE.Vector3(0, 4, 0));

  // Visual refs
  const sweepAngleRef = useRef(0);
  const sweepMeshRef = useRef<THREE.Group | null>(null);
  const radarDishRef = useRef<THREE.Group | null>(null);
  const contactsGroupRef = useRef<THREE.Group | null>(null);
  const zonesGroupRef = useRef<THREE.Group | null>(null);

  // Map contacts to dynamic 3D objects
  const targetMeshesRef = useRef<Map<string, THREE.Group>>(new Map());

  // Mouse Raycasting for selecting targets
  const mouseRef = useRef(new THREE.Vector2());
  const raycasterRef = useRef(new THREE.Raycaster());

  // Convert real world coords (km) into 3D world units (scale = 35 / maxRangeKm)
  const worldScale = useMemo(() => 35 / (maxRangeKm || 6), [maxRangeKm]);

  // Setup Three.js Scene
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // 1. Scene & Background Fog
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    const bgColor = theme === 'dark' ? 0x070b10 : 0xf1f5f9;
    scene.background = new THREE.Color(bgColor);
    scene.fog = new THREE.FogExp2(bgColor, 0.007);

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.5, 500);
    cameraRef.current = camera;
    updateCameraPosition();

    // 3. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = false;
    container.replaceChildren(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, theme === 'dark' ? 0.75 : 1.1);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x73ffdf, theme === 'dark' ? 1.6 : 1.2);
    dirLight.position.set(30, 60, 20);
    scene.add(dirLight);

    const radarPointLight = new THREE.PointLight(0x00e87a, 12, 45);
    radarPointLight.position.set(0, 5, 0);
    scene.add(radarPointLight);

    // 5. Tactical Ground Radar Floor & Range Rings
    const groundGroup = new THREE.Group();
    scene.add(groundGroup);

    // Concentric Circular Rings
    const ringRadii = [7, 14, 21, 28, 35];
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x00e87a,
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide,
    });

    ringRadii.forEach((rad) => {
      const ringGeo = new THREE.RingGeometry(rad - 0.05, rad + 0.05, 96);
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = -Math.PI / 2;
      ringMesh.position.y = 0.02;
      groundGroup.add(ringMesh);
    });

    // Radial Azimuth Spokes (every 30 deg)
    const spokeMat = new THREE.MeshBasicMaterial({
      color: 0x00e87a,
      transparent: true,
      opacity: 0.16,
    });
    for (let deg = 0; deg < 360; deg += 30) {
      const rad = (deg * Math.PI) / 180;
      const spokeGeo = new THREE.BoxGeometry(0.04, 0.01, 70);
      const spoke = new THREE.Mesh(spokeGeo, spokeMat);
      spoke.rotation.y = rad;
      spoke.position.y = 0.015;
      groundGroup.add(spoke);
    }

    // Grid wire helper
    const gridHelper = new THREE.GridHelper(70, 20, 0x00e87a, 0x16302b);
    gridHelper.position.y = 0.01;
    (gridHelper.material as THREE.Material).transparent = true;
    (gridHelper.material as THREE.Material).opacity = 0.2;
    groundGroup.add(gridHelper);

    // 6. Central Radar Tower with Rotating Dish
    const towerGroup = new THREE.Group();
    const towerGeo = new THREE.CylinderGeometry(0.4, 0.7, 3.2, 12);
    const towerMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.3 });
    const towerMesh = new THREE.Mesh(towerGeo, towerMat);
    towerMesh.position.y = 1.6;
    towerGroup.add(towerMesh);

    // Rotating Radar Dish Assembly
    const dishAssembly = new THREE.Group();
    dishAssembly.position.y = 3.3;
    const dishGeo = new THREE.SphereGeometry(0.8, 20, 12, 0, Math.PI);
    const dishMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.9,
      roughness: 0.2,
      emissive: 0x064e3b,
      emissiveIntensity: 0.6,
    });
    const dishMesh = new THREE.Mesh(dishGeo, dishMat);
    dishMesh.rotation.z = Math.PI / 2;
    dishAssembly.add(dishMesh);

    // Horn feed
    const hornGeo = new THREE.CylinderGeometry(0.08, 0.14, 0.6, 8);
    const hornMesh = new THREE.Mesh(hornGeo, new THREE.MeshBasicMaterial({ color: 0x00e87a }));
    hornMesh.position.set(0, 0, 0.5);
    hornMesh.rotation.x = Math.PI / 2;
    dishAssembly.add(hornMesh);

    towerGroup.add(dishAssembly);
    radarDishRef.current = dishAssembly;
    scene.add(towerGroup);

    // 7. 3D Volumetric Radar Sweep Fan
    const sweepFanGroup = new THREE.Group();
    const sweepSegments = 32;
    const sweepRange = 35;
    const sweepArc = Math.PI / 8; // 22.5 deg fan
    const sweepPositions = [0, 0.08, 0];
    const sweepIndices: number[] = [];

    for (let i = 0; i <= sweepSegments; i++) {
      const theta = -sweepArc + (sweepArc * i) / sweepSegments;
      sweepPositions.push(Math.sin(theta) * sweepRange, 0.08, Math.cos(theta) * sweepRange);
    }
    for (let i = 0; i < sweepSegments; i++) {
      sweepIndices.push(0, i + 1, i + 2);
    }

    const sweepGeo = new THREE.BufferGeometry();
    sweepGeo.setAttribute('position', new THREE.Float32BufferAttribute(sweepPositions, 3));
    sweepGeo.setIndex(sweepIndices);

    const sweepMat = new THREE.MeshBasicMaterial({
      color: 0x00e87a,
      transparent: true,
      opacity: 0.22,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const sweepMesh = new THREE.Mesh(sweepGeo, sweepMat);
    sweepFanGroup.add(sweepMesh);

    // Leading Sweep Laser Line
    const leadLineGeo = new THREE.BoxGeometry(0.08, 0.02, sweepRange);
    const leadLineMat = new THREE.MeshBasicMaterial({ color: 0x86efac, transparent: true, opacity: 0.85 });
    const leadLine = new THREE.Mesh(leadLineGeo, leadLineMat);
    leadLine.position.set(0, 0.09, sweepRange / 2);
    sweepFanGroup.add(leadLine);

    sweepMeshRef.current = sweepFanGroup;
    scene.add(sweepFanGroup);

    // 8. Contact & Zone Containers
    const cGroup = new THREE.Group();
    contactsGroupRef.current = cGroup;
    scene.add(cGroup);

    const zGroup = new THREE.Group();
    zonesGroupRef.current = zGroup;
    scene.add(zGroup);

    // Resize Handler
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Animation Loop
    let lastTime = performance.now();
    const animate = (time: number) => {
      reqIdRef.current = requestAnimationFrame(animate);
      const dt = (time - lastTime) / 1000;
      lastTime = time;

      // Rotate Sweep (rpm converted to rad/s)
      const sweepRadSpeed = (sweepSpeedRpm * 2 * Math.PI) / 60;
      sweepAngleRef.current += sweepRadSpeed * dt;
      if (sweepMeshRef.current) {
        sweepMeshRef.current.rotation.y = sweepAngleRef.current;
      }
      if (radarDishRef.current) {
        radarDishRef.current.rotation.y = sweepAngleRef.current;
      }

      // Smooth camera interpolation towards spherical coords
      updateCameraPosition();

      renderer.render(scene, camera);
    };

    reqIdRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
      renderer.dispose();
      scene.clear();
    };
  }, [theme, sweepSpeedRpm]);

  // Update Orbit Camera based on spherical coords
  const updateCameraPosition = () => {
    if (!cameraRef.current) return;
    const { radius, theta, phi } = sphericalRef.current;
    const x = radius * Math.sin(phi) * Math.sin(theta);
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.cos(theta);

    cameraRef.current.position.set(x, Math.max(y, 4), z);
    cameraRef.current.lookAt(targetLookAt.current);
  };

  // Render No-Engage Zones and Defended Asset
  useEffect(() => {
    const zGroup = zonesGroupRef.current;
    if (!zGroup) return;
    zGroup.clear();

    // Defended Base Asset
    if (defendedAsset) {
      const [ax, ay] = defendedAsset;
      const x = (ax / 1000) * worldScale;
      const z = -(ay / 1000) * worldScale;

      const baseGeo = new THREE.CylinderGeometry(0.8, 1.2, 0.6, 6);
      const baseMat = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        emissive: 0x0284c7,
        emissiveIntensity: 0.5,
      });
      const baseMesh = new THREE.Mesh(baseGeo, baseMat);
      baseMesh.position.set(x, 0.3, z);
      zGroup.add(baseMesh);

      // Defended Asset Ground Perimeter
      const ringGeo = new THREE.RingGeometry(1.6, 1.8, 32);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = -Math.PI / 2;
      ringMesh.position.set(x, 0.05, z);
      zGroup.add(ringMesh);
    }

    // No Engage Zones (Cylindrical volumetric domes)
    if (noEngageZones) {
      noEngageZones.forEach((zone) => {
        const [zx, zy] = zone.center;
        const x = (zx / 1000) * worldScale;
        const z = -(zy / 1000) * worldScale;
        const radWorld = ((zone.radius_m || 500) / 1000) * worldScale;

        const zoneGeo = new THREE.CylinderGeometry(radWorld, radWorld, 8, 32, 1, true);
        const zoneMat = new THREE.MeshBasicMaterial({
          color: 0xa855f7,
          transparent: true,
          opacity: 0.15,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        const mesh = new THREE.Mesh(zoneGeo, zoneMat);
        mesh.position.set(x, 4, z);
        zGroup.add(mesh);

        // Ground ring
        const borderGeo = new THREE.RingGeometry(radWorld - 0.06, radWorld + 0.06, 64);
        const borderMat = new THREE.MeshBasicMaterial({ color: 0xc084fc, transparent: true, opacity: 0.6, side: THREE.DoubleSide });
        const borderMesh = new THREE.Mesh(borderGeo, borderMat);
        borderMesh.rotation.x = -Math.PI / 2;
        borderMesh.position.set(x, 0.04, z);
        zGroup.add(borderMesh);
      });
    }
  }, [defendedAsset, noEngageZones, worldScale]);

  // Sync Contacts with 3D Models & Procedural Meshes
  useEffect(() => {
    const cGroup = contactsGroupRef.current;
    if (!cGroup) return;

    const currentIds = new Set<string>();

    contacts.forEach((c) => {
      currentIds.add(c.id);
      const isSelected = c.id === selectedContactId;
      const cls = (c.assigned_classification || c.classification || 'unknown').toLowerCase();
      const colorHex = CLASS_COLORS[cls] || CLASS_COLORS.unknown;

      // Position in 3D: (range, bearing) -> (x, z)
      const rangeKm = c.range_km ?? 0;
      const bearingRad = ((c.bearing_deg ?? 0) * Math.PI) / 180;
      const x = Math.sin(bearingRad) * rangeKm * worldScale;
      const z = -Math.cos(bearingRad) * rangeKm * worldScale;
      // Altitude in meters -> scaled Y
      const altM = c.altitude ?? 150;
      const y = Math.max(0.6, (altM / 100) * 1.5);

      let group = targetMeshesRef.current.get(c.id);

      if (!group) {
        group = new THREE.Group();
        group.userData = { contactId: c.id };

        // 1. Ground Projection Shadow Ring with pulsing glow
        const groundRingGeo = new THREE.RingGeometry(0.5, 0.8, 32);
        const groundRingMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(colorHex),
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
        });
        const groundRing = new THREE.Mesh(groundRingGeo, groundRingMat);
        groundRing.rotation.x = -Math.PI / 2;
        groundRing.position.y = 0.05;
        groundRing.name = 'groundRing';
        groundRing.userData = { contactId: c.id };
        group.add(groundRing);

        // 2. Altitude Stem Line (connects airborne model to ground ring)
        const stemGeo = new THREE.CylinderGeometry(0.04, 0.04, 1, 6);
        const stemMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(colorHex),
          transparent: true,
          opacity: 0.6,
        });
        const stemMesh = new THREE.Mesh(stemGeo, stemMat);
        stemMesh.name = 'stem';
        stemMesh.userData = { contactId: c.id };
        group.add(stemMesh);

        // 3. Glowing Tactical Marker Beacon on the target
        const beaconGeo = new THREE.SphereGeometry(0.45, 16, 16);
        const beaconMat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(colorHex),
          emissive: new THREE.Color(colorHex),
          emissiveIntensity: 1.5,
          roughness: 0.1,
        });
        const beacon = new THREE.Mesh(beaconGeo, beaconMat);
        beacon.name = 'beacon';
        beacon.userData = { contactId: c.id };
        group.add(beacon);

        // 4. Large Invisible Click Target Hitbox (radius: 2.2 units)
        const hitBoxGeo = new THREE.SphereGeometry(2.2, 8, 8);
        const hitBoxMat = new THREE.MeshBasicMaterial({ visible: false });
        const hitBox = new THREE.Mesh(hitBoxGeo, hitBoxMat);
        hitBox.name = 'hitbox';
        hitBox.userData = { contactId: c.id };
        group.add(hitBox);

        // 5. Procedural 3D Target Model Head
        const modelHead = createProceduralDroneMesh(cls, colorHex);
        modelHead.name = 'modelHead';
        modelHead.scale.set(1.4, 1.4, 1.4); // scaled up for visibility
        group.add(modelHead);

        // 6. Selection Halo / Box
        const haloGeo = new THREE.RingGeometry(1.2, 1.5, 32);
        const haloMat = new THREE.MeshBasicMaterial({
          color: 0x00e87a,
          transparent: true,
          opacity: 0.95,
          side: THREE.DoubleSide,
        });
        const halo = new THREE.Mesh(haloGeo, haloMat);
        halo.rotation.x = -Math.PI / 2;
        halo.position.y = 0.08;
        halo.name = 'halo';
        halo.visible = false;
        group.add(halo);

        cGroup.add(group);
        targetMeshesRef.current.set(c.id, group);
      }

      // Update positions
      group.position.set(x, 0, z);

      // Update Beacon & Hitbox heights
      const beacon = group.getObjectByName('beacon') as THREE.Mesh;
      if (beacon) beacon.position.set(0, y, 0);

      const hitBox = group.getObjectByName('hitbox') as THREE.Mesh;
      if (hitBox) hitBox.position.set(0, y, 0);

      // Update Stem length and center
      const stem = group.getObjectByName('stem') as THREE.Mesh;
      if (stem) {
        stem.scale.set(1, y, 1);
        stem.position.set(0, y / 2, 0);
      }

      // Update Target Head Position & Rotation
      const head = group.getObjectByName('modelHead') as THREE.Group;
      if (head) {
        head.position.set(0, y, 0);
        // Turn towards center if attacking, or heading
        const headingRad = (((c.heading ?? c.bearing_deg) ?? 180) * Math.PI) / 180;
        head.rotation.y = headingRad;
      }

      // Update Selection Halo
      const halo = group.getObjectByName('halo') as THREE.Mesh;
      if (halo) {
        halo.visible = isSelected;
      }
    });

    // Cleanup contacts that disappeared
    targetMeshesRef.current.forEach((mesh, id) => {
      if (!currentIds.has(id)) {
        cGroup.remove(mesh);
        targetMeshesRef.current.delete(id);
      }
    });
  }, [contacts, selectedContactId, worldScale]);

  // Helper: Procedural 3D Drone / Target Mesh Generator
  const createProceduralDroneMesh = (type: string, colorHex: string): THREE.Group => {
    const group = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(colorHex),
      emissive: new THREE.Color(colorHex),
      emissiveIntensity: 0.35,
      metalness: 0.7,
      roughness: 0.3,
    });

    if (type.includes('fixed') || type === 'aircraft') {
      // Sleek delta-wing drone
      const fuselage = new THREE.Mesh(new THREE.ConeGeometry(0.2, 1.2, 8), mat);
      fuselage.rotation.x = Math.PI / 2;
      group.add(fuselage);

      const wing = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.03, 0.4), mat);
      wing.position.set(0, 0, -0.1);
      group.add(wing);

      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.4, 0.25), mat);
      fin.position.set(0, 0.2, -0.4);
      group.add(fin);
    } else if (type.includes('bird')) {
      // Avian / bird profile
      const body = new THREE.Mesh(new THREE.SphereGeometry(0.2, 8, 8), mat);
      group.add(body);
      const leftWing = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.02, 0.2), mat);
      leftWing.position.set(-0.4, 0.05, 0);
      leftWing.rotation.z = -0.2;
      group.add(leftWing);
      const rightWing = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.02, 0.2), mat);
      rightWing.position.set(0.4, 0.05, 0);
      rightWing.rotation.z = 0.2;
      group.add(rightWing);
    } else {
      // Quadcopter Drone (central body + 4 arms + 4 rotor discs)
      const hub = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.12, 0.35), mat);
      group.add(hub);

      // Cross arms
      const arm1 = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.04, 0.04), mat);
      arm1.rotation.y = Math.PI / 4;
      const arm2 = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.04, 0.04), mat);
      arm2.rotation.y = -Math.PI / 4;
      group.add(arm1, arm2);

      // 4 Rotors
      const rotorMat = new THREE.MeshBasicMaterial({ color: 0x93c5fd, transparent: true, opacity: 0.65 });
      const rotorOffset = 0.38;
      const rotorPositions = [
        [rotorOffset, 0.08, rotorOffset],
        [-rotorOffset, 0.08, rotorOffset],
        [rotorOffset, 0.08, -rotorOffset],
        [-rotorOffset, 0.08, -rotorOffset],
      ];
      rotorPositions.forEach(([rx, ry, rz]) => {
        const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.02, 12), rotorMat);
        rotor.position.set(rx, ry, rz);
        group.add(rotor);
      });
    }

    return group;
  };

  const dragDistanceRef = useRef(0);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = true;
    dragDistanceRef.current = 0;
    prevMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - prevMouseRef.current.x;
    const dy = e.clientY - prevMouseRef.current.y;
    dragDistanceRef.current += Math.abs(dx) + Math.abs(dy);
    prevMouseRef.current = { x: e.clientX, y: e.clientY };

    // Update spherical camera coordinates
    sphericalRef.current.theta -= dx * 0.008;
    sphericalRef.current.phi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, sphericalRef.current.phi - dy * 0.008));
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = false;

    // Only pick if user clicked without dragging
    if (dragDistanceRef.current > 6) return;

    if (!containerRef.current || !cameraRef.current || !sceneRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const meshes: THREE.Object3D[] = [];
    targetMeshesRef.current.forEach((grp) => {
      grp.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) meshes.push(child);
      });
    });

    const intersects = raycasterRef.current.intersectObjects(meshes, false);
    if (intersects.length > 0) {
      let curr: THREE.Object3D | null = intersects[0].object;
      while (curr && !curr.userData.contactId) {
        curr = curr.parent;
      }
      if (curr && curr.userData.contactId && onSelectContact) {
        onSelectContact(curr.userData.contactId);
      }
    }
  };

  // Zoom helper
  const adjustZoom = (delta: number) => {
    sphericalRef.current.radius = Math.max(15, Math.min(180, sphericalRef.current.radius + delta));
  };

  const resetView = () => {
    sphericalRef.current = { radius: 75, theta: 0.8, phi: 1.05 };
    targetLookAt.current.set(0, 4, 0);
  };

  // Bind native non-passive wheel event to container
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const zoomFactor = e.deltaY > 0 ? 6 : -6;
      adjustZoom(zoomFactor);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      className={`relative w-full h-full overflow-hidden select-none cursor-grab active:cursor-grabbing ${className}`}
    >
      {/* 3D Viewport HUD Overlay */}
      <div className="absolute top-3 left-3 pointer-events-none font-mono text-[10px] space-y-1 bg-black/60 backdrop-blur-md p-2 rounded-xs border border-white/10 text-[var(--c-text-mid)] z-10">
        <div className="text-[var(--c-accent)] font-bold uppercase tracking-wider flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--c-accent)] animate-pulse" />
          VOLUMETRIC AIRSPACE PERSPECTIVE (3D)
        </div>
        <div>ORBIT: Drag Left Mouse · ZOOM: Scroll Wheel or +/- buttons</div>
        <div>ELEVATION STEMS: Real-time Altitude (m AGL)</div>
      </div>

      {/* On-screen Zoom & Reset Controls */}
      <div className="absolute bottom-4 right-4 z-20 flex items-center gap-1.5 bg-black/70 backdrop-blur-md border border-white/15 p-1 rounded-xs font-mono text-xs text-white">
        <button
          onClick={() => adjustZoom(-8)}
          title="Zoom In"
          className="w-7 h-7 flex items-center justify-center bg-white/10 hover:bg-white/20 rounded-xs text-white font-bold cursor-pointer transition-colors"
        >
          +
        </button>
        <button
          onClick={() => adjustZoom(8)}
          title="Zoom Out"
          className="w-7 h-7 flex items-center justify-center bg-white/10 hover:bg-white/20 rounded-xs text-white font-bold cursor-pointer transition-colors"
        >
          −
        </button>
        <button
          onClick={resetView}
          title="Reset Camera View"
          className="px-2 h-7 flex items-center justify-center bg-white/10 hover:bg-white/20 rounded-xs text-[10px] text-[var(--c-accent)] cursor-pointer transition-colors"
        >
          RESET
        </button>
      </div>
    </div>
  );
};
