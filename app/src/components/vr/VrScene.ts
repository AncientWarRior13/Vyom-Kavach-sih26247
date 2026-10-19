import * as THREE from 'three';
import type { Contact, Classification } from '../../api/types';
import { posAt, RawContact } from '../../utils/scenarioUtils';
import { colors } from '../../tokens';
import { vrAudio } from './VrAudio';

export const ALT_EXAG = 3;
export const WORLD_SCALE = 0.01; // 1 unit = 100 meters
export const EYE_HEIGHT = 1.6;   // Camera at defended asset ground post

export type VrViewMode = 'desktop' | 'splitscreen' | 'immersive';

export interface VrSceneOptions {
  onSelectContact?: (id: string | null) => void;
  onLockProgress?: (progress: number) => void; // 0 to 1 for gaze dwell ring
}

interface TargetMeshEntry {
  id: string;
  group: THREE.Group;
  shapeMesh: THREE.Mesh;
  dropLine: THREE.Line;
  dropGroundDisc: THREE.Mesh;
  billboardSprite: THREE.Sprite;
  trailLine: THREE.Line;
  reticle: THREE.Group;
  assigned: string;
  rawContact: any;
  fading: boolean;
  opacity: number;
}

export class VrScene {
  private container: HTMLDivElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private stereoCamera: THREE.StereoCamera;
  private renderer: THREE.WebGLRenderer;
  private rig: THREE.Group;

  // View & Mode State
  private viewMode: VrViewMode = 'desktop';
  private xrSession: any = null;
  private isDestroyed: boolean = false;

  // Desktop Controls State
  private yaw: number = 0;
  private pitch: number = 10;
  private isDragging: boolean = false;
  private dragStartX: number = 0;
  private dragStartY: number = 0;
  private dragStartYaw: number = 0;
  private dragStartPitch: number = 0;
  private targetYaw: number = 0;
  private targetPitch: number = 10;

  // Gyro Controls State
  private hasGyro: boolean = false;
  private gyroAlphaOffset: number = 0;
  private gyroQuaternion: THREE.Quaternion = new THREE.Quaternion();
  private useGyro: boolean = false;

  // Gaze Selection in Split-Screen Mode
  private gazeTargetId: string | null = null;
  private gazeDwellTime: number = 0;
  private gazeRaycaster: THREE.Raycaster = new THREE.Raycaster();

  // WebXR Controllers & HUD
  private controller0: any = null;
  private controller1: any = null;
  private controllerGrip0: any = null;
  private controllerGrip1: any = null;
  private xrRaycaster: THREE.Raycaster = new THREE.Raycaster();
  private wristPanelSprite: THREE.Sprite | null = null;
  private floatingInfoSprite: THREE.Sprite | null = null;
  private snapTurnCooloff: number = 0;

  private scenario: any = null;
  private selectedContactId: string | null = null;
  private targets: Map<string, TargetMeshEntry> = new Map();
  private destroyedIds: Set<string> = new Set();
  private noEngageGroup: THREE.Group;
  private roeGroup: THREE.Group;
  private skyMesh: THREE.Mesh | null = null;
  private starsPoints: THREE.Points | null = null;
  private domeWireframe: THREE.Group;
  private groundGroup: THREE.Group;

  // Engagement Visual Effects
  private fxGroup: THREE.Group;
  private activeHardKillEffects: any[] = [];
  private activeSoftKillEffects: Array<{ cone: THREE.Mesh; life: number }> = [];
  private turretGroup: THREE.Group | null = null;
  private turretHead: THREE.Group | null = null;

  // Callbacks
  private onSelectContact?: (id: string | null) => void;
  private onLockProgress?: (progress: number) => void;

  // Shared Geometries & Materials (Memory Optimization)
  private sharedGeometries = {
    unknown: new THREE.OctahedronGeometry(0.38),
    hostile: new THREE.ConeGeometry(0.38, 0.75, 8),
    friendly: new THREE.SphereGeometry(0.35, 12, 12),
    decoy: new THREE.BoxGeometry(0.5, 0.5, 0.5),
    bird: new THREE.TorusGeometry(0.32, 0.12, 8, 16),
    dropDisc: new THREE.RingGeometry(0.05, 0.45, 16),
  };

  private sharedMaterials = {
    unknown: new THREE.MeshBasicMaterial({ color: colors.contact.unknown, wireframe: false }),
    hostile: new THREE.MeshBasicMaterial({ color: colors.contact.hostile, wireframe: false }),
    friendly: new THREE.MeshBasicMaterial({ color: colors.contact.friendly, wireframe: false }),
    decoy: new THREE.MeshBasicMaterial({ color: colors.contact.decoy, wireframe: false }),
    bird: new THREE.MeshBasicMaterial({ color: colors.contact.neutral, wireframe: false }),
    dropLine: new THREE.LineBasicMaterial({ color: 0x4e718a, transparent: true, opacity: 0.4 }),
    trailLine: new THREE.LineBasicMaterial({ color: 0x00e87a, transparent: true, opacity: 0.6 }),
    reticle: new THREE.LineBasicMaterial({ color: 0x00e87a, linewidth: 2 }),
    laserBeam: new THREE.LineBasicMaterial({ color: 0x00e87a, linewidth: 3 }),
    jamCone: new THREE.MeshBasicMaterial({ color: 0xbf5af2, transparent: true, opacity: 0.35, side: THREE.DoubleSide }),
  };

  private lastTime: number = performance.now();

  constructor(container: HTMLDivElement, options?: VrSceneOptions) {
    this.container = container;
    this.onSelectContact = options?.onSelectContact;
    this.onLockProgress = options?.onLockProgress;

    // 1. Scene & Rig
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x020508);

    this.rig = new THREE.Group();
    this.scene.add(this.rig);

    // 2. Camera (Defended Asset eye height 1.6m)
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    this.camera = new THREE.PerspectiveCamera(70, w / h, 0.1, 1000);
    this.camera.position.set(0, EYE_HEIGHT, 0);
    this.rig.add(this.camera);

    this.stereoCamera = new THREE.StereoCamera();
    this.stereoCamera.aspect = 0.5;
    this.stereoCamera.eyeSep = 0.064; // Average human pupillary distance in meters

    // 3. Renderer with WebXR support
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.xr.enabled = true;
    container.appendChild(this.renderer.domElement);

    // 4. Feature Groups
    this.domeWireframe = new THREE.Group();
    this.scene.add(this.domeWireframe);

    this.groundGroup = new THREE.Group();
    this.scene.add(this.groundGroup);

    this.roeGroup = new THREE.Group();
    this.scene.add(this.roeGroup);

    this.noEngageGroup = new THREE.Group();
    this.scene.add(this.noEngageGroup);

    this.fxGroup = new THREE.Group();
    this.scene.add(this.fxGroup);

    // 5. Initialize Components
    this.initSkyAndStars();
    this.initDomeWireframe();
    this.initGroundAndRangeRings();
    this.initWebXrControllers();
    this.initInputListeners();

    // 6. Start Render Loop
    this.renderer.setAnimationLoop(this.animate.bind(this));
  }

  // --- Sky Dome & Stars Shader ---
  private initSkyAndStars() {
    const skyGeo = new THREE.SphereGeometry(50, 24, 24);
    const skyMat = new THREE.ShaderMaterial({
      uniforms: {
        topColor: { value: new THREE.Color(0x020508) },
        bottomColor: { value: new THREE.Color(0x081628) },
        offset: { value: 1.6 },
        exponent: { value: 0.8 },
      },
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPosition = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPosition.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 topColor;
        uniform vec3 bottomColor;
        uniform float offset;
        uniform float exponent;
        varying vec3 vWorldPosition;
        void main() {
          float h = normalize(vWorldPosition + offset).y;
          gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
        }
      `,
      side: THREE.BackSide,
      depthWrite: false,
    });

    this.skyMesh = new THREE.Mesh(skyGeo, skyMat);
    this.scene.add(this.skyMesh);

    // Stars on upper dome
    const starCount = 1200;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 0.95 + 0.05); // upper dome
      const r = 48 + Math.random() * 1.5;
      starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = r * Math.cos(phi) + EYE_HEIGHT;
      starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 1.2,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: false,
    });
    this.starsPoints = new THREE.Points(starGeo, starMat);
    this.scene.add(this.starsPoints);
  }

  // --- Wireframe Hemisphere: Elevation rings every 15°, Azimuth meridians every 30° ---
  private initDomeWireframe() {
    const domeRadius = 35; // 3.5km coverage
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00e87a,
      transparent: true,
      opacity: 0.16,
    });

    // Elevation rings: 15°, 30°, 45°, 60°, 75°
    for (let deg = 15; deg <= 75; deg += 15) {
      const rad = (deg * Math.PI) / 180;
      const r = domeRadius * Math.cos(rad);
      const y = EYE_HEIGHT + domeRadius * Math.sin(rad);

      const ringGeo = new THREE.BufferGeometry();
      const points: number[] = [];
      const segments = 64;
      for (let i = 0; i <= segments; i++) {
        const theta = (i * 2 * Math.PI) / segments;
        points.push(r * Math.sin(theta), y, r * Math.cos(theta));
      }
      ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
      const ring = new THREE.LineLoop(ringGeo, lineMat);
      this.domeWireframe.add(ring);
    }

    // Azimuth meridians every 30° (0° to 330°)
    for (let deg = 0; deg < 360; deg += 30) {
      const azRad = (deg * Math.PI) / 180;
      const points: number[] = [];
      const steps = 32;
      for (let i = 0; i <= steps; i++) {
        const elRad = (i * Math.PI) / (steps * 2); // 0 to 90 deg
        const r = domeRadius * Math.cos(elRad);
        const y = EYE_HEIGHT + domeRadius * Math.sin(elRad);
        points.push(r * Math.sin(azRad), y, -r * Math.cos(azRad));
      }
      const meridianGeo = new THREE.BufferGeometry();
      meridianGeo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
      const meridian = new THREE.Line(meridianGeo, lineMat);
      this.domeWireframe.add(meridian);
    }
  }

  // --- Ground Grid & Range Rings ---
  private initGroundAndRangeRings() {
    // Ground plane grid
    const grid = new THREE.GridHelper(70, 70, 0x1e3c64, 0x0d1e33);
    grid.position.y = 0;
    this.groundGroup.add(grid);

    // Range rings every 500m to 2.5km (5, 10, 15, 20, 25 world units)
    const ringMat = new THREE.LineBasicMaterial({ color: 0x00e87a, transparent: true, opacity: 0.35 });
    for (let rMeters = 500; rMeters <= 2500; rMeters += 500) {
      const r = rMeters * WORLD_SCALE;
      const points: number[] = [];
      for (let i = 0; i <= 64; i++) {
        const theta = (i * 2 * Math.PI) / 64;
        points.push(r * Math.sin(theta), 0.02, r * Math.cos(theta));
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
      const ring = new THREE.LineLoop(geo, ringMat);
      this.groundGroup.add(ring);

      // Label sprite on ring at North
      const label = this.createLabelSprite(`${rMeters}m`, '#00e87a');
      label.position.set(0, 0.4, -r);
      this.groundGroup.add(label);
    }

    // N / E / S / W Cardinal Markers
    const cardinals = [
      { text: 'NORTH [000°]', pos: [0, 0.5, -28] as [number, number, number], color: '#00e87a' },
      { text: 'EAST [090°]', pos: [28, 0.5, 0] as [number, number, number], color: '#38bdf8' },
      { text: 'SOUTH [180°]', pos: [0, 0.5, 28] as [number, number, number], color: '#38bdf8' },
      { text: 'WEST [270°]', pos: [-28, 0.5, 0] as [number, number, number], color: '#38bdf8' },
    ];
    for (const c of cardinals) {
      const sprite = this.createLabelSprite(c.text, c.color, 12);
      sprite.position.set(...c.pos);
      this.groundGroup.add(sprite);
    }

    // High Energy Laser Counter-Drone Battery on Ground
    this.initLaserDefenseTurret();
  }

  // --- Counter-UAS High Energy Laser (HEL) Defense Battery ---
  private initLaserDefenseTurret() {
    this.turretGroup = new THREE.Group();
    this.turretGroup.position.set(2.4, 0, 1.8);

    // 1. Armored Hexagonal Pedestal
    const baseGeo = new THREE.CylinderGeometry(0.7, 0.9, 0.35, 8);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0x1a2634 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.y = 0.175;
    this.turretGroup.add(base);

    // Neon Accent Ring
    const ringGeo = new THREE.RingGeometry(0.72, 0.78, 16);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.02;
    this.turretGroup.add(ring);

    // 2. Swivel Head
    this.turretHead = new THREE.Group();
    this.turretHead.position.set(0, 0.45, 0);

    // Turret Body Box
    const bodyGeo = new THREE.BoxGeometry(0.45, 0.35, 0.65);
    const bodyMat = new THREE.MeshBasicMaterial({ color: 0x243b53 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    this.turretHead.add(body);

    // Twin High-Energy Laser Emitter Barrels
    const barrelGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.5, 12);
    barrelGeo.rotateX(Math.PI / 2);
    const barrelMat = new THREE.MeshBasicMaterial({ color: 0x0f172a });

    const barrelL = new THREE.Mesh(barrelGeo, barrelMat);
    barrelL.position.set(-0.14, 0, 0.35);
    this.turretHead.add(barrelL);

    const barrelR = new THREE.Mesh(barrelGeo, barrelMat);
    barrelR.position.set(0.14, 0, 0.35);
    this.turretHead.add(barrelR);

    // Optics Apertures (Glowing Cyan Lenses)
    const lensGeo = new THREE.CircleGeometry(0.05, 12);
    const lensMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, side: THREE.DoubleSide });
    const lensL = new THREE.Mesh(lensGeo, lensMat);
    lensL.position.set(-0.14, 0, 0.61);
    this.turretHead.add(lensL);

    const lensR = new THREE.Mesh(lensGeo, lensMat);
    lensR.position.set(0.14, 0, 0.61);
    this.turretHead.add(lensR);

    this.turretGroup.add(this.turretHead);

    // Turret HUD Tag
    const tag = this.createLabelSprite('HEL AIR DEFENSE BATTERY [AUTO-AIM]', '#00e87a', 11);
    tag.position.set(0, 1.1, 0);
    this.turretGroup.add(tag);

    this.groundGroup.add(this.turretGroup);
  }

  // --- WebXR Controllers, Hands & Wrist HUD ---
  private initWebXrControllers() {
    // Controller 0 (Right)
    this.controller0 = this.renderer.xr.getController(0);
    this.controller0.addEventListener('select', this.onControllerSelect.bind(this));
    this.rig.add(this.controller0);

    // Controller 1 (Left - Wrist Controller)
    this.controller1 = this.renderer.xr.getController(1);
    this.controller1.addEventListener('select', this.onControllerSelect.bind(this));
    this.rig.add(this.controller1);

    // Laser Pointer Rays
    const laserGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -35),
    ]);
    const laserMat = new THREE.LineBasicMaterial({ color: 0x00e87a, transparent: true, opacity: 0.65 });
    const ray0 = new THREE.Line(laserGeo, laserMat);
    const ray1 = new THREE.Line(laserGeo.clone(), laserMat.clone());
    this.controller0.add(ray0);
    this.controller1.add(ray1);

    // Controller Grips with simple tactical mesh
    const gripGeo = new THREE.BoxGeometry(0.04, 0.05, 0.12);
    const gripMat = new THREE.MeshBasicMaterial({ color: 0x1e293b, wireframe: true });

    this.controllerGrip0 = this.renderer.xr.getControllerGrip(0);
    this.controllerGrip0.add(new THREE.Mesh(gripGeo, gripMat));
    this.rig.add(this.controllerGrip0);

    this.controllerGrip1 = this.renderer.xr.getControllerGrip(1);
    this.controllerGrip1.add(new THREE.Mesh(gripGeo, gripMat));
    this.rig.add(this.controllerGrip1);

    // Wrist Panel attached to Controller 1
    this.wristPanelSprite = this.createWristPanelSprite();
    this.wristPanelSprite.position.set(0, 0.08, -0.05);
    this.wristPanelSprite.scale.set(0.24, 0.16, 1);
    this.controller1.add(this.wristPanelSprite);
  }

  private onControllerSelect(e: any) {
    const controller = e.target;
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.identity().extractRotation(controller.matrixWorld);
    this.xrRaycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
    this.xrRaycaster.ray.direction.set(0, 0, -1).applyMatrix4(tempMatrix);

    const hit = this.raycastTargets(this.xrRaycaster);
    if (hit) {
      this.selectContact(hit);

      // Haptic feedback pulse on lock
      const session = this.renderer.xr.getSession();
      if (session) {
        const source = (session as any).inputSources?.[controller === this.controller0 ? 0 : 1];
        if (source?.gamepad?.hapticActuators?.[0]) {
          source.gamepad.hapticActuators[0].pulse(0.8, 60);
        }
      }
    }
  }

  // --- Input & Drag Handlers ---
  private initInputListeners() {
    const el = this.renderer.domElement;

    el.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX;
      this.dragStartY = e.clientY;
      this.dragStartYaw = this.yaw;
      this.dragStartPitch = this.pitch;
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.dragStartX;
      const dy = e.clientY - this.dragStartY;
      this.targetYaw = this.dragStartYaw - dx * 0.22;
      this.targetPitch = Math.max(-45, Math.min(85, this.dragStartPitch + dy * 0.22));
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // FOV Wheel (40 to 90 deg)
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const newFov = Math.max(40, Math.min(90, this.camera.fov + (e.deltaY > 0 ? 3 : -3)));
      this.camera.fov = newFov;
      this.camera.updateProjectionMatrix();
    }, { passive: false });

    // Click Raycasting to Select Target
    el.addEventListener('click', (e) => {
      if (this.isDragging) return;
      const rect = el.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, this.camera);
      const hit = this.raycastTargets(raycaster);
      if (hit) {
        this.selectContact(hit);
      }
    });

    // Keyboard Shortcuts: 1-9 / Tab cycle, Esc deselects
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.selectContact(null);
      } else if (e.key === 'Tab') {
        e.preventDefault();
        const ids = Array.from(this.targets.keys());
        if (ids.length > 0) {
          const idx = ids.indexOf(this.selectedContactId || '');
          const nextId = ids[(idx + 1) % ids.length];
          this.selectContact(nextId);
        }
      } else if (/^[1-9]$/.test(e.key)) {
        const num = parseInt(e.key, 10);
        const ids = Array.from(this.targets.keys());
        if (num <= ids.length) {
          this.selectContact(ids[num - 1]);
        }
      }
    });

    // Resize Observer
    const ro = new ResizeObserver(() => {
      this.resize();
    });
    ro.observe(this.container);

    // WebGL Context Recovery
    el.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      console.warn('WebGL context lost in VR Threat Dome');
    });
    el.addEventListener('webglcontextrestored', () => {
      console.info('WebGL context restored');
      this.initSkyAndStars();
      this.initDomeWireframe();
      this.initGroundAndRangeRings();
    });

    // Device Orientation (Gyro Look for Mobile)
    if (typeof window !== 'undefined' && 'DeviceOrientationEvent' in window) {
      window.addEventListener('deviceorientation', (e) => {
        if (e.alpha !== null && e.beta !== null && e.gamma !== null) {
          this.hasGyro = true;
          if (this.useGyro) {
            const alpha = THREE.MathUtils.degToRad(e.alpha + this.gyroAlphaOffset);
            const beta = THREE.MathUtils.degToRad(e.beta);
            const gamma = THREE.MathUtils.degToRad(e.gamma);
            const euler = new THREE.Euler(beta, alpha, -gamma, 'YXZ');
            this.gyroQuaternion.setFromEuler(euler);
          }
        }
      });
    }
  }

  private raycastTargets(raycaster: THREE.Raycaster): string | null {
    const meshes: THREE.Object3D[] = [];
    const map = new Map<THREE.Object3D, string>();

    this.targets.forEach((entry) => {
      meshes.push(entry.shapeMesh);
      map.set(entry.shapeMesh, entry.id);
    });

    const hits = raycaster.intersectObjects(meshes, false);
    if (hits.length > 0) {
      return map.get(hits[0].object) || null;
    }
    return null;
  }

  public selectContact(id: string | null) {
    if (this.selectedContactId !== id) {
      this.selectedContactId = id;
      this.onSelectContact?.(id);
      if (id) {
        vrAudio.playTargetLock();
      }
    }
  }

  // --- External Data Ingestion ---
  public setData(data: { contacts: Contact[]; scenario: any; selectedContactId?: string | null }) {
    if (this.scenario?.scenario_id !== data.scenario?.scenario_id) {
      this.destroyedIds.clear();
    }
    if (data.scenario?.destroyed_contact_ids && Array.isArray(data.scenario.destroyed_contact_ids)) {
      for (const id of data.scenario.destroyed_contact_ids) {
        this.destroyedIds.add(id);
      }
    }
    this.scenario = data.scenario;
    if (data.selectedContactId !== undefined) {
      this.selectedContactId = data.selectedContactId;
    }

    this.updateScenarioZonesAndRoe();
    this.diffTargets(data.contacts);
  }

  private updateScenarioZonesAndRoe() {
    if (!this.scenario) return;

    // Update ROE boundary rings
    this.roeGroup.clear();
    const roe = this.scenario.roe_card || {};
    const maxHard = Math.max(2500, roe.max_hard_kill_range_m || 2500);
    const maxSoft = Math.max(3500, roe.max_soft_kill_range_m || 3500);

    // Hard-kill ring (Hostile Red)
    const hardR = maxHard * WORLD_SCALE;
    const hardPoints: number[] = [];
    for (let i = 0; i <= 64; i++) {
      const th = (i * 2 * Math.PI) / 64;
      hardPoints.push(hardR * Math.sin(th), 0.04, hardR * Math.cos(th));
    }
    const hardGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(hardPoints, 3));
    const hardLine = new THREE.LineLoop(hardGeo, new THREE.LineBasicMaterial({ color: 0xff3b3b, linewidth: 2 }));
    this.roeGroup.add(hardLine);

    const hardLabel = this.createLabelSprite(`ROE: HARD-KILL LIMIT (${maxHard}m)`, '#ff3b3b');
    hardLabel.position.set(0, 0.8, -hardR);
    this.roeGroup.add(hardLabel);

    // Soft-kill ring (Decoy Violet)
    const softR = maxSoft * WORLD_SCALE;
    const softPoints: number[] = [];
    for (let i = 0; i <= 64; i++) {
      const th = (i * 2 * Math.PI) / 64;
      softPoints.push(softR * Math.sin(th), 0.04, softR * Math.cos(th));
    }
    const softGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(softPoints, 3));
    const softLine = new THREE.LineLoop(softGeo, new THREE.LineBasicMaterial({ color: 0xbf5af2, linewidth: 1.5 }));
    this.roeGroup.add(softLine);

    // No-engage zones
    this.noEngageGroup.clear();
    const zones = this.scenario.no_engage_zones || [];
    for (const z of zones) {
      const r = (z.radius_m || 300) * WORLD_SCALE;
      const x = (z.center?.[0] || 0) * WORLD_SCALE;
      const zPos = -(z.center?.[1] || 0) * WORLD_SCALE;

      const cylGeo = new THREE.CylinderGeometry(r, r, 5, 24, 1, true);
      const cylMat = new THREE.MeshBasicMaterial({
        color: 0xff3b3b,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
      });
      const cyl = new THREE.Mesh(cylGeo, cylMat);
      cyl.position.set(x, 2.5, zPos);
      this.noEngageGroup.add(cyl);

      const label = this.createLabelSprite(`NO-ENGAGE: ${z.name || 'CIVILIAN AIRWAY'}`, '#ff3b3b');
      label.position.set(x, 5.2, zPos);
      this.noEngageGroup.add(label);
    }
  }

  // --- Target Diffing & Shape Synthesis ---
  private diffTargets(contacts: Contact[]) {
    const activeIds = new Set<string>();

    for (const c of contacts) {
      if (
        this.destroyedIds.has(c.id) ||
        c.state === 'neutralized' ||
        c.state === 'destroyed' ||
        (c as any).engagement_action === 'hard_kill'
      ) {
        // Destroyed target — if lingering in targets map, immediately mark fading
        const existing = this.targets.get(c.id);
        if (existing && !existing.fading) {
          existing.fading = true;
        }
        continue;
      }

      activeIds.add(c.id);
      const assigned = (c as any).assigned || c.assigned_classification || 'unknown';

      let entry = this.targets.get(c.id);
      if (!entry) {
        // Create new target visual
        const raw = this.scenario?.contacts?.find((rc: any) => (rc.contact_id || rc.id) === c.id) || c;
        entry = this.createTargetMeshEntry(c.id, assigned, raw);
        this.targets.set(c.id, entry);
        this.scene.add(entry.group);
      } else {
        // Update shape if classification changed
        if (entry.assigned !== assigned) {
          entry.assigned = assigned;
          this.updateTargetGeometryAndColor(entry, assigned);
        }
      }
    }

    // Mark missing targets for clean fade-out
    this.targets.forEach((entry, id) => {
      if (!activeIds.has(id)) {
        entry.fading = true;
      }
    });
  }

  private createTargetMeshEntry(id: string, assigned: string, rawContact: any): TargetMeshEntry {
    const group = new THREE.Group();

    // 1. Shape Mesh based on assigned classification
    const shapeMesh = new THREE.Mesh();
    group.add(shapeMesh);

    // 2. Altitude drop line to ground
    const dropGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, -10, 0),
    ]);
    const dropLine = new THREE.Line(dropGeo, this.sharedMaterials.dropLine);
    group.add(dropLine);

    // Ground footprint disc
    const dropGroundDisc = new THREE.Mesh(this.sharedGeometries.dropDisc, this.sharedMaterials.dropLine as any);
    dropGroundDisc.rotation.x = -Math.PI / 2;
    group.add(dropGroundDisc);

    // 3. Billboard HUD Tag
    const billboardSprite = this.createLabelSprite(`${id.slice(0, 8).toUpperCase()}\n...`, '#ffffff');
    billboardSprite.position.set(0, 0.75, 0);
    billboardSprite.scale.set(1.5, 0.75, 1);
    group.add(billboardSprite);

    // 4. Past breadcrumb trail line
    const trailGeo = new THREE.BufferGeometry();
    const trailLine = new THREE.Line(trailGeo, this.sharedMaterials.trailLine);
    this.scene.add(trailLine);

    // 5. Pulsing reticle for selection
    const reticle = this.createReticleGroup();
    reticle.visible = false;
    group.add(reticle);

    const entry: TargetMeshEntry = {
      id,
      group,
      shapeMesh,
      dropLine,
      dropGroundDisc,
      billboardSprite,
      trailLine,
      reticle,
      assigned,
      rawContact,
      fading: false,
      opacity: 1,
    };

    this.updateTargetGeometryAndColor(entry, assigned);
    return entry;
  }

  private updateTargetGeometryAndColor(entry: TargetMeshEntry, assigned: string) {
    let geo: THREE.BufferGeometry;
    let mat: THREE.Material;

    switch (assigned) {
      case 'hostile_drone':
      case 'hostile_swarm_member':
      case 'drone':
      case 'quadcopter':
        geo = this.sharedGeometries.hostile;
        mat = this.sharedMaterials.hostile;
        break;
      case 'friendly_drone':
        geo = this.sharedGeometries.friendly;
        mat = this.sharedMaterials.friendly;
        break;
      case 'decoy':
        geo = this.sharedGeometries.decoy;
        mat = this.sharedMaterials.decoy;
        break;
      case 'bird':
      case 'neutral':
        geo = this.sharedGeometries.bird;
        mat = this.sharedMaterials.bird;
        break;
      default:
        geo = this.sharedGeometries.unknown;
        mat = this.sharedMaterials.unknown;
        break;
    }

    entry.shapeMesh.geometry = geo;
    entry.shapeMesh.material = mat;
  }

  private createReticleGroup(): THREE.Group {
    const grp = new THREE.Group();
    const pts = [
      new THREE.Vector3(-0.55, 0, 0),
      new THREE.Vector3(0, 0.55, 0),
      new THREE.Vector3(0.55, 0, 0),
      new THREE.Vector3(0, -0.55, 0),
      new THREE.Vector3(-0.55, 0, 0),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, this.sharedMaterials.reticle);
    grp.add(line);
    return grp;
  }

  // --- Dynamic Per-Frame Kinematics Interpolation ---
  public update(dt: number) {
    const now = performance.now();
    const epoch = this.scenario?.server_start_epoch_ms || now;
    const duration = this.scenario?.duration_s || 300;
    const rawElapsed = Math.max(0, (Date.now() - epoch) / 1000);
    const t = rawElapsed > duration ? (rawElapsed % duration) : rawElapsed;

    // 1. Smooth Desktop Pan / Orbit Damping
    this.yaw += (this.targetYaw - this.yaw) * Math.min(1, dt * 14);
    this.pitch += (this.targetPitch - this.pitch) * Math.min(1, dt * 14);

    if (this.useGyro && this.hasGyro) {
      this.camera.quaternion.copy(this.gyroQuaternion);
    } else if (!this.renderer.xr.isPresenting) {
      const euler = new THREE.Euler(
        THREE.MathUtils.degToRad(this.pitch),
        THREE.MathUtils.degToRad(this.yaw),
        0,
        'YXZ'
      );
      this.camera.quaternion.setFromEuler(euler);
    }

    // 2. Gaze Selection Dwell in Split-Screen Mode
    if (this.viewMode === 'splitscreen') {
      this.gazeRaycaster.setFromCamera(new THREE.Vector2(0, 0), this.camera);
      const hit = this.raycastTargets(this.gazeRaycaster);

      if (hit) {
        if (this.gazeTargetId === hit) {
          this.gazeDwellTime += dt;
          this.onLockProgress?.(Math.min(1, this.gazeDwellTime / 1.0));
          if (this.gazeDwellTime >= 1.0) {
            this.selectContact(hit);
          }
        } else {
          this.gazeTargetId = hit;
          this.gazeDwellTime = 0;
          this.onLockProgress?.(0);
        }
      } else {
        this.gazeTargetId = null;
        this.gazeDwellTime = 0;
        this.onLockProgress?.(0);
      }
    } else {
      this.onLockProgress?.(0);
    }

    // 3. Update Targets Kinematics
    const toDelete: string[] = [];

    this.targets.forEach((entry, id) => {
      if (entry.fading || this.destroyedIds.has(id)) {
        entry.opacity -= dt * 2.5;
        entry.group.scale.setScalar(Math.max(0.01, entry.opacity));
        if (entry.opacity <= 0) {
          toDelete.push(id);
          this.scene.remove(entry.group);
          this.scene.remove(entry.trailLine);
          return;
        }
      }

      const pos = posAt(entry.rawContact, t);
      const x_world = pos.x * WORLD_SCALE;
      const z_world = -pos.y * WORLD_SCALE; // North = -Z
      const y_world = EYE_HEIGHT + (pos.z * ALT_EXAG) * WORLD_SCALE;

      entry.group.position.set(x_world, y_world, z_world);

      // Adjust altitude drop line down to ground y=0
      const dropPositions = (entry.dropLine.geometry.attributes.position as THREE.BufferAttribute);
      dropPositions.setXYZ(0, 0, 0, 0);
      dropPositions.setXYZ(1, 0, -y_world, 0);
      dropPositions.needsUpdate = true;
      entry.dropGroundDisc.position.set(0, -y_world + 0.02, 0);

      // Reticle pulse and rotation if selected
      const isSelected = this.selectedContactId === id;
      entry.reticle.visible = isSelected;
      if (isSelected) {
        entry.reticle.rotation.z += dt * 3;
        const s = 1.0 + Math.sin(now * 0.008) * 0.2;
        entry.reticle.scale.set(s, s, 1);
        entry.billboardSprite.scale.set(1.8, 0.9, 1);
      } else {
        entry.billboardSprite.scale.set(1.4, 0.7, 1);
      }

      // Update Billboard HUD Text
      const rangeM = Math.round(Math.hypot(pos.x, pos.y));
      const altM = Math.round(pos.z);
      this.updateSpriteLabel(
        entry.billboardSprite,
        `${id.slice(0, 8).toUpperCase()}\n${rangeM}m | ${altM}m`,
        isSelected ? '#00e87a' : '#c8deef'
      );

      // Trail history line
      if (entry.rawContact.path && entry.rawContact.path.length > 1) {
        const trailPts: THREE.Vector3[] = [];
        for (const wp of entry.rawContact.path) {
          if (wp.t <= t) {
            trailPts.push(new THREE.Vector3(wp.x * WORLD_SCALE, EYE_HEIGHT + (wp.z * ALT_EXAG) * WORLD_SCALE, -wp.y * WORLD_SCALE));
          }
        }
        trailPts.push(new THREE.Vector3(x_world, y_world, z_world));
        entry.trailLine.geometry.setFromPoints(trailPts);
      }
    });

    for (const id of toDelete) {
      this.targets.delete(id);
    }

    // 3b. Aim HEL Battery Turret at Selected Target
    if (this.turretHead && this.selectedContactId) {
      const selectedEntry = this.targets.get(this.selectedContactId);
      if (selectedEntry) {
        this.turretHead.lookAt(selectedEntry.group.position);
      }
    }

    // 4. Update FX Volumetric Beams, Tracers, and Explosions
    for (let i = this.activeHardKillEffects.length - 1; i >= 0; i--) {
      const fx = this.activeHardKillEffects[i];
      fx.life -= dt;
      const progress = 1.0 - Math.max(0, fx.life / fx.maxLife); // 0.0 -> 1.0

      // High-velocity tracer moves from emitter to target in first 0.25s
      const tracerProg = Math.min(1.0, progress * 4.0);
      fx.tracer.position.lerpVectors(fx.emitterPos, fx.targetPos, tracerProg);
      if (tracerProg >= 1.0) {
        fx.tracer.visible = false;
      }

      // Beam sustains with high-energy laser flicker for the first 0.6s, then fades
      if (fx.life > 0.8) {
        const jitter = 0.85 + Math.random() * 0.15;
        (fx.coreBeam.material as THREE.Material).opacity = jitter;
        (fx.glowBeam.material as THREE.Material).opacity = 0.85 * jitter;
      } else {
        const fade = Math.max(0, fx.life / 0.8);
        (fx.coreBeam.material as THREE.Material).opacity = fade;
        (fx.glowBeam.material as THREE.Material).opacity = fade * 0.85;
      }

      // Muzzle flash dissipates quickly
      const muzzleFade = Math.max(0, 1.0 - progress * 4.0);
      (fx.muzzleFlash.material as THREE.Material).opacity = muzzleFade;
      fx.muzzleFlash.scale.setScalar(1.0 + progress * 2.0);

      // Impact explosion blast expands and fades
      const blastScale = 1.0 + progress * 3.5;
      fx.blastCore.scale.setScalar(blastScale);
      fx.blastWire.scale.setScalar(blastScale * 1.25);
      fx.shockwaveRing.scale.setScalar(1.0 + progress * 5.0);

      const blastFade = Math.max(0, fx.life / fx.maxLife);
      (fx.blastCore.material as THREE.Material).opacity = blastFade;
      (fx.blastWire.material as THREE.Material).opacity = blastFade * 0.75;
      (fx.shockwaveRing.material as THREE.Material).opacity = blastFade * 0.85;

      // Sparks fly outwards with radial velocity and gravity
      if (fx.sparks && fx.sparkVelocities) {
        const positions = fx.sparks.geometry.attributes.position.array as Float32Array;
        for (let j = 0; j < fx.sparkVelocities.length; j++) {
          positions[j * 3] += fx.sparkVelocities[j].x * dt;
          positions[j * 3 + 1] += (fx.sparkVelocities[j].y - 2.5 * progress) * dt;
          positions[j * 3 + 2] += fx.sparkVelocities[j].z * dt;
        }
        fx.sparks.geometry.attributes.position.needsUpdate = true;
        (fx.sparks.material as THREE.PointsMaterial).opacity = blastFade;
      }

      if (fx.life <= 0) {
        this.fxGroup.remove(fx.group);
        fx.coreBeam.geometry.dispose();
        fx.coreBeam.material.dispose();
        fx.glowBeam.geometry.dispose();
        fx.glowBeam.material.dispose();
        fx.blastCore.geometry.dispose();
        fx.blastCore.material.dispose();
        fx.blastWire.geometry.dispose();
        fx.blastWire.material.dispose();
        fx.shockwaveRing.geometry.dispose();
        fx.shockwaveRing.material.dispose();
        if (fx.sparks) {
          fx.sparks.geometry.dispose();
          fx.sparks.material.dispose();
        }
        this.activeHardKillEffects.splice(i, 1);
      }
    }

    for (let i = this.activeSoftKillEffects.length - 1; i >= 0; i--) {
      const fx = this.activeSoftKillEffects[i];
      fx.life -= dt * 2.0;
      (fx.cone.material as THREE.Material).opacity = Math.max(0, fx.life * 0.4);
      if (fx.life <= 0) {
        this.fxGroup.remove(fx.cone);
        this.activeSoftKillEffects.splice(i, 1);
      }
    }
  }

  // --- Tactical FX Deployment ---
  public triggerHardKillEffect(targetId: string) {
    this.destroyedIds.add(targetId);
    if (this.selectedContactId === targetId) {
      this.selectContact(null);
    }

    const entry = this.targets.get(targetId);
    if (!entry) return;

    entry.fading = true;
    vrAudio.playHardKill();

    let targetPos = entry.group.position.clone();
    if (targetPos.lengthSq() < 0.1 && entry.rawContact) {
      const now = performance.now();
      const epoch = this.scenario?.server_start_epoch_ms || now;
      const duration = this.scenario?.duration_s || 300;
      const rawElapsed = Math.max(0, (Date.now() - epoch) / 1000);
      const t = rawElapsed > duration ? (rawElapsed % duration) : rawElapsed;
      const pos = posAt(entry.rawContact, t);
      targetPos = new THREE.Vector3(
        pos.x * WORLD_SCALE,
        EYE_HEIGHT + (pos.z * ALT_EXAG) * WORLD_SCALE,
        -pos.y * WORLD_SCALE
      );
    }

    // Emitter: Ground Counter-UAS Laser Battery at (2.4, 0.45, 1.8)
    const emitterPos = this.turretHead
      ? this.turretHead.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.2, 0))
      : new THREE.Vector3(2.4, 0.65, 1.8);

    // Aim turret at target immediately
    if (this.turretHead) {
      this.turretHead.lookAt(targetPos);
    }

    const dir = new THREE.Vector3().subVectors(targetPos, emitterPos);
    const dist = dir.length();
    const mid = emitterPos.clone().add(targetPos).multiplyScalar(0.5);

    const fxGrp = new THREE.Group();

    // 1. Core White Laser Beam (Volumetric 3D Cylinder)
    const coreGeo = new THREE.CylinderGeometry(0.12, 0.12, dist, 8, 1, true);
    coreGeo.rotateX(Math.PI / 2);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 1.0,
      depthWrite: false,
    });
    const coreBeam = new THREE.Mesh(coreGeo, coreMat);
    coreBeam.position.copy(mid);
    coreBeam.lookAt(targetPos);
    fxGrp.add(coreBeam);

    // 2. High-Energy Cyan / Azure Laser Glow Sheath (Volumetric Additive)
    const glowGeo = new THREE.CylinderGeometry(0.42, 0.42, dist, 8, 1, true);
    glowGeo.rotateX(Math.PI / 2);
    const glowMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const glowBeam = new THREE.Mesh(glowGeo, glowMat);
    glowBeam.position.copy(mid);
    glowBeam.lookAt(targetPos);
    fxGrp.add(glowBeam);

    // 3. High-Velocity Laser Tracer Pulse
    const tracerGeo = new THREE.SphereGeometry(0.45, 12, 12);
    const tracerMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const tracer = new THREE.Mesh(tracerGeo, tracerMat);
    tracer.position.copy(emitterPos);
    fxGrp.add(tracer);

    // 4. Muzzle Flash at Turret
    const muzzleGeo = new THREE.SphereGeometry(0.7, 12, 12);
    const muzzleMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const muzzleFlash = new THREE.Mesh(muzzleGeo, muzzleMat);
    muzzleFlash.position.copy(emitterPos);
    fxGrp.add(muzzleFlash);

    // 5. Target Impact Detonation
    // 5a. Fiery Core Blast Sphere
    const blastGeo = new THREE.SphereGeometry(1.4, 16, 16);
    const blastMat = new THREE.MeshBasicMaterial({
      color: 0xff3b3b,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const blastCore = new THREE.Mesh(blastGeo, blastMat);
    blastCore.position.copy(targetPos);
    fxGrp.add(blastCore);

    // 5b. Wireframe Blast Cage
    const wireGeo = new THREE.SphereGeometry(1.9, 12, 12);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0xffaa00,
      transparent: true,
      opacity: 0.9,
      wireframe: true,
      depthWrite: false,
    });
    const blastWire = new THREE.Mesh(wireGeo, wireMat);
    blastWire.position.copy(targetPos);
    fxGrp.add(blastWire);

    // 5c. Expanding Shockwave Ring
    const ringGeo = new THREE.RingGeometry(0.3, 1.2, 24);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff4422,
      transparent: true,
      opacity: 0.95,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const shockwaveRing = new THREE.Mesh(ringGeo, ringMat);
    shockwaveRing.position.copy(targetPos);
    shockwaveRing.rotation.x = Math.PI / 2;
    fxGrp.add(shockwaveRing);

    // 5d. Kinetic Spark Fragments
    const sparkCount = 20;
    const sparkPositions = new Float32Array(sparkCount * 3);
    const sparkVelocities: THREE.Vector3[] = [];
    for (let i = 0; i < sparkCount; i++) {
      sparkPositions[i * 3] = targetPos.x;
      sparkPositions[i * 3 + 1] = targetPos.y;
      sparkPositions[i * 3 + 2] = targetPos.z;
      const v = new THREE.Vector3(
        (Math.random() - 0.5) * 8,
        (Math.random() - 0.5) * 8 + 1.0,
        (Math.random() - 0.5) * 8
      );
      sparkVelocities.push(v);
    }
    const sparkGeo = new THREE.BufferGeometry();
    sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPositions, 3));
    const sparkMat = new THREE.PointsMaterial({
      color: 0xffcc00,
      size: 0.35,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const sparks = new THREE.Points(sparkGeo, sparkMat);
    fxGrp.add(sparks);

    this.fxGroup.add(fxGrp);

    this.activeHardKillEffects.push({
      group: fxGrp,
      coreBeam,
      glowBeam,
      muzzleFlash,
      blastCore,
      blastWire,
      shockwaveRing,
      tracer,
      sparks,
      sparkVelocities,
      emitterPos,
      targetPos,
      life: 1.4,
      maxLife: 1.4,
    });
  }

  public triggerSoftKillEffect(targetId: string) {
    const entry = this.targets.get(targetId);
    if (!entry) return;

    vrAudio.playSoftKill();

    let targetPos = entry.group.position.clone();
    if (targetPos.lengthSq() < 0.1 && entry.rawContact) {
      const now = performance.now();
      const epoch = this.scenario?.server_start_epoch_ms || now;
      const duration = this.scenario?.duration_s || 300;
      const rawElapsed = Math.max(0, (Date.now() - epoch) / 1000);
      const t = rawElapsed > duration ? (rawElapsed % duration) : rawElapsed;
      const pos = posAt(entry.rawContact, t);
      targetPos = new THREE.Vector3(
        pos.x * WORLD_SCALE,
        EYE_HEIGHT + (pos.z * ALT_EXAG) * WORLD_SCALE,
        -pos.y * WORLD_SCALE
      );
    }

    const emitterPos = this.turretHead
      ? this.turretHead.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.2, 0))
      : new THREE.Vector3(2.4, 0.65, 1.8);

    if (this.turretHead) {
      this.turretHead.lookAt(targetPos);
    }

    const dir = new THREE.Vector3().subVectors(targetPos, emitterPos);
    const dist = dir.length();

    // Jamming cone from emitter to target
    const coneGeo = new THREE.ConeGeometry(dist * 0.18, dist, 16, 1, true);
    coneGeo.translate(0, -dist / 2, 0);
    coneGeo.rotateX(-Math.PI / 2);

    const cone = new THREE.Mesh(coneGeo, this.sharedMaterials.jamCone.clone());
    cone.position.copy(emitterPos);
    cone.lookAt(targetPos);
    this.fxGroup.add(cone);

    this.activeSoftKillEffects.push({ cone, life: 1.4 });
  }

  // --- Render & Animation Loop ---
  private animate() {
    if (this.isDestroyed) return;

    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastTime) / 1000);
    this.lastTime = now;

    this.update(dt);

    if (this.renderer.xr.isPresenting) {
      this.renderer.render(this.scene, this.camera);
    } else if (this.viewMode === 'splitscreen') {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.stereoCamera.update(this.camera);

      this.renderer.setScissorTest(true);

      // Left Eye
      this.renderer.setScissor(0, 0, w / 2, h);
      this.renderer.setViewport(0, 0, w / 2, h);
      this.renderer.render(this.scene, this.stereoCamera.cameraL);

      // Right Eye
      this.renderer.setScissor(w / 2, 0, w / 2, h);
      this.renderer.setViewport(w / 2, 0, w / 2, h);
      this.renderer.render(this.scene, this.stereoCamera.cameraR);

      this.renderer.setScissorTest(false);
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  // --- Mode Switches ---
  public setViewMode(mode: VrViewMode) {
    this.viewMode = mode;
    this.resize();
  }

  public getViewMode(): VrViewMode {
    return this.viewMode;
  }

  public async enterImmersiveVr(): Promise<{ success: boolean; reason?: string }> {
    if (typeof navigator === 'undefined' || !(navigator as any).xr) {
      return { success: false, reason: 'WebXR is not supported in this browser.' };
    }

    try {
      const isSupported = await (navigator as any).xr.isSessionSupported('immersive-vr');
      if (!isSupported) {
        return { success: false, reason: 'No VR headset or immersive WebXR runtime detected.' };
      }

      const session = await (navigator as any).xr.requestSession('immersive-vr', {
        optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking'],
      });

      this.xrSession = session;
      await this.renderer.xr.setSession(session);
      this.setViewMode('immersive');

      session.addEventListener('end', () => {
        this.xrSession = null;
        this.setViewMode('desktop');
      });

      return { success: true };
    } catch (err: any) {
      return { success: false, reason: err.message || 'Failed to initialize immersive WebXR session.' };
    }
  }

  public exitVr() {
    if (this.xrSession) {
      this.xrSession.end();
      this.xrSession = null;
    }
    this.setViewMode('desktop');
  }

  public recenterGyro() {
    this.gyroAlphaOffset = -this.yaw;
    this.targetYaw = 0;
    this.targetPitch = 10;
  }

  public setUseGyro(use: boolean) {
    this.useGyro = use;
  }

  // --- Dynamic Canvas Sprites for Billboards & HUDs ---
  private createLabelSprite(text: string, color: string, fontSize = 16): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    const ctx = canvas.getContext('2d')!;

    this.drawLabelCanvas(ctx, text, color, fontSize);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(spriteMat);
    (sprite as any)._canvas = canvas;
    (sprite as any)._ctx = ctx;
    (sprite as any)._lastText = text;
    (sprite as any)._lastColor = color;
    return sprite;
  }

  private updateSpriteLabel(sprite: THREE.Sprite, text: string, color: string) {
    if ((sprite as any)._lastText === text && (sprite as any)._lastColor === color) return;
    (sprite as any)._lastText = text;
    (sprite as any)._lastColor = color;

    const ctx = (sprite as any)._ctx as CanvasRenderingContext2D;
    const canvas = (sprite as any)._canvas as HTMLCanvasElement;
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.drawLabelCanvas(ctx, text, color, 16);
    sprite.material.map!.needsUpdate = true;
  }

  private drawLabelCanvas(ctx: CanvasRenderingContext2D, text: string, color: string, fontSize: number) {
    ctx.fillStyle = 'rgba(8, 18, 30, 0.88)';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 120, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = `bold ${fontSize}px "Geist Mono", monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const lines = text.split('\n');
    if (lines.length === 1) {
      ctx.fillText(lines[0], 128, 64);
    } else {
      ctx.fillText(lines[0], 128, 48);
      ctx.fillStyle = '#ffffff';
      ctx.font = `${fontSize - 2}px "Geist Mono", monospace`;
      ctx.fillText(lines[1], 128, 80);
    }
  }

  private createWristPanelSprite(): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 160;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'rgba(5, 12, 20, 0.95)';
    ctx.strokeStyle = '#00e87a';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(4, 4, 248, 152, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#00e87a';
    ctx.font = 'bold 16px "Chakra Petch", sans-serif';
    ctx.fillText('VR DOME WRIST C2', 18, 32);

    ctx.fillStyle = '#8aaabf';
    ctx.font = '12px "Geist Mono", monospace';
    ctx.fillText('TRIGGER : LOCK TARGET', 18, 68);
    ctx.fillText('THUMBSTICK : SNAP 30°', 18, 96);
    ctx.fillText('ALTITUDE EXAG : 3X', 18, 124);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    return new THREE.Sprite(spriteMat);
  }

  public resize() {
    if (!this.container || !this.renderer) return;
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  public dispose() {
    this.isDestroyed = true;
    this.renderer.setAnimationLoop(null);
    if (this.xrSession) {
      this.xrSession.end();
    }
    this.targets.forEach((entry) => {
      this.scene.remove(entry.group);
      this.scene.remove(entry.trailLine);
    });
    this.targets.clear();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }
  }
}
