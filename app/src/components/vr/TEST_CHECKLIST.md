# VR Threat Dome — Manual Verification Checklist

Follow this checklist to manually verify the VR Airspace Dome implementation across all supported modes and tactical edge cases.

---

### Pre-requisites:
1. Ensure the Python FastAPI backend is running:
   ```bash
   python -m uvicorn server:app --port 8090
   ```
2. Run the Vite application:
   ```bash
   npm --prefix app run dev
   ```

---

### Test Checklist:

1. **[x] Honest Empty State (No Active Scenario)**
   - Open `http://localhost:5173/vr` when no scenario has been launched on the server.
   - **Expected**: A clean, honest empty state renders: *"No active scenario. Start one in Trainer"* with a primary action button linking to `/trainer?autostart=1`.
   - **Verify**: Zero fake, mock, or hardcoded contacts appear.

2. **[x] Live Scenario Launch & Auto-Sync**
   - Click **"Start one in Trainer"** or open `http://localhost:5173/trainer?autostart=1`.
   - Wait for the scenario to initialize, then navigate to `/vr`.
   - **Expected**: Status badge changes to `LIVE TELEMETRY`. The 360° wireframe hemisphere, range rings, cardinal labels (NORTH [000°], EAST, SOUTH, WEST), and active contact meshes appear around the defended asset.

3. **[x] Scenario Dropdown Switching**
   - In the top-right toolbar, click the **SCENARIO** dropdown.
   - **Expected**: All active scenarios on the backend are listed with track counts and terrain. Switching to a different scenario immediately re-binds telemetry, ROE cards, and airspace bounds.

4. **[x] Classified vs. Unknown Shape Coding (MIL-STD Shapes)**
   - View targets in the 3D scene:
     - **Unknown**: Yellow Octahedron (`◆`)
     - **Hostile Drone / Swarm**: Red Cone (`▲`)
     - **Friendly Drone**: Blue Sphere (`●`)
     - **Decoy**: Violet Box (`□`)
     - **Bird / Neutral**: Teal Torus (`○`)
   - In `/trainer`, classify a contact (using hotkeys `1–4` or buttons). Return to `/vr`.
   - **Expected**: The target shape instantly morphs in 3D to match the assigned classification without relying on ground truth.

5. **[x] Desktop Orbit Controls & Dynamic FOV**
   - Click and drag with mouse/trackpad.
   - **Expected**: Smooth, damped 360° yaw and pitch rotation around the defended asset observation post.
   - Scroll the mouse wheel up and down.
   - **Expected**: Camera Field of View adjusts smoothly between 40° (telephoto zoom) and 90° (wide angle).
   - Press `Tab` or keys `1–9`.
   - **Expected**: Selection reticle locks onto contacts sequentially. Press `Esc` to deselect.

6. **[x] Split-Screen (Cardboard VR) & Gaze Dwell Lock**
   - Click the **Split-Screen** button in the top toolbar.
   - **Expected**: Screen splits into side-by-side stereoscopic view with center divider.
   - Aim the center reticle at any contact for **1.0 second**.
   - **Expected**: Circular green dwell progress ring fills around reticle; once full, target locks with tactical audio chirp.
   - Click the **Fullscreen** button to toggle immersive fullscreen.

7. **[x] Mobile Gyroscope 360° Tracking**
   - Open `/vr` on a mobile browser or emulate device sensors in Chrome DevTools.
   - Tap the **Compass** icon in the toolbar.
   - **Expected**: Physical orientation of the phone controls camera pitch and heading. Tap **Recenter** to zero current viewpoint to North.

8. **[x] WebXR Immersive Headset Mode (Immersive Web Emulator)**
   - Open DevTools → **WebXR** tab → select headset profile (e.g. Meta Quest 2).
   - Toggle WebXR emulation ON.
   - In `/vr`, click **Enter Immersive VR**.
   - **Expected**: Enters stereoscopic XR loop at 72+ fps. Controller laser rays point into the dome. Squeeze trigger on a target to lock with haptic feedback. Virtual wrist C2 panel floats on left controller.

9. **[x] ROE Weapons-Hold Block**
   - In a scenario configured with `weapons_hold = true` (or difficulty with hold order):
   - **Expected**: Top bar shows red `ROE: WEAPONS HOLD ACTIVE` alert. The **HARD-KILL** button is disabled, preventing kinetic engagement.

10. **[x] No-Engage Zone & Range Boundary Interception**
    - Select a target located inside a red cylindrical **No-Engage Zone** (or > 2500m away).
    - **Expected**: Banner displays `HARD-KILL PROHIBITED: INSIDE NO-ENGAGE ZONE` or `OUT OF RANGE`. Hard-kill is locked out.

11. **[x] 600ms Hold-to-Confirm Hard-Kill Deployment**
    - Select an active target within ROE parameters.
    - Press and hold the **HARD-KILL** button.
    - **Expected**: Button shows dynamic countdown (`HOLD (600ms)` ... `0ms`) with red progress fill.
    - Releasing before 600ms cancels fire.
    - Holding for full 600ms deploys kinetic countermeasure:
      - Directed laser beam and explosion burst flash in 3D.
      - Dispatches `POST /api/contact/engage` with `{ scenario_id, contact_id, t, action: 'hard_kill' }`.
      - Target begins clean fade-out.
