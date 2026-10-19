# WebXR Spatial Threat Dome — Developer & Verification Guide

This component implements a 360° viewer-centred Air Defense C2 spatial twin rendered with **Three.js** and **WebXR Device API**.

---

## 1. Zero-Mock Data Architecture

- Every target is streamed directly from the live Python FastAPI server via `/api/scenario/{id}/refresh`.
- When no scenario is active on the server, an **honest empty state** is displayed ("No active scenario. Start one in Trainer"). Fake or hardcoded demo targets are strictly forbidden.
- Real-time kinematic positions are computed per animation frame based on the server's sanitized waypoints (`(Date.now() - server_start_epoch_ms) / 1000`).
- Target classification tags derive from live multi-operator team roles (`/api/team/state`) or the local trainee log. Ground-truth classifications are never revealed prior to scoring.

---

## 2. Testing WebXR with the Immersive Web Emulator

The WebXR Device API requires a **secure origin** (`https://` or `http://localhost`).

### Browser Extension Setup:
1. Install the **[Immersive Web Emulator](https://chrome.google.com/webstore/detail/immersive-web-emulator/cgffilbngfmgjamahijjkdeamidikhaf)** extension for Google Chrome or Microsoft Edge.
2. Open DevTools (`F12` or `Ctrl+Shift+I`).
3. Select the **WebXR** tab in DevTools.
4. Choose a headset profile (e.g., **Meta Quest 2 / Quest 3**).
5. Click the toggle to enable headset emulation.
6. In the Vyom Kavach VR page (`/vr`), the **"Enter Immersive VR"** button will now be active.
7. Click **"Enter Immersive VR"**:
   - The scene enters full WebXR stereoscopic rendering.
   - Use the emulator's controller gizmos to point the laser ray at targets and press trigger to lock.
   - The virtual wrist C2 panel appears on the left controller.

---

## 3. Alternative VR Modes (No Headset Required)

- **Desktop Orbit**: Drag with mouse/trackpad to look around the 360° threat dome. Scroll mouse wheel to dynamically adjust camera Field of View (40° to 90°). Keys `1–9` / `Tab` cycle through live targets.
- **Split-Screen (Google Cardboard)**: Renders stereoscopic side-by-side view with a center divider. Features a **1.0-second gaze dwell ring**—simply look at any target for 1 second to lock it.
- **Mobile Gyroscope**: On phones and tablets, tap the **Compass** icon in the top toolbar to enable native 360° motion tracking. Use **Recenter** to zero the heading to your current orientation.

---

## 4. Tactical Counter-UAS Engagements

- **Warn / Hail**: Broadcasts advisory warning on emergency defense frequencies.
- **RF Soft-Kill**: Emits directed multi-band RF jamming cone towards the target.
- **Hard-Kill**:
  - Requires holding the **HARD-KILL** button for **600 milliseconds** (visible progress ring).
  - Automatically blocked with a warning banner if the target is inside a **No-Engage Zone** or beyond the ROE hard-kill boundary (`2500m`).
  - Calls `POST /api/contact/engage` with `{ scenario_id, contact_id, t, action: 'hard_kill' }`.
  - Triggers a directed kinetic laser beam and expansion burst in 3D.
