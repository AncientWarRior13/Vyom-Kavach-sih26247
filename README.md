# DroneThreatTrainer

A desktop- and VR-capable drone & swarm threat recognition training
simulator: scripted and procedurally generated scenarios across
day/night, degraded-sensor, and urban/rural/coastal/mountain terrain
conditions, decision-tree scoring on detection time / classification
accuracy / engagement decisions, and an after-action-review (AAR) dashboard
tracking individual/unit performance over repeated sessions.

## Status: fully built, every problem-statement requirement covered

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8090
```

- `http://localhost:8090/` — desktop trainee exercise screen
- `http://localhost:8090/vr` — WebXR/VR trainee exercise screen (works as a
  regular 3D view in any browser; enters headset immersive mode on a
  WebXR-capable browser/device, e.g. Meta Quest Browser)
- `http://localhost:8090/aar` — after-action-review dashboard

Flow: pick difficulty/time-of-day/terrain/sensor-degradation and start a
procedural scenario, **or** load a fixed scripted drill from the dropdown
→ click/tag blips as you spot them, optionally request an "AI sensor hint",
classify each contact, choose engage/monitor/ignore → "Finish & score" →
the breakdown appears and shows up immediately in the AAR dashboard under
that trainee/unit ID.

## Layout

| Folder | Role |
|---|---|
| `scenario_engine/` | Procedurally generates scenarios (hostile drones, swarm attacks, decoys, birds, friendly drones) with day/night/terrain/sensor-degradation parameters that have a real effect on detection difficulty (see below); `scripted.py` + `scripted/*.json` load fixed, instructor-selected drills instead |
| `scoring_engine/` | Auditable decision-tree scorer: grades detection time, classification accuracy, and engagement decisions against ground truth, flags critical misses (threat not engaged) and wasted engagements |
| `aar_dashboard/` | SQLite-backed session store + a dashboard page (per-trainee score trend over time, per-unit leaderboard) |
| `trainer_frontend/` | `index.html` — desktop exercise UI (SVG scope, click-to-detect, classify, engage/monitor/ignore, "AI sensor hint", scripted-drill picker). `vr.html` — the same exercise loop rendered in WebXR (Three.js scene, controller/trigger-select to tag a contact, a DOM-overlay panel for classify/engage) |
| `ml_bridge/` | Wires the two reused repos' real engines into the live scenario/scoring loop (see below) |
| `server.py` | FastAPI app tying everything together + serving all three frontends |
| `radar_classification/` | Reused from RadarSense: real radar DSP + ML classifier, FastAPI backend, Next.js frontend, trained model artifacts — the ONNX model is loaded live by `ml_bridge` |
| `swarm_simulation/` | Reused from ares-x: multi-UAV swarm simulator (allocator/relay/recovery) — driven live by `ml_bridge` for swarm-attack contacts |

## How day/night, terrain, and sensor degradation actually affect the exercise

`scenario_engine/generator.py` folds `time_of_day` and `terrain` into an
`effective_sensor_degradation` on top of the manually chosen
`sensor_degradation` slider (night +0.35, dusk +0.15 of a penalty; urban
+0.25, mountain +0.15, coastal +0.05 of a penalty — scaled 0.6× and added).
That effective value drives: how much position noise/dropout the trainee's
sensor picture gets (`sanitize_for_client`), and how much feature noise the
real ONNX classifier's input gets (`ml_bridge.radar_classifier`). Urban and
mountain terrain also spawn extra decoys/birds as clutter. Verified:
day+rural → 0.0 effective degradation; night+urban → 0.36 even with the
slider at 0; night+urban+0.3 slider → 0.66.

## How the ML/simulation engines are actually wired in (`ml_bridge/`)

**`ml_bridge/radar_classifier.py`** — scenario_engine's contacts carry ground
truth (what they actually are); a real sensor wouldn't know that. This
module stands in for the missing radar front end: it derives RadarSense's
frozen 11-feature schema (doppler velocity/freq shift, RCS estimate,
micro-Doppler bandwidth/jitter, range, azimuth, altitude, velocity variance,
acceleration, SNR) from a contact's kinematics plus a type-conditioned
physical signature prior, adds noise scaled by the scenario's
`sensor_degradation`, and runs the **real trained CatBoost ONNX model**
(`radar_classification/artifacts/onnx/radarsense.onnx`, loaded via
`onnxruntime`) to get a classification. The trainee sees this model's call
via the "AI sensor hint" button (`POST /api/contact/classify`) — it can be
wrong, especially for decoys, exactly like a real classifier would be. If
`onnxruntime` isn't installed, it falls back to a simple heuristic so the
trainer still runs, but the real model is what actually loads and executes
when the dependency is present (verified: see `source: "radarsense_onnx"`
in the response).

**`ml_bridge/swarm_kinematics.py`** — swarm-attack contacts are no longer
straight lines. This module builds a real `ares.scenario.Scenario` (the
"PoIs" are points around the asset the swarm is converging on) and runs the
**actual `ares.sim.Simulator`** from `swarm_simulation` for the scenario's
duration, with a scripted `UAV_FAILURE` event injected partway through
(an automated point-defense kill). The remaining swarm members' paths come
from ares's real allocator/relay/recovery logic reacting to that loss —
not a scripted animation. If `ares` can't be imported (e.g. PyYAML
missing), `scenario_engine` falls back to synthetic straight-line swarm
paths so the trainer keeps working either way.

## Decision-tree scoring logic (`scoring_engine/scorer.py`)

Per contact: detection (full credit within 20s of spawn, decaying linearly
after), classification (full credit exact match, half credit for right
hostile/non-hostile bucket, zero otherwise — graded against the trainee's
own final call, not the AI hint, so relying on a wrong hint is still scored
honestly), and engagement (correct action per threat level — high→engage,
low→monitor, none→ignore — with explicit `threat_not_engaged` and
`wasted_or_fratricide_engagement` flags). Weighted 25/35/40 into an overall
0–100 score. Every number is traceable back to a specific contact and rule,
by design, so an AAR can explain exactly why a trainee lost points.

## Problem-statement coverage

| Requirement | Where |
|---|---|
| Desktop simulator | `trainer_frontend/index.html` |
| VR-capable | `trainer_frontend/vr.html` — WebXR (Three.js), controller-select to detect, DOM-overlay panel for classify/engage |
| Scripted scenarios | `scenario_engine/scripted.py` + `scenario_engine/scripted/*.json` (3 fixed drills) |
| Procedurally generated scenarios | `scenario_engine/generator.py` |
| Day/night, degraded sensors, urban/rural terrain | `effective_degradation()` in `generator.py` — see above, has a measured effect |
| Single-drone and swarm attacks | `generator.py` (`hostile_drone` contacts) + `ml_bridge/swarm_kinematics.py` (real ares sim) |
| Minimal specialized hardware | Runs in any browser; VR mode degrades gracefully to a plain 3D view without a headset |
| Decision-tree scoring: detection time / classification accuracy / engagement decisions | `scoring_engine/scorer.py` |
| AAR dashboard, individual/unit performance over repeated sessions | `aar_dashboard/` |
| Difficulty + randomization to prevent rote learning | `difficulty` param + procedural generation; scripted drills exist as the deliberate counterpart for fixed repeatable drills |

## Known scope limits (honest, not hidden)

- The ares swarm sim reacts to one **scripted** failure event, not live to
  the trainee's own engagement timing mid-flight — making it fully reactive
  would mean running the sim stepwise against real-time trainee input
  instead of precomputing the whole trajectory up front. The sim itself is
  real and running; only the trigger timing is scripted rather than
  interactive.
- The synthetic radar feature derivation in `radar_classifier.py` is a
  stand-in for RadarSense's real DSP front end (range-Doppler FFT, CA-CFAR,
  Kalman) — it's not reading a real I/Q stream, since this trainer has no
  physical radar. The model it feeds is the real, trained artifact.
- VR mode was built and code-reviewed against the WebXR spec but not
  hardware-tested on a physical headset in this environment (none
  available here) — the non-VR desktop 3D fallback in the same file was
  verified.
