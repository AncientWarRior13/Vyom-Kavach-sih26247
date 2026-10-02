# DroneThreatTrainer

A desktop drone & swarm threat recognition training simulator: procedurally
generated scenarios, decision-tree scoring on detection time / classification
accuracy / engagement decisions, and an after-action-review (AAR) dashboard
tracking individual/unit performance over repeated sessions.

## Status: fully built and wired, real ML/sim engines included

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8090
```

Open `http://localhost:8090/` for the trainee exercise screen, and
`http://localhost:8090/aar` for the after-action-review dashboard.

Flow: pick difficulty/time-of-day/terrain/sensor-degradation → "New scenario"
→ click blips on the scope as you spot them, optionally request an "AI
sensor hint", classify each contact, choose engage/monitor/ignore →
"Finish & score" → the breakdown appears, and shows up immediately in the
AAR dashboard under that trainee/unit ID.

## Layout

| Folder | Role |
|---|---|
| `scenario_engine/` | Procedurally generates scenarios: hostile drones, swarm attacks, decoys, birds, friendly drones, with day/night/terrain/sensor-degradation parameters and a `seed` for reproducible scripted runs |
| `scoring_engine/` | Auditable decision-tree scorer: grades detection time, classification accuracy, and engagement decisions against ground truth, flags critical misses (threat not engaged) and wasted engagements |
| `aar_dashboard/` | SQLite-backed session store + a dashboard page (per-trainee score trend over time, per-unit leaderboard) |
| `trainer_frontend/` | Trainee-facing exercise UI: a live radar scope (SVG), click-to-detect, classify dropdown, "AI sensor hint" button, engage/monitor/ignore buttons, session timer |
| `ml_bridge/` | Wires the two reused repos' real engines into the live scenario/scoring loop (see below) |
| `server.py` | FastAPI app tying everything together + serving both frontends |
| `radar_classification/` | Reused from RadarSense: real radar DSP + ML classifier, FastAPI backend, Next.js frontend, trained model artifacts — the ONNX model is loaded live by `ml_bridge` |
| `swarm_simulation/` | Reused from ares-x: multi-UAV swarm simulator (allocator/relay/recovery) — driven live by `ml_bridge` for swarm-attack contacts |

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
