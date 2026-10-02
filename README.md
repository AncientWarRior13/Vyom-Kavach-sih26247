# DroneThreatTrainer

A desktop drone & swarm threat recognition training simulator: procedurally
generated scenarios, decision-tree scoring on detection time / classification
accuracy / engagement decisions, and an after-action-review (AAR) dashboard
tracking individual/unit performance over repeated sessions.

## Status: functional end-to-end

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8090
```

Open `http://localhost:8090/` for the trainee exercise screen, and
`http://localhost:8090/aar` for the after-action-review dashboard.

Flow: pick difficulty/time-of-day/terrain/sensor-degradation → "New scenario"
→ click blips on the scope as you spot them, classify each contact, choose
engage/monitor/ignore → "Finish & score" → the breakdown appears, and shows
up immediately in the AAR dashboard under that trainee/unit ID.

## Layout

| Folder | Role | Status |
|---|---|---|
| `scenario_engine/` | Procedurally generates scenarios: hostile drones, swarm attacks, decoys, birds, friendly drones, with day/night/terrain/sensor-degradation parameters and a `seed` for reproducible scripted runs | **Built** |
| `scoring_engine/` | Auditable decision-tree scorer: grades detection time, classification accuracy, and engagement decisions (engage/monitor/ignore) against ground truth, flags critical misses (threat not engaged) and wasted engagements | **Built** |
| `aar_dashboard/` | SQLite-backed session store + a dashboard page (per-trainee score trend over time, per-unit leaderboard) | **Built** |
| `trainer_frontend/` | Trainee-facing exercise UI: a live radar scope (SVG), click-to-detect, classify dropdown, engage/monitor/ignore buttons, session timer | **Built** |
| `server.py` | FastAPI app wiring all four together + serving both frontends | **Built** |
| `radar_classification/` | Reused from RadarSense: real radar DSP + ML classifier, FastAPI backend, Next.js frontend, trained model artifacts | Reused as-is, **not yet wired in** |
| `swarm_simulation/` | Reused from ares-x: multi-UAV swarm simulator (kinematics, comm graph, relay/recovery), scenario YAMLs | Reused as-is, **not yet wired in** |

## What's intentionally not wired in yet

`scenario_engine` currently generates its own lightweight contact
kinematics rather than driving the much heavier `swarm_simulation/ares`
engine, and scoring is self-contained rather than calling out to
`radar_classification`'s trained classifier. That was a deliberate scoping
call to get a working, scorable, playable loop first. Two concrete upgrade
paths once the core loop is validated:

1. **Richer swarm behavior** — replace `scenario_engine`'s straight-line
   contact paths with `swarm_simulation/ares`'s allocator/relay/recovery
   simulator for swarm members, so swarm attacks exhibit real
   reconfiguration behavior (relay reassignment, response to a "shoot-down")
   instead of scripted motion.
2. **Real sensor realism** — instead of `sanitize_for_client`'s synthetic
   position noise, route contacts through `radar_classification`'s DSP
   pipeline (range-Doppler/CFAR/Kalman) so "sensor degradation" is a real
   simulated radar signal, and use its classifier as a difficulty-scaled
   hint/assist rather than keeping classification 100% manual.

Neither upgrade is required for the trainer to work today — both are pure
additions behind the existing `scenario_engine` / `scoring_engine`
interfaces.

## Decision-tree scoring logic (`scoring_engine/scorer.py`)

Per contact: detection (full credit within 20s of spawn, decaying linearly
after), classification (full credit exact match, half credit for right
hostile/non-hostile bucket, zero otherwise), and engagement (correct action
per threat level — high→engage, low→monitor, none→ignore — with explicit
`threat_not_engaged` and `wasted_or_fratricide_engagement` flags). Weighted
25/35/40 into an overall 0–100 score. Every number is traceable back to a
specific contact and rule, by design, so an AAR can explain exactly why a
trainee lost points.
