# ARES-X — Autonomous Resilient Emergency Swarm

Stage 1 prototype for the **UAV-X: Resilient BVLOS Swarm Challenge** (PUSHPAK
Grand Challenge 2026, Techfest IIT Bombay). Full design rationale lives in
[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md); implementation detail lives in
[docs/](docs/).

A communication-aware multi-UAV autonomy stack that surveys disaster PoIs
while keeping a resilient multi-hop link back to the GCS, dynamically
reassigning relay/survey roles and repairing the network graph after
failures — driven entirely by scenario files, with no scripted or faked
outcomes.

## Three layers, one autonomy core

| Layer | What it is | Status |
|---|---|---|
| `ares/` | Pure-Python offline simulator: kinematics + comm graph + allocator/relay/recovery/safety | **Working** — this is the source of truth, fully tested |
| `ares/control/`, `ares/dynamics.py`, `ares/flight.py` | Adaptive sim-to-real flight controller (online system identification) + point-mass plant + velocity-aware separation checking, opt-in via `Simulator(flight_model="dynamic")` | **Working, sim-only** — near-oracle single-vehicle tracking; no swarm-level difference on standard missions, large benefit (coverage 0.85 → 1.00 over 100 seeds) when survey precision matters. See `docs/adaptive_control.md` |
| `ros2_ws/` | The same autonomy code (`ares/`) wrapped in real ROS 2 nodes (`ares_msgs`, `ares_swarm_manager`, `ares_uav_agent`) talking over real topics | **Working** — builds clean from a fresh `rm -rf build install log`, all nodes launch, topic graph verified via a repeatable check (`scripts/ros2_smoke_test.sh`), not just a one-off manual session |
| `px4_workspace/` | PX4 SITL + Gazebo for physics-grounded telemetry | **Working end-to-end for one vehicle** — `ares_swarm_manager`'s real task allocator flew a real PX4 vehicle to a real PoI through Gazebo physics and correctly detected arrival, not just a manual test command. 3-vehicle Gazebo spawn + telemetry also proven; the ROS2 *control* path for vehicles beyond the first hit one specific, unresolved DDS bug (see `docs/simulation.md`) |

The autonomy code (`ares/allocator.py`, `relay.py`, `recovery.py`,
`safety.py`, `resilience.py`) is identical across all three — only the
source of UAV telemetry/motion changes. See `docs/architecture.md` for the
exact seam where PX4 plugs in.

## Quickstart — pure Python (always works, no external deps beyond PyYAML)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/run_simulation.py config/scenarios/baseline.yaml
.venv/bin/python -m pytest tests/ -q

# compare dynamic vs static baselines (section 31)
.venv/bin/python scripts/run_batch_experiment.py --scenario config/scenarios/baseline.yaml

# view results
.venv/bin/python dashboard/serve.py   # http://localhost:8765/dashboard/
```

Output lands in `results/raw/<scenario_name>_{metrics.json,timeseries.csv,events.json,events.csv}`
and `results/processed/comparison.{json,csv}`.

## Quickstart — ROS 2 distributed demo

```bash
# one-time: ROS 2 Jazzy via RoboStack conda (no official arm64 macOS ROS 2 build)
conda env create -f environment/ares_ros.yml
conda activate ares_ros
mamba install -y -c conda-forge -c robostack-jazzy colcon-common-extensions

cd ros2_ws && colcon build && source install/setup.zsh
export ARES_PROJECT_ROOT=$(cd .. && pwd)
ros2 launch ares_swarm_manager ares_swarm.launch.py scenario_path:=config/scenarios/baseline.yaml
```

See `docs/simulation.md` for what's been verified here vs. what's still a
known sandbox quirk (DDS `topic echo` flakiness, unrelated to the code).

## What the baseline scenario demonstrates

`config/scenarios/baseline.yaml`: 8 UAVs, 6 PoIs (+1 injected mid-run), 600s.

- **t=180s** — `uav_02` (a relay) is killed outright (`UAV_FAILURE`)
- **t=300s** — a link is degraded with 45% packet loss for 60s
- **t=400s** — a new priority-1.0 PoI appears
- **t=450s** — a UAV is recalled to base

None of this is scripted response — the fault detector, recovery planner,
and relay allocator react to the resulting graph state each timestep. The
`RECOVERY_STARTED`/`RECOVERY_COMPLETED` log entries and
`mean_reconfiguration_time_s` metric are measured, not asserted. See
`docs/experiments.md` for the actual (not cherry-picked) comparison
against static baselines — the resilience-potential-map policy wins
clearly on the 4 hand-built scenarios, but under proper statistical
testing across 40 randomized-geometry seeds it **trades connectivity
availability for packet delivery ratio**: PDR improves significantly
(p=0.004), but connectivity availability is significantly *worse*
(-10.1 points, p=0.026), not just "not proven better" — a real,
measured tradeoff a team has to deliberately choose, reported as the
project's actual headline finding.

## Architecture

```
ares/
├── scenario.py    Scenario/PoI/Event loading from YAML — no code changes for new scenarios
├── state.py        Canonical SwarmState / UAVState / PoI data model
├── network.py       Comm graph builder, PDR/latency model, connectivity health (§6, §8)
├── energy.py         Battery accounting + safety-reserve constraint (§12)
├── roles.py           RoleScore-style relay suitability scoring (§5)
├── allocator.py         TaskAllocator interface + greedy priority-based impl (§20)
├── relay.py               RelayAllocator: reactive graph repair + proactive pre-positioning (§9, §10)
├── resilience.py            Resilience Potential Map — predictive relay placement (§21)
├── baselines.py               Static Task/Relay allocators for the A/B comparison (§31)
├── recovery.py                  FaultDetector + RecoveryPlanner: detects failures, measures recovery time
├── safety.py                     SafetySupervisor: geofence, separation, battery reserve (§13) — deterministic, overrides everything else
├── metrics.py                     MetricsEngine + internal score estimator (§18, §19)
├── eventlog.py                    Structured explainable decision log (§33)
└── sim.py                          Event-driven simulation loop tying it all together

ros2_ws/src/
├── ares_msgs/            Custom messages/actions (UAVState, SwarmGraph, UAVCommand, SurveyPoI, ...) — builds with colcon/rosidl
├── ares_swarm_manager/   Level-1 node: same ares/ allocator+relay+recovery+safety, driven by ROS topics
└── ares_uav_agent/       Level-2 node: uav_agent_node.py (fake kinematics) or
                          px4_bridge_node.py (real PX4 telemetry/control) - same
                          ares_msgs/UAVState+UAVCommand interface either way

px4_workspace/PX4-Autopilot/   Full PX4 SITL source, cloned with submodules
patches/px4-disable-optical-flow-opencv5.patch   Required patch, see docs/simulation.md
```

Two-level split per §4: `sim.py` + `relay.py`/`allocator.py`/`recovery.py` are
the "swarm intelligence" layer; per-UAV kinematics/battery/safety in
`sim.py`/`safety.py`/`energy.py` are the "UAV autonomy" layer. They only
communicate through `SwarmState` (or, in `ros2_ws/`, through the
equivalent ROS 2 topics).

## Interfaces designed for extension (§20)

`TaskAllocator` and `RelayAllocator` are abstract base classes.
Implementations shipped: `GreedyPriorityAllocator`/`GraphRelayAllocator`
(proposed, dynamic) and `StaticTaskAllocator`/`StaticRelayAllocator`
(Baselines A/B). Swap in Hungarian assignment, min-cost flow, or a learned
policy later without touching `sim.py` or the ROS 2 nodes.

## Status vs. the 9-phase roadmap in PROJECT_CONTEXT.md

- **Phase 0-6:** done — scenario engine, multi-UAV sim, comm graph, mission
  planning, dynamic relay allocation, fault recovery, emerging-priority PoIs.
- **Phase 7:** done — resilience potential map (`resilience.py`), proactive
  pre-positioning, multi-hop chain repair, and a relay-commitment cap all
  implemented. Clearly wins on all 4 hand-built scenarios after several
  real bug fixes along the way (see `docs/experiments.md` and
  `docs/connectivity_analysis.md`), but a **paired statistical test
  across 40 randomized-geometry seeds shows it trades connectivity
  availability for packet delivery ratio**: PDR significantly better
  (p=0.004), connectivity availability significantly *worse* (-10.1
  points, p=0.026) - a real measured tradeoff, not an unproven-but-maybe
  win. Reported as the honest headline, not softened either direction.
  A second, independent research
  contribution — a provable, algorithm-independent lower bound on relay
  count (`ares/connectivity_bounds.py`), empirically validated with zero
  violations across 196 real gap-repair events (including a scenario
  built specifically to force `k_min ≥ 2`, which found and fixed a real
  bug: relay chains weren't re-extended as a gap UAV kept moving further
  away) — is in `docs/connectivity_analysis.md`. A third (`ares/link_predictor.py`,
  PROJECT_CONTEXT.md §34 Option B) is an honestly-reported **negative**
  result: a learned link-quality predictor, properly trained and
  evaluated, loses to a naive baseline by 14.2%, and is deliberately kept
  out of any allocator decision — see `docs/link_predictor_analysis.md`.
  It is wired into the live `Simulator` as an opt-in diagnostic observer
  (`enable_link_predictor_diagnostics=True`, off by default) so its
  predictions vs. actual PDR are runtime-observable, not untouched dead
  code — see `scripts/run_link_predictor_diagnostics.py`.
- **Phase 8:** done — `scripts/run_batch_experiment.py` runs the four
  policies from §31 across the 4 fixed scenarios or N randomized-geometry
  scenarios (`--generated N`, via `ares/scenario_generator.py`, so seeds
  actually vary topology). `scripts/statistical_analysis.py` adds paired
  bootstrap CIs and permutation tests instead of trusting raw means.
- **Phase 9 / PX4-Gazebo:** working end-to-end for one vehicle — real PX4
  SITL + Gazebo physics, with `ares_swarm_manager`'s actual task allocator
  (not a manual command) flying it to a real PoI and correctly detecting
  arrival. 3-vehicle Gazebo spawn + telemetry proven; the ROS2 control
  path for vehicles beyond the first hit one specific, unresolved DDS bug
  — see `docs/simulation.md`'s "Multi-vehicle SITL."

## Known limitations (be honest about these, not smoothed over)

- **The resilience map's connectivity benefit is not statistically
  significant on randomized geometry, even though it wins clearly on the
  4 hand-built scenarios.** 40-seed paired testing only supports a
  packet-delivery-ratio improvement; treat "the resilience map improves
  connectivity" as unproven at current sample size, not as an established
  result — see `docs/experiments.md`.
- ~~Relay allocation has no cap on how many UAVs it commits to relay
  duty~~ — **fixed**: `config/planner.yaml::max_relay_fraction` (default
  0.6) caps this. Mission completion on the randomized 30-seed sweep went
  from 99.0% → 100%, matching the static baselines exactly. See
  `docs/experiments.md`'s "The relay-commitment cap."
- **PX4 multi-vehicle: Gazebo spawn and telemetry work, ROS2 control
  doesn't yet for vehicles beyond the first.** 3 real PX4 vehicles ran
  simultaneously with correct per-instance telemetry; direct MAVLink
  control works on all of them, but the ROS2 DDS command topics
  (`vehicle_command`, `trajectory_setpoint`) only reliably reach instance
  0. One blocking bug found and fixed along the way (stale per-instance
  `rootfs/N` parameter state can hang PX4's startup before it even reaches
  `uxrce_dds_client start` — always `rm -rf` it before relaunching a
  killed instance). After that fix, DDS entity creation is confirmed
  correct for instance 1 (verified via the Agent's own log — 414 entities
  created, matching a healthy instance 0), yet command messages still
  don't reach PX4 — narrowed to routing after entity creation, not yet
  root-caused further. See `docs/simulation.md`'s "Multi-vehicle SITL."
- Multi-hop relay chaining is implemented and tested
  (`tests/test_relay_chain.py`) but only exercised by a synthetic unit
  test so far — no scenario yet has a gap wide enough to require 2+ chained
  relays in practice.
- `ros2 topic echo` was flaky against this specific sandboxed shell (DDS
  shared-memory transport contention between rapidly-spawned CLI
  processes) — verified instead via `ros2 node info` showing the correct
  typed pub/sub graph. Worth re-checking live echo on a normal machine.
