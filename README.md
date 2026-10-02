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
| `research/` | Research-grade additions grounded in the literature (see `research/PROTOCOLS.md`): two live A/B experiments (AAR-timing, scenario-sequencing) with stable arm assignment and an analysis export endpoint, plus a swarm-density-aware classifier feature |
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

**`ml_bridge/radar_classifier.py` + `ml_bridge/iq_synthesizer.py`** —
scenario_engine's contacts carry ground truth (what they actually are); a
real sensor wouldn't know that, and there's no physical radar here to
capture real I/Q. Rather than stop at formula-estimated features,
`iq_synthesizer.py` embeds a contact's true radial velocity, RCS, and a
type-conditioned rotor/wing micro-motion signature as actual
frequency-domain content in a synthesized 77GHz FMCW I/Q segment, then runs
it through **`radar_classification/radarsense/radar_dsp.py`'s real
range-Doppler FFT + CA-CFAR detector** — the same signal-processing chain
RadarSense validated on real 77GHz data. `doppler_velocity`, `snr_db`, and
`micro_doppler_bandwidth`/`jitter` are therefore *measured* by that real
DSP off synthesized-but-physically-modeled I/Q (and can come back
bin-quantized, noisier than the ground truth, or even undetected at low
SNR — a genuine missed detection), not assigned from a lookup table.
`altitude`/`range`/`azimuth` stay direct geometry (correctly — CFAR doesn't
need to "recover" those). The resulting 11-feature vector runs through the
**real trained CatBoost ONNX model**
(`radar_classification/artifacts/onnx/radarsense.onnx`, via `onnxruntime`).
The trainee sees this model's call via the "AI sensor hint" button
(`POST /api/contact/classify`) — it can be wrong, especially for decoys,
exactly like a real classifier would be. Verified: classifying the same
kinematics multiple times shows measurably different CFAR-detected
bandwidths across contact types (bird vs. drone vs. aircraft), confirming
real signal processing — not a formula — produces the feature values.

**`ml_bridge/swarm_kinematics.py`** — swarm-attack contacts are no longer
straight lines, and no failure is scripted. This module builds a real
`ares.scenario.Scenario` (the "PoIs" are points around the asset the swarm
is converging on) and runs the **actual `ares.sim.Simulator`** from
`swarm_simulation`. Swarm construction is fully seeded (a real bug fix: the
previous version took a `seed` parameter but used Python's *global* RNG for
scenario layout, so a given seed didn't actually reproduce anything — now
every random choice goes through a `random.Random(seed)` instance local to
that build). A swarm member only goes down when the **trainee actually
engages it** — `server.py`'s `POST /api/contact/engage` fires the instant
that happens (not just at session scoring), deterministically replays the
swarm's simulation from t=0 up to that moment (same seed ⇒ byte-identical
to what the trainee already saw), injects a real `UAV_FAILURE` at that
exact instant, and lets ares's actual recovery/relay logic react forward
from there, live. Verified: engaging a swarm member mid-flight leaves every
other contact's pre-engagement path exactly unchanged, and produces a
measurable position divergence for the rest of the swarm afterward — a real
reallocation reacting to when and which member was shot down, not a
scripted timer. If `ares` can't be imported (e.g. PyYAML missing),
`scenario_engine` falls back to synthetic straight-line swarm paths so the
trainer keeps working either way (engagement on those is scored normally
but has no live reallocation to react with).

## Decision-tree scoring logic (`scoring_engine/scorer.py`)

Four axes per contact, weighted 20/25/35/20 into an overall 0–100 score:
**detection** (full credit within 20s of spawn, decaying linearly after),
**classification** (full credit exact match, half credit for right
hostile/non-hostile bucket, zero otherwise — graded against the trainee's
own final call, not the AI hint, so relying on a wrong hint is still scored
honestly), **engagement/ROE** (graduated warn/soft-kill/hard-kill response
checked against threat level, no-engage zones, and warn-before-escalation
requirements — see the Tier-1 roadmap section below for the full doctrine),
and **timeliness** (engagement latency scored against each hostile
contact's real time-to-impact window, not a fixed clock). An `ai_trust`
block is reported alongside the score (not folded into it) categorizing
whether the trainee followed or overrode each AI hint they requested, and
whether that hint was right. Every number is traceable back to a specific
contact and rule, by design, so an AAR can explain exactly why a trainee
lost points.

## Tier-1 roadmap features (`DroneThreatTrainer_Roadmap_and_Training_Plan.docx`)

A later review identified the main defense-credibility gap: the trainee's
only action was engage/monitor/ignore, and scoring didn't consider rules of
engagement, time pressure, or over-reliance on the AI hint. Five features
(the roadmap's Tier 1, "highest impact for the effort") close that gap —
all live in `scoring_engine/scorer.py` and `scenario_engine/generator.py`:

1. **Rules-of-engagement (ROE) layer** — every scenario now generates 1-2
   fixed no-engage zones (hospital/school/civilian-housing/marketplace) and
   an ROE card (warn-before-hard-kill requirement, a time-critical
   self-defense exception, max effective ranges). A `hard_kill` inside a
   no-engage zone is a violation regardless of threat level; skipping the
   required warn before a non-time-critical hard-kill is a procedural
   violation (partial credit, not zero — the kill may still be materially
   correct). Shown to the trainee as dashed red circles on the scope and a
   plain-text ROE summary.
2. **Time-to-impact scoring** — a new `timeliness_score` axis computes each
   hostile contact's actual closing-speed ETA to the defended asset and
   scores engagement latency against that real window instead of a fixed
   clock; an engagement that lands after the threat would have reached the
   asset scores 0 and raises a `threat_reached_asset` flag.
3. **Graduated response ladder** — `warn` → `soft_kill` (jam/spoof) →
   `hard_kill` (kinetic), replacing the old single `engage` action
   (`engage` is kept as a scored legacy alias for `hard_kill`). Each tier
   has its own effective range and correct-use doctrine per threat level
   (decoys: warn/soft-kill/monitor correct, hard-kill is overkill;
   friendlies/birds: monitor/ignore correct, any kill is fratricide).
4. **AI trust calibration** — every `ai_hint_requested`/`ai_hint_label` is
   logged and compared against the trainee's own final classification,
   categorized into followed-correct / overrode-correct / followed-wrong /
   overrode-wrong. Session results include an `ai_trust` block with an
   `automation_bias_score` (fraction of hints that were wrong *and*
   followed anyway) — reported, not folded into the score, since the
   roadmap asks to measure it, not penalize it.
5. **Adaptive curriculum** — `POST /api/scenario/adaptive` (an "Adaptive
   scenario" button on the trainer page) pulls a trainee's last 10
   sessions' per-contact-type scores from the AAR store and weights the
   next scenario's contact counts toward whatever they're weakest on,
   instead of pure random repetition. Verified: a trainee who missed every
   contact across several sessions gets measurably upweighted spawn counts
   (`{"hostile": 2.5, "decoy": 1.94, ...}`) on their next adaptive scenario.

All five verified with an 18-case unit-test sweep of every scoring branch
(ROE violations, time-critical exception, graduated-response correctness
per threat tier, timeliness scoring, automation-bias categorization) plus
a full HTTP integration pass and a real-browser click-through of the new
5-button action ladder. Tier 2/3 of the roadmap (multi-sensor fusion,
instructor console, AAR replay, certification thresholds, team roles,
offline/audit mode, spoofed-signature scenarios) and the model
training-performance section are not yet implemented — each is
independently large; ask for them specifically when ready.

## Research-grade additions (`research/PROTOCOLS.md`)

Four directions were identified from the published literature (reviewed via
Consensus) closest to this trainer's design — chosen because the evidence
either doesn't exist for this task domain yet or is genuinely contested:

1. **AAR-timing ablation** (`aar_ablation`, live A/B experiment) — tests
   whether showing the score breakdown immediately vs. withholding it for a
   later debrief changes outcomes, building on AAR's large documented
   effect (d=0.67–0.92 across meta-analyses) but testing the one variable
   (timing) that hasn't been isolated for this task type.
2. **Scenario-sequencing experiment** (`sequencing`, live A/B experiment) —
   blocked (ramping) vs. randomized difficulty order per attempt, testing
   the contested contextual-interference effect for a decision-under-
   uncertainty task, which no prior study covers (existing evidence is all
   motor-skill tasks: aiming, surgery, music).
3. **Swarm-density-aware classification** (implemented in `ml_bridge`) —
   nearby swarm members measurably widen each other's synthesized
   micro-Doppler signature before hitting the real trained classifier,
   modeling an effect the (thin) swarm-classification literature doesn't
   address at all.
4. **Transfer-validity study** (protocol only, for a live human-subjects
   run) — the two-arm design, sample-size estimate, and analysis plan for
   testing whether simulator training here actually transfers to live-drill
   performance, reusing this app's own `scoring_engine` as the assessment
   instrument.

Both live experiments are opt-in (a "Participate in research" checkbox on
the trainer page), use stable hash-based arm assignment so a trainee stays
in the same condition across sessions, and are queryable for offline
analysis via `GET /api/research/results/{aar_ablation|sequencing}`. Full
citations in `research/PROTOCOLS.md`.

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

## Previously known scope limits — resolved

These were honestly flagged in earlier versions of this README and have
since been closed, with evidence:

- ~~The ares swarm sim reacts to one scripted failure event, not live to
  the trainee's own engagement timing.~~ **Fixed** — `POST
  /api/contact/engage` makes a trainee's own engagement the only way a
  swarm member fails, via deterministic replay-and-diverge of the real
  ares sim (see above). Verified: pre-engagement paths are byte-identical
  across repeated engagement scenarios; post-engagement paths measurably
  diverge due to real reallocation; engaging a target before its own spawn
  time clamps sensibly instead of crashing; a full sequential swarm
  wipeout (every member engaged one by one) completes cleanly; the
  `dusk_rural_swarm_raid` scripted drill — generated before this feature
  existed — was regenerated so it carries the reactive-engagement metadata
  too (the other two scripted drills were also regenerated, removing an
  incidental random-chance swarm that had snuck into the decoy-mix drill,
  keeping the three drills meaningfully distinct: single-drone / swarm-raid
  / decoy-mix).
- ~~The synthetic radar feature derivation is a formula stand-in for
  RadarSense's real DSP front end.~~ **Fixed** — `iq_synthesizer.py` now
  synthesizes physically-modeled I/Q and runs it through the project's
  actual range-Doppler FFT + CA-CFAR chain (`radar_dsp.py`) to *measure*
  doppler velocity, SNR, and micro-Doppler bandwidth/jitter, rather than
  assigning them. Verified: measured bandwidth differs meaningfully across
  contact types from the same DSP code path.

## ares-x's adaptive flight controller — deliberately not integrated

ares-x gained a trained adaptive flight controller (MLP-based sim-to-real
correction) and a dynamic point-mass physics model after this project
copied `swarm_simulation/`. It was evaluated and **not** pulled in: it
solves precision *survey-task tracking* for a friendly swarm hitting PoIs
accurately, which has no equivalent in this trainer — the swarm here is a
one-way attacker with no survey target, so "precision tracking" isn't a
meaningful concept for it. The one piece that could still add value —
`ares/dynamics.py`'s point-mass physics for more inertia-realistic swarm
motion instead of instant-velocity kinematics — was left out too, as a
cosmetic upgrade not required by the problem statement, to avoid
integration risk on the now-verified reactive-engagement path above. Happy
to wire it in on request; flagging the decision here instead of silently
dropping it.

## Remaining limitation (not fixable from here)

- **VR mode has not been tested on a physical headset.** The code is
  written to the WebXR spec and was verified running correctly in a real
  browser — the Three.js scene initializes, the scenario/contact data
  pipeline populates the DOM-overlay panel from live data with zero JS
  errors, and `VRButton` correctly detects the absence of an XR device and
  reports "VR NOT SUPPORTED" rather than crashing. What's left — an actual
  headset entering immersive mode and a human pulling a controller trigger
  — needs physical VR hardware (or at minimum a WebXR device emulator),
  neither of which is available in this environment. This is a hardware
  constraint, not an engineering gap: there is nothing further to build
  here that would close it without that hardware.
