# DroneThreatTrainer

A desktop- and VR-capable drone & swarm threat recognition training
simulator: scripted and procedurally generated scenarios across
day/night, degraded-sensor, and urban/rural/coastal/mountain terrain
conditions, decision-tree scoring on detection time / classification
accuracy / engagement decisions, and an after-action-review (AAR) dashboard
tracking individual/unit performance over repeated sessions.

## Status: fully built, every problem-statement requirement covered

## Quickstart (one command, does everything)

**Windows:** double-click `start.bat` (or run it from a terminal).
**Mac/Linux:** run `./start.sh` from a terminal.

Either one creates a virtual environment if needed, installs/updates every
dependency, picks a free port (8090, falling back to 8099 if that's taken),
starts the server, and opens the trainer in your browser automatically.
Nothing else to do first — no manual `pip install`, no activating a venv
yourself. If something's wrong with your Python install, the script tells
you exactly what, instead of a raw traceback.

### Manual setup (if you'd rather control each step)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\pip install -r requirements.txt
# Mac/Linux: .venv/bin/pip install -r requirements.txt
pip install -r requirements.txt
uvicorn server:app --reload --port 8090
```

Requires **Python 3.10+**. If `pip install` fails partway through, re-run
it — it's safe to run repeatedly, it just skips what's already installed.

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
5-button action ladder.

## Tier-2 roadmap features

1. **Multi-sensor fusion** (`ml_bridge/multi_sensor.py`) — EO/IR, RF, and
   acoustic channels alongside radar. Each scenario randomly jams/degrades
   one non-radar channel for its duration (`scenario["sensor_outage"]`,
   shown to the trainee), forcing cross-cueing with whichever channels
   remain. EO/IR degrades at night and in cluttered terrain and can only
   tell gross shape (bird vs. drone), never intent — a real sensor
   limitation, not a shortcut. `POST /api/contact/sensor_read` serves all
   three; radar stays on the existing `/api/contact/classify`.
2. **Adaptive red-team tactics** (`scenario_engine/generator.py`) — a
   `red_team_tactics` flag (auto-enabled ~50% of the time at difficulty≥4)
   layers in feints (approach then peel away before engagement range),
   swarm splitting (two independent ares-driven subgroups converging from
   different launch points — each gets its own reactive-engagement replay
   metadata, verified independently reactive to separate kills), terrain-
   masked low approaches (very low altitude + higher radar dropout chance),
   RF-silent drones (invisible to the RF channel specifically), and decoy
   saturation (+80% decoy count). `scenario["red_team_tactics_used"]`
   records which fired, for instructor/AAR visibility.
3. **Instructor console** (`/instructor`, `trainer_frontend/instructor.html`)
   — lists currently-active trainee scenarios, injects a new threat (any
   contact type) into a running session at a chosen time, and jams/clears a
   sensor mid-session. The trainee page polls `GET
   /api/scenario/{id}/refresh` every 3s to pick up both without a reload.
4. **AAR replay with ghost track** — sessions now persist the full
   ground-truth scenario and the trainee's raw action log (not just the
   score), enabling `GET /api/aar/session/{id}/replay` for timeline/ghost-
   track playback. The AAR dashboard also shows an **error taxonomy**
   (counts of each scoring-flag type), a **missed-contact heatmap** (canvas
   gradient plot of where contacts go undetected), and a **CSV export**;
   "PDF export" is a print-friendly `/aar/session/{id}/print` page (browser
   print-to-PDF) rather than a bundled PDF-generation dependency.
5. **Certification threshold** — `GET /api/aar/certification/{trainee_id}`
   grades pass/fail readiness (default: avg ≥80 over the last 5 sessions);
   `GET /api/aar/certification/unit/{unit_id}` rolls that up per-unit. Shown
   as a badge on the AAR dashboard.

All five verified with a 16-check HTTP integration pass plus real-browser
click-throughs of the multi-sensor buttons and the updated AAR dashboard
(certification badges, error taxonomy, report links all confirmed
populating from live data).

## Tier-3 roadmap features

11. **Team roles with comms delay** (`/team`, `trainer_frontend/team.html`)
    — a scenario can be played by three separate operators (detector,
    classifier, engagement authority) instead of one trainee doing
    everything. Each role only learns what the previous role found after
    `team_comms_delay_s` (default 5s): a classifier can't act on a contact
    until the detector's detection has had time to "reach" them, and the
    engagement authority can't act until the classifier's call has. `POST
    /api/team/action` enforces this server-side (returns 403 with the wait
    time if a role acts too early); `GET /api/team/state` filters what each
    role can even see. No new scoring axis was needed — the delay just
    makes the existing detection-window and timeliness scoring harder,
    honestly, by adding real relay latency. `POST /api/session/submit_team`
    assembles the standard trainee_log from all three roles' actions and
    scores it through the same `scoring_engine` as a solo session. Open the
    page in up to three tabs/windows with the same scenario ID (one role
    per tab) to actually play it as a team.
12. **Offline mode and audit** — session records now form a hash chain
    (`aar_dashboard/store.py`): each row's hash covers its own data (every
    score column, not just the JSON blob — a tampering test against a
    column-only edit was caught and fixed during verification) plus the
    previous row's hash, so editing or reinserting a past session breaks
    every hash after it, detectably, with no central server to cross-check
    against (`GET /api/audit/verify_chain`, with a "Verify audit chain"
    button on the AAR dashboard). Scripted scenarios can be exported as
    **signed packs** (`scenario_engine/signing.py`, HMAC-SHA256 over the
    scenario JSON, `GET /api/scenario/scripted/{name}/pack`) — verifiable
    (`POST /api/scenario/pack/verify`) without any network connection,
    for distributing scenario content to an air-gapped site. (The backend
    itself makes no runtime network calls; full air-gap for the frontend
    would additionally need vendoring the two CDN-hosted libraries,
    Three.js and Chart.js — not done, flagged honestly rather than silently
    glossed over.)
13. **Spoofed-signature scenarios** (`ml_bridge/radar_classifier.py`) — a
    contact can carry `spoofed_as`, the contact type whose radar signature
    it presents instead of its own ground truth (e.g. a decoy spoofed to
    mimic a bird) — directly exercising RadarSense's adversarial-evasion
    framing. Only the radar/ML signature is spoofable; EO/IR's visual shape
    reading is deliberately left reading the true type, since a camera
    can't be fooled by RF spoofing — a trainee who cross-cues with EO/IR
    catches what pure-radar trust would miss. Verified: the real trained
    classifier calls a spoofed decoy "bird" far more consistently (13/15
    samples) than an ordinary decoy (6/15).

All three verified: a 15-check HTTP integration pass for team roles
(including the exact 403-with-wait-time rejection and successful action
after the delay, confirmed through a real browser clicking through all
three roles in sequence), a tamper-injection test proving the audit chain
actually catches an edited score (not just a theoretical hash), and a
before/after classifier comparison proving the spoofing has a measurable
effect on the real model's output.

## Model training-performance section (roadmap sections 3-5)

1. **Faster CatBoost + real early stopping** — `radar_classification/radarsense/feat31_train_300.py`
   previously trained a fixed 500 iterations despite passing an `eval_set`,
   so early stopping never fired. Replaced its model construction with the
   roadmap's exact config (`iterations=2000, learning_rate=0.15,
   border_count=64, early_stopping_rounds=40, use_best_model=True`).
   Verified directly: on a synthetic dataset, training stopped at iteration
   25 of 2000 in 0.23s — early stopping actually engages.
2. **Feature/dataset caching** and **3. parallelized I/Q synthesis**
   (`ml_bridge/training_utils.py`) — `cached_features()` memoizes a
   `build_fn(cfg)` DataFrame to parquet keyed by a hash of `cfg`, so
   unchanged config is never regenerated; `parallel_map()` runs a per-sample
   function (I/Q synthesis + CFAR is CPU-bound and independent per sample)
   across cores via `joblib`, falling back to a serial loop if `joblib`
   isn't installed. Verified working.
4. **Warm-start retraining** — covered by the roadmap's own documented
   pattern (`CatBoostClassifier().load_model(...)` then `fit(...,
   init_model=base)`); not separately wired into this trainer since there's
   no live trainee-data-driven retraining loop that would call it yet (see
   #5's honest caveat below).
5. **Background retrain + hot-swap** (`ml_bridge/radar_classifier.py`) —
   `retrain_async()` runs a training function in a background thread,
   never blocking the classifier; `GET /api/model/status` reports
   idle/training/trained/timeout/failed; on success, the serving ONNX
   session is atomically swapped under a lock — `classify_contact` never
   blocks on it. Verified all three outcomes directly (success swaps the
   model and keeps classifying; a timeout keeps the old model; a raised
   exception keeps the old model) and via a full HTTP round-trip
   (`POST /api/model/retrain` with a real script → `GET /api/model/status`
   → classify still works post-swap).
6. **Wall-clock cutoff via subprocess** (`ml_bridge/training_utils.py`'s
   `train_with_cutoff`) — runs the training script as a real OS subprocess
   with `subprocess.run(..., timeout=...)`, so a runaway job is actually
   killed, not just abandoned (a Python thread can't force-kill code stuck
   in a C extension; a subprocess can be). Verified: a deliberately slow
   dummy script (5s sleep) was killed at its 1s budget, confirmed by
   `duration_s: 1.0` in the status, not 5.
7. **Precomputed AI hints at scenario creation** — every `/api/scenario/*`
   creation endpoint now kicks a background thread
   (`ml_bridge.radar_classifier.precompute_hints`) that runs the real
   CFAR+ONNX path for each contact at a few random draws across its active
   window; `/api/contact/classify` answers from that cache when available
   (near-instant), falling back to a live compute if asked before the
   background pass finishes. This trainer's classification is already
   randomized-by-design per draw (synthesized I/Q noise), so serving a
   cached draw instead of a fresh one changes nothing the trainee
   experiences. Verified via HTTP immediately after scenario creation and
   again after the background pass should have completed.
8. **ares adaptive controller (numpy MLP, DAgger) training speed** — not
   touched. This trainer deliberately does not integrate ares-x's adaptive
   flight controller at all (see the Tier-2 section above — it solves a
   precision-tracking problem this trainer's one-way attacker swarm doesn't
   have), so speeding up training code that's never invoked here would be
   dead-code optimization. Flagged explicitly rather than silently skipped.
9. **Cheaper inference (optional)** — RadarSense's own `export_onnx.py`
   already benchmarks an int8 dynamically-quantized ONNX variant alongside
   the full-precision one (discovered while implementing this section, not
   something this round added) — nothing further was needed here.

## Research-grade additions (`research/PROTOCOLS.md`)

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
