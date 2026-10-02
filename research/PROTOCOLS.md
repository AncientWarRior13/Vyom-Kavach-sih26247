# Research protocols

Four research-grade directions were identified by reviewing the published
literature closest to this trainer's design, via Consensus. Two are
implemented as live, running A/B experiments inside the app (not just
proposals) — `aar_ablation` and `sequencing`. One (swarm-density-aware
classification) is implemented as a model feature. One (transfer validity)
is a protocol for a study to run *outside* the app, with live subjects,
since it needs a real-world comparison condition this software can't
provide on its own.

All four exist because the underlying literature either (a) is thin or
nonexistent for this specific task domain (threat recognition/engagement
decisions, as opposed to surgery, sports, or single-drone-only radar
classification), or (b) is contested enough that running the test here
adds real evidence rather than restating a settled result.

---

## 1. Transfer-validity study (run outside the app, this app instruments it)

**Gap**: VR/simulation training reliably improves performance *on the
simulator* (faster, more accurate, better checklist scores), but evidence
of transfer to real-world task performance is "limited and heterogeneous"
even in the best-studied domain, surgery — see the 2026 systematic review
finding only low-to-moderate certainty for translation to live operative
performance (Sarmento et al., 2026) and the 2021 review noting the same gap
(Mao et al., 2021). **No equivalent evidence exists at all for military
threat-recognition/engagement-decision training** — every paper found is
medical or surgical.

**Design**: Two-arm, between-subjects. Recruit N trainees (power analysis:
for a medium effect size d=0.5 at α=0.05, power=0.8, need ~64 per arm, so
~128 total — smaller N is still publishable as a pilot with the limitation
noted). Randomize to:
- **Treatment**: K sessions on this trainer (recommend K≥5, spaced across
  ≥1 week, to leave room for retention effects per contextual-interference
  literature below) before the live assessment.
- **Control**: no simulator exposure, or a non-interactive briefing/slide
  equivalent of the same material (time-matched, to isolate the
  interactive-practice effect from mere exposure).

**Outcome measure**: a live or high-fidelity tabletop exercise (e.g. an
instructor-run drone-ID drill with real or simulated contacts the trainee
has never seen) scored on the *same three axes* this trainer already
uses — detection time, classification accuracy, engagement-decision
correctness — so the simulator's own scoring rubric doubles as the
transfer-validity instrument. This is the one piece of infrastructure this
codebase can't provide (a live assessment), so this protocol is written
for a human researcher to execute; this app's `scoring_engine` can be
reused as-is to grade the live assessment consistently with simulator
sessions.

**Analysis**: independent-samples t-test (or Mann-Whitney if scores are
non-normal) on live-assessment scores, treatment vs. control. Report effect
size (Cohen's d) alongside p-value, per current best practice in this
literature (Sarmento et al. explicitly flag GRADE/imprecision issues from
underpowered prior studies — don't repeat that mistake).

---

## 2. AAR-timing ablation (`aar_ablation` — implemented, live in the app)

**Gap**: AAR/debriefing is one of the best-evidenced training interventions
in the literature (d=0.67–0.92 across three independent meta-analyses:
Tannenbaum & Cerasoli 2013, Keiser & Arthur 2020/2022), and effectiveness
depends on objective-vs-subjective review media and individual-vs-team
alignment. This trainer already does *objective, individual-aligned* AAR
(exactly the combination Keiser et al. found strongest) — what's untested
is the **timing**: does showing the scored breakdown immediately after a
session change anything for this specific task, versus withholding it for
a later batched debrief?

**Implementation**: `aar_dashboard.get_or_assign_arm("aar_ablation",
trainee_id, ["aar_shown", "aar_withheld"])` — a stable hash-based 50/50
split, persisted in the `experiment_assignments` table so a trainee stays
in the same arm across all their sessions. A trainee opts in via the
"Participate in research" checkbox on the trainer page
(`trainer_frontend/index.html`). `server.py`'s `/api/session/submit`:
- always scores and stores the session (so unit/trainee trend data is
  complete either way)
- for `aar_shown`: returns the full breakdown immediately, as normal
- for `aar_withheld`: returns only a confirmation message; the score is
  recorded but not shown to the trainee until a later instructor-led
  debrief (which an instructor can pull via `/api/research/results/aar_ablation`
  or the normal AAR dashboard, trainee by trainee)

**Analysis**: pull `/api/research/results/aar_ablation`, compare
`overall_score` trajectories (attempt_number 1..N) between arms with a
mixed-effects model (random intercept per trainee) or, for a quick check, a
repeated-measures ANOVA on improvement from attempt 1 to attempt N.

---

## 3. Scenario-sequencing experiment (`sequencing` — implemented, live in the app)

**Gap**: contextual interference (random vs. blocked practice order) shows
a real retention benefit in lab motor-learning settings (Shea & Morgan,
1979; medium effect in recent meta-analyses: Czyż et al. 2024, SMD≈0.55 for
transfer) but the effect is small, age-dependent, and "almost negligible"
in applied/real-world settings per the same authors' retention-focused
meta-analysis, and a separate 2023 meta-analysis calls the sports-practice
version of the effect a "myth" (Ammar et al., 2023; only 21% of outcomes
agreed with the predicted pattern). **This has never been tested for a
decision-under-uncertainty task like threat classification/engagement**,
which is a meaningfully different skill type than motor tasks (aiming,
surgery, music) — all of which is what the existing literature covers.

**Implementation**: `research.sequencing.next_difficulty(arm,
attempt_number)`:
- `"blocked"`: difficulty ramps monotonically, 1,1,2,2,3,3,4,4,5,5,...
- `"randomized"`: difficulty drawn i.i.d. from {1..5} each attempt

Arm assignment uses the same stable-hash mechanism as `aar_ablation`.
`/api/research/assignment/{trainee_id}` returns the trainee's arm and the
recommended difficulty for their *next* attempt — an instructor or the
trainee sets the difficulty dropdown to that value before starting a new
procedural scenario (this is advisory, not yet auto-applied, since forcing
it silently would remove the trainee's own sense of control, a confound
the training literature treats as mattering in its own right).

**Analysis**: the actual outcome of interest is *retention*, not
in-session score — per the contextual-interference literature, blocked
practice often *outperforms* during acquisition but randomized can win on
a later retention/transfer test. So: compare each arm's score on a late
attempt (e.g. attempt 8+) relative to their own early attempts (e.g.
attempt 1-2), not raw scores between arms. Pull via
`/api/research/results/sequencing`.

---

## 4. Swarm-density-aware classification feature (implemented, in `ml_bridge`)

**Gap**: micro-Doppler drone classification is a mature, crowded field for
single targets (Rahman et al. 2020: 99%+ on CNN spectrogram classification;
Kumawat et al. 2021's DIAT-μSAT dataset is a standard benchmark). Only two
papers on swarm/multi-drone-type discrimination were found, both limited
to classifying which single model a radar is looking at (85–95% accuracy,
Bernard-Cooper et al. 2022; Bourn et al. 2024) — **none address how nearby
swarm members' returns interfere with and distort each other's apparent
signature**, which is exactly the regime a real counter-swarm sensor would
operate in.

**Implementation**: `ml_bridge/radar_classifier.py::_swarm_density_at()`
counts same-swarm contacts within 150m of a given member at time `t`, and
widens that member's synthesized `micro_doppler_bandwidth` (+15Hz per
nearby member) and `micro_doppler_jitter` (+0.03 per nearby member) before
it reaches the real trained classifier — modeling the physically expected
effect of overlapping returns, rather than treating each swarm member as if
it were an isolated target (which is what every one of the found papers'
training data implicitly assumes). Verified: classifying the same swarm
member with vs. without sibling context changes its derived
`micro_doppler_bandwidth` by a measurable amount (e.g. 260Hz → 311Hz for a
7-member swarm in one test run), and the real RadarSense CatBoost model
consumes that shifted feature.

**Follow-up this enables**: measure how much the real classifier's accuracy
degrades as swarm density increases — a genuine, quantifiable contribution
to the swarm-classification gap identified above, using an existing
trained model rather than needing new training data.

---

## Bibliography

VR/simulation training transfer:
- Sarmento, K.M. et al. (2026). *Evidence of Skill Transferability From
  Training in Virtual Reality Robotic Surgery Simulators to Nonvirtual
  Environments: A Systematic Review.* Journal of Surgical Education.
- Mao, R.Q. et al. (2021). *Immersive Virtual Reality for Surgical
  Training: A Systematic Review.* Journal of Surgical Research.
- Gallagher, A. et al. (2013). *Prospective, Randomized Assessment of
  Transfer of Training (ToT) and Transfer Effectiveness Ratio (TER) of
  Virtual Reality Simulation Training for Laparoscopic Skill Acquisition.*
  Annals of Surgery.
- Abich, J. et al. (2021). *A review of the evidence for training
  effectiveness with virtual reality technology.* Virtual Reality.
- Xie, B. et al. (2021). *A Review on Virtual Reality Skill Training
  Applications.*

After-action review / debriefing:
- Keiser, N.L. & Arthur, W. (2020). *A meta-analysis of the effectiveness
  of the after-action review (or debrief) and factors that influence its
  effectiveness.* Journal of Applied Psychology.
- Keiser, N.L. et al. (2022). *A Meta-Analysis of Task and Training
  Characteristics that Contribute to or Attenuate the Effectiveness of the
  After-Action Review.* Journal of Business and Psychology.
- Tannenbaum, S. & Cerasoli, C. (2013). *Do Team and Individual Debriefs
  Enhance Performance? A Meta-Analysis.* Human Factors.
- Villado, A.J. & Arthur, W. (2013). *The comparative effect of subjective
  and objective after-action reviews on team performance on a complex
  task.* Journal of Applied Psychology.

Contextual interference / practice sequencing:
- Shea, J. & Morgan, R. (1979). *Contextual interference effects on the
  acquisition, retention, and transfer of a motor skill.* Journal of
  Experimental Psychology: Human Learning & Memory.
- Czyż, S.H. et al. (2024). *High contextual interference improves
  retention in motor learning: systematic review and meta-analysis.*
  Scientific Reports.
- Czyż, S.H. et al. (2024). *The effect of contextual interference on
  transfer in motor learning — a systematic review and meta-analysis.*
  Frontiers in Psychology.
- Ammar, A. et al. (2023). *The myth of contextual interference learning
  benefit in sports practice: A systematic review and meta-analysis.*
  Educational Research Review.

Micro-Doppler radar drone classification:
- Rahman, S. & Robertson, D. (2020). *Classification of drones and birds
  using convolutional neural networks applied to radar micro-Doppler
  spectrogram images.* IET Radar, Sonar & Navigation.
- Kumawat, H.C. et al. (2021). *DIAT-μSAT: Small Aerial Targets' Micro-
  Doppler Signatures and Their Classification Using CNN.* IEEE Geoscience
  and Remote Sensing Letters.
- Narayanan, R. et al. (2023). *Classification and Discrimination of Birds
  and Small Drones Using Radar Micro-Doppler Spectrogram Images.* Signals.
- Bernard-Cooper, J. et al. (2022). *Multiple drone type classification
  using machine learning techniques based on FMCW radar micro-Doppler
  data.*
- Bourn, W. et al. (2024). *Micro-Doppler Signature Classification of
  Unmanned Aerial Vehicles Using FMCW Radar.* 2024 International Radar
  Conference (RADAR).
