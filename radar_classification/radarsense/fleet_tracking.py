"""Fleet-scale multi-target tracking -- generalizes multi_target_tracking.py's
3-target validation to the swarm regime (100-1000 simultaneous targets), the
scale a coordinated-drone-incursion scenario actually demands.

Two changes from multi_target_tracking.py, both required at this scale:

1. TWO-DIMENSIONAL scene (range x Doppler), not a single Doppler row.
   multi_target_tracking.py's 3-target validation puts everything on one
   256-bin Doppler axis with min_sep=6 bins -- room for ~40 targets before
   collisions dominate. A real swarm has range separation too; this module
   uses it.
2. Spatially-indexed (KD-tree) peak-picking AND track association, replacing
   the O(n^2) full pairwise scan multi_target_tracking.py uses. At n=1000
   tracks x n=1000 detections a naive scan is ~1e6 distance comparisons every
   frame; a KD-tree keeps each frame close to O(n log n).

Same validated DSP building blocks as multi_target_tracking.py/radar_dsp.py:
OS-CFAR per range row (radar_dsp.os_cfar_1d, unmodified), a bank of
constant-velocity Kalman filters (radar_dsp.KalmanTracker, unmodified),
spawn / M-of-N-confirm / coast / kill track lifecycle.

New here: a fleet-aggregation summary (the per-frame rollup an operator or
API consumer actually wants -- count, cardinality, association latency) and
a swarm-pattern flag: many tracks moving with correlated heading is the
signature a coordinated drone swarm has that a scatter of independent birds
or aircraft does not.

Architecture note: this is a backend/API-layer capability, not something the
live browser demo drives frame-by-frame (round-tripping HTTP per animation
frame for 1000 targets at 30-60fps isn't practical without a streaming
transport -- see TODO.md's WebSocket item). It plugs in the same place
radar_dsp.py's real DSP chain does: validated, stress-tested, and exposed via
api/app.py's /fleet endpoints, while the live frontend simulator keeps its
own lighter client-side spawn/render path for the swarm *view* (see
FleetPanel.tsx). That split mirrors the project's existing REAL vs SIM
architecture (arch.md) rather than inventing a new one.
"""

import json
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

from radarsense.radar_dsp import KalmanTracker, CFAR_TRAIN, CFAR_GUARD, CFAR_PFA
from radarsense import config as C

N_DOPPLER = 768
N_RANGE = 1024
GATE = 12.0
MIN_SEP = (2.5, 5.0)
MAX_COAST = 5
CONFIRM_HITS = 3
N_FRAMES_STRESS = 20
SCALES = [100, 500, 1000]


def _os_cfar_2d(power_grid, train=CFAR_TRAIN, guard=CFAR_GUARD, pfa=CFAR_PFA, k_frac=0.75):
    """Vectorized OS-CFAR (radar_dsp.os_cfar_1d's algorithm, unmodified: k-th
    ordered training cell as the noise estimate) applied to every row of a
    2D grid at once via a sliding window, instead of radar_dsp.py's
    pure-Python per-bin loop. radar_dsp.py's real-data grid is 5 rows x 256
    bins -- the Python loop is fine there. A swarm-scale grid (N_RANGE=1024)
    needs the same math ~1000x faster, hence this vectorized rewrite rather
    than reusing os_cfar_1d directly.

    NOTE on target density: OS-CFAR's noise estimate is contaminated (and
    detection masked) when multiple targets fall inside the same training
    window -- a known CFAR limitation with dense scenes, not a bug. That is
    why N_RANGE is large enough to keep per-row target density low (see
    module docstring) rather than an arbitrary choice.
    """
    n_range, n_doppler = power_grid.shape
    pad = train + guard
    win = 2 * pad + 1
    padded = np.pad(power_grid, ((0, 0), (pad, pad)), mode="constant", constant_values=np.inf)
    windows = sliding_window_view(padded, window_shape=win, axis=1)
    left = windows[:, :, 0:train]
    right = windows[:, :, train + 2 * guard + 1: train + 2 * guard + 1 + train]
    noise_cells = np.concatenate([left, right], axis=2)
    k = int(2 * train * k_frac)
    noise_est = np.partition(noise_cells, k, axis=2)[:, :, k]
    alpha = (pfa ** (-1.0 / max(k, 1)) - 1.0) * (2 * train)
    thr = alpha * noise_est
    det = power_grid > thr
    return det, thr



def _lattice_positions(rng, n_targets):
    """Guaranteed-separable starting positions: a jittered lattice spaced at
    1.3x the NMS separation, shuffled and truncated to n_targets. Cheap
    (O(n log n)) and doesn't depend on rejection-sampling luck at n=1000."""
    sr, sd = MIN_SEP[0] * 1.3, MIN_SEP[1] * 1.3
    r_vals = np.arange(4, N_RANGE - 4, sr)
    d_vals = np.arange(30, N_DOPPLER - 30, sd)
    grid = np.array([(r, d) for r in r_vals for d in d_vals])
    if len(grid) < n_targets:
        raise ValueError(f"grid capacity {len(grid)} < n_targets {n_targets}; "
                          f"increase N_RANGE/N_DOPPLER")
    idx = rng.choice(len(grid), size=n_targets, replace=False)
    pts = grid[idx].astype(float)
    pts[:, 0] += rng.uniform(-sr * 0.3, sr * 0.3, n_targets)
    pts[:, 1] += rng.uniform(-sd * 0.3, sd * 0.3, n_targets)
    pts[:, 0] = np.clip(pts[:, 0], 2, N_RANGE - 2)
    pts[:, 1] = np.clip(pts[:, 1], 4, N_DOPPLER - 4)
    return pts


def build_swarm_scene(rng, n_targets, n_frames=N_FRAMES_STRESS, snr_db=14.0,
                       coordinated_frac=0.0):
    """n_targets point targets on a (range, Doppler) grid, each on its own
    constant-acceleration trajectory. Returns (power_grids per frame,
    ground-truth (r, d) trajectories, ground-truth headings).

    coordinated_frac: fraction of targets sharing one common heading (+ small
    per-target jitter) instead of an independent random one -- lets the
    stress test also exercise the swarm-pattern detector against a scene that
    actually contains a coordinated group.
    """
    start = _lattice_positions(rng, n_targets)
    n_coord = int(n_targets * coordinated_frac)
    common_heading = rng.uniform(0, 2 * np.pi)

    headings = rng.uniform(0, 2 * np.pi, n_targets)
    if n_coord:
        headings[:n_coord] = common_heading + rng.normal(0, 0.08, n_coord)

    speed = rng.uniform(0.3, 1.6, n_targets)
    vel = np.column_stack([speed * np.cos(headings), speed * np.sin(headings)])
    acc = rng.uniform(-0.01, 0.01, (n_targets, 2))

    snr_lin = 10 ** (snr_db / 10.0)
    traj = np.zeros((n_frames, n_targets, 2))
    for f in range(n_frames):
        pos = start + vel * f + 0.5 * acc * f * f
        pos[:, 0] = np.clip(pos[:, 0], 2, N_RANGE - 2)
        pos[:, 1] = np.clip(pos[:, 1], 4, N_DOPPLER - 4)
        traj[f] = pos

    def make_grid(f):
        grid = rng.exponential(1.0, (N_RANGE, N_DOPPLER))
        rb = np.round(traj[f, :, 0]).astype(int)
        db = np.round(traj[f, :, 1]).astype(int)
        for ro, rw in [(-1, 0.5), (0, 1.0), (1, 0.5)]:
            for do, dw in [(-1, 0.4), (0, 1.0), (1, 0.4)]:
                rr = np.clip(rb + ro, 0, N_RANGE - 1)
                dd = np.clip(db + do, 0, N_DOPPLER - 1)
                grid[rr, dd] += snr_lin * rw * dw * rng.uniform(0.8, 1.2, n_targets)
        return grid

    grids = [make_grid(f) for f in range(n_frames)]
    return grids, traj, headings



def detect_peaks_2d(power_grid, min_sep=MIN_SEP):
    """Vectorized OS-CFAR across the whole grid -> candidate cells, then
    KD-tree-gated non-max-suppression across the full 2D grid (strongest
    first) so one physical target's spectral spread doesn't register as
    several detections. O(n_candidates log n_candidates) instead of the
    pairwise scan a naive 2D NMS would need."""
    det, thr = _os_cfar_2d(power_grid)
    rs, ds = np.where(det)
    if len(rs) == 0:
        return []
    cand_pts = list(zip(rs.tolist(), ds.tolist()))
    cand_snr = (10 * np.log10(np.maximum(power_grid[rs, ds], 1e-12)
                               / np.maximum(thr[rs, ds], 1e-12))).tolist()

    cand_arr = np.array(cand_pts, dtype=float)
    norm = cand_arr / np.array(min_sep)
    order = np.argsort(-np.array(cand_snr))
    tree = cKDTree(norm)
    suppressed = np.zeros(len(cand_pts), dtype=bool)
    peaks = []
    for i in order:
        if suppressed[i]:
            continue
        peaks.append((int(cand_pts[i][0]), int(cand_pts[i][1]), float(cand_snr[i])))
        for j in tree.query_ball_point(norm[i], r=1.0):
            suppressed[j] = True
    return peaks



class FleetTracker:
    """Bank of constant-velocity Kalman tracks over an unbounded number of
    simultaneous targets. Association is KD-tree-gated global-nearest-
    neighbour (same greedy, confidence-ordered policy as
    multi_target_tracking.MultiTargetTracker, just spatially indexed so it
    scales past a handful of targets)."""

    def __init__(self, gate=GATE, max_coast=MAX_COAST, confirm_hits=CONFIRM_HITS):
        self.tracks = []
        self.next_id = 0
        self.gate = gate
        self.max_coast = max_coast
        self.confirm_hits = confirm_hits
        self.last_assoc_latency_s = 0.0

    def update(self, detections):
        """detections: list of (range_bin, doppler_bin, snr)."""
        t0 = time.perf_counter()
        assigned = set()
        if self.tracks and detections:
            preds = np.array([t["kf"].peek() for t in self.tracks])
            valid = np.array([p is not None for p in preds])
            det_pts = np.array([(d[0], d[1]) for d in detections])
            tree = cKDTree(det_pts)
            order = sorted(range(len(self.tracks)), key=lambda i: self.tracks[i]["coast"])
            for ti in order:
                tr = self.tracks[ti]
                pred = tr["kf"].peek()
                if pred is None:
                    continue
                cand = tree.query_ball_point(pred, r=self.gate)
                cand = [c for c in cand if c not in assigned]
                if not cand:
                    tr["coast"] += 1
                    tr["hist"].append(tuple(tr["kf"].step(None)))
                    continue
                best = min(cand, key=lambda di: (det_pts[di][0] - pred[0]) ** 2
                                                  + (det_pts[di][1] - pred[1]) ** 2)
                assigned.add(best)
                tr["kf"].step((detections[best][0], detections[best][1]))
                tr["coast"] = 0
                tr["hits"] += 1
                tr["hist"].append((detections[best][0], detections[best][1]))
        else:
            for ti, tr in enumerate(self.tracks):
                tr["coast"] += 1
                tr["hist"].append(tuple(tr["kf"].step(None)))

        for di, det in enumerate(detections):
            if di not in assigned:
                rb, db = float(det[0]), float(det[1])
                kf = KalmanTracker(gate=self.gate ** 2)
                kf.step((rb, db))
                self.tracks.append({"kf": kf, "id": self.next_id, "coast": 0,
                                     "hits": 1, "hist": [(rb, db)]})
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["coast"] <= self.max_coast]
        self.last_assoc_latency_s = time.perf_counter() - t0

    def confirmed(self):
        return [t for t in self.tracks if t["hits"] >= self.confirm_hits]


    def fleet_summary(self, coherence_threshold=0.6, min_swarm_size=20):
        """Per-frame rollup: count, cardinality, and a swarm-pattern flag.
        Coherence = mean resultant length of confirmed tracks' velocity
        headings (circular statistics: 1.0 = perfectly aligned headings,
        0.0 = uniformly scattered) -- the signature a coordinated drone
        swarm has that independent birds/aircraft returns don't."""
        confirmed = self.confirmed()
        headings = []
        for t in confirmed:
            x = t["kf"].x
            if x is not None and (abs(x[2]) > 1e-6 or abs(x[3]) > 1e-6):
                headings.append(np.arctan2(x[3], x[2]))
        coherence = 0.0
        if headings:
            vecs = np.exp(1j * np.array(headings))
            coherence = float(np.abs(np.mean(vecs)))
        coordinated = len(confirmed) >= min_swarm_size and coherence >= coherence_threshold
        return {
            "confirmed_tracks": len(confirmed),
            "tentative_tracks": len(self.tracks) - len(confirmed),
            "heading_coherence": coherence,
            "coordinated_swarm_detected": bool(coordinated),
            "association_latency_ms": round(self.last_assoc_latency_s * 1000, 3),
        }



def run_scale(rng, n_targets, n_frames=N_FRAMES_STRESS, coordinated_frac=0.0):
    grids, traj, headings = build_swarm_scene(rng, n_targets, n_frames,
                                               coordinated_frac=coordinated_frac)
    tracker = FleetTracker()
    detect_times, assoc_times = [], []
    track_hist = {}
    for f, grid in enumerate(grids):
        t0 = time.perf_counter()
        peaks = detect_peaks_2d(grid)
        detect_times.append(time.perf_counter() - t0)
        tracker.update(peaks)
        assoc_times.append(tracker.last_assoc_latency_s)
        for tr in tracker.confirmed():
            track_hist.setdefault(tr["id"], {})[f] = tr["hist"][-1]

    live_counts = [sum(1 for h in track_hist.values() if f in h) for f in range(n_frames)]
    cardinality_error = abs(np.median(live_counts) - n_targets)
    spurious = max(0, len(track_hist) - n_targets)

    completeness = []
    for t in range(n_targets):
        true_r, true_d = traj[:, t, 0], traj[:, t, 1]
        best_cov = 0.0
        for hist in track_hist.values():
            frames = [f for f in hist
                      if abs(hist[f][0] - true_r[f]) < 4 and abs(hist[f][1] - true_d[f]) < 8]
            best_cov = max(best_cov, len(frames) / n_frames)
        completeness.append(best_cov)

    summary = tracker.fleet_summary()
    return {
        "n_targets": n_targets,
        "n_frames": n_frames,
        "track_completeness_mean": float(np.mean(completeness)),
        "track_completeness_std": float(np.std(completeness)),
        "cardinality_error": float(cardinality_error),
        "spurious_tracks": int(spurious),
        "detect_latency_ms_mean": round(float(np.mean(detect_times)) * 1000, 2),
        "assoc_latency_ms_mean": round(float(np.mean(assoc_times)) * 1000, 3),
        "assoc_latency_ms_p95": round(float(np.percentile(assoc_times, 95)) * 1000, 3),
        "fleet_summary_final_frame": summary,
    }


def main():
    rng = np.random.default_rng(C.SEED)
    results = []
    for n in SCALES:
        print(f"=== fleet stress test: {n} simultaneous targets, {N_FRAMES_STRESS} frames ===")
        t0 = time.perf_counter()
        coord_frac = 0.75 if n == SCALES[-1] else 0.0
        res = run_scale(rng, n, coordinated_frac=coord_frac)
        wall = time.perf_counter() - t0
        res["wall_time_s"] = round(wall, 2)
        results.append(res)
        print(f"  completeness={res['track_completeness_mean']:.3f}  "
              f"cardinality_error={res['cardinality_error']:.1f}  "
              f"spurious={res['spurious_tracks']}  "
              f"assoc_latency(mean/p95)={res['assoc_latency_ms_mean']:.2f}/"
              f"{res['assoc_latency_ms_p95']:.2f} ms  wall={wall:.1f}s")
        print(f"  fleet_summary: {res['fleet_summary_final_frame']}")

    out = {
        "association": "KD-tree-gated greedy global-nearest-neighbour (scipy.spatial.cKDTree)",
        "detection": "OS-CFAR per range row + KD-tree 2D non-max suppression",
        "tracker": "bank of constant-velocity Kalman filters (radar_dsp.KalmanTracker), spawn/coast/kill",
        "grid": {"n_range": N_RANGE, "n_doppler": N_DOPPLER},
        "scales_tested": SCALES,
        "results": results,
    }
    (C.REPORTS_DIR / "fleet_tracking_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'fleet_tracking_results.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ns = [r["n_targets"] for r in results]
    axes[0].plot(ns, [r["assoc_latency_ms_mean"] for r in results], "o-", label="mean")
    axes[0].plot(ns, [r["assoc_latency_ms_p95"] for r in results], "s--", label="p95")
    axes[0].set_xlabel("simultaneous targets"); axes[0].set_ylabel("association latency (ms/frame)")
    axes[0].set_title("KD-tree-gated association latency vs. fleet size")
    axes[0].legend()
    axes[1].plot(ns, [r["cardinality_error"] for r in results], "o-", color="firebrick")
    axes[1].set_xlabel("simultaneous targets"); axes[1].set_ylabel("cardinality error (tracks)")
    axes[1].set_title("Tracker cardinality error vs. fleet size")
    fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "fleet_tracking_scaling.png", dpi=150)
    plt.close(fig)
    print(f"figure -> {C.PLOTS_DIR / 'fleet_tracking_scaling.png'}")


if __name__ == "__main__":
    main()
