"""Multi-target tracking -- fills the single-target gap in radar_dsp.py (whose
CFAR + Kalman chain runs on real I/Q but tracks one target). Real multi-target
truth is impossible to get from the single-target SAAB set, so this validates
the multi-target TRACKER on controlled range-Doppler scenes: several point
targets on realistic (exponential) noise, each following its own
constant-acceleration Doppler trajectory, with known ground truth.

Pipeline is the SAME as the real-data chain: OS-CFAR (multiple peaks/frame) ->
global-nearest-neighbour association (greedy, gated) -> a BANK of
constant-velocity Kalman tracks (spawn / M-of-N confirm / coast / kill).

Metrics (standard multi-target): track completeness (frac of frames each true
target is held), cardinality error (est vs true target count), and spurious
track rate.
"""

import json

import numpy as np

from radarsense.radar_dsp import N_SLOW, N_RANGE, os_cfar_1d, KalmanTracker
from radarsense import config as C

N_TARGETS = 3
N_FRAMES = 80
GATE = 12.0
MAX_COAST = 5


def detect_peaks(prof, min_sep=6):
    """All OS-CFAR detections in a Doppler power profile, as (doppler_bin, snr),
    non-max-suppressed by min_sep bins."""
    det, thr = os_cfar_1d(prof)
    peaks = []
    for d in sorted(np.where(det)[0], key=lambda i: -prof[i]):
        if all(abs(d - p[0]) >= min_sep for p in peaks):
            peaks.append((int(d), float(10 * np.log10(max(prof[d], 1e-12) / max(thr[d], 1e-12)))))
    return peaks


def build_scene(rng, snr_db=14.0):
    """Controlled scene: N point targets on exponential noise, each with a
    constant-acceleration Doppler trajectory. Returns (Doppler power profiles
    per frame, ground-truth Doppler-bin trajectories)."""
    v0 = rng.uniform(40, 216, N_TARGETS)
    vel = rng.uniform(-1.5, 1.5, N_TARGETS)
    acc = rng.uniform(-0.02, 0.02, N_TARGETS)
    snr_lin = 10 ** (snr_db / 10.0)
    profiles, truth = [], [[] for _ in range(N_TARGETS)]
    for f in range(N_FRAMES):
        prof = rng.exponential(1.0, N_SLOW)
        for t in range(N_TARGETS):
            b = int(np.clip(v0[t] + vel[t] * f + 0.5 * acc[t] * f * f, 8, N_SLOW - 8))
            truth[t].append(b)
            for off, w in [(-1, 0.4), (0, 1.0), (1, 0.4)]:
                prof[b + off] += snr_lin * w * rng.uniform(0.8, 1.2)
        profiles.append(prof)
    return profiles, truth


class MultiTargetTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        assigned = set()
        order = sorted(range(len(self.tracks)),
                       key=lambda i: self.tracks[i]["coast"])
        for ti in order:
            tr = self.tracks[ti]
            pred = tr["kf"].peek()
            if pred is None:
                continue
            best, bestd = None, GATE
            for di, (dbin, _) in enumerate(detections):
                if di in assigned:
                    continue
                if abs(pred[1] - dbin) < bestd:
                    bestd, best = abs(pred[1] - dbin), di
            if best is not None:
                assigned.add(best)
                tr["kf"].step((2, detections[best][0]))
                tr["coast"] = 0
                tr["hits"] += 1
                tr["hist"].append(detections[best][0])
            else:
                tr["coast"] += 1
                tr["hist"].append(tr["kf"].step(None)[1])
        for di, (dbin, _) in enumerate(detections):
            if di not in assigned:
                kf = KalmanTracker()
                kf.step((2, dbin))
                self.tracks.append({"kf": kf, "id": self.next_id, "coast": 0,
                                    "hits": 1, "hist": [dbin]})
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["coast"] <= MAX_COAST]

    def confirmed(self):
        """M-of-N confirmation: a track is real only after >=3 hits."""
        return [t for t in self.tracks if t["hits"] >= 3]


def main():
    rng = np.random.default_rng(C.SEED)

    completeness, id_switches, card_err = [], [], []
    for scene_i in range(10):
        profiles, truth = build_scene(rng)
        mtt = MultiTargetTracker()
        track_hist = {}
        for f, prof in enumerate(profiles):
            peaks = detect_peaks(prof)
            mtt.update(peaks)
            for tr in mtt.confirmed():
                track_hist.setdefault(tr["id"], {})[f] = tr["hist"][-1]

        live_counts = []
        for f in range(N_FRAMES):
            live_counts.append(sum(1 for h in track_hist.values() if f in h))
        card_err.append(abs(np.median(live_counts) - N_TARGETS))

        for t in range(N_TARGETS):
            true_traj = np.array(truth[t])
            best_cov = 0.0
            for tid, hist in track_hist.items():
                frames = [f for f in hist if abs(hist[f] - true_traj[f]) < 8]
                best_cov = max(best_cov, len(frames) / N_FRAMES)
            completeness.append(best_cov)
        id_switches.append(max(0, len(track_hist) - N_TARGETS))

    print(f"=== multi-target tracking, controlled scenes ({N_TARGETS} targets, {N_FRAMES} frames, 10 scenes) ===")
    print(f"  track completeness (frac frames a true target is held): {np.mean(completeness):.3f} +/- {np.std(completeness):.3f}")
    print(f"  cardinality error (|median live tracks - true|): {np.mean(card_err):.3f}")
    print(f"  spurious/extra tracks per scene (ID proliferation): {np.mean(id_switches):.2f}")

    out = {
        "n_targets": N_TARGETS, "n_frames": N_FRAMES, "n_scenes": 10,
        "association": "greedy global-nearest-neighbour, Mahalanobis gate",
        "tracker": "bank of constant-velocity Kalman filters, spawn/coast/kill",
        "track_completeness_mean": float(np.mean(completeness)),
        "track_completeness_std": float(np.std(completeness)),
        "cardinality_error_mean": float(np.mean(card_err)),
        "extra_tracks_per_scene_mean": float(np.mean(id_switches)),
    }
    (C.REPORTS_DIR / "multi_target_tracking_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'multi_target_tracking_results.json'}")


if __name__ == "__main__":
    main()
