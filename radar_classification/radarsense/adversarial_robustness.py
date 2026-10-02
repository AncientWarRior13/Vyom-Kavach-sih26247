"""Black-box adversarial spoofing search against the deployed 11-feature
model -- the two items THREAT_MODEL.md explicitly names as "not attempted
here": (1) an optimized adversarial search rather than the physically-
motivated uniform countermeasures already tested in robustness.py, and
(2) spoofing/decoy evasion (crafting a feature vector that mimics a benign
class) rather than pure signature suppression.

Why black-box, not gradient-based: CatBoost is a tree ensemble, not
differentiable the way a neural net is -- THREAT_MODEL.md names gradient-
free/black-box search as the appropriate method, followed here.

Attack: for each real drone the model already correctly classifies, search
for the SMALLEST perturbation (in per-feature standard-deviation units, so
features on very different natural scales -- m/s vs. a [0,1] jitter ratio
-- are compared fairly) across the SIX features an adversary could
plausibly influence via flight behavior or physical countermeasures:
doppler_velocity, rcs_estimate, micro_doppler_bandwidth,
micro_doppler_jitter, velocity_variance, acceleration. Excludes range_m/
azimuth_deg/altitude_m (changes mission profile, not signature) and snr_db
(determined by the sensor/environment, not attacker-controlled).
doppler_freq_shift is DERIVED from doppler_velocity (recomputed after each
perturbation, not perturbed independently) -- an adversary changing speed
changes the Doppler shift as a physical consequence, not as an independent
free parameter.

Perturbations are clipped to the SAME physical bounds simulator.py itself
enforces (doppler_velocity in [-50,50], rcs_estimate in [-20,20], etc.) --
so this never searches outside physically valid sensor readings. It does
NOT verify that a given perturbation is ACHIEVABLE by a real operator (that
would need a per-feature cost model an adversary would have to pay, e.g.
"how much RCS reduction does N mm of RAM coating buy" -- a harder, separate
question). This is disclosed as the same limitation robustness.py's own
physically-motivated tests already sidestep by testing named, real
countermeasures instead of a free-form search.

Search method: black-box random search / basin-hopping (simple, no
adversarial-robustness-toolbox dependency, matches this project's existing
style of self-contained sweeps like optimize.py's threshold search).
"""

import json

import joblib
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.optimize import threshold_predict

ATTACK_FEATURES = ["doppler_velocity", "rcs_estimate", "micro_doppler_bandwidth",
                   "micro_doppler_jitter", "velocity_variance", "acceleration"]
BOUNDS = {
    "doppler_velocity": (-50, 50), "rcs_estimate": (-20, 20),
    "micro_doppler_bandwidth": (0, 500), "micro_doppler_jitter": (0, 1),
    "velocity_variance": (0, 200), "acceleration": (-10, 10),
}
N_RESTARTS = 8
N_ITERS = 150
N_SAMPLES = 200


def reorder_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, [list(model.classes_).index(c) for c in C.CLASSES]]


VEL_IDX = C.FEATURE_COLUMNS.index("doppler_velocity")
FSHIFT_IDX = C.FEATURE_COLUMNS.index("doppler_freq_shift")
ATTACK_IDX = [C.FEATURE_COLUMNS.index(f) for f in ATTACK_FEATURES]
BOUNDS_ARR = np.array([BOUNDS[f] for f in ATTACK_FEATURES])


def apply_perturbation(x0, delta):
    """x0: (11,) float32 array in C.FEATURE_COLUMNS order. delta: (6,) array
    over ATTACK_FEATURES. Returns a new (11,) array, clipped to physical
    bounds, with doppler_freq_shift re-derived consistently."""
    out = x0.copy()
    for j, idx in enumerate(ATTACK_IDX):
        out[idx] = np.clip(x0[idx] + delta[j], BOUNDS_ARR[j, 0], BOUNDS_ARR[j, 1])
    out[FSHIFT_IDX] = 2 * out[VEL_IDX] * C.CARRIER_FREQ_HZ / C.C_LIGHT
    return out


def attack_one(model, feat_std_arr, x0, threshold, drone_idx, rng):
    """Black-box random search for the minimal-norm perturbation that drops
    p(drone) below `threshold`. Returns (success, min_norm, best_delta)."""
    best_norm = None
    best_delta = None
    for restart in range(N_RESTARTS):
        delta = np.zeros(len(ATTACK_FEATURES))
        step = feat_std_arr * 0.5
        x = apply_perturbation(x0, delta)
        best_p = reorder_proba(model, x[None, :].astype(np.float32))[0, drone_idx]
        for it in range(N_ITERS):
            j = rng.integers(len(ATTACK_FEATURES))
            trial = delta.copy()
            trial[j] = delta[j] + rng.normal(0, step[j])
            x_trial = apply_perturbation(x0, trial)
            p_drone = reorder_proba(model, x_trial[None, :].astype(np.float32))[0, drone_idx]
            norm = float(np.sqrt(np.sum((trial / np.maximum(feat_std_arr, 1e-9)) ** 2)))
            if p_drone < best_p or (p_drone < threshold and
                                     (best_norm is None or norm < best_norm)):
                delta = trial
                best_p = p_drone
                if p_drone < threshold and (best_norm is None or norm < best_norm):
                    best_norm = norm
                    best_delta = trial.copy()
            step = step * 0.995
    return (best_norm is not None), best_norm, best_delta


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    feat_std = {f: float(train[f].std()) for f in ATTACK_FEATURES}
    feat_std_arr = np.array([feat_std[f] for f in ATTACK_FEATURES])
    print("per-feature std (attack step-size basis):", {k: round(v, 3) for k, v in feat_std.items()})

    drones = test[test[C.LABEL_COLUMN] == "drone"].copy()
    proba = reorder_proba(model, drones[C.FEATURE_COLUMNS])
    pred = threshold_predict(proba, threshold)
    correctly_classified = drones[pred == "drone"].reset_index(drop=True)
    print(f"real drones correctly classified in test set: {len(correctly_classified)} "
          f"(attacking {min(N_SAMPLES, len(correctly_classified))} of them)")

    sample = correctly_classified.sample(min(N_SAMPLES, len(correctly_classified)), random_state=C.SEED)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)

    rng = np.random.default_rng(C.SEED)
    results = []
    for i in range(len(sample)):
        success, norm, delta = attack_one(model, feat_std_arr, X0[i], threshold, drone_idx, rng)
        results.append({
            "success": success,
            "min_perturbation_norm": norm,
            "delta": dict(zip(ATTACK_FEATURES, delta.tolist())) if success else None,
        })
        if (i + 1) % 50 == 0:
            print(f"  attacked {i+1}/{len(sample)}...")

    n_success = sum(r["success"] for r in results)
    success_rate = n_success / len(results)
    print(f"\nattack success rate: {n_success}/{len(results)} ({success_rate:.1%})")

    successful = [r for r in results if r["success"]]
    if successful:
        norms = np.array([r["min_perturbation_norm"] for r in successful])
        print(f"minimal perturbation norm (std-units): median={np.median(norms):.3f} "
              f"p25={np.percentile(norms,25):.3f} p75={np.percentile(norms,75):.3f}")

        mean_abs_delta = {}
        for f in ATTACK_FEATURES:
            vals = [abs(r["delta"][f]) / feat_std[f] for r in successful]
            mean_abs_delta[f] = float(np.mean(vals))
        ranked = sorted(mean_abs_delta.items(), key=lambda kv: -kv[1])
        print("features most exploited by successful attacks (mean |delta|/std):")
        for f, v in ranked:
            print(f"  {f:28s} {v:.3f}")
    else:
        norms = np.array([])
        ranked = []

    out = {
        "n_attacked": len(results),
        "n_correctly_classified_drones_in_test": len(correctly_classified),
        "attack_features": ATTACK_FEATURES,
        "excluded_features": [f for f in C.FEATURE_COLUMNS if f not in ATTACK_FEATURES + ["doppler_freq_shift"]],
        "physical_bounds": BOUNDS,
        "search_method": "black-box random search / basin-hopping, 8 restarts x 150 iters/restart",
        "deployed_threshold": threshold,
        "success_rate": success_rate,
        "median_min_perturbation_norm_std_units": float(np.median(norms)) if len(norms) else None,
        "features_most_exploited_mean_abs_delta_over_std": dict(ranked),
        "per_sample_results": results,
        "caveat": (
            "Perturbations are clipped to physically valid SENSOR READING bounds, but this "
            "does NOT verify a real operator could ACHIEVE a given perturbation (e.g. how much "
            "RCS reduction a given amount of RAM coating buys is a separate, harder question, "
            "not modeled here) -- same limitation the physically-motivated tests in "
            "robustness.py sidestep by testing named real countermeasures at fixed severities "
            "instead of a free-form search."),
    }
    (C.REPORTS_DIR / "adversarial_robustness_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'adversarial_robustness_results.json'}")


if __name__ == "__main__":
    main()
