"""WS2 -- the COMPLETE adversarial-evasion treatment.

ADVERSARIAL_TRAINING.md ran ONE round of adversarial training and cut the
evasion rate 100% -> 80.1%. Partial, by construction: one round only teaches
the model the perturbations it already saw, and the attacker just re-runs the
search. This closes it three ways:

  1. ITERATIVE adversarial training to PLATEAU. Retrain -> re-attack the new
     model -> harvest the fresh evasions it's still vulnerable to -> retrain
     again, round after round, until the evasion rate stops falling. The
     plateau value (not a single round's number) is the honest measure of
     how far empirical adversarial training gets.

  2. CERTIFIED robustness via randomized smoothing (Cohen et al. 2019). The
     empirical evasion rate is an upper bound an attacker might beat with a
     better search; a certificate is a PROVABLE lower bound no attacker can.
     The smoothed classifier g(x) = "what the base model predicts most often
     under Gaussian feature noise" comes with a radius R such that NO
     perturbation of L2 norm < R (in the same per-feature-std units the
     attack itself uses) can change g's prediction. We report the certified
     radius distribution for the original vs the hardened model, and what
     fraction of drones are certified beyond the attack's own median
     perturbation norm (1.04 std-units) -- i.e. provably un-evadable by that
     attack.

  3. A PHYSICS PLAUSIBILITY gate. The attack is bounded to valid per-feature
     ranges but can still push the JOINT feature vector into a combination no
     real drone produces. A novelty detector fit on real drone feature
     vectors flags those, catching evasions that escaped the classifier.

Reuses adversarial_robustness.py's exact attack (attack_one, apply_
perturbation, ATTACK_FEATURES) and adversarial_training.py's harvest logic,
imported unmodified, so every number is comparable to what's already
reported.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import binomtest, norm
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.adversarial_robustness import (ATTACK_FEATURES, ATTACK_IDX, apply_perturbation,
                                               attack_one, reorder_proba)
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

N_ROUNDS = 4
N_HARVEST = 250
N_AUGMENT_PER_DELTA = 3
N_REATTACK = 150
SMOOTH_SIGMA = 1.0
SMOOTH_N0 = 100
SMOOTH_N = 500
SMOOTH_ALPHA = 0.05
N_CERTIFY = 120


def harvest_evasions(model, threshold, train_drones, feat_std_arr, drone_idx, n_attack, seed):
    proba = reorder_proba(model, train_drones[C.FEATURE_COLUMNS])
    correct = train_drones[threshold_predict(proba, threshold) == "drone"].reset_index(drop=True)
    sample = correct.sample(min(n_attack, len(correct)), random_state=seed)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)
    rng = np.random.default_rng(seed)
    deltas = []
    for i in range(len(sample)):
        ok, _, delta = attack_one(model, feat_std_arr, X0[i], threshold, drone_idx, rng)
        if ok:
            deltas.append(delta)
    return deltas


def augment(train, deltas, rng):
    all_drones = train[train[C.LABEL_COLUMN] == "drone"]
    rows = []
    for delta in deltas:
        for _, row in all_drones.sample(N_AUGMENT_PER_DELTA, random_state=int(rng.integers(1e6))).iterrows():
            x_adv = apply_perturbation(row[C.FEATURE_COLUMNS].values.astype(np.float64), delta)
            r = dict(zip(C.FEATURE_COLUMNS, x_adv)); r[C.LABEL_COLUMN] = "drone"
            rows.append(r)
    return pd.DataFrame(rows)


def retrain(train_aug):
    sw = compute_sample_weight("balanced", train_aug[C.LABEL_COLUMN])
    clf = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train_aug[C.FEATURE_COLUMNS], train_aug[C.LABEL_COLUMN], sample_weight=sw)
    return clf


def evasion_rate(model, threshold, test_drones, feat_std_arr, drone_idx, seed):
    proba = reorder_proba(model, test_drones[C.FEATURE_COLUMNS])
    correct = test_drones[threshold_predict(proba, threshold) == "drone"].reset_index(drop=True)
    sample = correct.sample(min(N_REATTACK, len(correct)), random_state=seed)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)
    rng = np.random.default_rng(seed)
    succ, norms = 0, []
    for i in range(len(sample)):
        ok, nrm, _ = attack_one(model, feat_std_arr, X0[i], threshold, drone_idx, rng)
        succ += int(ok)
        if ok:
            norms.append(nrm)
    rate = succ / len(sample) if len(sample) else 0.0
    return rate, (float(np.median(norms)) if norms else None), len(sample)


def certified_radius(model, threshold, X, feat_std_arr, drone_idx, sigma, rng):
    """Randomized-smoothing certificate for ONE sample. Returns (predicted_drone,
    certified_radius_in_std_units). Radius 0 => abstain/not-certified-drone."""
    def smooth_pred_counts(n):
        noise = rng.normal(0, sigma, size=(n, len(ATTACK_FEATURES))) * feat_std_arr
        batch = np.repeat(X[None, :], n, axis=0)
        for j, idx in enumerate(ATTACK_IDX):
            batch[:, idx] = batch[:, idx] + noise[:, j]
        vel_i = C.FEATURE_COLUMNS.index("doppler_velocity")
        fs_i = C.FEATURE_COLUMNS.index("doppler_freq_shift")
        batch[:, fs_i] = 2 * batch[:, vel_i] * C.CARRIER_FREQ_HZ / C.C_LIGHT
        proba = reorder_proba(model, batch.astype(np.float32))
        is_drone = (proba[:, drone_idx] >= threshold)
        return int(is_drone.sum())
    c0 = smooth_pred_counts(SMOOTH_N0)
    if c0 < SMOOTH_N0 / 2:
        return False, 0.0
    k = smooth_pred_counts(SMOOTH_N)
    p_lower = binomtest(k, SMOOTH_N, 0.5).proportion_ci(confidence_level=1 - 2 * SMOOTH_ALPHA)[0]
    if p_lower <= 0.5:
        return True, 0.0
    return True, float(sigma * norm.ppf(p_lower))


def certify_set(model, threshold, drones, feat_std_arr, drone_idx, sigma):
    rng = np.random.default_rng(C.SEED)
    sample = drones.sample(min(N_CERTIFY, len(drones)), random_state=C.SEED)
    radii, drone_preds = [], 0
    for _, row in sample.iterrows():
        x = row[C.FEATURE_COLUMNS].values.astype(np.float64)
        is_drone, R = certified_radius(model, threshold, x, feat_std_arr, drone_idx, sigma, rng)
        drone_preds += int(is_drone)
        radii.append(R if is_drone else 0.0)
    radii = np.array(radii)
    return {
        "n_certified": len(radii),
        "smoothed_drone_recall": drone_preds / len(radii),
        "mean_certified_radius": float(radii.mean()),
        "median_certified_radius": float(np.median(radii)),
        "frac_certified_beyond_attack_median_1.04": float((radii >= 1.04).mean()),
        "frac_certified_beyond_0.5": float((radii >= 0.5).mean()),
    }


def main():
    original = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    t0 = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    feat_std_arr = np.array([float(train[f].std()) for f in ATTACK_FEATURES])
    train_drones = train[train[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    test_drones = test[test[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)

    print("=== 1: iterative adversarial training ===")
    rate0, norm0, n0 = evasion_rate(original, t0, test_drones, feat_std_arr, drone_idx, C.SEED)
    m0 = full_metrics(test[C.LABEL_COLUMN], threshold_predict(reorder_proba(original, test[C.FEATURE_COLUMNS]), t0),
                      reorder_proba(original, test[C.FEATURE_COLUMNS]))
    rounds = [{"round": 0, "evasion_rate": rate0, "median_norm": norm0, "n_attacked": n0,
               "clean_acc": m0["accuracy"], "clean_drone_recall": m0["recall_drone"]}]
    print(f"  round 0 (original): evasion={rate0:.1%} median_norm={norm0} "
          f"clean_acc={m0['accuracy']:.4f} recall={m0['recall_drone']:.4f}")

    cur_model, cur_t, cur_train = original, t0, train.copy()
    rng = np.random.default_rng(C.SEED)
    for r in range(1, N_ROUNDS + 1):
        deltas = harvest_evasions(cur_model, cur_t, train_drones, feat_std_arr, drone_idx,
                                  N_HARVEST, seed=C.SEED + r)
        aug = augment(cur_train, deltas, rng)
        cur_train = pd.concat([cur_train, aug], ignore_index=True)
        cur_model = retrain(cur_train)
        cur_t, _, _ = tune_threshold(reorder_proba(cur_model, val[C.FEATURE_COLUMNS]), val[C.LABEL_COLUMN])
        rate, med, n = evasion_rate(cur_model, cur_t, test_drones, feat_std_arr, drone_idx, C.SEED)
        proba_te = reorder_proba(cur_model, test[C.FEATURE_COLUMNS])
        m = full_metrics(test[C.LABEL_COLUMN], threshold_predict(proba_te, cur_t), proba_te)
        rounds.append({"round": r, "n_evasions_harvested": len(deltas), "n_augmented": len(aug),
                       "threshold": cur_t, "evasion_rate": rate, "median_norm": med, "n_attacked": n,
                       "clean_acc": m["accuracy"], "clean_drone_recall": m["recall_drone"]})
        print(f"  round {r}: harvested={len(deltas)} evasion={rate:.1%} median_norm={med} "
              f"clean_acc={m['accuracy']:.4f} recall={m['recall_drone']:.4f}")

    evasion_curve = [rd["evasion_rate"] for rd in rounds]
    plateaued = len(evasion_curve) >= 3 and abs(evasion_curve[-1] - evasion_curve[-2]) < 0.05
    hardened_model, hardened_t = cur_model, cur_t
    joblib.dump(hardened_model, C.MODELS_DIR / "final_model_iteratively_hardened.joblib")

    print("\n=== 2: certified robustness via randomized smoothing ===")
    cert_original = certify_set(original, t0, test_drones, feat_std_arr, drone_idx, SMOOTH_SIGMA)
    cert_hardened = certify_set(hardened_model, hardened_t, test_drones, feat_std_arr, drone_idx, SMOOTH_SIGMA)
    print(f"  original  : mean_R={cert_original['mean_certified_radius']:.3f} "
          f"frac_R>=1.04={cert_original['frac_certified_beyond_attack_median_1.04']:.1%} "
          f"smoothed_recall={cert_original['smoothed_drone_recall']:.3f}")
    print(f"  hardened  : mean_R={cert_hardened['mean_certified_radius']:.3f} "
          f"frac_R>=1.04={cert_hardened['frac_certified_beyond_attack_median_1.04']:.1%} "
          f"smoothed_recall={cert_hardened['smoothed_drone_recall']:.3f}")

    print("\n=== 3: physics plausibility gate on harvested evasions ===")
    gate = IsolationForest(n_estimators=300, contamination=0.05, random_state=C.SEED)
    gate.fit(train_drones[C.FEATURE_COLUMNS])
    proba = reorder_proba(original, test_drones[C.FEATURE_COLUMNS])
    correct = test_drones[threshold_predict(proba, t0) == "drone"].reset_index(drop=True)
    sample = correct.sample(min(N_REATTACK, len(correct)), random_state=C.SEED)
    rng_g = np.random.default_rng(C.SEED)
    adv_vectors, clean_vectors = [], []
    for _, row in sample.iterrows():
        x0 = row[C.FEATURE_COLUMNS].values.astype(np.float64)
        ok, _, delta = attack_one(original, feat_std_arr, x0, t0, drone_idx, rng_g)
        clean_vectors.append(x0)
        if ok:
            adv_vectors.append(apply_perturbation(x0, delta))
    adv_flagged = (gate.predict(np.array(adv_vectors)) == -1).mean() if adv_vectors else None
    clean_flagged = (gate.predict(np.array(clean_vectors)) == -1).mean()
    print(f"  plausibility gate flags {adv_flagged:.1%} of successful evasions as anomalous "
          f"(vs {clean_flagged:.1%} false-flag on clean real drones)")

    out = {
        "iterative_adversarial_training": {
            "n_rounds": N_ROUNDS, "rounds": rounds, "evasion_curve": evasion_curve,
            "plateaued": bool(plateaued), "final_evasion_rate": evasion_curve[-1],
        },
        "certified_robustness_randomized_smoothing": {
            "sigma_std_units": SMOOTH_SIGMA, "n0": SMOOTH_N0, "n": SMOOTH_N, "alpha": SMOOTH_ALPHA,
            "original_model": cert_original, "hardened_model": cert_hardened,
        },
        "physics_plausibility_gate": {
            "n_evasions_tested": len(adv_vectors),
            "frac_evasions_flagged_anomalous": adv_flagged,
            "frac_clean_drones_false_flagged": float(clean_flagged),
        },
    }
    (C.REPORTS_DIR / "adversarial_defense_complete_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'final_model_iteratively_hardened.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'adversarial_defense_complete_results.json'}")


if __name__ == "__main__":
    main()
