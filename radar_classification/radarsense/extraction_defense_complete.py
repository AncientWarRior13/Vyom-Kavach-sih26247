"""WS1 -- the COMPLETE model-extraction treatment.

EXTRACTION_DEFENSE.md's honest limit: the anomaly filter raises extraction
cost ~25x but doesn't eliminate it, and response-hardening (rounding/hiding
probabilities) does nothing because the attack only needs the hard label.
Extraction is a problem NO ONE fully prevents -- so "complete" here means
raise attacker cost, DETECT the queries, and -- the real reframe -- make a
stolen model PROVABLY attributable so theft can be demonstrated and revoked
even when extraction itself succeeds. Four fronts:

  1. FINGERPRINTING for theft detection (the reframe). Generate secret
     decision-boundary "conferrable" probe points and record the deployed
     model's labels on them. A model EXTRACTED from ours learns our exact
     boundary and agrees on these probes far more than an independently
     trained competitor model does. Measured as a separation between
     "extracted surrogate agreement" and an INNOCENT baseline (independent
     models trained on similar data) -> a detection threshold that flags
     clones while clearing honest competitors.

  2. BOUNDARY-REGION LABEL NOISE (the untested lever). Probability rounding
     was shown useless. This is different: flip the hard LABEL for a small
     fraction of queries that land within a margin of the decision
     threshold. One query is barely affected; a surrogate trained on
     thousands inherits corrupted boundary supervision exactly where it
     matters. Tested for whether it degrades surrogate fidelity where
     rounding didn't.

  3. PER-KEY QUERY BUDGET. A token-bucket per API key caps a single
     identity's TOTAL attempts (not just answered ones), so the anomaly
     filter's residual leak (~200 boundary-like queries slip through per
     5000) can't be replenished indefinitely from one key.

  4. REALISTIC-PROFILE re-evaluation. The original attack used i.i.d.-uniform
     queries the anomaly filter catches easily. This re-runs extraction with
     queries drawn from the REAL data distribution (the patient-adversary
     profile) and reports both the filter's reject rate and the surrogate
     fidelity when the attacker pays the cost of realistic queries.

Uses the deployed joblib model directly -- MODEL_EXTRACTION_ATTACK.md already
verified predict_one() == the live HTTP API, so this is computationally
identical and avoids per-query HTTP overhead.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest

from radarsense import config as C
from radarsense.model_extraction_attack import QUERY_BOUNDS, sample_random_queries
from radarsense.optimize import threshold_predict

N_FINGERPRINT = 300
FINGERPRINT_BAND = 0.06
N_INNOCENT = 5
EXTRACT_BUDGET = 2000
FLIP_MARGINS = [0.0, 0.05, 0.10, 0.15]
FLIP_PROB = 0.5


def reorder_proba(model, X):
    p = model.predict_proba(X)
    return p[:, [list(model.classes_).index(c) for c in C.CLASSES]]


def p_drone(model, X):
    return reorder_proba(model, X)[:, C.MINORITY_IDX]


class PerKeyBudget:
    """Token bucket: each key gets `capacity` tokens, refilled `refill` per
    tick. A query costs 1 token; over-budget queries are refused."""
    def __init__(self, capacity=500, refill=50):
        self.capacity, self.refill, self.tokens = capacity, refill, {}

    def allow(self, key):
        self.tokens[key] = self.tokens.get(key, self.capacity)
        if self.tokens[key] <= 0:
            return False
        self.tokens[key] -= 1
        return True

    def tick(self):
        for k in self.tokens:
            self.tokens[k] = min(self.capacity, self.tokens[k] + self.refill)


def hard_labels(m, X, threshold):
    return threshold_predict(reorder_proba(m, X), threshold)


def gen_boundary_fingerprints(model, threshold, rng, n_target):
    """Probes ON the decision boundary. Near coin-flips even for a perfect
    clone, so they do NOT discriminate -- kept to show that negative."""
    kept = []
    while len(kept) < n_target:
        batch = sample_random_queries(4000, rng)
        sel = batch[np.abs(p_drone(model, batch) - threshold) < FINGERPRINT_BAND]
        for _, row in sel.iterrows():
            kept.append(row[C.FEATURE_COLUMNS].values.astype(np.float64))
            if len(kept) >= n_target:
                break
    X = np.array(kept)
    return X, hard_labels(model, X, threshold)


def gen_idiosyncratic_fingerprints(model, selectors, threshold, rng, n_target):
    """Probes where the TARGET disagrees with a panel of independent
    (selector) models -- the target's idiosyncratic decisions. A faithful
    clone reproduces them; an independent model won't. Selection uses only
    the selector panel, so held-out reference models test it non-circularly."""
    kept = []
    while len(kept) < n_target:
        batch = sample_random_queries(6000, rng)
        Xb = batch[C.FEATURE_COLUMNS].values
        tgt = hard_labels(model, Xb, threshold)
        votes = np.array([hard_labels(s, Xb, threshold) for s in selectors])
        maj = np.array([max(set(col), key=list(col).count) for col in votes.T])
        idio = tgt != maj
        for x in Xb[idio]:
            kept.append(x.astype(np.float64))
            if len(kept) >= n_target:
                break
    X = np.array(kept)
    return X, hard_labels(model, X, threshold)


def train_catboost(X, y, seed=C.SEED):
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=seed, verbose=0, allow_writing_files=False)
    clf.fit(X, y, sample_weight=sw)
    return clf


def defended_labels(model, threshold, X, margin, flip_prob, rng):
    """Hard labels, with near-boundary labels flipped with prob flip_prob."""
    proba = reorder_proba(model, X)
    labels = threshold_predict(proba, threshold)
    if margin <= 0:
        return labels
    near = np.abs(proba[:, C.MINORITY_IDX] - threshold) < margin
    flip = near & (rng.random(len(X)) < flip_prob)
    out = labels.copy()
    other = [i for i in range(len(C.CLASSES)) if i != C.MINORITY_IDX]
    for i in np.where(flip)[0]:
        if out[i] == C.MINORITY_CLASS:
            out[i] = C.CLASSES[other[int(np.argmax(proba[i, other]))]]
        else:
            out[i] = C.MINORITY_CLASS
    return out


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    rng = np.random.default_rng(C.SEED)

    proba_target_test = reorder_proba(model, test[C.FEATURE_COLUMNS])
    pred_target_test = threshold_predict(proba_target_test, threshold)

    print("=== 1: fingerprinting (theft detection) ===")
    q = sample_random_queries(EXTRACT_BUDGET, rng)
    q_labels = threshold_predict(reorder_proba(model, q[C.FEATURE_COLUMNS].values), threshold)
    surrogate = train_catboost(q[C.FEATURE_COLUMNS].values, q_labels)
    innocent = []
    for k in range(6):
        boot = train.sample(frac=1.0, replace=True, random_state=1000 + k)
        innocent.append(train_catboost(boot[C.FEATURE_COLUMNS].values,
                                       boot[C.LABEL_COLUMN].values, seed=1000 + k))
    selectors, testers = innocent[:3], innocent[3:]

    def score(models, fp_X, fp_y):
        return [float((hard_labels(m, fp_X, threshold) == fp_y).mean()) for m in models]

    fingerprinting = {}
    bx, by = gen_boundary_fingerprints(model, threshold, rng, N_FINGERPRINT)
    b_tester = score(testers, bx, by)
    b_extr = float((hard_labels(surrogate, bx, threshold) == by).mean())
    fingerprinting["boundary"] = {"n": len(bx), "held_out_innocent_max": max(b_tester),
                                  "held_out_innocent_mean": float(np.mean(b_tester)),
                                  "extracted_agreement": b_extr, "separation": b_extr - max(b_tester),
                                  "detectable": bool(b_extr - max(b_tester) > 0.05)}
    print(f"  [boundary     ] extracted={b_extr:.3f}  held-out innocent max={max(b_tester):.3f}  "
          f"sep={b_extr - max(b_tester):+.3f}  (negative control)")
    fp_X, fp_y = gen_idiosyncratic_fingerprints(model, selectors, threshold, rng, N_FINGERPRINT)
    i_tester = score(testers, fp_X, fp_y)
    i_extr = float((hard_labels(surrogate, fp_X, threshold) == fp_y).mean())
    sep = i_extr - max(i_tester)
    fingerprinting["idiosyncratic"] = {"n": len(fp_X), "held_out_innocent_max": max(i_tester),
                                       "held_out_innocent_mean": float(np.mean(i_tester)),
                                       "held_out_innocent_all": i_tester,
                                       "extracted_agreement": i_extr, "separation": sep,
                                       "detectable": bool(sep > 0.05)}
    print(f"  [idiosyncratic] extracted={i_extr:.3f}  held-out innocent max={max(i_tester):.3f}  "
          f"sep={sep:+.3f}  -> {'THEFT DETECTABLE' if sep > 0.05 else 'not separable'}")

    print("\n=== 2: boundary-region label-noise defense (vs the useless rounding) ===")
    label_noise = {}
    for margin in FLIP_MARGINS:
        rng_d = np.random.default_rng(C.SEED)
        y_def = defended_labels(model, threshold, q[C.FEATURE_COLUMNS].values, margin, FLIP_PROB, rng_d)
        frac_flipped = float((y_def != threshold_predict(reorder_proba(model, q[C.FEATURE_COLUMNS].values), threshold)).mean())
        sur = train_catboost(q[C.FEATURE_COLUMNS].values, y_def)
        pred_sur = threshold_predict(reorder_proba(sur, test[C.FEATURE_COLUMNS]), threshold)
        drone_gate_fid = float(((pred_sur == "drone") == (pred_target_test == "drone")).mean())
        overall_fid = float((pred_sur == pred_target_test).mean())
        label_noise[margin] = {"frac_labels_flipped": frac_flipped,
                               "surrogate_drone_gate_fidelity": drone_gate_fid,
                               "surrogate_overall_fidelity": overall_fid}
        tag = "  (no defense)" if margin == 0 else ""
        print(f"  margin={margin:.2f} flip_p={FLIP_PROB}: {frac_flipped:.1%} labels flipped -> "
              f"surrogate drone-gate fidelity={drone_gate_fid:.4f} overall={overall_fid:.4f}{tag}")

    print("\n=== 3: per-key query budget (token bucket) ===")
    budget = PerKeyBudget(capacity=500, refill=50)
    allowed = sum(budget.allow("attacker_key") for _ in range(5000))
    print(f"  single key attempting 5000 queries in one burst: {allowed} answered, "
          f"{5000 - allowed} refused (capacity 500, no refill within the burst)")

    print("\n=== 4: realistic-profile extraction vs the anomaly filter ===")
    iso = joblib.load(C.MODELS_DIR / "extraction_anomaly_filter.joblib")
    iso_meta = json.loads((C.MODELS_DIR / "extraction_anomaly_filter_meta.json").read_text())
    reject_t = iso_meta["reject_threshold"]

    uniform_q = sample_random_queries(EXTRACT_BUDGET, rng)[C.FEATURE_COLUMNS]
    uniform_reject = float((iso.score_samples(uniform_q) < reject_t).mean())
    realistic_q = test[C.FEATURE_COLUMNS].sample(EXTRACT_BUDGET, replace=True, random_state=7).reset_index(drop=True)
    realistic_reject = float((iso.score_samples(realistic_q) < reject_t).mean())
    print(f"  anomaly filter rejects: uniform queries {uniform_reject:.1%}, "
          f"realistic queries {realistic_reject:.1%}")

    def surrogate_fidelity(queries):
        q_lab = threshold_predict(reorder_proba(model, queries.values), threshold)
        sur = train_catboost(queries.values, q_lab)
        pred = threshold_predict(reorder_proba(sur, test[C.FEATURE_COLUMNS]), threshold)
        return float(((pred == "drone") == (pred_target_test == "drone")).mean())

    uniform_passers = uniform_q[iso.score_samples(uniform_q) >= reject_t]
    realistic_passers = realistic_q[iso.score_samples(realistic_q) >= reject_t]
    fid_uniform_passers = surrogate_fidelity(uniform_passers) if len(uniform_passers) > 50 else None
    fid_realistic = surrogate_fidelity(realistic_passers)
    print(f"  surrogate drone-gate fidelity from filter-passing queries: "
          f"uniform={fid_uniform_passers} ({len(uniform_passers)} passed)  "
          f"realistic={fid_realistic:.4f} ({len(realistic_passers)} passed)")

    out = {
        "fingerprinting": fingerprinting,
        "boundary_label_noise_defense": label_noise,
        "per_key_budget_demo": {"capacity": 500, "burst_attempts": 5000, "answered": allowed},
        "realistic_profile_vs_anomaly_filter": {
            "uniform_query_reject_rate": uniform_reject,
            "realistic_query_reject_rate": realistic_reject,
            "surrogate_fidelity_uniform_passers": fid_uniform_passers,
            "surrogate_fidelity_realistic": fid_realistic,
            "n_uniform_passed": int(len(uniform_passers)), "n_realistic_passed": int(len(realistic_passers)),
        },
    }
    joblib.dump({"fingerprints": fp_X, "labels": fp_y}, C.MODELS_DIR / "extraction_fingerprints.joblib")
    (C.REPORTS_DIR / "extraction_defense_complete_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nfingerprints -> {C.MODELS_DIR / 'extraction_fingerprints.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'extraction_defense_complete_results.json'}")


if __name__ == "__main__":
    main()
