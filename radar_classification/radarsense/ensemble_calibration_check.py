"""Answers a concrete question raised by ensemble_decision_explainer.py's own
failure case: the ensemble's "dominant_model" explanation and the stacking
meta-model both trust a model's RAW probability as a genuine confidence
signal. If one model's confidence is miscalibrated -- says "80% sure" but
is only right 50% of the time it says that -- the ensemble is being misled
by loud confidence, not real reliability, and that's not visible from the
decision log alone.

Reuses classical_quantum_ensemble.py's exact setup (same qsvc/csvc fitting,
same embeddings, same subsampling), but pools predictions across ALL 15
seed x train-size combinations (5 seeds x 3 sizes) into one set for a
statistically meaningful reliability diagram -- a single n=300 test pool
per run is too small on its own.

Checks:
  1. Raw calibration (ECE, Brier, reliability diagram) for quantum vs
     classical, pooled.
  2. Specifically: overconfidence check -- among predictions where a model
     said p>=0.8 (or p<=0.2), what fraction were actually correct? If
     that's well below ~80%/~20% for one model, it's a real, measurable
     overconfidence problem, not a hypothetical one.
  3. If calibration is meaningfully different between the two, re-runs the
     stacking ensemble using ISOTONIC-CALIBRATED probabilities instead of
     raw ones (calibrators fit on a held-out split), to see whether fixing
     the confidence signal changes the ensemble's own accuracy/recall.
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from qiskit.circuit.library import ZZFeatureMap
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.calibrate import ece
from radarsense.raddar_cnn import (batched_predict, fidelity_kernel, list_matrix_files,
                                    load_matrices, statevectors, stratified_subsample)

QUANTUM_TRAIN_SIZES = [200, 400, 800]
QUANTUM_TEST_SIZE = 300
N_SEEDS = 5
BOTTLENECK_DIM = 8
QUANTUM_REPS = 2


def main():
    print("reproducing the RAD-DAR ensemble setup, pooling predictions across "
          "5 seeds x 3 train sizes for calibration analysis...")
    rows = list_matrix_files()
    X, y_str, sessions = load_matrices(rows)
    y = (y_str == "drone").astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=C.SEED)
    tr_i, rest_i = next(gss.split(X, groups=sessions))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=C.SEED)
    va_i_rel, te_i_rel = next(gss2.split(rest_i, groups=sessions[rest_i]))
    te_i = rest_i[te_i_rel]
    Xtr, ytr = X[tr_i], y[tr_i]
    Xte, yte = X[te_i], y[te_i]

    model = tf.keras.models.load_model(C.MODELS_DIR / "raddar_cnn.keras")
    encoder = tf.keras.Model(model.input, model.get_layer("bottleneck").output)
    Ztr = batched_predict(encoder, Xtr)
    Zte = batched_predict(encoder, Xte)

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)

    pool_y, pool_q, pool_c = [], [], []
    for n_train in QUANTUM_TRAIN_SIZES:
        for seed in range(N_SEEDS):
            rng = C.SEED + seed
            Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, n_train, rng)
            Zte_sub, yte_sub = stratified_subsample(Zte, yte, QUANTUM_TEST_SIZE, rng)
            scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
            Ztr_s, Zte_s = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

            S_tr, S_te = statevectors(fmap, Ztr_s), statevectors(fmap, Zte_s)
            Ktr, Kte = fidelity_kernel(S_tr), fidelity_kernel(S_te, S_tr)
            qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=rng)
            qsvc.fit(Ktr, ytr_sub)
            q_p = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]

            csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=rng)
            csvc.fit(Ztr_s, ytr_sub)
            c_p = csvc.predict_proba(Zte_s)[:, list(csvc.classes_).index(1)]

            pool_y.append(yte_sub); pool_q.append(q_p); pool_c.append(c_p)
        print(f"n_train={n_train}: done")

    y_all = np.concatenate(pool_y)
    q_all = np.concatenate(pool_q)
    c_all = np.concatenate(pool_c)
    print(f"\npooled {len(y_all)} predictions across all 15 runs")

    report = {}
    for name, p in [("quantum", q_all), ("classical_rbf", c_all)]:
        report[name] = {
            "brier": float(brier_score_loss(y_all, p)),
            "ece": ece(y_all, p),
        }
        print(f"{name}: Brier={report[name]['brier']:.4f} ECE={report[name]['ece']:.4f}")

    print("\n--- overconfidence check ---")
    for name, p in [("quantum", q_all), ("classical_rbf", c_all)]:
        hi_mask = p >= 0.8
        lo_mask = p <= 0.2
        hi_actual = float(y_all[hi_mask].mean()) if hi_mask.sum() else None
        lo_actual = float(1 - y_all[lo_mask].mean()) if lo_mask.sum() else None
        report[name]["high_confidence_check"] = {
            "n_predictions_p_geq_0.8": int(hi_mask.sum()),
            "actual_drone_fraction_when_p_geq_0.8": hi_actual,
            "n_predictions_p_leq_0.2": int(lo_mask.sum()),
            "actual_not_drone_fraction_when_p_leq_0.2": lo_actual,
        }
        print(f"{name}: when p>=0.8 (n={hi_mask.sum()}), actually drone {hi_actual:.1%} of the time "
              f"({'OVERCONFIDENT' if hi_actual < 0.75 else 'roughly trustworthy'})")
        print(f"{name}: when p<=0.2 (n={lo_mask.sum()}), actually not-drone {lo_actual:.1%} of the time "
              f"({'OVERCONFIDENT' if lo_actual < 0.75 else 'roughly trustworthy'})")

    print("\n--- does isotonic calibration change the stacking ensemble's own accuracy? ---")
    rng_split = np.random.default_rng(C.SEED)
    perm = rng_split.permutation(len(y_all))
    half = len(perm) // 2
    cal_idx, test_idx = perm[:half], perm[half:]

    iso_q = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(q_all[cal_idx], y_all[cal_idx])
    iso_c = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c_all[cal_idx], y_all[cal_idx])
    q_cal, c_cal = iso_q.predict(q_all[test_idx]), iso_c.predict(c_all[test_idx])
    q_raw, c_raw, y_test = q_all[test_idx], c_all[test_idx], y_all[test_idx]

    def stack_eval(q, c, y, label):
        rng2 = np.random.default_rng(C.SEED)
        perm2 = rng2.permutation(len(y))
        fit_i, eval_i = perm2[:len(y) // 2], perm2[len(y) // 2:]
        meta = LogisticRegression().fit(np.column_stack([q[fit_i], c[fit_i]]), y[fit_i])
        proba = meta.predict_proba(np.column_stack([q[eval_i], c[eval_i]]))[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = {
            "accuracy": float((pred == y[eval_i]).mean()),
            "recall_drone": float(recall_score(y[eval_i], pred, zero_division=0)),
            "precision_drone": float(precision_score(y[eval_i], pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y[eval_i], proba)),
            "meta_coef": {"quantum": float(meta.coef_[0, 0]), "classical": float(meta.coef_[0, 1])},
        }
        print(f"{label}: acc={m['accuracy']:.4f} recall={m['recall_drone']:.4f} "
              f"precision={m['precision_drone']:.4f} pr_auc={m['pr_auc']:.4f} coef={m['meta_coef']}")
        return m

    m_raw = stack_eval(q_raw, c_raw, y_test, "stacking on RAW probabilities    ")
    m_cal = stack_eval(q_cal, c_cal, y_test, "stacking on CALIBRATED probabilities")

    out = {
        "n_pooled_predictions": int(len(y_all)),
        "raw_calibration": report,
        "stacking_raw_vs_calibrated": {"raw": m_raw, "calibrated": m_cal},
    }
    (C.REPORTS_DIR / "ensemble_calibration_check_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'ensemble_calibration_check_results.json'}")


if __name__ == "__main__":
    main()
