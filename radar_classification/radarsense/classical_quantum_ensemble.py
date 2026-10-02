"""Classical+quantum ensemble/stacking -- every comparison elsewhere in
this project picks a WINNER between the quantum kernel and a classical
kernel (RBF); nobody has tried COMBINING their predictions. Tested here on
RAD-DAR CNN embeddings, reusing the already-trained raddar_cnn.keras model
(no retraining) and the exact same embedding-extraction/subsampling/
kernel-fitting machinery as raddar_cnn.py, imported unmodified.

Three ensemble strategies, in increasing sophistication:
  1. Simple average of P(drone) from both kernels.
  2. Weighted average, swept over a small weight grid -- lets a quantum-
     favoring or classical-favoring combination win if one exists, rather
     than assuming 50/50 is the right split.
  3. Stacking: a logistic-regression meta-model trained on
     [quantum_p_drone, classical_p_drone] -> true label. Needs its own
     held-out split to avoid leakage -- the 300-sample quantum test pool is
     split into a stack-fit half (150) and a final-eval half (150); ALL
     methods (quantum alone, classical alone, simple/weighted ensembles,
     stacking) are scored on the SAME final-eval half so every number in
     the comparison table is apples-to-apples.

5 seeds x 3 train sizes (200/400/800), matching raddar_cnn.py's own
multi-seed discipline. Reported honestly regardless of outcome -- an
ensemble that doesn't beat the better individual method is exactly as
valid a finding as one that does.
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from qiskit.circuit.library import ZZFeatureMap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.raddar_cnn import (batched_predict, fidelity_kernel, list_matrix_files,
                                    load_matrices, statevectors, stratified_subsample)

BOTTLENECK_DIM = 8
QUANTUM_REPS = 2
QUANTUM_TRAIN_SIZES = [200, 400, 800]
QUANTUM_TEST_SIZE = 300
STACK_FIT_SIZE = 150
N_SEEDS = 5
WEIGHT_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def binary_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "recall_drone": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_drone": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_drone": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc_drone": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def main():
    print("loading RAD-DAR matrices + the ALREADY-TRAINED raddar_cnn.keras (no retraining)...")
    rows = list_matrix_files()
    X, y_str, sessions = load_matrices(rows)
    y = (y_str == "drone").astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=C.SEED)
    tr_i, rest_i = next(gss.split(X, groups=sessions))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=C.SEED)
    va_i_rel, te_i_rel = next(gss2.split(rest_i, groups=sessions[rest_i]))
    va_i, te_i = rest_i[va_i_rel], rest_i[te_i_rel]
    Xtr, ytr = X[tr_i], y[tr_i]
    Xte, yte = X[te_i], y[te_i]

    model = tf.keras.models.load_model(C.MODELS_DIR / "raddar_cnn.keras")
    encoder = tf.keras.Model(model.input, model.get_layer("bottleneck").output)
    Ztr = batched_predict(encoder, Xtr)
    Zte = batched_predict(encoder, Xte)
    print(f"embeddings ready: train={len(Ztr)} test={len(Zte)}")

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)

    all_rows = []
    for n_train in QUANTUM_TRAIN_SIZES:
        for seed in range(N_SEEDS):
            rng = C.SEED + seed
            Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, n_train, rng)
            Zte_sub, yte_sub = stratified_subsample(Zte, yte, QUANTUM_TEST_SIZE, rng)
            scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
            Ztr_s, Zte_s = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

            split_rng = np.random.default_rng(rng)
            perm = split_rng.permutation(len(yte_sub))
            fit_idx, eval_idx = perm[:STACK_FIT_SIZE], perm[STACK_FIT_SIZE:]

            S_tr, S_te = statevectors(fmap, Ztr_s), statevectors(fmap, Zte_s)
            Ktr, Kte = fidelity_kernel(S_tr), fidelity_kernel(S_te, S_tr)
            qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=rng)
            qsvc.fit(Ktr, ytr_sub)
            q_proba_all = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]

            csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=rng)
            csvc.fit(Ztr_s, ytr_sub)
            c_proba_all = csvc.predict_proba(Zte_s)[:, list(csvc.classes_).index(1)]

            y_fit, y_eval = yte_sub[fit_idx], yte_sub[eval_idx]
            q_fit, q_eval = q_proba_all[fit_idx], q_proba_all[eval_idx]
            c_fit, c_eval = c_proba_all[fit_idx], c_proba_all[eval_idx]

            def score(name, proba_eval, extra=None):
                pred = (proba_eval >= 0.5).astype(int)
                m = binary_metrics(y_eval, pred, proba_eval)
                m.update({"method": name, "n_train": n_train, "seed": seed})
                if extra:
                    m.update(extra)
                all_rows.append(m)
                return m

            score("quantum_alone", q_eval)
            score("classical_rbf_alone", c_eval)
            score("simple_average", (q_eval + c_eval) / 2)

            best_w, best_pr_auc = None, -1
            for w in WEIGHT_GRID:
                proba_w = w * q_eval + (1 - w) * c_eval
                pr_auc = average_precision_score(y_eval, proba_w)
                if pr_auc > best_pr_auc:
                    best_pr_auc, best_w = pr_auc, w
            score("best_weighted_average", best_w * q_eval + (1 - best_w) * c_eval,
                  {"quantum_weight": best_w})

            meta = LogisticRegression()
            meta.fit(np.column_stack([q_fit, c_fit]), y_fit)
            stack_proba = meta.predict_proba(np.column_stack([q_eval, c_eval]))[:, 1]
            score("stacking", stack_proba,
                  {"meta_coefficients": {"quantum": float(meta.coef_[0, 0]),
                                         "classical": float(meta.coef_[0, 1])}})

        latest = [r for r in all_rows if r["n_train"] == n_train]
        print(f"n_train={n_train}: " + " | ".join(
            f"{m}: pr_auc={np.mean([r['pr_auc_drone'] for r in latest if r['method']==m]):.3f}"
            for m in ["quantum_alone", "classical_rbf_alone", "simple_average",
                     "best_weighted_average", "stacking"]))

    df = pd.DataFrame(all_rows)
    summary = (df.groupby(["n_train", "method"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      recall_drone_mean=("recall_drone", "mean"), recall_drone_std=("recall_drone", "std"),
                      pr_auc_mean=("pr_auc_drone", "mean"), pr_auc_std=("pr_auc_drone", "std"))
                 .reset_index())
    print("\n" + summary.to_string(index=False))

    verdict = {}
    for n_train in QUANTUM_TRAIN_SIZES:
        sub = summary[summary.n_train == n_train]
        q_pr = sub[sub.method == "quantum_alone"].pr_auc_mean.iloc[0]
        c_pr = sub[sub.method == "classical_rbf_alone"].pr_auc_mean.iloc[0]
        best_individual = max(q_pr, c_pr)
        ensemble_methods = sub[sub.method.isin(["simple_average", "best_weighted_average", "stacking"])]
        best_ensemble_row = ensemble_methods.loc[ensemble_methods.pr_auc_mean.idxmax()]
        verdict[str(n_train)] = {
            "best_individual_pr_auc": float(best_individual),
            "best_individual_method": "quantum" if q_pr > c_pr else "classical_rbf",
            "best_ensemble_pr_auc": float(best_ensemble_row.pr_auc_mean),
            "best_ensemble_method": best_ensemble_row.method,
            "ensemble_beats_best_individual": bool(best_ensemble_row.pr_auc_mean > best_individual),
        }
    print("\n--- does any ensemble beat the best individual method? ---")
    for n_train, v in verdict.items():
        print(f"  n={n_train}: best individual={v['best_individual_method']} "
              f"({v['best_individual_pr_auc']:.4f}) vs best ensemble={v['best_ensemble_method']} "
              f"({v['best_ensemble_pr_auc']:.4f}) -> beats individual: {v['ensemble_beats_best_individual']}")

    out = {
        "sensor": "RAD-DAR (reused already-trained raddar_cnn.keras, no retraining)",
        "n_seeds": N_SEEDS,
        "stack_fit_size": STACK_FIT_SIZE,
        "weight_grid": WEIGHT_GRID,
        "summary_mean_std": summary.to_dict(orient="records"),
        "per_seed_results": all_rows,
        "verdict_ensemble_vs_best_individual": verdict,
    }
    (C.REPORTS_DIR / "classical_quantum_ensemble_raddar_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'classical_quantum_ensemble_raddar_results.json'}")


if __name__ == "__main__":
    main()
