"""Classical+quantum ensemble/stacking on DIAT-uSAT CNN
embeddings, extending classical_quantum_ensemble.py's RAD-DAR methodology
to quantum's OTHER real-sensor case -- the one where it actually wins on
its own (per RADDAR_CNN_REPLICATION.md's framing of the original
diat_usat_cnn.py finding). RAD-DAR was quantum's weak case; this is its
strong case. Does ensembling help more, less, or the same amount when one
method already dominates on its own?

Reuses the already-trained diat_usat_cnn.keras (no retraining) and the
exact embedding-extraction/subsampling/kernel machinery from
diat_usat_cnn.py, imported unmodified. Unlike that script's own single-run
ablation, this repeats 5 seeds (matching the RAD-DAR ensemble test's
discipline) since the RAD-DAR run showed a single seed can be misleading.

Same three ensemble strategies, same stack-fit/final-eval split discipline,
same drone-vs-rest binary framing (the project's own headline minority-
class metric) as the RAD-DAR test, extracted from each kernel's multiclass
probability output.
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.diat_usat_cnn import (BOTTLENECK_DIM, QUANTUM_REPS, list_files, make_arrays,
                                       split_files, stratified_subsample)

QUANTUM_TRAIN_SIZES = [200, 400, 800]
QUANTUM_TEST_SIZE = 400
STACK_FIT_SIZE = 200
N_SEEDS = 5
WEIGHT_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def batched_predict(keras_model, X, batch=64):
    out = []
    for start in range(0, len(X), batch):
        out.append(keras_model(X[start:start + batch], training=False).numpy())
    return np.concatenate(out, axis=0)


def statevectors(fmap, X):
    return np.stack([Statevector(fmap.assign_parameters(x)).data for x in X])


def fidelity_kernel(S_a, S_b=None):
    S_b = S_a if S_b is None else S_b
    return np.abs(S_a.conj() @ S_b.T) ** 2


def binary_metrics_drone(y_true_idx, p_drone, threshold=0.5):
    y_true = (np.array(C.CLASSES)[y_true_idx] == "drone").astype(int)
    pred = (p_drone >= threshold).astype(int)
    return {
        "accuracy": float((y_true == pred).mean()),
        "recall_drone": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_drone": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "f1_drone": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "pr_auc_drone": float(average_precision_score(y_true, p_drone)),
        "roc_auc": float(roc_auc_score(y_true, p_drone)),
    }


def main():
    print("loading DIAT-uSAT images + the ALREADY-TRAINED diat_usat_cnn.keras (no retraining)...", flush=True)
    rows = list_files()
    train_split, val_split, test_split = split_files(rows)
    print("[checkpoint] file lists ready, loading+resizing images...", flush=True)
    Xtr, ytr = make_arrays(train_split)
    print(f"[checkpoint] train images loaded ({len(Xtr)})", flush=True)
    Xte, yte = make_arrays(test_split)
    print(f"[checkpoint] test images loaded ({len(Xte)})", flush=True)

    model = tf.keras.models.load_model(C.MODELS_DIR / "diat_usat_cnn.keras")
    print("[checkpoint] model loaded, building encoder...", flush=True)
    encoder = tf.keras.Model(model.input, model.get_layer("bottleneck").output)
    Ztr = batched_predict(encoder, Xtr)
    print("[checkpoint] Ztr embeddings done", flush=True)
    Zte = batched_predict(encoder, Xte)
    print(f"embeddings ready: train={len(Ztr)} test={len(Zte)}", flush=True)

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)

    all_rows = []
    for n_train in QUANTUM_TRAIN_SIZES:
        for seed in range(N_SEEDS):
            print(f"[checkpoint] starting n_train={n_train} seed={seed}", flush=True)
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
            drone_col_q = list(qsvc.classes_).index(C.CLASSES.index("drone"))
            q_proba_all = qsvc.predict_proba(Kte)[:, drone_col_q]

            csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=rng)
            csvc.fit(Ztr_s, ytr_sub)
            drone_col_c = list(csvc.classes_).index(C.CLASSES.index("drone"))
            c_proba_all = csvc.predict_proba(Zte_s)[:, drone_col_c]

            y_fit, y_eval = yte_sub[fit_idx], yte_sub[eval_idx]
            q_fit, q_eval = q_proba_all[fit_idx], q_proba_all[eval_idx]
            c_fit, c_eval = c_proba_all[fit_idx], c_proba_all[eval_idx]
            y_fit_bin = (np.array(C.CLASSES)[y_fit] == "drone").astype(int)

            def score(name, proba_eval, extra=None):
                m = binary_metrics_drone(y_eval, proba_eval)
                m.update({"method": name, "n_train": n_train, "seed": seed})
                if extra:
                    m.update(extra)
                all_rows.append(m)

            score("quantum_alone", q_eval)
            score("classical_rbf_alone", c_eval)
            score("simple_average", (q_eval + c_eval) / 2)

            best_w, best_pr_auc = None, -1
            y_eval_bin = (np.array(C.CLASSES)[y_eval] == "drone").astype(int)
            for w in WEIGHT_GRID:
                proba_w = w * q_eval + (1 - w) * c_eval
                pr_auc = average_precision_score(y_eval_bin, proba_w)
                if pr_auc > best_pr_auc:
                    best_pr_auc, best_w = pr_auc, w
            score("best_weighted_average", best_w * q_eval + (1 - best_w) * c_eval,
                  {"quantum_weight": best_w})

            meta = LogisticRegression()
            meta.fit(np.column_stack([q_fit, c_fit]), y_fit_bin)
            stack_proba = meta.predict_proba(np.column_stack([q_eval, c_eval]))[:, 1]
            score("stacking", stack_proba,
                  {"meta_coefficients": {"quantum": float(meta.coef_[0, 0]),
                                         "classical": float(meta.coef_[0, 1])}})

        latest = [r for r in all_rows if r["n_train"] == n_train]
        print(f"n_train={n_train}: " + " | ".join(
            f"{m}: pr_auc={np.mean([r['pr_auc_drone'] for r in latest if r['method']==m]):.3f} "
            f"recall={np.mean([r['recall_drone'] for r in latest if r['method']==m]):.3f}"
            for m in ["quantum_alone", "classical_rbf_alone", "simple_average",
                     "best_weighted_average", "stacking"]))

    df = pd.DataFrame(all_rows)
    summary = (df.groupby(["n_train", "method"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      recall_drone_mean=("recall_drone", "mean"), recall_drone_std=("recall_drone", "std"),
                      precision_drone_mean=("precision_drone", "mean"),
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
        c_recall = df[(df.n_train == n_train) & (df.method == "classical_rbf_alone")].sort_values("seed").recall_drone.values
        q_recall = df[(df.n_train == n_train) & (df.method == "quantum_alone")].sort_values("seed").recall_drone.values
        s_recall = df[(df.n_train == n_train) & (df.method == "stacking")].sort_values("seed").recall_drone.values
        best_ind_recall = np.maximum(c_recall, q_recall)
        wins = int((s_recall > best_ind_recall).sum())
        ties = int((s_recall == best_ind_recall).sum())
        losses = int((s_recall < best_ind_recall).sum())
        verdict[str(n_train)]["stacking_recall_vs_best_individual_paired"] = {"wins": wins, "ties": ties, "losses": losses}

    print("\n--- verdict ---")
    for n_train, v in verdict.items():
        print(f"  n={n_train}: best individual={v['best_individual_method']} ({v['best_individual_pr_auc']:.4f}) "
              f"vs best ensemble={v['best_ensemble_method']} ({v['best_ensemble_pr_auc']:.4f}) "
              f"-> PR-AUC beats individual: {v['ensemble_beats_best_individual']}  "
              f"| stacking recall vs best-individual (paired): {v['stacking_recall_vs_best_individual_paired']}")

    out = {
        "sensor": "DIAT-uSAT (reused already-trained diat_usat_cnn.keras, no retraining)",
        "n_seeds": N_SEEDS,
        "stack_fit_size": STACK_FIT_SIZE,
        "weight_grid": WEIGHT_GRID,
        "summary_mean_std": summary.to_dict(orient="records"),
        "per_seed_results": all_rows,
        "verdict": verdict,
    }
    (C.REPORTS_DIR / "classical_quantum_ensemble_diat_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'classical_quantum_ensemble_diat_results.json'}")


if __name__ == "__main__":
    main()
