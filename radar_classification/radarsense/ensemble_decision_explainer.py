"""Per-detection COMPARISON LOG for the classical+quantum ensemble
(classical_quantum_ensemble.py), so the ensemble's decision is auditable
rather than a single opaque probability number. For a sample of real RAD-DAR
test detections, logs:

  1. What each individual model said (classical RBF SVM, quantum ZZ-kernel
     SVM) and what the ensemble (stacking) said, side by side.
  2. EXACTLY how much each model's vote drove the final ensemble decision --
     not an approximation: the stacking meta-model is a 2-input logistic
     regression, so its contribution breakdown (coef * input, per model) is
     exact, not SHAP/LIME-estimated. No point approximating something that's
     already fully transparent.
  3. WHY each individual model made ITS OWN call -- this is the genuinely
     opaque part (RBF/quantum-kernel SVMs aren't inherently interpretable
     the way the meta-model is), explained via LIME on the 8-dim CNN
     bottleneck embedding, run separately for the classical model and for
     the quantum model. Comparing which embedding dimensions each one
     leans on is itself informative: if they lean on similar dimensions,
     that explains why ensembling barely moves PR-AUC; if they lean on
     different ones, that's the mechanism behind the recall improvement
     found in CLASSICAL_QUANTUM_ENSEMBLE.md.

Reuses the SAME trained qsvc/csvc/meta setup as
classical_quantum_ensemble.py (n_train=800, seed=0 -- the largest, most
stable configuration tested there), no retraining, same embeddings.
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from lime.lime_tabular import LimeTabularExplainer
from qiskit.circuit.library import ZZFeatureMap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.raddar_cnn import (batched_predict, fidelity_kernel, list_matrix_files,
                                    load_matrices, statevectors, stratified_subsample)

N_TRAIN = 800
STACK_FIT_SIZE = 150
N_LOG_SAMPLES = 20
LIME_NUM_SAMPLES = 500
BOTTLENECK_DIM = 8
QUANTUM_REPS = 2
EMBED_FEATURE_NAMES = [f"cnn_embed_dim_{i}" for i in range(BOTTLENECK_DIM)]


def main():
    print("reproducing the RAD-DAR ensemble setup (n_train=800, seed=0)...")
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

    rng = C.SEED
    Ztr_sub, ytr_sub = stratified_subsample(Ztr, ytr, N_TRAIN, rng)
    Zte_sub, yte_sub = stratified_subsample(Zte, yte, 300, rng)
    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Ztr_sub)
    Ztr_s, Zte_s = scaler.transform(Ztr_sub), scaler.transform(Zte_sub)

    split_rng = np.random.default_rng(rng)
    perm = split_rng.permutation(len(yte_sub))
    fit_idx, eval_idx = perm[:STACK_FIT_SIZE], perm[STACK_FIT_SIZE:]

    fmap = ZZFeatureMap(feature_dimension=BOTTLENECK_DIM, reps=QUANTUM_REPS)
    S_tr = statevectors(fmap, Ztr_s)
    S_eval_pool = statevectors(fmap, Zte_s)
    Ktr = fidelity_kernel(S_tr)
    Kte_full = fidelity_kernel(S_eval_pool, S_tr)

    qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=rng)
    qsvc.fit(Ktr, ytr_sub)
    csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=rng)
    csvc.fit(Ztr_s, ytr_sub)

    q_col = list(qsvc.classes_).index(1)
    c_col = list(csvc.classes_).index(1)
    q_proba_all = qsvc.predict_proba(Kte_full)[:, q_col]
    c_proba_all = csvc.predict_proba(Zte_s)[:, c_col]

    meta = LogisticRegression()
    meta.fit(np.column_stack([q_proba_all[fit_idx], c_proba_all[fit_idx]]), yte_sub[fit_idx])
    coef_q, coef_c = float(meta.coef_[0, 0]), float(meta.coef_[0, 1])
    bias = float(meta.intercept_[0])
    print(f"meta-model (exact, linear): p(drone) = sigmoid({coef_q:.3f}*quantum + "
          f"{coef_c:.3f}*classical + {bias:.3f})")

    print("\nbuilding LIME explainers for classical and quantum models on the "
          "8-dim CNN embedding...")
    lime_explainer = LimeTabularExplainer(
        Ztr_s, feature_names=EMBED_FEATURE_NAMES, class_names=["not_drone", "drone"],
        mode="classification", discretize_continuous=True, random_state=C.SEED)

    def classical_predict_fn(Z):
        return csvc.predict_proba(Z)

    def quantum_predict_fn(Z):
        S = statevectors(fmap, Z)
        K = fidelity_kernel(S, S_tr)
        return qsvc.predict_proba(K)

    eval_global_idx = np.arange(len(yte_sub))[eval_idx]
    q_eval, c_eval = q_proba_all[eval_idx], c_proba_all[eval_idx]
    y_eval = yte_sub[eval_idx]
    ens_eval = 1 / (1 + np.exp(-(coef_q * q_eval + coef_c * c_eval + bias)))

    disagreement = np.abs((q_eval >= 0.5).astype(int) - (c_eval >= 0.5).astype(int)).astype(bool)
    disagree_idx = np.where(disagreement)[0]
    agree_idx = np.where(~disagreement)[0]
    print(f"\n{len(disagree_idx)}/{len(y_eval)} eval samples: classical and quantum DISAGREE on the hard label")

    rng_sample = np.random.default_rng(C.SEED)
    chosen = list(rng_sample.choice(disagree_idx, size=min(N_LOG_SAMPLES // 2, len(disagree_idx)), replace=False))
    chosen += list(rng_sample.choice(agree_idx, size=min(N_LOG_SAMPLES - len(chosen), len(agree_idx)), replace=False))

    log_entries = []
    for k in chosen:
        true_label = "drone" if y_eval[k] == 1 else "not_drone"
        z = Zte_s[eval_idx][k]

        q_p, c_p, e_p = float(q_eval[k]), float(c_eval[k]), float(ens_eval[k])
        contrib_q, contrib_c = coef_q * q_p, coef_c * c_p

        lime_c = lime_explainer.explain_instance(z, classical_predict_fn, labels=(1,),
                                                  num_features=5, num_samples=LIME_NUM_SAMPLES)
        lime_q = lime_explainer.explain_instance(z, quantum_predict_fn, labels=(1,),
                                                  num_features=5, num_samples=LIME_NUM_SAMPLES)

        entry = {
            "true_label": true_label,
            "classical": {"p_drone": round(c_p, 4), "decision": "drone" if c_p >= 0.5 else "not_drone",
                         "top_lime_features": lime_c.as_list(label=1)},
            "quantum": {"p_drone": round(q_p, 4), "decision": "drone" if q_p >= 0.5 else "not_drone",
                       "top_lime_features": lime_q.as_list(label=1)},
            "ensemble": {
                "p_drone": round(e_p, 4), "decision": "drone" if e_p >= 0.5 else "not_drone",
                "exact_contribution_breakdown": {
                    "quantum_vote_x_weight": round(contrib_q, 4),
                    "classical_vote_x_weight": round(contrib_c, 4),
                    "bias": round(bias, 4),
                    "dominant_model": "quantum" if abs(contrib_q) > abs(contrib_c) else "classical",
                },
            },
            "models_agreed_on_hard_label": bool((c_p >= 0.5) == (q_p >= 0.5)),
        }
        log_entries.append(entry)
        print(f"\n[{true_label:9s}] classical={entry['classical']['decision']}({c_p:.2f}) "
              f"quantum={entry['quantum']['decision']}({q_p:.2f}) "
              f"-> ensemble={entry['ensemble']['decision']}({e_p:.2f}) "
              f"[driven mainly by {entry['ensemble']['exact_contribution_breakdown']['dominant_model']}]")

    def agg_lime_features(entries, model_key):
        agg = {f: 0.0 for f in EMBED_FEATURE_NAMES}
        for e in entries:
            for feat_desc, weight in e[model_key]["top_lime_features"]:
                matched = next((f for f in EMBED_FEATURE_NAMES if f in feat_desc), None)
                if matched:
                    agg[matched] += abs(weight)
        return sorted(agg.items(), key=lambda kv: -kv[1])

    classical_ranked = agg_lime_features(log_entries, "classical")
    quantum_ranked = agg_lime_features(log_entries, "quantum")
    print(f"\nclassical model leans on (aggregate |LIME weight|): {classical_ranked[:4]}")
    print(f"quantum model leans on (aggregate |LIME weight|): {quantum_ranked[:4]}")

    out = {
        "setup": {"n_train": N_TRAIN, "seed": rng, "meta_model": {"coef_quantum": coef_q,
                  "coef_classical": coef_c, "bias": bias}},
        "n_disagreement_cases_in_eval_pool": int(len(disagree_idx)),
        "n_total_eval_pool": int(len(y_eval)),
        "log_entries": log_entries,
        "aggregate_classical_embedding_reliance": classical_ranked,
        "aggregate_quantum_embedding_reliance": quantum_ranked,
    }
    (C.REPORTS_DIR / "ensemble_decision_log.json").write_text(json.dumps(out, indent=2))
    print(f"\nlog -> {C.REPORTS_DIR / 'ensemble_decision_log.json'}")


if __name__ == "__main__":
    main()
