"""Does hardening the API's response actually blunt model extraction?
Tested BEFORE touching api/server.py -- measure first, then apply the
defense that's confirmed to work, matching the same discipline every
other finding in this project was held to.

Reuses model_extraction_attack.py's exact query-sampling and transfer-
attack methodology (imported, not duplicated), varying only what the
"API" returns to the attacker:

  baseline            full float probabilities (what MODEL_EXTRACTION_ATTACK.md measured)
  rounded_2dp         probabilities rounded to 2 decimals (e.g. 0.83)
  rounded_1dp         probabilities rounded to 1 decimal (e.g. 0.8)
  label_only          hard label only, no probability at all (attacker
                      must train the surrogate on 0/1 labels, the
                      standard harder "label-only" extraction setting)

For each, re-runs the FULL two-stage attack (extract a surrogate at the
same 5,000-query budget, then the transfer attack -- craft evasions
against the surrogate only, check if they still fool the real target)
so the comparison is apples-to-apples with the original finding.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from radarsense import config as C
from radarsense.adversarial_robustness import ATTACK_FEATURES, apply_perturbation
from radarsense.model_extraction_attack import sample_random_queries
from radarsense.optimize import threshold_predict

N_QUERIES = 5000
N_TRANSFER_ATTACK_SAMPLES = 100
N_RESTARTS_TRANSFER = 8
N_ITERS_TRANSFER = 150


def reorder_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, [list(model.classes_).index(c) for c in C.CLASSES]]


def hardened_response(model, threshold, sample_dict, mode):
    """Simulates what a hardened /predict endpoint would return, at the
    SAME decision logic as api.predict.predict_one, just with a coarsened
    or withheld probability."""
    x = np.array([[float(sample_dict[f]) for f in C.FEATURE_COLUMNS]], dtype=np.float32)
    proba = reorder_proba(model, x)[0]
    drone_idx = C.CLASSES.index("drone")
    label = "drone" if proba[drone_idx] >= threshold else \
        C.CLASSES[[i for i in range(len(C.CLASSES)) if i != drone_idx][
            int(np.argmax(proba[[i for i in range(len(C.CLASSES)) if i != drone_idx]]))]]

    if mode == "baseline":
        p_drone = float(proba[drone_idx])
    elif mode == "rounded_2dp":
        p_drone = round(float(proba[drone_idx]), 2)
    elif mode == "rounded_1dp":
        p_drone = round(float(proba[drone_idx]), 1)
    elif mode == "label_only":
        p_drone = None
    else:
        raise ValueError(mode)
    return {"label": label, "p_drone": p_drone}


def extract_surrogate(model, threshold, mode, rng):
    queries = sample_random_queries(N_QUERIES, rng)
    labels, p_drones = [], []
    for _, row in queries.iterrows():
        resp = hardened_response(model, threshold, row.to_dict(), mode)
        labels.append(resp["label"])
        p_drones.append(resp["p_drone"])

    surrogate = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                   random_seed=C.SEED, verbose=0, allow_writing_files=False)
    surrogate.fit(queries, labels)
    return surrogate, queries, labels, p_drones


def transfer_attack(model, surrogate, threshold, drone_idx, test, rng):
    proba_real = reorder_proba(model, test[C.FEATURE_COLUMNS])
    pred_real = threshold_predict(proba_real, threshold)
    correctly_classified = test[pred_real == "drone"].reset_index(drop=True)
    sample = correctly_classified.sample(min(N_TRANSFER_ATTACK_SAMPLES, len(correctly_classified)),
                                         random_state=C.SEED)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)

    def surrogate_p_drone(x):
        return reorder_proba(surrogate, x[None, :].astype(np.float32))[0, drone_idx]

    def real_p_drone(x):
        return reorder_proba(model, x[None, :].astype(np.float32))[0, drone_idx]

    n_surrogate_evaded, n_transferred = 0, 0
    for i in range(len(sample)):
        x0 = X0[i]
        delta = np.zeros(len(ATTACK_FEATURES))
        step = np.array([50, 20, 200, 0.5, 100, 10]) * 0.1
        best_p_sur = surrogate_p_drone(apply_perturbation(x0, delta))
        crafted = None
        for restart in range(N_RESTARTS_TRANSFER):
            for it in range(N_ITERS_TRANSFER):
                j = rng.integers(len(ATTACK_FEATURES))
                trial = delta.copy()
                trial[j] = delta[j] + rng.normal(0, step[j])
                x_trial = apply_perturbation(x0, trial)
                p_sur = surrogate_p_drone(x_trial)
                if p_sur < best_p_sur:
                    delta = trial
                    best_p_sur = p_sur
                    if p_sur < threshold:
                        crafted = x_trial
            step = step * 0.99
        if crafted is not None:
            n_surrogate_evaded += 1
            if real_p_drone(crafted) < threshold:
                n_transferred += 1
    return n_surrogate_evaded, n_transferred, len(sample)


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    proba_target_test = reorder_proba(model, test[C.FEATURE_COLUMNS])
    pred_target_test = threshold_predict(proba_target_test, threshold)

    results = {}
    for mode in ["baseline", "rounded_2dp", "rounded_1dp", "label_only"]:
        print(f"\n=== mode: {mode} ===")
        rng = np.random.default_rng(C.SEED)
        surrogate, queries, labels, p_drones = extract_surrogate(model, threshold, mode, rng)

        proba_sur_test = reorder_proba(surrogate, test[C.FEATURE_COLUMNS])
        pred_sur_test = threshold_predict(proba_sur_test, threshold)
        agreement = float((pred_sur_test == pred_target_test).mean())
        drone_agreement = float(((pred_sur_test == "drone") == (pred_target_test == "drone")).mean())
        print(f"  extraction fidelity: overall={agreement:.4f} drone_gate={drone_agreement:.4f}")

        rng2 = np.random.default_rng(C.SEED)
        n_evaded, n_transferred, n_attacked = transfer_attack(model, surrogate, threshold, drone_idx, test, rng2)
        transfer_rate = n_transferred / n_attacked
        print(f"  transfer attack: evaded={n_evaded}/{n_attacked} transferred={n_transferred}/{n_attacked} "
              f"({transfer_rate:.1%})")

        results[mode] = {
            "extraction_overall_agreement": agreement,
            "extraction_drone_gate_agreement": drone_agreement,
            "n_surrogate_evaded": n_evaded,
            "n_transferred_to_real_target": n_transferred,
            "n_attacked": n_attacked,
            "transfer_rate": transfer_rate,
        }

    print("\n--- summary: does hardening the response actually help? ---")
    for mode, r in results.items():
        print(f"  {mode:14s} drone_gate_fidelity={r['extraction_drone_gate_agreement']:.4f}  "
              f"transfer_rate={r['transfer_rate']:.1%}")

    (C.REPORTS_DIR / "extraction_defense_test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'extraction_defense_test_results.json'}")


if __name__ == "__main__":
    main()
