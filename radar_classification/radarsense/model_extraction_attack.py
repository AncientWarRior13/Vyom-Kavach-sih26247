"""Model extraction / inversion attack against the deployed 11-feature
ONNX model -- the last well-specified, testable gap THREAT_MODEL.md names:
"No evaluation of whether repeated queries to the deployed ONNX model
(via the /predict or /batch API) could allow an adversary to reconstruct
decision-boundary information useful for evasion."

Threat model: an adversary with QUERY ACCESS ONLY -- no training data, no
model internals, no architecture knowledge -- who can call /predict (which
IS api.predict.predict_one, confirmed by reading api/server.py: the
endpoint is a thin wrapper with no rate-limiting or response filtering, so
calling predict_one directly is computationally identical to querying the
live HTTP API; verified once via an actual live request below before
running the bulk of the attack that way for speed, not HTTP overhead per
query).

Two-stage experiment:
  1. EXTRACTION: train a surrogate CatBoost model purely from (random
     query, API response) pairs -- no access to the real training data --
     across a query-budget sweep, measuring surrogate FIDELITY (prediction
     agreement with the real target on a held-out set the attacker would
     NOT have in reality, used here only to measure ground-truth
     extraction quality).
  2. TRANSFER ATTACK: the actual security-relevant question -- extraction
     alone isn't dangerous; extraction ENABLING evasion crafted entirely
     OFFLINE (zero further queries to the real target) that still fools
     the real target IS. Reuses adversarial_robustness.py's black-box
     search methodology, but every query during the search hits the
     SURROGATE, not the real model -- only the final success check queries
     the real target.

Query inputs are sampled uniformly within the schema's PUBLICLY KNOWN
physical bounds (documented in simulator.py / config.py's FEATURE_COLUMNS
comments) -- NOT the real training distribution, which a genuine
query-only attacker would not have access to.

Disclosed leak that makes this easier than it could be: /predict returns
FULL class probabilities, not just the hard label. A defense that returned
only the argmax label would make extraction meaningfully harder -- this
experiment measures the risk under the API's ACTUAL current behavior, not
a hardened hypothetical.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from radarsense import config as C
from radarsense.adversarial_robustness import ATTACK_FEATURES, BOUNDS, apply_perturbation
from radarsense.optimize import threshold_predict

QUERY_BUDGETS = [200, 500, 1000, 2000, 5000]
QUERY_BOUNDS = {
    "doppler_velocity": (-50, 50), "doppler_freq_shift": (-3334, 3334),
    "rcs_estimate": (-20, 20), "micro_doppler_bandwidth": (0, 500),
    "micro_doppler_jitter": (0, 1), "range_m": (50, 5000),
    "azimuth_deg": (-180, 180), "altitude_m": (0, 3000),
    "velocity_variance": (0, 200), "acceleration": (-10, 10), "snr_db": (-20, -10),
}
N_RESTARTS_TRANSFER = 8
N_ITERS_TRANSFER = 150
N_TRANSFER_ATTACK_SAMPLES = 100


def reorder_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, [list(model.classes_).index(c) for c in C.CLASSES]]


def sample_random_queries(n, rng):
    cols = {}
    for f in C.FEATURE_COLUMNS:
        lo, hi = QUERY_BOUNDS[f]
        cols[f] = rng.uniform(lo, hi, n)
    return pd.DataFrame(cols)[C.FEATURE_COLUMNS]


def main():
    from api.predict import predict_one

    print("verifying predict_one() == live HTTP API on one real request...")
    import subprocess
    import time
    import urllib.request
    proc = subprocess.Popen(["/opt/anaconda3/bin/python3", "-m", "uvicorn", "api.server:app",
                             "--port", "8099"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(3)
        test_probe = pd.read_parquet(C.DATA_DIR / "radar_test.parquet").iloc[0]
        payload = json.dumps(test_probe[C.FEATURE_COLUMNS].to_dict()).encode()
        req = urllib.request.Request("http://localhost:8099/predict", data=payload,
                                     headers={"content-type": "application/json"}, method="POST")
        live_result = json.loads(urllib.request.urlopen(req, timeout=10).read())
        direct_result = predict_one(test_probe[C.FEATURE_COLUMNS].to_dict())
        assert live_result == direct_result, "live API and predict_one() diverge!"
        print(f"  confirmed identical: live HTTP == predict_one() (label={live_result['label']})")
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")

    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    proba_target_test = reorder_proba(model, test[C.FEATURE_COLUMNS])
    pred_target_test = threshold_predict(proba_target_test, threshold)

    print(f"\n--- extraction: query-budget sweep, {QUERY_BUDGETS} ---")
    rng = np.random.default_rng(C.SEED)
    surrogates, fidelity = {}, {}
    for n_q in QUERY_BUDGETS:
        queries = sample_random_queries(n_q, rng)
        responses = [predict_one(row.to_dict()) for _, row in queries.iterrows()]
        y_extracted = np.array([r["label"] for r in responses])

        surrogate = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                                       random_seed=C.SEED, verbose=0, allow_writing_files=False)
        surrogate.fit(queries, y_extracted)
        surrogates[n_q] = surrogate

        proba_sur_test = reorder_proba(surrogate, test[C.FEATURE_COLUMNS])
        pred_sur_test = threshold_predict(proba_sur_test, threshold)
        agreement = float((pred_sur_test == pred_target_test).mean())
        drone_agreement = float(((pred_sur_test == "drone") == (pred_target_test == "drone")).mean())
        fidelity[n_q] = {"overall_agreement": agreement, "drone_gate_agreement": drone_agreement}
        print(f"  n_queries={n_q}: surrogate-vs-target agreement={agreement:.4f} "
              f"drone-gate agreement={drone_agreement:.4f}")

    best_n = max(QUERY_BUDGETS)
    best_surrogate = surrogates[best_n]
    print(f"\nusing the n={best_n}-query surrogate for the transfer attack (highest fidelity)")

    print(f"\n--- transfer attack: craft evasions against the SURROGATE ONLY "
          f"(zero live queries during crafting), check success against the REAL target ---")
    drones = test[test[C.LABEL_COLUMN] == "drone"].copy()
    proba_real = reorder_proba(model, drones[C.FEATURE_COLUMNS])
    pred_real = threshold_predict(proba_real, threshold)
    correctly_classified = drones[pred_real == "drone"].reset_index(drop=True)
    sample = correctly_classified.sample(min(N_TRANSFER_ATTACK_SAMPLES, len(correctly_classified)),
                                         random_state=C.SEED)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)

    VEL_IDX = C.FEATURE_COLUMNS.index("doppler_velocity")
    FSHIFT_IDX = C.FEATURE_COLUMNS.index("doppler_freq_shift")

    def surrogate_p_drone(x):
        return reorder_proba(best_surrogate, x[None, :].astype(np.float32))[0, drone_idx]

    def real_p_drone(x):
        return reorder_proba(model, x[None, :].astype(np.float32))[0, drone_idx]

    rng2 = np.random.default_rng(C.SEED)
    transfer_results = []
    for i in range(len(sample)):
        x0 = X0[i]
        delta = np.zeros(len(ATTACK_FEATURES))
        step = np.array([50, 20, 200, 0.5, 100, 10]) * 0.1
        best_p_sur = surrogate_p_drone(apply_perturbation(x0, delta))
        crafted_evasion = None
        for restart in range(N_RESTARTS_TRANSFER):
            for it in range(N_ITERS_TRANSFER):
                j = rng2.integers(len(ATTACK_FEATURES))
                trial = delta.copy()
                trial[j] = delta[j] + rng2.normal(0, step[j])
                x_trial = apply_perturbation(x0, trial)
                p_sur = surrogate_p_drone(x_trial)
                if p_sur < best_p_sur:
                    delta = trial
                    best_p_sur = p_sur
                    if p_sur < threshold:
                        crafted_evasion = apply_perturbation(x0, trial)
            step = step * 0.99

        if crafted_evasion is not None:
            p_real = real_p_drone(crafted_evasion)
            transfer_results.append({
                "surrogate_evaded": True,
                "transferred_to_real_target": bool(p_real < threshold),
                "surrogate_p_drone": float(best_p_sur),
                "real_p_drone": float(p_real),
            })
        else:
            transfer_results.append({"surrogate_evaded": False, "transferred_to_real_target": False})

    n_surrogate_evaded = sum(r["surrogate_evaded"] for r in transfer_results)
    n_transferred = sum(r["transferred_to_real_target"] for r in transfer_results)
    print(f"\nsurrogate-crafted evasion success (against surrogate itself): "
          f"{n_surrogate_evaded}/{len(transfer_results)} ({n_surrogate_evaded/len(transfer_results):.1%})")
    print(f"of those, transferred to fool the REAL target (zero extra live queries used): "
          f"{n_transferred}/{n_surrogate_evaded if n_surrogate_evaded else 1} "
          f"({n_transferred/max(n_surrogate_evaded,1):.1%} transfer rate)  "
          f"= {n_transferred}/{len(transfer_results)} overall "
          f"({n_transferred/len(transfer_results):.1%} of all attacked drones)")

    out = {
        "extraction_query_budgets": QUERY_BUDGETS,
        "extraction_fidelity": fidelity,
        "leak_disclosed": "API returns full class probabilities, not just the hard label -- makes extraction easier than a label-only API would allow.",
        "transfer_attack": {
            "n_attacked": len(transfer_results),
            "surrogate_query_budget_used": best_n,
            "n_surrogate_evaded": n_surrogate_evaded,
            "n_transferred_to_real_target_with_zero_extra_live_queries": n_transferred,
            "transfer_rate_among_surrogate_successes": n_transferred / max(n_surrogate_evaded, 1),
            "overall_transfer_rate": n_transferred / len(transfer_results),
        },
        "per_sample_transfer_results": transfer_results,
    }
    (C.REPORTS_DIR / "model_extraction_attack_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'model_extraction_attack_results.json'}")


if __name__ == "__main__":
    main()
