"""Extraction is copyable -- so make the copy USELESS. The reason model
extraction matters is the transfer attack: steal a clone, craft evasions
offline against it, use them on the real model. EXTRACTION_DEFENSE_COMPLETE.md
established the clone can't be PREVENTED. This closes the loop from the other
end: if the deployed model is adversarially hardened (WS2), do evasions
crafted on a clone still transfer?

Setup: extract a surrogate from each target (uniform queries -> target hard
labels), craft minimal evasions against the surrogate ONLY (zero live queries
during crafting), then check transfer to the real target. Compare:
  - original (undefended) target   -> the 100% transfer baseline
  - iteratively-hardened target    -> does hardening break transfer?

If transfer collapses against the hardened model, extraction is defanged: the
adversary can still copy the model, but the copy no longer yields working
attacks -- which is the only reason they wanted it.

Reuses adversarial_robustness.py's attack primitives and the extraction
surrogate recipe from model_extraction_attack.py, unmodified.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from radarsense import config as C
from radarsense.adversarial_robustness import (ATTACK_FEATURES, apply_perturbation, reorder_proba)
from radarsense.model_extraction_attack import sample_random_queries
from radarsense.optimize import threshold_predict, tune_threshold

EXTRACT_BUDGET = 2000
N_TRANSFER = 100
N_RESTARTS, N_ITERS = 8, 150


def extract_surrogate(target, threshold, rng):
    q = sample_random_queries(EXTRACT_BUDGET, rng)
    labels = threshold_predict(reorder_proba(target, q[C.FEATURE_COLUMNS].values), threshold)
    sur = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    sur.fit(q[C.FEATURE_COLUMNS].values, labels)
    return sur


def craft_and_transfer(surrogate, target, sur_thr, tgt_thr, drones, feat_std_arr, drone_idx):
    proba = reorder_proba(target, drones[C.FEATURE_COLUMNS])
    correct = drones[threshold_predict(proba, tgt_thr) == "drone"].reset_index(drop=True)
    sample = correct.sample(min(N_TRANSFER, len(correct)), random_state=C.SEED)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)
    rng = np.random.default_rng(C.SEED)

    def sur_p(x):
        return reorder_proba(surrogate, x[None, :].astype(np.float32))[0, drone_idx]

    def tgt_p(x):
        return reorder_proba(target, x[None, :].astype(np.float32))[0, drone_idx]

    n_sur_evaded, n_transferred = 0, 0
    for i in range(len(sample)):
        x0 = X0[i]
        delta = np.zeros(len(ATTACK_FEATURES))
        step = feat_std_arr * 0.5
        best = sur_p(apply_perturbation(x0, delta))
        crafted = None
        for _ in range(N_RESTARTS):
            for _ in range(N_ITERS):
                j = rng.integers(len(ATTACK_FEATURES))
                trial = delta.copy()
                trial[j] += rng.normal(0, step[j])
                p = sur_p(apply_perturbation(x0, trial))
                if p < best:
                    delta, best = trial, p
                    if p < sur_thr:
                        crafted = apply_perturbation(x0, trial)
            step *= 0.99
        if crafted is not None:
            n_sur_evaded += 1
            if tgt_p(crafted) < tgt_thr:
                n_transferred += 1
    return n_sur_evaded, n_transferred, len(sample)


def main():
    original = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    t_orig = meta["drone_threshold_deployed"]
    hardened = joblib.load(C.MODELS_DIR / "final_model_iteratively_hardened.joblib")
    drone_idx = C.CLASSES.index("drone")

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    feat_std_arr = np.array([float(train[f].std()) for f in ATTACK_FEATURES])
    drones = test[test[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    t_hard, _, _ = tune_threshold(reorder_proba(hardened, val[C.FEATURE_COLUMNS]), val[C.LABEL_COLUMN])

    rng = np.random.default_rng(C.SEED)
    results = {}
    for name, target, thr in [("original", original, t_orig), ("iteratively_hardened", hardened, t_hard)]:
        print(f"--- extracting surrogate from {name} target, crafting offline, checking transfer ---")
        sur = extract_surrogate(target, thr, rng)
        n_ev, n_tr, n = craft_and_transfer(sur, target, thr, thr, drones, feat_std_arr, drone_idx)
        transfer_rate = n_tr / max(n_ev, 1)
        results[name] = {"n_attacked": n, "n_surrogate_evaded": n_ev,
                         "n_transferred": n_tr, "transfer_rate": transfer_rate,
                         "overall_success": n_tr / n}
        print(f"  surrogate evaded {n_ev}/{n}, transferred to real target {n_tr}/{n_ev} "
              f"({transfer_rate:.1%}), overall {n_tr}/{n} ({n_tr/n:.1%})")

    print(f"\n=== extraction defanged? ===")
    print(f"  transfer rate: original={results['original']['transfer_rate']:.1%}  "
          f"hardened={results['iteratively_hardened']['transfer_rate']:.1%}")
    (C.REPORTS_DIR / "extraction_defanged_results.json").write_text(json.dumps(results, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'extraction_defanged_results.json'}")


if __name__ == "__main__":
    main()
