"""Two critiques answered at once:

  (1) "your evasion attack is unconstrained in feature space -- is it even
      PHYSICALLY achievable?" The reports already disclosed this as a limit;
      this replaces the disclosure with a measurement. Each attack feature is
      given a physical-achievability COST (how hard is it for a real operator
      to move that feature by one std): reducing RCS needs radar-absorbent
      material (expensive), shrouding rotors moves micro-Doppler (moderate),
      changing speed/acceleration is cheap flight behavior. An evasion only
      counts as achievable if its total weighted cost is under a budget. The
      "achievable evasion rate" under a realistic budget is far below the
      unconstrained 71-100%.

  (2) "single seed." The attack is run across multiple seeds and reported as
      mean +/- std, so the headline numbers carry a confidence interval.

Reuses adversarial_robustness.py's exact attack primitives, unmodified.
"""

import json

import joblib
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.adversarial_robustness import ATTACK_FEATURES, attack_one, reorder_proba
from radarsense.optimize import threshold_predict

PHYSICAL_COST = {
    "rcs_estimate": 5.0,
    "micro_doppler_bandwidth": 2.0,
    "micro_doppler_jitter": 2.0,
    "doppler_velocity": 1.0,
    "velocity_variance": 1.0,
    "acceleration": 1.0,
}
COST_BUDGETS = [1.0, 2.0, 4.0, 8.0, np.inf]
SEEDS = [42, 7, 123]
N_SAMPLES = 120


def physical_cost(delta, feat_std_arr):
    cost_w = np.array([PHYSICAL_COST[f] for f in ATTACK_FEATURES])
    return float(np.sum(np.abs(delta) / np.maximum(feat_std_arr, 1e-9) * cost_w))


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    feat_std_arr = np.array([float(train[f].std()) for f in ATTACK_FEATURES])
    drones = test[test[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    proba = reorder_proba(model, drones[C.FEATURE_COLUMNS])
    correct = drones[threshold_predict(proba, threshold) == "drone"].reset_index(drop=True)

    per_seed_uncon, per_seed_budget = [], {b: [] for b in COST_BUDGETS}
    for seed in SEEDS:
        sample = correct.sample(min(N_SAMPLES, len(correct)), random_state=seed)
        X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)
        rng = np.random.default_rng(seed)
        costs = []
        for i in range(len(sample)):
            ok, _, delta = attack_one(model, feat_std_arr, X0[i], threshold, drone_idx, rng)
            costs.append(physical_cost(delta, feat_std_arr) if ok else np.inf)
        costs = np.array(costs)
        per_seed_uncon.append(float(np.isfinite(costs).mean()))
        for b in COST_BUDGETS:
            per_seed_budget[b].append(float((costs <= b).mean()))
        print(f"  seed {seed}: unconstrained evasion={np.isfinite(costs).mean():.1%}  "
              f"achievable@budget2={ (costs<=2.0).mean():.1%}  @budget4={(costs<=4.0).mean():.1%}")

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    print("\n=== evasion rate vs physical-achievability budget (mean +/- std over seeds) ===")
    print(f"  unconstrained (feature-space):  {np.mean(per_seed_uncon):.1%} +/- {np.std(per_seed_uncon):.1%}")
    for b in COST_BUDGETS:
        tag = "inf (=unconstrained)" if not np.isfinite(b) else f"{b:.0f}"
        print(f"  achievable @ cost budget {tag:>18}: "
              f"{np.mean(per_seed_budget[b]):.1%} +/- {np.std(per_seed_budget[b]):.1%}")

    out = {
        "physical_cost_weights": PHYSICAL_COST,
        "seeds": SEEDS,
        "unconstrained_evasion_rate": ms(per_seed_uncon),
        "achievable_evasion_rate_by_budget": {("inf" if not np.isfinite(b) else b): ms(v)
                                              for b, v in per_seed_budget.items()},
    }
    (C.REPORTS_DIR / "adversarial_physical_cost_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'adversarial_physical_cost_results.json'}")


if __name__ == "__main__":
    main()
