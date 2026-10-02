"""Adversarial training: does retraining on the evasions
adversarial_robustness.py actually found reduce the 100% evasion rate?
The standard defense against a demonstrated evasion attack -- augment
training data with adversarial examples, correctly labeled, so the model
learns a decision boundary that isn't fooled by them.

Same CatBoost config as the deployed "class_weight" winning variant
(train_baseline.py's build_models()['catboost'] + optimize.py's balanced
sample weighting) so the comparison isolates the effect of the augmented
data, not a training-recipe change.

Procedure:
  1. Run the SAME black-box search as adversarial_robustness.py against
     the CURRENT model on a sample of TRAINING drones (not test -- the
     test set stays held out for honest re-evaluation), collecting real
     found evasion perturbations.
  2. Amplify: apply each found perturbation direction to several
     additional drone training rows (still real drones, still correctly
     labeled "drone" after perturbation -- the physical target didn't
     change, only the measured feature vector did) for broader coverage
     without needing full per-sample optimization at training-set scale.
  3. Retrain on original + augmented-adversarial training data.
  4. Sanity check: does normal (unperturbed) test performance hold up?
  5. Re-run the EXACT SAME attack (same 200 held-out test drones, same
     search config) against the retrained model -- the real question.
"""

import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.adversarial_robustness import (ATTACK_FEATURES, BOUNDS, apply_perturbation,
                                                attack_one, reorder_proba)
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

N_SEED_ATTACKS = 300
N_AUGMENT_PER_DELTA = 3


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")

    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    val = pd.read_parquet(C.DATA_DIR / "radar_val.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    feat_std = {f: float(train[f].std()) for f in ATTACK_FEATURES}
    feat_std_arr = np.array([feat_std[f] for f in ATTACK_FEATURES])

    print("--- stage 1: harvesting real evasion perturbations from TRAINING drones ---")
    train_drones = train[train[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    proba = reorder_proba(model, train_drones[C.FEATURE_COLUMNS])
    pred = threshold_predict(proba, threshold)
    correctly_classified = train_drones[pred == "drone"].reset_index(drop=True)
    sample = correctly_classified.sample(min(N_SEED_ATTACKS, len(correctly_classified)), random_state=C.SEED)
    X0 = sample[C.FEATURE_COLUMNS].values.astype(np.float64)

    rng = np.random.default_rng(C.SEED)
    found_deltas = []
    for i in range(len(sample)):
        success, norm, delta = attack_one(model, feat_std_arr, X0[i], threshold, drone_idx, rng)
        if success:
            found_deltas.append(delta)
        if (i + 1) % 100 == 0:
            print(f"  attacked {i+1}/{len(sample)}, {len(found_deltas)} evasions found so far")
    print(f"harvested {len(found_deltas)} real evasion perturbation directions")

    print("\n--- stage 2: augmenting the training set with adversarial examples ---")
    all_train_drones = train[train[C.LABEL_COLUMN] == "drone"]
    aug_rows = []
    for delta in found_deltas:
        targets = all_train_drones.sample(N_AUGMENT_PER_DELTA, random_state=int(rng.integers(1e6)))
        for _, row in targets.iterrows():
            x0 = row[C.FEATURE_COLUMNS].values.astype(np.float64)
            x_adv = apply_perturbation(x0, delta)
            aug_row = dict(zip(C.FEATURE_COLUMNS, x_adv))
            aug_row[C.LABEL_COLUMN] = "drone"
            aug_rows.append(aug_row)
    aug_df = pd.DataFrame(aug_rows)
    print(f"generated {len(aug_df)} adversarially-augmented drone training rows")

    train_augmented = pd.concat([train, aug_df], ignore_index=True)
    print(f"augmented train set: {len(train)} -> {len(train_augmented)} rows "
          f"({train_augmented[C.LABEL_COLUMN].value_counts().to_dict()})")

    print("\n--- stage 3: retraining (same CatBoost config as the deployed model) ---")
    sw = compute_sample_weight("balanced", train_augmented[C.LABEL_COLUMN])
    clf = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(train_augmented[C.FEATURE_COLUMNS], train_augmented[C.LABEL_COLUMN], sample_weight=sw)

    proba_val = reorder_proba(clf, val[C.FEATURE_COLUMNS])
    t_fbeta, t_cost, _ = tune_threshold(proba_val, val[C.LABEL_COLUMN])
    print(f"tuned drone threshold on retrained model: fbeta2={t_fbeta:.3f} cost={t_cost:.3f}")

    print("\n--- stage 4: sanity check -- normal (unperturbed) test performance ---")
    proba_test_orig = reorder_proba(model, test[C.FEATURE_COLUMNS])
    pred_test_orig = threshold_predict(proba_test_orig, threshold)
    m_orig = full_metrics(test[C.LABEL_COLUMN], pred_test_orig, proba_test_orig)

    proba_test_new = reorder_proba(clf, test[C.FEATURE_COLUMNS])
    pred_test_new = threshold_predict(proba_test_new, t_fbeta)
    m_new = full_metrics(test[C.LABEL_COLUMN], pred_test_new, proba_test_new)

    print(f"  original model : accuracy={m_orig['accuracy']:.4f} drone_recall={m_orig['recall_drone']:.4f}")
    print(f"  retrained model: accuracy={m_new['accuracy']:.4f} drone_recall={m_new['recall_drone']:.4f}")

    print("\n--- stage 5: THE question -- re-run the SAME attack against the retrained model ---")
    test_drones = test[test[C.LABEL_COLUMN] == "drone"].reset_index(drop=True)
    proba_new_drones = reorder_proba(clf, test_drones[C.FEATURE_COLUMNS])
    pred_new_drones = threshold_predict(proba_new_drones, t_fbeta)
    correctly_new = test_drones[pred_new_drones == "drone"].reset_index(drop=True)
    print(f"real drones correctly classified by RETRAINED model: {len(correctly_new)}")

    sample2 = correctly_new.sample(min(200, len(correctly_new)), random_state=C.SEED)
    X0_2 = sample2[C.FEATURE_COLUMNS].values.astype(np.float64)
    rng2 = np.random.default_rng(C.SEED)
    results = []
    for i in range(len(sample2)):
        success, norm, delta = attack_one(clf, feat_std_arr, X0_2[i], t_fbeta, drone_idx, rng2)
        results.append({"success": success, "min_perturbation_norm": norm})
        if (i + 1) % 50 == 0:
            print(f"  re-attacked {i+1}/{len(sample2)}...")

    n_success = sum(r["success"] for r in results)
    new_evasion_rate = n_success / len(results)
    successful = [r for r in results if r["success"]]
    new_median_norm = float(np.median([r["min_perturbation_norm"] for r in successful])) if successful else None

    print(f"\n=== BEFORE (original model): 100% evasion (179/179), median norm 1.04 std-units ===")
    print(f"=== AFTER (adversarially-trained model): {new_evasion_rate:.1%} evasion "
          f"({n_success}/{len(results)}), median norm "
          f"{new_median_norm if new_median_norm is None else round(new_median_norm, 3)} std-units ===")

    joblib.dump(clf, C.MODELS_DIR / "final_model_adversarially_trained.joblib")
    out = {
        "n_seed_attacks_for_harvesting": N_SEED_ATTACKS,
        "n_evasions_harvested": len(found_deltas),
        "n_augmented_rows": len(aug_df),
        "retrained_threshold_fbeta2": t_fbeta,
        "sanity_check_normal_test_performance": {
            "original_model": {"accuracy": m_orig["accuracy"], "recall_drone": m_orig["recall_drone"]},
            "retrained_model": {"accuracy": m_new["accuracy"], "recall_drone": m_new["recall_drone"]},
        },
        "re_attack_results": {
            "n_attacked": len(results),
            "success_rate_before": 1.0,
            "success_rate_after": new_evasion_rate,
            "median_perturbation_norm_after": new_median_norm,
        },
        "per_sample_results": results,
    }
    (C.REPORTS_DIR / "adversarial_training_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'final_model_adversarially_trained.joblib'}")
    print(f"results -> {C.REPORTS_DIR / 'adversarial_training_results.json'}")


if __name__ == "__main__":
    main()
