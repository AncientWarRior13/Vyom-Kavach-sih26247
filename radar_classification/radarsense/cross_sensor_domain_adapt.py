"""WS4 -- the COMPLETE cross-sensor transfer treatment.

CROSS_SENSOR_TRANSFER.md found zero-shot transfer degrades substantially
(mean cross-sensor accuracy ~49% vs 85-99% in-distribution).
CROSS_SENSOR_CALIBRATION_ABLATION.md showed dropping the two
inconsistently-calibrated features recovers ~11 points but throws signal
away. CROSS_SENSOR_FEW_SHOT.md showed 50 labeled target examples recover
~21 points. Each was partial. This closes the gap as far as the data
allows, along four fronts, all measured on the same 12 ordered sensor pairs
and the same drone-vs-anomaly task:

  1. UNSUPERVISED domain adaptation, ZERO target labels. Two alignment
     transforms that use only the target's UNLABELED feature distribution
     (a deployment can collect unlabeled detections for free):
       - moment_matching: per-feature affine map of source into the
         target's mean/variance (a LEARNED calibration transform -- recovers
         the miscalibrated-feature signal instead of DROPPING it like the
         ablation did; a diagonal special case of CORAL).
       - CORAL: full second-order (covariance) alignment of source features
         to the target, the standard correlation-alignment DA method.

  2. LEAVE-ONE-SENSOR-OUT pooled training: train on the OTHER three sensors
     pooled, test on the held-out one -- does seeing multiple sensors'
     physics generalize to an unseen sensor better than any single source?

  3. EXTENDED few-shot: CROSS_SENSOR_FEW_SHOT.md capped at 50 shots with the
     curve still rising; extend to 100/200/500 and find where it plateaus.

  4. COMBINED: CORAL alignment PLUS a small labeled few-shot set -- do the
     zero-label and few-label recoveries stack?

Everything reuses cross_sensor_transfer.py's exact data loading, task
definition, and metrics (imported unmodified) so every number here is
directly comparable to the zero-shot baseline already reported.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.cross_sensor_transfer import SENSORS, load_binary
from radarsense.raddar_cnn import binary_metrics

FEATURES = C.FEATURE_COLUMNS
RIDGE = 1e-3
SHOT_BUDGETS = [0, 50, 100, 200, 500]


def _cov_sqrt(X, ridge=RIDGE):
    """Symmetric PSD square root and inverse square root of cov(X)+ridge*I."""
    C_ = np.cov(X, rowvar=False) + ridge * np.eye(X.shape[1])
    w, V = np.linalg.eigh(C_)
    w = np.clip(w, ridge, None)
    sqrt = V @ np.diag(np.sqrt(w)) @ V.T
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return sqrt, inv_sqrt


def coral_align(Xs, Xt):
    """Recolor SOURCE features to match TARGET first+second moments.
    Uses target features only (labels never touched) -> unsupervised."""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    _, inv_sqrt_s = _cov_sqrt(Xs)
    sqrt_t, _ = _cov_sqrt(Xt)
    whitened = (Xs - mu_s) @ inv_sqrt_s
    return whitened @ sqrt_t + mu_t


def moment_match(Xs, Xt):
    """Per-feature affine map of source into target mean/std (diagonal CORAL)."""
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-8
    mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-8
    return (Xs - mu_s) / sd_s * sd_t + mu_t


def fit_model(X, y):
    sw = compute_sample_weight("balanced", y)
    clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0, allow_writing_files=False)
    clf.fit(X, y, sample_weight=sw)
    return clf


def acc_pr(clf, X, y):
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = binary_metrics(y, pred, proba)
    return m["accuracy"], m["pr_auc_drone"], m["recall_drone"]


def main():
    data = {name: load_binary(path) for name, path in SENSORS.items()}
    splits = {}
    for name, df in data.items():
        tr, te = train_test_split(df, test_size=0.3, stratify=df.y, random_state=C.SEED)
        splits[name] = (tr.reset_index(drop=True), te.reset_index(drop=True))

    names = list(SENSORS)

    print("=== zero-label methods (baseline / moment-matching / CORAL), all 12 pairs ===")
    per_pair = []
    for src in names:
        src_tr = splits[src][0]
        Xs, ys = src_tr[FEATURES].values, src_tr.y.values
        base_model = fit_model(Xs, ys)
        for tgt in names:
            if tgt == src:
                continue
            tgt_tr, tgt_te = splits[tgt]
            Xt_unlab = tgt_tr[FEATURES].values
            Xe, ye = tgt_te[FEATURES].values, tgt_te.y.values

            base_acc, base_pr, base_rec = acc_pr(base_model, Xe, ye)

            Xs_mm = moment_match(Xs, Xt_unlab)
            mm_model = fit_model(Xs_mm, ys)
            mm_acc, mm_pr, mm_rec = acc_pr(mm_model, Xe, ye)

            Xs_co = coral_align(Xs, Xt_unlab)
            co_model = fit_model(Xs_co, ys)
            co_acc, co_pr, co_rec = acc_pr(co_model, Xe, ye)

            per_pair.append({"src": src, "tgt": tgt,
                             "baseline": {"acc": base_acc, "pr_auc": base_pr, "recall": base_rec},
                             "moment_match": {"acc": mm_acc, "pr_auc": mm_pr, "recall": mm_rec},
                             "coral": {"acc": co_acc, "pr_auc": co_pr, "recall": co_rec}})

    def mean_over_pairs(method, field):
        return float(np.mean([p[method][field] for p in per_pair]))

    zero_label = {m: {f: mean_over_pairs(m, f) for f in ("acc", "pr_auc", "recall")}
                  for m in ("baseline", "moment_match", "coral")}
    for m in ("baseline", "moment_match", "coral"):
        print(f"  {m:14s} mean cross-sensor acc={zero_label[m]['acc']:.4f}  "
              f"pr_auc={zero_label[m]['pr_auc']:.4f}  recall={zero_label[m]['recall']:.4f}")

    print("\n=== leave-one-sensor-out: train on the other 3 pooled, test on held-out ===")
    loso = {}
    for held in names:
        pool = pd.concat([splits[s][0] for s in names if s != held], ignore_index=True)
        model = fit_model(pool[FEATURES].values, pool.y.values)
        Xe, ye = splits[held][1][FEATURES].values, splits[held][1].y.values
        acc, pr, rec = acc_pr(model, Xe, ye)
        loso[held] = {"acc": acc, "pr_auc": pr, "recall": rec}
        print(f"  held-out {held:24s} acc={acc:.4f} pr_auc={pr:.4f} recall={rec:.4f}")
    loso_mean = {f: float(np.mean([loso[h][f] for h in names])) for f in ("acc", "pr_auc", "recall")}
    print(f"  MEAN leave-one-out       acc={loso_mean['acc']:.4f} pr_auc={loso_mean['pr_auc']:.4f} "
          f"recall={loso_mean['recall']:.4f}")

    print("\n=== extended few-shot (full retrain) and CORAL+few-shot combined ===")
    rng = np.random.default_rng(C.SEED)
    fewshot = {b: [] for b in SHOT_BUDGETS}
    combined = {b: [] for b in SHOT_BUDGETS if b > 0}
    for src in names:
        src_tr = splits[src][0]
        Xs, ys = src_tr[FEATURES].values, src_tr.y.values
        for tgt in names:
            if tgt == src:
                continue
            tgt_tr, tgt_te = splits[tgt]
            Xt_unlab = tgt_tr[FEATURES].values
            Xe, ye = tgt_te[FEATURES].values, tgt_te.y.values
            Xs_co = coral_align(Xs, Xt_unlab)
            for b in SHOT_BUDGETS:
                if b == 0:
                    acc, _, _ = acc_pr(fit_model(Xs, ys), Xe, ye)
                    fewshot[b].append(acc)
                    continue
                npc = max(1, b // 2)
                shots = pd.concat([
                    tgt_tr[tgt_tr.y == 0].sample(min(npc, int((tgt_tr.y == 0).sum())), random_state=C.SEED),
                    tgt_tr[tgt_tr.y == 1].sample(min(npc, int((tgt_tr.y == 1).sum())), random_state=C.SEED),
                ])
                Xsh, ysh = shots[FEATURES].values, shots.y.values
                fs_model = fit_model(np.vstack([Xs, Xsh]), np.concatenate([ys, ysh]))
                fewshot[b].append(acc_pr(fs_model, Xe, ye)[0])
                cb_model = fit_model(np.vstack([Xs_co, Xsh]), np.concatenate([ys, ysh]))
                combined[b].append(acc_pr(cb_model, Xe, ye)[0])
        print(f"  {src}: done")

    fewshot_mean = {b: float(np.mean(v)) for b, v in fewshot.items()}
    combined_mean = {b: float(np.mean(v)) for b, v in combined.items()}
    print("\n  shots |  full-retrain  |  CORAL+few-shot")
    for b in SHOT_BUDGETS:
        cb = f"{combined_mean[b]:.4f}" if b in combined_mean else "   --  "
        print(f"  {b:5d} |    {fewshot_mean[b]:.4f}    |    {cb}")

    out = {
        "task": "drone vs anomaly, 11-feature schema, 12 ordered sensor pairs",
        "zero_label_domain_adaptation": zero_label,
        "leave_one_sensor_out": {"per_sensor": loso, "mean": loso_mean},
        "extended_few_shot_full_retrain_mean_acc": fewshot_mean,
        "coral_plus_few_shot_mean_acc": combined_mean,
        "per_pair_zero_label": per_pair,
    }
    (C.REPORTS_DIR / "cross_sensor_domain_adapt_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'cross_sensor_domain_adapt_results.json'}")


if __name__ == "__main__":
    main()
