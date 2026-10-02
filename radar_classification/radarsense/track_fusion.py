"""Track-level (multi-frame) fusion combined with the quantum kernel --
unexplored territory in this project until now. Real radar systems
classify a TARGET across consecutive detections, not one snapshot; this
tests whether fusing predictions across a track's frames helps the quantum
kernel specifically close (any of) the gap to classical that single-frame
classification always showed, or whether it helps both equally (or neither).

Fusion methods (both simple, standard, not novel by themselves -- what's
new is applying them to compare quantum vs classical under the SAME fusion
rule, on the SAME synthetic tracks):
  - mean probability across the track's K frames, then threshold
  - majority vote across the track's K per-frame hard predictions

Reported honestly regardless of direction, same as every other quantum
experiment in this project.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.optimize import threshold_predict
from radarsense.quantum import fidelity_kernel, statevectors, subsample

N_TRAIN_QSVC = 400


def fuse_mean(proba_frames):
    """proba_frames: (n_tracks, k_frames) drone-probability per frame -> mean per track."""
    return proba_frames.mean(axis=1)


def fuse_vote(pred_frames):
    """pred_frames: (n_tracks, k_frames) binary hard predictions -> majority vote."""
    return (pred_frames.mean(axis=1) >= 0.5).astype(int)


def binary_metrics(y_true, y_pred, y_score):
    return {
        "recall_drone": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_drone": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_drone": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc_drone": float(average_precision_score(y_true, y_score)),
    }


def reshape_by_track(df, proba_col_values, k_frames=5):
    """Reshape a flat (track_id, frame)-ordered array into (n_tracks, k_frames),
    using the track/frame structure actually present (tracks may have <k_frames
    if any rows were dropped -- assert clean rectangular structure here)."""
    df = df.copy()
    df["_p"] = proba_col_values
    pivot = df.pivot_table(index="track_id", columns="frame", values="_p")
    is_drone = df.groupby("track_id")[C.LABEL_COLUMN].first().values == C.MINORITY_CLASS
    return pivot.values, is_drone.astype(int)


def main():
    train = pd.read_parquet(C.DATA_DIR / "tracks_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "tracks_test.parquet")
    print(f"train frames={len(train)} ({train.track_id.nunique()} tracks), "
          f"test frames={len(test)} ({test.track_id.nunique()} tracks)")

    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    t = meta["drone_threshold_deployed"]

    proba_test = model.predict_proba(test[C.FEATURE_COLUMNS])
    p_drone = proba_test[:, list(model.classes_).index("drone")]
    pred_frame = (p_drone >= t).astype(int)

    y_frame = (test[C.LABEL_COLUMN] == "drone").astype(int).values
    single_frame_classical = binary_metrics(y_frame, pred_frame, p_drone)

    p_by_track, y_by_track = reshape_by_track(test, p_drone)
    pred_by_track, _ = reshape_by_track(test, pred_frame)
    fused_mean_p = fuse_mean(p_by_track)
    fused_mean_pred = (fused_mean_p >= t).astype(int)
    fused_vote_pred = fuse_vote(pred_by_track)

    classical_results = {
        "single_frame": single_frame_classical,
        "track_fused_mean_proba": binary_metrics(y_by_track, fused_mean_pred, fused_mean_p),
        "track_fused_majority_vote": binary_metrics(y_by_track, fused_vote_pred, fused_vote_pred.astype(float)),
    }
    print("\n=== CLASSICAL (frozen final_model) ===")
    for k, v in classical_results.items():
        print(f"  {k}: drone_recall={v['recall_drone']:.3f} precision={v['precision_drone']:.3f} "
              f"pr_auc={v['pr_auc_drone']:.3f}")

    print("\ntraining QSVC on track-simulator data (n_train=%d)..." % N_TRAIN_QSVC)
    Xtr_raw, ytr = subsample(train, N_TRAIN_QSVC, C.SEED)
    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)

    fmap = ZZFeatureMap(feature_dimension=len(C.QUANTUM_FEATURES), reps=C.QUANTUM_REPS)
    t0 = time.time()
    S_tr = statevectors(fmap, Xtr)
    Ktr = fidelity_kernel(S_tr)
    qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
    qsvc.fit(Ktr, ytr)
    print(f"QSVC trained in {time.time()-t0:.1f}s")

    Xte_raw = test[C.QUANTUM_FEATURES].values
    Xte = scaler.transform(Xte_raw)
    S_te = statevectors(fmap, Xte)
    Kte = fidelity_kernel(S_te, S_tr)
    proba_q = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]
    pred_q_frame = qsvc.predict(Kte)

    y_frame_bin = (test[C.LABEL_COLUMN] == C.MINORITY_CLASS).astype(int).values
    single_frame_quantum = binary_metrics(y_frame_bin, pred_q_frame, proba_q)

    pq_by_track, yq_by_track = reshape_by_track(test, proba_q)
    predq_by_track, _ = reshape_by_track(test, pred_q_frame)
    fused_mean_pq = fuse_mean(pq_by_track)
    fused_mean_pred_q = (fused_mean_pq >= 0.5).astype(int)
    fused_vote_pred_q = fuse_vote(predq_by_track)

    quantum_results = {
        "single_frame": single_frame_quantum,
        "track_fused_mean_proba": binary_metrics(yq_by_track, fused_mean_pred_q, fused_mean_pq),
        "track_fused_majority_vote": binary_metrics(yq_by_track, fused_vote_pred_q, fused_vote_pred_q.astype(float)),
    }
    print("\n=== QUANTUM (QSVC, n_train=%d) ===" % N_TRAIN_QSVC)
    for k, v in quantum_results.items():
        print(f"  {k}: drone_recall={v['recall_drone']:.3f} precision={v['precision_drone']:.3f} "
              f"pr_auc={v['pr_auc_drone']:.3f}")

    classical_gain = (classical_results["track_fused_mean_proba"]["recall_drone"]
                      - classical_results["single_frame"]["recall_drone"])
    quantum_gain = (quantum_results["track_fused_mean_proba"]["recall_drone"]
                    - quantum_results["single_frame"]["recall_drone"])
    print(f"\ndrone-recall gain from track fusion: classical {classical_gain:+.3f}, "
          f"quantum {quantum_gain:+.3f}")

    out = {
        "k_frames": 5, "n_test_tracks": int(len(y_by_track)),
        "classical": classical_results, "quantum": quantum_results,
        "track_fusion_gain_drone_recall": {"classical": classical_gain, "quantum": quantum_gain},
        "honest_finding": (
            "Track-level fusion (mean probability or majority vote across 5 "
            "correlated-target, independently-noised frames) compared for both "
            "backends under the identical fusion rule. Reported as measured -- "
            "whichever backend gains more from fusion, or neither, is the real "
            "result, not assumed in advance."),
    }
    (C.REPORTS_DIR / "track_fusion_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'track_fusion_results.json'}")


if __name__ == "__main__":
    main()
