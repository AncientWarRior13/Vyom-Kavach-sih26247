"""Quantum-kernel accuracy at FIXED SNR levels on the 31-feature schema --
a fresh, controlled test built specifically to compare against a teammate's
independently-run quantum-kernel result at the same nominal SNR levels
(-10/-15/-20/-25 dB), same dataset, same evaluation protocol.

"Same SNR levels" here means a controlled test set generated with SNR HELD
FIXED per level, not a random draw from a continuous band that happens to
land near a target value -- this is what a clean per-level comparison
requires.

Same underlying physics as feat31_simulator.py's _class_waveforms (imported,
not duplicated) and the same feature extraction as feat31_features.py
(imported, not duplicated) -- this is not a new dataset, it's the same
31-feature schema's generator with SNR pinned instead of sampled.

Training set: subsampled directly from feat31_train_features.parquet, the
SAME data the deployed 31-feature CatBoost model was trained on -- not a
separately-generated pool. Class-balanced (100/class, not the dataset's
native 20:1 aircraft:drone ratio) for the SAME reason quantum.py's
DRONE_FRAC does this for the 11-feature schema, documented there: at native
proportions a 400-sample pool contains ~9 drones, too few for a QSVC to
learn any drone boundary at all -- confirmed empirically on a first pass of
this exact script (natural-proportion pool -> 0.0 drone recall at every
SNR level, a degenerate, uninformative result, not a real finding about
quantum-kernel capability). Class-balanced training is the established
project convention for kernel studies at this scale, not a new ad-hoc
choice.

Quantum feature subset (4, for a tractable 4-qubit ZZFeatureMap): the
model's own top-ranked real SHAP features for the drone class, from
artifacts/reports/feat31_shap_report.json, excluding snr_db itself (which
would make an SNR-stratified test circular) -- energy_spectral_bandwidth,
amplitude_mean, amplitude_spectral_bandwidth, amplitude_min. Same
selection-by-measured-importance methodology as the frozen 11-feature
schema's QUANTUM_FEATURES in config.py.

Circuit: ZZFeatureMap(4, reps=2), exact noiseless statevector fidelity
kernel -- identical circuit family to every other quantum-kernel experiment
in this project (quantum.py, raddar_cnn.py, diat_usat_cnn.py).

Reports BOTH:
  - overall 4-class accuracy (multiclass QSVC via SVC's native one-vs-one
    support for a precomputed kernel) -- the metric being compared against
  - drone-vs-rest recall/precision/F1 -- this project's own standing
    convention elsewhere, kept for internal consistency

5 seeds per SNR level (fresh train pool + fresh test set each seed), same
repeated-measurement discipline used everywhere else in this project
(quantum.py, raddar_cnn.py) -- a first single-seed pass showed a
non-monotonic swing across levels (-10dB lower than -20dB, which has no
physical explanation) that mean+-std across seeds shows is sampling noise
at this small n_test, not a real effect. Reporting mean+-std rather than
one lucky/unlucky draw.

Output: one CSV row per SNR level (mean +/- std over 5 seeds), for direct
handoff.
"""

import json

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.feat31_features import extract_features
from radarsense.feat31_simulator import SNR_RANGE_DB, _class_waveforms
from radarsense.quantum import fidelity_kernel, statevectors

SNR_LEVELS = [-10.0, -15.0, -20.0, -25.0]
QUANTUM_FEAT31 = ["energy_spectral_bandwidth", "amplitude_mean",
                  "amplitude_spectral_bandwidth", "amplitude_min"]
N_TRAIN_PER_CLASS = 100
N_TEST_PER_CLASS = 80
QUANTUM_REPS = 2
N_SEEDS = 5


def generate_fixed_snr_test(snr_level, n_per_class, seed):
    """Same per-class waveform physics as feat31_simulator._class_waveforms,
    SNR held fixed (not sampled) so every point in this set is genuinely at
    the requested level."""
    rng = np.random.default_rng(seed)
    noise_frac = (SNR_RANGE_DB[1] - snr_level) / (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])
    frames = []
    for cls in C.CLASSES:
        block = _class_waveforms(rng, cls, n_per_class)
        for key, scale in [("amplitude", 0.25), ("velocity", 4.0), ("energy", 0.35)]:
            wf = block[key]
            wf_scale = (np.abs(wf).mean(axis=1, keepdims=True) + 1e-6) * scale
            block[key] = wf + noise_frac * wf_scale * rng.standard_normal(wf.shape)
        rows = [{
            "amplitude_waveform": block["amplitude"][i],
            "velocity_waveform": block["velocity"][i],
            "energy_waveform": block["energy"][i],
            "snr_db": float(snr_level),
            C.LABEL_COLUMN: cls,
        } for i in range(n_per_class)]
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def run_one_seed(train, fmap, seed):
    train_sub = pd.concat([
        train[train[C.LABEL_COLUMN] == cls].sample(
            n=min(N_TRAIN_PER_CLASS, (train[C.LABEL_COLUMN] == cls).sum()), random_state=seed)
        for cls in C.CLASSES
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(train_sub[QUANTUM_FEAT31])
    Xtr = scaler.transform(train_sub[QUANTUM_FEAT31])
    ytr = train_sub[C.LABEL_COLUMN].values

    S_tr = statevectors(fmap, Xtr)
    Ktr = fidelity_kernel(S_tr)
    qsvc = SVC(kernel="precomputed", C=1.0, probability=False, random_state=seed)
    qsvc.fit(Ktr, ytr)

    per_level = []
    for level in SNR_LEVELS:
        raw = generate_fixed_snr_test(level, N_TEST_PER_CLASS, seed=seed * 1000 + int(abs(level)))
        feats = extract_features(raw)
        Xte = scaler.transform(feats[QUANTUM_FEAT31])
        yte = feats[C.LABEL_COLUMN].values

        S_te = statevectors(fmap, Xte)
        Kte = fidelity_kernel(S_te, S_tr)
        pred = qsvc.predict(Kte)

        y_drone_true = (yte == "drone").astype(int)
        y_drone_pred = (pred == "drone").astype(int)
        per_level.append({
            "snr_db": level,
            "seed": seed,
            "n_test": len(yte),
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
            "drone_recall": float(recall_score(y_drone_true, y_drone_pred, zero_division=0)),
            "drone_precision": float(precision_score(y_drone_true, y_drone_pred, zero_division=0)),
            "drone_f1": float(f1_score(y_drone_true, y_drone_pred, zero_division=0)),
        })
    return per_level, len(train_sub), train_sub[C.LABEL_COLUMN].value_counts().to_dict()


def main():
    print("loading QSVC training pool from feat31_train_features.parquet "
          "(the SAME data the deployed 31-feature model was trained on), "
          f"class-balanced {N_TRAIN_PER_CLASS}/class, {N_SEEDS} seeds...")
    train = pd.read_parquet(C.DATA_DIR / "feat31_train_features.parquet")

    fmap = ZZFeatureMap(feature_dimension=4, reps=QUANTUM_REPS)
    circuit_meta = {"feature_map": "ZZFeatureMap", "qubits": 4, "reps": QUANTUM_REPS,
                    "circuit_depth": int(fmap.decompose().depth()),
                    "backend": "Qiskit exact statevector simulation (noiseless)",
                    "quantum_features": QUANTUM_FEAT31}
    print("circuit:", circuit_meta)

    all_rows = []
    n_train_seen, train_counts_seen = None, None
    for seed in range(C.SEED, C.SEED + N_SEEDS):
        print(f"\n=== seed {seed} ===")
        per_level, n_train_seen, train_counts_seen = run_one_seed(train, fmap, seed)
        for row in per_level:
            print(f"  SNR={row['snr_db']} accuracy={row['accuracy']:.4f} "
                  f"drone_recall={row['drone_recall']:.4f}")
        all_rows.extend(per_level)

    df_all = pd.DataFrame(all_rows)
    summary = (df_all.groupby("snr_db")
               .agg(n_test=("n_test", "first"),
                    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                    drone_recall_mean=("drone_recall", "mean"), drone_recall_std=("drone_recall", "std"),
                    drone_precision_mean=("drone_precision", "mean"),
                    drone_f1_mean=("drone_f1", "mean"))
               .reset_index()
               .sort_values("snr_db", ascending=False)
               .round(4))

    csv_path = C.REPORTS_DIR / "feat31_quantum_snr_comparison.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nCSV -> {csv_path}")

    out = {
        "purpose": ("Fresh quantum-kernel test at fixed SNR levels, requested for direct "
                    "comparison against an independently-run teammate result at the same "
                    "nominal SNR levels -- same dataset (feat31_train_features.parquet), "
                    "same quantum feature-selection methodology (top real SHAP features, "
                    "excluding snr_db), same circuit family used throughout this project."),
        "n_seeds": N_SEEDS,
        "n_train_per_seed": n_train_seen,
        "train_class_counts_example": train_counts_seen,
        "n_test_per_level": N_TEST_PER_CLASS * len(C.CLASSES),
        "circuit": circuit_meta,
        "summary_mean_std": summary.to_dict(orient="records"),
        "per_seed_results": all_rows,
    }
    (C.REPORTS_DIR / "feat31_quantum_snr_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"JSON -> {C.REPORTS_DIR / 'feat31_quantum_snr_comparison.json'}")


if __name__ == "__main__":
    main()
