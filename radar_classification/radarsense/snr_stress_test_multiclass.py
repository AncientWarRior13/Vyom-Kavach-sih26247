"""4-Class Multi-Class SNR Noise Stress-Test Script.

Evaluates and compares ALL THREE ONNX models:
  1. Classic 11-Feature Model (radarsense.onnx)
  2. 31-Feature Waveform Model (feat31_model.onnx)
  3. 42-Feature Combined Model (feat42_model.onnx)

Across all THREE inference modalities (without requiring an external server):
  - Classic ONNX (Direct ONNX Runtime inference)
  - Pure Quantum Kernel (QSVC with 4-class prior-weighted mapping)
  - Classical + Quantum Stacking Ensemble (LogisticRegression meta-model)

Evaluates on identical 4-class balanced test datasets (aircraft, bird, drone, anomaly)
sweeping SNR from -5.0 dB (clean) down to -30.0 dB (extreme noise).
Exports 4-class benchmark tables to CSV and JSON artifacts.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from qiskit.circuit.library import ZZFeatureMap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.quantum import fidelity_kernel, statevectors, subsample
from radarsense.feat42_simulator import _class_params_and_trajectories, _physics11, _waveforms31
from radarsense.feat31_features import extract_features, FEAT31_COLUMNS

FEATURES_QUANTUM = C.QUANTUM_FEATURES
FEATURES_11 = C.FEATURE_COLUMNS
CLASSES_4 = C.CLASSES

META_11 = json.loads((C.ROOT / "artifacts/models/final_model_meta.json").read_text())
META_31 = json.loads((C.ROOT / "artifacts/models/feat31_model_meta.json").read_text())
META_42 = json.loads((C.ROOT / "artifacts/models/feat42_model_meta.json").read_text())

FEAT_ORDER_11 = META_11["feature_order"]
FEAT_ORDER_31 = META_31["feature_order"]
FEAT_ORDER_42 = META_42["feature_order"]

DRONE_T_11 = META_11.get("drone_threshold_deployed", 0.54)
DRONE_T_31 = META_31.get("drone_threshold_deployed", 0.33)
DRONE_T_42 = META_42.get("drone_threshold_deployed", 0.365)

N_TRAIN = 800
N_TEST_PER_BIN = 600
SNR_LEVELS = [-5.0, -7.5, -10.0, -12.5, -15.0, -17.5, -20.0, -22.5, -25.0, -27.5, -30.0]

CSV_OUTPUT_PATH = C.REPORTS_DIR / "snr_stress_test_multiclass_results.csv"
ENSEMBLE_CSV_PATH = C.REPORTS_DIR / "snr_stress_test_ensemble_results.csv"
JSON_OUTPUT_PATH = C.REPORTS_DIR / "snr_stress_test_multiclass_summary.json"


def generate_balanced_multifeat_data(snr_target: float, n_samples: int = 600, seed: int = C.SEED + 42):
    """Generates synthetic test dataset containing 11-feature, 31-feature, and 42-feature representations
    for the EXACT SAME 4-class targets at a given SNR level."""
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // len(CLASSES_4)
    
    phys_frames, wave_rows = [], []
    noise_frac = max(0.0, (C.SNR_RANGE_DB[1] - snr_target) / (C.SNR_RANGE_DB[1] - C.SNR_RANGE_DB[0]))

    for cls in CLASSES_4:
        params = _class_params_and_trajectories(rng, cls, n_per_class)

        phys = _physics11(params, rng, n_per_class)
        noise_scales = {
            "doppler_velocity": 4.0, "rcs_estimate": 3.0,
            "micro_doppler_bandwidth": 60.0, "micro_doppler_jitter": 0.12,
            "range_m": 30.0, "azimuth_deg": 2.0, "altitude_m": 50.0,
            "velocity_variance": 2.0, "acceleration": 1.5,
        }
        for feat, scale in noise_scales.items():
            phys[feat] = phys[feat] + noise_frac * scale * rng.standard_normal(n_per_class)
            
        phys["doppler_velocity"] = np.clip(phys["doppler_velocity"], -50, 50)
        phys["rcs_estimate"] = np.clip(phys["rcs_estimate"], -20, 20)
        phys["micro_doppler_bandwidth"] = np.clip(phys["micro_doppler_bandwidth"], 0, 500)
        phys["micro_doppler_jitter"] = np.clip(phys["micro_doppler_jitter"], 0, 1)
        phys["range_m"] = np.clip(phys["range_m"], 50, 5000)
        phys["azimuth_deg"] = np.clip(phys["azimuth_deg"], -180, 180)
        phys["altitude_m"] = np.clip(phys["altitude_m"], 0, 3000)
        phys["velocity_variance"] = np.clip(phys["velocity_variance"], 0, None)
        phys["acceleration"] = np.clip(phys["acceleration"], -10, 10)
        phys["doppler_freq_shift"] = 2 * phys["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT
        phys["snr_db"] = np.full(n_per_class, snr_target)
        
        phys_df = pd.DataFrame(phys)
        phys_df[C.LABEL_COLUMN] = cls
        phys_frames.append(phys_df)

        wave = _waveforms31(params, rng, n_per_class)
        for key, scale in [("amplitude", 0.25), ("velocity", 4.0), ("energy", 0.35)]:
            wf = wave[key]
            wf_scale = (np.abs(wf).mean(axis=1, keepdims=True) + 1e-6) * scale
            wave[key] = wf + noise_frac * wf_scale * rng.standard_normal(wf.shape)
        for i in range(n_per_class):
            wave_rows.append({
                "amplitude_waveform": wave["amplitude"][i],
                "velocity_waveform": wave["velocity"][i],
                "energy_waveform": wave["energy"][i],
                "snr_db": snr_target,
                C.LABEL_COLUMN: cls,
            })

    phys_all = pd.concat(phys_frames, ignore_index=True)
    wave_all_raw = pd.DataFrame(wave_rows)
    wave_feats = extract_features(wave_all_raw)

    combined = pd.concat([phys_all, wave_feats[[c for c in FEAT31_COLUMNS if c != "snr_db"]]], axis=1)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined


def calculate_multiclass_metrics(y_true, y_pred):
    """Calculates 4-class multi-class classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    recalls = recall_score(y_true, y_pred, labels=CLASSES_4, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=CLASSES_4, average=None, zero_division=0)
    
    rec_dict = {f"recall_{cls}": round(float(recalls[i]), 4) for i, cls in enumerate(CLASSES_4)}
    prec_dict = {f"precision_{cls}": round(float(precisions[i]), 4) for i, cls in enumerate(CLASSES_4)}
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        **rec_dict,
        **prec_dict,
    }


def map_frontend_4class(p_drone: float, onnx_probs: dict, drone_threshold: float = 0.54) -> str:
    """Frontend decision logic: maps binary quantum/ensemble p_drone to 4-class using ONNX priors."""
    if p_drone >= drone_threshold:
        return "drone"
    
    p_not_drone = 1.0 - p_drone
    other_classes = ["bird", "aircraft", "anomaly"]
    
    sum_others = sum(onnx_probs.get(cls, 0.0) for cls in other_classes)
    
    confidences = {}
    for cls in other_classes:
        prop = (onnx_probs.get(cls, 0.0) / sum_others) if sum_others > 0 else (1.0 / 3.0)
        confidences[cls] = prop * p_not_drone
        
    best_cls = max(other_classes, key=lambda c: confidences[c])
    return best_cls


def run_onnx_inference(sess, input_name, df, feature_order, drone_threshold):
    """Runs ONNX inference for a given model and returns predictions and probability dicts."""
    X = df[feature_order].values.astype(np.float32)
    raw_out = sess.run(None, {input_name: X})[-1]
    
    preds = []
    probs_list = []
    for item in raw_out:
        if isinstance(item, dict):
            p_dict = item
        else:
            p_dict = {c: float(item[i]) for i, c in enumerate(CLASSES_4)}
            
        probs_list.append(p_dict)
        if p_dict.get("drone", 0.0) >= drone_threshold:
            preds.append("drone")
        else:
            others = [c for c in CLASSES_4 if c != "drone"]
            preds.append(max(others, key=lambda c: p_dict.get(c, 0.0)))
            
    return preds, probs_list


def main():
    print("=" * 115)
    print("   RADARSENSE: COMPREHENSIVE 3-MODEL x 3-MODE SNR STRESS-TEST (11-Feat, 31-Feat, 42-Feat)   ")
    print("=" * 115)
    print(f"Sweep SNR Levels: {SNR_LEVELS} dB")
    print(f"Evaluation Dataset: Balanced 4-class (600 test samples per SNR level: 150 each)")
    print("-" * 115)

    sess_11 = ort.InferenceSession(str(C.ROOT / "artifacts/onnx/radarsense.onnx"), providers=["CPUExecutionProvider"])
    sess_31 = ort.InferenceSession(str(C.ROOT / "artifacts/onnx/feat31_model.onnx"), providers=["CPUExecutionProvider"])
    sess_42 = ort.InferenceSession(str(C.ROOT / "artifacts/onnx/feat42_model.onnx"), providers=["CPUExecutionProvider"])

    in_11 = sess_11.get_inputs()[0].name
    in_31 = sess_31.get_inputs()[0].name
    in_42 = sess_42.get_inputs()[0].name
    print("[1/3] Loaded all 3 ONNX models (radarsense.onnx, feat31_model.onnx, feat42_model.onnx).")

    print("[2/3] Fitting Quantum (QSVC) & Classical RBF & Stacking Meta-Model on train set...")
    train_df = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    Xtr_raw, ytr_binary = subsample(train_df, N_TRAIN, C.SEED)

    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)

    fmap = ZZFeatureMap(feature_dimension=len(FEATURES_QUANTUM), reps=C.QUANTUM_REPS)
    S_tr = statevectors(fmap, Xtr)
    Ktr = fidelity_kernel(S_tr)

    qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
    qsvc.fit(Ktr, ytr_binary)

    csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=C.SEED)
    csvc.fit(Xtr, ytr_binary)

    q_tr_proba = qsvc.predict_proba(Ktr)[:, list(qsvc.classes_).index(1)]
    c_tr_proba = csvc.predict_proba(Xtr)[:, list(csvc.classes_).index(1)]
    meta_model = LogisticRegression(random_state=C.SEED)
    meta_model.fit(np.column_stack([q_tr_proba, c_tr_proba]), ytr_binary)
    print("      Quantum & Ensemble models successfully trained.\n")

    print("[3/3] Running 4-Class SNR Stress Sweep across all models...")
    records = []

    for snr in SNR_LEVELS:
        test_df = generate_balanced_multifeat_data(snr_target=snr, n_samples=N_TEST_PER_BIN, seed=C.SEED + int(abs(snr) * 10))
        y_true = test_df["label"].values

        preds_11, probs_11 = run_onnx_inference(sess_11, in_11, test_df, FEAT_ORDER_11, DRONE_T_11)
        preds_31, probs_31 = run_onnx_inference(sess_31, in_31, test_df, FEAT_ORDER_31, DRONE_T_31)
        preds_42, probs_42 = run_onnx_inference(sess_42, in_42, test_df, FEAT_ORDER_42, DRONE_T_42)

        X_q_raw = test_df[FEATURES_QUANTUM].values
        X_q = scaler.transform(X_q_raw)

        S_te = statevectors(fmap, X_q)
        Kte = fidelity_kernel(S_te, S_tr)
        q_proba = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]
        c_proba = csvc.predict_proba(X_q)[:, list(csvc.classes_).index(1)]
        
        stacked_input = np.column_stack([q_proba, c_proba])
        ens_proba = meta_model.predict_proba(stacked_input)[:, list(meta_model.classes_).index(1)]

        q_preds_11 = [map_frontend_4class(q_proba[i], probs_11[i]) for i in range(len(test_df))]
        ens_preds_11 = [map_frontend_4class(ens_proba[i], probs_11[i]) for i in range(len(test_df))]

        q_preds_31 = [map_frontend_4class(q_proba[i], probs_31[i]) for i in range(len(test_df))]
        ens_preds_31 = [map_frontend_4class(ens_proba[i], probs_31[i]) for i in range(len(test_df))]

        q_preds_42 = [map_frontend_4class(q_proba[i], probs_42[i]) for i in range(len(test_df))]
        ens_preds_42 = [map_frontend_4class(ens_proba[i], probs_42[i]) for i in range(len(test_df))]

        m11_onnx = calculate_multiclass_metrics(y_true, preds_11)
        m11_q    = calculate_multiclass_metrics(y_true, q_preds_11)
        m11_ens  = calculate_multiclass_metrics(y_true, ens_preds_11)

        m31_onnx = calculate_multiclass_metrics(y_true, preds_31)
        m31_q    = calculate_multiclass_metrics(y_true, q_preds_31)
        m31_ens  = calculate_multiclass_metrics(y_true, ens_preds_31)

        m42_onnx = calculate_multiclass_metrics(y_true, preds_42)
        m42_q    = calculate_multiclass_metrics(y_true, q_preds_42)
        m42_ens  = calculate_multiclass_metrics(y_true, ens_preds_42)

        record = {
            "snr_db": snr,
            "n_test_samples": len(y_true),

            "onnx_accuracy": m11_onnx["accuracy"],
            "onnx_macro_f1": m11_onnx["macro_f1"],
            "onnx_recall_drone": m11_onnx["recall_drone"],
            "quantum_accuracy": m11_q["accuracy"],
            "quantum_macro_f1": m11_q["macro_f1"],
            "quantum_recall_drone": m11_q["recall_drone"],
            "ensemble_accuracy": m11_ens["accuracy"],
            "ensemble_macro_f1": m11_ens["macro_f1"],
            "ensemble_recall_drone": m11_ens["recall_drone"],

            "m11_onnx_acc": m11_onnx["accuracy"], "m11_onnx_f1": m11_onnx["macro_f1"],
            "m11_quantum_acc": m11_q["accuracy"], "m11_quantum_f1": m11_q["macro_f1"],
            "m11_ensemble_acc": m11_ens["accuracy"], "m11_ensemble_f1": m11_ens["macro_f1"],

            "m31_onnx_acc": m31_onnx["accuracy"], "m31_onnx_f1": m31_onnx["macro_f1"],
            "m31_quantum_acc": m31_q["accuracy"], "m31_quantum_f1": m31_q["macro_f1"],
            "m31_ensemble_acc": m31_ens["accuracy"], "m31_ensemble_f1": m31_ens["macro_f1"],

            "m42_onnx_acc": m42_onnx["accuracy"], "m42_onnx_f1": m42_onnx["macro_f1"],
            "m42_quantum_acc": m42_q["accuracy"], "m42_quantum_f1": m42_q["macro_f1"],
            "m42_ensemble_acc": m42_ens["accuracy"], "m42_ensemble_f1": m42_ens["macro_f1"],
        }
        records.append(record)

    results_df = pd.DataFrame(records)

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CSV_OUTPUT_PATH, index=False)
    results_df.to_csv(ENSEMBLE_CSV_PATH, index=False)
    print(f"Exported CSV -> {CSV_OUTPUT_PATH}")
    print(f"Exported CSV -> {ENSEMBLE_CSV_PATH}")

    json_output = {
        "title": "RadarSense Multi-Model (11, 31, 42 Feat) x Multi-Mode (ONNX, Quantum, Ensemble) Stress Test",
        "snr_levels_db": SNR_LEVELS,
        "n_train": N_TRAIN,
        "n_test_per_bin": N_TEST_PER_BIN,
        "results": records,
    }
    JSON_OUTPUT_PATH.write_text(json.dumps(json_output, indent=2))
    print(f"Exported JSON -> {JSON_OUTPUT_PATH}")

    print("\n" + "=" * 130)
    print(f"{'SNR':<6} | {'--- 11-FEATURE MODEL ---':<36} | {'--- 31-FEATURE MODEL ---':<36} | {'--- 42-FEATURE MODEL ---':<36}")
    print(f"{'(dB)':<6} | {'ONNX':<10} {'QUANTUM':<10} {'ENSEMBLE':<10} | {'ONNX':<10} {'QUANTUM':<10} {'ENSEMBLE':<10} | {'ONNX':<10} {'QUANTUM':<10} {'ENSEMBLE':<10}")
    print("=" * 130)
    for r in records:
        print(
            f"{r['snr_db']:>5.1f}  | "
            f"{r['m11_onnx_acc']*100:5.1f}%     {r['m11_quantum_acc']*100:5.1f}%     {r['m11_ensemble_acc']*100:5.1f}%     | "
            f"{r['m31_onnx_acc']*100:5.1f}%     {r['m31_quantum_acc']*100:5.1f}%     {r['m31_ensemble_acc']*100:5.1f}%     | "
            f"{r['m42_onnx_acc']*100:5.1f}%     {r['m42_quantum_acc']*100:5.1f}%     {r['m42_ensemble_acc']*100:5.1f}%"
        )
    print("=" * 130)
    print("Multi-model multi-mode 4-class stress test completed successfully!\n")


if __name__ == "__main__":
    main()
