"""SNR Noise Stress-Test Script for Classical + Quantum Stacking Ensemble.

Evaluates the performance and robustness of:
  1. Pure Quantum Kernel (QSVC with ZZFeatureMap)
  2. Classical RBF Kernel (SVC)
  3. Classical + Quantum Stacking Ensemble (LogisticRegression meta-model)

Sweeps SNR from -10 dB (clean) down to -25 dB (CRITICAL NOISE) in steps of 2.5 dB.
Exports all evaluated metrics and breakdown tables to CSV and JSON artifacts.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from radarsense import config as C
from radarsense.quantum import fidelity_kernel, statevectors, subsample
from radarsense.simulator import _class_block

FEATURES = C.QUANTUM_FEATURES
N_TRAIN = 800
N_TEST_PER_BIN = 600
SNR_LEVELS = [-5.0, -7.5, -10.0, -12.5, -15.0, -17.5, -20.0, -22.5, -25.0, -27.5, -30.0]

CSV_OUTPUT_PATH = C.REPORTS_DIR / "snr_stress_test_ensemble_results.csv"
JSON_OUTPUT_PATH = C.REPORTS_DIR / "snr_stress_test_ensemble_summary.json"


def generate_snr_stressed_test_data(snr_target: float, n_samples: int = 600, seed: int = C.SEED + 42):
    """Generates synthetic test Detections fixed at a specific SNR level with SNR-dependent measurement noise."""
    rng = np.random.default_rng(seed)
    frames = []
    
    for cls, frac in C.CLASS_PROPORTIONS.items():
        n = int(round(n_samples * frac))
        block = _class_block(rng, cls, n)
        
        noise_frac = max(0.0, (C.SNR_RANGE_DB[1] - snr_target) / (C.SNR_RANGE_DB[1] - C.SNR_RANGE_DB[0]))
        noise_scales = {
            "doppler_velocity": 4.0,
            "rcs_estimate": 3.0,
            "micro_doppler_bandwidth": 60.0,
            "micro_doppler_jitter": 0.12,
        }
        
        for feat, scale in noise_scales.items():
            block[feat] = block[feat] + noise_frac * scale * rng.standard_normal(n)
            
        block["doppler_velocity"] = np.clip(block["doppler_velocity"], -50, 50)
        block["rcs_estimate"] = np.clip(block["rcs_estimate"], -20, 20)
        block["micro_doppler_bandwidth"] = np.clip(block["micro_doppler_bandwidth"], 0, 500)
        block["micro_doppler_jitter"] = np.clip(block["micro_doppler_jitter"], 0, 1)
        block["snr_db"] = np.full(n, snr_target)
        block["label"] = cls
        
        frames.append(pd.DataFrame(block))
        
    df = pd.concat(frames, ignore_index=True)
    return df


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculates classification metrics for binary drone detection."""
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "recall_drone": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_drone": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_drone": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "pr_auc_drone": round(float(average_precision_score(y_true, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def main():
    print("=" * 105)
    print("      RADARSENSE: SNR NOISE STRESS-TEST (QUANTUM vs CLASSICAL vs STACKING)      ")
    print("=" * 105)
    print(f"Sweep SNR Levels: {SNR_LEVELS} dB")
    print(f"Training Set Size: {N_TRAIN} | Test Samples per SNR Level: {N_TEST_PER_BIN}")
    print("-" * 105)

    print("\n[1/3] Loading training dataset & training Quantum, Classical, and Stacking Meta-Model...")
    train_df = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    Xtr_raw, ytr = subsample(train_df, N_TRAIN, C.SEED)

    scaler = MinMaxScaler(feature_range=(0.0, 2.0)).fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)

    fmap = ZZFeatureMap(feature_dimension=len(FEATURES), reps=C.QUANTUM_REPS)
    S_tr = statevectors(fmap, Xtr)
    Ktr = fidelity_kernel(S_tr)

    qsvc = SVC(kernel="precomputed", C=1.0, probability=True, random_state=C.SEED)
    qsvc.fit(Ktr, ytr)

    csvc = SVC(kernel="rbf", C=1.0, probability=True, random_state=C.SEED)
    csvc.fit(Xtr, ytr)

    q_tr_proba = qsvc.predict_proba(Ktr)[:, list(qsvc.classes_).index(1)]
    c_tr_proba = csvc.predict_proba(Xtr)[:, list(csvc.classes_).index(1)]
    meta_model = LogisticRegression(random_state=C.SEED)
    meta_model.fit(np.column_stack([q_tr_proba, c_tr_proba]), ytr)

    coef_q = float(meta_model.coef_[0, 0])
    coef_c = float(meta_model.coef_[0, 1])
    bias = float(meta_model.intercept_[0])
    print(f"    Meta-Model Equation: P(drone) = sigmoid({coef_q:.3f} * P_q + {coef_c:.3f} * P_c + {bias:.3f})")
    print("    Models successfully fitted.")

    print("\n[2/3] Running SNR Stress Sweep from -10dB to -25dB (CRITICAL NOISE)...")
    records = []

    for snr in SNR_LEVELS:
        test_df = generate_snr_stressed_test_data(snr_target=snr, n_samples=N_TEST_PER_BIN, seed=C.SEED + int(abs(snr) * 10))
        Xte_raw = test_df[FEATURES].values
        yte = (test_df["label"] == "drone").astype(int).values

        Xte = scaler.transform(Xte_raw)

        t0 = time.perf_counter()
        S_te = statevectors(fmap, Xte)
        Kte = fidelity_kernel(S_te, S_tr)
        q_proba = qsvc.predict_proba(Kte)[:, list(qsvc.classes_).index(1)]
        q_pred = (q_proba >= 0.5).astype(int)
        q_latency = ((time.perf_counter() - t0) * 1000) / len(Xte)
        q_metrics = calculate_metrics(yte, q_pred, q_proba)

        t0 = time.perf_counter()
        c_proba = csvc.predict_proba(Xte)[:, list(csvc.classes_).index(1)]
        c_pred = (c_proba >= 0.5).astype(int)
        c_latency = ((time.perf_counter() - t0) * 1000) / len(Xte)
        c_metrics = calculate_metrics(yte, c_pred, c_proba)

        t0 = time.perf_counter()
        stacked_input = np.column_stack([q_proba, c_proba])
        ens_proba = meta_model.predict_proba(stacked_input)[:, list(meta_model.classes_).index(1)]
        ens_pred = (ens_proba >= 0.5).astype(int)
        ens_latency = q_latency + c_latency + (((time.perf_counter() - t0) * 1000) / len(Xte))
        ens_metrics = calculate_metrics(yte, ens_pred, ens_proba)

        dom_q_votes = (np.abs(coef_q * q_proba) >= np.abs(coef_c * c_proba)).sum()
        dominant_model = "quantum" if dom_q_votes >= (len(Xte) / 2) else "classical"

        record = {
            "snr_db": snr,
            "n_test_samples": len(yte),
            "quantum_accuracy": q_metrics["accuracy"],
            "quantum_recall_drone": q_metrics["recall_drone"],
            "quantum_precision_drone": q_metrics["precision_drone"],
            "quantum_f1_drone": q_metrics["f1_drone"],
            "quantum_pr_auc": q_metrics["pr_auc_drone"],
            "quantum_roc_auc": q_metrics["roc_auc"],
            "quantum_latency_ms": round(q_latency, 3),
            "classical_accuracy": c_metrics["accuracy"],
            "classical_recall_drone": c_metrics["recall_drone"],
            "classical_precision_drone": c_metrics["precision_drone"],
            "classical_f1_drone": c_metrics["f1_drone"],
            "classical_pr_auc": c_metrics["pr_auc_drone"],
            "classical_roc_auc": c_metrics["roc_auc"],
            "classical_latency_ms": round(c_latency, 3),
            "ensemble_accuracy": ens_metrics["accuracy"],
            "ensemble_recall_drone": ens_metrics["recall_drone"],
            "ensemble_precision_drone": ens_metrics["precision_drone"],
            "ensemble_f1_drone": ens_metrics["f1_drone"],
            "ensemble_pr_auc": ens_metrics["pr_auc_drone"],
            "ensemble_roc_auc": ens_metrics["roc_auc"],
            "ensemble_latency_ms": round(ens_latency, 3),
            "ensemble_dominant_model": dominant_model,
            "ensemble_beats_quantum_pr_auc": bool(ens_metrics["pr_auc_drone"] > q_metrics["pr_auc_drone"]),
            "ensemble_beats_classical_pr_auc": bool(ens_metrics["pr_auc_drone"] > c_metrics["pr_auc_drone"]),
            "ensemble_recall_gain_vs_quantum": round(ens_metrics["recall_drone"] - q_metrics["recall_drone"], 4),
            "ensemble_recall_gain_vs_classical": round(ens_metrics["recall_drone"] - c_metrics["recall_drone"], 4),
        }
        records.append(record)

    results_df = pd.DataFrame(records)

    print("\n[3/3] Exporting stress-test results to CSV & JSON...")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"    CSV exported -> {CSV_OUTPUT_PATH}")

    json_output = {
        "title": "SNR Noise Stress-Test: Quantum vs Classical vs Stacking Ensemble",
        "snr_levels_db": SNR_LEVELS,
        "n_train": N_TRAIN,
        "n_test_per_bin": N_TEST_PER_BIN,
        "meta_model_coefficients": {"coef_quantum": coef_q, "coef_classical": coef_c, "bias": bias},
        "results": records,
    }
    JSON_OUTPUT_PATH.write_text(json.dumps(json_output, indent=2))
    print(f"    JSON exported -> {JSON_OUTPUT_PATH}")

    print("\n" + "=" * 105)
    print(f"{'SNR (dB)':<9} | {'QUANTUM (PR-AUC / Recall)':<26} | {'CLASSICAL (PR-AUC / Recall)':<26} | {'ENSEMBLE (PR-AUC / Recall)':<26} | {'DOMINANT'}")
    print("=" * 105)
    for r in records:
        q_str = f"{r['quantum_pr_auc']:.4f} / {r['quantum_recall_drone']:.4f}"
        c_str = f"{r['classical_pr_auc']:.4f} / {r['classical_recall_drone']:.4f}"
        e_str = f"{r['ensemble_pr_auc']:.4f} / {r['ensemble_recall_drone']:.4f}"
        flag = f"({r['ensemble_dominant_model'].upper()})"
        print(f"{r['snr_db']:>7.1f} dB | {q_str:<26} | {c_str:<26} | {e_str:<26} | {flag}")
    print("=" * 105)
    print("Stress-test completed successfully!\n")


if __name__ == "__main__":
    main()
