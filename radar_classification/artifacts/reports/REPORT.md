# RadarSense — Performance Evaluation Report

*Every number in this report is generated from artifact JSONs produced by `python run_pipeline.py`; nothing is hand-typed.*

## 1. Dataset
- 52,000 samples, imbalance 20.0:1 (in 15–25 spec band: True), SNR realized [-20.0, -10.0] dB
- leakage: 0 duplicates, 0 train/test overlaps
- see `DATA_DICTIONARY.md` + `data_quality_report.json`

## 2. Classical bake-off (stratified 5-fold CV, train+val pool)
| model | CV drone PR-AUC | CV macro-F1 | test acc | test drone recall | test drone precision |
|---|---|---|---|---|---|
| logreg | 0.868 ± 0.024 | 0.878 ± 0.009 | 0.965 | 0.764 | 0.837 |
| random_forest | 0.909 ± 0.018 | 0.928 ± 0.006 | 0.981 | 0.821 | 0.860 |
| balanced_rf | 0.901 ± 0.023 | 0.903 ± 0.006 | 0.969 | 0.944 | 0.710 |
| xgboost | 0.923 ± 0.013 | 0.934 ± 0.004 | 0.981 | 0.826 | 0.847 |
| catboost | 0.928 ± 0.014 | 0.934 ± 0.006 | 0.981 | 0.821 | 0.860 |
| svm_rbf | 0.864 ± 0.016 | 0.911 ± 0.004 | 0.978 | 0.795 | 0.842 |

Selected by CV drone PR-AUC: **catboost**.

**Reproduced vanilla baseline (catboost, test set):** accuracy 98.1%, drone recall 82.1%, drone precision 86.0%. The prior-cycle numbers (97.2% acc / 38.5% recall) were re-run and replaced with these — our regenerated dataset is more separable than the prior simulation. The accuracy-hides-misses lesson still holds: 98.1% accuracy coexists with a 17.9% drone miss rate.

## 3. Minority-class optimization (distinct pipeline stage)
| variant | drone recall | drone precision | drone PR-AUC | macro-F1 |
|---|---|---|---|---|
| baseline | 0.826 | 0.856 | 0.938 | 0.931 |
| class_weight **(winner)** | 0.918 | 0.758 | 0.938 | 0.920 |
| smote | 0.918 | 0.772 | 0.931 | 0.924 |
| borderline_smote | 0.872 | 0.776 | 0.928 | 0.920 |
| adasyn | 0.908 | 0.760 | 0.923 | 0.920 |
| smote_tomek | 0.903 | 0.772 | 0.934 | 0.922 |
| smote+class_weight | 0.918 | 0.772 | 0.931 | 0.924 |

Threshold tuned on **validation only**: F2-optimal t=0.54, cost-optimal t=0.06 under cost matrix FN:FP = 25:1 (a missed drone is an incident; a false alarm is one operator check).

**Final deployed numbers (F2 threshold t=0.54, held-out test):** drone recall **91.8%**, drone precision **77.5%**, accuracy 97.6%, macro-F1 0.923. (argmax default: recall 91.8% / precision 75.8%)

Alternative **max-recall mode** (cost-optimal t=0.06 under the 25:1 matrix): recall 99.0% at precision 47.7% — offered as a switchable operating mode for heightened-alert postures; not the default because it trades below the 65% precision target.

Spec targets: recall >85% — **MET**; precision >65% — **MET**.

SMOTE-on-synthetic-data risk note: SMOTE interpolates synthetic minority samples; on simulated data this can amplify simulator artifacts. Mitigation evidence: the SMOTE-variant test PR-AUC vs the class-weight variant (no synthetic samples) is reported side by side above — if SMOTE materially outperformed class-weighting on synthetic data but not on real data, that gap would be the artifact signature to re-examine. Both variants' numbers are real held-out results.

## 4. Probability calibration (isotonic, fit on validation)
| class | Brier raw → cal | ECE raw → cal |
|---|---|---|
| aircraft | 0.0090 → 0.0069 | 0.0136 → 0.0020 |
| anomaly | 0.0152 → 0.0132 | 0.0158 → 0.0039 |
| bird | 0.0067 → 0.0045 | 0.0095 → 0.0022 |
| drone | 0.0064 → 0.0054 | 0.0073 → 0.0014 |

## 5. Robustness
- SNR-stratified recall curves: `plots/robustness_snr_curves.png`
- Adversarial stress (drone recall, frozen model): unperturbed 0.92, rcs_suppression_-3dBsm 0.93, rcs_suppression_-6dBsm 0.93, rcs_suppression_-9dBsm 0.94, jamming_snr_floor_-16dB 0.92, jamming_snr_floor_-18dB 0.95, jamming_snr_floor_-20dB 0.96, micro_doppler_masking_x0.7 0.76, micro_doppler_masking_x0.5 0.52, micro_doppler_masking_x0.3 0.12

## 6. Quantum kernel ablation (summary — full writeup in QUANTUM_MINIPAPER.md)
- ZZFeatureMap, 4 qubits, reps=2, depth 31, backend: Qiskit exact statevector simulation (noiseless); fidelity kernel K_ij = |<psi_i|psi_j>|^2 computed as a Gram matrix

| n_train | quantum PR-AUC | RBF PR-AUC | linear PR-AUC |
|---|---|---|---|
| 100 | 0.718 | 0.975 | 0.982 |
| 200 | 0.793 | 0.975 | 0.981 |
| 400 | 0.849 | 0.976 | 0.981 |
| 800 | 0.882 | 0.976 | 0.982 |

Kernel-target alignment at n=800: {'quantum_zz': 0.1583775305687742, 'rbf': 0.2851728927403711}

## 7. Explainability (SHAP)
- Drone-class top features: micro_doppler_bandwidth, range_m, altitude_m (physics-sane: True)
- Artifacts: `shap_summary_drone.png`, `shap_summary_global.png`, `shap_waterfall_drone_detection.png`

## 8. Deployment (ONNX, CPU-only)
- CPU: Intel64 Family 6 Model 142 Stepping 9, GenuineIntel | Windows 11 (AMD64) | onnxruntime 1.27.0 | GPU dependency: none
- Parity vs original model: label agreement 1.0000, max |Δp| 5.8e-07
- Single-sample latency: median **0.1069 ms** (p95 0.1686 ms) — target <10 ms
- Batched throughput: **38,249 samples/s** (batch 7800)

## 9. Real-radar validation (SAAB SIRS 77 GHz FMCW, Zenodo 5896641)
- 75,868 REAL measured segments: 6 drone types, 5+ bird species, humans, corner reflector (CC-BY-4.0). Features extracted from raw I/Q by `real_data_bridge.py` (RCS radiometrically calibrated on the 100 m² corner reflector; azimuth/altitude not measurable → constant).
- **Experiment A — zero-shot (frozen sim-trained model):** drone recall 30.4%, precision 82.0%, accuracy 39.0%. This is the measured sim-to-real gap — it converts our 'generalization unverified' disclaimer into a number.
- **Experiment B — same architecture retrained on real data (official splits):** drone recall **99.2%**, precision **94.3%**, bird recall 76.1%, accuracy **95.5%**. The pipeline transfers; the simulator's feature distributions don't — realism, not architecture, is the bottleneck.
- Figure: `plots/real_data_sim_to_real.png`. Caveat: official splits are segment-level (per [Karlsson et al. 2022]), not independent-flight; no aircraft class exists in this dataset.

### 9b. Second real dataset: RAD-DAR (UPM, 8.75 GHz FMCW)
- 17,485 real range-Doppler matrices (drones / cars / people), 80 measurement sessions. Cars+people mapped to anomaly; no bird class.
- Zero-shot frozen sim model: drone recall 2.1% — near-total failure, expected and documented: the sensor's ±2.8 m/s Doppler window and uncalibrated power proxy invalidate most schema features.
- Session-aware retrain (no session spans splits — stricter than segment-level): drone recall **98.5%**, precision 65.7%, accuracy 76.3%.
- Extraction assumptions listed in `raddar_results.json`.

### 9c. Third real dataset: TSMS-Drone CW radar (figshare)
- 2250 real CW frames (4 drones + corner reflector, 2-30 m): zero-shot drone recall 21.7%; range-generalization retrain (train even / test odd distances) drone recall 77.6%, accuracy 71.5%.

## 10. Honesty checkpoints (state unprompted)
- Statistical simulator, not full EM physics; real-sensor generalization unverified.
- All headline numbers are single-pass held-out test results; confusion matrices in the JSONs.
- Quantum result reported as measured, including where classical kernels match or beat it.
- The vanilla-baseline numbers were re-reproduced this cycle, not inherited from slides.