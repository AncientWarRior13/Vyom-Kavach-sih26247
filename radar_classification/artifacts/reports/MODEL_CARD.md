# Model Card — RadarSense Final Classifier

**Model:** catboost (class_weight variant) | **Task:** 4-class radar target classification (aircraft / anomaly / bird / drone) with cost-optimal drone gate

## Intended use
Decision support for counter-UAS radar screening on simulated X-band returns. NOT qualified for operational deployment: trained purely on simulation; requires real-sensor validation first.

## Performance (held-out test, n=7,800)
- Drone recall 91.8%, precision 77.5% at deployed threshold t=0.54; accuracy 97.6%; macro-F1 0.923
- Confusion matrices: `optimization_results.json`
- Worst stress-test drone recall: 0.12 (micro_doppler_masking_x0.3)

## Factors
- Performance degrades with SNR (see `robustness_snr_curves.png`); the drone gate threshold trades ~precision for recall by design.

## Training data
- 52,000 simulated samples, 20:1 imbalance, seed-42 reproducible (`DATA_DICTIONARY.md`).

## Ethical considerations & caveats
- Defense-adjacent screening use; system outputs are advisory and must keep a human in the loop for any response action.
- False positives burden operators; false negatives are the costlier error — encoded explicitly in the 25:1 cost matrix rather than hidden.
- Synthetic-only training data; distribution shift on real sensors is expected and unquantified.