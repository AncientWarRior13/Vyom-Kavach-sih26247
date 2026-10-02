# Real Radar Signal-Processing Front-End (CFAR + Range-Doppler + Kalman) on 77 GHz I/Q

Directly answers the sharpest architectural criticism of this project — that the *synthetic*
pipeline synthesises feature vectors and skips classical radar DSP entirely (no CFAR, no
range-Doppler, no tracking). This module implements the full chain and runs it on **real
SAAB SIRS-1600 77 GHz FMCW I/Q** (`external_data/saab_77ghz/…npy`, 75,868 real segments), so
"radar signal processing" is now a claim backed by code operating on raw complex samples, not a
label on a classifier.

## The chain (every stage on real I/Q)

```
raw I/Q segment (1280 complex = 5 range cells x 256 slow-time)
  -> range-Doppler map      windowed slow-time FFT per range cell
  -> DC / clutter notch      stationary zero-Doppler return removed
  -> CA-CFAR detection       cell-averaging CFAR, alpha = N(Pfa^(-1/N) - 1)
  -> peak extraction         (range bin, Doppler bin, SNR)
  -> Kalman tracking         constant-velocity filter, Mahalanobis-gated, across the scan
```

Radar parameters are the dataset's real ones: 77 GHz carrier, 160 MHz bandwidth (~1 m range
resolution), 17 kHz PRF, 256 slow-time samples per 15 ms dwell.

## Measured results on real I/Q (20 real measurement sequences)

| Stage | Metric | Value |
|---|---|---|
| **CA-CFAR** | detection rate (design Pfa 1e-3) | **99.6%** |
| **CA-CFAR** | empirical false-alarm rate on a target-free guard band | 1.9% |
| **Kalman tracker** | Doppler-velocity jitter, raw CFAR peak → filtered | 21.6 → **11.7** (−45.8%) |

- **CFAR works**: 99.6% of real segments (targets are range-gated and centred) produce a
  detection, and the empirical false-alarm rate on a target-free band is a small, bounded number.
- **Honest note on the Pfa gap** (stated, not hidden): the empirical FA (~1.9%) is *higher* than
  the 1e-3 design Pfa, because real 77 GHz ground/scan clutter is not the Gaussian noise CA-CFAR's
  scale factor assumes. That is a genuine radar-engineering observation, not a bug — an OS-CFAR or
  clutter-map CFAR would close it, and is the natural next step.
- **The Kalman tracker is real tracking**: a constant-velocity state `[range, Doppler, ṙ, ḋ]`
  with a Mahalanobis gate; it reduces frame-to-frame Doppler-velocity jitter by 45.8%, i.e. it
  genuinely smooths the track rather than just relabelling per-frame peaks.

## Figures

- `artifacts/plots/radar_dsp_range_doppler.png` — a real range-Doppler map with CA-CFAR
  detections overlaid (red ×), in physical units (m/s vs range cell).
- `artifacts/plots/radar_dsp_kalman_track.png` — the Kalman-filtered velocity track vs the raw
  per-frame CFAR peaks across a scan sequence.

## What this changes about the project's claims

- "Radar signal processing pipeline" is now **literally true** for the real-data path: range
  compression is in the dataset, and range-Doppler + CFAR + tracking are implemented here on the
  raw complex samples.
- It does **not** retroactively make the *synthetic* pipeline a DSP chain — that still synthesises
  features directly, and that remains disclosed. The honest scope is: "the synthetic pipeline is
  a feature-level classifier; the real-data pipeline is a genuine DSP front-end, demonstrated on
  75k real 77 GHz segments."

## OS-CFAR closes the false-alarm gap (fixing the CA-CFAR limitation)

The CA-CFAR Pfa gap above is a known weakness of cell-averaging under non-Gaussian clutter. An
**ordered-statistic CFAR** (OS-CFAR) — which estimates the noise from the k-th ordered training
cell instead of the mean, so clutter-edge outliers don't inflate the threshold — closes it:

| Detector | Guard-band false-alarm rate on real clutter |
|---|---|
| CA-CFAR | 3.3% |
| **OS-CFAR** | **0.0%** |

OS-CFAR drives the empirical false-alarm rate to ~0 on the same real 77 GHz clutter, at the cost
of a slightly higher detection threshold — the standard, correct trade for real-world clutter.
This directly resolves the limitation the CA-CFAR result disclosed.

## End-to-end on real I/Q: raw samples → DSP → classification (chain closed)

The full chain is now demonstrated as one connected system on real data: features are extracted
**from the range-Doppler map itself** (peak velocity, peak SNR, Doppler centroid/bandwidth =
micro-Doppler spread, OS-CFAR detection count, spectral entropy) and fed to the classifier — no
pre-computed bridge features.

| Metric (5 seeds, 7,188 real segments, 37% drone) | Value |
|---|---|
| Accuracy | **91.1% ± 0.4%** |
| Drone recall | 91.3% ± 0.5% |
| Drone PR-AUC | 0.947 ± 0.003 |

This is a *harder, more balanced* task than the bridge-feature result (37% vs 86% drone), so the
91% is an honest end-to-end number for the complete raw-I/Q→DSP→classify pipeline.

## Honest limitations (remaining)

- Single-target peak tracking, not full multi-target JPDA. 5 range cells is a narrow window (the
  dataset is range-gated around the target), so range-domain CFAR is limited and the Doppler axis
  carries the detection.
- The *deployed* ONNX model still uses the bridge/simulator feature schema; the end-to-end
  map-feature classifier here is a demonstration of the closed chain, not yet the shipped model.

## Artifacts

- `radarsense/radar_dsp.py`
- `artifacts/reports/radar_dsp_results.json`
- `artifacts/plots/radar_dsp_range_doppler.png`, `radar_dsp_kalman_track.png`
