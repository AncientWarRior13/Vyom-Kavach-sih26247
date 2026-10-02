# SNR-Aware (Piecewise) Threshold Tuning: Does It Help?

Tests whether tuning a separate drone-decision threshold per SNR band — instead of one
global threshold tuned across the full [-25,-10]dB range — improves on the 31-feature
model's current approach. [`feat31_snr_threshold.py`](../../radarsense/feat31_snr_threshold.py)
reuses `tune_threshold` from `optimize.py` unmodified, tunes it separately on each of the
same 8 SNR bins `feat31_train.py` already reports per-bin metrics for, then applies each
test sample's own bin's threshold at inference.

**Predicted going in** (correctly, as it turned out): this is the same F-beta2 threshold-
tuning lever already pulled once for the global case — diminishing returns were expected,
and that's what was found.

## Per-bin thresholds actually tuned

| SNR bin | n (val) | Tuned threshold |
|---|---|---|
| [-25,-24) | 826 | 0.285 |
| [-24,-22) | 1581 | 0.330 (= global) |
| [-22,-20) | 1691 | 0.515 |
| [-20,-18) | 1289 | 0.210 |
| [-18,-16) | 898 | 0.345 |
| [-16,-14) | 638 | 0.535 |
| [-14,-12) | 421 | 0.815 |
| [-12,-10) | 261 | 0.280 |

(Global threshold: 0.330.) The per-bin thresholds swing widely (0.21 to 0.82) — the
underlying probability calibration genuinely does shift across SNR bands, which is exactly
the premise this experiment set out to exploit.

## Result: no improvement, slightly worse overall

| | Global threshold | Piecewise threshold |
|---|---|---|
| Overall accuracy | **96.56%** | 96.46% |
| Overall drone recall | **83.08%** | 82.05% |
| Overall drone precision | **59.78%** | 58.18% |
| SNR≤-23dB accuracy | **96.10%** | 95.83% |
| SNR≤-23dB drone recall | 76.92% | 76.92% (identical) |
| SNR≤-23dB drone precision | **44.12%** | 41.67% |

Drone recall delta (piecewise − global): **−1.03 points overall, exactly 0 at the
deep-low-SNR slice.** The deep-low-SNR tie isn't a coincidence — most of that slice falls in
the [-24,-22) bin, whose independently-tuned threshold (0.330) happens to equal the global
one, so those samples get identical predictions either way; the [-25,-24) bin's different
threshold (0.285) changed some predictions but only enough to cost precision, not gain
recall.

## Honest takeaway

The premise was real (probability calibration does shift with SNR) but exploiting it with
per-bin thresholds doesn't pay off here — it adds real complexity (8 thresholds to maintain,
a live SNR-bin lookup at inference time, more surface area for a deployment bug) for a result
that's flat-to-slightly-worse on every metric checked. The single global threshold currently
deployed is the better choice; this was worth testing rather than assuming, but the answer is
"no, don't do this."

Plausible reason it doesn't help: each per-bin threshold is tuned on a smaller validation
slice (261–1691 samples vs. the global threshold's full ~7,800), so individual bin thresholds
are noisier estimates even before accounting for the fact that F-beta2-optimal thresholds
found on validation don't always transfer cleanly to test — the global threshold benefits
from averaging over more data, which apparently outweighs whatever gain a bin-matched
threshold could offer.

## Caveats

- Single run, not repeated across seeds.
- Only tested on the 31-feature model; the same experiment on the 11-feature or 42-feature
  combined model might behave differently (not run here — this was scoped to the model this
  request specifically asked about).
- Bin boundaries reused as-is from `feat31_train.py`'s existing reporting bins, not
  independently optimized — different bin widths might shift the result, though given the
  clean negative outcome here that seems unlikely to change the conclusion materially.

## Artifacts

- `radarsense/feat31_snr_threshold.py`
- `artifacts/reports/feat31_snr_threshold_results.json`
