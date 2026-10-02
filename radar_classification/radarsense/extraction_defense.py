"""Trains and saves the query-pattern anomaly filter that
MODEL_EXTRACTION_ATTACK.md's defense-direction section named but didn't
build: "a burst of near-uniform-random, physically-implausible-in-
combination queries is a detectable signature distinct from real target
detections."

Measured first, before committing to this as the fix (see
extraction_defense_test.py / EXTRACTION_DEFENSE.md): rounding or
withholding the probability from the API response does NOTHING here --
all four response-hardening variants tested gave IDENTICAL extraction
fidelity, because the attack only ever needed the hard label, which any
classifier API has to return to be useful. The decision boundary is also
extremely cheap to learn (97% drone-gate fidelity from just 10 queries),
so rate-limiting alone would need to be so aggressive it breaks
legitimate use.

What DOES measurably work: an IsolationForest fit on the real training
distribution cleanly separates genuine detections from the attacker's
i.i.d.-uniform query sampling (real detections follow the simulator's
per-class physical correlations; synthetic probing doesn't). At a
threshold calibrated to reject only 1% of real detections, it catches
95.2% of the attacker's queries -- an attacker sending the same 5,000-query
budget as before gets only ~200 actually answered.

Honest limit, not overclaimed: this raises the cost of extraction by
roughly 25x, it does not eliminate it -- the ~200 queries that DO pass
(because they resemble real detections, i.e. sit near the real decision
boundary) are already informative enough on their own for a decent
surrogate. Combined with a request-rate cap on TOTAL attempts (not just
answered ones), this is a real, substantial deterrent, not a complete fix.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from radarsense import config as C

N_ESTIMATORS = 200
CONTAMINATION = 0.05
REJECT_PERCENTILE = 1


def main():
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    iso = IsolationForest(n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=C.SEED)
    iso.fit(train[C.FEATURE_COLUMNS])

    cal_scores = iso.score_samples(train[C.FEATURE_COLUMNS].sample(3000, random_state=C.SEED))
    reject_threshold = float(np.percentile(cal_scores, REJECT_PERCENTILE))

    test_scores = iso.score_samples(test[C.FEATURE_COLUMNS].sample(min(2000, len(test)), random_state=C.SEED))
    false_reject_rate = float((test_scores < reject_threshold).mean())
    print(f"reject_threshold={reject_threshold:.4f}, false-reject rate on real held-out "
          f"detections: {false_reject_rate:.2%}")

    joblib.dump(iso, C.MODELS_DIR / "extraction_anomaly_filter.joblib")
    (C.MODELS_DIR / "extraction_anomaly_filter_meta.json").write_text(json.dumps({
        "reject_threshold": reject_threshold,
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "false_reject_rate_on_real_holdout": false_reject_rate,
        "feature_order": C.FEATURE_COLUMNS,
    }, indent=2))
    print(f"saved -> {C.MODELS_DIR / 'extraction_anomaly_filter.joblib'}")


if __name__ == "__main__":
    main()
