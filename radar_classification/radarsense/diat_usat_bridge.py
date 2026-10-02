"""DIAT-uSAT micro-Doppler spectrogram bridge + experiments.

Real X-band (~10 GHz) CW radar spectrogram IMAGES (not raw I/Q), DIAT
(Defence Institute of Advanced Technology, Pune), 4,849 JPGs across 10
class folders, average reported SNR ~4.32 dB (Paul & Bandyopadhyay,
IEEE Access 2026, "UAV Classification Using Attentive Binarized CNN for
Micro-Doppler Spectrograms"). Source: DIAT-uSAT_dataset.zip, extracted to
external_data/diat_usat/<folder>/figureN.jpg (4,849 images total --
verified to match the paper's stated dataset size exactly).

Unlike the other three real-data bridges, there is no raw signal here --
only rendered jet-colormap spectrogram plots with NO burned-in axis
numbers. This bridge:
  1. crops the white plot margin,
  2. inverts the jet colormap (nearest-neighbour against a 256-entry LUT
     built from matplotlib's own 'jet' colormap) to reconstruct a scalar
     power(time, frequency) matrix,
  3. derives the schema's spectral features from that matrix under a
     documented axis-calibration ASSUMPTION (frequency axis +-500 Hz,
     time axis 3.0 s -- no ground-truth axis values exist in the image;
     this is the analogue of RAD-DAR's assumed 20 km/h Doppler span).

Schema mapping notes:
- doppler_velocity / micro_doppler_bandwidth: power-weighted centroid /
  spread of the reconstructed spectrum. The source radar is already
  X-band ~10 GHz == the schema's CARRIER_FREQ_HZ, so (uniquely among the
  four real bridges) NO carrier rescale is needed.
- micro_doppler_jitter: centroid wander across time-columns within one
  image, clipped/scaled with the same convention every other bridge uses.
- velocity_variance / acceleration: derived from the per-column centroid
  track WITHIN a single image's own time axis, not across files -- there
  is no cross-file session/sequence metadata linking consecutive images
  (this differs structurally from the SAAB/RAD-DAR/TSMS bridges, which
  all derive kinematics from consecutive frames/files).
- rcs_estimate: no known-RCS reference target in this dataset (unlike the
  SAAB/TSMS corner reflector) -> uncalibrated z-scored power proxy, same
  convention as raddar_experiments.py.
- range_m: not measurable -> constant at the schema floor (50 m).
- azimuth_deg / altitude_m: constant 0.0 (unmeasurable, as in all bridges).
- classes: the extracted folder names use physical descriptors, not the
  paper's plain-English labels -- ground truth confirmed by exact image
  count (4,849, matching the paper): 2_blade_rotor_{1,2} (2-blade
  propeller RC plane -- "RC plane" in the paper) -> aircraft;
  3_long_blades_rotor, 3_short_blade_rotor_{1,2}, drone_{1,2} -> drone;
  Bird -> bird; Bird+2_Blade_rotor_{1,2} (bird near a hovering 2-blade
  mini-helicopter -- "mini-helicopter + bionic bird" in the paper, a
  confuser signal) -> anomaly.

Experiments:
  A) zero-shot frozen sim-trained model (drone recall) -- first real
     bridge where all 4 schema classes (incl. aircraft) are populated.
  B) variant-holdout retrain: for the 4 classes split across two numbered
     folders, train on the "_1" folder (minus a held-out validation tail),
     test on the "_2" folder -- a whole-folder holdout. Figure numbering
     is contiguous across each pair (e.g. 2_blade_rotor_2 starts at
     figure401), so this also acts as a chronological holdout, following
     the project's "split discipline stronger than the last" progression
     (SAAB official split -> RAD-DAR session-aware -> TSMS range-
     generalization -> DIAT-uSAT variant/chronological holdout). For the
     2 single-folder classes (Bird, 3_long_blades_rotor) contiguous
     figureN-sorted 70/15/15 index blocks are used for train/val/test,
     for the same anti-leakage reason (never a random shuffle).
"""

import glob
import json
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from radarsense import config as C
from radarsense.metrics import full_metrics
from radarsense.optimize import threshold_predict, tune_threshold

DIAT = C.ROOT / "external_data/diat_usat"

CLASS_MAP = {
    "2_blade_rotor_1": "aircraft",
    "2_blade_rotor_2": "aircraft",
    "3_long_blades_rotor": "drone",
    "3_short_blade_rotor_1": "drone",
    "3_short_blade_rotor_2": "drone",
    "drone_1": "drone",
    "drone_2": "drone",
    "Bird": "bird",
    "Bird+2_Blade_rotor_1": "anomaly",
    "Bird+2_Blade_rotor_2": "anomaly",
}

VARIANT_PAIRS = [
    ("2_blade_rotor_1", "2_blade_rotor_2"),
    ("3_short_blade_rotor_1", "3_short_blade_rotor_2"),
    ("drone_1", "drone_2"),
    ("Bird+2_Blade_rotor_1", "Bird+2_Blade_rotor_2"),
]
SINGLE_FOLDERS = ["Bird", "3_long_blades_rotor"]

F_DIAT = 10e9
FREQ_HALF_SPAN_HZ = 500.0
T_ASSUMED_SEC = 3.0
THUMB_SIZE = (140, 105)

_JET_LUT = cm.jet(np.linspace(0, 1, 256))[:, :3]
_JET_TREE = cKDTree(_JET_LUT)


def crop_plot_region(img_rgb):
    """Crop the white margin around the jet-colormap plot area."""
    non_white = np.any(img_rgb < 240, axis=-1)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img_rgb
    return img_rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def rgb_to_power(img_rgb):
    """Invert the jet colormap: RGB image -> scalar power(freq, time) matrix in [0,1]."""
    small = np.asarray(
        Image.fromarray(img_rgb).resize(THUMB_SIZE, Image.BILINEAR)
    ).astype(np.float64) / 255.0
    h, w, _ = small.shape
    _, idx = _JET_TREE.query(small.reshape(-1, 3))
    return (idx.astype(np.float64) / 255.0).reshape(h, w)


def image_features(fp):
    """Load one spectrogram JPG -> dict of schema spectral features."""
    img = np.asarray(Image.open(fp).convert("RGB"))
    power = rgb_to_power(crop_plot_region(img))
    h, w = power.shape

    freqs = np.linspace(FREQ_HALF_SPAN_HZ, -FREQ_HALF_SPAN_HZ, h)
    times = np.linspace(0, T_ASSUMED_SEC, w)

    total = power.sum()
    if total > 0:
        cen = float((freqs[:, None] * power).sum() / total)
        bw = float(np.sqrt(((freqs[:, None] - cen) ** 2 * power).sum() / total))
    else:
        cen, bw = 0.0, 0.0
    v = float(np.clip(cen * C.C_LIGHT / (2 * F_DIAT), -50, 50))
    bw10 = float(np.clip(bw, 0, 500))

    col_sum = power.sum(axis=0)
    weighted = (freqs[:, None] * power).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        col_cent = np.where(col_sum > 0, weighted / col_sum, np.nan)
    valid = ~np.isnan(col_cent)
    if valid.sum() >= 2:
        cv, tv = col_cent[valid], times[valid]
        jitter = float(np.clip(np.std(cv) / 10.0, 0, 1))
        v_track = cv * C.C_LIGHT / (2 * F_DIAT)
        velocity_variance = float(np.var(v_track))
        acceleration = (float(np.clip(np.polyfit(tv, v_track, 1)[0], -10, 10))
                         if valid.sum() >= 3 else 0.0)
    else:
        jitter, velocity_variance, acceleration = 0.0, 0.0, 0.0

    power_db = float(10 * np.log10(max(total, 1e-30)))
    row_energy = power.sum(axis=1)
    floor = float(np.median(row_energy[row_energy > 0])) if (row_energy > 0).any() else 1e-30
    peak = float(row_energy.max()) if row_energy.size else 1e-30
    snr_raw = float(10 * np.log10(max(peak, 1e-30) / max(floor, 1e-30)))

    return {"doppler_velocity": v, "micro_doppler_bandwidth": bw10,
            "micro_doppler_jitter": jitter, "velocity_variance": velocity_variance,
            "acceleration": acceleration, "_power_db": power_db, "_snr_raw": snr_raw}


def build_dataframe():
    rows = []
    for folder, label in CLASS_MAP.items():
        files = sorted(
            glob.glob(str(DIAT / folder / "figure*.jpg")),
            key=lambda p: int(re.search(r"figure(\d+)\.jpg$", p).group(1)))
        for fp in files:
            f = image_features(fp)
            f["label"] = label
            f["folder"] = folder
            f["figure_idx"] = int(re.search(r"figure(\d+)\.jpg$", fp).group(1))
            rows.append(f)
    df = pd.DataFrame(rows)

    p = df["_power_db"]
    df["rcs_estimate"] = ((p - p.mean()) / max(p.std(), 1e-9) * 5 - 5).clip(-20, 20)
    s = df["_snr_raw"]
    scale = 5.0 / max(float(s.quantile(0.95) - s.median()), 1e-6)
    df["snr_db"] = (-15.0 + (s - s.median()) * scale).clip(-20, -10)
    df["doppler_freq_shift"] = 2 * df["doppler_velocity"] * C.CARRIER_FREQ_HZ / C.C_LIGHT
    df["range_m"] = 50.0
    df["azimuth_deg"] = 0.0
    df["altitude_m"] = 0.0
    return df


def make_splits(df):
    """Train on '_1'/first-index-block folders (minus a val tail), test on
    '_2'/last-index-block folders. See module docstring, Experiment B."""
    train_idx, val_idx, test_idx = [], [], []
    for a, b in VARIANT_PAIRS:
        sub = df[df.folder == a].sort_values("figure_idx")
        cut = int(len(sub) * 0.85)
        train_idx += sub.index[:cut].tolist()
        val_idx += sub.index[cut:].tolist()
        test_idx += df[df.folder == b].index.tolist()
    for folder in SINGLE_FOLDERS:
        sub = df[df.folder == folder].sort_values("figure_idx")
        c1, c2 = int(len(sub) * 0.70), int(len(sub) * 0.85)
        train_idx += sub.index[:c1].tolist()
        val_idx += sub.index[c1:c2].tolist()
        test_idx += sub.index[c2:].tolist()
    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()


def reorder_proba(proba, classes_):
    """CatBoost's predict_proba column order follows clf.classes_; reindex to C.CLASSES."""
    classes_ = list(classes_)
    return proba[:, [classes_.index(c) for c in C.CLASSES]]


def main():
    print("extracting features from DIAT-uSAT spectrogram images (this reads "
          "~4,849 JPGs, a few minutes)...")
    df = build_dataframe()
    out_cols = C.FEATURE_COLUMNS + [C.LABEL_COLUMN, "folder", "figure_idx",
                                     "_power_db", "_snr_raw"]
    df = df[out_cols]
    df.to_parquet(C.DATA_DIR / "real_diat_usat.parquet")
    print(f"{len(df)} images | {df.label.value_counts().to_dict()} | "
          f"{df.folder.nunique()} folders")

    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    meta = json.loads((C.MODELS_DIR / "final_model_meta.json").read_text())
    proba = reorder_proba(model.predict_proba(df[C.FEATURE_COLUMNS]), model.classes_)
    pred = threshold_predict(proba, meta["drone_threshold_deployed"])
    a = full_metrics(df["label"], pred, proba)
    a["confusion_matrix"] = confusion_matrix(df["label"], pred, labels=C.CLASSES).tolist()
    print(f"A zero-shot: drone recall {a['recall_drone']:.3f} "
          f"precision {a['precision_drone']:.3f} acc {a['accuracy']:.3f}")

    train, val, test = make_splits(df)
    clf = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                             random_seed=C.SEED, verbose=0,
                             allow_writing_files=False)
    clf.fit(train[C.FEATURE_COLUMNS], train["label"],
            sample_weight=compute_sample_weight("balanced", train["label"]))

    proba_val = reorder_proba(clf.predict_proba(val[C.FEATURE_COLUMNS]), clf.classes_)
    t_fbeta, t_cost, _ = tune_threshold(proba_val, val["label"])

    proba_test = reorder_proba(clf.predict_proba(test[C.FEATURE_COLUMNS]), clf.classes_)
    pred_b = threshold_predict(proba_test, t_fbeta)
    b = full_metrics(test["label"], pred_b, proba_test)
    b["confusion_matrix"] = confusion_matrix(test["label"], pred_b, labels=C.CLASSES).tolist()
    b["threshold_fbeta2"] = t_fbeta
    b["threshold_cost_optimal"] = t_cost
    b["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    b["test_folders"] = sorted(test.folder.unique().tolist())
    print(f"B variant-holdout retrain: drone recall {b['recall_drone']:.3f} "
          f"precision {b['precision_drone']:.3f} acc {b['accuracy']:.3f} "
          f"(t={t_fbeta:.2f})")

    joblib.dump(clf, C.MODELS_DIR / "real_diat_usat_model.joblib")
    out = {
        "dataset": "DIAT-uSAT micro-Doppler spectrogram images (DIAT, Pune; "
                   "X-band ~10 GHz CW radar; Paul & Bandyopadhyay, IEEE Access 2026)",
        "n_samples": len(df),
        "assumptions": [
            "frequency axis calibrated to +-500 Hz (no ground-truth axis values "
            "burned into the images)",
            "time axis calibrated to 3.0 s",
            "rcs/power proxy uncalibrated (no known-RCS reference target in "
            "this dataset)",
            "range/azimuth/altitude unavailable -> constants",
            "velocity_variance/acceleration derived from the per-image time "
            "axis, not a cross-file sequence",
        ],
        "experiment_a_zero_shot": a,
        "experiment_b_variant_holdout_retrain": b,
        "summary": (
            f"{len(df)} real DIAT-uSAT spectrogram images (10 class-folders, "
            f"all 4 schema classes populated -- first real bridge to include "
            f"aircraft): zero-shot drone recall {a['recall_drone']:.1%}; "
            f"variant-holdout retrain (train on '_1'/first-index-block folders, "
            f"test on '_2'/last-index-block folders) drone recall "
            f"{b['recall_drone']:.1%}, accuracy {b['accuracy']:.1%}."),
    }
    (C.REPORTS_DIR / "diat_usat_results.json").write_text(json.dumps(out, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'diat_usat_results.json'}")


if __name__ == "__main__":
    main()
