"""Does the 42-feature combined model resist micro-Doppler masking better
than the 11-feature model, since it has redundant signal from a second
feature set? Tested, not assumed -- and the answer depends on doing this
FAIRLY: the 31-feature schema's amplitude/energy waveform statistics are
themselves driven by the SAME micro_doppler_bandwidth/jitter values as the
11-feature physics side (feat42_simulator.py's _waveforms31: mod_freq =
md_bw/4, mod_depth = md_jit) -- so masking only the physics-side columns
while leaving the waveform-stat columns untouched would NOT be a
physically faithful simulation of shrouded rotors (which would suppress
BOTH), and would make the combined model look more robust than it really
is.

This regenerates drone test instances with the SAME attenuated md_bw/
md_jit fed consistently into BOTH _physics11 and _waveforms31 (reusing
feat42_simulator.py's own functions, imported unmodified), at the exact
same attenuation levels robustness.py used for the 11-feature model
(x0.7, x0.5, x0.3) so the two are directly comparable.
"""

import json

import joblib
import numpy as np
import pandas as pd

from radarsense import config as C
from radarsense.feat42_simulator import FEAT42_COLUMNS, _physics11, _waveforms31
from radarsense.feat31_features import extract_features as extract_31
from radarsense.optimize import threshold_predict

ATTENUATIONS = (1.0, 0.7, 0.5, 0.3)
N_DRONE_SAMPLES = 1000


def generate_masked_drones(attenuation, n, seed):
    """Regenerate n drone instances with md_bw/md_jit attenuated BEFORE
    driving either feature set -- same physical event, both views masked
    consistently."""
    rng = np.random.default_rng(seed)
    from radarsense.feat42_simulator import _class_params_and_trajectories, SNR_RANGE_DB

    params = _class_params_and_trajectories(rng, "drone", n)
    params["md_bw"] = params["md_bw"] * attenuation
    params["md_jit"] = np.clip(params["md_jit"] * attenuation, 0, 1)

    raw = params["rcs"] - 40 * np.log10(params["rng_m"])
    lo, hi = np.percentile(raw, [2, 98])
    snr = SNR_RANGE_DB[0] + (raw - lo) / (hi - lo) * (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])
    snr = np.clip(snr + rng.normal(0, 1.0, n), *SNR_RANGE_DB)
    noise_frac = (SNR_RANGE_DB[1] - snr) / (SNR_RANGE_DB[1] - SNR_RANGE_DB[0])

    phys = _physics11(params, rng, n)
    noise_scales = {
        "doppler_velocity": 4.0, "rcs_estimate": 3.0,
        "micro_doppler_bandwidth": 60.0, "micro_doppler_jitter": 0.12,
        "range_m": 30.0, "azimuth_deg": 2.0, "altitude_m": 50.0,
        "velocity_variance": 2.0, "acceleration": 1.5,
    }
    for feat, scale in noise_scales.items():
        phys[feat] = phys[feat] + noise_frac * scale * rng.standard_normal(n)
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
    phys["snr_db"] = snr
    phys_df = pd.DataFrame(phys)
    phys_df[C.LABEL_COLUMN] = "drone"

    wave = _waveforms31(params, rng, n)
    for key, scale in [("amplitude", 0.25), ("velocity", 4.0), ("energy", 0.35)]:
        wf = wave[key]
        wf_scale = (np.abs(wf).mean(axis=1, keepdims=True) + 1e-6) * scale
        wave[key] = wf + noise_frac[:, None] * wf_scale * rng.standard_normal(wf.shape)
    wave_rows = [{
        "amplitude_waveform": wave["amplitude"][i], "velocity_waveform": wave["velocity"][i],
        "energy_waveform": wave["energy"][i], "snr_db": float(snr[i]), C.LABEL_COLUMN: "drone",
    } for i in range(n)]
    wave_feats = extract_31(pd.DataFrame(wave_rows))

    combined = pd.concat([phys_df[C.FEATURE_COLUMNS],
                          wave_feats[[c for c in FEAT42_COLUMNS if c not in C.FEATURE_COLUMNS]]], axis=1)
    combined[C.LABEL_COLUMN] = "drone"
    return combined[FEAT42_COLUMNS + [C.LABEL_COLUMN]]


def main():
    model = joblib.load(C.MODELS_DIR / "feat42_model.joblib")
    meta = json.loads((C.MODELS_DIR / "feat42_model_meta.json").read_text())
    threshold = meta["drone_threshold_deployed"]
    drone_idx = C.CLASSES.index("drone")

    results = {}
    for att in ATTENUATIONS:
        df = generate_masked_drones(att, N_DRONE_SAMPLES, seed=C.SEED + int(att * 100))
        proba = model.predict_proba(df[FEAT42_COLUMNS])
        proba_r = proba[:, [list(model.classes_).index(c) for c in C.CLASSES]]
        pred = threshold_predict(proba_r, threshold)
        recall = float((pred == "drone").mean())
        results[f"x{att}"] = recall
        label = "unperturbed" if att == 1.0 else f"micro_doppler_masking_x{att}"
        print(f"{label:32s} drone_recall={recall:.4f}")

    comparison = {
        "11_feature_model_reported": {"unperturbed": 0.918, "x0.7": 0.764, "x0.5": 0.518, "x0.3": 0.123},
        "42_feature_model_measured": {"x1.0": results["x1.0"], "x0.7": results["x0.7"],
                                      "x0.5": results["x0.5"], "x0.3": results["x0.3"]},
    }
    print("\n--- head-to-head at matching attenuation levels ---")
    for level, key11 in [("unperturbed/x1.0", "unperturbed"), ("x0.7", "x0.7"),
                         ("x0.5", "x0.5"), ("x0.3", "x0.3")]:
        v11 = comparison["11_feature_model_reported"][key11]
        v42 = comparison["42_feature_model_measured"][key11 if key11 != "unperturbed" else "x1.0"]
        print(f"  {level:20s} 11-feature={v11:.3f}  42-feature={v42:.3f}  "
              f"({'42-feature MORE robust' if v42 > v11 else '11-feature more robust' if v11 > v42 else 'tie'})")

    (C.REPORTS_DIR / "feat42_masking_robustness_results.json").write_text(json.dumps({
        "attenuation_levels": list(ATTENUATIONS),
        "n_drone_samples_per_level": N_DRONE_SAMPLES,
        "results": results,
        "comparison_vs_11_feature_model": comparison,
    }, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'feat42_masking_robustness_results.json'}")


if __name__ == "__main__":
    main()
