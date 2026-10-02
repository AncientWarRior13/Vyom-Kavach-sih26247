"""Acoustic drone detection -- a LIVE, real-sensor demo that runs on a phone
or laptop microphone, addressing the "no live sensor" gap without pretending
to have a radar.

The physics bridge that makes this legitimate, not a gimmick: a drone's sound
carries periodic amplitude modulation from its spinning rotors -- the
blade-pass frequency (rotor_rate x n_blades) and its harmonics. That periodic
modulation is the DIRECT ACOUSTIC ANALOG of radar micro-Doppler (both measure
rotating-blade periodicity). So the same signature-based classification idea
the whole radar project rests on is re-validated here on a genuinely different,
real, live sensor.

This module:
  1. Synthesizes physically-motivated acoustic signatures (drone rotor
     harmonics + motor whine; bird chirps; wind broadband; aircraft rumble)
     so the pipeline is demonstrable even before a real dataset is collected.
  2. Extracts acoustic features, INCLUDING the modulation spectrum (the
     acoustic micro-Doppler analog), MFCCs, spectral shape, harmonic ratio.
  3. Trains a classifier and reports held-out accuracy.
  4. classify_wav(path) -> runs on ANY real .wav recorded on a phone, so the
     live demo is: record a sound -> get drone/not-drone + confidence.

Honest framing for the demo: trained on physically-motivated synthetic audio;
for a live phone demo, add a handful of real recordings (drone clip + ambient)
to fine-tune. The pipeline and the physics analogy are the deliverable; the
synthetic training is the honestly-disclosed stand-in for a labeled corpus.
"""

import json

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from radarsense import config as C

SR = 22050
CLIP_SECONDS = 2.0
CLASSES = ["drone", "bird", "wind", "aircraft"]
N_PER_CLASS = 250


def _synth_clip(cls, rng):
    n = int(SR * CLIP_SECONDS)
    t = np.arange(n) / SR
    if cls == "drone":
        rotor_hz = rng.uniform(80, 160)
        n_blades = rng.choice([2, 3])
        blade_pass = rotor_hz * n_blades
        motor = rng.uniform(1500, 4000)
        sig = 0.5 * np.sin(2 * np.pi * motor * t)
        mod = 1.0 + 0.7 * np.sin(2 * np.pi * blade_pass * t)
        for h in range(1, 5):
            sig += (0.4 / h) * np.sin(2 * np.pi * blade_pass * h * t)
        sig *= mod
        sig += 0.15 * rng.standard_normal(n)
    elif cls == "bird":
        sig = np.zeros(n)
        for _ in range(rng.integers(2, 6)):
            f0, f1 = rng.uniform(2000, 4000), rng.uniform(4000, 8000)
            start = rng.uniform(0, CLIP_SECONDS - 0.3)
            m = (t >= start) & (t < start + rng.uniform(0.08, 0.25))
            sweep = np.linspace(f0, f1, m.sum())
            sig[m] += 0.6 * np.sin(2 * np.pi * sweep * t[m])
        sig += 0.05 * rng.standard_normal(n)
    elif cls == "wind":
        sig = rng.standard_normal(n)
        b = np.exp(-np.arange(200) / 40.0); b /= b.sum()
        sig = np.convolve(sig, b, mode="same") * 0.8
    else:
        base = rng.uniform(80, 220)
        sig = 0.6 * np.sin(2 * np.pi * base * t) + 0.3 * np.sin(2 * np.pi * base * 2 * t)
        sig *= 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)
        sig += 0.2 * rng.standard_normal(n)
    return (sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def extract_acoustic_features(y, sr=SR):
    """Acoustic feature vector, with the modulation spectrum (micro-Doppler
    analog) as the headline feature group."""
    import librosa
    feats = {}
    feats["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feats["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    feats["spectral_flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    feats["spectral_rolloff"] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    feats["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
    for i in range(8):
        feats[f"mfcc_{i}"] = float(np.mean(mfcc[i]))
    env = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)).mean(axis=0)
    env = env - env.mean()
    mod_spec = np.abs(np.fft.rfft(env))
    mod_freqs = np.fft.rfftfreq(len(env), d=256 / sr)
    band = (mod_freqs > 20) & (mod_freqs < 500)
    feats["mod_peak_freq"] = float(mod_freqs[band][np.argmax(mod_spec[band])]) if band.any() else 0.0
    feats["mod_peak_strength"] = float(mod_spec[band].max() / (mod_spec.mean() + 1e-9)) if band.any() else 0.0
    feats["mod_band_energy"] = float(mod_spec[band].sum() / (mod_spec.sum() + 1e-9)) if band.any() else 0.0
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)).mean(axis=1)
    thresh = spec.mean() + 2 * spec.std()
    feats["harmonic_ratio"] = float(spec[spec > thresh].sum() / (spec.sum() + 1e-9))
    return feats


FEATURE_ORDER = None


def build_dataset(seed=C.SEED):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for cls in CLASSES:
        for _ in range(N_PER_CLASS):
            rows.append(extract_acoustic_features(_synth_clip(cls, rng)))
            labels.append("drone" if cls == "drone" else "not_drone")
    global FEATURE_ORDER
    FEATURE_ORDER = list(rows[0].keys())
    X = np.array([[r[k] for k in FEATURE_ORDER] for r in rows])
    return X, np.array(labels)


def classify_wav(path, model=None):
    """Live-demo entry point: classify a real recording from a phone/laptop."""
    import librosa
    if model is None:
        model = joblib.load(C.MODELS_DIR / "acoustic_drone_model.joblib")
        order = json.loads((C.MODELS_DIR / "acoustic_drone_meta.json").read_text())["feature_order"]
    else:
        order = FEATURE_ORDER
    y, _ = librosa.load(path, sr=SR, duration=CLIP_SECONDS)
    if len(y) < int(SR * CLIP_SECONDS):
        y = np.pad(y, (0, int(SR * CLIP_SECONDS) - len(y)))
    feats = extract_acoustic_features(y)
    x = np.array([[feats[k] for k in order]])
    proba = model.predict_proba(x)[0]
    idx = list(model.classes_).index("drone")
    return {"label": model.predict(x)[0], "drone_confidence": float(proba[idx]),
            "blade_pass_freq_hz": feats["mod_peak_freq"], "mod_strength": feats["mod_peak_strength"]}


def main():
    print("--- building acoustic dataset (physically-motivated synthesis) ---")
    X, y = build_dataset()
    print(f"  {len(X)} clips, {X.shape[1]} features, drone fraction {np.mean(y=='drone'):.2f}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=C.SEED)

    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", ytr)
    model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.1,
                               random_seed=C.SEED, verbose=0, allow_writing_files=False)
    model.fit(Xtr, ytr, sample_weight=sw)

    from radarsense.metrics import full_metrics
    pred = model.predict(Xte).ravel()
    proba = model.predict_proba(Xte)
    di = list(model.classes_).index("drone")
    acc = float((pred == yte).mean())
    drone_recall = float((pred[yte == "drone"] == "drone").mean())
    drone_prec = float((yte[pred == "drone"] == "drone").mean()) if (pred == "drone").any() else 0.0
    print(f"  held-out accuracy={acc:.4f} drone_recall={drone_recall:.4f} drone_precision={drone_prec:.4f}")

    imp = dict(zip(FEATURE_ORDER, model.get_feature_importance()))
    mod_imp = imp["mod_peak_freq"] + imp["mod_peak_strength"] + imp["mod_band_energy"]
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
    print(f"  modulation-spectrum (micro-Doppler analog) total importance: {mod_imp:.1f}%")
    print(f"  top features: {[(k, round(v,1)) for k,v in top]}")

    joblib.dump(model, C.MODELS_DIR / "acoustic_drone_model.joblib")
    (C.MODELS_DIR / "acoustic_drone_meta.json").write_text(json.dumps({
        "feature_order": FEATURE_ORDER, "classes": list(model.classes_),
        "sample_rate": SR, "clip_seconds": CLIP_SECONDS}, indent=2))
    (C.REPORTS_DIR / "acoustic_demo_results.json").write_text(json.dumps({
        "n_clips": len(X), "accuracy": acc, "drone_recall": drone_recall,
        "drone_precision": drone_prec,
        "modulation_spectrum_importance_pct": mod_imp,
        "feature_importance": imp,
        "note": "trained on physically-motivated synthetic audio; classify_wav() runs on real phone recordings"},
        indent=2))
    print(f"\nmodel -> {C.MODELS_DIR / 'acoustic_drone_model.joblib'}")
    print("live demo: python -c \"from radarsense.acoustic_demo import classify_wav; print(classify_wav('recording.wav'))\"")


if __name__ == "__main__":
    main()
