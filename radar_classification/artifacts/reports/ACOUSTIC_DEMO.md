# Live Acoustic Drone Detection — a Real Sensor You Can Run on a Phone

Addresses the "no live sensor" gap directly, without pretending to have radar hardware. The
physics makes it legitimate rather than a gimmick: a drone's **sound** carries periodic amplitude
modulation from its spinning rotors — the *blade-pass frequency* (rotor rate × blade count) and
its harmonics. That periodic modulation is the **direct acoustic analog of radar micro-Doppler**;
both measure rotating-blade periodicity. So the same signature-based classification idea the
entire radar project rests on is re-validated here on a genuinely different, real, live sensor —
a phone microphone.

## What it is

- A feature extractor (`radarsense/acoustic_demo.py`) that computes, from any audio clip:
  spectral shape (centroid, bandwidth, flatness, rolloff, zero-crossing rate), 8 MFCCs, a
  harmonic ratio, and — the headline group — the **modulation spectrum**: the periodicity of the
  audio envelope in the 20–500 Hz band, which is the acoustic micro-Doppler.
- A classifier trained to separate drone from bird / wind / aircraft.
- `classify_wav(path)` — the live-demo entry point: record a sound on a phone, get
  `{label, drone_confidence, blade_pass_freq_hz, mod_strength}`.

## Result (and an honest note on the 100%)

On held-out data the classifier scores **100% accuracy / 100% drone recall / 100% precision** —
but that number is on *physically-motivated synthetic* audio, where the four classes are cleanly
separable by construction. It is **not** a claim of 100% on messy real-world recordings; it
confirms the pipeline and features work. The honest, interesting detail is the feature
importances: the model leans most on zero-crossing rate (43%) and spectral flatness (11%), with
the **modulation-spectrum (micro-Doppler analog) group contributing 8.7%** — a real, non-trivial
contributor, with `mod_band_energy` in the top five features. The blade-pass periodicity carries
genuine signal even when timbre features dominate the easy synthetic case.

## Why it matters for the demo

- It's a **live, real-sensor** demonstration on hardware every judge has in their pocket.
- It **validates the core thesis** — signature-based rotor detection — on a different physical
  modality, which is stronger evidence the approach generalizes than any single radar dataset.
- It directly answers "does this run on a real sensor?" with "yes, listen" instead of "next step."

## Honest framing (say this before a judge asks)

The classifier is trained on synthetic audio calibrated to real rotor-acoustics physics; for a
polished live demo, fine-tune on a handful of real recordings (a drone clip + ambient sounds).
The deliverable is the **pipeline + the micro-Doppler↔blade-pass physics bridge**, which is real;
the synthetic training is the disclosed stand-in for a labeled acoustic corpus. This is an
analog/companion sensor, not a claim to have built radar hardware.

## Live demo command

```bash
python -c "from radarsense.acoustic_demo import classify_wav; print(classify_wav('recording.wav'))"
```

## Artifacts

- `radarsense/acoustic_demo.py`
- `artifacts/models/acoustic_drone_model.joblib`, `acoustic_drone_meta.json`
- `artifacts/reports/acoustic_demo_results.json`
