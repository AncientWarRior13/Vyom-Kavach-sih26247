"""Synthesizes a physically-modeled 77GHz FMCW I/Q segment from a contact's
kinematics, in the exact (5 range cells x 256 slow-time) format
`radar_classification/radarsense/radar_dsp.py`'s real CA-CFAR/range-Doppler
pipeline expects.

This replaces the earlier approach of *formula-estimating* doppler_velocity/
snr_db/micro_doppler_bandwidth directly. There's no physical radar in this
trainer, so there's no real captured I/Q to feed the DSP chain — but we can
still do real signal processing on *synthesized* I/Q, instead of skipping
straight to invented feature values. The target's radial velocity, RCS, and
rotor/wing micro-motion are embedded as actual phase/frequency content in a
synthetic complex baseband signal; `radar_dsp.detect_segment` (genuine
range-Doppler FFT + DC notch + CA-CFAR) then has to *recover* velocity/SNR
from that signal exactly as it would from a real capture, and the measured
micro-Doppler spectral spread is read directly off the resulting
Doppler spectrum rather than assigned from a type-lookup table.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

RADARSENSE_ROOT = Path(__file__).resolve().parent.parent / "radar_classification"
if str(RADARSENSE_ROOT) not in sys.path:
    sys.path.insert(0, str(RADARSENSE_ROOT))

from radarsense.radar_dsp import (  # noqa: E402
    N_RANGE, N_SLOW, PRF, LAMBDA, detect_segment, doppler_bin_to_velocity, range_doppler_map, notch_dc, ca_cfar_1d,
)

TARGET_RANGE_CELL = 2  # arbitrary mid cell; others carry pure noise, like a real uncluttered range bin


def synthesize_segment(doppler_velocity_mps: float, rcs_db: float, snr_db: float,
                        micro_motion_hz: list[tuple[float, float]] | None = None,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Builds a (1280,) complex128 I/Q segment: one range cell carries a
    target echo (main Doppler tone from `doppler_velocity_mps` + optional
    micro-motion sidebands from rotor/wing modulation), the rest is noise.

    `micro_motion_hz`: list of (frequency_hz, relative_amplitude) sidebands
    riding on the main Doppler tone — this is what produces measurable
    micro-Doppler spectral spread in the resulting range-Doppler map.
    """
    rng = rng or np.random.default_rng()
    n = np.arange(N_SLOW)
    f_d = 2.0 * doppler_velocity_mps / LAMBDA  # radar Doppler equation
    f_d = np.clip(f_d, -PRF / 2, PRF / 2 - 1)  # stay within the unambiguous Doppler band

    amplitude = 10.0 ** (rcs_db / 20.0)
    phase = 2 * np.pi * f_d * n / PRF
    signal = amplitude * np.exp(1j * phase)

    for freq_hz, rel_amp in (micro_motion_hz or []):
        sideband_phase = 2 * np.pi * freq_hz * n / PRF
        signal = signal + amplitude * rel_amp * np.exp(1j * (phase + sideband_phase))

    noise_power = amplitude ** 2 / (10.0 ** (snr_db / 10.0))
    noise_std = math.sqrt(noise_power / 2.0)

    cells = (rng.normal(0, noise_std, (N_RANGE, N_SLOW)) + 1j * rng.normal(0, noise_std, (N_RANGE, N_SLOW)))
    cells[TARGET_RANGE_CELL] += signal
    return cells.reshape(-1)


def measure_micro_doppler(seg1280: np.ndarray) -> tuple[float, float]:
    """Measures spectral bandwidth/roughness around the target range cell's
    peak using the same CA-CFAR detector `radar_dsp.detect_segment` uses for
    primary detection — reusing a real adaptive-threshold detector instead of
    an ad hoc fixed fraction of peak power, which breaks down at this
    schema's low SNR range (-20..-10dB, by design: see FEATURE_SCHEMAS.md)
    where the noise floor alone can rival a naive peak-relative threshold.
    Returns (bandwidth_hz, jitter_0_to_1); (0, 0) if CFAR finds no spread at
    all around the peak (a real "signal too weak to resolve micro-Doppler").
    """
    power = notch_dc(range_doppler_map(seg1280))
    row = power[TARGET_RANGE_CELL]
    peak_idx = int(np.argmax(row))

    det, _ = ca_cfar_1d(row)
    # restrict to a window around the main peak so an unrelated CFAR false
    # alarm elsewhere in the row doesn't get counted as "spread"
    window = 15
    lo, hi = max(0, peak_idx - window), min(len(row), peak_idx + window + 1)
    local_det_idx = np.where(det[lo:hi])[0] + lo
    if len(local_det_idx) == 0:
        local_det_idx = np.array([peak_idx])  # at minimum, the peak bin itself

    bin_hz = PRF / N_SLOW
    bandwidth_hz = float((local_det_idx.max() - local_det_idx.min() + 1) * bin_hz)

    spread_power = row[local_det_idx]
    jitter = float(np.std(spread_power) / (np.mean(spread_power) + 1e-9))
    jitter = max(0.0, min(1.0, jitter))
    return bandwidth_hz, jitter


def dsp_measure(doppler_velocity_mps: float, rcs_db: float, snr_db_target: float,
                 micro_motion_hz: list[tuple[float, float]] | None = None,
                 rng: np.random.Generator | None = None) -> dict:
    """End-to-end: synthesize I/Q -> run the real CFAR/range-Doppler chain
    -> return what was actually *measured*, which is what a real sensor
    would hand the classifier (possibly noisier/different from the ground-
    truth kinematics that generated it — that's the point).
    """
    seg = synthesize_segment(doppler_velocity_mps, rcs_db, snr_db_target, micro_motion_hz, rng)
    detection, _ = detect_segment(seg)
    bandwidth_hz, jitter = measure_micro_doppler(seg)

    if detection is None:
        # CFAR didn't trigger — a real missed detection at this range/SNR.
        # Fall back to the ground-truth kinematics rather than fabricating
        # a detection that didn't happen, same as the pre-DSP behavior.
        return {
            "doppler_velocity": doppler_velocity_mps,
            "snr_db": snr_db_target,
            "micro_doppler_bandwidth": bandwidth_hz,
            "micro_doppler_jitter": jitter,
            "cfar_detected": False,
        }

    range_bin, doppler_bin, measured_snr = detection
    measured_velocity = doppler_bin_to_velocity(doppler_bin)
    return {
        "doppler_velocity": measured_velocity,
        "snr_db": measured_snr,
        "micro_doppler_bandwidth": bandwidth_hz,
        "micro_doppler_jitter": jitter,
        "cfar_detected": True,
    }
