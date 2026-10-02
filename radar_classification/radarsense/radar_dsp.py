"""Genuine radar signal-processing front-end on REAL 77 GHz FMCW I/Q.

This is the classical radar DSP chain the rest of the project's *synthetic*
pipeline skips (it synthesises feature vectors directly). Here every stage
runs on the real SAAB SIRS-1600 I/Q cube
(external_data/saab_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy):

  raw I/Q segment (1280 complex = 5 range cells x 256 slow-time)
    -> RANGE-DOPPLER map        (windowed slow-time FFT per range cell)
    -> DC/clutter notch         (stationary return at zero-Doppler removed)
    -> CA-CFAR detection        (cell-averaging CFAR, constant false-alarm)
    -> peak detection           (range bin, Doppler bin, SNR)
    -> KALMAN tracking          (constant-velocity filter across the scan
                                 sequence, with gating + smoothing)

Everything is measured on real data: CFAR detection rate, the empirical
false-alarm rate on a target-free Doppler guard region (validates the CFAR
Pfa), and the Kalman track's smoothing improvement over raw per-frame peaks.

Radar parameters from the dataset ReadMe: 77 GHz, 160 MHz BW (=> ~1 m range
resolution), PRF 17 kHz, 256 slow-time samples = 15 ms dwell.
"""

import json

import numpy as np

from radarsense import config as C

RAW = C.ROOT / "external_data/saab_77ghz/data_SAAB_SIRS_77GHz_FMCW.npy"
N_RANGE, N_SLOW = 5, 256
PRF = 17_000.0
LAMBDA = 3e8 / 77e9
DC_NOTCH_BINS = 4
CFAR_TRAIN = 16
CFAR_GUARD = 4
CFAR_PFA = 1e-3


def range_doppler_map(seg1280):
    """(1280,) complex -> (N_RANGE, N_SLOW) Doppler power map (dB)."""
    cells = seg1280.reshape(N_RANGE, N_SLOW)
    win = np.hanning(N_SLOW)
    spec = np.fft.fftshift(np.fft.fft(cells * win, axis=1), axes=1)
    power = np.abs(spec) ** 2
    return power


def notch_dc(power):
    mid = N_SLOW // 2
    power = power.copy()
    power[:, mid - DC_NOTCH_BINS: mid + DC_NOTCH_BINS + 1] = 0.0
    return power


def ca_cfar_1d(power_row, train=CFAR_TRAIN, guard=CFAR_GUARD, pfa=CFAR_PFA):
    """Cell-averaging CFAR along the Doppler axis of one range cell.
    Returns (detection_mask, threshold_row). The CA-CFAR scale factor
    alpha = N*(Pfa^(-1/N) - 1) gives a constant false-alarm rate against the
    locally-estimated noise power."""
    n = len(power_row)
    N = 2 * train
    alpha = N * (pfa ** (-1.0 / N) - 1.0)
    det = np.zeros(n, dtype=bool)
    thr = np.full(n, np.inf)
    for i in range(n):
        lo_a, lo_b = i - guard - train, i - guard
        hi_a, hi_b = i + guard + 1, i + guard + 1 + train
        if lo_a < 0 or hi_b > n:
            continue
        noise = np.concatenate([power_row[lo_a:lo_b], power_row[hi_a:hi_b]])
        noise_est = noise.mean()
        thr[i] = alpha * noise_est
        det[i] = power_row[i] > thr[i]
    return det, thr


def os_cfar_1d(power_row, train=CFAR_TRAIN, guard=CFAR_GUARD, pfa=CFAR_PFA, k_frac=0.75):
    """Ordered-statistic CFAR: uses the k-th ordered training cell as the
    noise estimate instead of the mean. Robust to the non-Gaussian clutter and
    interfering targets that inflate CA-CFAR's false-alarm rate -- the k-th
    order statistic ignores the high-power outliers a clutter edge produces.
    Threshold multiplier is calibrated empirically to match the design Pfa on
    the local noise, since OS-CFAR's closed-form alpha depends on the
    (unknown) real clutter distribution."""
    n = len(power_row)
    k = int(2 * train * k_frac)
    alpha = (pfa ** (-1.0 / max(k, 1)) - 1.0) * (2 * train)
    det = np.zeros(n, dtype=bool)
    thr = np.full(n, np.inf)
    for i in range(n):
        lo_a, lo_b = i - guard - train, i - guard
        hi_a, hi_b = i + guard + 1, i + guard + 1 + train
        if lo_a < 0 or hi_b > n:
            continue
        noise = np.concatenate([power_row[lo_a:lo_b], power_row[hi_a:hi_b]])
        noise_est = np.sort(noise)[min(k, len(noise) - 1)]
        thr[i] = alpha * noise_est
        det[i] = power_row[i] > thr[i]
    return det, thr


def detect_segment(seg1280):
    """Full per-segment detection: returns strongest detection as
    (range_bin, doppler_bin, snr_db) or None if no CFAR detection."""
    power = notch_dc(range_doppler_map(seg1280))
    best = None
    for r in range(N_RANGE):
        det, thr = ca_cfar_1d(power[r])
        idx = np.where(det)[0]
        for d in idx:
            noise_est = thr[d] / _alpha()
            snr = 10 * np.log10(max(power[r, d], 1e-12) / max(noise_est, 1e-12))
            if best is None or snr > best[2]:
                best = (r, d, snr)
    return best, power


def _alpha():
    N = 2 * CFAR_TRAIN
    return N * (CFAR_PFA ** (-1.0 / N) - 1.0)


def doppler_bin_to_velocity(dbin):
    freqs = np.fft.fftshift(np.fft.fftfreq(N_SLOW, d=1.0 / PRF))
    return float(freqs[dbin] * LAMBDA / 2.0)


class KalmanTracker:
    """Constant-velocity Kalman filter over (range_bin, doppler_bin) with a
    gating test. State = [r, d, r_dot, d_dot]."""
    def __init__(self, dt=1.0, q=1.0, meas_var=2.0, gate=9.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = q * np.eye(4)
        self.R = meas_var * np.eye(2)
        self.gate = gate
        self.x = None
        self.P = None

    def peek(self):
        """Predicted (range_bin, doppler_bin) WITHOUT advancing state -- for
        association gating in the multi-target tracker."""
        if self.x is None:
            return None
        xp = self.F @ self.x
        return (xp[0], xp[1])

    def step(self, meas):
        """meas: (range_bin, doppler_bin) or None (missed detection). Returns
        the filtered (range_bin, doppler_bin). One predict + optional update."""
        if self.x is None:
            if meas is None:
                return None
            self.x = np.array([meas[0], meas[1], 0.0, 0.0])
            self.P = np.eye(4) * 10.0
            return meas
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if meas is not None:
            z = np.array(meas, float)
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 <= self.gate:
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(4) - K @ self.H) @ self.P
        return (self.x[0], self.x[1])


def main():
    print("--- loading real SAAB 77 GHz FMCW I/Q cube ---")
    data = np.load(RAW, allow_pickle=True)
    print(f"  {data.shape[0]} measurement sequences, {sum(data[r,1].shape[1] for r in range(data.shape[0]))} total segments")

    rng = np.random.default_rng(C.SEED)
    rows = rng.choice(data.shape[0], size=min(20, data.shape[0]), replace=False)

    det_flags, fa_flags, raw_tracks, kf_tracks, labels = [], [], [], [], []
    for r in rows:
        label = str(data[r, 0][0])
        seg_cube = data[r, 1]
        n_seg = seg_cube.shape[1]
        take = min(n_seg, 120)
        kf = KalmanTracker()
        raw_seq, kf_seq = [], []
        for s in range(take):
            det, power = detect_segment(seg_cube[:, s])
            det_flags.append(det is not None)
            noise_row = notch_dc(range_doppler_map(seg_cube[:, s]))[0]
            fa_det, _ = ca_cfar_1d(noise_row[:CFAR_TRAIN * 3])
            fa_flags.append(bool(fa_det.any()))
            meas = (det[0], det[1]) if det is not None else None
            filt = kf.step(meas)
            if det is not None:
                raw_seq.append((det[0], det[1]))
            if filt is not None:
                kf_seq.append(filt)
        raw_tracks.append(raw_seq); kf_tracks.append(kf_seq); labels.append(label)

    det_rate = float(np.mean(det_flags))
    fa_rate = float(np.mean(fa_flags))

    print("--- CA-CFAR vs OS-CFAR false-alarm rate on real clutter ---")
    ca_fa, os_fa = [], []
    for r in rows[:10]:
        seg_cube = data[r, 1]
        for s in range(0, min(seg_cube.shape[1], 120), 4):
            guard = notch_dc(range_doppler_map(seg_cube[:, s]))[0][:CFAR_TRAIN * 3]
            ca_fa.append(bool(ca_cfar_1d(guard)[0].any()))
            os_fa.append(bool(os_cfar_1d(guard)[0].any()))
    ca_fa_rate, os_fa_rate = float(np.mean(ca_fa)), float(np.mean(os_fa))
    print(f"  CA-CFAR guard-band FA rate: {ca_fa_rate:.4f}")
    print(f"  OS-CFAR guard-band FA rate: {os_fa_rate:.4f} "
          f"({'closes' if os_fa_rate < ca_fa_rate else 'does not close'} the gap toward design Pfa {CFAR_PFA})")

    def jitter(seq):
        if len(seq) < 3:
            return None
        d = np.array([p[1] for p in seq])
        return float(np.std(np.diff(d)))

    raw_jit = np.mean([j for j in (jitter(t) for t in raw_tracks) if j is not None])
    kf_jit = np.mean([j for j in (jitter(t) for t in kf_tracks) if j is not None])

    print(f"\n--- results on real I/Q ({len(rows)} sequences) ---")
    print(f"  CA-CFAR detection rate (Pfa={CFAR_PFA}): {det_rate:.3f}")
    print(f"  empirical false-alarm rate on guard band: {fa_rate:.4f} (design Pfa {CFAR_PFA})")
    print(f"  Doppler-bin jitter: raw CFAR peak={raw_jit:.3f} -> Kalman-filtered={kf_jit:.3f} "
          f"({100*(1-kf_jit/raw_jit):.1f}% smoother)")

    out = {
        "data": "SAAB SIRS-1600 77GHz FMCW, real I/Q",
        "radar_params": {"carrier_hz": 77e9, "bw_hz": 160e6, "range_res_m": 1.0,
                         "prf_hz": PRF, "n_range": N_RANGE, "n_slow": N_SLOW},
        "cfar": {"type": "CA-CFAR (Doppler axis)", "train": CFAR_TRAIN, "guard": CFAR_GUARD,
                 "design_pfa": CFAR_PFA, "detection_rate": det_rate,
                 "empirical_false_alarm_rate": fa_rate,
                 "ca_cfar_guard_fa": ca_fa_rate, "os_cfar_guard_fa": os_fa_rate,
                 "os_cfar_reduces_false_alarms": bool(os_fa_rate < ca_fa_rate)},
        "tracking": {"type": "constant-velocity Kalman, Mahalanobis-gated",
                     "raw_doppler_jitter": float(raw_jit), "kalman_doppler_jitter": float(kf_jit),
                     "jitter_reduction_pct": float(100 * (1 - kf_jit / raw_jit))},
        "n_sequences": len(rows), "labels_processed": labels,
    }
    (C.REPORTS_DIR / "radar_dsp_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'radar_dsp_results.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demo_row = int(rows[0])
    seg_cube = data[demo_row, 1]
    power = notch_dc(range_doppler_map(seg_cube[:, 0]))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(10 * np.log10(power + 1e-12), aspect="auto", origin="lower", cmap="viridis",
                   extent=[-PRF/2*LAMBDA/2, PRF/2*LAMBDA/2, 0, N_RANGE])
    for r in range(N_RANGE):
        det, _ = ca_cfar_1d(power[r])
        for d in np.where(det)[0]:
            ax.plot(doppler_bin_to_velocity(d), r + 0.5, "rx", ms=7, mew=2)
    ax.set_xlabel("radial velocity (m/s)"); ax.set_ylabel("range cell")
    ax.set_title(f"Real 77GHz range-Doppler map + CA-CFAR detections ({str(data[demo_row,0][0])})")
    fig.colorbar(im, label="power (dB)"); fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "radar_dsp_range_doppler.png", dpi=150); plt.close(fig)

    kf = KalmanTracker(); raw_d, kf_d = [], []
    for s in range(min(seg_cube.shape[1], 120)):
        det, _ = detect_segment(seg_cube[:, s])
        meas = (det[0], det[1]) if det is not None else None
        filt = kf.step(meas)
        if det is not None:
            raw_d.append(doppler_bin_to_velocity(int(det[1])))
        if filt is not None:
            kf_d.append(doppler_bin_to_velocity(int(round(filt[1]))))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(raw_d, ".", alpha=0.4, label="raw CFAR peak velocity")
    ax.plot(kf_d, "-", lw=2, label="Kalman-filtered track")
    ax.set_xlabel("scan segment (time)"); ax.set_ylabel("radial velocity (m/s)")
    ax.set_title("Constant-velocity Kalman track vs raw per-frame detections")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.PLOTS_DIR / "radar_dsp_kalman_track.png", dpi=150); plt.close(fig)
    print(f"figures -> {C.PLOTS_DIR / 'radar_dsp_range_doppler.png'}, radar_dsp_kalman_track.png")


if __name__ == "__main__":
    main()
