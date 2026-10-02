"""RadarSense — central config and the FROZEN feature schema contract.

The FEATURE_COLUMNS list below is the shared contract with Pair 2.
Order matters: the ONNX model consumes features in exactly this order.
Any change here after handoff silently breaks the live demo — don't.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
MODELS_DIR = ARTIFACTS / "models"
PLOTS_DIR = ARTIFACTS / "plots"
REPORTS_DIR = ARTIFACTS / "reports"
ONNX_DIR = ARTIFACTS / "onnx"

SEED = 42

N_SAMPLES = 52_000
CLASS_PROPORTIONS = {
    "aircraft": 0.50,
    "bird": 0.40,
    "drone": 0.025,
    "anomaly": 0.075,
}
SNR_RANGE_DB = (-20.0, -10.0)

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

FEATURE_COLUMNS = [
    "doppler_velocity",
    "doppler_freq_shift",
    "rcs_estimate",
    "micro_doppler_bandwidth",
    "micro_doppler_jitter",
    "range_m",
    "azimuth_deg",
    "altitude_m",
    "velocity_variance",
    "acceleration",
    "snr_db",
]
LABEL_COLUMN = "label"
CLASSES = ["aircraft", "anomaly", "bird", "drone"]
MINORITY_CLASS = "drone"
MINORITY_IDX = CLASSES.index(MINORITY_CLASS)

CARRIER_FREQ_HZ = 10e9
C_LIGHT = 3e8

COST_FN_DRONE = 25.0
COST_FP_DRONE = 1.0

QUANTUM_FEATURES = [
    "doppler_velocity",
    "rcs_estimate",
    "micro_doppler_bandwidth",
    "micro_doppler_jitter",
]
QUANTUM_TRAIN_SIZES = [100, 200, 400, 800]
QUANTUM_TEST_SIZE = 300
QUANTUM_REPS = 2

for _d in (DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, ONNX_DIR):
    _d.mkdir(parents=True, exist_ok=True)
