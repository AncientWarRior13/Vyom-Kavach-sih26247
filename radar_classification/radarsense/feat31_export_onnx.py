"""ONNX export + parity verification + CPU benchmark for the 31-feature model.

Same methodology as radarsense/export_onnx.py (the incumbent's export
script): CatBoost's native ONNX export, parity check against the original
model's predictions, then real measured latency/throughput on this CPU.
"""

import json
import platform
import subprocess
import time

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from radarsense import config as C
from radarsense.feat31_features import FEAT31_COLUMNS

ONNX_PATH = C.ONNX_DIR / "feat31_model.onnx"


def cpu_name():
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    return platform.processor() or platform.machine()


def main():
    model = joblib.load(C.MODELS_DIR / "feat31_model.joblib")
    test = pd.read_parquet(C.DATA_DIR / "feat31_test_features.parquet")
    X = test[FEAT31_COLUMNS].values.astype(np.float32)

    model.save_model(str(ONNX_PATH), format="onnx")
    print(f"exported -> {ONNX_PATH} ({ONNX_PATH.stat().st_size/1024:.0f} KiB)")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def proba_from_outputs(outputs):
        last = outputs[-1]
        if isinstance(last, list) and last and isinstance(last[0], dict):
            classes_ = list(model.classes_)
            return np.array([[row[c] for c in classes_] for row in last])
        return np.asarray(last)

    n_par = min(2000, len(X))
    ref_proba = model.predict_proba(test[FEAT31_COLUMNS].iloc[:n_par])
    onnx_proba = proba_from_outputs(sess.run(None, {in_name: X[:n_par]}))
    label_match = float((np.argmax(ref_proba, 1) == np.argmax(onnx_proba, 1)).mean())
    max_dev = float(np.abs(ref_proba - onnx_proba).max())
    print(f"parity: label agreement {label_match:.4f}, max |dp| {max_dev:.2e}")
    if label_match < 0.999:
        raise SystemExit("PARITY FAIL: ONNX model does not reproduce predictions")

    def bench(session):
        for _ in range(50):
            session.run(None, {in_name: X[:1]})
        lat = []
        for i in range(min(2000, len(X))):
            x1 = X[i % len(X):i % len(X) + 1]
            t0 = time.perf_counter()
            session.run(None, {in_name: x1})
            lat.append((time.perf_counter() - t0) * 1000)
        lat = np.array(lat)
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            session.run(None, {in_name: X})
        thr = len(X) * reps / (time.perf_counter() - t0)
        return {"single_sample_latency_ms_median": round(float(np.median(lat)), 4),
                "single_sample_latency_ms_p95": round(float(np.percentile(lat, 95)), 4),
                "batched_throughput_samples_per_sec": int(thr),
                "batch_size_used": len(X)}

    report = {
        "cpu": cpu_name(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "onnxruntime_version": ort.__version__,
        "gpu_dependency": "none - CPUExecutionProvider only",
        "model_file": ONNX_PATH.name,
        "model_size_kib": round(ONNX_PATH.stat().st_size / 1024, 1),
        "parity_label_agreement": label_match,
        "parity_max_proba_deviation": max_dev,
        "fp32": bench(sess),
    }
    (C.REPORTS_DIR / "feat31_onnx_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
