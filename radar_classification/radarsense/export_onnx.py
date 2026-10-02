"""Stage 7: ONNX export + parity verification + real CPU benchmark.

- Convert the final model to ONNX (skl2onnx / onnxmltools converters).
- Parity: exported model must reproduce original predictions on 2,000
  held-out samples (hard fail otherwise).
- Benchmark: measured single-sample latency (median + p95 over 2,000 runs)
  and batched throughput on this machine's CPU, recorded with the CPU model.
- Int8 dynamic quantization variant benchmarked alongside (edge story).
"""

import json
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
from skl2onnx.common.data_types import FloatTensorType

from radarsense import config as C


def cpu_name():
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    return platform.processor() or platform.machine()


def convert(model):
    """Route to the right converter for the final model's library."""
    initial_types = [("features", FloatTensorType([None, len(C.FEATURE_COLUMNS)]))]
    cls_name = type(model).__name__
    if cls_name == "XGBClassifier":
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes)
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost)
        update_registered_converter(
            type(model), "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes, convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False]})
        return convert_sklearn(model, initial_types=initial_types,
                               options={id(model): {"zipmap": False}},
                               target_opset={"": 15, "ai.onnx.ml": 3})
    if cls_name == "CatBoostClassifier":
        model.save_model(str(C.ONNX_DIR / "radarsense.onnx"), format="onnx")
        return None
    from skl2onnx import convert_sklearn
    return convert_sklearn(model, initial_types=initial_types,
                           options={id(model): {"zipmap": False}},
                           target_opset={"": 15, "ai.onnx.ml": 3})


def main():
    model = joblib.load(C.MODELS_DIR / "final_model.joblib")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")
    X = test[C.FEATURE_COLUMNS].values.astype(np.float32)

    onnx_path = C.ONNX_DIR / "radarsense.onnx"
    onx = convert(model)
    if onx is not None:
        onnx_path.write_bytes(onx.SerializeToString())
    print(f"exported -> {onnx_path} ({onnx_path.stat().st_size/1024:.0f} KiB)")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    def proba_from_outputs(outputs):
        """Normalize ONNX probability output: plain tensor OR catboost's
        seq(map(class,float)) zipmap form."""
        last = outputs[-1]
        if isinstance(last, list) and last and isinstance(last[0], dict):
            return np.array([[row[c] for c in C.CLASSES] for row in last])
        return np.asarray(last)

    n_par = 2000
    ref_proba = model.predict_proba(test[C.FEATURE_COLUMNS].iloc[:n_par])
    onnx_proba = proba_from_outputs(sess.run(None, {in_name: X[:n_par]}))
    if onnx_proba.ndim == 1 or onnx_proba.shape[-1] != len(C.CLASSES):
        raise SystemExit(f"unexpected ONNX output shape {onnx_proba.shape}")
    label_match = float((np.argmax(ref_proba, 1) == np.argmax(onnx_proba, 1)).mean())
    max_dev = float(np.abs(ref_proba - onnx_proba).max())
    print(f"parity: label agreement {label_match:.4f}, max |dp| {max_dev:.2e}")
    if label_match < 0.999:
        raise SystemExit("PARITY FAIL: ONNX model does not reproduce predictions")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    q_path = C.ONNX_DIR / "radarsense_int8.onnx"
    try:
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
        sess_q = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
        q_proba = proba_from_outputs(sess_q.run(None, {in_name: X[:n_par]}))
        q_match = float((np.argmax(ref_proba, 1) == np.argmax(q_proba, 1)).mean())
    except Exception as e:
        sess_q, q_match = None, None
        print(f"int8 quantization skipped: {e}")

    def bench(session, tag):
        for _ in range(50):
            session.run(None, {in_name: X[:1]})
        lat = []
        for i in range(2000):
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
        "model_file": onnx_path.name,
        "model_size_kib": round(onnx_path.stat().st_size / 1024, 1),
        "parity_label_agreement": label_match,
        "parity_max_proba_deviation": max_dev,
        "fp32": bench(sess, "fp32"),
    }
    if sess_q is not None:
        report["int8"] = bench(sess_q, "int8")
        report["int8"]["label_agreement_vs_original"] = q_match
        report["int8"]["model_size_kib"] = round(q_path.stat().st_size / 1024, 1)

    (C.REPORTS_DIR / "onnx_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = report["fp32"]["single_sample_latency_ms_median"] < 10
    print(f"\n<10ms latency target: {'MET' if ok else 'MISSED'}")


if __name__ == "__main__":
    main()
