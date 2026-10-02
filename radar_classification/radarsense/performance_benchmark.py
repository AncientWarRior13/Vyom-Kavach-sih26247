"""Performance benchmark: single-sample latency percentiles, batched
throughput, and peak memory for the deployed ONNX inference path -- the
numbers a 'real-time / production' claim needs to be backed by."""

import json
import time
import tracemalloc

import numpy as np

from radarsense import config as C

BATCH_SIZES = [1, 32, 256, 1024, 4096]
N_WARMUP, N_TRIALS = 20, 200


def main():
    import onnxruntime as ort
    sess = ort.InferenceSession(str(C.ONNX_DIR / "radarsense.onnx"),
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(C.SEED)

    tracemalloc.start()

    x1 = rng.standard_normal((1, 11)).astype(np.float32)
    for _ in range(N_WARMUP):
        sess.run(None, {inp: x1})
    lat = []
    for _ in range(N_TRIALS):
        t = time.perf_counter()
        sess.run(None, {inp: x1})
        lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    print("--- single-sample latency (ms) ---")
    print(f"  p50={np.percentile(lat,50):.3f}  p95={np.percentile(lat,95):.3f}  "
          f"p99={np.percentile(lat,99):.3f}")

    print("\n--- batched throughput ---")
    throughput = {}
    for b in BATCH_SIZES:
        xb = rng.standard_normal((b, 11)).astype(np.float32)
        for _ in range(5):
            sess.run(None, {inp: xb})
        t = time.perf_counter()
        reps = 50
        for _ in range(reps):
            sess.run(None, {inp: xb})
        dt = time.perf_counter() - t
        sps = b * reps / dt
        throughput[b] = sps
        print(f"  batch={b:5d}: {sps:>12,.0f} samples/sec ({dt/reps*1000:.3f} ms/batch)")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n--- memory ---\n  peak traced (inference session + buffers): {peak/1e6:.1f} MB")

    out = {
        "single_sample_latency_ms": {"p50": float(np.percentile(lat, 50)),
                                     "p95": float(np.percentile(lat, 95)),
                                     "p99": float(np.percentile(lat, 99))},
        "batched_throughput_samples_per_sec": {str(k): float(v) for k, v in throughput.items()},
        "peak_traced_memory_mb": peak / 1e6,
        "provider": "CPUExecutionProvider",
    }
    (C.REPORTS_DIR / "performance_benchmark_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'performance_benchmark_results.json'}")


if __name__ == "__main__":
    main()
