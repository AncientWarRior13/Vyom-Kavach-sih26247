"""One-command reproducibility: regenerates every number and figure in the report.

    python run_pipeline.py            # full pipeline
    python run_pipeline.py --from 4   # resume from stage index 4

Stages run in dependency order; each writes its artifacts under artifacts/.
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    ("simulator",       "generate dataset + splits"),
    ("validate_schema", "schema contract gate"),
    ("data_quality",    "QC checks + plots"),
    ("train_baseline",  "6-model bake-off, 5-fold CV"),
    ("optimize",        "minority-class optimization + thresholds"),
    ("calibrate",       "isotonic calibration + reliability"),
    ("robustness",      "SNR curves + stress tests"),
    ("quantum",         "quantum kernel ablation + scaling"),
    ("explain",         "SHAP summary + waterfall"),
    ("export_onnx",     "ONNX export + CPU benchmark"),
    ("final_report",    "assemble report + headline figure"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    args = ap.parse_args()
    t_all = time.time()
    for i, (mod, desc) in enumerate(STAGES[args.start:], start=args.start):
        print(f"\n=== [{i}] {mod}: {desc} " + "=" * max(1, 40 - len(mod)))
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", f"radarsense.{mod}"])
        if r.returncode != 0:
            print(f"stage {mod} FAILED (exit {r.returncode})")
            sys.exit(r.returncode)
        print(f"[{i}] {mod} done in {time.time()-t0:.0f}s")
    print(f"\npipeline complete in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
