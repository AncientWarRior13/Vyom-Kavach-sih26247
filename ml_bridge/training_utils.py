"""Roadmap training-time-cutoff steps 2, 3, and 6 — shared helpers for
whatever training/data-prep script `retrain_async` runs.

Step 6 (wall-clock cutoff) runs the training script as a real OS subprocess
with `subprocess.run(..., timeout=...)`: Python can't forcibly kill another
thread stuck inside a C extension (e.g. a training library's native code),
but the OS can kill a child process. This is the actual enforcement
mechanism; `radar_classifier.retrain_async`'s own thread+Event wait is a
second, more general safety net for non-subprocess train functions.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def train_with_cutoff(script: str, args: list[str] | None = None, budget_s: float = 600.0) -> str:
    """Runs `script` as `python script [args...]`, killed by the OS if it
    exceeds `budget_s`. Returns one of "trained" / "timeout_keep_current_model"
    / "failed_keep_current_model" — never raises, so a caller (e.g.
    `retrain_async`) can always report a clean status instead of crashing.
    """
    try:
        subprocess.run([sys.executable, script, *(args or [])], timeout=budget_s, check=True)
        return "trained"
    except subprocess.TimeoutExpired:
        return "timeout_keep_current_model"
    except subprocess.CalledProcessError:
        return "failed_keep_current_model"


def cached_features(cfg: dict, build_fn, cache_dir: str | Path = "cache"):
    """Roadmap step 2: memoizes `build_fn(cfg)` (expected to return a
    pandas DataFrame) to a parquet file keyed by a hash of `cfg` — unchanged
    config means unchanged data, so it's never regenerated. `pandas` is
    imported lazily so this module has no hard dependency on it for callers
    that only need `train_with_cutoff`.
    """
    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f"feat_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build_fn(cfg)
    df.to_parquet(path)
    return df


def parallel_map(fn, items: list, n_jobs: int = -1, batch_size: int = 64) -> list:
    """Roadmap step 3: parallelizes a per-sample function (e.g. I/Q
    synthesis + CFAR feature extraction, which is CPU-bound and
    independent per sample) across cores via joblib. Falls back to a plain
    serial loop if joblib isn't installed, so callers keep working either
    way — just slower.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        return [fn(item) for item in items]
    return Parallel(n_jobs=n_jobs, batch_size=batch_size)(delayed(fn)(item) for item in items)
