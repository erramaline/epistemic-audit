"""Generic bootstrap confidence intervals used by bootstrap_v2."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    data,
    stat_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """Return bootstrap mean/CI/std for a statistic function.

    Args:
        data: 1D or 2D array-like sample.
        stat_fn: Callable applied to each bootstrap sample.
        n_bootstrap: Number of resamples.
        ci: Confidence interval level.
        seed: RNG seed.

    Returns:
        Dict with mean, lower, upper, std.
    """
    rng = np.random.RandomState(seed)
    arr = np.asarray(data)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    stats = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = arr[idx]
        try:
            stats.append(float(stat_fn(sample)))
        except Exception:
            continue

    if not stats:
        val = float(stat_fn(arr))
        return {"mean": val, "lower": val, "upper": val, "std": 0.0}

    stats = np.asarray(stats, dtype=float)
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(stats)),
        "lower": float(np.percentile(stats, alpha * 100)),
        "upper": float(np.percentile(stats, (1 - alpha) * 100)),
        "std": float(np.std(stats)),
    }


def compute_metric_confidence_intervals(
    correctness: list[bool],
    confidences: list[float],
    n_bootstrap: int = 1000,
) -> dict:
    """Backward-compatible helper returning accuracy and Brier CIs."""
    correct_arr = np.array(correctness, dtype=float)
    conf_arr = np.array(confidences)

    accuracy_ci = bootstrap_ci(correct_arr, np.mean, n_bootstrap)
    brier_data = np.column_stack([conf_arr, correct_arr])
    brier_ci = bootstrap_ci(
        brier_data,
        lambda d: float(np.mean((d[:, 0] - d[:, 1]) ** 2)),
        n_bootstrap,
    )

    return {"accuracy_ci": accuracy_ci, "brier_ci": brier_ci}
