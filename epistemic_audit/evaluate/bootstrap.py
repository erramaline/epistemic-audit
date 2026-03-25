"""Bootstrap confidence intervals for all benchmark metrics."""

import numpy as np
from typing import Callable


def bootstrap_ci(
    data: list | np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for any statistic.

    Args:
        data: Raw data array.
        statistic_fn: Function that computes the statistic from data.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with 'mean', 'lower', 'upper', 'std'.
    """
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        # Sample indices to correctly handle multidimensional data
        indices = rng.choice(n, size=n, replace=True)
        sample = data[indices]
        boot_stats.append(statistic_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2

    return {
        "mean": float(np.mean(boot_stats)),
        "lower": float(np.percentile(boot_stats, alpha * 100)),
        "upper": float(np.percentile(boot_stats, (1 - alpha) * 100)),
        "std": float(np.std(boot_stats)),
    }


def compute_metric_confidence_intervals(
    correctness: list[bool],
    confidences: list[float],
    n_bootstrap: int = 1000,
) -> dict:
    """Compute 95% CIs for accuracy and Brier score.

    Args:
        correctness: Boolean list of correct/incorrect.
        confidences: Model confidence scores (0-1 scale).
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with 'accuracy_ci' and 'brier_ci', each containing mean/lower/upper.
    """
    correct_arr = np.array(correctness, dtype=float)
    conf_arr = np.array(confidences)

    accuracy_ci = bootstrap_ci(correct_arr, np.mean, n_bootstrap)
    brier_data = np.column_stack([conf_arr, correct_arr])
    brier_ci = bootstrap_ci(
        brier_data,
        lambda d: np.mean((d[:, 0] - d[:, 1]) ** 2),
        n_bootstrap,
    )

    return {"accuracy_ci": accuracy_ci, "brier_ci": brier_ci}
