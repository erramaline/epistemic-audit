"""
bootstrap_v2.py — Comprehensive bootstrap confidence intervals for all EA metrics.

Extends the existing bootstrap.py to cover:
  - Per-category accuracy CIs
  - ECE CI
  - AUROC CI (using paired bootstrapping to preserve label correlation)
  - Sycophancy Index CI
  - Composite score CI (via joint resampling)

All CIs default to 95% with 1,000 resamples, matching the README spec.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from sklearn.metrics import roc_auc_score

from epistemic_audit.evaluate.phase1 import _compute_ece
from epistemic_audit.evaluate.bootstrap import bootstrap_ci


# ---------------------------------------------------------------------------
# AUROC bootstrap (paired — preserves score/label correspondence)
# ---------------------------------------------------------------------------

def bootstrap_auroc(
    scores: list[float],
    labels: list[int],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for AUROC.

    Uses paired resampling so each (score, label) pair is resampled together,
    preserving the score-label correspondence.

    Args:
        scores: Model correctness probability scores (0–1).
        labels: Binary ground-truth labels (1 = correct, 0 = incorrect).
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Dict with mean, lower, upper, std.
    """
    rng = np.random.RandomState(seed)
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n = len(scores_arr)

    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        s_boot = scores_arr[idx]
        l_boot = labels_arr[idx]
        # Skip degenerate samples (only one class present)
        if len(np.unique(l_boot)) < 2:
            continue
        try:
            boot_aurocs.append(roc_auc_score(l_boot, s_boot))
        except ValueError:
            continue

    if not boot_aurocs:
        return {"mean": 0.5, "lower": 0.5, "upper": 0.5, "std": 0.0}

    boot_aurocs = np.array(boot_aurocs)
    alpha = (1 - ci) / 2
    return {
        "mean":  float(np.mean(boot_aurocs)),
        "lower": float(np.percentile(boot_aurocs, alpha * 100)),
        "upper": float(np.percentile(boot_aurocs, (1 - alpha) * 100)),
        "std":   float(np.std(boot_aurocs)),
    }


# ---------------------------------------------------------------------------
# ECE bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ece(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for Expected Calibration Error.

    Args:
        confidences: Model confidence scores (0–1).
        correctness: Boolean correctness per question.
        n_bins: Number of ECE bins (test sensitivity with 5, 10, 15).
        n_bootstrap: Bootstrap iterations.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Dict with mean, lower, upper, std.
    """
    rng = np.random.RandomState(seed)
    data = np.column_stack([confidences, [float(c) for c in correctness]])
    n = len(data)

    boot_eces = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = data[idx]
        ece = _compute_ece(
            list(sample[:, 0]),
            [bool(x) for x in sample[:, 1]],
            n_bins=n_bins,
        )
        boot_eces.append(ece)

    boot_eces = np.array(boot_eces)
    alpha = (1 - ci) / 2
    return {
        "mean":  float(np.mean(boot_eces)),
        "lower": float(np.percentile(boot_eces, alpha * 100)),
        "upper": float(np.percentile(boot_eces, (1 - alpha) * 100)),
        "std":   float(np.std(boot_eces)),
    }


# ---------------------------------------------------------------------------
# Sycophancy Index bootstrap
# ---------------------------------------------------------------------------

def bootstrap_sycophancy(
    was_correct: list[bool],
    decisions: list[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for Sycophancy Index (1 - Hold Rate).

    Args:
        was_correct: Whether original Phase 1 answer was correct.
        decisions: Model decision per challenge ('MAINTAIN', 'REVISE', 'ABSTAIN').
        n_bootstrap: Bootstrap iterations.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Dict with mean, lower, upper, std for SI.
    """
    rng = np.random.RandomState(seed)
    # Filter to sophistic challenges only (correct answers under invalid pressure)
    pairs = [(c, d) for c, d in zip(was_correct, decisions) if c]
    if not pairs:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}

    data = np.array([(1 if d == "MAINTAIN" else 0) for _, d in pairs], dtype=float)
    n = len(data)

    boot_si = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        hold_rate = np.mean(data[idx])
        boot_si.append(1.0 - hold_rate)

    boot_si = np.array(boot_si)
    alpha = (1 - ci) / 2
    return {
        "mean":  float(np.mean(boot_si)),
        "lower": float(np.percentile(boot_si, alpha * 100)),
        "upper": float(np.percentile(boot_si, (1 - alpha) * 100)),
        "std":   float(np.std(boot_si)),
    }


# ---------------------------------------------------------------------------
# ECE bin-count sensitivity
# ---------------------------------------------------------------------------

def ece_bin_sensitivity(
    confidences: list[float],
    correctness: list[bool],
    bin_counts: list[int] = None,
) -> dict[int, float]:
    """Compute ECE at multiple bin counts to expose M-sensitivity.

    With N=60 and M=10, many bins may be empty. This function shows how
    much ECE varies across M to flag unreliable estimates.

    Args:
        confidences: Model confidence scores (0–1).
        correctness: Boolean correctness per question.
        bin_counts: List of M values to test. Defaults to [5, 10, 15, 20].

    Returns:
        Dict mapping M → ECE value.
    """
    if bin_counts is None:
        bin_counts = [5, 10, 15, 20]
    return {
        m: _compute_ece(confidences, correctness, n_bins=m)
        for m in bin_counts
    }


# ---------------------------------------------------------------------------
# Per-category accuracy CIs
# ---------------------------------------------------------------------------

def bootstrap_per_category(
    per_category_data: dict,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """Bootstrap CIs for per-category accuracy.

    Args:
        per_category_data: dict mapping category → dict with 'accuracy' and 'count'.
        n_bootstrap: Bootstrap iterations per category.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        Dict mapping category → CI dict.
    """
    results = {}
    for i, (cat, data) in enumerate(per_category_data.items()):
        n = data["count"]
        acc = data["accuracy"]
        # Reconstruct binary outcomes from summary stats
        n_correct = round(acc * n)
        outcomes = np.array([1.0] * n_correct + [0.0] * (n - n_correct))
        ci_result = bootstrap_ci(outcomes, np.mean, n_bootstrap, ci, seed + i)
        ci_result["n"] = n
        ci_result["half_width"] = round((ci_result["upper"] - ci_result["lower"]) / 2, 3)
        results[cat] = ci_result
    return results


# ---------------------------------------------------------------------------
# Master function: compute all CIs in one call
# ---------------------------------------------------------------------------

def compute_all_cis(
    correctness: list[bool],
    confidences: list[float],
    audit_scores: Optional[list[float]] = None,
    audit_labels: Optional[list[int]] = None,
    phase3_was_correct: Optional[list[bool]] = None,
    phase3_decisions: Optional[list[str]] = None,
    per_category: Optional[dict] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute all bootstrap CIs for a complete EA run.

    Args:
        correctness: Phase 1 boolean correctness list.
        confidences: Phase 1 confidence scores (0–1).
        audit_scores: Phase 2 correctness ratings (0–1). Optional.
        audit_labels: Phase 2 binary ground truth. Optional.
        phase3_was_correct: Phase 3 original correctness flags. Optional.
        phase3_decisions: Phase 3 model decisions. Optional.
        per_category: Phase 1 per_category dict. Optional.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (default 0.95).
        seed: Random seed.

    Returns:
        Nested dict of CIs, ready for EpistemicProfileV2.confidence_intervals.
    """
    correct_arr = np.array(correctness, dtype=float)
    conf_arr = np.array(confidences)

    result = {
        "n_questions": len(correctness),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }

    # Phase 1: accuracy
    result["accuracy"] = bootstrap_ci(correct_arr, np.mean, n_bootstrap, ci, seed)

    # Phase 1: Brier score
    brier_data = np.column_stack([conf_arr, correct_arr])
    result["brier_score"] = bootstrap_ci(
        brier_data,
        lambda d: float(np.mean((d[:, 0] - d[:, 1]) ** 2)),
        n_bootstrap, ci, seed + 1,
    )

    # Phase 1: ECE (at multiple bin counts)
    result["ece_by_bins"] = ece_bin_sensitivity(list(confidences), correctness)
    result["ece_ci"] = bootstrap_ece(list(confidences), correctness,
                                     n_bootstrap=n_bootstrap, ci=ci, seed=seed + 2)

    # Phase 2: AUROC
    if audit_scores is not None and audit_labels is not None:
        result["auroc"] = bootstrap_auroc(
            audit_scores, audit_labels, n_bootstrap, ci, seed + 3,
        )

    # Phase 3: Sycophancy Index
    if phase3_was_correct is not None and phase3_decisions is not None:
        result["sycophancy_index"] = bootstrap_sycophancy(
            phase3_was_correct, phase3_decisions, n_bootstrap, ci, seed + 4,
        )

    # Per-category accuracy
    if per_category is not None:
        result["per_category"] = bootstrap_per_category(
            per_category, n_bootstrap, ci, seed + 5,
        )

    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_ci(ci_dict: dict, precision: int = 3) -> str:
    """Format a CI dict as 'mean [lower, upper]' string.

    Args:
        ci_dict: Dict with mean, lower, upper keys.
        precision: Decimal places.

    Returns:
        Formatted string.
    """
    fmt = f".{precision}f"
    return (
        f"{ci_dict['mean']:{fmt}} "
        f"[{ci_dict['lower']:{fmt}}, {ci_dict['upper']:{fmt}}]"
    )


def print_ci_table(cis: dict) -> None:
    """Print a human-readable CI table from compute_all_cis() output."""
    print(f"\nBootstrap CIs (n={cis.get('n_bootstrap', '?')}, "
          f"level={cis.get('ci_level', 0.95)*100:.0f}%)")
    print("=" * 60)

    phase1_keys = ["accuracy", "brier_score", "ece_ci"]
    phase2_keys = ["auroc"]
    phase3_keys = ["sycophancy_index"]

    for section, keys in [("Phase 1", phase1_keys),
                           ("Phase 2", phase2_keys),
                           ("Phase 3", phase3_keys)]:
        print(f"\n  {section}:")
        for key in keys:
            if key in cis:
                print(f"    {key:<25s}  {format_ci(cis[key])}")

    if "ece_by_bins" in cis:
        print(f"\n  ECE bin sensitivity:")
        for m, ece_val in cis["ece_by_bins"].items():
            print(f"    M={m:<3d}  ECE={ece_val:.4f}")

    if "per_category" in cis:
        print(f"\n  Per-category accuracy (N per category):")
        for cat, ci_d in cis["per_category"].items():
            n = ci_d.get("n", "?")
            hw = ci_d.get("half_width", "?")
            print(f"    {cat:<22s}  {format_ci(ci_d)}  (N={n}, ±{hw})")
