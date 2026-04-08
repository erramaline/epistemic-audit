"""Calibration curves, radar charts, and multi-model comparison plots."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color palette for multi-model overlays
_MODEL_COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]


# ─────────────────────────────────────────────────────────────────────────────
# Single-model charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curve(
    confidences: list[float],
    correctness: list[bool],
    model_name: str = "Model",
    save_path: str = "calibration_curve.png",
) -> None:
    """Plot a calibration curve (reliability diagram) for one model.

    Args:
        confidences: List of confidence values in [0, 1].
        correctness: List of booleans for each item.
        model_name: Display name for the legend.
        save_path: Output file path.
    """
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(len(bins) - 1):
        mask = [(b >= bins[i]) and (b < bins[i+1]) for b in confidences]
        if sum(mask) > 0:
            bin_accs.append(np.mean([c for c, m in zip(correctness, mask) if m]))
            bin_confs.append(np.mean([b for b, m in zip(confidences, mask) if m]))
            bin_counts.append(sum(mask))
        else:
            bin_accs.append(None); bin_confs.append(None); bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

    valid = [(c, a) for c, a in zip(bin_confs, bin_accs) if c is not None]
    if valid:
        confs, accs = zip(*valid)
        ax1.plot(confs, accs, "o-", color=_MODEL_COLORS[0], linewidth=2, markersize=8, label=model_name)
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence"); ax1.set_ylabel("Actual Accuracy")
    ax1.set_title("Calibration Curve"); ax1.legend(); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    ax2.bar(bin_centers, bin_counts, width=0.08, color=_MODEL_COLORS[0], alpha=0.7)
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count"); ax2.set_title("Confidence Distribution")

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_radar_chart(
    profile_dict: dict,
    model_name: str = "Model",
    save_path: str = "radar_chart.png",
) -> None:
    """Plot a single-model radar/spider chart of all epistemic metrics.

    Args:
        profile_dict: Dict from EpistemicProfile.to_dict().
        model_name: Display name for the title.
        save_path: Output file path.
    """
    metrics = {
        "Accuracy": profile_dict["phase1"]["accuracy"],
        "Calibration\n(1-Brier)": 1 - profile_dict["phase1"]["brier_score"],
        "Abstention\nF1": profile_dict["phase1"]["abstention_f1"],
        "Audit\nAUROC": profile_dict["phase2"]["audit_auroc"],
        "Planted\nDetection": profile_dict["phase2"]["planted_detection_rate"],
        "Hold Rate": profile_dict["phase3"]["appropriate_hold_rate"],
        "Revise Rate": profile_dict["phase3"]["appropriate_revise_rate"],
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    n = len(labels)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color=_MODEL_COLORS[0], alpha=0.25)
    ax.plot(angles, values, "o-", color=_MODEL_COLORS[0], linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1); ax.set_title(f"Epistemic Profile: {model_name}", fontsize=14, pad=20)

    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison charts
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metric_vector(profile: dict) -> dict[str, float]:
    """Extract a flat metric dict from a profile dict.

    Args:
        profile: Dict from EpistemicProfile.to_dict().

    Returns:
        Flat dict of metric_name → float.
    """
    return {
        "Accuracy":      profile["phase1"]["accuracy"],
        "1-Brier":       1 - profile["phase1"]["brier_score"],
        "Abstention F1": profile["phase1"]["abstention_f1"],
        "Audit AUROC":   profile["phase2"]["audit_auroc"],
        "Planted Det.":  profile["phase2"]["planted_detection_rate"],
        "Hold Rate":     profile["phase3"]["appropriate_hold_rate"],
        "Revise Rate":   profile["phase3"]["appropriate_revise_rate"],
    }


def plot_multi_model_radar(
    profiles: dict[str, dict],
    save_path: str = "multi_model_radar.png",
) -> None:
    """Overlay multiple models on a single radar chart with a legend.

    Args:
        profiles: Dict mapping model_name → profile dict.
        save_path: Output file path.
    """
    metric_labels = ["Accuracy", "1-Brier", "Abstention F1",
                     "Audit AUROC", "Planted Det.", "Hold Rate", "Revise Rate"]
    n = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_xticks(angles); ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)

    for idx, (model_name, profile) in enumerate(profiles.items()):
        mv = _extract_metric_vector(profile)
        values = [mv[m] for m in metric_labels] + [mv[metric_labels[0]]]
        color = _MODEL_COLORS[idx % len(_MODEL_COLORS)]
        ax.fill(angles_closed, values, color=color, alpha=0.15)
        ax.plot(angles_closed, values, "o-", color=color, linewidth=2, label=model_name)

    ax.set_title("Multi-Model Epistemic Profile", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_multi_model_comparison_bar(
    profiles: dict[str, dict],
    save_path: str = "multi_model_bar.png",
) -> None:
    """Grouped bar chart comparing all models across key metrics.

    Args:
        profiles: Dict mapping model_name → profile dict.
        save_path: Output file path.
    """
    metric_labels = ["Accuracy", "1-Brier", "Audit AUROC",
                     "Hold Rate", "Revise Rate", "Planted Det."]
    models = list(profiles.keys())
    x = np.arange(len(metric_labels))
    bar_width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, model_name in enumerate(models):
        mv = _extract_metric_vector(profiles[model_name])
        vals = [mv[m] for m in metric_labels]
        offset = (idx - len(models) / 2 + 0.5) * bar_width
        color = _MODEL_COLORS[idx % len(_MODEL_COLORS)]
        bars = ax.bar(x + offset, vals, bar_width * 0.9, label=model_name, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Epistemic Audit Metrics")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_category_heatmap(
    profiles: dict[str, dict],
    save_path: str = "category_heatmap.png",
) -> None:
    """Heatmap of per-category accuracy broken down by model.

    Args:
        profiles: Dict mapping model_name → profile dict (must include per_category).
        save_path: Output file path.
    """
    all_cats: list[str] = []
    for profile in profiles.values():
        for cat in profile.get("per_category", {}).keys():
            if cat not in all_cats:
                all_cats.append(cat)

    models = list(profiles.keys())
    data = np.zeros((len(models), len(all_cats)))
    for i, model in enumerate(models):
        per_cat = profiles[model].get("per_category", {})
        for j, cat in enumerate(all_cats):
            data[i, j] = per_cat.get(cat, {}).get("accuracy", float("nan"))

    fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 1.5), max(4, len(models) * 1.2)))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(all_cats))); ax.set_xticklabels(all_cats, rotation=30, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    ax.set_title("Per-Category Accuracy by Model")
    for i in range(len(models)):
        for j in range(len(all_cats)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.8 else "white")

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_sycophancy_comparison(
    profiles: dict[str, dict],
    save_path: str = "sycophancy_comparison.png",
) -> None:
    """Horizontal bar chart of sycophancy index per model.

    Lower is better. A value of 0 means the model never flips on correct answers.

    Args:
        profiles: Dict mapping model_name → profile dict.
        save_path: Output file path.
    """
    models = list(profiles.keys())
    syco_vals = [profiles[m]["phase3"]["sycophancy_index"] for m in models]
    colors = [_MODEL_COLORS[i % len(_MODEL_COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(8, max(3, len(models) * 0.7)))
    bars = ax.barh(models, syco_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, syco_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Sycophancy Index (lower = better)")
    ax.set_title("Sycophancy Index by Model\n(0 = never flips correct answers, 1 = always flips)")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance baseline")
    ax.legend(); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")
