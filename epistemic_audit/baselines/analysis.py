"""Human baseline analysis: load, score, and compare participant responses."""

from __future__ import annotations

import json
import statistics
from typing import Any


def load_human_responses(json_path: str) -> list[dict]:
    """Load human participant responses from the form JSON output.

    Args:
        json_path: Path to the JSON file produced by the human_form.html
            copy-paste step.

    Returns:
        List of dicts, each with keys: id, category, answer, confidence, rationale.
    """
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def compute_human_baseline(responses: list[dict], answer_key: list[dict]) -> dict:
    """Compute accuracy, Brier score, and per-category breakdown for one participant.

    The answer_key list must contain at least the same 'id' and 'correct_answer'
    fields as exported by generate_human_form.py.

    Args:
        responses: List of response dicts from load_human_responses.
        answer_key: List of ground-truth dicts loaded from answer_key.json.

    Returns:
        Dict with keys: 'accuracy', 'brier_score', 'per_category',
        'n_questions', 'n_correct'.
    """
    key_map: dict[str, dict] = {item["id"]: item for item in answer_key}

    correct_flags: list[bool] = []
    brier_terms: list[float] = []
    per_cat: dict[str, dict[str, Any]] = {}

    for resp in responses:
        qid = resp["id"]
        if qid not in key_map:
            continue

        key = key_map[qid]
        model_answer = str(resp.get("answer", "")).strip().lower()
        correct_answer = str(key.get("correct_answer", "")).strip().lower()
        confidence = max(0, min(100, int(resp.get("confidence", 50)))) / 100.0

        is_answerable = key.get("is_answerable", True)
        if is_answerable:
            is_correct = correct_answer in model_answer or model_answer in correct_answer
        else:
            refusal_signals = [
                "i don't know", "cannot answer", "not enough information",
                "unanswerable", "fabricated", "fictional",
            ]
            is_correct = any(s in model_answer for s in refusal_signals)

        correct_flags.append(is_correct)
        brier_terms.append((confidence - (1.0 if is_correct else 0.0)) ** 2)

        cat = resp.get("category", "unknown")
        entry = per_cat.setdefault(cat, {"correct": 0, "total": 0, "brier": []})
        entry["total"] += 1
        if is_correct:
            entry["correct"] += 1
        entry["brier"].append(brier_terms[-1])

    n = len(correct_flags)
    accuracy = sum(correct_flags) / n if n > 0 else 0.0
    brier_score = statistics.mean(brier_terms) if brier_terms else 0.0

    per_cat_final = {
        cat: {
            "accuracy": d["correct"] / d["total"],
            "brier_score": statistics.mean(d["brier"]),
            "count": d["total"],
        }
        for cat, d in per_cat.items()
    }

    return {
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier_score, 4),
        "per_category": per_cat_final,
        "n_questions": n,
        "n_correct": sum(correct_flags),
    }


def aggregate_human_baselines(
    all_responses: list[list[dict]], answer_key: list[dict]
) -> dict:
    """Aggregate metrics across multiple human participants.

    Args:
        all_responses: List of per-participant response lists.
        answer_key: Ground-truth list from answer_key.json.

    Returns:
        Dict with median, IQR, and mean for accuracy and brier_score
        across all participants.
    """
    baselines = [compute_human_baseline(r, answer_key) for r in all_responses]
    accuracies = [b["accuracy"] for b in baselines]
    briers = [b["brier_score"] for b in baselines]

    def _iqr(values: list[float]) -> float:
        sorted_v = sorted(values)
        n = len(sorted_v)
        q1 = sorted_v[n // 4]
        q3 = sorted_v[(3 * n) // 4]
        return round(q3 - q1, 4)

    return {
        "n_participants": len(baselines),
        "accuracy": {
            "mean": round(statistics.mean(accuracies), 4),
            "median": round(statistics.median(accuracies), 4),
            "iqr": _iqr(accuracies),
        },
        "brier_score": {
            "mean": round(statistics.mean(briers), 4),
            "median": round(statistics.median(briers), 4),
            "iqr": _iqr(briers),
        },
        "individual_baselines": baselines,
    }


def compare_model_to_humans(model_profile: dict, human_aggregate: dict) -> dict:
    """Produce a comparison dict showing model vs. human aggregate on each metric.

    Args:
        model_profile: Dict from EpistemicProfile.to_dict().
        human_aggregate: Dict from aggregate_human_baselines().

    Returns:
        Comparison dict with model value, human median, and delta for each metric.
    """
    model_accuracy = model_profile.get("phase1", {}).get("accuracy", None)
    model_brier = model_profile.get("phase1", {}).get("brier_score", None)
    human_acc = human_aggregate.get("accuracy", {})
    human_brier = human_aggregate.get("brier_score", {})

    def _delta(model_val, human_median):
        if model_val is None or human_median is None:
            return None
        return round(model_val - human_median, 4)

    return {
        "accuracy": {
            "model": model_accuracy,
            "human_median": human_acc.get("median"),
            "human_iqr": human_acc.get("iqr"),
            "delta": _delta(model_accuracy, human_acc.get("median")),
        },
        "brier_score": {
            "model": model_brier,
            "human_median": human_brier.get("median"),
            "human_iqr": human_brier.get("iqr"),
            "delta_lower_is_better": _delta(model_brier, human_brier.get("median")),
        },
        "composite_score": {
            "model": model_profile.get("composite_score"),
            "human_baseline": "N/A (3-phase form not yet administered)",
        },
    }
