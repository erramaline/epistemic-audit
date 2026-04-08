"""
Phase 1: accuracy, Brier score, abstention precision/recall, and multi-bin ECE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from epistemic_audit import Question, ModelResponse


FABRICATED_ONLY_SIGNALS = [
    "there is no",
    "no way to know",
    "no such",
    "does not exist",
    "never happened",
    "this is not a real",
    "this doesn't appear to be real",
    "cannot verify",
    "no evidence",
]

UNIVERSAL_ABSTENTION_SIGNALS = [
    "i don't know",
    "cannot answer",
    "not enough information",
    "unanswerable",
    "fabricated",
    "fictional",
    "i'm not sure",
    "cannot determine",
    "i don't have information",
    "not a real",
]


@dataclass
class Phase1Results:
    accuracy: float
    brier_score: float
    ece: float
    abstention_f1: float
    n_questions: int
    n_correct: int
    per_category: dict
    responses: list[ModelResponse]
    correctness: list[bool]
    ece_by_bins: dict[int, float] = field(default_factory=dict)
    # Backward-compatible defaults for callers that only passed abstention_f1.
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0


def _extract_final_number(text: str) -> str:
    """Extract the last standalone number from arithmetic working."""
    nums = re.findall(
        r"(?<![.\d\w])"
        r"(-?\d+(?:\.\d+)?)"
        r"(?:[.\s,;:!?]|$)",
        text,
    )
    return nums[-1] if nums else text.strip()


def parse_phase1_response(raw: str, category: str = "") -> dict:
    """Parse a raw Phase 1 model response into structured fields."""
    answer_match = re.search(r"ANSWER:\s*(.*?)(?=\s*CONFIDENCE:)", raw, re.DOTALL | re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*(-?\d+)", raw, re.IGNORECASE)
    rat_match = re.search(r"RATIONALE:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)

    if answer_match:
        raw_answer = answer_match.group(1).strip()
    else:
        fallback = re.search(r"ANSWER:\s*(.+)", raw, re.IGNORECASE)
        raw_answer = fallback.group(1).strip() if fallback else raw.strip()

    if category == "arithmetic" and len(raw_answer) > 20:
        raw_answer = _extract_final_number(raw_answer)

    return {
        "answer": raw_answer,
        "confidence": max(0, min(100, int(conf_match.group(1)))) if conf_match else 50,
        "rationale": rat_match.group(1).strip() if rat_match else "",
    }


def _compute_ece(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not confidences:
        return 0.0

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = [(c >= bins[i]) and (c <= bins[i + 1]) for c in confidences]
        else:
            mask = [(c >= bins[i]) and (c < bins[i + 1]) for c in confidences]
        count = sum(mask)
        if count == 0:
            continue
        bin_acc = np.mean([cor for cor, m in zip(correctness, mask) if m])
        bin_conf = np.mean([c for c, m in zip(confidences, mask) if m])
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_phase1(questions: list[Question], raw_responses: list[str]) -> Phase1Results:
    assert len(questions) == len(raw_responses)

    responses, correctness, confidences = [], [], []
    per_cat: dict = {}

    for q, raw in zip(questions, raw_responses):
        p = parse_phase1_response(raw, category=q.category)
        responses.append(ModelResponse(q.id, p["answer"], p["confidence"], p["rationale"]))

        is_correct = q.verify(p["answer"])
        correctness.append(is_correct)
        confidences.append(p["confidence"] / 100.0)

        cat = per_cat.setdefault(q.category, {"correct": 0, "total": 0, "brier": []})
        cat["total"] += 1
        if is_correct:
            cat["correct"] += 1
        cat["brier"].append((p["confidence"] / 100.0 - (1.0 if is_correct else 0.0)) ** 2)

    n = len(questions)
    accuracy = sum(correctness) / n if n else 0.0
    brier = float(np.mean([(c - (1.0 if cor else 0.0)) ** 2 for c, cor in zip(confidences, correctness)])) if n else 0.0
    ece = _compute_ece(confidences, correctness)

    tp = fp = fn = 0
    for q, raw in zip(questions, raw_responses):
        is_fabricated = not q.is_answerable
        raw_lower = raw.lower()
        universal_fired = any(s in raw_lower for s in UNIVERSAL_ABSTENTION_SIGNALS)
        fabricated_scope_fired = is_fabricated and any(s in raw_lower for s in FABRICATED_ONLY_SIGNALS)
        model_abstained = universal_fired or fabricated_scope_fired
        if is_fabricated and model_abstained:
            tp += 1
        elif (not is_fabricated) and model_abstained:
            fp += 1
        elif is_fabricated and (not model_abstained):
            fn += 1

    abstention_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    abstention_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = abstention_precision + abstention_recall
    abstention_f1 = 2 * abstention_precision * abstention_recall / denom if denom > 0 else 0.0

    per_cat_final = {
        cat: {
            "accuracy": d["correct"] / d["total"] if d["total"] else 0.0,
            "brier_score": float(np.mean(d["brier"])) if d["brier"] else 0.0,
            "count": d["total"],
        }
        for cat, d in per_cat.items()
    }

    ece_by_bins = {m: _compute_ece(confidences, correctness, n_bins=m) for m in (5, 10, 15, 20)}

    return Phase1Results(
        accuracy=accuracy,
        brier_score=brier,
        ece=ece,
        abstention_f1=abstention_f1,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        n_questions=n,
        n_correct=sum(correctness),
        per_category=per_cat_final,
        responses=responses,
        correctness=correctness,
        ece_by_bins=ece_by_bins,
    )
