"""Phase 1: accuracy, Brier score, abstention F1."""

import re
import numpy as np
from dataclasses import dataclass
from epistemic_audit import Question, ModelResponse


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


def parse_phase1_response(raw: str) -> dict:
    """Parse a raw Phase 1 model response into structured fields.

    Extracts ANSWER, CONFIDENCE (0-100, clamped), and RATIONALE from the
    response string. Gracefully handles missing or malformed fields by
    returning safe defaults.

    Args:
        raw: Raw model response string.

    Returns:
        Dict with 'answer' (str), 'confidence' (int 0-100), 'rationale' (str).
    """
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|CONFIDENCE:)", raw, re.DOTALL)
    conf_match = re.search(r"CONFIDENCE:\s*(-?\d+)", raw)
    rat_match = re.search(r"RATIONALE:\s*(.+)", raw, re.DOTALL)

    return {
        "answer": answer_match.group(1).strip() if answer_match else raw.strip(),
        "confidence": max(0, min(100, int(conf_match.group(1)))) if conf_match else 50,
        "rationale": rat_match.group(1).strip() if rat_match else "",
    }


def evaluate_phase1(questions: list[Question], raw_responses: list[str]) -> Phase1Results:
    assert len(questions) == len(raw_responses)

    responses, correctness, confidences = [], [], []
    per_cat: dict = {}

    for q, raw in zip(questions, raw_responses):
        p = parse_phase1_response(raw)
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
    accuracy = sum(correctness) / n
    brier = float(np.mean([(c - (1.0 if cor else 0.0)) ** 2 for c, cor in zip(confidences, correctness)]))
    ece = _compute_ece(confidences, correctness)

    # Abstention F1 for fabricated questions
    fab = [(q, cor) for q, cor in zip(questions, correctness) if q.category == "fabricated"]
    if fab:
        tp = sum(1 for _, cor in fab if cor)
        fn = sum(1 for _, cor in fab if not cor)
        # FP: non-fabricated questions where model abstained anyway
        fp = sum(
            1 for q, raw in zip(questions, raw_responses)
            if q.category != "fabricated" and any(s in raw.lower() for s in ["i don't know", "fabricated", "fictional"])
        )
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        abstention_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    else:
        abstention_f1 = 0.0

    per_cat_final = {
        cat: {"accuracy": d["correct"] / d["total"], "brier_score": float(np.mean(d["brier"])), "count": d["total"]}
        for cat, d in per_cat.items()
    }

    return Phase1Results(
        accuracy=accuracy, brier_score=brier, ece=ece, abstention_f1=abstention_f1,
        n_questions=n, n_correct=sum(correctness),
        per_category=per_cat_final, responses=responses, correctness=correctness,
    )


def _compute_ece(confidences: list[float], correctness: list[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    ECE measures the gap between predicted confidence and actual accuracy
    across binned confidence ranges. Lower = better calibrated.

    Args:
        confidences: Model confidence scores (0-1 scale).
        correctness: Boolean correct/incorrect per question.
        n_bins: Number of confidence bins.

    Returns:
        ECE score (0-1, lower is better).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        # Use <= for the last bin to include confidence 1.0
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
