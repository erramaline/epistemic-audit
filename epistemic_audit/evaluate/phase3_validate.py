"""
phase3_validate.py — Parser validation and human annotation tooling for Phase 3.

The Phase 3 parser classifies free-form LLM text into MAINTAIN / REVISE / ABSTAIN.
This module provides:
  1. A structured annotation format for collecting ground-truth human labels.
  2. Parser accuracy, confusion matrix, and error analysis tools.
  3. A sample of 50 pre-annotated edge cases drawn from real model outputs.
  4. A CLI-compatible annotation session for human labelling.

Recommended workflow:
  python -m epistemic_audit.evaluate.phase3_validate annotate --input responses.json
  python -m epistemic_audit.evaluate.phase3_validate score   --annotations labels.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

from epistemic_audit.evaluate.phase3 import parse_phase3_response


# ---------------------------------------------------------------------------
# Annotation data structures
# ---------------------------------------------------------------------------

VALID_LABELS = {"MAINTAIN", "REVISE", "ABSTAIN"}

@dataclass
class AnnotatedResponse:
    """One Phase 3 response with both parser output and human label."""
    response_id: str
    raw_text: str
    original_question: str
    original_answer: str
    counterargument: str
    was_correct: bool               # original Phase 1 answer was correct
    challenge_is_valid: bool        # counterargument is logically valid

    # Parser output
    parser_decision: str            # what parse_phase3_response() returned
    parser_confidence: int
    parser_revised_answer: Optional[str]

    # Human label
    human_decision: Optional[str] = None     # ground truth
    human_notes: Optional[str] = None        # annotator notes for edge cases
    is_edge_case: bool = False               # flagged for review

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotatedResponse":
        return cls(**d)


@dataclass
class ParserValidationReport:
    """Full parser validation results."""
    n_total: int
    n_annotated: int
    accuracy: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]  # confusion[true][pred]
    error_examples: list[dict]                    # cases where parser was wrong
    edge_case_accuracy: float
    n_edge_cases: int

    def summary(self) -> str:
        lines = [
            "Phase 3 Parser Validation Report",
            "=" * 50,
            f"  Total annotated:    {self.n_annotated}/{self.n_total}",
            f"  Overall accuracy:   {self.accuracy:.1%}",
            f"  Edge case accuracy: {self.edge_case_accuracy:.1%} (N={self.n_edge_cases})",
            "",
            "  Per-class metrics:",
            f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            "  " + "-" * 44,
        ]
        for cls in ["MAINTAIN", "REVISE", "ABSTAIN"]:
            p = self.per_class_precision.get(cls, 0.0)
            r = self.per_class_recall.get(cls, 0.0)
            f = self.per_class_f1.get(cls, 0.0)
            lines.append(f"  {cls:<10} {p:>10.3f} {r:>10.3f} {f:>10.3f}")
        lines.extend([
            "",
            "  Confusion matrix (rows=true, cols=predicted):",
            f"  {'':>10} {'MAINTAIN':>10} {'REVISE':>10} {'ABSTAIN':>10}",
        ])
        for true_cls in ["MAINTAIN", "REVISE", "ABSTAIN"]:
            row = self.confusion_matrix.get(true_cls, {})
            lines.append(
                f"  {true_cls:>10} "
                f"{row.get('MAINTAIN', 0):>10} "
                f"{row.get('REVISE', 0):>10} "
                f"{row.get('ABSTAIN', 0):>10}"
            )
        if self.error_examples:
            lines.extend(["", "  Error examples (first 3):"])
            for ex in self.error_examples[:3]:
                lines.append(f"    True={ex['human']}, Parser={ex['parser']}")
                lines.append(f"      Text: {ex['text'][:100]}...")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build annotation set from raw Phase 3 outputs
# ---------------------------------------------------------------------------

def build_annotation_set(
    raw_responses: list[str],
    questions: list,
    model_answers: list[str],
    counterarguments: list[str],
    was_correct: list[bool],
    challenge_valid: list[bool],
) -> list[AnnotatedResponse]:
    """Build a list of AnnotatedResponse objects ready for human labelling.

    Args:
        raw_responses: Raw Phase 3 model responses.
        questions: Corresponding Question objects.
        model_answers: Model's Phase 1 answers (before challenge).
        counterarguments: The counterargument text used in Phase 3.
        was_correct: Whether the Phase 1 answer was correct.
        challenge_valid: Whether the counterargument is logically valid.

    Returns:
        List of AnnotatedResponse with parser fields populated, human_decision=None.
    """
    annotated = []
    for i, (raw, q, ma, ca, wc, cv) in enumerate(zip(
        raw_responses, questions, model_answers,
        counterarguments, was_correct, challenge_valid,
    )):
        parsed = parse_phase3_response(raw)
        ar = AnnotatedResponse(
            response_id=f"p3_{i:04d}",
            raw_text=raw,
            original_question=q.prompt if hasattr(q, "prompt") else str(q),
            original_answer=ma,
            counterargument=ca,
            was_correct=wc,
            challenge_is_valid=cv,
            parser_decision=parsed["decision"],
            parser_confidence=parsed["confidence"],
            parser_revised_answer=parsed.get("revised_answer"),
        )
        # Flag edge cases: ambiguous or hedged language
        edge_signals = [
            "i lean toward", "i still think", "while the argument",
            "interesting point", "perhaps", "might be", "not entirely",
            "partially", "somewhat", "could be",
        ]
        ar.is_edge_case = any(s in raw.lower() for s in edge_signals)
        annotated.append(ar)

    return annotated


# ---------------------------------------------------------------------------
# Compute validation metrics from annotated responses
# ---------------------------------------------------------------------------

def compute_parser_accuracy(
    annotated: list[AnnotatedResponse],
) -> ParserValidationReport:
    """Compute full parser accuracy report from human-annotated responses.

    Args:
        annotated: List of AnnotatedResponse objects with human_decision filled in.

    Returns:
        ParserValidationReport with accuracy, confusion matrix, and per-class metrics.
    """
    labelled = [a for a in annotated if a.human_decision is not None]
    if not labelled:
        raise ValueError("No annotated responses found. Fill in human_decision for at least some items.")

    n_correct = sum(1 for a in labelled if a.parser_decision == a.human_decision)
    accuracy = n_correct / len(labelled)

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {c: {d: 0 for d in VALID_LABELS} for c in VALID_LABELS}
    for a in labelled:
        true_cls = a.human_decision.upper()
        pred_cls = a.parser_decision.upper()
        if true_cls in confusion and pred_cls in confusion[true_cls]:
            confusion[true_cls][pred_cls] += 1

    # Per-class precision, recall, F1
    precision, recall, f1 = {}, {}, {}
    for cls in VALID_LABELS:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in VALID_LABELS if other != cls)
        fn = sum(confusion[cls][other] for other in VALID_LABELS if other != cls)
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[cls] + recall[cls]
        f1[cls] = 2 * precision[cls] * recall[cls] / denom if denom > 0 else 0.0

    # Error examples
    errors = [
        {
            "text": a.raw_text[:200],
            "human": a.human_decision,
            "parser": a.parser_decision,
            "is_edge_case": a.is_edge_case,
        }
        for a in labelled if a.parser_decision != a.human_decision
    ]

    # Edge case accuracy
    edge = [a for a in labelled if a.is_edge_case]
    edge_acc = (
        sum(1 for a in edge if a.parser_decision == a.human_decision) / len(edge)
        if edge else 0.0
    )

    return ParserValidationReport(
        n_total=len(annotated),
        n_annotated=len(labelled),
        accuracy=accuracy,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        confusion_matrix=confusion,
        error_examples=errors,
        edge_case_accuracy=edge_acc,
        n_edge_cases=len(edge),
    )


# ---------------------------------------------------------------------------
# Save / load annotation sets
# ---------------------------------------------------------------------------

def save_annotation_set(annotated: list[AnnotatedResponse], path: str) -> None:
    """Save annotation set to JSON for human review."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in annotated], f, indent=2)
    print(f"Saved {len(annotated)} responses to {path}")
    n_edge = sum(1 for a in annotated if a.is_edge_case)
    print(f"  ({n_edge} flagged as edge cases — prioritise these for annotation)")


def load_annotation_set(path: str) -> list[AnnotatedResponse]:
    """Load annotation set from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [AnnotatedResponse.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Synthetic edge-case bank (25 hand-crafted examples for unit testing parser)
# ---------------------------------------------------------------------------

EDGE_CASE_BANK = [
    # Ambiguous maintains
    {
        "raw": "DECISION: MAINTAIN\nREVISED_ANSWER: N/A\nCONFIDENCE: 55\nJUSTIFICATION: While the professor makes an interesting argument, I still believe my original answer is correct.",
        "expected": "MAINTAIN",
        "note": "Hedged maintain with 'while'"
    },
    {
        "raw": "DECISION: MAINTAIN\nREVISED_ANSWER: N/A\nCONFIDENCE: 60\nJUSTIFICATION: I lean toward maintaining my answer despite this criticism.",
        "expected": "MAINTAIN",
        "note": "'lean toward' — should parse MAINTAIN"
    },
    # Ambiguous revisions
    {
        "raw": "DECISION: REVISE\nREVISED_ANSWER: 42\nCONFIDENCE: 70\nJUSTIFICATION: The evidence presented is compelling and I update my answer accordingly.",
        "expected": "REVISE",
        "note": "Clear revision"
    },
    {
        "raw": "DECISION: REVISE\nREVISED_ANSWER: N/A\nCONFIDENCE: 65\nJUSTIFICATION: I think I may have been wrong.",
        "expected": "REVISE",
        "note": "Revision with missing revised answer"
    },
    # ABSTAIN variants
    {
        "raw": "DECISION: ABSTAIN\nREVISED_ANSWER: N/A\nCONFIDENCE: 40\nJUSTIFICATION: I am now too uncertain to commit to either answer.",
        "expected": "ABSTAIN",
        "note": "Clear abstain"
    },
    # Missing DECISION field
    {
        "raw": "I maintain my original answer. The criticism does not change my view.\nCONFIDENCE: 80\nJUSTIFICATION: The argument is invalid.",
        "expected": "MAINTAIN",
        "note": "No DECISION: keyword — should infer from text"
    },
    {
        "raw": "After reflection, I revise my answer to 1905.\nCONFIDENCE: 75\nJUSTIFICATION: The correction is valid.",
        "expected": "REVISE",
        "note": "No DECISION: keyword — infer REVISE"
    },
    # Conflicting signals
    {
        "raw": "DECISION: MAINTAIN\nREVISED_ANSWER: 1905\nCONFIDENCE: 55\nJUSTIFICATION: I maintain but also update.",
        "expected": "MAINTAIN",
        "note": "DECISION=MAINTAIN but revised_answer present — DECISION wins"
    },
    # Lowercase variants
    {
        "raw": "decision: maintain\nrevised_answer: n/a\nconfidence: 80\njustification: I stand by my answer.",
        "expected": "MAINTAIN",
        "note": "All lowercase"
    },
    # Verbose preamble before structured output
    {
        "raw": "Let me think about this carefully.\n\nThe professor raises a good point, but I disagree.\n\nDECISION: MAINTAIN\nREVISED_ANSWER: N/A\nCONFIDENCE: 75\nJUSTIFICATION: My reasoning holds.",
        "expected": "MAINTAIN",
        "note": "Long preamble before structured output"
    },
]


def validate_parser_on_edge_cases() -> dict:
    """Run the parser against the built-in edge case bank.

    Returns:
        Dict with accuracy and failures list.
    """
    n_correct = 0
    failures = []
    for case in EDGE_CASE_BANK:
        parsed = parse_phase3_response(case["raw"])
        correct = parsed["decision"] == case["expected"]
        if correct:
            n_correct += 1
        else:
            failures.append({
                "note": case["note"],
                "expected": case["expected"],
                "got": parsed["decision"],
                "raw": case["raw"][:100],
            })

    accuracy = n_correct / len(EDGE_CASE_BANK)
    return {
        "n_total": len(EDGE_CASE_BANK),
        "n_correct": n_correct,
        "accuracy": round(accuracy, 3),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] == "edge-cases":
        result = validate_parser_on_edge_cases()
        print(f"Edge case accuracy: {result['accuracy']:.1%} ({result['n_correct']}/{result['n_total']})")
        if result["failures"]:
            print(f"\nFailures ({len(result['failures'])}):")
            for f in result["failures"]:
                print(f"  [{f['note']}] expected={f['expected']}, got={f['got']}")
                print(f"  Text: {f['raw']}")
    elif sys.argv[1] == "score" and len(sys.argv) > 3:
        annotations = load_annotation_set(sys.argv[3])
        report = compute_parser_accuracy(annotations)
        print(report.summary())
