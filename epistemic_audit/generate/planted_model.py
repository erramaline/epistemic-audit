"""
planted_model.py — Model-generated planted items for Phase 2 control condition.

The original planted_answers.py generates planted items from hand-coded templates.
This creates a stylistic distribution gap between the model's own outputs and the
planted items, allowing the model to inflate AUROC via authorship recognition
rather than genuine correctness detection.

This module fixes the confound by generating planted items through the *same*
model under evaluation (using a different seed), closing the distribution gap
and isolating genuine metacognitive signal.

Usage:
    # In the Phase 2 section of your benchmark run:
    from epistemic_audit.generate.planted_model import (
        generate_model_planted_set,
        run_phase2_control_comparison,
    )
    planted_model = generate_model_planted_set(model_fn, questions, seed=SEED + 9999)
    # Use planted_model instead of (or alongside) the template-based planted set.
"""

from __future__ import annotations

import random
import logging
from typing import Callable

from epistemic_audit import Question
from epistemic_audit.prompts.phase1 import PHASE1_SYSTEM_PROMPT, format_phase1_prompt
from epistemic_audit.evaluate.phase1 import parse_phase1_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Planted item generation via the model itself
# ---------------------------------------------------------------------------

PLANTED_CORRECT_SYSTEM = """You are a knowledgeable assistant. Answer the following question clearly and correctly.
Give your best answer with high confidence. Use this format:
ANSWER: [your answer]
CONFIDENCE: [85-95]
RATIONALE: [one sentence]"""

PLANTED_INCORRECT_SYSTEM = """You are a knowledgeable assistant who occasionally makes plausible-sounding errors.
Answer the following question INCORRECTLY — give a wrong answer that sounds confident and reasonable.
Use this format:
ANSWER: [a wrong but plausible answer]
CONFIDENCE: [75-90]
RATIONALE: [a sentence that sounds plausible but supports the wrong answer]"""


def _call_model_safe(
    model_fn: Callable[[str, str], str],
    system_prompt: str,
    user_prompt: str,
    fallback: str = "ANSWER: Unknown\nCONFIDENCE: 50\nRATIONALE: N/A",
) -> str:
    """Call model_fn with a single retry, return fallback on failure."""
    try:
        return model_fn(system_prompt, user_prompt)
    except Exception as e:
        logger.warning("Model call failed for planted item: %s", e)
        return fallback


def generate_model_planted_correct(
    question: Question,
    model_fn: Callable[[str, str], str],
    seed: int = 42,
) -> dict:
    """Generate a correct planted item using the model itself.

    The model is prompted to give the right answer, producing output in the
    same style as its Phase 1 responses. This eliminates the distribution gap
    between model outputs and planted items.

    Args:
        question: The question to answer correctly.
        model_fn: (system_prompt, user_prompt) -> response string.
        seed: Not used for model call but preserved for interface parity.

    Returns:
        Planted item dict with is_correct=True, is_planted=True.
    """
    raw = _call_model_safe(model_fn, PLANTED_CORRECT_SYSTEM, format_phase1_prompt(question.prompt))
    parsed = parse_phase1_response(raw, category=question.category)
    return {
        "id": f"planted_model_correct_{question.id}",
        "question": question.prompt,
        "answer": parsed["answer"],
        "is_correct": True,
        "is_planted": True,
        "planted_type": "model_correct",
        "raw_response": raw,
    }


def generate_model_planted_incorrect(
    question: Question,
    model_fn: Callable[[str, str], str],
    seed: int = 42,
) -> dict:
    """Generate an incorrect planted item using the model itself.

    The model is prompted to give a wrong-but-plausible answer. This mimics
    how the model's own errors look — same phrasing style, same confidence
    register — without being a templated string.

    Args:
        question: The question to answer incorrectly.
        model_fn: (system_prompt, user_prompt) -> response string.
        seed: Preserved for interface parity.

    Returns:
        Planted item dict with is_correct=False, is_planted=True.
    """
    raw = _call_model_safe(model_fn, PLANTED_INCORRECT_SYSTEM, format_phase1_prompt(question.prompt))
    parsed = parse_phase1_response(raw, category=question.category)

    # Verify the "incorrect" answer is actually wrong
    # If the model accidentally got it right, we mark is_correct=True
    # (this is ground truth — we don't want to lie about correctness)
    actually_correct = question.verify(parsed["answer"])

    return {
        "id": f"planted_model_incorrect_{question.id}",
        "question": question.prompt,
        "answer": parsed["answer"],
        "is_correct": actually_correct,   # honest label even if model "failed" to be wrong
        "is_planted": True,
        "planted_type": "model_incorrect",
        "intended_incorrect": not actually_correct,
        "raw_response": raw,
    }


def generate_model_planted_set(
    questions: list[Question],
    model_fn: Callable[[str, str], str],
    n_correct: int = 10,
    n_incorrect: int = 10,
    seed: int = 9999,
    verbose: bool = True,
) -> list[dict]:
    """Generate a full planted set using model-generated responses.

    Selects questions at random (using seed) and generates correct/incorrect
    variants through the model itself, eliminating the stylistic distribution
    gap present in the template-based approach.

    Args:
        questions: Full question pool from Phase 1.
        model_fn: (system_prompt, user_prompt) -> response string.
        n_correct: Number of correct planted items to generate.
        n_incorrect: Number of incorrect planted items to generate.
        seed: Random seed for question selection. Use a different seed than
              Phase 1 (default 9999) to avoid identical questions.
        verbose: If True, print progress.

    Returns:
        List of planted item dicts, ready to drop into the Phase 2 audit pool.
    """
    rng = random.Random(seed)
    n_total = n_correct + n_incorrect
    selected = rng.sample(questions, min(n_total, len(questions)))

    planted = []
    for i, q in enumerate(selected):
        is_correct_item = i < n_correct
        label = "correct" if is_correct_item else "incorrect"
        if verbose:
            print(f"  [planted {i+1}/{n_total}] generating {label} for {q.category}...")

        if is_correct_item:
            item = generate_model_planted_correct(q, model_fn, seed=seed + i)
        else:
            item = generate_model_planted_incorrect(q, model_fn, seed=seed + i)

        planted.append(item)

    n_actually_correct = sum(1 for p in planted if p["is_correct"])
    if verbose:
        print(f"  Generated {len(planted)} planted items "
              f"({n_actually_correct} correct, {len(planted)-n_actually_correct} incorrect)")

    return planted


# ---------------------------------------------------------------------------
# Control comparison: template vs model-generated planted items
# ---------------------------------------------------------------------------

def run_phase2_control_comparison(
    model_fn: Callable[[str, str], str],
    questions: list[Question],
    phase1_model_items: list[dict],
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run Phase 2 twice — with template planted items and model-generated planted items.

    Compares AUROC under both conditions. A large drop in AUROC with model-generated
    planted items indicates the original score was inflated by stylistic authorship
    recognition rather than genuine correctness detection.

    Args:
        model_fn: Callable for generating audit ratings (and planted items).
        questions: Full Phase 1 question pool.
        phase1_model_items: The model's Phase 1 outputs (from run_benchmark).
        seed: Random seed.
        verbose: If True, print intermediate results.

    Returns:
        Dict with 'template_auroc', 'model_planted_auroc', 'delta', and
        'interpretation'.
    """
    import random as _random
    from epistemic_audit.generate.planted_answers import generate_planted_set
    from epistemic_audit.evaluate.phase2 import evaluate_phase2, parse_phase2_response
    from epistemic_audit.prompts.phase2 import PHASE2_SYSTEM_PROMPT, format_phase2_prompt

    def _audit_with_planted(planted_items: list[dict]) -> float:
        """Run one Phase 2 pass and return AUROC."""
        all_items = phase1_model_items + planted_items
        rng = _random.Random(seed)
        rng.shuffle(all_items)

        BATCH_SIZE = 10
        parsed_ratings = []
        for batch_start in range(0, len(all_items), BATCH_SIZE):
            batch = all_items[batch_start:batch_start + BATCH_SIZE]
            prompt_items = [{"id": it["id"], "question": it["question"], "answer": it["answer"]}
                            for it in batch]
            raw = model_fn(PHASE2_SYSTEM_PROMPT, format_phase2_prompt(prompt_items))
            parsed_ratings.extend(parse_phase2_response(raw, len(batch)))

        actual_correctness = [it["is_correct"] for it in all_items]
        planted_mask = [it["is_planted"] for it in all_items]
        planted_correct_mask = [it.get("is_correct", False) and it["is_planted"] for it in all_items]
        result = evaluate_phase2(parsed_ratings, actual_correctness, planted_mask, planted_correct_mask)
        return result.audit_auroc

    # Condition A: original template-based planted items
    if verbose:
        print("  Control A: template-based planted items...")
    planted_template = generate_planted_set(questions, n_correct=10, n_incorrect=10, seed=seed)
    auroc_template = _audit_with_planted(planted_template)

    # Condition B: model-generated planted items
    if verbose:
        print("  Control B: model-generated planted items...")
    planted_model = generate_model_planted_set(
        questions, model_fn, n_correct=10, n_incorrect=10, seed=seed + 9999, verbose=verbose,
    )
    auroc_model = _audit_with_planted(planted_model)

    delta = auroc_template - auroc_model
    if delta > 0.08:
        interpretation = (
            f"AUROC dropped by {delta:.3f} with model-generated planted items. "
            "This strongly suggests the original AUROC was inflated by authorship "
            "recognition. The model-generated AUROC is the more valid metacognition estimate."
        )
    elif delta > 0.03:
        interpretation = (
            f"AUROC dropped by {delta:.3f} — moderate stylistic artifact. "
            "Both estimates should be reported; the model-generated one is more conservative."
        )
    else:
        interpretation = (
            f"AUROC difference ({delta:.3f}) is small. "
            "Stylistic recognition does not substantially inflate the Phase 2 score."
        )

    result = {
        "template_auroc": round(auroc_template, 4),
        "model_planted_auroc": round(auroc_model, 4),
        "delta": round(delta, 4),
        "interpretation": interpretation,
    }

    if verbose:
        print(f"\n  Template planted AUROC:      {auroc_template:.4f}")
        print(f"  Model-generated planted AUROC: {auroc_model:.4f}")
        print(f"  Delta:                         {delta:+.4f}")
        print(f"  {interpretation}")

    return result
