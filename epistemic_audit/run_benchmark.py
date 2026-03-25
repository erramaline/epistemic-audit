"""Main orchestrator: runs all 3 phases of the Epistemic Audit benchmark."""

import json
import os
import random
import time
import logging
from typing import Callable
import re

from epistemic_audit.generate.questions import QuestionGenerator
from epistemic_audit.generate.planted_answers import generate_planted_set
from epistemic_audit.generate.counterarguments import (
    generate_sophistic_counterargument, generate_valid_counterargument,
)
from epistemic_audit.prompts.phase1 import PHASE1_SYSTEM_PROMPT, format_phase1_prompt
from epistemic_audit.prompts.phase2 import PHASE2_SYSTEM_PROMPT, format_phase2_prompt
from epistemic_audit.prompts.phase3 import PHASE3_SYSTEM_PROMPT, format_phase3_prompt
from epistemic_audit.evaluate.phase1 import evaluate_phase1, parse_phase1_response
from epistemic_audit.evaluate.phase2 import evaluate_phase2, parse_phase2_response
from epistemic_audit.evaluate.phase3 import evaluate_phase3, parse_phase3_response
from epistemic_audit.evaluate.bootstrap import compute_metric_confidence_intervals
from epistemic_audit.evaluate.composite import compute_epistemic_score, EpistemicProfile

logger = logging.getLogger(__name__)

_DEFAULT_API_ERROR_RESPONSE = (
    "ANSWER: ERROR\nCONFIDENCE: 50\nRATIONALE: API call failed after all retries."
)

# Phase 2 batch size: send this many items per API call
PHASE2_BATCH_SIZE = 10


def _strip_thinking_tags(response: str) -> str:
    """Remove <think>...</think> reasoning blocks from model responses.
    
    Some reasoning models (e.g., DeepSeek R1) include internal reasoning
    wrapped in <think> tags. These must be stripped before parsing the
    structured ANSWER/CONFIDENCE/RATIONALE format.
    """
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()


class EpistemicAuditBenchmark:
    """Run all 3 phases of the Epistemic Audit metacognition benchmark.

    Args:
        model_fn: Callable(system_prompt, user_prompt) -> response string.
        seed: Random seed for reproducibility.
        n_per_category: Questions per category in Phase 1 (default 10).
        verbose: If True, print progress to stdout (default True).
        checkpoint_dir: Directory for intermediate checkpoint files.
        throttle_seconds: Sleep duration in seconds per model call (default 0.0).
    """

    def __init__(
        self,
        model_fn: Callable[[str, str], str],
        seed: int = 42,
        n_per_category: int = 10,
        verbose: bool = True,
        checkpoint_dir: str = "data/results",
        throttle_seconds: float = 0.0,
    ):
        """Initialise the benchmark runner.

        Args:
            model_fn: Takes (system_prompt, user_prompt), returns response string.
            seed: Random seed for reproducibility.
            n_per_category: Questions per category in Phase 1.
            verbose: If True, print progress to stdout.
            checkpoint_dir: Directory for intermediate checkpoint files.
            throttle_seconds: Sleep duration in seconds per model call.
        """
        self.model_fn = model_fn
        self.seed = seed
        self.n_per_category = n_per_category
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        self.throttle_seconds = throttle_seconds
        self.rng = random.Random(seed)

        # Stored for later visualization access
        self._phase1_results = None

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def run(self) -> EpistemicProfile:
        """Execute all 3 phases and return the final epistemic profile.

        Returns:
            EpistemicProfile dataclass with composite score + all sub-metrics.
        """
        self._log("=" * 60)
        self._log("EPISTEMIC AUDIT BENCHMARK")
        self._log("=" * 60)

        # Phase 1
        self._log("\n--- Phase 1: Generate (Knowledge Baseline) ---")
        questions, raw_responses, p1 = self._run_phase1()
        self._phase1_results = p1
        self.save_checkpoint("phase1", p1.__dict__ if hasattr(p1, "__dict__") else {})

        # Phase 2
        self._log("\n--- Phase 2: Audit (Self-Knowledge) ---")
        p2 = self._run_phase2(questions, raw_responses, p1)
        self.save_checkpoint("phase2", {"audit_auroc": p2.audit_auroc,
                                         "planted_detection_rate": p2.planted_detection_rate})

        # Phase 3
        self._log("\n--- Phase 3: Challenge (Belief Revision) ---")
        p3 = self._run_phase3(questions, raw_responses, p1)
        self.save_checkpoint("phase3", {"hold_rate": p3.appropriate_hold_rate,
                                         "revise_rate": p3.appropriate_revise_rate,
                                         "sycophancy": p3.sycophancy_index})

        # Composite and Confidence Intervals
        ci = compute_metric_confidence_intervals(p1.correctness, [(r.confidence / 100.0) for r in p1.responses])
        profile = compute_epistemic_score(p1, p2, p3, confidence_intervals=ci)
        
        self.export_detailed_results(
            profile, questions, raw_responses, 
            os.path.join(self.checkpoint_dir, "detailed_results.json")
        )

        self._log(f"\n{'=' * 60}")
        self._log(f"COMPOSITE SCORE: {profile.composite_score:.4f}")
        self._log(f"LEVEL: {profile.level}")
        self._log(f"{'=' * 60}")

        return profile

    def save_checkpoint(self, phase: str, data: dict, path: str | None = None) -> None:
        """Persist intermediate phase results to disk to guard against crashes.

        Args:
            phase: Phase identifier string (e.g. 'phase1').
            data: Serialisable dict of results to checkpoint.
            path: Override output path; defaults to checkpoint_dir/checkpoint_{phase}.json.
        """
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{phase}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"phase": phase, "data": data}, f, indent=2, default=str)
            self._log(f"  [checkpoint] saved -> {path}")
        except Exception as exc:
            logger.warning("Checkpoint write failed for %s: %s", phase, exc)

    def export_detailed_results(self, profile: EpistemicProfile, questions, raw_responses, path: str):
        """Export per-question detailed results for analysis.

        Saves a JSON with every question, model response, correctness,
        and confidence — useful for error analysis and paper figures.
        """
        details = []
        for i, (q, raw) in enumerate(zip(questions, raw_responses)):
            parsed = parse_phase1_response(raw)
            details.append({
                "question_id": q.id,
                "category": q.category,
                "difficulty": q.difficulty,
                "prompt": q.prompt[:200],  # Truncate for readability
                "correct_answer": q.correct_answer,
                "is_answerable": q.is_answerable,
                "model_answer": parsed["answer"][:200],
                "model_confidence": parsed["confidence"],
                "is_correct": q.verify(parsed["answer"]),
            })

        output = {
            "profile": profile.to_dict(),
            "per_question": details,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        self._log(f"  [detailed export] saved -> {path}")

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        """Print only when verbose is True.

        Args:
            msg: Message to print.
        """
        if self.verbose:
            print(msg)

    def _call_model(self, system_prompt: str, user_prompt: str,
                    max_retries: int = 3) -> str:
        """Wrap model_fn with exponential back-off retry logic.

        Args:
            system_prompt: System-level instruction string.
            user_prompt: User-level query string.
            max_retries: Number of attempts before giving up (default 3).

        Returns:
            Model response string, or a safe default on total failure.
        """
        delays = [10, 15, 30]
        if self.throttle_seconds > 0:
            time.sleep(self.throttle_seconds)
        for attempt in range(max_retries):
            try:
                result = self.model_fn(system_prompt, user_prompt)
                return _strip_thinking_tags(result)
            except Exception as exc:
                wait = delays[attempt] if attempt < len(delays) else delays[-1]
                if "429" in str(exc) or "quota" in str(exc).lower():
                    wait = max(wait, 30)  # Moderate wait for rate limits
                
                logger.warning(
                    "model_fn call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, max_retries, exc, wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
        logger.error("All %d retries exhausted; returning safe default.", max_retries)
        return _DEFAULT_API_ERROR_RESPONSE

    def _run_phase1(self):
        """Generate questions, query the model, and evaluate Phase 1.

        Returns:
            Tuple of (questions, raw_responses, Phase1Results).
        """
        gen = QuestionGenerator(seed=self.seed)
        questions = gen.generate_set(n_per_category=self.n_per_category)

        raw_responses = []
        for i, q in enumerate(questions):
            self._log(f"  [{i+1}/{len(questions)}] {q.category} (difficulty {q.difficulty})")
            resp = self._call_model(PHASE1_SYSTEM_PROMPT, format_phase1_prompt(q.prompt))
            raw_responses.append(resp)

        results = evaluate_phase1(questions, raw_responses)
        self._log(f"\n  Accuracy:       {results.accuracy:.2%}")
        self._log(f"  Brier Score:    {results.brier_score:.4f}")
        self._log(f"  Abstention F1:  {results.abstention_f1:.4f}")

        return questions, raw_responses, results

    def _run_phase2(self, questions, raw_responses, p1):
        """Build the 80-item audit set and evaluate in batches of PHASE2_BATCH_SIZE.

        Batching reduces the risk of long-context failures where models produce
        fewer than n_items structured responses.

        Args:
            questions: List of Question objects from Phase 1.
            raw_responses: Raw string responses from Phase 1.
            p1: Phase1Results (used for correctness labels).

        Returns:
            Phase2Results dataclass.
        """
        model_items = []
        for q, raw, correct in zip(questions, raw_responses, p1.correctness):
            parsed = parse_phase1_response(raw)
            model_items.append({
                "id": q.id, "question": q.prompt, "answer": parsed["answer"],
                "is_correct": correct, "is_planted": False,
            })

        planted = generate_planted_set(questions, n_correct=10, n_incorrect=10, seed=self.seed)
        all_items = model_items + planted
        self.rng.shuffle(all_items)

        self._log(f"  Auditing {len(all_items)} items in batches of {PHASE2_BATCH_SIZE}...")

        parsed_ratings: list[dict] = []
        for batch_start in range(0, len(all_items), PHASE2_BATCH_SIZE):
            batch = all_items[batch_start: batch_start + PHASE2_BATCH_SIZE]
            # Renumber items 1-based within this batch so the prompt is clean
            prompt_items = [
                {"id": it["id"], "question": it["question"], "answer": it["answer"]}
                for it in batch
            ]
            audit_prompt = format_phase2_prompt(prompt_items)
            batch_num = batch_start // PHASE2_BATCH_SIZE + 1
            total_batches = (len(all_items) + PHASE2_BATCH_SIZE - 1) // PHASE2_BATCH_SIZE
            self._log(f"    Batch {batch_num}/{total_batches} ({len(batch)} items)...")

            raw_audit = self._call_model(PHASE2_SYSTEM_PROMPT, audit_prompt)
            batch_ratings = parse_phase2_response(raw_audit, len(batch))
            parsed_ratings.extend(batch_ratings)

        actual_correctness = [it["is_correct"] for it in all_items]
        planted_mask = [it["is_planted"] for it in all_items]
        planted_correct_mask = [it.get("is_correct", False) and it["is_planted"] for it in all_items]

        results = evaluate_phase2(parsed_ratings, actual_correctness, planted_mask, planted_correct_mask)
        self._log(f"  Audit AUROC:        {results.audit_auroc:.4f}")
        self._log(f"  Planted Detection:  {results.planted_detection_rate:.2%}")

        return results

    def _run_phase3(self, questions, raw_responses, p1):
        """Challenge selected correct + incorrect answers and evaluate belief revision.

        Args:
            questions: List of Question objects from Phase 1.
            raw_responses: Raw string responses from Phase 1.
            p1: Phase1Results.

        Returns:
            Phase3Results dataclass.
        """
        correct_idx = [i for i, c in enumerate(p1.correctness) if c]
        incorrect_idx = [i for i, c in enumerate(p1.correctness) if not c]

        sel_correct = self.rng.sample(correct_idx, min(10, len(correct_idx)))
        sel_incorrect = self.rng.sample(incorrect_idx, min(10, len(incorrect_idx)))

        parsed_responses, was_correct, challenge_valid = [], [], []

        for idx in sel_correct:
            q = questions[idx]
            p = parse_phase1_response(raw_responses[idx])
            counter = generate_sophistic_counterargument(q.prompt, q.correct_answer, self.seed + idx)
            raw = self._call_model(PHASE3_SYSTEM_PROMPT, format_phase3_prompt(q.prompt, p["answer"], counter))
            parsed_responses.append(parse_phase3_response(raw))
            was_correct.append(True)
            challenge_valid.append(False)

        for idx in sel_incorrect:
            q = questions[idx]
            p = parse_phase1_response(raw_responses[idx])
            counter = generate_valid_counterargument(
                q.prompt, p["answer"], q.correct_answer, q.category, self.seed + idx
            )
            raw = self._call_model(PHASE3_SYSTEM_PROMPT, format_phase3_prompt(q.prompt, p["answer"], counter))
            parsed_responses.append(parse_phase3_response(raw))
            was_correct.append(False)
            challenge_valid.append(True)

        results = evaluate_phase3(parsed_responses, was_correct, challenge_valid)
        self._log(f"  Hold Rate:   {results.appropriate_hold_rate:.2%}")
        self._log(f"  Revise Rate: {results.appropriate_revise_rate:.2%}")
        self._log(f"  Sycophancy:  {results.sycophancy_index:.4f}")

        return results
