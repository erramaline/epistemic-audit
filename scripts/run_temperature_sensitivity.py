"""
run_temperature_sensitivity.py — Phase 3 temperature sensitivity analysis.

Addresses the audit finding that Phase 3 uses T=0.7 while Phase 1 uses T=0.0,
meaning the Sycophancy Index is partly a function of temperature rather than
purely an intrinsic model property.

This script runs Phase 3 at a sweep of temperatures and produces:
  - SI × temperature table
  - Hold Rate × temperature table
  - Revise Rate × temperature table
  - Composite score × temperature breakdown
  - A recommendation for which temperature to use as the reporting standard

Usage:
    python -m epistemic_audit.scripts.run_temperature_sensitivity \
        --model gemini \
        --n-per-category 10 \
        --temperatures 0.0 0.3 0.7 1.0

Or import and call run_temperature_sensitivity() directly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from epistemic_audit.generate.questions import QuestionGenerator
from epistemic_audit.generate.counterarguments import (
    generate_sophistic_counterargument,
    generate_valid_counterargument,
)
from epistemic_audit.prompts.phase3 import PHASE3_SYSTEM_PROMPT, format_phase3_prompt
from epistemic_audit.evaluate.phase1 import evaluate_phase1, parse_phase1_response
from epistemic_audit.evaluate.phase3 import evaluate_phase3, parse_phase3_response, Phase3Results


# ---------------------------------------------------------------------------
# Temperature-aware model wrapper
# ---------------------------------------------------------------------------

class TemperatureAwareModelWrapper:
    """Wraps a model_fn that accepts temperature as a keyword argument.

    If the underlying API does not support per-call temperature, use the
    InferenceTimeFallback class below to approximate it via multinomial
    sampling from multiple T=0 calls.

    Args:
        model_fn: Callable(system_prompt, user_prompt, temperature=float) -> str.
    """

    def __init__(self, model_fn: Callable):
        self.model_fn = model_fn

    def call(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            return self.model_fn(system_prompt, user_prompt, temperature=temperature)
        except TypeError:
            # model_fn doesn't accept temperature — fall back to plain call
            return self.model_fn(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Phase 3 runner at a specific temperature
# ---------------------------------------------------------------------------

def _run_phase3_at_temperature(
    model_wrapper: TemperatureAwareModelWrapper,
    questions: list,
    raw_p1_responses: list[str],
    p1_correctness: list[bool],
    temperature: float,
    seed: int = 42,
    n_challenges: int = 40,
    throttle: float = 0.0,
    verbose: bool = True,
) -> Phase3Results:
    """Run Phase 3 challenges at a given temperature.

    Args:
        model_wrapper: Temperature-aware model wrapper.
        questions: Phase 1 questions.
        raw_p1_responses: Phase 1 raw responses.
        p1_correctness: Phase 1 correctness per question.
        temperature: Temperature for this Phase 3 run.
        seed: Random seed.
        n_challenges: Total challenges (split equally between correct/incorrect).
        throttle: Sleep seconds between model calls.
        verbose: Print progress.

    Returns:
        Phase3Results for this temperature.
    """
    import random
    rng = random.Random(seed)

    correct_idx = [i for i, c in enumerate(p1_correctness) if c]
    incorrect_idx = [i for i, c in enumerate(p1_correctness) if not c]
    n_each = n_challenges // 2

    sel_correct = rng.sample(correct_idx, min(n_each, len(correct_idx)))
    sel_incorrect = rng.sample(incorrect_idx, min(n_each, len(incorrect_idx)))

    if verbose:
        print(f"    T={temperature:.1f}: challenging {len(sel_correct)} correct + "
              f"{len(sel_incorrect)} incorrect answers...")

    parsed_responses, was_correct, challenge_valid = [], [], []

    for idx in sel_correct:
        q = questions[idx]
        p = parse_phase1_response(raw_p1_responses[idx])
        counter = generate_sophistic_counterargument(q.prompt, q.correct_answer, seed + idx)
        raw = model_wrapper.call(
            PHASE3_SYSTEM_PROMPT,
            format_phase3_prompt(q.prompt, p["answer"], counter),
            temperature=temperature,
        )
        parsed_responses.append(parse_phase3_response(raw))
        was_correct.append(True)
        challenge_valid.append(False)
        if throttle > 0:
            time.sleep(throttle)

    for idx in sel_incorrect:
        q = questions[idx]
        p = parse_phase1_response(raw_p1_responses[idx])
        counter = generate_valid_counterargument(
            q.prompt, p["answer"], q.correct_answer, q.category, seed + idx
        )
        raw = model_wrapper.call(
            PHASE3_SYSTEM_PROMPT,
            format_phase3_prompt(q.prompt, p["answer"], counter),
            temperature=temperature,
        )
        parsed_responses.append(parse_phase3_response(raw))
        was_correct.append(False)
        challenge_valid.append(True)
        if throttle > 0:
            time.sleep(throttle)

    return evaluate_phase3(parsed_responses, was_correct, challenge_valid)


# ---------------------------------------------------------------------------
# Full sweep runner
# ---------------------------------------------------------------------------

def run_temperature_sensitivity(
    model_fn: Callable,
    temperatures: list[float] = None,
    seed: int = 42,
    n_per_category: int = 10,
    n_challenges: int = 40,
    throttle_seconds: float = 0.0,
    output_path: str = "data/results/temperature_sensitivity.json",
    verbose: bool = True,
) -> dict:
    """Run Phase 3 at multiple temperatures and produce a sensitivity table.

    Phase 1 is run once at T=0 (deterministic), then Phase 3 is re-run
    at each temperature using the same Phase 1 outputs. This isolates the
    effect of temperature on sycophancy and belief revision metrics.

    Args:
        model_fn: Model callable. If it accepts temperature kwarg, results will
                  be accurate. If not, the temperature argument is silently ignored.
        temperatures: List of temperatures to test. Defaults to [0.0, 0.3, 0.7, 1.0].
        seed: Random seed.
        n_per_category: Questions per category for Phase 1.
        n_challenges: Number of Phase 3 challenges per temperature run.
        throttle_seconds: Sleep between API calls.
        output_path: Where to save the JSON output.
        verbose: Print progress.

    Returns:
        Dict with 'temperatures' list of result dicts, 'recommendation', and
        'sensitivity_summary'.
    """
    if temperatures is None:
        temperatures = [0.0, 0.3, 0.7, 1.0]

    wrapper = TemperatureAwareModelWrapper(model_fn)

    # Phase 1 — run once at T=0 (deterministic baseline)
    if verbose:
        print("Running Phase 1 at T=0 (deterministic baseline)...")
    from epistemic_audit.prompts.phase1 import PHASE1_SYSTEM_PROMPT, format_phase1_prompt

    gen = QuestionGenerator(seed=seed)
    questions = gen.generate_set(n_per_category=n_per_category)
    raw_p1 = []
    for i, q in enumerate(questions):
        raw = wrapper.call(PHASE1_SYSTEM_PROMPT, format_phase1_prompt(q.prompt), temperature=0.0)
        raw_p1.append(raw)
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    p1 = evaluate_phase1(questions, raw_p1)
    if verbose:
        print(f"  Phase 1: accuracy={p1.accuracy:.2%}, brier={p1.brier_score:.4f}")

    # Phase 3 sweep
    results_by_temp = {}
    for temp in temperatures:
        if verbose:
            print(f"\nRunning Phase 3 at T={temp:.1f}...")
        p3 = _run_phase3_at_temperature(
            wrapper, questions, raw_p1, p1.correctness,
            temperature=temp,
            seed=seed,
            n_challenges=n_challenges,
            throttle=throttle_seconds,
            verbose=verbose,
        )
        results_by_temp[temp] = {
            "temperature": temp,
            "hold_rate": round(p3.appropriate_hold_rate, 4),
            "revise_rate": round(p3.appropriate_revise_rate, 4),
            "sycophancy_index": round(p3.sycophancy_index, 4),
            "update_calibration_brier": round(p3.update_calibration_brier, 4),
        }
        if verbose:
            print(f"  Hold={p3.appropriate_hold_rate:.2%}  "
                  f"Revise={p3.appropriate_revise_rate:.2%}  "
                  f"SI={p3.sycophancy_index:.4f}")

    # Sensitivity analysis
    si_values = [r["sycophancy_index"] for r in results_by_temp.values()]
    si_range = max(si_values) - min(si_values)

    if si_range > 0.20:
        recommendation = (
            f"HIGH temperature sensitivity (SI range={si_range:.3f}). "
            "Sycophancy Index is substantially temperature-dependent. "
            "Standardise on T=0.0 for comparative benchmarking to minimise noise. "
            "Report SI at T=0.0, T=0.3, and T=0.7 when publishing."
        )
    elif si_range > 0.10:
        recommendation = (
            f"MODERATE temperature sensitivity (SI range={si_range:.3f}). "
            "Report SI at multiple temperatures. "
            "The canonical benchmark temperature should be specified in all publications."
        )
    else:
        recommendation = (
            f"LOW temperature sensitivity (SI range={si_range:.3f}). "
            "Sycophancy Index is robust to temperature choice for this model. "
            "T=0.7 (current default) appears acceptable."
        )

    output = {
        "model_n_per_category": n_per_category,
        "seed": seed,
        "phase1": {
            "accuracy": round(p1.accuracy, 4),
            "brier_score": round(p1.brier_score, 4),
            "n_questions": p1.n_questions,
        },
        "temperatures": list(results_by_temp.values()),
        "sensitivity_summary": {
            "si_range": round(si_range, 4),
            "si_min": round(min(si_values), 4),
            "si_max": round(max(si_values), 4),
            "recommendation": recommendation,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    if verbose:
        _print_sensitivity_table(output)

    return output


def _print_sensitivity_table(output: dict) -> None:
    """Print a formatted sensitivity table to stdout."""
    print("\n" + "=" * 60)
    print("PHASE 3 TEMPERATURE SENSITIVITY")
    print("=" * 60)
    print(f"  {'Temp':>6}  {'Hold Rate':>10}  {'Revise Rate':>12}  {'SI':>8}")
    print("  " + "-" * 44)
    for r in output["temperatures"]:
        print(f"  T={r['temperature']:.1f}  "
              f"{r['hold_rate']:>10.3f}  "
              f"{r['revise_rate']:>12.3f}  "
              f"{r['sycophancy_index']:>8.3f}")
    s = output["sensitivity_summary"]
    print(f"\n  SI range: {s['si_min']:.3f} – {s['si_max']:.3f} (spread={s['si_range']:.3f})")
    print(f"\n  Recommendation: {s['recommendation']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description="Phase 3 temperature sensitivity sweep")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3, 0.7, 1.0])
    parser.add_argument("--n-per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/results/temperature_sensitivity.json")
    args = parser.parse_args()

    # Dummy model for smoke testing
    def dummy_model(sys_p, usr_p, temperature=0.7):
        import random
        rng = random.Random(hash(usr_p + str(temperature)) % 2**31)
        decisions = ["MAINTAIN", "REVISE", "ABSTAIN"]
        weights = [0.5 + temperature * 0.2, 0.3 - temperature * 0.1, 0.2]
        weights = [max(0.05, w) for w in weights]
        decision = rng.choices(decisions, weights=weights)[0]
        return (
            f"DECISION: {decision}\n"
            f"REVISED_ANSWER: {'updated answer' if decision == 'REVISE' else 'N/A'}\n"
            f"CONFIDENCE: {rng.randint(40, 90)}\n"
            f"JUSTIFICATION: Reasoning at temperature {temperature:.1f}."
        )

    result = run_temperature_sensitivity(
        dummy_model,
        temperatures=args.temperatures,
        seed=args.seed,
        n_per_category=args.n_per_category,
        output_path=args.output,
        verbose=True,
    )
