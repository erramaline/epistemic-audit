"""
cross_model_audit.py — Cross-model Phase 2 control for testing dissociation.

The "capability–introspection dissociation" hypothesis (Section 5.5 of the paper)
claims that high first-order reasoning skill does not predict metacognitive
self-monitoring. But the evidence in the paper is observational: we see a strong
reasoner with low AUROC. It could equally be explained by poor error discrimination
in general, not specifically about self-knowledge.

The cross-model audit resolves this:
  - If Model B auditing Model A's outputs achieves *higher* AUROC than A auditing
    its own outputs, the deficit is specifically about self-knowledge.
  - If B achieves similar AUROC, the deficit is about general error discrimination.

This module implements:
  1. CrossModelAudit — runs Phase 2 with an external auditor model.
  2. dissociation_test() — statistical comparison of self vs. cross-model AUROC.
  3. build_cross_model_report() — generates the full comparison table.

Design:
  The auditor model receives exactly the same prompt as the target model would
  in self-audit (same items, same format). The auditor does NOT know it is
  auditing another model's outputs — this controls for potential "charitable"
  vs. "critical" framing effects.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

from sklearn.metrics import roc_auc_score

from epistemic_audit.evaluate.phase2 import evaluate_phase2, parse_phase2_response, Phase2Results
from epistemic_audit.generate.planted_answers import generate_planted_set
from epistemic_audit.evaluate.phase1 import parse_phase1_response
from epistemic_audit.prompts.phase2 import PHASE2_SYSTEM_PROMPT, format_phase2_prompt
from epistemic_audit.evaluate.bootstrap_v2 import bootstrap_auroc

PHASE2_BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# Cross-model auditor
# ---------------------------------------------------------------------------

@dataclass
class CrossModelAuditResult:
    """Results from one model auditing another model's Phase 2 outputs."""

    target_model_name: str
    auditor_model_name: str

    # Self-audit AUROC (target model auditing its own outputs)
    self_auroc: float

    # Cross-model AUROC (auditor model auditing target model's outputs)
    cross_auroc: float

    # Delta: positive means cross > self (external model can better detect errors)
    auroc_delta: float

    # Full Phase2Results for both conditions
    self_phase2: Optional[Phase2Results] = None
    cross_phase2: Optional[Phase2Results] = None

    # Bootstrap CIs
    self_auroc_ci: Optional[dict] = None
    cross_auroc_ci: Optional[dict] = None

    def interpretation(self) -> str:
        """Interpret the cross-model AUROC comparison."""
        if self.auroc_delta > 0.10:
            return (
                f"Strong dissociation (Δ={self.auroc_delta:+.3f}): "
                f"{self.auditor_model_name} detects {self.target_model_name}'s errors "
                "substantially better than the target model can detect its own. "
                "This confirms the capability–introspection gap is specifically about "
                "self-knowledge, not general error discrimination ability."
            )
        elif self.auroc_delta > 0.04:
            return (
                f"Moderate dissociation (Δ={self.auroc_delta:+.3f}): "
                "External auditor has some advantage. Partial self-knowledge deficit."
            )
        elif self.auroc_delta > -0.04:
            return (
                f"No meaningful dissociation (Δ={self.auroc_delta:+.3f}): "
                "Self-audit and cross-model AUROC are comparable. "
                "The deficit may reflect general error discrimination, not specifically "
                "self-knowledge."
            )
        else:
            return (
                f"Reverse pattern (Δ={self.auroc_delta:+.3f}): "
                f"{self.target_model_name} audits itself *better* than {self.auditor_model_name} "
                "can audit it. Possible explanations: stylistic familiarity advantage, "
                "or the auditor model has lower general discrimination ability."
            )

    def to_dict(self) -> dict:
        return {
            "target_model": self.target_model_name,
            "auditor_model": self.auditor_model_name,
            "self_auroc": round(self.self_auroc, 4),
            "cross_auroc": round(self.cross_auroc, 4),
            "auroc_delta": round(self.auroc_delta, 4),
            "self_auroc_ci": self.self_auroc_ci,
            "cross_auroc_ci": self.cross_auroc_ci,
            "interpretation": self.interpretation(),
        }


# ---------------------------------------------------------------------------
# Core audit runner
# ---------------------------------------------------------------------------

def _run_phase2_with_auditor(
    auditor_fn: Callable[[str, str], str],
    audit_items: list[dict],
    seed: int = 42,
    verbose: bool = False,
) -> Phase2Results:
    """Run Phase 2 using a specific auditor model.

    Args:
        auditor_fn: The model to use for auditing (may differ from target).
        audit_items: Shuffled list of items (model + planted) to audit.
        seed: Random seed (for any randomness in the auditor).
        verbose: Print progress.

    Returns:
        Phase2Results from the auditor's perspective.
    """
    parsed_ratings = []
    n_batches = (len(audit_items) + PHASE2_BATCH_SIZE - 1) // PHASE2_BATCH_SIZE

    for batch_start in range(0, len(audit_items), PHASE2_BATCH_SIZE):
        batch = audit_items[batch_start:batch_start + PHASE2_BATCH_SIZE]
        prompt_items = [
            {"id": it["id"], "question": it["question"], "answer": it["answer"]}
            for it in batch
        ]
        batch_num = batch_start // PHASE2_BATCH_SIZE + 1
        if verbose:
            print(f"      Batch {batch_num}/{n_batches}...")
        raw = auditor_fn(PHASE2_SYSTEM_PROMPT, format_phase2_prompt(prompt_items))
        parsed_ratings.extend(parse_phase2_response(raw, len(batch)))

    actual_correctness = [it["is_correct"] for it in audit_items]
    planted_mask = [it["is_planted"] for it in audit_items]
    planted_correct_mask = [it.get("is_correct", False) and it["is_planted"] for it in audit_items]
    return evaluate_phase2(parsed_ratings, actual_correctness, planted_mask, planted_correct_mask)


def run_cross_model_audit(
    target_model_fn: Callable[[str, str], str],
    auditor_model_fn: Callable[[str, str], str],
    questions: list,
    raw_p1_responses: list[str],
    p1_correctness: list[bool],
    target_model_name: str = "Target",
    auditor_model_name: str = "Auditor",
    seed: int = 42,
    bootstrap_cis: bool = True,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> CrossModelAuditResult:
    """Run Phase 2 under two conditions: self-audit and cross-model audit.

    Both conditions use the same 80-item audit pool (model outputs + planted items)
    and the same shuffled order, ensuring comparability.

    Args:
        target_model_fn: The model being evaluated (used for self-audit condition).
        auditor_model_fn: An external model used for cross-model condition.
        questions: Phase 1 question list.
        raw_p1_responses: Phase 1 raw responses from the target model.
        p1_correctness: Phase 1 correctness flags for the target model.
        target_model_name: Display name for the target model.
        auditor_model_name: Display name for the auditor model.
        seed: Random seed.
        bootstrap_cis: Whether to compute bootstrap CIs (adds ~30s per run).
        n_bootstrap: Bootstrap iterations if bootstrap_cis is True.
        verbose: Print progress.

    Returns:
        CrossModelAuditResult with self_auroc, cross_auroc, delta, and interpretation.
    """
    rng = random.Random(seed)

    # Build the shared audit pool
    model_items = []
    for q, raw, correct in zip(questions, raw_p1_responses, p1_correctness):
        parsed = parse_phase1_response(raw, category=q.category)
        model_items.append({
            "id": q.id,
            "question": q.prompt,
            "answer": parsed["answer"],
            "is_correct": correct,
            "is_planted": False,
        })

    planted = generate_planted_set(questions, n_correct=10, n_incorrect=10, seed=seed)
    all_items = model_items + planted
    rng.shuffle(all_items)

    if verbose:
        print(f"\n  Phase 2 cross-model audit")
        print(f"  Target:  {target_model_name}")
        print(f"  Auditor: {auditor_model_name}")
        print(f"  Pool:    {len(model_items)} model items + {len(planted)} planted = {len(all_items)}")

    # Condition A: target model audits its own outputs
    if verbose:
        print(f"\n  Condition A: {target_model_name} self-audit...")
    self_p2 = _run_phase2_with_auditor(target_model_fn, all_items, seed, verbose)
    if verbose:
        print(f"    Self-audit AUROC: {self_p2.audit_auroc:.4f}")

    # Condition B: auditor model audits target model's outputs
    if verbose:
        print(f"\n  Condition B: {auditor_model_name} cross-audit...")
    cross_p2 = _run_phase2_with_auditor(auditor_model_fn, all_items, seed, verbose)
    if verbose:
        print(f"    Cross-audit AUROC: {cross_p2.audit_auroc:.4f}")

    delta = cross_p2.audit_auroc - self_p2.audit_auroc

    # Bootstrap CIs
    self_ci = cross_ci = None
    if bootstrap_cis:
        actual_labels = [int(it["is_correct"]) for it in all_items]

        self_scores = [r.correctness_rating / 100.0 for r in self_p2.ratings]
        self_ci = bootstrap_auroc(self_scores, actual_labels, n_bootstrap, seed=seed)

        cross_scores = [r.correctness_rating / 100.0 for r in cross_p2.ratings]
        cross_ci = bootstrap_auroc(cross_scores, actual_labels, n_bootstrap, seed=seed + 1)

        if verbose:
            print(f"\n  Self-audit  CI: [{self_ci['lower']:.4f}, {self_ci['upper']:.4f}]")
            print(f"  Cross-audit CI: [{cross_ci['lower']:.4f}, {cross_ci['upper']:.4f}]")

    result = CrossModelAuditResult(
        target_model_name=target_model_name,
        auditor_model_name=auditor_model_name,
        self_auroc=self_p2.audit_auroc,
        cross_auroc=cross_p2.audit_auroc,
        auroc_delta=delta,
        self_phase2=self_p2,
        cross_phase2=cross_p2,
        self_auroc_ci=self_ci,
        cross_auroc_ci=cross_ci,
    )

    if verbose:
        print(f"\n  Delta (cross - self): {delta:+.4f}")
        print(f"  {result.interpretation()}")

    return result


# ---------------------------------------------------------------------------
# Multi-model comparison table
# ---------------------------------------------------------------------------

def build_cross_model_report(
    model_fns: dict[str, Callable[[str, str], str]],
    questions: list,
    p1_outputs: dict[str, dict],   # model_name -> {raw_responses, correctness}
    output_path: str = "data/results/cross_model_audit.json",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Build a full N×N cross-model audit table.

    For each (target, auditor) pair where target != auditor, runs Phase 2 with
    the auditor model and records the AUROC. The diagonal is the self-audit AUROC.

    Args:
        model_fns: Dict mapping model name → model callable.
        questions: Common Phase 1 question set used for all models.
        p1_outputs: Dict mapping model name → {'raw_responses': list, 'correctness': list}.
        output_path: JSON output path.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict with full results table and dissociation rankings.
    """
    model_names = list(model_fns.keys())
    results_matrix = {}

    for target_name in model_names:
        results_matrix[target_name] = {}
        p1 = p1_outputs[target_name]

        for auditor_name in model_names:
            if auditor_name == target_name:
                # Self-audit — run once and cache
                if "_self" not in results_matrix[target_name]:
                    result = run_cross_model_audit(
                        model_fns[target_name], model_fns[target_name],
                        questions, p1["raw_responses"], p1["correctness"],
                        target_model_name=target_name,
                        auditor_model_name=f"{target_name} (self)",
                        seed=seed, verbose=verbose,
                    )
                    results_matrix[target_name]["_self"] = result.self_auroc
                results_matrix[target_name][auditor_name] = results_matrix[target_name]["_self"]
            else:
                result = run_cross_model_audit(
                    model_fns[target_name], model_fns[auditor_name],
                    questions, p1["raw_responses"], p1["correctness"],
                    target_model_name=target_name,
                    auditor_model_name=auditor_name,
                    seed=seed, verbose=verbose,
                )
                results_matrix[target_name][auditor_name] = result.cross_auroc

    # Compute dissociation score per target model:
    # max(cross AUROC from any auditor) - self AUROC
    dissociation = {}
    for target_name in model_names:
        self_auroc = results_matrix[target_name].get("_self", 0.0)
        cross_aurocs = [
            v for k, v in results_matrix[target_name].items()
            if k != "_self" and k != target_name
        ]
        max_cross = max(cross_aurocs) if cross_aurocs else self_auroc
        dissociation[target_name] = {
            "self_auroc": round(self_auroc, 4),
            "max_cross_auroc": round(max_cross, 4),
            "dissociation_score": round(max_cross - self_auroc, 4),
        }

    output = {
        "n_questions": len(questions),
        "models": model_names,
        "auroc_matrix": {
            target: {
                auditor: round(auroc, 4)
                for auditor, auroc in by_auditor.items()
                if not auditor.startswith("_")
            }
            for target, by_auditor in results_matrix.items()
        },
        "dissociation_scores": dict(
            sorted(dissociation.items(), key=lambda x: -x[1]["dissociation_score"])
        ),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        _print_cross_model_table(output)

    return output


def _print_cross_model_table(output: dict) -> None:
    """Pretty-print the cross-model AUROC matrix."""
    models = output["models"]
    matrix = output["auroc_matrix"]

    print("\n" + "=" * 70)
    print("CROSS-MODEL PHASE 2 AUDIT — AUROC MATRIX")
    print("Rows = Target model (whose outputs are audited)")
    print("Cols = Auditor model (who performs the audit)")
    print("Diagonal = self-audit")
    print("=" * 70)

    col_w = 14
    header = f"{'Target / Auditor':<22}" + "".join(f"{m[:col_w]:>{col_w}}" for m in models)
    print(header)
    print("-" * len(header))
    for target in models:
        row = f"{target[:22]:<22}"
        for auditor in models:
            val = matrix.get(target, {}).get(auditor, float("nan"))
            marker = "*" if target == auditor else " "
            row += f"{val:.4f}{marker}".rjust(col_w)
        print(row)

    print("\nDissociation scores (max_cross_AUROC - self_AUROC):")
    for model, d in output["dissociation_scores"].items():
        print(f"  {model:<30} self={d['self_auroc']:.4f}  "
              f"max_cross={d['max_cross_auroc']:.4f}  "
              f"Δ={d['dissociation_score']:+.4f}")
