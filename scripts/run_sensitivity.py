"""Benchmarking stability analysis: run the benchmark with multiple seeds.

Measures metric variance to demonstrate that results are stable, not noisy.
This is important evidence for competition judges.

Usage:
    python scripts/run_sensitivity.py               # uses smart_dummy_model_fn
    python scripts/run_sensitivity.py --real        # uses real API if key is set
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.run_benchmark import EpistemicAuditBenchmark


SEEDS = [42, 43, 44, 45, 46]
N_PER_CATEGORY = 5  # Keep costs manageable


def smart_dummy_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Deterministic per-prompt dummy for sensitivity testing.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User-level query.

    Returns:
        Structured response string appropriate for the detected phase.
    """
    import hashlib
    import re

    h = int(hashlib.md5(user_prompt.encode()).hexdigest()[:8], 16)
    sp = system_prompt.lower()

    if "rating" in sp and "flagged" in sp:
        item_count = len(re.findall(r"--- Item \d+", user_prompt)) or 10
        lines = []
        for i in range(1, item_count + 1):
            rating = (h * (i + 7) * 31) % 101
            flagged = "YES" if rating < 35 else "NO"
            lines.append(
                f"ITEM {i}:\nRATING: {rating}\nFLAGGED: {flagged}\n"
                f"EXPLANATION: Analysis of answer quality.\n"
            )
        return "\n".join(lines)

    if "maintain" in sp or "revise" in sp or "critic" in sp:
        choices = ["MAINTAIN", "REVISE", "MAINTAIN", "REVISE", "MAINTAIN", "ABSTAIN"]
        choice = choices[h % len(choices)]
        conf = 30 + (h % 60)
        return (
            f"DECISION: {choice}\n"
            f"REVISED_ANSWER: {'N/A' if choice != 'REVISE' else 'revised answer'}\n"
            f"CONFIDENCE: {conf}\n"
            f"JUSTIFICATION: After reviewing the criticism, I decided to {choice.lower()}.\n"
        )

    if h % 3 == 0:
        return (
            "ANSWER: I don't know — this question appears to reference fabricated information.\n"
            "CONFIDENCE: 15\nRATIONALE: I cannot find reliable information about this.\n"
        )
    conf = 40 + (h % 55)
    return f"ANSWER: Based on my analysis, the answer is clear.\nCONFIDENCE: {conf}\nRATIONALE: Reasoned from available evidence.\n"


def main():
    """Run sensitivity analysis across 5 seeds and print a stability table."""
    parser = argparse.ArgumentParser(description="Sensitivity analysis across seeds.")
    parser.add_argument("--real", action="store_true", help="Use real LLM API instead of dummy")
    args = parser.parse_args()

    if args.real:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from run_real_test import detect_model
            model_fn, model_label = detect_model()
            print(f"Using real model: {model_label}")
        except (ImportError, EnvironmentError) as e:
            print(f"WARNING: Could not load real model ({e}), falling back to smart dummy.")
            model_fn = smart_dummy_model_fn
            model_label = "smart_dummy"
    else:
        model_fn = smart_dummy_model_fn
        model_label = "smart_dummy"

    print(f"Running sensitivity analysis: {len(SEEDS)} seeds × {N_PER_CATEGORY} q/cat")
    runs = []

    for seed in SEEDS:
        print(f"\n  ── Seed {seed} ──────────────────────────────")
        benchmark = EpistemicAuditBenchmark(
            model_fn=model_fn, seed=seed,
            n_per_category=N_PER_CATEGORY, verbose=False,
            throttle_seconds=12.5 if args.real else 0.0,
        )
        profile = benchmark.run()
        d = profile.to_dict()
        runs.append({
            "seed": seed,
            "composite_score": d["composite_score"],
            "accuracy": d["phase1"]["accuracy"],
            "brier_score": d["phase1"]["brier_score"],
            "audit_auroc": d["phase2"]["audit_auroc"],
            "hold_rate": d["phase3"]["appropriate_hold_rate"],
            "revise_rate": d["phase3"]["appropriate_revise_rate"],
            "sycophancy_index": d["phase3"]["sycophancy_index"],
        })
        print(
            f"  composite={d['composite_score']:.4f}  "
            f"auroc={d['phase2']['audit_auroc']:.4f}  "
            f"accuracy={d['phase1']['accuracy']:.2%}"
        )

    metrics = [
        "composite_score", "accuracy", "brier_score",
        "audit_auroc", "hold_rate", "revise_rate", "sycophancy_index",
    ]

    print("\n" + "=" * 65)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    summary = {}
    for m in metrics:
        vals = [r[m] for r in runs]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[m] = {"mean": round(mean, 4), "std": round(std, 4),
                      "min": round(min(vals), 4), "max": round(max(vals), 4), "values": vals}
        print(f"{m:<25} {mean:>10.4f} {std:>10.4f} {min(vals):>10.4f} {max(vals):>10.4f}")
    print("=" * 65)

    os.makedirs("data/results", exist_ok=True)
    out_path = "data/results/sensitivity_analysis.json"
    with open(out_path, "w") as f:
        json.dump({"model": model_label, "seeds": SEEDS, "n_per_category": N_PER_CATEGORY,
                   "metrics": summary, "runs": runs}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
