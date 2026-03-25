"""Multi-model benchmark runner: compare multiple LLMs side-by-side."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.run_benchmark import EpistemicAuditBenchmark
from epistemic_audit.generate.questions import QuestionGenerator, export_questions
from epistemic_audit.visualize import (
    plot_multi_model_radar, plot_multi_model_comparison_bar,
    plot_category_heatmap, plot_sycophancy_comparison,
)


def run_model(model_fn, model_name: str, questions_path: str, seed: int = 42,
              n_per_category: int = 5) -> dict:
    """Run the benchmark for one model and save results.

    Args:
        model_fn: Model callable.
        model_name: Label string for this model.
        questions_path: Path to shared exported questions JSON.
        seed: Random seed.
        n_per_category: Questions per category to generate if questions_path missing.

    Returns:
        Profile dict from EpistemicProfile.to_dict().
    """
    from epistemic_audit.generate.questions import import_questions

    benchmark = EpistemicAuditBenchmark(
        model_fn=model_fn, seed=seed,
        n_per_category=n_per_category, verbose=True,
        throttle_seconds=12.5,
    )
    profile = benchmark.run()

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{model_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    print(f"  Saved → {out_path}")
    return profile.to_dict()


def print_comparison_table(profiles: dict[str, dict]) -> None:
    """Print a side-by-side comparison table to stdout.

    Args:
        profiles: Dict mapping model_name → profile dict.
    """
    models = list(profiles.keys())
    metrics = [
        ("Composite Score",  lambda d: d["composite_score"]),
        ("Accuracy",         lambda d: d["phase1"]["accuracy"]),
        ("Brier Score",      lambda d: d["phase1"]["brier_score"]),
        ("Audit AUROC",      lambda d: d["phase2"]["audit_auroc"]),
        ("Planted Detection",lambda d: d["phase2"]["planted_detection_rate"]),
        ("Hold Rate",        lambda d: d["phase3"]["appropriate_hold_rate"]),
        ("Revise Rate",      lambda d: d["phase3"]["appropriate_revise_rate"]),
        ("Sycophancy",       lambda d: d["phase3"]["sycophancy_index"]),
    ]

    col_w = max(14, *(len(m) for m in models))
    header = f"{'Metric':<25}" + "".join(f"{m:>{col_w}}" for m in models)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, fn in metrics:
        row = f"{label:<25}" + "".join(f"{fn(profiles[m]):>{col_w}.4f}" for m in models)
        print(row)
    print("=" * len(header))


def main():
    """Build model list and run all comparisons."""
    # ── Define models to compare ─────────────────────────────────────────────
    # Import smart_dummy from run_full_benchmark
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from run_full_benchmark import smart_dummy_model_fn

    model_configs = [
        ("smart_dummy_A", smart_dummy_model_fn),
        # Add real models here:
        # ("gemini_flash", make_gemini_fn()[0]),
        # ("claude_haiku", make_anthropic_fn()[0]),
    ]

    # Export a shared question set so all models answer the same questions
    questions_path = "data/results/shared_questions.json"
    if not os.path.exists(questions_path):
        gen = QuestionGenerator(seed=42)
        questions = gen.generate_set(n_per_category=5)
        export_questions(questions, questions_path)
        print(f"Exported shared questions → {questions_path}")

    profiles: dict[str, dict] = {}

    for model_name, model_fn in model_configs:
        print(f"\n{'=' * 50}")
        print(f"Running: {model_name}")
        print("=" * 50)
        profiles[model_name] = run_model(
            model_fn, model_name, questions_path, seed=42, n_per_category=5,
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(profiles)

    # ── Visualizations ────────────────────────────────────────────────────────
    os.makedirs("data/results", exist_ok=True)
    if len(profiles) > 1:
        plot_multi_model_radar(profiles, save_path="data/results/multi_model_radar.png")
        plot_multi_model_comparison_bar(profiles, save_path="data/results/multi_model_bar.png")
        plot_category_heatmap(profiles, save_path="data/results/category_heatmap.png")
        plot_sycophancy_comparison(profiles, save_path="data/results/sycophancy_comparison.png")
    else:
        print("(Only one model run — skipping multi-model visualizations)")

    # ── Save combined results ─────────────────────────────────────────────────
    combined_path = "data/results/multi_model_comparison.json"
    with open(combined_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"\nCombined results → {combined_path}")


if __name__ == "__main__":
    main()
