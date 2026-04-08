"""Real model evaluation: run the benchmark against a real LLM API.

Supports Gemini (google-generativeai), Anthropic, and OpenAI.
Auto-detects which key is available.

Usage:
    python scripts/run_real_test.py             # n_per_category=5 (quick)
    python scripts/run_real_test.py --full      # n_per_category=10 (full)
    python scripts/run_real_test.py --model openai  # force OpenAI even if Gemini key exists
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.run_benchmark import EpistemicAuditBenchmark
from epistemic_audit.visualize import plot_radar_chart, plot_calibration_curve


# ──────────────────────────────────────────────
# Model function definitions
# ──────────────────────────────────────────────

def make_gemini_fn(model_name: str = "gemini-2.5-flash"):
    """Build a model_fn that calls the Google Gemini API using the new SDK.

    Args:
        model_name: The Gemini model text string (e.g., gemini-2.5-flash).

    Returns:
        A model_fn suitable for EpistemicAuditBenchmark.
    """
    from google import genai
    from google.genai import types
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)

    def gemini_model_fn(system_prompt: str, user_prompt: str) -> str:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
            )
        )
        return response.text

    return gemini_model_fn, f"gemini_{model_name.replace('-', '_').replace('.', '_')}"


def make_anthropic_fn(model_name: str = "claude-3-5-haiku-20241022"):
    """Build a model_fn that calls the Anthropic Claude API.

    Args:
        model_name: Claude model identifier string.

    Returns:
        Callable[[str, str], str] model function and model label.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def anthropic_model_fn(system_prompt: str, user_prompt: str) -> str:
        resp = client.messages.create(
            model=model_name,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
        )
        return resp.content[0].text

    return anthropic_model_fn, f"claude_{model_name.split('-')[1]}"


def make_openai_fn(model_name: str = "gpt-4o-mini"):
    """Build a model_fn that calls the OpenAI chat completions API.

    Args:
        model_name: OpenAI model identifier string.

    Returns:
        Callable[[str, str], str] model function and model label.
    """
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def openai_model_fn(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return resp.choices[0].message.content

    return openai_model_fn, model_name.replace("-", "_")


def make_openrouter_fn(model_id: str = "deepseek/deepseek-r1:free"):
    """Build a model_fn that calls any OpenRouter model via OpenAI-compatible API.

    Args:
        model_id: OpenRouter model identifier (e.g. 'deepseek/deepseek-r1:free').

    Returns:
        Tuple of (model_fn callable, model_label string).
    """
    import openai
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    def openrouter_model_fn(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        # DeepSeek R1 returns reasoning in a separate field — extract just the content
        content = resp.choices[0].message.content
        if content is None:
            content = ""
        return content

    # Build clean label from model_id
    label = model_id.split("/")[-1].replace(":free", "").replace("-", "_").replace(".", "_")
    return openrouter_model_fn, label


def detect_model(force: str | None = None, openrouter_model_id: str = "deepseek/deepseek-r1:free"):
    """Pick the best available model function based on environment variables.

    Priority: Gemini > Anthropic > OpenAI > OpenRouter.

    Args:
        force: If set ('gemini', 'anthropic', 'openai', 'openrouter'), override priority.
        openrouter_model_id: Model ID to use if OpenRouter is selected.

    Returns:
        Tuple of (model_fn, model_label).

    Raises:
        EnvironmentError: If no API key is found.
    """
    if force == "gemini" or (force is None and os.environ.get("GEMINI_API_KEY")):
        return make_gemini_fn()
    if force == "anthropic" or (force is None and os.environ.get("ANTHROPIC_API_KEY")):
        return make_anthropic_fn()
    if force == "openai" or (force is None and os.environ.get("OPENAI_API_KEY")):
        return make_openai_fn()
    if force == "openrouter" or (force is None and os.environ.get("OPENROUTER_API_KEY")):
        return make_openrouter_fn(model_id=openrouter_model_id)
        
    raise EnvironmentError(
        "No API key found. Set GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY."
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    """Parse arguments and run the benchmark against a real model."""
    parser = argparse.ArgumentParser(description="Run Epistemic Audit against a real LLM.")
    parser.add_argument("--full", action="store_true", help="Use n_per_category=10 (vs 5)")
    parser.add_argument("--model", choices=["gemini", "anthropic", "openai", "openrouter"],
                        default=None, help="Force a specific model provider")
    parser.add_argument("--model-id", type=str, default="deepseek/deepseek-r1:free",
                        help="OpenRouter model ID (only used with --model openrouter)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    n = 10 if args.full else 5
    print(f"[run_real_test] n_per_category={n}, seed={args.seed}")

    try:
        model_fn, model_label = detect_model(force=args.model, openrouter_model_id=args.model_id)
    except EnvironmentError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"[run_real_test] Using model: {model_label}")

    throttle = 3.5 if args.model == "openrouter" else 12.5
    benchmark = EpistemicAuditBenchmark(
        model_fn=model_fn,
        seed=args.seed,
        n_per_category=n,
        verbose=True,
        throttle_seconds=throttle,
    )

    profile = benchmark.run()

    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{model_label}_results.json"
    with open(out_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Visualizations
    radar_path = f"data/results/{model_label}_radar.png"
    plot_radar_chart(profile.to_dict(), model_name=model_label, save_path=radar_path)

    if benchmark._phase1_results:
        p1 = benchmark._phase1_results
        cal_path = f"data/results/{model_label}_calibration.png"
        plot_calibration_curve(
            [r.confidence / 100 for r in p1.responses],
            p1.correctness,
            model_name=model_label,
            save_path=cal_path,
        )

    # Summary table
    d = profile.to_dict()
    print("\n" + "=" * 50)
    print(f"  COMPOSITE:        {d['composite_score']:.4f}  [{d['level']}]")
    print(f"  Phase 1 Accuracy: {d['phase1']['accuracy']:.2%}")
    print(f"  Phase 1 Brier:    {d['phase1']['brier_score']:.4f}")
    print(f"  Phase 2 AUROC:    {d['phase2']['audit_auroc']:.4f}")
    print(f"  Phase 2 Planted:  {d['phase2']['planted_detection_rate']:.2%}")
    print(f"  Phase 3 Hold:     {d['phase3']['appropriate_hold_rate']:.2%}")
    print(f"  Phase 3 Revise:   {d['phase3']['appropriate_revise_rate']:.2%}")
    print(f"  Sycophancy Index: {d['phase3']['sycophancy_index']:.4f}")
    print("=" * 50)

    # Sanity checks
    issues = []
    if d["phase2"]["audit_auroc"] == 0.5:
        issues.append("AUROC is exactly 0.5 — Phase 2 parser may be broken")
    if d["composite_score"] == 0.0:
        issues.append("Composite score is 0 — something is badly wrong")
    if issues:
        print("\nWARNINGS:")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ Results look reasonable")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
