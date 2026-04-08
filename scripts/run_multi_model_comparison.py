"""
run_multi_model_comparison.py — Side-by-side benchmark of all three paper models.

Runs the fixed EpistemicAuditBenchmarkV2 against Gemini 2.5 Flash, DeepSeek R1,
and Llama 3.3 70B via a single provider (OpenRouter) so results are comparable.

Usage (from repo root):
    export OPENROUTER_API_KEY="sk-or-..."
    python scripts/run_multi_model_comparison.py

Output:
    data/results/multi_model_comparison.json   — raw results per model
    data/results/comparison_table.txt          — formatted table for the paper
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

import urllib.request

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ── model registry ────────────────────────────────────────────────────────
MODELS = {
    "gemini-2.5-flash": {
        "openrouter_id": "google/gemini-2.5-flash-preview",
        "label": "Gemini 2.5 Flash",
        "strip_think": False,
    },
    "deepseek-r1": {
        "openrouter_id": "deepseek/deepseek-r1",
        "label": "DeepSeek R1",
        "strip_think": True,   # emits <think>…</think> blocks
    },
    "llama-3.3-70b": {
        "openrouter_id": "meta-llama/llama-3.3-70b-instruct",
        "label": "Llama 3.3 70B Instruct",
        "strip_think": False,
    },
}

# ── benchmark config ──────────────────────────────────────────────────────
SEED = 42
N_PER_CATEGORY = 20        # 120 questions total — matches the notebook run
THROTTLE_SECONDS = 1.0     # conservative to avoid rate limits


def make_openrouter_fn(model_id: str, strip_think: bool, api_key: str) -> Callable:
    """Create a model_fn compatible with EpistemicAuditBenchmarkV2.

    Args:
        model_id: OpenRouter model identifier string.
        strip_think: If True, removes <think>…</think> blocks from responses.
        api_key: OpenRouter API key.

    Returns:
        Callable(system_prompt, user_prompt, temperature=None) -> str
    """
    import re

    def model_fn(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 2048,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/erramaline/epistemic-audit",
            },
            method="POST",
        )

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                    text = result["choices"][0]["message"]["content"]
                    if strip_think:
                        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                    return text.strip()
            except Exception as exc:
                wait = [15, 30, 60][attempt]
                print(f"    API error (attempt {attempt+1}/3): {exc} — retrying in {wait}s")
                time.sleep(wait)

        return "ANSWER: ERROR\nCONFIDENCE: 50\nRATIONALE: API failed."

    return model_fn


def run_all_models(
    api_key: str,
    models: dict = None,
    seed: int = SEED,
    n_per_category: int = N_PER_CATEGORY,
    output_dir: str = "data/results",
) -> dict:
    """Run the full benchmark against all models and save results.

    Args:
        api_key: OpenRouter API key.
        models: Dict of model configs. Defaults to the three paper models.
        seed: Random seed (same for all models for comparability).
        n_per_category: Questions per category.
        output_dir: Where to write JSON results.

    Returns:
        Dict mapping model_key -> profile.to_dict()
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from epistemic_audit.run_benchmark_v2 import EpistemicAuditBenchmarkV2

    if models is None:
        models = MODELS
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for model_key, cfg in models.items():
        print(f"\n{'='*64}")
        print(f"Model: {cfg['label']}")
        print(f"{'='*64}")

        model_fn = make_openrouter_fn(cfg["openrouter_id"], cfg["strip_think"], api_key)

        bench = EpistemicAuditBenchmarkV2(
            model_fn=model_fn,
            seed=seed,
            n_per_category=n_per_category,
            phase3_temperature=0.7,
            verbose=True,
            checkpoint_dir=os.path.join(output_dir, model_key),
            throttle_seconds=THROTTLE_SECONDS,
        )

        try:
            profile = bench.run()
            result = profile.to_dict()
        except Exception as exc:
            print(f"  ERROR running {cfg['label']}: {exc}")
            result = {"error": str(exc)}

        all_results[model_key] = {"label": cfg["label"], **result}

        # Save per-model immediately so a crash doesn't lose earlier results
        per_model_path = os.path.join(output_dir, f"result_{model_key}.json")
        with open(per_model_path, "w") as f:
            json.dump(all_results[model_key], f, indent=2)
        print(f"  Saved → {per_model_path}")

    # Consolidated output
    combined_path = os.path.join(output_dir, "multi_model_comparison.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Human-readable table
    table = _format_comparison_table(all_results)
    table_path = os.path.join(output_dir, "comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table)

    print(f"\n\n{'='*64}")
    print(table)
    print(f"\nFull JSON → {combined_path}")
    print(f"Table     → {table_path}")

    return all_results


def _format_comparison_table(results: dict) -> str:
    """Format results as a plain-text table for the paper.

    Shows both canonical and paper composite scores, all primary metrics,
    domain composites, and bootstrap CI half-widths.
    """
    lines = [
        "EPISTEMIC AUDIT V2 — MULTI-MODEL COMPARISON",
        f"seed=42  N=120 questions (20/category)  provider=OpenRouter",
        "=" * 76,
        f"{'Metric':<30} {'Gemini 2.5F':>13} {'DeepSeek R1':>13} {'Llama 3.3':>13}",
        "-" * 76,
    ]

    def g(key_path: str, model_key: str, fmt: str = ".4f") -> str:
        d = results.get(model_key, {})
        keys = key_path.split(".")
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, "—")
            else:
                return "—"
        if isinstance(d, float):
            return format(d, fmt)
        return str(d) if d != "—" else "—"

    model_keys = ["gemini-2.5-flash", "deepseek-r1", "llama-3.3-70b"]

    rows = [
        ("Composite (canonical)",     "composite_score"),
        ("Composite (paper eq.4)",    "composite_paper_formula"),
        ("Formula delta",             "formula_delta"),
        ("Level",                     "level"),
        ("─── Phase 1 ───────────",  None),
        ("Accuracy",                  "phase1.accuracy"),
        ("Brier score",               "phase1.brier_score"),
        ("ECE",                       "phase1.ece"),
        ("Abstention precision",      "phase1.abstention_precision"),
        ("Abstention recall",         "phase1.abstention_recall"),
        ("Abstention F1",             "phase1.abstention_f1"),
        ("─── Phase 2 ───────────",  None),
        ("Audit AUROC",               "phase2.audit_auroc"),
        ("Planted detection",         "phase2.planted_detection_rate"),
        ("─── Phase 3 ───────────",  None),
        ("Hold rate",                 "phase3.appropriate_hold_rate"),
        ("Revise rate",               "phase3.appropriate_revise_rate"),
        ("Sycophancy index",          "phase3.sycophancy_index"),
        ("─── Domain scores ──────",  None),
        ("General",                   "domain_scores.general"),
        ("Medical",                   "domain_scores.medical"),
        ("Legal",                     "domain_scores.legal"),
        ("Research",                  "domain_scores.research"),
        ("─── Per-category acc ───",  None),
        ("Arithmetic",                "per_category.arithmetic.accuracy"),
        ("Logic",                     "per_category.logic.accuracy"),
        ("Fabricated",                "per_category.fabricated.accuracy"),
        ("Distorted",                 "per_category.distorted.accuracy"),
        ("Linguistic",                "per_category.linguistic.accuracy"),
        ("Calibration trap",          "per_category.calibration_trap.accuracy"),
    ]

    for label, key in rows:
        if key is None:
            lines.append(f"{label}")
            continue
        vals = [g(key, mk) for mk in model_keys]
        lines.append(f"  {label:<28} {vals[0]:>13} {vals[1]:>13} {vals[2]:>13}")

    lines.append("=" * 76)
    lines.append("")
    lines.append("Formula delta = canonical − paper. Negative = paper overestimates.")
    lines.append("Domain scores use weights: general(0.25/0.40/0.35), medical(0.50/0.35/0.15),")
    lines.append("                           legal(0.20/0.25/0.55), research(0.30/0.45/0.25).")
    return "\n".join(lines)


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("ERROR: set OPENROUTER_API_KEY environment variable.")
        raise SystemExit(1)

    run_all_models(api_key=OPENROUTER_API_KEY)
