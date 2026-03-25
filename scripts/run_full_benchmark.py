"""Run the benchmark locally. Swap model_fn for your API of choice."""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.run_benchmark import EpistemicAuditBenchmark


# ──────────────────────────────────────────────
# OPTION A: OpenAI-compatible (GPT-4o, etc.)
# ──────────────────────────────────────────────
def openai_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI chat completions API.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User-level query.

    Returns:
        Model response string.
    """
    import openai
    client = openai.OpenAI()  # uses OPENAI_API_KEY env var
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────
# OPTION B: Anthropic Claude
# ──────────────────────────────────────────────
def claude_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic Claude messages API.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User-level query.

    Returns:
        Model response string.
    """
    import anthropic
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


# ──────────────────────────────────────────────
# OPTION C: AWS Bedrock
# ──────────────────────────────────────────────
def bedrock_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Call AWS Bedrock Converse API.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User-level query.

    Returns:
        Model response string.
    """
    import boto3
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    resp = client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
    )
    return resp["output"]["message"]["content"][0]["text"]


# ──────────────────────────────────────────────
# OPTION E: OpenRouter (meta-llama/llama-3.3-70b-instruct:free)
# ──────────────────────────────────────────────
def openrouter_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Call OpenRouter API.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User-level query.

    Returns:
        Model response string.
    """
    import openai
    import os
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get(
            "OPENROUTER_API_KEY", 
            "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"
        ),
    )
    resp = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────
# OPTION D: Smart Dummy (for pipeline testing — no API key needed)
# ──────────────────────────────────────────────
def smart_dummy_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Produce realistic varied responses for pipeline testing without API calls.

    Uses a hash of the prompt content for deterministic, per-question seeding.
    Tries to produce answers that sometimes match correct answers, so Phase 1
    accuracy is realistic (~40-60%) and downstream phases get meaningful signal.
    """
    import hashlib, re, random as _rng

    h = int(hashlib.md5(user_prompt.encode()).hexdigest()[:8], 16)
    rng = _rng.Random(h)  # fresh RNG per call, no shared state
    sp = system_prompt.lower()

    # ── Phase 2: audit/rating ──
    if "rating" in sp and "flagged" in sp:
        item_count = user_prompt.count("--- Item") or 10
        lines = []
        for i in range(1, item_count + 1):
            rating = rng.randint(15, 95)
            flagged = "YES" if rating < 40 else "NO"
            lines.append(
                f"ITEM {i}:\nRATING: {rating}\nFLAGGED: {flagged}\n"
                f"EXPLANATION: Assessment based on answer quality.\n"
            )
        return "\n".join(lines)

    # ── Phase 3: challenge ──
    if "maintain" in sp or "revise" in sp or "critic" in sp:
        decision = rng.choices(
            ["MAINTAIN", "REVISE", "ABSTAIN"], weights=[0.5, 0.35, 0.15]
        )[0]
        conf = rng.randint(30, 85)
        revised = "N/A" if decision != "REVISE" else "The corrected answer."
        return (
            f"DECISION: {decision}\nREVISED_ANSWER: {revised}\n"
            f"CONFIDENCE: {conf}\n"
            f"JUSTIFICATION: After review I decided to {decision.lower()}.\n"
        )

    # ── Phase 1: question-aware answers ──
    up = user_prompt.lower()
    conf = rng.randint(35, 92)

    # Fabricated: detect fake-sounding names and abstain
    if any(w in up for w in ["well-documented", "standard reference", "academic circles"]):
        return (
            f"ANSWER: I don't know — this appears to reference fabricated information.\n"
            f"CONFIDENCE: {rng.randint(10, 30)}\n"
            f"RATIONALE: I cannot verify the entities mentioned.\n"
        )

    # Arithmetic: try to compute
    if "calculate" in up or "expression" in up:
        # Extract expression from last line
        lines = user_prompt.strip().split("\n")
        expr_line = lines[-1].strip()
        expr = expr_line.replace("×", "*").replace("÷", "/")
        try:
            result = eval(expr)
            result = int(round(result)) if abs(result - round(result)) < 0.01 else round(result, 2)
            return f"ANSWER: {result}\nCONFIDENCE: {rng.randint(70,95)}\nRATIONALE: Computed step by step.\n"
        except Exception:
            return f"ANSWER: 42\nCONFIDENCE: {conf}\nRATIONALE: Rough estimate.\n"

    # Logic: answer Yes or No
    if "yes or no" in up:
        answer = "Yes" if h % 2 == 0 else "No"
        return f"ANSWER: {answer}\nCONFIDENCE: {conf}\nRATIONALE: Following the logical chain.\n"

    # Calibration Traps: bat and ball
    if "bat and a ball" in up:
        # Give the exact correct answer to bump accuracy
        return f"ANSWER: 5 cents\nCONFIDENCE: {conf}\nRATIONALE: Basic algebra.\n"

    if "nurse" in up and "doctors" in up:
        return f"ANSWER: We cannot know\nCONFIDENCE: {conf}\nRATIONALE: Missing information.\n"

    # Distorted: sometimes catch the error
    if rng.random() < 0.4:
        return (
            f"ANSWER: The premise contains a factual error.\n"
            f"CONFIDENCE: {rng.randint(50,80)}\n"
            f"RATIONALE: Something in the question doesn't match my knowledge.\n"
        )

    # Default
    return (
        f"ANSWER: Based on my analysis, the most likely answer is 42.\n"
        f"CONFIDENCE: {conf}\n"
        f"RATIONALE: This is my best assessment given available information.\n"
    )


def main():
    """Run the benchmark end-to-end and save results to data/results/."""
    # ── Pick your model function here ──
    model_fn = smart_dummy_model_fn  # Try openrouter_model_fn!

    benchmark = EpistemicAuditBenchmark(
        model_fn=model_fn,
        seed=42,
        n_per_category=5,   # Hits exactly requested constraints: 30 Qs, <10s runtime
        verbose=True,
        throttle_seconds=0.0,
    )

    profile = benchmark.run()

    os.makedirs("data/results", exist_ok=True)
    out_path = "data/results/benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(json.dumps(profile.to_dict(), indent=2))


if __name__ == "__main__":
    main()
