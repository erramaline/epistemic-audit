"""Automate benchmark runs for multiple models via OpenRouter for comparison."""

import json
import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.run_benchmark import EpistemicAuditBenchmark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_openrouter_fn(model_id: str):
    """Build a model_fn for a specific OpenRouter model."""
    import openai
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Fallback to the one provided in the prompt if not in env
        api_key = "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"

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    def model_fn(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    return model_fn

def run_model_benchmark(model_id: str, label: str):
    """Run full benchmark for a single model and save results."""
    logger.info(f"\n" + "="*60)
    logger.info(f"RUNNING BENCHMARK FOR: {label} ({model_id})")
    logger.info("="*60 + "\n")

    # Use a faster run for comparison (5 per category = 30 questions)
    benchmark = EpistemicAuditBenchmark(
        model_fn=make_openrouter_fn(model_id),
        seed=42,
        n_per_category=5,  # 30 total questions
        verbose=True,
        checkpoint_dir=f"data/results/{label.lower().replace(' ', '_')}",
        throttle_seconds=1.0,  # Respect free tier rate limits
    )

    profile = benchmark.run()
    
    os.makedirs("data/results", exist_ok=True)
    out_path = f"data/results/{label.lower().replace(' ', '_')}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)
    
    logger.info(f"\n[DONE] Results for {label} saved to {out_path}")
    return out_path

def main():
    models = [
        {"id": "google/gemma-3-27b-it:free", "label": "Gemma 3 27B"},
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
        {"id": "deepseek/deepseek-r1:free", "label": "DeepSeek R1"},
        {"id": "qwen/qwen-2.5-72b-instruct:free", "label": "Qwen 2.5 72B"},
        {"id": "google/gemini-2.0-flash-exp:free", "label": "Gemini 2.0 Flash"},
    ]

    results_paths = []
    for model in models:
        try:
            path = run_model_benchmark(model["id"], model["label"])
            results_paths.append(path)
            # Short cooldown between models
            time.sleep(5)
        except Exception as e:
            logger.error(f"Failed to run benchmark for {model['label']}: {e}")

    logger.info("\n" + "="*60)
    logger.info("ALL BENCHMARK RUNS COMPLETED")
    logger.info(f"Results available in: {', '.join(results_paths)}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
