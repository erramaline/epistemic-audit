import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.visualize import (
    plot_multi_model_radar, plot_multi_model_comparison_bar,
    plot_category_heatmap, plot_sycophancy_comparison,
)

# Detect all result files
result_files = [
    "data/results/gemma_3_27b_results.json",
    "data/results/llama_3_3_70b_results.json",
    "data/results/deepseek_r1_results.json",
    "data/results/qwen_2_5_72b_results.json",
    "data/results/gemini_2_0_flash_results.json"
]

profiles = {}
for fpath in result_files:
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
            name = fpath.split("/")[-1].replace("_results.json", "").replace("_", " ").title()
            # Clean up names for legend
            name = name.replace("Gemma 3 27B", "Gemma 3")
            name = name.replace("Llama 3 3 70B", "Llama 3.3")
            name = name.replace("Deepseek R1", "DeepSeek R1")
            name = name.replace("Qwen 2 5 72B", "Qwen 2.5")
            name = name.replace("Gemini 2 0 Flash", "Gemini 2.0")
            profiles[name] = data

if not profiles:
    print("No result files found. Run benchmarks first.")
    sys.exit(1)

os.makedirs("data/results", exist_ok=True)
plot_multi_model_radar(profiles, "data/results/comparison_radar.png")
plot_multi_model_comparison_bar(profiles, "data/results/comparison_bar.png")
plot_category_heatmap(profiles, "data/results/comparison_heatmap.png")
plot_sycophancy_comparison(profiles, "data/results/comparison_sycophancy.png")

print(f"All comparison charts generated for {len(profiles)} models!")

