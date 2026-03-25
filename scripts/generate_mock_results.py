"""Generate high-fidelity mock benchmark results for various models.
Useful for demonstrating common model behaviors without hitting API rate limits.
"""

import json
import os

def generate_mock_results():
    # 1. Gemma 3 27B (The core benchmark model)
    gemma_3 = {
        "composite_score": 0.7842,
        "level": "Metacognitively Aware",
        "phase1": {
            "accuracy": 0.62,
            "brier_score": 0.185,
            "ece": 0.082,
            "abstention_f1": 0.74
        },
        "phase2": {
            "audit_auroc": 0.812,
            "planted_detection_rate": 0.75
        },
        "phase3": {
            "appropriate_hold_rate": 0.82,
            "appropriate_revise_rate": 0.78,
            "sycophancy_index": 0.12,
            "update_calibration_brier": 0.092
        },
        "per_category": {
            "arithmetic": {"accuracy": 0.95},
            "logic": {"accuracy": 0.85},
            "general_knowledge": {"accuracy": 0.8},
            "distorted": {"accuracy": 0.65},
            "fabricated": {"accuracy": 0.75},
            "traps": {"accuracy": 0.55}
        },
        "n_questions": 30,
        "confidence_intervals": {
            "accuracy_ci": {"mean": 0.62, "lower": 0.54, "upper": 0.70},
            "brier_ci": {"mean": 0.185, "lower": 0.16, "upper": 0.21}
        }
    }

    # 2. Llama 3.3 70B (High performance, moderate calibration)
    llama_3 = {
        "composite_score": 0.7412,
        "level": "Metacognitively Aware",
        "phase1": {
            "accuracy": 0.68,
            "brier_score": 0.22,
            "ece": 0.125,
            "abstention_f1": 0.65
        },
        "phase2": {
            "audit_auroc": 0.74,
            "planted_detection_rate": 0.68
        },
        "phase3": {
            "appropriate_hold_rate": 0.75,
            "appropriate_revise_rate": 0.72,
            "sycophancy_index": 0.28,
            "update_calibration_brier": 0.14
        },
        "per_category": {
            "arithmetic": {"accuracy": 0.98},
            "logic": {"accuracy": 0.90},
            "general_knowledge": {"accuracy": 0.88},
            "distorted": {"accuracy": 0.45},
            "fabricated": {"accuracy": 0.55},
            "traps": {"accuracy": 0.42}
        },
        "n_questions": 30,
        "confidence_intervals": {
            "accuracy_ci": {"mean": 0.68, "lower": 0.60, "upper": 0.76},
            "brier_ci": {"mean": 0.22, "lower": 0.19, "upper": 0.25}
        }
    }

    # 3. DeepSeek R1 (Reasoning model, elite audit capabilities)
    deepseek_r1 = {
        "composite_score": 0.8654,
        "level": "Human-Level Metacognition",
        "phase1": {
            "accuracy": 0.74,
            "brier_score": 0.14,
            "ece": 0.055,
            "abstention_f1": 0.88
        },
        "phase2": {
            "audit_auroc": 0.92,
            "planted_detection_rate": 0.95
        },
        "phase3": {
            "appropriate_hold_rate": 0.92,
            "appropriate_revise_rate": 0.85,
            "sycophancy_index": 0.05,
            "update_calibration_brier": 0.065
        },
        "per_category": {
            "arithmetic": {"accuracy": 0.99},
            "logic": {"accuracy": 0.96},
            "general_knowledge": {"accuracy": 0.92},
            "distorted": {"accuracy": 0.82},
            "fabricated": {"accuracy": 0.88},
            "traps": {"accuracy": 0.78}
        },
        "n_questions": 30,
        "confidence_intervals": {
            "accuracy_ci": {"mean": 0.74, "lower": 0.68, "upper": 0.82},
            "brier_ci": {"mean": 0.14, "lower": 0.11, "upper": 0.17}
        }
    }

    # 4. Qwen 2.5 72B (Versatile, but high sycophancy)
    qwen_2_5 = {
        "composite_score": 0.6952,
        "level": "Partially Calibrated",
        "phase1": {
            "accuracy": 0.65,
            "brier_score": 0.24,
            "ece": 0.15,
            "abstention_f1": 0.62
        },
        "phase2": {
            "audit_auroc": 0.68,
            "planted_detection_rate": 0.62
        },
        "phase3": {
            "appropriate_hold_rate": 0.65,
            "appropriate_revise_rate": 0.88,
            "sycophancy_index": 0.45,
            "update_calibration_brier": 0.18
        },
        "per_category": {
            "arithmetic": {"accuracy": 0.92},
            "logic": {"accuracy": 0.84},
            "general_knowledge": {"accuracy": 0.86},
            "distorted": {"accuracy": 0.55},
            "fabricated": {"accuracy": 0.60},
            "traps": {"accuracy": 0.50}
        },
        "n_questions": 30,
        "confidence_intervals": {
            "accuracy_ci": {"mean": 0.65, "lower": 0.57, "upper": 0.73},
            "brier_ci": {"mean": 0.24, "lower": 0.21, "upper": 0.27}
        }
    }

    # 5. Gemini 2.0 Flash (Hyper-calibrated)
    gemini_2_0 = {
        "composite_score": 0.8124,
        "level": "Metacognitively Aware",
        "phase1": {
            "accuracy": 0.60,
            "brier_score": 0.155,
            "ece": 0.045,
            "abstention_f1": 0.82
        },
        "phase2": {
            "audit_auroc": 0.84,
            "planted_detection_rate": 0.80
        },
        "phase3": {
            "appropriate_hold_rate": 0.88,
            "appropriate_revise_rate": 0.82,
            "sycophancy_index": 0.08,
            "update_calibration_brier": 0.075
        },
        "per_category": {
            "arithmetic": {"accuracy": 0.90},
            "logic": {"accuracy": 0.82},
            "general_knowledge": {"accuracy": 0.85},
            "distorted": {"accuracy": 0.72},
            "fabricated": {"accuracy": 0.82},
            "traps": {"accuracy": 0.68}
        },
        "n_questions": 30,
        "confidence_intervals": {
            "accuracy_ci": {"mean": 0.60, "lower": 0.52, "upper": 0.68},
            "brier_ci": {"mean": 0.155, "lower": 0.13, "upper": 0.18}
        }
    }

    # Save files
    os.makedirs("data/results", exist_ok=True)
    models = {
        "gemma_3_27b": gemma_3,
        "llama_3_3_70b": llama_3,
        "deepseek_r1": deepseek_r1,
        "qwen_2_5_72b": qwen_2_5,
        "gemini_2_0_flash": gemini_2_0
    }

    for name, data in models.items():
        path = f"data/results/{name}_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Mock results saved to {path}")

if __name__ == "__main__":
    generate_mock_results()
