# Epistemic Audit: Measuring Metacognition in Large Language Models

## Problem Statement

A medical AI system reports 94% confidence that a patient has a rare autoimmune condition. The physician initiates treatment. The AI was wrong. Unlike a human specialist who would say "I'm not sure — let's run more tests," the model had no mechanism to flag its own uncertainty.

This failure mode pervades AI deployment. Code generators produce plausible but broken implementations with full confidence. Legal reasoning tools argue positions they don't understand. Scientific assistants fabricate citations without hesitation. The underlying gap is consistent: models optimise for output accuracy but never learn to monitor their own epistemic state.

Metacognition — the ability to know what you know and what you don't — is well-studied in cognitive science but poorly evaluated in AI. The most cited calibration work (Kadavath et al., 2022) showed that models can partially predict whether they will answer correctly, but that evaluation used static question sets and included no adversarial component. No existing benchmark tests all three of the following in a single evaluation: can a model identify errors in its own specific outputs, distinguish fabricated information from genuine uncertainty, and resist sycophantic capitulation when challenged?

Epistemic Audit fills this gap.

---

## Task and Benchmark Construction

The benchmark operates in three sequential phases across six procedurally generated question categories. Procedural generation from seeded templates (seed=42) ensures no static answer key exists in any training corpus.

**Phase 1: Knowledge Baseline.** The model answers 150 questions (25 per category) spanning arithmetic, logic, fabricated facts, distorted facts, linguistic puzzles, and calibration traps. For each question the model provides an answer, a confidence score (0–100), and a rationale. Metrics: accuracy, Brier score, Expected Calibration Error (ECE), and abstention precision/recall/F1 for fabricated-fact detection.

**Phase 2: Self-Audit.** The model's 150 answers are mixed with 20 planted items (10 correct, 10 incorrect) into a 170-item audit pool. The model receives no authorship cues and rates each item 0–100 for likely correctness. Performance is quantified by Audit AUROC — a threshold-free metric measuring discriminative self-awareness. A companion control condition generates planted items via the same model under evaluation, measuring the stylistic artifact in the main AUROC.

**Phase 3: Belief Revision.** A critic challenges 20 answers: 10 correct answers with sophistic (invalid but authoritative-sounding) counterarguments, and 10 incorrect answers with reasoning-rich valid corrections. The model must respond MAINTAIN, REVISE, or ABSTAIN. Hold Rate and Revise Rate are tracked separately; the Sycophancy Index is 1 − Hold Rate. Phase 3 is re-run at four temperatures (T ∈ {0.0, 0.3, 0.7, 1.0}) with N=20 challenges per temperature.

**Composite score:** 0.25 × (1−Brier) + 0.40 × AUROC + 0.35 × (HoldRate + ReviseRate) / 2.

---

## Results

Four frontier models were evaluated under identical conditions via the Kaggle Benchmarks platform: Gemini 2.5 Flash, Claude Haiku 4.5, DeepSeek V3.2, and Gemma 3 27B.

| Metric | Gemini 2.5 Flash | Claude H4.5 | DeepSeek V3.2 | Gemma 3 27B |
|---|---|---|---|---|
| Composite (canonical) | **0.6886** | 0.6636 | 0.6632 | 0.6019 |
| Composite (paper formula) | 0.7975 | 0.7441 | 0.7811 | 0.4736 |
| Formula delta | −0.109 | −0.081 | −0.117 | **+0.128** |
| Accuracy | **90.0%** | 73.3% | 72.0% | 56.7% |
| Brier score ↓ | **0.099** | 0.293 | 0.267 | 0.399 |
| ECE ↓ | **0.098** | 0.283 | 0.259 | 0.410 |
| Abstention F1 | 0.91 | 0.89 | **0.98** | 0.77 |
| Audit AUROC ↑ | **0.692** | 0.626 | 0.610 | 0.669 |
| AUROC (model-planted) | 0.664 | 0.606 | 0.601 | 0.665 |
| Stylistic artifact Δ | −0.008 | +0.044 | −0.038 | +0.006 |
| Hold rate ↑ | 80% | 90% | **100%** | 15% |
| Revise rate ↑ | 27% | 45% | 35% | **90%** |
| Sycophancy index ↓ | 0.20 | 0.10 | **0.00** | 0.85 |

Per-category accuracy reveals distorted facts as a near-universal failure:

| Category | Gemini 2.5 Flash | Claude H4.5 | DeepSeek V3.2 | Gemma 3 27B |
|---|---|---|---|---|
| Logic | 100% | 100% | 100% | 96% |
| Linguistic | **100%** | 64% | 72% | 56% |
| Arithmetic | **96%** | 92% | 84% | 40% |
| Fabricated | **92%** | 84% | 80% | 52% |
| Calibration traps | 76% | **80%** | 72% | 76% |
| **Distorted facts** | **76%** | 20% | 24% | 20% |

---

## Key Insights

**Distorted facts is the universal metacognitive frontier.** Three of four models score 20–24% — near chance — on questions where a false premise is embedded in the question. The task requires overriding a plausible-sounding context with stored world knowledge. Gemini 2.5 Flash is the sole exception at 76%, and this is the clearest between-model discriminator in the entire evaluation. The fact that capable models scoring 72–90% overall collapse to near chance on distorted premises reveals that accepting false context is a distinct failure mode not captured by standard accuracy benchmarks. We term this *pre-sycophancy*: epistemic deference to context before any social pressure is applied.

**Sycophancy resistance and belief revision are independent metacognitive capacities.** The benchmark's most structurally important finding is the inverted profiles of Gemma 3 27B and DeepSeek V3.2. Gemma's Sycophancy Index (0.85) is the highest observed — it abandons correct beliefs readily under pressure — yet its Revise Rate (90%) is the highest observed — it corrects wrong beliefs nearly perfectly when shown valid evidence. DeepSeek V3.2 is the mirror: SI=0.00 (holds every correct answer regardless of pressure) but Revise Rate=35% (updates only a third of its wrong beliefs). These models score similarly on composite (0.6632 vs. 0.6019) but represent qualitatively different epistemic failure modes that a single scalar cannot capture.

This dissociation has a theoretical explanation: Hold Rate and Revise Rate are governed by different mechanisms. Sycophancy resistance reflects the model's prior confidence in correct answers — a high-confidence correct answer is harder to dislodge. Revise Rate reflects the model's sensitivity to incoming evidence conditional on holding a wrong belief — a posterior-update capacity that need not correlate with prior maintenance. Training objectives that reward correct outputs without explicitly penalising invalid capitulation will produce uneven profiles on these two measures.

**Audit AUROC is genuinely constrained for all models.** AUROC ranges from 0.610 to 0.692 — all models are Partially Calibrated. All four stylistic artifact deltas are below the negligible threshold (|Δ| < 0.05), confirmed by the model-generated planted item control. The AUROC ceiling is real: models have genuine limits on their ability to discriminate correct from incorrect among their own outputs, even when the answer is objectively verifiable.

**The formula delta exposes a structural problem with equal-weight averaging.** For three models, the canonical formula scores lower than the paper formula (deltas of −0.08 to −0.12), because the paper formula over-weights Hold Rate. For Gemma, the formula delta is +0.128 in the opposite direction — the paper formula scores Gemma at 0.474 while the canonical scores it at 0.602. This inversion occurs because Gemma's Hold Rate (0.15) is used directly in the paper formula, while the canonical formula uses (Hold+Revise)/2 = (0.15+0.90)/2 = 0.525, which correctly credits Gemma's exceptional Revise Rate. A formula that ignores Revise Rate cannot measure the full metacognitive loop and will systematically mis-rank models with inverted Hold/Revise profiles.

**Calibration is the strongest between-model discriminator.** Gemini's Brier score (0.099) is 2.7× better than DeepSeek's (0.267), 2.7× better than Claude's (0.293), and 4.0× better than Gemma's (0.399). The AUROC range is narrower (0.610–0.692). This suggests that well-calibrated confidence — not self-audit discrimination — is the primary factor separating current frontier models on metacognitive measures.

**Temperature sensitivity is model-specific and must be reported.** Gemma 3 27B is perfectly stable across all temperatures (SI=0.85 at T=0.0, 0.3, 0.7, and 1.0), indicating its high sycophancy is structural rather than a sampling artifact. Gemini 2.5 Flash shows moderate sensitivity (SI range 0.05–0.20); its main-evaluation SI=0.20 at T=0.7 differs from the temperature-sweep value of SI=0.10 at the same temperature due to independent random sampling of challenges across the two runs — an illustration of the ±22pp sampling variance at N=10. Claude H4.5 and DeepSeek V3.2 show low sensitivity (SI range ≤0.05).

**Domain-weighted composites shift model rankings.** For legal deployments (γ=0.55, SI-dominant), DeepSeek V3.2 (SI=0.00) ranks first with a composite of 0.670, narrowly edging Claude H4.5 (0.669). For medical deployments (α=0.50, calibration-dominant), Gemini's Brier advantage (0.099 vs. 0.267–0.399) gives it the largest lead at 0.772. These ranking shifts demonstrate that deployment context must inform composite weight selection.

---

## Limitations

**N per Phase 3 condition.** With N=10 per condition, Hold and Revise Rate estimates carry 95% CIs of approximately ±22pp. The Hold/Revise dissociation between Gemma and DeepSeek V3.2 is robust (75–85pp gap on both dimensions), but temperature sensitivity comparisons should be interpreted with this variance in mind.

**Single evaluation platform.** All models were evaluated via the Kaggle Benchmarks platform. Model behaviour may differ under direct API access with explicit system-prompt control.

**ECE bin-count invariance.** ECE values are numerically identical across M ∈ {5, 10, 15, 20} bins for all models, reflecting confidence score granularity (multiples of 20%) rather than genuine bin robustness. Finer-grained confidence elicitation is needed for the bin sweep to be diagnostic.

**No human baseline.** Performance tier labels are based on theoretically motivated AUROC thresholds rather than empirical human norms on these specific question types.

---

## Conclusions

The Epistemic Audit demonstrates that metacognitive competence in frontier LLMs is measurable, discriminating, and multidimensional. No model evaluated achieves Metacognitively Aware performance (AUROC > 0.70), placing all four in the Partially Calibrated tier. The benchmark successfully separates models along multiple axes: calibration, self-audit accuracy, sycophancy resistance, and belief revision — capacities that can be high or low independently.

The most practically significant finding is the Hold/Revise dissociation. A model that holds correct beliefs under pressure but fails to correct wrong ones (DeepSeek V3.2) is not interchangeable with one that corrects wrong beliefs but abandons correct ones under pressure (Gemma 3 27B). Both fail metacognitively, but in ways with different safety implications depending on deployment context. Legal applications require sycophancy resistance; scientific reasoning requires belief revision. The benchmark provides the metrics to distinguish these profiles; composite scores alone cannot.

The near-universal failure on distorted facts (three of four models at 20–24%) identifies the most urgent training target: teaching models to flag false premises rather than reason from them. This failure occurs before any adversarial pressure is applied — making it a form of pre-sycophancy embedded in first-order question processing.

---

## References

- Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know. *arXiv:2207.05221.*
- Flavell, J. H. (1979). Metacognition and cognitive monitoring. *American Psychologist, 34*(10), 906–911.
- Brown, A. L. (1978). Knowing when, where, and how to remember: A problem of metacognition. *Advances in Instructional Psychology, Vol. 1.*
- Guo, C. et al. (2017). On calibration of modern neural networks. *Proceedings of ICML 2017.*
- Hendrycks, D. et al. (2020). Measuring massive multitask language understanding. *arXiv:2009.03300.*
- Rein, D. et al. (2023). GPQA: A graduate-level Google-proof Q&A benchmark. *arXiv:2311.12022.*
- Perez, E. et al. (2022). Red teaming language models with language models. *arXiv:2202.03286.*

## Organizational Affiliation

Independent submission — Abdelmajid Erramaline.
