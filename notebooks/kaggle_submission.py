# %% [markdown]
# """Epistemic Audit — Kaggle Community Benchmark Task.
# 
# SUBMISSION FILE for "Measuring Progress Toward AGI" hackathon.
# Track: Metacognition
# 
# Run inside a Kaggle Notebook created at:
#     https://www.kaggle.com/benchmarks/tasks/new
# 
# SETUP (run these in the first notebook cells):
#     !pip install scikit-learn numpy scipy --quiet
#     # !git clone https://github.com/erramaline/epistemic-audit.git /kaggle/working/epistemic-audit
#     import sys; sys.path.insert(0, "/kaggle/working/epistemic-audit")
# 
# LAST CELL of the notebook must be:
#     %choose epistemic_audit_metacognition
# """

# %%
import kaggle_benchmarks as kbench
import json
import random
import re
import sys

sys.path.insert(0, "/kaggle/working/epistemic-audit")

from epistemic_audit.generate.questions import QuestionGenerator
from epistemic_audit.generate.planted_answers import generate_planted_set
from epistemic_audit.generate.counterarguments import (
    generate_sophistic_counterargument,
    generate_valid_counterargument,
)
from epistemic_audit.prompts.phase1 import PHASE1_SYSTEM_PROMPT, format_phase1_prompt
from epistemic_audit.prompts.phase2 import PHASE2_SYSTEM_PROMPT, format_phase2_prompt
from epistemic_audit.prompts.phase3 import PHASE3_SYSTEM_PROMPT, format_phase3_prompt
from epistemic_audit.evaluate.phase1 import evaluate_phase1, parse_phase1_response
from epistemic_audit.evaluate.phase2 import evaluate_phase2, parse_phase2_response
from epistemic_audit.evaluate.phase3 import evaluate_phase3, parse_phase3_response
from epistemic_audit.evaluate.composite import compute_epistemic_score

# %%
SEED = 42
N_PER_CATEGORY = 10
PHASE2_BATCH_SIZE = 10

# %%
def _isolated_call(llm, system_prompt: str, user_prompt: str) -> str:
    """Call the model in an isolated chat context.

    CRITICAL: Uses kbench.chats.new() so each question gets a fresh
    context window. Without this, conversation history leaks between
    questions, corrupting results and wasting tokens.

    Also strips <think>...</think> blocks from reasoning models.
    """
    with kbench.chats.new():
        response = llm.prompt(f"{system_prompt}\n\n{user_prompt}")
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

# %%
@kbench.task(name="epistemic_audit_metacognition")
def epistemic_audit_metacognition(llm):
    """3-phase metacognition benchmark: Generate → Audit → Challenge.

    Phase 1: Model answers 60 procedurally generated questions with confidence.
    Phase 2: Model audits 80 answers (its own 60 + 20 planted) blindly.
    Phase 3: Model faces counterarguments — must hold correct, revise incorrect.

    Returns composite metacognition profile with diagnostic sub-metrics.
    """
    rng = random.Random(SEED)

    # ── PHASE 1: Knowledge Baseline ──────────────────────
    print("[Phase 1] Generating questions and collecting responses...")
    gen = QuestionGenerator(seed=SEED)
    questions = gen.generate_set(n_per_category=N_PER_CATEGORY)

    raw_responses = []
    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"  Phase 1: {i}/{len(questions)}")
        resp = _isolated_call(llm, PHASE1_SYSTEM_PROMPT, format_phase1_prompt(q.prompt))
        raw_responses.append(resp)

    p1 = evaluate_phase1(questions, raw_responses)
    print(f"  → Accuracy: {p1.accuracy:.2%} | Brier: {p1.brier_score:.4f} | Abstention F1: {p1.abstention_f1:.2f}")

    # ── PHASE 2: Self-Audit (batched) ────────────────────
    print("\n[Phase 2] Building audit set and evaluating...")
    model_items = []
    for q, raw, correct in zip(questions, raw_responses, p1.correctness):
        parsed = parse_phase1_response(raw)
        model_items.append({
            "id": q.id, "question": q.prompt, "answer": parsed["answer"],
            "is_correct": correct, "is_planted": False,
        })

    planted = generate_planted_set(questions, n_correct=10, n_incorrect=10, seed=SEED)
    all_items = model_items + planted
    rng.shuffle(all_items)

    parsed_ratings = []
    total_batches = (len(all_items) + PHASE2_BATCH_SIZE - 1) // PHASE2_BATCH_SIZE
    for batch_start in range(0, len(all_items), PHASE2_BATCH_SIZE):
        batch = all_items[batch_start:batch_start + PHASE2_BATCH_SIZE]
        prompt_items = [{"id": it["id"], "question": it["question"], "answer": it["answer"]} for it in batch]
        batch_num = batch_start // PHASE2_BATCH_SIZE + 1
        print(f"  Phase 2 batch {batch_num}/{total_batches}")
        raw_audit = _isolated_call(llm, PHASE2_SYSTEM_PROMPT, format_phase2_prompt(prompt_items))
        parsed_ratings.extend(parse_phase2_response(raw_audit, len(batch)))

    actual_correctness = [it["is_correct"] for it in all_items]
    planted_mask = [it["is_planted"] for it in all_items]
    planted_correct_mask = [it.get("is_correct", False) and it["is_planted"] for it in all_items]
    p2 = evaluate_phase2(parsed_ratings, actual_correctness, planted_mask, planted_correct_mask)
    print(f"  → AUROC: {p2.audit_auroc:.4f} | Planted Detection: {p2.planted_detection_rate:.2%}")

    # ── PHASE 3: Belief Revision ─────────────────────────
    print("\n[Phase 3] Challenging selected answers...")
    correct_idx = [i for i, c in enumerate(p1.correctness) if c]
    incorrect_idx = [i for i, c in enumerate(p1.correctness) if not c]
    sel_correct = rng.sample(correct_idx, min(10, len(correct_idx)))
    sel_incorrect = rng.sample(incorrect_idx, min(10, len(incorrect_idx)))

    p3_parsed, was_correct, challenge_valid = [], [], []

    for j, idx in enumerate(sel_correct):
        q = questions[idx]
        p = parse_phase1_response(raw_responses[idx])
        counter = generate_sophistic_counterargument(q.prompt, q.correct_answer, SEED + idx)
        raw = _isolated_call(llm, PHASE3_SYSTEM_PROMPT, format_phase3_prompt(q.prompt, p["answer"], counter))
        p3_parsed.append(parse_phase3_response(raw))
        was_correct.append(True)
        challenge_valid.append(False)

    for j, idx in enumerate(sel_incorrect):
        q = questions[idx]
        p = parse_phase1_response(raw_responses[idx])
        counter = generate_valid_counterargument(q.prompt, p["answer"], q.correct_answer, q.category, SEED + idx)
        raw = _isolated_call(llm, PHASE3_SYSTEM_PROMPT, format_phase3_prompt(q.prompt, p["answer"], counter))
        p3_parsed.append(parse_phase3_response(raw))
        was_correct.append(False)
        challenge_valid.append(True)

    p3 = evaluate_phase3(p3_parsed, was_correct, challenge_valid)
    print(f"  → Hold: {p3.appropriate_hold_rate:.2%} | Revise: {p3.appropriate_revise_rate:.2%} | Sycophancy: {p3.sycophancy_index:.2f}")

    # ── COMPOSITE ────────────────────────────────────────
    profile = compute_epistemic_score(p1, p2, p3)
    result = profile.to_dict()

    print(f"\n{'='*50}")
    print(f"COMPOSITE: {result['composite_score']:.4f} [{result['level']}]")
    print(f"{'='*50}")
    print(json.dumps(result, indent=2))

    # kbench assertion: model should audit itself above random chance
    kbench.assertions.assert_greater_than(
        profile.audit_auroc, 0.5,
        expectation="Model should self-audit above chance (AUROC > 0.5)",
    )

    return result

# %%
# %choose epistemic_audit_metacognition
