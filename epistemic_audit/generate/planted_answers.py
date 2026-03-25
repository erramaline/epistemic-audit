"""Planted correct/incorrect answers mixed into Phase 2 to prevent gaming."""

import random
from epistemic_audit import Question


def generate_planted_correct(question: Question, seed: int = 42) -> dict:
    return {
        "id": f"planted_correct_{question.id}",
        "question": question.prompt,
        "answer": f"The answer is {question.correct_answer}. This follows directly from the given information.",
        "is_correct": True,
        "is_planted": True,
    }


def generate_planted_incorrect(question: Question, seed: int = 42) -> dict:
    rng = random.Random(seed)
    if question.category == "arithmetic":
        try:
            n = float(question.correct_answer)
            wrong = str(int(n + rng.choice([-10, 10, -1, 1])))
        except ValueError:
            wrong = "42"
    elif question.category == "logic":
        wrong = "No" if question.correct_answer == "Yes" else "Yes"
    elif question.category == "fabricated":
        wrong = f"This occurred in {rng.randint(1950, 2010)}, according to historical records."
    else:
        wrong = "Based on my analysis, the answer is different from the expected one."

    return {
        "id": f"planted_incorrect_{question.id}",
        "question": question.prompt,
        "answer": f"The answer is {wrong}. I'm quite confident in my reasoning here.",
        "is_correct": False,
        "is_planted": True,
    }


def generate_planted_set(questions: list[Question], n_correct=10, n_incorrect=10, seed=42) -> list[dict]:
    rng = random.Random(seed)
    selected = rng.sample(questions, min(n_correct + n_incorrect, len(questions)))
    planted = []
    for i, q in enumerate(selected):
        if i < n_correct:
            planted.append(generate_planted_correct(q, seed + i))
        else:
            planted.append(generate_planted_incorrect(q, seed + i))
    return planted
