"""Master generator: produces balanced question sets across all 6 categories."""

import random
from typing import Optional
from epistemic_audit import Question
from epistemic_audit.generate.templates import (
    arithmetic, logic, fabricated, distorted, linguistic, calibration_traps,
)

GENERATORS = {
    "arithmetic": arithmetic.generate,
    "logic": logic.generate,
    "fabricated": fabricated.generate,
    "distorted": distorted.generate,
    "linguistic": linguistic.generate,
    "calibration_trap": calibration_traps.generate,
}


class QuestionGenerator:
    """Generates balanced question sets across all 6 categories.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42):
        """Initialise with a fixed seed.

        Args:
            seed: Integer seed for the internal RNG.
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_set(
        self,
        n_per_category: int = 10,
        difficulty_range: tuple[int, int] = (1, 5),
        categories: Optional[list[str]] = None,
    ) -> list[Question]:
        """Generate a balanced question set shuffled across all categories.

        Args:
            n_per_category: Number of questions to generate per category.
            difficulty_range: (min, max) difficulty levels, inclusive.
            categories: Subset of category names; defaults to all 6.

        Returns:
            Shuffled list of Question objects.
        """
        cats = categories or list(GENERATORS.keys())
        questions = []
        for cat in cats:
            gen_fn = GENERATORS[cat]
            cat_questions = []
            seen_prompts = set()
            while len(cat_questions) < n_per_category:
                seed_i = self.rng.randint(0, 999999)
                diff = self.rng.randint(*difficulty_range)
                q = gen_fn(seed=seed_i, difficulty=diff)
                if q.prompt not in seen_prompts:
                    seen_prompts.add(q.prompt)
                    cat_questions.append(q)
            questions.extend(cat_questions)
        self.rng.shuffle(questions)
        return questions

    def generate_multiple_sets(self, n_sets: int = 5, n_per_category: int = 10):
        """Generate multiple independent question sets for sensitivity analysis.

        Args:
            n_sets: Number of separate sets to generate.
            n_per_category: Questions per category in each set.

        Returns:
            List of question lists.
        """
        return [
            QuestionGenerator(seed=self.seed + i * 1000).generate_set(n_per_category=n_per_category)
            for i in range(n_sets)
        ]


def export_questions(questions: list[Question], path: str) -> None:
    """Serialise a list of questions to a JSON file.

    Enables running the exact same question set against different models
    for fair, reproducible comparisons.

    Args:
        questions: List of Question objects to export.
        path: Destination file path (created if it does not exist).
    """
    import json
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)


def import_questions(path: str) -> list[Question]:
    """Load a question list from a previously exported JSON file.

    Args:
        path: Path to the JSON file produced by export_questions.

    Returns:
        List of Question objects identical to the originals.
    """
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Question.from_dict(d) for d in data]
