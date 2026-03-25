"""Epistemic Audit — A metacognition benchmark for LLMs."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    """A single benchmark question with verifiable ground truth."""
    id: str
    category: str
    prompt: str
    correct_answer: str
    is_answerable: bool
    difficulty: int
    metadata: dict = field(default_factory=dict)

    def verify(self, model_answer: str) -> bool:
        """Check if the model's answer is correct."""
        if not self.is_answerable:
            refusal_signals = [
                "i don't know", "cannot answer", "not enough information",
                "unanswerable", "no way to know", "fabricated", "fictional",
                "i'm not sure", "cannot determine", "there is no",
                "this is not a real", "this doesn't appear to be real",
                "i don't have information", "cannot verify",
                "no evidence", "does not exist", "never happened",
                "no such", "not a real",
            ]
            return any(s in model_answer.lower() for s in refusal_signals)

        # For distorted questions: check if model identified the distortion
        if self.category == "distorted":
            return self._check_distortion_detected(model_answer)

        return self._fuzzy_match(model_answer, self.correct_answer)

    def _check_distortion_detected(self, model_answer: str) -> bool:
        """Check if the model caught the factual distortion in the premise.

        For distorted questions, we check if the model's response contains
        key correction signals rather than exact string matching.
        """
        answer_lower = model_answer.lower()
        correct_lower = self.correct_answer.lower()

        # Direct match
        if self._fuzzy_match(model_answer, self.correct_answer):
            return True

        # Check for correction signals: the model points out the error
        correction_signals = [
            "incorrect", "inaccurate", "error", "wrong", "actually",
            "not correct", "mistaken", "false premise", "factual error",
            "should be", "was actually", "in fact", "correction",
            "not true", "misstated", "the correct",
        ]
        has_correction = any(s in answer_lower for s in correction_signals)

        # Check if key words from the correct answer appear
        # e.g., correct_answer = "fell in 1989, not 1991" → check for "1989"
        import re
        key_numbers = re.findall(r'\d{4}', self.correct_answer)
        key_names = [w for w in self.correct_answer.split() if len(w) > 4 and w[0].isupper()]

        has_key_info = any(n in answer_lower for n in key_numbers) or \
                       any(n.lower() in answer_lower for n in key_names)

        return has_correction and has_key_info

    def _fuzzy_match(self, model_answer: str, correct: str) -> bool:
        m = model_answer.strip().lower().replace(",", "")
        c = correct.strip().lower().replace(",", "")
        if c in m:
            return True
        # Check if the correct answer's key phrases appear
        # Split correct answer into meaningful chunks and check if most appear
        words = [w for w in c.split() if len(w) > 3]
        if words and sum(1 for w in words if w in m) >= len(words) * 0.6:
            return True
        try:
            return abs(float(m) - float(c)) < 0.01
        except ValueError:
            pass
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id, "category": self.category,
            "prompt": self.prompt, "correct_answer": self.correct_answer,
            "is_answerable": self.is_answerable, "difficulty": self.difficulty,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(**d)


@dataclass
class ModelResponse:
    question_id: str
    answer: str
    confidence: float
    rationale: str


@dataclass
class AuditRating:
    item_id: str
    correctness_rating: float
    flagged_as_wrong: bool
    explanation: str


@dataclass
class ChallengeResponse:
    question_id: str
    decision: str
    updated_confidence: float
    justification: str
    revised_answer: Optional[str] = None
