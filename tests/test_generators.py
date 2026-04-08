import pytest
from epistemic_audit import Question
from epistemic_audit.generate.questions import QuestionGenerator
from epistemic_audit.generate.templates import (
    arithmetic, logic, fabricated, distorted, linguistic, calibration_traps
)


class TestArithmetic:
    def test_generates_valid(self):
        q = arithmetic.generate(seed=42, difficulty=3)
        assert isinstance(q, Question)
        assert q.category == "arithmetic"
        assert q.is_answerable is True
        float(q.correct_answer)  # Should not raise

    def test_different_seeds(self):
        assert arithmetic.generate(1, 3).prompt != arithmetic.generate(2, 3).prompt

    def test_reproducible(self):
        assert arithmetic.generate(42, 3).correct_answer == arithmetic.generate(42, 3).correct_answer


class TestLogic:
    def test_valid_answer(self):
        q = logic.generate(seed=42, difficulty=3)
        assert q.correct_answer in ("Yes", "No")

    def test_all_difficulties(self):
        for d in range(1, 6):
            q = logic.generate(seed=42, difficulty=d)
            assert q.correct_answer in ("Yes", "No")


class TestFabricated:
    def test_unanswerable(self):
        q = fabricated.generate(seed=42, difficulty=3)
        assert q.is_answerable is False


class TestDistorted:
    def test_has_distortion(self):
        q = distorted.generate(seed=42, difficulty=3)
        assert "distortion" in q.metadata


class TestCalibrationTraps:
    def test_correct_differs_from_trap(self):
        q = calibration_traps.generate(seed=42, difficulty=3)
        assert q.correct_answer != q.metadata["trap_answer"]


class TestQuestionGenerator:
    def test_balanced_set(self):
        gen = QuestionGenerator(seed=42)
        qs = gen.generate_set(n_per_category=5)
        assert len(qs) == 30
        assert len({q.category for q in qs}) == 6

    def test_multiple_sets_differ(self):
        gen = QuestionGenerator(seed=42)
        sets = gen.generate_multiple_sets(n_sets=3, n_per_category=3)
        assert {q.id for q in sets[0]} != {q.id for q in sets[1]}
