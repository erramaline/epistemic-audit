"""Multi-step arithmetic questions with verifiable answers."""

import random
from epistemic_audit import Question

DIFFICULTY_CONFIG = {
    1: {"num_ops": 2, "max_num": 20, "ops": ["+", "-"]},
    2: {"num_ops": 3, "max_num": 50, "ops": ["+", "-", "×"]},
    3: {"num_ops": 4, "max_num": 100, "ops": ["+", "-", "×"]},
    4: {"num_ops": 5, "max_num": 200, "ops": ["+", "-", "×", "÷"]},
    5: {"num_ops": 6, "max_num": 500, "ops": ["+", "-", "×", "÷"]},
}


def generate(seed: int, difficulty: int = 3) -> Question:
    rng = random.Random(seed)
    config = DIFFICULTY_CONFIG[difficulty]

    nums = [rng.randint(2, config["max_num"]) for _ in range(config["num_ops"] + 1)]
    ops = [rng.choice(config["ops"]) for _ in range(config["num_ops"])]

    # Ensure clean division
    for i, op in enumerate(ops):
        if op == "÷":
            divisors = [d for d in range(2, min(nums[i], 20) + 1) if nums[i] % d == 0]
            if divisors:
                nums[i + 1] = rng.choice(divisors)
            else:
                ops[i] = "×"

    expression_parts = [str(nums[0])]
    for i, op in enumerate(ops):
        expression_parts.append(op)
        expression_parts.append(str(nums[i + 1]))
    expression_str = " ".join(expression_parts)

    eval_expr = expression_str.replace("×", "*").replace("÷", "/")
    try:
        result = eval(eval_expr)
        result = int(round(result)) if abs(result - round(result)) < 0.001 else round(result, 2)
    except ZeroDivisionError:
        return generate(seed + 10000, difficulty)

    prompt = f"Calculate the following expression. Show your work, then give the final numerical answer.\n\n{expression_str}"

    return Question(
        id=f"arith_{seed}_{difficulty}", category="arithmetic",
        prompt=prompt, correct_answer=str(result),
        is_answerable=True, difficulty=difficulty,
        metadata={"expression": expression_str, "operations": ops},
    )
