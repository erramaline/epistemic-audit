"""Synthetic syllogism questions with made-up entity names."""

import random
from epistemic_audit import Question

ENTITY_POOLS = [
    ["bloops", "ramps", "clinks", "daxes", "forps", "grells", "hixes", "jombs"],
    ["wugs", "zilps", "trebs", "morfs", "pleks", "snorgs", "bivs", "quands"],
    ["flims", "zarks", "punds", "krebs", "tolms", "glefs", "drups", "yolks"],
]


def generate(seed: int, difficulty: int = 3) -> Question:
    rng = random.Random(seed)
    pool = list(rng.choice(ENTITY_POOLS))
    rng.shuffle(pool)

    chain_length = {1: 2, 2: 3, 3: 3, 4: 4, 5: 5}[difficulty]
    use_negation = difficulty >= 4
    use_distractors = difficulty >= 3

    chain = pool[:chain_length + 1]
    premises = [f"All {chain[i]} are {chain[i+1]}." for i in range(chain_length)]

    distractors = []
    if use_distractors:
        extra = pool[chain_length + 1:chain_length + 3]
        if len(extra) >= 2:
            distractors.append(f"All {extra[0]} are {extra[1]}.")
        if len(extra) >= 1:
            distractors.append(f"Some {extra[0]} are {chain[0]}.")

    if use_negation and rng.random() < 0.5:
        query_start = chain[0]
        non_chain = [e for e in pool if e not in chain]
        if non_chain:
            query_end = rng.choice(non_chain)
            correct_answer = "No"
        else:
            query_end = chain[-1]
            correct_answer = "Yes"
    else:
        if rng.random() < 0.5:
            query_start, query_end, correct_answer = chain[0], chain[-1], "Yes"
        else:
            query_start, query_end, correct_answer = chain[-1], chain[0], "No"

    all_statements = premises + distractors
    rng.shuffle(all_statements)
    premises_text = "\n".join(f"- {s}" for s in all_statements)

    prompt = (
        f"Given the following statements:\n{premises_text}\n\n"
        f"Question: Are all {query_start} also {query_end}? "
        f"Answer Yes or No, and explain your reasoning step by step."
    )

    return Question(
        id=f"logic_{seed}_{difficulty}", category="logic",
        prompt=prompt, correct_answer=correct_answer,
        is_answerable=True, difficulty=difficulty,
        metadata={"chain": chain, "premises": premises, "distractors": distractors},
    )
