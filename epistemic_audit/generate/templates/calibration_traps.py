"""Questions with tempting obvious answers that are WRONG.
Based on cognitive reflection test patterns."""

import random
from epistemic_audit import Question

TRAP_TEMPLATES = [
    {
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "correct_answer": "$0.05",
        "trap_answer": "$0.10",
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "correct_answer": "5 minutes",
        "trap_answer": "100 minutes",
    },
    {
        "prompt": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
        "correct_answer": "47 days",
        "trap_answer": "24 days",
    },
    {
        "prompt": "A farmer has 15 sheep. All but 8 die. How many sheep does the farmer have left?",
        "correct_answer": "8",
        "trap_answer": "7",
    },
    {
        "prompt": "You're running a race and you pass the person in second place. What place are you in now?",
        "correct_answer": "2nd place",
        "trap_answer": "1st place",
    },
    {
        "prompt": "A doctor gives you 3 pills and tells you to take one every 30 minutes. How many minutes will it take to finish all the pills?",
        "correct_answer": "60 minutes",
        "trap_answer": "90 minutes",
    },
    {
        "prompt": "How many times can you subtract 5 from 25?",
        "correct_answer": "1 time",
        "trap_answer": "5 times",
    },
    {
        "prompt": "If you have a bowl with six apples and you take away four, how many do you have?",
        "correct_answer": "4",
        "trap_answer": "2",
    },
    {
        "prompt": "Some months have 31 days, some have 30 days. How many months have 28 days?",
        "correct_answer": "12 (all months)",
        "trap_answer": "1 (February)",
    },
    {
        "prompt": "A brick weighs one kilogram plus half a brick. How much does the whole brick weigh?",
        "correct_answer": "2 kilograms",
        "trap_answer": "1.5 kilograms",
    },
    {
        "prompt": "A snail is at the bottom of a 10-meter well. Each day it climbs 3 meters, and each night it slides back 2 meters. How many days does it take to reach the top?",
        "correct_answer": "8 days",
        "trap_answer": "10 days",
    },
    {
        "prompt": "You are in a dark room with a box of matches, a candle, a wood stove, and a kerosene lamp. You have only one match. What do you light first?",
        "correct_answer": "The match",
        "trap_answer": "The candle (or lamp or stove)",
    },
    {
        "prompt": "Four cards lie on a table showing: 'E', 'K', '4', '7'. Each card has a letter on one side and a number on the other. Which cards must you turn over to test the rule: 'If a card has a vowel on one side, it has an even number on the other'?",
        "correct_answer": "E and 7",
        "trap_answer": "E and 4",
    },
    {
        "prompt": "A plane crashes exactly on the border between Canada and the United States. Where do they bury the survivors?",
        "correct_answer": "You don't bury survivors",
        "trap_answer": "Half in Canada, half in the US",
    },
    {
        "prompt": "Is it legal in the United States for a man to marry his widow's sister?",
        "correct_answer": "No — a man with a widow is dead and cannot marry anyone",
        "trap_answer": "Yes, it is legal",
    },
]


def generate(seed: int, difficulty: int = 3) -> Question:
    rng = random.Random(seed)
    trap = rng.choice(TRAP_TEMPLATES)
    prompt = trap["prompt"]

    if difficulty <= 2:
        pass  # No hint
    elif difficulty <= 3:
        prompt += "\n\nThink carefully before answering."
    else:
        prompt += f'\n\n(Hint: the obvious answer is {trap["trap_answer"]}, but think again.)'

    return Question(
        id=f"trap_{seed}_{difficulty}", category="calibration_trap",
        prompt=prompt, correct_answer=trap["correct_answer"],
        is_answerable=True, difficulty=difficulty,
        metadata={"trap_answer": trap["trap_answer"]},
    )
