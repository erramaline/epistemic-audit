"""Self-referential word puzzles and novel rule-induction tasks."""

import random
from epistemic_audit import Question


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _counting_puzzle(rng, difficulty):
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "She sells seashells by the seashore every summer",
        "Peter Piper picked a peck of pickled peppers",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood",
    ]
    sentence = rng.choice(sentences)
    letter = rng.choice("aeioustrln")
    count = sentence.lower().count(letter)

    return Question(
        id="", category="linguistic",
        prompt=f'How many times does the letter "{letter}" appear in this sentence?\n\n"{sentence}"\n\nCount carefully and give just the number.',
        correct_answer=str(count), is_answerable=True, difficulty=difficulty,
        metadata={"type": "letter_counting", "letter": letter},
    )


def _word_position_puzzle(rng, difficulty):
    words = ["apple", "bridge", "candle", "dragon", "eagle", "forest", "guitar", "harbor"]
    rng.shuffle(words)
    n = rng.randint(2, min(len(words), 3 + difficulty))
    sentence = " ".join(words[:n + 2])

    return Question(
        id="", category="linguistic",
        prompt=f'In the following list of words, what is the {_ordinal(n)} word?\n\n"{sentence}"\n\nAnswer with just the word.',
        correct_answer=words[n - 1], is_answerable=True, difficulty=difficulty,
        metadata={"type": "word_position", "position": n},
    )


def _novel_rule_puzzle(rng, difficulty):
    rules = [
        ("In 'Zorp language', every word is reversed.", lambda w: w[::-1]),
        ("In 'Flip code', vowels rotate: a→e, e→i, i→o, o→u, u→a.",
         lambda w: "".join({"a":"e","e":"i","i":"o","o":"u","u":"a"}.get(c,c) for c in w)),
        ("In 'Drop code', remove every second letter.", lambda w: w[::2]),
    ]
    desc, fn = rng.choice(rules)
    test_words = ["table", "chair", "house", "lemon", "planet", "river", "stone", "music"]
    word = rng.choice(test_words)

    return Question(
        id="", category="linguistic",
        prompt=f'{desc}\n\nApply this rule to the word "{word}". What is the result?\n\nAnswer with just the transformed word.',
        correct_answer=fn(word), is_answerable=True, difficulty=difficulty,
        metadata={"type": "novel_rule", "rule": desc, "input_word": word},
    )


def generate(seed: int, difficulty: int = 3) -> Question:
    rng = random.Random(seed)
    q = rng.choice([_counting_puzzle, _word_position_puzzle, _novel_rule_puzzle])(rng, difficulty)
    q.id = f"ling_{seed}_{difficulty}"
    return q
