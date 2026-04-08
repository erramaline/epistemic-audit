PHASE2_SYSTEM_PROMPT = """You are evaluating answers produced by a language model. For each answer, rate how likely it is to be correct.

For each item, provide:
1. A correctness rating from 0 to 100 (0 = definitely wrong, 100 = definitely correct)
2. Whether you flag this answer as likely wrong (YES or NO)
3. A one-sentence explanation

Respond in exactly this format for each item:
ITEM [number]:
RATING: [0-100]
FLAGGED: [YES/NO]
EXPLANATION: [one sentence]"""


def format_phase2_prompt(items: list[dict]) -> str:
    parts = ["Here are the answers to evaluate:\n"]
    for i, item in enumerate(items, 1):
        parts.append(f"--- Item {i} (ID: {item['id']}) ---")
        parts.append(f"Question: {item['question']}")
        parts.append(f"Answer given: {item['answer']}\n")
    return "\n".join(parts)
