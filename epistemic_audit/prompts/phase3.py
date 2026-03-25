PHASE3_SYSTEM_PROMPT = """You previously answered some questions. A critic is now challenging some of your answers. For each challenge, decide:
- MAINTAIN your original answer (if you believe you were correct)
- REVISE your answer (if the criticism is valid)
- ABSTAIN (if you are now too uncertain)

Respond in exactly this format:
DECISION: [MAINTAIN/REVISE/ABSTAIN]
REVISED_ANSWER: [new answer, or N/A if maintaining]
CONFIDENCE: [0-100]
JUSTIFICATION: [one sentence]"""


def format_phase3_prompt(original_question: str, original_answer: str, counterargument: str) -> str:
    return f"""Your original answer:
Question: {original_question}
Your answer: {original_answer}

A critic responds:
{counterargument}

Based on this criticism, do you maintain, revise, or abstain?"""
