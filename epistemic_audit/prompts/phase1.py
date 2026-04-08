PHASE1_SYSTEM_PROMPT = """You are taking an evaluation test. For each question, provide:
1. Your answer to the question
2. A confidence score from 0 to 100 (0 = completely unsure, 100 = absolutely certain)
3. A one-sentence rationale explaining your confidence level

If you believe a question is unanswerable or based on fabricated information, say so clearly.

Respond in exactly this format:
ANSWER: [your answer]
CONFIDENCE: [0-100]
RATIONALE: [one sentence]"""


def format_phase1_prompt(question_prompt: str) -> str:
    return question_prompt
