"""Generate a human-participant evaluation form and answer key for baseline collection."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epistemic_audit.generate.questions import QuestionGenerator


def generate_human_form(
    seed: int = 42,
    n_per_category: int = 5,
    output_dir: str = "data/human_baselines",
) -> None:
    """Generate an HTML question form and a JSON answer key for human baseline data.

    Creates two files:
    - human_form.html: A self-contained single-page app where participants answer
      questions one at a time, rate their confidence, provide a rationale, and
      copy the resulting JSON to send to the researcher.
    - answer_key.json: Ground-truth answers for scoring human responses.

    Args:
        seed: Random seed for the question generator (default 42).
        n_per_category: Questions per category (default 5 → 30 total).
        output_dir: Directory to write outputs into.
    """
    os.makedirs(output_dir, exist_ok=True)

    gen = QuestionGenerator(seed=seed)
    questions = gen.generate_set(n_per_category=n_per_category)

    # ── Answer key ────────────────────────────────────────────────────────────
    answer_key = [
        {
            "id": q.id,
            "category": q.category,
            "prompt": q.prompt,
            "correct_answer": q.correct_answer,
            "is_answerable": q.is_answerable,
            "difficulty": q.difficulty,
        }
        for q in questions
    ]
    key_path = os.path.join(output_dir, "answer_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2)
    print(f"Answer key saved → {key_path}")

    # ── HTML form ─────────────────────────────────────────────────────────────
    questions_js = json.dumps(
        [{"id": q.id, "category": q.category, "prompt": q.prompt} for q in questions],
        indent=2,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Epistemic Audit — Human Baseline</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f5f6fa;
      color: #1a1a2e;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      padding: 2rem 1rem;
    }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; color: #2d3561; }}
    .subtitle {{ color: #666; font-size: 0.95rem; margin-bottom: 2rem; }}
    .card {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2rem;
      max-width: 720px;
      width: 100%;
    }}
    .progress-bar-outer {{
      background: #e8ecef;
      border-radius: 99px;
      height: 8px;
      margin-bottom: 1.5rem;
    }}
    .progress-bar-inner {{
      background: linear-gradient(90deg, #4f6ef7, #7c3aed);
      height: 8px;
      border-radius: 99px;
      transition: width 0.4s ease;
    }}
    .q-meta {{
      font-size: 0.8rem;
      color: #888;
      margin-bottom: 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .q-text {{
      font-size: 1.1rem;
      line-height: 1.65;
      margin-bottom: 1.5rem;
      white-space: pre-wrap;
    }}
    label {{ display: block; font-weight: 600; margin-bottom: 0.4rem; color: #444; font-size: 0.95rem; }}
    textarea {{
      width: 100%;
      border: 1.5px solid #d1d5db;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      font-size: 0.95rem;
      resize: vertical;
      min-height: 80px;
      font-family: inherit;
      transition: border-color 0.2s;
      margin-bottom: 1.25rem;
    }}
    textarea:focus {{ outline: none; border-color: #4f6ef7; }}
    .slider-group {{ margin-bottom: 1.5rem; }}
    .slider-labels {{ display: flex; justify-content: space-between; font-size: 0.8rem; color: #777; margin-top: 0.3rem; }}
    input[type=range] {{ width: 100%; accent-color: #4f6ef7; cursor: pointer; }}
    #conf-display {{ font-size: 1.5rem; font-weight: 700; color: #4f6ef7; text-align: center; margin-bottom: 0.3rem; }}
    .btn {{
      display: inline-block;
      padding: 0.75rem 2rem;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      border: none;
      background: linear-gradient(90deg, #4f6ef7, #7c3aed);
      color: #fff;
      transition: opacity 0.2s, transform 0.1s;
      width: 100%;
    }}
    .btn:hover {{ opacity: 0.9; transform: translateY(-1px); }}
    .final-section {{ display: none; }}
    .final-section h2 {{ font-size: 1.3rem; margin-bottom: 0.75rem; color: #2d3561; }}
    .final-section p {{ color: #555; margin-bottom: 1rem; font-size: 0.95rem; }}
    #json-output {{
      width: 100%;
      min-height: 200px;
      font-family: monospace;
      font-size: 0.82rem;
      border: 1.5px solid #d1d5db;
      border-radius: 8px;
      padding: 0.75rem;
      background: #f9fafb;
      resize: vertical;
      margin-bottom: 1rem;
    }}
    .copy-btn {{ background: #16a34a; }}
  </style>
</head>
<body>
  <h1>Epistemic Audit — Human Baseline</h1>
  <p class="subtitle">Answer each question as accurately as you can. Be honest about your confidence.</p>
  <div class="card" id="question-card">
    <div class="progress-bar-outer">
      <div class="progress-bar-inner" id="progress-bar" style="width:0%"></div>
    </div>
    <div class="q-meta" id="q-meta"></div>
    <div class="q-text" id="q-text"></div>

    <label for="answer-input">Your Answer</label>
    <textarea id="answer-input" placeholder="Type your answer here…" rows="3"></textarea>

    <div class="slider-group">
      <label for="conf-slider">Confidence (0 = wild guess, 100 = certain)</label>
      <div id="conf-display">50</div>
      <input type="range" id="conf-slider" min="0" max="100" value="50"
             oninput="document.getElementById('conf-display').textContent=this.value" />
      <div class="slider-labels"><span>0 — No idea</span><span>50 — Uncertain</span><span>100 — Certain</span></div>
    </div>

    <label for="rationale-input">One-sentence rationale</label>
    <textarea id="rationale-input" placeholder="Why do you think this is correct?" rows="2"></textarea>

    <button class="btn" id="next-btn" onclick="nextQuestion()">Next →</button>
  </div>

  <div class="card final-section" id="final-card">
    <h2>✓ All done! Thank you.</h2>
    <p>Copy the JSON below and send it to the researcher.</p>
    <textarea id="json-output" readonly></textarea>
    <button class="btn copy-btn" onclick="copyJSON()">Copy to Clipboard</button>
  </div>

  <script>
    const QUESTIONS = {questions_js};
    const answers = [];
    let current = 0;

    function showQuestion(idx) {{
      const q = QUESTIONS[idx];
      document.getElementById('q-meta').textContent =
        `Question ${{idx + 1}} of ${{QUESTIONS.length}} · ${{q.category}}`;
      document.getElementById('q-text').textContent = q.prompt;
      document.getElementById('answer-input').value = '';
      document.getElementById('conf-slider').value = 50;
      document.getElementById('conf-display').textContent = '50';
      document.getElementById('rationale-input').value = '';
      const pct = (idx / QUESTIONS.length * 100).toFixed(1);
      document.getElementById('progress-bar').style.width = pct + '%';
    }}

    function nextQuestion() {{
      const q = QUESTIONS[current];
      const answer = document.getElementById('answer-input').value.trim();
      if (!answer) {{ alert('Please enter an answer before continuing.'); return; }}
      answers.push({{
        id: q.id,
        category: q.category,
        answer: answer,
        confidence: parseInt(document.getElementById('conf-slider').value),
        rationale: document.getElementById('rationale-input').value.trim(),
      }});
      current++;
      if (current < QUESTIONS.length) {{
        showQuestion(current);
      }} else {{
        document.getElementById('question-card').style.display = 'none';
        const finalCard = document.getElementById('final-card');
        finalCard.style.display = 'block';
        document.getElementById('json-output').value = JSON.stringify(answers, null, 2);
      }}
    }}

    function copyJSON() {{
      const ta = document.getElementById('json-output');
      ta.select();
      document.execCommand('copy');
      alert('Copied to clipboard!');
    }}

    showQuestion(0);
  </script>
</body>
</html>"""

    html_path = os.path.join(output_dir, "human_form.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Human form saved  → {html_path}")
    print(f"Total questions:   {len(questions)}")


if __name__ == "__main__":
    generate_human_form()
