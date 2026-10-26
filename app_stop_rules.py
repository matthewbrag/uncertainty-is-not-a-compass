"""
Dependency-free copies of app.py's answer parsing and repetition detection,
so the vLLM venv (which doesn't have app.py's FastAPI/datasets stack) uses the
exact same rules. Keep in sync with app.py: DEFAULT_TEMPLATE, ANSWER_MARKER_RE,
LAST_NUMBER_RE, extract_answer, detect_repetition.
"""
import re

DEFAULT_TEMPLATE = (
    "{question}\n\nSolve this step by step. Write each step on its own line starting with "
    "'Step 1:', 'Step 2:', etc. After the final step, on its own line, write your final "
    "numeric answer in exactly this format: '#### <number>' (for example: '#### 42')."
)
ANSWER_MARKER_RE = re.compile(r"####\s*\$?(-?[\d,]*\.?\d+)")
LAST_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def extract_answer(text):
    m = None
    for m in ANSWER_MARKER_RE.finditer(text):
        pass  # take the last marker match
    if m is None:
        matches = LAST_NUMBER_RE.findall(text)
        if not matches:
            return None
        raw = matches[-1]
    else:
        raw = m.group(1)
    raw = raw.replace(",", "").replace("$", "")
    try:
        return float(raw)
    except ValueError:
        return None


def detect_repetition(ids, min_ngram=3, max_ngram=16, min_repeats=6):
    n_total = len(ids)
    for n in range(min_ngram, max_ngram + 1):
        window = n * min_repeats
        if n_total < window:
            continue
        tail = ids[-window:]
        pattern = tail[:n]
        if all(tail[i * n:(i + 1) * n] == pattern for i in range(min_repeats)):
            return True
    return False
