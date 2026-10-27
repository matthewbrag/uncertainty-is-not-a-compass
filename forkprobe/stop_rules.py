"""
Prompt template, answer parsing and repetition detection shared by every
experiment. Dependency-free, so both the HF venv and the vLLM venv import the
exact same rules.
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
    # min_ngram=3 skips 1- and 2-token patterns entirely (blank lines, "- "
    # bullets, digit runs, "$$" / markdown dividers all legitimately repeat
    # 2-3x and would otherwise false-trigger). min_repeats=6 gives a wide
    # margin above normal formatting while still catching genuine degenerate
    # loops early.
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
