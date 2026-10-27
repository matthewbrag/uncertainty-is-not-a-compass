"""
HF transformers helpers shared by the experiment scripts: model loading, full-vocab
entropy, and the logits processors that stop generation once a trace has written
its final answer or fallen into a repetition loop.

Split out of the token-tree app (tools/token_tree_app/app.py) so experiments don't
import a web server, which used to load a model and a dataset at import time.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from .stop_rules import (  # noqa: F401  (re-exported for the experiment scripts)
    ANSWER_MARKER_RE,
    DEFAULT_TEMPLATE,
    LAST_NUMBER_RE,
    detect_repetition,
    extract_answer,
)

MODEL_REGISTRY = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
}
DEVICE = "cuda:0"
THINK_CLOSE = "</think>"

_loaded_models = {}


def get_model(name):
    """(tokenizer, model) for a MODEL_REGISTRY key, loaded once in bf16 on DEVICE."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}' (known: {', '.join(MODEL_REGISTRY)})")
    if name not in _loaded_models:
        repo = MODEL_REGISTRY[name]
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(DEVICE).eval()
        _loaded_models[name] = (tok, mdl)
    return _loaded_models[name]


def compute_entropy(logits):
    """Shannon entropy in bits of the full-vocab softmax, from raw (unfiltered,
    temperature-1) logits — independent of whatever decoding mode/temperature
    is later used to actually pick a token."""
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log2(probs.clamp_min(1e-12))
    return float(-(probs * logp).sum().item())


class ForceEosAfterAnswer(LogitsProcessor):
    """Forces EOS once a row's post-</think> text contains '#### <number>'.

    The model rarely emits its own EOS after the chat template, even once it has
    already written the final answer, so batched generation otherwise runs to
    max_new_tokens on every example. Pinning the logits to EOS here lets
    transformers' own unfinished-sequence tracking retire that row immediately.
    """

    def __init__(self, tokenizer, prompt_len, eos_token_id, enable_thinking):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.eos_token_id = eos_token_id
        self.enable_thinking = enable_thinking
        self.matched = None

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        if self.matched is None:
            self.matched = [False] * batch_size

        for i in range(batch_size):
            if not self.matched[i]:
                text = self.tokenizer.decode(input_ids[i, self.prompt_len:], skip_special_tokens=True)
                if self.enable_thinking:
                    idx = text.rfind(THINK_CLOSE)
                    segment = text[idx + len(THINK_CLOSE):] if idx != -1 else None
                else:
                    segment = text
                if segment is not None:
                    # A match ending exactly at the current text boundary means
                    # the number might still be mid-stream (e.g. "2" generated,
                    # "60" not yet sampled) — only trust it once at least one
                    # more character has been generated past the number, proving
                    # the model itself moved on rather than us guessing mid-digit.
                    # "#### 4." is not done yet either: a "." or "," right after
                    # the digits may be the middle of "4.5" / "1,250", so wait for
                    # a character that can't continue the number (this used to cut
                    # decimal answers off at the point)
                    for m in ANSWER_MARKER_RE.finditer(segment):
                        nxt = segment[m.end(1):m.end(1) + 2]
                        if nxt and not (nxt[0] in ".," and (len(nxt) < 2 or nxt[1].isdigit())):
                            self.matched[i] = True
                            break

            if self.matched[i]:
                scores[i, :] = -float("inf")
                scores[i, self.eos_token_id] = 0.0

        return scores


class ForceEosOnRepetition(LogitsProcessor):
    """Same repetition guard as the token-tree app's run loop, as a logits
    processor so batched generation (which otherwise has no mid-generation hook)
    doesn't burn the whole token budget on a loop."""

    def __init__(self, prompt_len, eos_token_id):
        self.prompt_len = prompt_len
        self.eos_token_id = eos_token_id
        self.matched = None

    def __call__(self, input_ids, scores):
        batch_size = input_ids.shape[0]
        if self.matched is None:
            self.matched = [False] * batch_size

        for i in range(batch_size):
            if not self.matched[i]:
                gen_ids = input_ids[i, self.prompt_len:].tolist()
                # skip rows that already ended via some other condition (e.g.
                # ForceEosAfterAnswer) - HF pads them with repeated pad_token_id,
                # which would otherwise trivially look like a detected loop
                if self.eos_token_id not in gen_ids and detect_repetition(gen_ids):
                    self.matched[i] = True
            if self.matched[i]:
                scores[i, :] = -float("inf")
                scores[i, self.eos_token_id] = 0.0

        return scores
