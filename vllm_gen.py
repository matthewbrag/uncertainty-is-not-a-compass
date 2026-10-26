"""
vLLM drop-in for the HF generate helpers used by the tournament/collection
scripts. Runs in .venv-vllm (vLLM pins its own torch; the HF venv stays
untouched). Hidden-state / probe scoring still goes through HF -- vLLM is only
used for generating traces and branches, which is where the time goes.

Why vLLM here: every problem forks 25-150 branches off one shared trace
(prefix caching computes that prefix once), and branch lengths vary 10x
(continuous batching retires finished branches instead of waiting on the
slowest one in a static batch).

The two custom stop rules from app.py are ported as a per-request logits
processor so behaviour matches the HF runs:
  - ForceEosAfterAnswer: force EOS once '#### <number>' has appeared in the
    generated text AND at least one more character follows the number
  - ForceEosOnRepetition: force EOS once detect_repetition() fires
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep the logits processor in-process
# 0.32 defaults to the V2 model runner, whose logits-processor API only sees
# on-device token buffers; the V1 runner still supports the per-request
# AdapterLogitsProcessor below, which is what the text-based stop rules need
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
# FlashInfer JIT-compiles its sampler with the system nvcc, which is too old
# for its flags ("Unknown option '--compress-mode=size'"); vLLM's own sampler is fine
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from app_stop_rules import ANSWER_MARKER_RE, detect_repetition

MODEL = "Qwen/Qwen3-1.7B"
SAMPLE_KW = dict(temperature=0.7, top_k=20, top_p=0.8)
_tok = AutoTokenizer.from_pretrained(MODEL)
EOS = _tok.eos_token_id


class StopRules(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self):
        return False  # it changes the argmax (forces EOS), so it must also run for greedy requests

    def new_req_logits_processor(self, params):
        if not (params.extra_args or {}).get("stop_rules"):
            return None
        state = {"done": False}

        def apply(output_ids, logits):
            if not state["done"] and output_ids:
                # only the generated text is checked (same as HF: prompt_len
                # there includes the forced prefix); a short tail is enough
                # because the check runs every step
                tail = _tok.decode(output_ids[-24:], skip_special_tokens=True)
                for m in ANSWER_MARKER_RE.finditer(tail):
                    if m.end(1) < len(tail):
                        state["done"] = True
                        break
                if not state["done"] and detect_repetition(list(output_ids)):
                    state["done"] = True
            if state["done"]:
                logits.fill_(-float("inf"))
                logits[EOS] = 0.0
            return logits

        return apply


def make_llm(gpu_memory_utilization=0.85, max_model_len=4096):
    return LLM(model=MODEL, dtype="bfloat16", enable_prefix_caching=True, seed=0,
               gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
               logits_processors=[StopRules])


def generate(llm, id_lists, max_new_tokens, sample, seeds=None):
    """Same contract as the HF helpers: one list of new token ids per input,
    cut just after EOS (EOS included) when the model stopped on its own."""
    params = []
    for k in range(len(id_lists)):
        kw = dict(SAMPLE_KW, seed=None if seeds is None else int(seeds[k])) if sample else dict(temperature=0.0)
        params.append(SamplingParams(max_tokens=max_new_tokens, stop_token_ids=[EOS], detokenize=False,
                                     extra_args={"stop_rules": True}, **kw))
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in id_lists], params, use_tqdm=False)
    rows = []
    for o in outs:
        row = list(o.outputs[0].token_ids)
        if EOS in row:
            row = row[: row.index(EOS) + 1]
        rows.append(row)
    return rows
