import json
import re
import uuid
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset

from forkprobe.hf import ForceEosAfterAnswer, ForceEosOnRepetition
from forkprobe.stop_rules import DEFAULT_TEMPLATE, detect_repetition, extract_answer

MODEL_REGISTRY = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
}
DEFAULT_MODEL = "qwen3-0.6b"
DEVICE = "cuda:0"
TOP_K_DISPLAY = 10


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

app = FastAPI()

_loaded_models = {}


def get_model(name):
    if name not in MODEL_REGISTRY:
        raise HTTPException(400, f"unknown model '{name}'")
    if name not in _loaded_models:
        repo = MODEL_REGISTRY[name]
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(DEVICE).eval()
        _loaded_models[name] = (tok, mdl)
    return _loaded_models[name]


get_model(DEFAULT_MODEL)  # warm the default at startup

sessions = {}
evals = {}
gsm8k_test = load_dataset("openai/gsm8k", "main")["test"]

# Linear probe trained overnight: predicts outcome-entropy (ground truth via
# forced top-N branching + resampling) directly from the full hidden-state
# vector at one layer, rather than a hand-picked scalar entropy reduction of
# it. Only valid for the exact model it was trained on - different models
# have different hidden-state spaces entirely.
PROBE_MODEL_KEY = "qwen3-0.6b"
try:
    with open(REPO_ROOT / "probes" / "entropy_probe_layer22.json") as f:
        _probe_artifact = json.load(f)
    PROBE_LAYER = _probe_artifact["layer"]
    _probe_mean = torch.tensor(_probe_artifact["scaler_mean"], dtype=torch.float32, device=DEVICE)
    _probe_scale = torch.tensor(_probe_artifact["scaler_scale"], dtype=torch.float32, device=DEVICE)
    _probe_coef = torch.tensor(_probe_artifact["coef"], dtype=torch.float32, device=DEVICE)
    _probe_intercept = _probe_artifact["intercept"]
    PROBE_AVAILABLE = True
except FileNotFoundError:
    PROBE_AVAILABLE = False


def predict_outcome_entropy(hidden_state_vec):
    x = (hidden_state_vec.float() - _probe_mean) / _probe_scale
    return float(torch.dot(x, _probe_coef).item() + _probe_intercept)


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id])


def forward_logits(model, ids):
    probe_applicable = PROBE_AVAILABLE and model is _loaded_models.get(PROBE_MODEL_KEY, (None, None))[1]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=DEVICE), output_hidden_states=probe_applicable)
    logits = out.logits[0, -1]
    predicted_oe = None
    if probe_applicable:
        predicted_oe = predict_outcome_entropy(out.hidden_states[PROBE_LAYER][0, -1])
    return logits, predicted_oe


def top_k_distribution(tokenizer, logits, k=TOP_K_DISPLAY):
    probs = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, k)
    logit_vals = logits[topi]
    return [
        {
            "token_id": int(tid),
            "token_text": decode_token(tokenizer, int(tid)),
            "prob": float(p),
            "logit": float(lv),
        }
        for tid, p, lv in zip(topi.tolist(), topv.tolist(), logit_vals.tolist())
    ]


def compute_entropy(logits):
    """Shannon entropy in bits of the full-vocab softmax, from raw (unfiltered,
    temperature-1) logits — independent of whatever decoding mode/temperature
    is later used to actually pick a token."""
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log2(probs.clamp_min(1e-12))
    return float(-(probs * logp).sum().item())


def sample_from_logits(logits, mode, temperature, top_k, top_p, min_p=0.0):
    if mode == "greedy":
        return int(torch.argmax(logits).item())

    scaled = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)

    if top_k and top_k > 0:
        topv, topi = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs)
        mask.scatter_(0, topi, topv)
        probs = mask

    if top_p and 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        # keep the first token that crosses top_p
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        sorted_probs[cutoff] = 0.0
        probs = torch.zeros_like(probs)
        probs.scatter_(0, sorted_idx, sorted_probs)

    if min_p and min_p > 0:
        threshold = min_p * probs.max()
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))

    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(logits).item())
    probs = probs / total
    return int(torch.multinomial(probs, 1).item())


def full_ids(session):
    return session["prompt_ids"] + [h["token_id"] for h in session["history"]]


def state_payload(session):
    return {
        "session_id": session["id"],
        "model": session["model_name"],
        "prompt_tokens": session["prompt_tokens"],
        "history": session["history"],
        "next_distribution": session["pending_topk"],
        "next_entropy": session["pending_entropy"],
        "next_predicted_outcome_entropy": session.get("pending_predicted_oe"),
    }


class CreateSessionBody(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL


class ImportSessionBody(BaseModel):
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    model: str = DEFAULT_MODEL


class SelectBody(BaseModel):
    step_index: int
    token_id: int | None = None


class NextBody(BaseModel):
    mode: str = "greedy"  # "greedy" | "sample"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0


class ForceBody(BaseModel):
    text: str


class RunBody(BaseModel):
    mode: str = "greedy"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    steps: int = 20
    stop_on_eos: bool = True


@app.post("/api/session")
def create_session(body: CreateSessionBody):
    tokenizer, model = get_model(body.model)
    ids = tokenizer.encode(body.prompt, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(" ", add_special_tokens=False)

    logits, predicted_oe = forward_logits(model, ids)
    session_id = uuid.uuid4().hex
    session = {
        "id": session_id,
        "model_name": body.model,
        "prompt_ids": ids,
        "prompt_tokens": [{"token_id": tid, "token_text": decode_token(tokenizer, tid)} for tid in ids],
        "history": [],
        "pending_logits": logits,
        "pending_topk": top_k_distribution(tokenizer, logits),
        "pending_entropy": compute_entropy(logits),
        "pending_predicted_oe": predicted_oe,
    }
    sessions[session_id] = session
    return state_payload(session)


@app.post("/api/session/import")
def import_session(body: ImportSessionBody):
    tokenizer, model = get_model(body.model)
    prompt_ids = body.prompt_token_ids
    gen_ids = body.generated_token_ids
    if not prompt_ids:
        raise HTTPException(400, "prompt_token_ids must be non-empty")

    probe_applicable = PROBE_AVAILABLE and body.model == PROBE_MODEL_KEY
    full = prompt_ids + gen_ids
    with torch.no_grad():
        out = model(input_ids=torch.tensor([full], device=DEVICE), output_hidden_states=probe_applicable)
    logits_full = out.logits[0]

    base = len(prompt_ids)
    history = []
    for i, tid in enumerate(gen_ids):
        step_logits = logits_full[base + i - 1]
        predicted_oe = None
        if probe_applicable:
            predicted_oe = predict_outcome_entropy(out.hidden_states[PROBE_LAYER][0, base + i - 1])
        history.append({
            "step_index": i,
            "token_id": tid,
            "token_text": decode_token(tokenizer, tid),
            "source": "imported",
            "topk": top_k_distribution(tokenizer, step_logits),
            "entropy": compute_entropy(step_logits),
            "predicted_outcome_entropy": predicted_oe,
        })

    pending_logits = logits_full[-1]
    pending_predicted_oe = None
    if probe_applicable:
        pending_predicted_oe = predict_outcome_entropy(out.hidden_states[PROBE_LAYER][0, -1])
    session_id = uuid.uuid4().hex
    session = {
        "id": session_id,
        "model_name": body.model,
        "prompt_ids": prompt_ids,
        "prompt_tokens": [{"token_id": tid, "token_text": decode_token(tokenizer, tid)} for tid in prompt_ids],
        "history": history,
        "pending_logits": pending_logits,
        "pending_topk": top_k_distribution(tokenizer, pending_logits),
        "pending_entropy": compute_entropy(pending_logits),
        "pending_predicted_oe": pending_predicted_oe,
    }
    sessions[session_id] = session
    return state_payload(session)


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return state_payload(session)


@app.post("/api/session/{session_id}/select")
def select_token(session_id: str, body: SelectBody):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    if not (0 <= body.step_index <= len(session["history"])):
        raise HTTPException(400, "step_index out of range")
    tokenizer, model = get_model(session["model_name"])

    truncated = session["history"][: body.step_index]
    session["history"] = truncated

    if body.token_id is not None:
        current_ids = full_ids(session)
        logits, predicted_oe = forward_logits(model, current_ids)
        new_entry = {
            "step_index": body.step_index,
            "token_id": body.token_id,
            "token_text": decode_token(tokenizer, body.token_id),
            "source": "selected",
            "topk": top_k_distribution(tokenizer, logits),
            "entropy": compute_entropy(logits),
            "predicted_outcome_entropy": predicted_oe,
        }
        session["history"].append(new_entry)

    new_ids = full_ids(session)
    pending_logits, pending_predicted_oe = forward_logits(model, new_ids)
    session["pending_logits"] = pending_logits
    session["pending_topk"] = top_k_distribution(tokenizer, pending_logits)
    session["pending_entropy"] = compute_entropy(pending_logits)
    session["pending_predicted_oe"] = pending_predicted_oe
    return state_payload(session)


def step_once(session, mode, temperature, top_k, top_p, min_p=0.0):
    tokenizer, model = get_model(session["model_name"])
    pending_logits = session["pending_logits"]
    token_id = sample_from_logits(pending_logits, mode, temperature, top_k, top_p, min_p)
    entry = {
        "step_index": len(session["history"]),
        "token_id": token_id,
        "token_text": decode_token(tokenizer, token_id),
        "source": "sampled",
        "mode": mode,
        "topk": session["pending_topk"],
        "entropy": session["pending_entropy"],
        "predicted_outcome_entropy": session.get("pending_predicted_oe"),
    }
    session["history"].append(entry)

    new_ids = full_ids(session)
    pending_logits, pending_predicted_oe = forward_logits(model, new_ids)
    session["pending_logits"] = pending_logits
    session["pending_topk"] = top_k_distribution(tokenizer, pending_logits)
    session["pending_entropy"] = compute_entropy(pending_logits)
    session["pending_predicted_oe"] = pending_predicted_oe
    return token_id


@app.post("/api/session/{session_id}/next")
def next_token(session_id: str, body: NextBody):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    step_once(session, body.mode, body.temperature, body.top_k, body.top_p, body.min_p)
    return state_payload(session)


def run_steps_generator(session, mode, temperature, top_k, top_p, min_p, steps, stop_on_eos):
    tokenizer, _ = get_model(session["model_name"])
    steps = max(1, min(steps, 4000))
    eos_id = tokenizer.eos_token_id
    for _ in range(steps):
        token_id = step_once(session, mode, temperature, top_k, top_p, min_p)
        gen_ids = [h["token_id"] for h in session["history"]]
        reason = None
        if stop_on_eos and eos_id is not None and token_id == eos_id:
            reason = "eos"
        elif detect_repetition(gen_ids):
            reason = "repetition"
        yield reason
        if reason:
            return


@app.post("/api/session/{session_id}/run")
def run_tokens(session_id: str, body: RunBody):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")

    stop_reason = None
    for reason in run_steps_generator(
        session, body.mode, body.temperature, body.top_k, body.top_p, body.min_p, body.steps, body.stop_on_eos
    ):
        stop_reason = reason
    payload = state_payload(session)
    payload["stop_reason"] = stop_reason
    return payload


@app.post("/api/session/{session_id}/run_stream")
async def run_stream(session_id: str, body: RunBody, request: Request):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")

    async def gen():
        for reason in run_steps_generator(
            session, body.mode, body.temperature, body.top_k, body.top_p, body.min_p, body.steps, body.stop_on_eos
        ):
            payload = state_payload(session)
            payload["stop_reason"] = reason
            yield json.dumps(payload) + "\n"
            if await request.is_disconnected():
                break

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/session/{session_id}/force")
def force_sequence(session_id: str, body: ForceBody):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    tokenizer, model = get_model(session["model_name"])

    tok_ids = tokenizer.encode(body.text, add_special_tokens=False)
    if not tok_ids:
        return state_payload(session)

    pending_logits = session["pending_logits"]
    pending_predicted_oe = session.get("pending_predicted_oe")
    for i, tid in enumerate(tok_ids):
        if i == 0:
            logits, predicted_oe = pending_logits, pending_predicted_oe
        else:
            logits, predicted_oe = forward_logits(model, full_ids(session))
        entry = {
            "step_index": len(session["history"]),
            "token_id": tid,
            "token_text": decode_token(tokenizer, tid),
            "source": "forced",
            "topk": top_k_distribution(tokenizer, logits),
            "entropy": compute_entropy(logits),
            "predicted_outcome_entropy": predicted_oe,
        }
        session["history"].append(entry)

    new_ids = full_ids(session)
    pending_logits, pending_predicted_oe = forward_logits(model, new_ids)
    session["pending_logits"] = pending_logits
    session["pending_topk"] = top_k_distribution(tokenizer, pending_logits)
    session["pending_entropy"] = compute_entropy(pending_logits)
    session["pending_predicted_oe"] = pending_predicted_oe
    return state_payload(session)


class EvalRunBody(BaseModel):
    # Defaults use non-thinking mode: thinking mode's own internal, uncontrolled
    # deliberation has no stable stopping point for this model (observed 45-55%
    # truncation even at 1000+ tokens) and gives no discrete boundaries to branch
    # on. Non-thinking mode + an explicit step-by-step template moves "generate
    # candidates / explore" out of the model's own run-on monologue and into
    # something this toolkit can actually branch on - concludes ~5x shorter,
    # effectively never loops once paired with the repetition guard, and gives
    # explicit 'Step N:' boundaries. Settings match Qwen's officially
    # recommended NON-thinking decoding params (model card): temperature 0.7,
    # top_k 20, top_p 0.8.
    model: str = DEFAULT_MODEL
    n_samples: int = 20
    offset: int = 0
    batch_size: int = 8
    max_new_tokens: int = 512
    enable_thinking: bool = False
    mode: str = "sample"
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    template: str = DEFAULT_TEMPLATE


@app.post("/api/eval/run_stream")
async def eval_run_stream(body: EvalRunBody, request: Request):
    if "{question}" not in body.template:
        raise HTTPException(400, "template must contain {question}")

    tokenizer, model = get_model(body.model)
    n = max(1, min(body.n_samples, 500))
    batch_size = max(1, min(body.batch_size, 32))
    offset = max(0, body.offset)
    max_new_tokens = max(16, min(body.max_new_tokens, 2048))

    eval_id = uuid.uuid4().hex
    evals[eval_id] = {"params": body.model_dump(), "results": [], "status": "running"}

    async def gen():
        tokenizer.padding_side = "left"
        results = evals[eval_id]["results"]
        indices = list(range(offset, min(offset + n, len(gsm8k_test))))

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            examples = [gsm8k_test[i] for i in batch_idx]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body.template.replace("{question}", ex["question"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=body.enable_thinking,
                )
                for ex in examples
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            prompt_len = enc["input_ids"].shape[1]

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([
                    ForceEosAfterAnswer(tokenizer, prompt_len, tokenizer.eos_token_id, body.enable_thinking),
                    ForceEosOnRepetition(prompt_len, tokenizer.eos_token_id),
                ]),
            )
            if body.mode == "sample":
                gen_kwargs.update(
                    do_sample=True,
                    temperature=max(body.temperature, 1e-5),
                    top_k=body.top_k if body.top_k > 0 else 0,
                    top_p=body.top_p,
                    min_p=body.min_p if body.min_p > 0 else None,
                )
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)

            gen_tokens_batch = out[:, prompt_len:]

            new_results = []
            for j, ex in enumerate(examples):
                gen_ids_row = gen_tokens_batch[j].tolist()
                eos_hit = tokenizer.eos_token_id in gen_ids_row
                # trim trailing pad filler that HF appends after an early-finished row's EOS
                if eos_hit:
                    eos_pos = gen_ids_row.index(tokenizer.eos_token_id)
                    true_gen_ids = gen_ids_row[:eos_pos + 1]
                else:
                    true_gen_ids = gen_ids_row
                true_prompt_ids = enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist()

                gen_text = tokenizer.decode(gen_ids_row, skip_special_tokens=True)
                pred = extract_answer(gen_text)
                ref = extract_answer(ex["answer"])
                correct = pred is not None and ref is not None and abs(pred - ref) < 1e-4
                result = {
                    "index": batch_idx[j],
                    "model": body.model,
                    "question": ex["question"],
                    "reference_text": ex["answer"],
                    "reference_answer": ref,
                    "generated_text": gen_text,
                    "predicted_answer": pred,
                    "correct": correct,
                    "truncated": not eos_hit,
                    "prompt_token_ids": true_prompt_ids,
                    "generated_token_ids": true_gen_ids,
                }
                results.append(result)
                new_results.append(result)

            n_correct = sum(r["correct"] for r in results)
            payload = {
                "type": "progress",
                "eval_id": eval_id,
                "completed": len(results),
                "total": len(indices),
                "n_correct": n_correct,
                "batch_results": new_results,
            }
            yield json.dumps(payload) + "\n"
            if await request.is_disconnected():
                evals[eval_id]["status"] = "stopped"
                return

        evals[eval_id]["status"] = "done"
        n_correct = sum(r["correct"] for r in results)
        yield json.dumps({
            "type": "done",
            "eval_id": eval_id,
            "n_correct": n_correct,
            "n_total": len(results),
            "accuracy": n_correct / len(results) if results else 0,
        }) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/eval/{eval_id}")
def get_eval(eval_id: str):
    ev = evals.get(eval_id)
    if ev is None:
        raise HTTPException(404, "eval not found")
    n_correct = sum(r["correct"] for r in ev["results"])
    return {
        "eval_id": eval_id,
        "status": ev["status"],
        "params": ev["params"],
        "results": ev["results"],
        "n_correct": n_correct,
        "n_total": len(ev["results"]),
    }


@app.get("/api/models")
def list_models():
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {"key": key, "repo": repo, "loaded": key in _loaded_models}
            for key, repo in MODEL_REGISTRY.items()
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(HERE / "static" / "index.html") as f:
        return f.read()


@app.get("/eval", response_class=HTMLResponse)
def eval_page():
    with open(HERE / "static" / "eval.html") as f:
        return f.read()
