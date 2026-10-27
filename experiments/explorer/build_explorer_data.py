"""
Builds the Fork Explorer's data.json (the page at matthewbragonje.com/fork-explorer/)
from debug_strategy_trace.py dumps that add_local_entropy.py has annotated.

Keeps problems with integer reference answers (decimal answers were exposed to a
stop-rule bug in these runs), the top-5 alternatives per token, and each forced
branch's continuation truncated to 1200 characters.

Usage: python experiments/explorer/build_explorer_data.py <out.json> <dump.json> [<dump.json> ...]
"""
import json
import sys

out_path, dumps = sys.argv[1], sys.argv[2:]
problems = []
for path in dumps:
    for d in json.load(open(path)):
        if not float(d["reference"]).is_integer():
            continue
        if d.get("local_ent") is None:
            sys.exit(f"{path}: run experiments/explorer/add_local_entropy.py on this dump first")
        cuts = {
            k: [{"t": b["token"], "p": round(b["prob"], 4), "a": b["answer"], "ok": b["correct"], "x": b["text"][:1200]}
                for b in c["branches"]]
            for k, c in d["cuts"].items()
        }
        problems.append({
            "id": d["test_index"], "q": d["problem"], "ref": d["reference"], "ans": d["trace_answer"],
            "toks": d["token_texts"], "z": [round(v, 2) for v in d["probe_z"]], "le": d["local_ent"],
            "tk": [[[t, round(p, 4)] for t, p in row[:5]] for row in d["topk"]], "cuts": cuts,
        })
json.dump(problems, open(out_path, "w"), ensure_ascii=False, separators=(",", ":"))
print(f"{len(problems)} problems -> {out_path}")
