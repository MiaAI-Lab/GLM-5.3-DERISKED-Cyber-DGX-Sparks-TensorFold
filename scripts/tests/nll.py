#!/usr/bin/env python3
"""Mean negative log-likelihood of fixed texts under the served model (prompt_logprobs on /v1/completions).

A quality gate for numerics changes (DENSE bits, partial precision, kernels): run it before and after on the same
texts; a change that moves the mean NLL by more than run-to-run noise (about 1e-3 at these sizes) needs a reason.

Usage: nll.py FILE [FILE ...] [--url http://127.0.0.1:8888] [--max-chars 24000]
"""
import argparse
import json
import math
import urllib.request


def nll(url, text, k=0, keep=None):
    body = json.dumps({"model": "GLM-5.3-NVFP4", "prompt": text, "max_tokens": 1, "temperature": 0,
                       "prompt_logprobs": k}).encode()
    req = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        reply = json.loads(r.read())
    lps = reply.get("prompt_logprobs") or reply["choices"][0].get("prompt_logprobs") or []
    vals = []
    for entry in lps:
        if not entry:
            continue  # the first token has none
        if isinstance(entry, dict):  # {token_id: {"logprob": x, ...}} as vLLM writes it
            v = next(iter(entry.values()))
            vals.append(v["logprob"] if isinstance(v, dict) else float(v))
            if keep is not None:  # the prompt token (listed first) and the top k, for kl.py
                keep.append([next(iter(entry)), {t: x["logprob"] for t, x in entry.items() if 1 <= x.get("rank", 0) <= k}])
        else:
            vals.append(float(entry))
    return -sum(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--save", help="write each position's top --k to this JSON for scripts/tests/kl.py")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    saved = {}
    total, count = 0.0, 0
    for f in a.files:
        text = open(f, encoding="utf-8", errors="replace").read()[: a.max_chars]
        keep = [] if a.save else None
        s, n = nll(a.url, text, a.k if a.save else 0, keep)
        if a.save:
            saved[f] = keep
        total += s
        count += n
        print(f"{f}: {n} tokens, mean NLL {s / max(n, 1):.5f}", flush=True)
    if a.save:
        json.dump(saved, open(a.save, "w"))
    print(f"ALL: {count} tokens, mean NLL {total / max(count, 1):.5f} (ppl {math.exp(total / max(count, 1)):.4f})")


if __name__ == "__main__":
    main()
