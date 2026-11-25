#!/usr/bin/env python3
"""Decode path against prefill path on the same tokens: a served build's own consistency check.

For each frozen prompt (token ids from build_parity.py): generate up to N tokens greedily on the decode path
(/v1/completions, logprobs, return_token_ids), then score prompt + reply in one prefill pass (prompt_logprobs) and
compare position by position. A sound build has the decode path's greedy pick ranked first by the prefill path almost
everywhere, and their log-probabilities of it close. Compare a candidate's numbers with the baseline build's: the two
paths use different kernels, so a fault in either shows up as a gap the baseline does not have.

usage: parity.py PARITY.json [--url http://127.0.0.1:8888] [--model NAME] [--tokens 512] [--save OUT.json]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # the server's reason, not just the status
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parity")
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="GLM-5.3-NVFP4")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--save")
    a = ap.parse_args()
    data = json.load(open(a.parity))
    by_kind, saved = {}, []
    for i, p in enumerate(data["prompts"]):
        body = {"model": a.model, "prompt": p["ids"], "max_tokens": a.tokens, "temperature": 0, "return_token_ids": True}
        try:
            gen = post(a.url, body | {"logprobs": 0})
        except RuntimeError as err:  # decode logprobs refused: the rank check alone (the test's core) still runs
            print(f"p{i:02d}: {err}; retrying without decode logprobs", file=sys.stderr, flush=True)
            gen = post(a.url, body)
        reply = gen["tensorfold"]["token_ids"]
        dec_lp = ((gen["choices"][0].get("logprobs") or {}).get("token_logprobs")) or [None] * len(reply)
        scored = post(a.url, {"model": a.model, "prompt": p["ids"] + reply, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
        rows = scored.get("prompt_logprobs") or scored["choices"][0].get("prompt_logprobs")
        rows = rows[len(p["ids"]):]  # the reply's positions: row j scores reply token j
        n = min(len(reply), len(rows), len(dec_lp))
        not_top, dlp = 0, []
        for j in range(n):
            entry = rows[j]
            tok = str(reply[j])
            info = entry.get(tok) or entry.get(int(tok)) if isinstance(entry, dict) else None
            if info is None:
                not_top += 1
                continue
            if info.get("rank", 1) != 1:
                not_top += 1
            if dec_lp[j] is not None:
                dlp.append(abs(info["logprob"] - dec_lp[j]))
        k = by_kind.setdefault(p["kind"], {"tokens": 0, "not_top": 0, "dlp": []})
        k["tokens"] += n
        k["not_top"] += not_top
        k["dlp"] += dlp
        saved.append({"kind": p["kind"], "reply": reply, "decode_logprobs": dec_lp[:n]})
        print(f"p{i:02d} {p['kind']:5s} {n:4d} tokens, prefill disagrees with the decode pick at {not_top}, "
              f"mean |dlogp| {sum(dlp) / max(len(dlp), 1):.4f}", flush=True)
    for kind, k in by_kind.items():
        d = sorted(k["dlp"])
        print(f"{kind:5s}: {k['tokens']} tokens, decode pick not prefill's top-1 at {k['not_top']} "
              f"({100 * k['not_top'] / max(k['tokens'], 1):.2f}%), |dlogp| mean {sum(d) / max(len(d), 1):.4f} "
              f"p99 {d[int(0.99 * (len(d) - 1))] if d else 0:.4f}")
    if a.save:
        json.dump(saved, open(a.save, "w"))


if __name__ == "__main__":
    main()
