#!/usr/bin/env python3
"""Drafting exactness and speed in one boot: every prompt twice, with drafts and with "draft": false.

Drafts only propose, so a reply with drafts must carry the same token ids as the same request without them: greedy,
and sampled with a fixed seed. Any difference prints "PARITY DIFF" and the exit code is 1.

usage: draft_parity.py PARITY.json [--url http://127.0.0.1:8888] [--model NAME] [--tokens 512] [--limit N]
                       [--sampled] [--copy] [--save OUT.json]
  PARITY.json: {"prompts": [{"kind", "ids"}, ...]} (token-id prompts, /v1/completions).
  --sampled: also each prompt at temperature 1.0, top_p 0.95, seed 1234 (drafts on and off).
  --copy: also six copy-heavy chat prompts (rewrites of a given text or code), where copy rounds and deep drafts fire.
Per reply: tokens, wall seconds, decode tok/s (after the first token), and the server's speculative counters
(rounds, drafted, accepted). Summary per kind: acceptance, tokens a round, drafted/undrafted decode tok/s.
"""
import json
import sys
import time
import urllib.error
import urllib.request

COPY_SRC = '''def merge_intervals(intervals):
    """Merge overlapping [start, end] intervals and return them sorted."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
        self.order = []

    def get(self, key):
        if key not in self.data:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.order.remove(key)
        elif len(self.data) >= self.capacity:
            oldest = self.order.pop(0)
            del self.data[oldest]
        self.data[key] = value
        self.order.append(key)
'''
PROSE_SRC = ("The committee met on Tuesday to review the proposal for the new library wing. After a long discussion "
             "about the budget, the members agreed that the reading room should face the garden, that the archive "
             "should move to the basement where the temperature is stable, and that the children's section should "
             "keep its own entrance. The architect will present revised drawings next month, and the treasurer will "
             "report on the fundraising campaign, which has so far collected about two thirds of the target.")
COPY_PROMPTS = [
    ("copy-code", "Rewrite this Python code exactly as it is, but rename the variable `merged` to `result` everywhere. "
                  "Output only the code.\n\n" + COPY_SRC),
    ("copy-code", "Add type hints to every function in this code and keep everything else identical. Output only the "
                  "code.\n\n" + COPY_SRC),
    ("copy-code", "Repeat the following code verbatim, then add one line comment above each method.\n\n" + COPY_SRC),
    ("copy-prose", "Repeat this paragraph word for word, then repeat it again with every sentence in the past "
                   "perfect.\n\n" + PROSE_SRC),
    ("copy-prose", "Copy this text exactly, replacing 'library' with 'museum':\n\n" + PROSE_SRC),
    ("copy-prose", "Translate this paragraph into formal English while keeping its structure, sentence by sentence:\n\n"
                   + PROSE_SRC),
]


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}")
    return out, time.monotonic() - t0


def run(url, path, body, draft):
    out, wall = post(url, path, body | {"draft": draft, "return_token_ids": True})
    rt = out.get("tensorfold") or {}
    sp = out.get("speculative") or {}
    ids = rt.get("token_ids")
    if ids is None:
        raise SystemExit("the server returned no tensorfold.token_ids (return_token_ids)")
    ttft = rt.get("time_to_first_token") or 0.0
    n = len(ids)
    dec = (n - 1) / max(wall - ttft, 1e-9) if n > 1 else 0.0
    return {"ids": ids, "sha256": rt.get("token_sha256"), "n": n, "wall": wall, "ttft": ttft, "decode_tps": dec,
            "rounds": sp.get("rounds", 0), "drafted": sp.get("drafted", 0), "accepted": sp.get("accepted", 0)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sampled", action="store_true")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--kinds", default="", help="only these prompt kinds (comma-separated), e.g. tool")
    a = ap.parse_args()
    model = a.model
    if model is None:
        with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    prompts = json.load(open(a.panel))["prompts"]
    if a.kinds:
        prompts = [p for p in prompts if p["kind"] in a.kinds.split(",")]
    if a.limit:
        prompts = prompts[:a.limit]
    cases = []
    for i, p in enumerate(prompts):
        base = {"model": model, "prompt": p["ids"], "max_tokens": a.tokens}
        cases.append((f"{p['kind']}#{i}", p["kind"], "/v1/completions", base | {"temperature": 0}))
        if a.sampled:
            cases.append((f"{p['kind']}#{i}/s", p["kind"] + "-sampled", "/v1/completions",
                          base | {"temperature": 1.0, "top_p": 0.95, "seed": 1234}))
    if a.copy:
        for i, (kind, text) in enumerate(COPY_PROMPTS):
            cases.append((f"{kind}#{i}", kind, "/v1/chat/completions",
                          {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": a.tokens,
                           "temperature": 0}))
    results, diffs = [], 0
    for name, kind, path, body in cases:
        on = run(a.url, path, body, True)
        off = run(a.url, path, body, False)
        same = on["ids"] == off["ids"]
        if not same:
            diffs += 1
            first = next((j for j, (x, y) in enumerate(zip(on["ids"], off["ids"])) if x != y), min(on["n"], off["n"]))
            print(f"PARITY DIFF {name}: first differing token {first} (draft {on['ids'][first:first + 4]} vs "
                  f"serial {off['ids'][first:first + 4]}), lengths {on['n']} / {off['n']}", flush=True)
        acc = on["accepted"] / on["drafted"] if on["drafted"] else 0.0
        tpr = on["n"] / on["rounds"] if on["rounds"] else 0.0
        print(f"{name:16s} {'same' if same else 'DIFF':4s} n={on['n']:4d} drafted {on['decode_tps']:6.1f} tok/s, "
              f"serial {off['decode_tps']:6.1f} tok/s, acceptance {acc:.3f}, tokens/round {tpr:.2f}", flush=True)
        results.append({"name": name, "kind": kind, "same": same, "on": {k: v for k, v in on.items() if k != "ids"},
                        "off": {k: v for k, v in off.items() if k != "ids"}})
    print("\nper kind: replies, acceptance, tokens/round, drafted tok/s, serial tok/s, speedup")
    summary = {}
    for kind in dict.fromkeys(r["kind"] for r in results):
        rs = [r for r in results if r["kind"] == kind]
        dr = sum(r["on"]["drafted"] for r in rs)
        ac = sum(r["on"]["accepted"] for r in rs)
        rd = sum(r["on"]["rounds"] for r in rs)
        n = sum(r["on"]["n"] for r in rs)
        t_on = sum(r["on"]["wall"] - r["on"]["ttft"] for r in rs)
        t_off = sum(r["off"]["wall"] - r["off"]["ttft"] for r in rs)
        n_off = sum(r["off"]["n"] for r in rs)
        s = {"replies": len(rs), "acceptance": ac / dr if dr else 0.0, "tokens_per_round": n / rd if rd else 0.0,
             "drafted_tps": n / t_on if t_on else 0.0, "serial_tps": n_off / t_off if t_off else 0.0}
        s["speedup"] = s["drafted_tps"] / s["serial_tps"] if s["serial_tps"] else 0.0
        summary[kind] = s
        print(f"  {kind:14s} {s['replies']:3d} {s['acceptance']:.3f} {s['tokens_per_round']:.2f} "
              f"{s['drafted_tps']:7.1f} {s['serial_tps']:7.1f} {s['speedup']:.3f}")
    print(f"\nPARITY {'PASS' if diffs == 0 else 'FAIL'}: {diffs} differing replies of {len(results)}")
    if a.save:
        json.dump({"summary": summary, "results": results, "diffs": diffs}, open(a.save, "w"), indent=1)
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
