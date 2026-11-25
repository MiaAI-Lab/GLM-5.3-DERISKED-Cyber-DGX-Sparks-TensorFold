#!/usr/bin/env python3
"""Send a frozen KL panel's windows (token ids) to any OpenAI-style /v1/completions server with prompt_logprobs=K and
keep, for every predicted position, the server's top-K ids and log-probabilities plus the actual next token's.

usage: panel_topk.py PANEL.json OUT.bin [--url http://127.0.0.1:8100] [--model NAME] [--k 20]
  For servers that return only the top K of each row (gr8: K <= 20), where TF_GLM_DUMP_LOGITS's full rows are not
  available. score_topk.py scores this file, and full-row dumps reduced to the same top K, identically.

OUT.bin: windows in panel order, each `rows` = tokens_per_window - 1 records (row r predicts token r+1):
  int32 ids[K], float32 lps[K] (descending), float32 actual_lp, int32 actual_rank (1 = top; 0 = not reported)
OUT.bin.json: {"k", "rows", "windows", "panel_sha256", "url", "model", "seconds"}.
"""
import argparse
import json
import time
import urllib.request

import numpy as np


def rec_dtype(k):
    return np.dtype([("ids", "<i4", (k,)), ("lps", "<f4", (k,)), ("act", "<f4"), ("rank", "<i4")])


def fetch(url, model, ids, k):
    body = {"prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": k}
    if model:
        body["model"] = model
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        res = json.loads(r.read())
    plp = res.get("prompt_logprobs")
    if plp is None:
        plp = (res.get("choices") or [{}])[0].get("prompt_logprobs")
    if plp is None:
        raise SystemExit("no prompt_logprobs in the response: " + json.dumps(res)[:400])
    return plp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    panel = json.load(open(a.panel))
    rows = panel["tokens_per_window"] - 1
    k = a.k
    dt = rec_dtype(k)
    t0 = time.time()
    with open(a.out, "wb") as f:
        for w, win in enumerate(panel["windows"]):
            ids = win["ids"]
            assert len(ids) == rows + 1, (w, len(ids))
            plp = fetch(a.url, a.model, ids, k)
            if len(plp) != rows + 1:
                raise SystemExit(f"w{w}: {len(plp)} prompt_logprobs entries for {rows + 1} tokens")
            out = np.zeros(rows, dtype=dt)
            out["ids"] = -1
            out["lps"] = -np.inf
            for r in range(rows):
                d = plp[r + 1]  # position j = r + 1: the distribution predicted at r, scored on token j
                actual = ids[r + 1]
                top = []
                for tid, e in d.items():
                    tid = int(tid)
                    rk = e.get("rank")
                    if tid == actual:
                        out["act"][r] = e["logprob"]
                        out["rank"][r] = rk if rk is not None else 0
                    if rk is not None and 1 <= rk <= k:
                        top.append((rk, tid, e["logprob"]))
                top.sort()
                for i, (_, tid, lp) in enumerate(top[:k]):
                    out["ids"][r, i] = tid
                    out["lps"][r, i] = lp
                if str(actual) not in d:
                    out["act"][r] = np.nan
            out.tofile(f)
            f.flush()
            print(f"w{w:02d} {win['domain']:5s} rows {rows} mean actual NLL {-np.nanmean(out['act']):.4f}"
                  f" top-1 = actual {100 * np.mean(out['ids'][:, 0] == np.array(ids[1:])):.1f}%"
                  f" {time.time() - t0:.0f} s", flush=True)
    json.dump({"k": k, "rows": rows, "windows": len(panel["windows"]), "panel_sha256": panel.get("sha256"),
               "url": a.url, "model": a.model, "seconds": time.time() - t0}, open(a.out + ".json", "w"))
    print("done", a.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
