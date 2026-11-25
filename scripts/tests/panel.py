#!/usr/bin/env python3
"""Send a frozen KL panel's windows (token ids) through the served model with prompt_logprobs, in order.

With the ranks started under TF_GLM_DUMP_LOGITS=path (engine patch 0154), rank 0 appends every window's whole bf16
logit rows to `path` ([2047][vocab] a 2,048-token window, row r scoring token r + 1); panel_kl.py compares them with
a reference's log-probabilities. Prints each window's mean NLL as a sanity check.

usage: panel.py PANEL.json [--url http://127.0.0.1:8888] [--model NAME]
"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="GLM-5.3-NVFP4")
    a = ap.parse_args()
    panel = json.load(open(a.panel))
    print("panel", panel["sha256"], len(panel["windows"]), "windows", flush=True)
    for i, w in enumerate(panel["windows"]):
        body = json.dumps({"model": a.model, "prompt": w["ids"], "max_tokens": 1, "temperature": 0,
                           "prompt_logprobs": 0}).encode()
        req = urllib.request.Request(a.url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            reply = json.loads(r.read())
        lps = reply.get("prompt_logprobs") or reply["choices"][0].get("prompt_logprobs") or []
        vals = []
        for e in lps:
            if e:
                v = next(iter(e.values())) if isinstance(e, dict) else e
                vals.append(v["logprob"] if isinstance(v, dict) else float(v))
        print(f"w{i:02d} {w['domain']:5s} rows {len(vals)} mean NLL {-sum(vals) / max(len(vals), 1):.4f}", flush=True)


if __name__ == "__main__":
    main()
