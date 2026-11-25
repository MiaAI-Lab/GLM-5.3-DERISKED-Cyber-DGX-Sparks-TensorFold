#!/usr/bin/env python3
"""Prefill and decode throughput for the TP6 recipe, one request at a time (the old 3x/4x recipes' figures:
prefill at 8k / 32k / 128k prompt tokens, decode of 512 tokens on prose and on code).

Prefill tok/s = prompt tokens / time to first token, from the client (stream=True). Decode tok/s = (tokens - 1) /
(last token time - first token time). Each case runs --runs times; the median is reported.

Usage: bench.py [--url http://127.0.0.1:8888] [--prefill 8192,32768,131072] [--decode 512] [--runs 3]
"""
import argparse
import json
import os
import random
import statistics
import time
import urllib.request

WORDS = ("river stone garden lantern copper falcon meadow harbor violet granite orbit ember thistle walnut saddle "
         "quartz bramble cobalt zephyr pepper maple signal archive engine voyage chamber silver compass winter "
         "summer forest valley mountain ocean island bridge castle market village tower lighthouse").split()


def filler(n_words, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n_words // 12):
        s = " ".join(rng.choice(WORDS) for _ in range(11))
        out.append(f"{s.capitalize()} {rng.randint(10, 9999)}.")
    return " ".join(out)


def stream(url, messages, max_tokens, fill=False):
    req = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
           "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("TEST_MODEL", "GLM-5.3-NVFP4"):  # TEST_MODEL="": omit the name (servers that check it)
        req["model"] = os.environ.get("TEST_MODEL", "GLM-5.3-NVFP4")
    if fill:  # sparkDash's DecodeBench: every reply runs to max_tokens
        req.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    body = json.dumps(req).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    pieces = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    now = time.monotonic()
                    first = first or now
                    last = now
                    pieces += 1
    return t0, first, last, usage


def prefill(url, target, seed):
    # ~1.3 tokens a word for this filler; calibrated once per size by the server's count
    msg = filler(int(target / 1.3), seed) + "\n\nReply with the single word: done"
    t0, first, _, usage = stream(url, [{"role": "user", "content": msg}], 1)
    pt = usage.get("prompt_tokens", 0)
    return pt, pt / (first - t0) if first else 0.0


PROSE = "Write a long, detailed short story about a lighthouse keeper who finds a map in a bottle. Do not stop early."
CODE = ("Write a complete Python module implementing a red-black tree (insert, delete, search, in-order iteration) "
        "with docstrings and a full unittest suite. Output only code.")


# sparkDash (github.com/MiaAI-Lab/sparkDash, src/shared/llmPrompts.js): the old 3x/4x recipes' decode figures used its
# DecodeBench: text prompts as "prose", structural as "code", a fill-to-max suffix, thinking off, ignore_eos.
SD_FILL = " Continue generating until you hit the maximum output length; do not stop early—keep expanding with more content."
SD_PROSE = [
    "Write a clear essay explaining unified memory on NVIDIA GB10 Sparks for a technical but non-specialist reader. Keep expanding with examples and analogies.",
    "Write a vivid sci-fi scene set in a liquid-cooled server room at 3 a.m. Keep expanding the scene with sensory detail and dialogue.",
    "Write a pirate-captain monologue explaining KV-cache pressure and prefill vs decode to the crew. Keep expanding with more shanties and metaphors.",
]
SD_CODE = [
    "Emit only a JSON array of fake GPU metrics rows. Each object needs host, gpuIndex, utilPct, tempC, powerW, memUsedMb. Invent many rows. No markdown. Keep expanding the array.",
    "Emit only an HTML FAQ about CUDA and vLLM. Use <h2> and <p> for many Q&A pairs. No markdown fences. Keep adding sections.",
    "Emit a Markdown comparison table: llama.cpp vs vLLM vs SGLang. Columns: feature, llama.cpp, vLLM, SGLang. Fill many rows and keep adding.",
]


def decode(url, prompt, n, fill=False):
    _, first, last, usage = stream(url, [{"role": "user", "content": prompt}], n, fill)
    toks = usage.get("completion_tokens", 0)
    return toks, (toks - 1) / (last - first) if first and last and last > first else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--prefill", default="8192,32768,131072")
    ap.add_argument("--decode", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--suite", choices=("own", "sparkdash"), default="own", help="decode prompts: ours, or sparkDash's")
    a = ap.parse_args()
    for n in [int(x) for x in a.prefill.split(",") if x]:
        rs = [prefill(a.url, n, seed=n * 10 + i) for i in range(a.runs)]
        print(f"prefill {rs[0][0]:>7} tok: {statistics.median(r[1] for r in rs):7.1f} tok/s  "
              f"(runs {', '.join(f'{r[1]:.1f}' for r in rs)})", flush=True)
    if a.decode and a.suite == "sparkdash":
        for name, ps in (("prose", SD_PROSE), ("code", SD_CODE)):
            rs = [decode(a.url, p + SD_FILL, a.decode, fill=True) for p in ps]
            print(f"sparkdash {name:5} {rs[0][0]:>4} tok: {statistics.median(r[1] for r in rs):6.1f} tok/s  "
                  f"(prompts {', '.join(f'{r[1]:.1f}' for r in rs)})", flush=True)
    elif a.decode:
        for name, p in (("prose", PROSE), ("code", CODE)):
            rs = [decode(a.url, p, a.decode) for _ in range(a.runs)]
            print(f"decode {name:5} {rs[0][0]:>4} tok: {statistics.median(r[1] for r in rs):6.1f} tok/s  "
                  f"(runs {', '.join(f'{r[1]:.1f}' for r in rs)})", flush=True)


if __name__ == "__main__":
    main()
