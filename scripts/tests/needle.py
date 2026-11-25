#!/usr/bin/env python3
"""Long-context needle-in-a-haystack test for the TP6 recipe.

Builds a prompt of ~target_tokens of filler with a unique needle fact buried
at a configurable depth, asks for the fact, and verifies the reply contains
it. Token count is calibrated against the server's own usage accounting:
a probe measures tokens-per-filler-block, then the filler is scaled to hit
the target.

Usage: needle.py [--target 200000] [--depth 0.85] [--url http://127.0.0.1:8888]
"""
import argparse
import json
import os
import sys
import urllib.request

NEEDLE_KEY = "TITANIUM-OWL-7314"
NEEDLE = ("The archived registry entry reads: the magic passkey for the "
          "northern vault is TITANIUM-OWL-7314, recorded by keeper Anselm Bray.")
QUESTION = ("\n\nBased on the text above, what is the magic passkey for the "
            "northern vault? Answer with just the passkey.\n")
FILLER_UNIT = ("Log entry {i}: the northern survey team catalogued ordinary "
               "mineral samples, weather readings, and supply inventories "
               "along the ridge; nothing of note was recorded that day.\n")


CHAT = True  # --raw: /v1/completions with no chat template (the model then rambles; not a retrieval test)


def post(url, prompt, max_tokens):
    req = {"max_tokens": max_tokens, "temperature": 0, "stream": False}
    if os.environ.get("TEST_MODEL", "GLM-5.3-NVFP4"):  # TEST_MODEL="": omit the name (servers that check it)
        req["model"] = os.environ.get("TEST_MODEL", "GLM-5.3-NVFP4")
    if CHAT:
        req["messages"] = [{"role": "user", "content": prompt}]
        req["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        req["prompt"] = prompt
    body = json.dumps(req).encode()
    path = "/v1/chat/completions" if CHAT else "/v1/completions"
    req = urllib.request.Request(url + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def reply_text(reply):
    try:
        c = reply["choices"][0]
        return (c.get("message") or {}).get("content") or c.get("text") or ""
    except (KeyError, IndexError):
        return ""


def prompt_tokens(reply):
    return reply.get("usage", {}).get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200000)
    ap.add_argument("--depth", type=float, default=0.85)
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--raw", action="store_true", help="raw /v1/completions, no chat template")
    args = ap.parse_args()
    global CHAT
    CHAT = not args.raw

    # Calibration: one block of 512 filler lines -> server-reported token count.
    probe_fill = "".join(FILLER_UNIT.format(i=i) for i in range(512))
    probe = post(args.url, probe_fill + QUESTION, 1)
    pt = prompt_tokens(probe)
    per_line = (pt - 30) / 512  # QUESTION is ~30 tokens
    if per_line <= 0:
        print("calibration failed:", json.dumps(probe)[:300])
        return 2
    print(f"calibration: {pt} tokens / 512 lines = {per_line:.3f} tok/line",
          flush=True)

    lines = int((args.target - 80) / per_line)
    needle_at = int(lines * args.depth)
    parts = []
    for i in range(lines):
        if i == needle_at:
            parts.append(NEEDLE + "\n")
        parts.append(FILLER_UNIT.format(i=i))
    prompt = "".join(parts) + QUESTION

    reply = post(args.url, prompt, args.max_tokens)
    pt = prompt_tokens(reply)
    text = reply_text(reply)
    ok = NEEDLE_KEY in text
    print(f"prompt_tokens={pt} needle@{args.depth:.2f} (line {needle_at}/{lines})")
    print(f"reply: {text.strip()[:200]}")
    print("NEEDLE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
