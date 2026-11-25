#!/usr/bin/env python3
"""DSpark trajectories for the NumPy twin (dspark_twin.py): a server booted with TF_GLM_TAP_DUMP=<file> (engine
commit "DSpark tap capture", drafts off) appends each prompt chunk's taps there. For each prompt: a greedy reply of
--tokens tokens ("draft": false), then prompt + reply sent as a prompt (max_tokens 1) so every position's taps are
prefilled; the new records of the dump file give taps for positions 0 .. L - 1 (a reused prefix keeps the first
request's rows). Saved per prompt as OUT/NN.npz: ids [L] i32, n_prompt, taps [L][5][hidden] (bf16 bits, u16).

usage: dspark_traj.py PARITY.json DUMP_FILE OUT_DIR [--url URL] [--tokens 256] [--long PANEL8K.json:N:LEN]
"""
import json
import os
import struct
import sys
import urllib.request

import numpy as np


def post(url, body):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def records(path, off):
    """(start, ids, taps) records of the dump from byte offset `off`; returns them and the new offset."""
    out = []
    with open(path, "rb") as f:
        f.seek(off)
        while True:
            head = f.read(20)
            if len(head) < 20:
                break
            magic, start, rows, hidden = head[:4], *struct.unpack("<qii", head[4:])
            if magic != b"TAPS":
                raise SystemExit(f"bad record at {off}")
            ids = np.frombuffer(f.read(4 * rows), dtype=np.int32)
            taps = np.frombuffer(f.read(rows * 5 * hidden * 2), dtype=np.uint16).reshape(rows, 5, hidden)
            out.append((start, ids, taps))
            off = f.tell()
    return out, off


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--long", default="", help="PANEL8K.json:N:LEN: N extra prompts, LEN tokens of panel windows")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    prompts = [(p["kind"], list(p["ids"])) for p in json.load(open(a.panel))["prompts"]]
    if a.long:
        path, n, ln = a.long.split(":")
        pan = json.load(open(path))
        wins = pan["windows"] if isinstance(pan, dict) and "windows" in pan else pan
        seen = set()
        for w in wins:            # the first window of each domain, up to N
            dom = w.get("domain", "long")
            if dom in seen or len(seen) >= int(n):
                continue
            seen.add(dom)
            prompts.append((dom + "-long", list(w["ids"][:int(ln)])))
    off = 0
    index = []
    for i, (kind, ids) in enumerate(prompts):
        r = post(a.url, {"model": model, "prompt": ids, "max_tokens": a.tokens, "temperature": 0, "draft": False,
                         "return_token_ids": True})
        reply = r["tensorfold"]["token_ids"]
        post(a.url, {"model": model, "prompt": ids + reply, "max_tokens": 1, "temperature": 0, "draft": False})
        recs, off = records(a.dump, off)
        seq = ids + reply
        L = len(seq)
        hidden = recs[0][2].shape[2] if recs else 0
        taps = np.zeros((L, 5, hidden), dtype=np.uint16)
        have = np.zeros(L, dtype=bool)
        for start, rid, t in recs:   # later records overwrite (the second request's rows are its own pass)
            n = min(len(rid), L - start)
            if n <= 0:
                continue
            if not np.array_equal(rid[:n], np.asarray(seq[start:start + n], dtype=np.int32)):
                if start == 0:
                    continue          # a record of another sequence (not ours): skip
                raise SystemExit(f"prompt {i}: record at {start} has other ids")
            taps[start:start + n] = t[:n]
            have[start:start + n] = True
        missing = int((~have).sum())
        np.savez(os.path.join(a.out, f"{i:02d}.npz"), ids=np.asarray(seq, dtype=np.int32), n_prompt=len(ids), taps=taps)
        index.append({"i": i, "kind": kind, "n_prompt": len(ids), "n_reply": len(reply), "missing": missing,
                      "sha256": r["tensorfold"].get("token_sha256")})
        print(f"{i:02d} {kind:12s} prompt {len(ids):5d} reply {len(reply):4d} records {len(recs)} missing {missing}",
              flush=True)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    bad = sum(1 for x in index if x["missing"])
    print(f"TRAJ {'PASS' if bad == 0 else 'FAIL'}: {len(index)} trajectories, {bad} with missing taps")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
