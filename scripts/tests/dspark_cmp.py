#!/usr/bin/env python3
"""G5: the live DSpark drafter (engine rounds logged with TF_GLM_DFLASH2_LOG=n) against the NumPy twin.

usage: dspark_cmp.py points ROUNDS.log TRAJ_DIR POINTS.json   (rounds -> twin points; prompts in parity order)
       dspark_cmp.py report POINTS_OUT.json                    (agreement after `dspark_twin.py points`)
A round's request boundaries are where its position falls (draft_parity.py sends the prompts in order, drafted
replies only log rounds). Pass: slot-0 drafts equal on >= 97% of rounds (4-bit drafter, Mythos 4.8), and the mean
confidence difference small (< 0.05).
"""
import json
import re
import sys

import numpy as np

PAT = re.compile(r"dspark round (\d+): slot (\d+) pos (\d+) kept (\d+) anchor (\d+) drafts \{ ([^}]*) \} chances \{ ([^}]*) \}")


def main():
    if sys.argv[1] == "points":
        log, traj, out = sys.argv[2:5]
        idx = json.load(open(f"{traj}/index.json"))
        seqs = [np.load(f"{traj}/{x['i']:02d}.npz")["ids"] for x in idx]
        groups, cur, last = [], [], -1
        for line in open(log, errors="replace"):
            m = PAT.search(line)
            if not m:
                continue
            pos = int(m.group(3))
            if pos < last and cur:
                groups.append(cur)
                cur = []
            last = pos
            cur.append((pos, int(m.group(5)), [int(x) for x in m.group(6).split(",")], [float(x) for x in m.group(7).split(",")]))
        if cur:
            groups.append(cur)
        pts = []
        for g in groups:   # the trajectory whose tokens agree with the group's anchors (a group of another request: none)
            best, hits = None, 0
            for t, ids in enumerate(seqs):
                h = sum(1 for pos, anc, _, _ in g if pos < len(ids) and int(ids[pos]) == anc)
                if h > hits:
                    best, hits = t, h
            if best is None or hits < 0.9 * len(g):
                print(f"group of {len(g)} rounds from pos {g[0][0]}: no trajectory ({hits} anchors agree at best)")
                continue
            ids = seqs[best]
            for pos, anc, d, c in g:
                if pos < len(ids) and int(ids[pos]) == anc:
                    pts.append({"traj": idx[best]["i"], "p": pos, "anchor": anc, "drafts": d, "conf": c})
            print(f"group of {len(g)} rounds -> trajectory {idx[best]['i']} ({hits} anchors agree)")
        json.dump(pts, open(out, "w"))
        print(f"{len(pts)} rounds")
        return
    pts = json.load(open(sys.argv[2]))
    n = len(pts)
    s0 = sum(p["drafts"][0] == p["twin_drafts"][0] for p in pts) / max(n, 1)
    full = sum(p["drafts"] == p["twin_drafts"] for p in pts) / max(n, 1)
    pref = []
    for p in pts:
        k = 0
        while k < len(p["drafts"]) and p["drafts"][k] == p["twin_drafts"][k]:
            k += 1
        pref.append(k)
    dc = float(np.mean([abs(a - b) for p in pts for a, b in zip(p["conf"][:1], p["twin_conf"][:1])])) if n else 0
    ok = s0 >= 0.97 and dc < 0.05
    print(f"rounds {n}: slot-0 drafts equal {100 * s0:.1f}%, whole chain equal {100 * full:.1f}%, mean common prefix "
          f"{np.mean(pref):.2f} of 7, mean |conf0 diff| {dc:.4f}")
    print(f"GATE G5 {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
