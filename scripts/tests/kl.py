#!/usr/bin/env python3
"""Compare two nll.py --save dumps of the same texts: a reference (e.g. DENSE=q8) and a candidate (e.g. q6).

Per position: KL(ref || cand) over the union of both top-k sets (each side's mass outside its top-k as one "rest"
bucket, a missing token taking a share of it), top-1 agreement, and the prompt token's logprob change. A numerics
change passes when mean KL and top-1 flips stay near the reference's own run-to-run level (run the reference twice).

Usage: kl.py REF.json CAND.json
"""
import json
import math
import sys


def dist(top):
    p = {t: math.exp(v) for t, v in top.items()}
    return p, max(0.0, 1.0 - sum(p.values()))


def main():
    ref, cand = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    kls, flips, n, dlp = [], 0, 0, []
    for f in ref:
        for (tok_r, top_r), (tok_c, top_c) in zip(ref[f], cand.get(f, [])):
            if tok_r != tok_c or not top_r or not top_c:
                continue
            p, pr = dist(top_r)
            q, qr = dist(top_c)
            kl = 0.0
            for t in set(p) | set(q):
                pt, qt = p.get(t, pr / 1000), q.get(t, qr / 1000)
                if pt > 0 and qt > 0:
                    kl += pt * math.log(pt / qt)
            if pr > 1e-9 and qr > 1e-9:
                kl += pr * math.log(pr / qr)
            kls.append(max(0.0, kl))
            flips += max(top_r, key=top_r.get) != max(top_c, key=top_c.get)
            if tok_r in top_r and tok_r in top_c:
                dlp.append(top_c[tok_r] - top_r[tok_r])
            n += 1
    kls.sort()
    pct = lambda q: kls[min(len(kls) - 1, int(q * len(kls)))] if kls else 0.0
    print(f"positions {n}: mean KL {sum(kls) / max(n, 1):.5f}  p99 {pct(0.99):.4f}  max {pct(1.0):.4f}  "
          f"top-1 flips {flips} ({100.0 * flips / max(n, 1):.2f}%)  mean dlogp {sum(dlp) / max(len(dlp), 1):+.5f}")


if __name__ == "__main__":
    main()
