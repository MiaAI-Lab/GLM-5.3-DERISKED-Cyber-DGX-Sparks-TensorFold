#!/usr/bin/env python3
"""G3: DSpark's predicted decode speed against measured MTP, from the twin's walk and draft_parity.py's MTP replies.

usage: g3.py TWIN.json MTP_PARITY.json [--d 1.5,3,6] [--out G3.json]
Costs: per MTP reply, decode seconds = alpha * rounds + beta * (rounds + drafted) (least squares; alpha holds the
verify's fixed cost and the MTP chain), serial seconds a token s = beta + a0 from the "draft": false replies. A DSpark
round offering k drafts costs a0 + beta * (1 + k) + d (d: the drafter's pass, ms). Policies over the twin's per-anchor
accepted prefixes and confidences: fixed depth k = 1..7; product cut c; expected-gain (the depth maximizing
(1 + sum_i prod_{j<=i} c_j) / cost(k)). Per kind (code / prose from parity kinds), DSpark tok/s against MTP's.
"""
import json
import sys

import numpy as np


def kind_group(k):
    k = k.lower()
    if "code" in k:
        return "code"
    return "prose"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("mtp")
    ap.add_argument("--d", default="1.5,3,6")
    ap.add_argument("--out", default="")
    ap.add_argument("--mtp-step", type=float, default=0.0,
                    help="ms of one MTP draft step inside beta (a DSpark row then costs beta - this)")
    a = ap.parse_args()
    tw = json.load(open(a.twin))
    mp = json.load(open(a.mtp))
    rs = [r for r in mp["results"] if not r["kind"].endswith("-sampled") and not r["kind"].startswith("copy")]
    X, y = [], []
    for r in rs:
        on = r["on"]
        X.append([on["rounds"], on["rounds"] + on["drafted"]])
        y.append(on["wall"] - on["ttft"])
    X, y = np.array(X, float), np.array(y, float)
    (alpha, beta), *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ np.array([alpha, beta])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n_off = sum(r["off"]["n"] for r in rs)
    t_off = sum(r["off"]["wall"] - r["off"]["ttft"] for r in rs)
    s = t_off / n_off
    a0 = s - beta
    mtp = {}
    for g in ("code", "prose"):
        sel = [r for r in rs if kind_group(r["kind"]) == g]
        if sel:
            n = sum(r["on"]["n"] for r in sel)
            t = sum(r["on"]["wall"] - r["on"]["ttft"] for r in sel)
            n2 = sum(r["off"]["n"] for r in sel)
            t2 = sum(r["off"]["wall"] - r["off"]["ttft"] for r in sel)
            mtp[g] = {"tps": n / t, "serial_tps": n2 / t2,
                      "tokens_per_round": n / max(1, sum(r["on"]["rounds"] for r in sel))}
    print(f"cost fit: alpha {alpha * 1e3:.2f} ms/round, beta {beta * 1e3:.2f} ms/row, r2 {r2:.3f}; serial {s * 1e3:.2f} ms/token "
          f"-> a0 {a0 * 1e3:.2f} ms")
    for g, v in mtp.items():
        print(f"MTP {g:6s} {v['tps']:.2f} tok/s (serial {v['serial_tps']:.2f}), tokens/round {v['tokens_per_round']:.2f}")
    out = {"alpha_ms": alpha * 1e3, "beta_ms": beta * 1e3, "r2": r2, "serial_ms": s * 1e3, "a0_ms": a0 * 1e3, "mtp": mtp, "dspark": {}}
    for d_ms in [float(x) for x in a.d.split(",")]:
        d = d_ms / 1e3
        bv = beta - a.mtp_step / 1e3
        cost = lambda k: a0 + a.mtp_step / 1e3 + bv * (1 + k) + d   # a serial token is a0 + beta = (a0 + m) + bv
        for g in ("code", "prose"):
            trajs = [p for p in tw["per"] if kind_group(p["kind"]) == g]
            if not trajs or g not in mtp:
                continue
            res = {}
            policies = [("fixed", k) for k in range(1, 8)] + [("cut", c) for c in (0.1, 0.2, 0.3, 0.4, 0.5)] + [("gain", 0)]
            for pol, par in policies:
                tok, tim = 0, 0.0
                for p in trajs:
                    det = {q: (acc, conf) for q, acc, conf in p["anchors_detail"]}
                    q = p["n_prompt"]
                    L = p["length"]
                    while q < L - 1 and q in det:
                        acc, conf = det[q]
                        if pol == "fixed":
                            k = par
                        elif pol == "cut":
                            k, alive = 0, 1.0
                            for i in range(7):
                                alive *= conf[i]
                                if i > 0 and alive < par:
                                    break
                                k = i + 1
                        else:
                            best, k, alive, exp = -1, 1, 1.0, 1.0
                            for i in range(7):
                                alive *= conf[i]
                                exp += alive
                                v = exp / cost(i + 1)
                                if v > best:
                                    best, k = v, i + 1
                        a_ = min(acc, k)
                        tok += a_ + 1
                        tim += cost(k)
                        q += a_ + 1
                res[f"{pol}{par if pol != 'gain' else ''}"] = tok / tim
            best = max(res, key=res.get)
            ratio = res[best] / mtp[g]["tps"]
            out["dspark"].setdefault(str(d_ms), {})[g] = {"policies": res, "best": best, "tps": res[best], "vs_mtp": ratio}
            print(f"d={d_ms:4.1f} ms {g:6s} DSpark best {best:8s} {res[best]:.2f} tok/s = {ratio:.3f}x MTP "
                  f"(fixed7 {res['fixed7']:.2f}, gain {res['gain']:.2f})")
    d3 = out["dspark"].get("3.0", {})
    go = d3.get("code", {}).get("vs_mtp", 0) >= 1.05 and d3.get("prose", {}).get("vs_mtp", 0) >= 0.98
    out["go"] = bool(go)
    print(f"GATE G3 {'GO' if go else 'NO-GO'} (d = 3 ms: code {d3.get('code', {}).get('vs_mtp', 0):.3f}x >= 1.05, "
          f"prose {d3.get('prose', {}).get('vs_mtp', 0):.3f}x >= 0.98)")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
