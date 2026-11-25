#!/usr/bin/env python3
"""Top-K fidelity against a reference, computed identically for servers that return only their top K (panel_topk.py)
and for full-row dumps (TF_GLM_DUMP_LOGITS) reduced to their own top K.

usage:
  score_topk.py compute PANEL.json REF_DIR SRC OUT.npz [--k 20] [--vocab V]
      SRC: a panel_topk.py file (it has SRC.json beside it) or a full-row bf16 dump (panel_kl.py's DUMP.bin).
  score_topk.py report PANEL.json BASE.npz CAND.npz [CAND.npz ...]
      the first is the baseline every other is paired against (paired t on window means, exact McNemar on flips).

Per position, with S the candidate's top-K ids, p the reference and q the candidate:
  partition KL = sum_{i in S} p_i log(p_i / q_i) + p_rest log(p_rest / q_rest),  p_rest = 1 - sum_S p, q_rest = 1 - sum_S q
  (the KL between the two distributions coarsened to S + one "other" bucket: a lower bound of the full KL that a
  top-K server can be scored on); top-1 agreement with the reference; flips where the reference's top is above 0.9.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_kl import logsoftmax_bf16, mcnemar_worse_p, paired_ci  # noqa: E402

CHUNK = 1024
EPS = 1e-12


def topk_rows(src, k, rows, vocab, windows):
    """Yield per window: (ids [rows, k] int64, lps [rows, k] float32)."""
    if os.path.exists(src + ".json"):
        meta = json.load(open(src + ".json"))
        kk = meta["k"]
        assert kk >= k and meta["rows"] == rows, (meta, k, rows)
        dt = np.dtype([("ids", "<i4", (kk,)), ("lps", "<f4", (kk,)), ("act", "<f4"), ("rank", "<i4")])
        rec = np.memmap(src, dtype=dt, mode="r")
        assert len(rec) >= rows * windows, (len(rec), rows * windows)
        for w in range(windows):
            r = rec[w * rows:(w + 1) * rows]
            yield np.asarray(r["ids"][:, :k], dtype=np.int64), np.asarray(r["lps"][:, :k], dtype=np.float32)
        return
    f = open(src, "rb")
    for w in range(windows):
        ids = np.empty((rows, k), dtype=np.int64)
        lps = np.empty((rows, k), dtype=np.float32)
        for r0 in range(0, rows, CHUNK):
            n = min(CHUNK, rows - r0)
            q = logsoftmax_bf16(f.read(n * vocab * 2), n, vocab)
            part = np.argpartition(-q, k, axis=1)[:, :k]
            v = np.take_along_axis(q, part, axis=1)
            o = np.argsort(-v, axis=1)
            ids[r0:r0 + n] = np.take_along_axis(part, o, axis=1)
            lps[r0:r0 + n] = np.take_along_axis(v, o, axis=1)
        yield ids, lps


def compute(panel_p, ref_dir, src, out, k=20, vocab=0):
    panel = json.load(open(panel_p))
    meta = json.load(open(os.path.join(ref_dir, "meta.json")))
    assert meta["panel_sha256"] == panel["sha256"], "reference was made on another panel"
    vocab = vocab or meta["vocab"]
    rows = panel["tokens_per_window"] - 1
    W = min(len(panel["windows"]), meta["windows"], int(os.environ.get("SCORE_WINDOWS", 10 ** 6)))  # a quick check
    kl_w, top1_w, flips, confs = [], [], [], []
    for w, (ids, lps) in enumerate(topk_rows(src, k, rows, vocab, W)):
        ref = np.memmap(os.path.join(ref_dir, f"ref-w{w:02d}.f16"), dtype=np.float16, mode="r", shape=(rows, vocab))
        kl = np.empty(rows, dtype=np.float64)
        agree = np.empty(rows, dtype=bool)
        conf = np.empty(rows, dtype=bool)
        for r0 in range(0, rows, CHUNK):
            n = min(CHUNK, rows - r0)
            rr = np.asarray(ref[r0:r0 + n], dtype=np.float32)
            top = rr.argmax(axis=1)
            pmax = np.exp(rr[np.arange(n), top])
            idn = ids[r0:r0 + n]
            valid = idn >= 0
            lp_ref = np.take_along_axis(rr, np.where(valid, idn, 0), axis=1).astype(np.float64)
            lq = lps[r0:r0 + n].astype(np.float64)
            p = np.where(valid, np.exp(lp_ref), 0.0)
            q = np.where(valid, np.exp(lq), 0.0)
            term = np.where(valid & (p > 0), p * (lp_ref - lq), 0.0).sum(axis=1)
            pr = np.clip(1.0 - p.sum(axis=1), EPS, 1.0)
            qr = np.clip(1.0 - q.sum(axis=1), EPS, 1.0)
            kl[r0:r0 + n] = term + pr * (np.log(pr) - np.log(qr))
            agree[r0:r0 + n] = idn[:, 0] == top
            conf[r0:r0 + n] = pmax > 0.9
        kl_w.append(kl.mean())
        top1_w.append(agree.mean())
        flips.append(~agree & conf)
        confs.append(conf)
        print(f"w{w:02d} {panel['windows'][w]['domain']:5s} top{k} KL {kl_w[-1]:.5f} top-1 {100 * top1_w[-1]:.1f}%", flush=True)
    np.savez_compressed(out, kl=np.array(kl_w), top1=np.array(top1_w), flip=np.stack(flips), conf=np.stack(confs),
                        k=k, panel_sha256=panel["sha256"], src=os.path.basename(src))
    print("saved", out)


def report(panel_p, files):
    panel = json.load(open(panel_p))
    names = [os.path.basename(f).rsplit(".", 1)[0] for f in files]
    D = [np.load(f) for f in files]
    W = min(len(d["kl"]) for d in D)
    wins = panel["windows"][:W]
    k = int(D[0]["k"])
    print(f"per candidate: top-{k} partition KL vs reference, mean [paired-t 95% on window means] | top-1 | flips where ref > 0.9")
    for dom in ["all"] + sorted({w["domain"] for w in wins}):
        idx = np.array([i for i in range(W) if dom == "all" or wins[i]["domain"] == dom])
        conf = int(D[0]["conf"][idx].sum())
        print(f"\n[{dom}] {len(idx)} windows, {conf} confident positions")
        for n, d in zip(names, D):
            kv = d["kl"][idx]
            _, lo, hi = paired_ci(kv)
            fl = int(d["flip"][idx].sum())
            print(f"  {n:10s} topKL {kv.mean():.5f} [{lo:.5f}, {hi:.5f}] | top-1 {100 * d['top1'][idx].mean():.2f}%"
                  f" | conf flips {fl} ({100 * fl / max(conf, 1):.2f}%)")
        b0, d0 = names[0], D[0]
        for n, d in zip(names[1:], D[1:]):
            diff = d["kl"][idx] - d0["kl"][idx]
            m, lo, hi = paired_ci(diff)
            base = d0["kl"][idx].mean()
            dt = d["top1"][idx] - d0["top1"][idx]
            tm, tlo, thi = paired_ci(dt)
            fn, fb = d["flip"][idx], d0["flip"][idx]
            b, c = int((fn & ~fb).sum()), int((fb & ~fn).sum())
            print(f"  {n} - {b0}: topkl {m:+.5f} [{lo:+.5f}, {hi:+.5f}] ({100 * hi / max(base, 1e-12):+.1f}% upper);"
                  f" top-1 {100 * tm:+.2f} pts [{100 * tlo:+.2f}, {100 * thi:+.2f}];"
                  f" conf flips only-{n} {b} only-{b0} {c} one-sided p (candidate worse) {mcnemar_worse_p(b, c):.3f}")


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    opt = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    if a and a[0] == "compute" and len(a) == 5:
        compute(a[1], a[2], a[3], a[4], k=int(opt.get("k", 20)), vocab=int(opt.get("vocab", 0)))
    elif a and a[0] == "report" and len(a) >= 3:
        report(a[1], a[2:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
