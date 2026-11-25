#!/usr/bin/env python3
"""Full-vocabulary fidelity of served models against a reference on a frozen panel (panel.py + TF_GLM_DUMP_LOGITS).

usage: panel_kl.py PANEL.json REF_DIR NAME=DUMP.bin [NAME=DUMP.bin ...] [--vocab=V]
  REF_DIR: ref-wNN.f16 files ([rows][vocab] fp16 log-softmax, ref_run.py) and meta.json. DUMP.bin: a candidate's bf16
  logit rows, every window's in panel order. The first candidate is the baseline the others are paired against.

Per domain and pooled, for each candidate:
  KL(ref || cand) mean / median / p99 in nats; KL(cand || ref) mean (a build that puts mass where the reference puts
  almost none: the sampled-garbage direction); the candidate's mass outside the reference's 0.95 nucleus; top-1
  agreement; top-1 flips where the reference holds its top token above 0.9.
Against the baseline, paired by window: the mean difference with a paired t 95% interval on window means (the
  percentile bootstrap over 12 windows runs ~15% narrow), relative to the baseline; top-1 difference; and an exact
  one-sided McNemar test on confident flips (positions where only one of the two flips; small p: the candidate flips significantly more).
"""
import json
import math
import os
import sys

import numpy as np


def logsoftmax_bf16(raw, rows, vocab):
    x = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, vocab)
    m = x.max(axis=1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


def t975(df):
    """Student t 0.975 quantile (Cornish-Fisher in 1/df; within ~0.3% from df 5)."""
    z = 1.959963985
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2) + \
        (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3)


def paired_ci(diff):
    n = len(diff)
    if n < 2:
        return float(diff.mean()), float("nan"), float("nan")
    h = t975(n - 1) * diff.std(ddof=1) / math.sqrt(n)
    return float(diff.mean()), float(diff.mean() - h), float(diff.mean() + h)


def mcnemar_worse_p(b, c):
    """One-sided exact p that the candidate is not worse: P(X >= b) for X ~ Bin(b + c, 1/2); b = only the candidate flips."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def nucleus_mask(p, mass=0.95, k=4096):
    """Rows' smallest top sets holding `mass` of p (top-k by partition, a full sort only where k is not enough)."""
    rows, vocab = p.shape
    idx = np.argpartition(-p, k, axis=1)[:, :k]
    top = np.take_along_axis(p, idx, axis=1)
    order = np.argsort(-top, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    top = np.take_along_axis(top, order, axis=1)
    cum = np.cumsum(top, axis=1)
    keep = (cum - top) < mass  # an index stays while the mass before it is under `mass`
    mask = np.zeros_like(p, dtype=bool)
    np.put_along_axis(mask, idx, keep, axis=1)
    short = cum[:, -1] < mass
    for r in np.nonzero(short)[0]:  # rare: a flat row whose top-k holds under `mass`
        o = np.argsort(-p[r])
        cr = np.cumsum(p[r, o])
        mask[r, o[: int(np.searchsorted(cr, mass)) + 1]] = True
    return mask


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    vocab = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--vocab=")), 0))
    panel = json.load(open(args[0]))
    ref_dir = args[1]
    cands = [x.split("=", 1) for x in args[2:]]
    names = [n for n, _ in cands]
    wins = panel["windows"]
    rows = panel["tokens_per_window"] - 1
    meta = json.load(open(os.path.join(ref_dir, "meta.json")))
    assert meta["panel_sha256"] == panel["sha256"], "reference was made on another panel"
    vocab = vocab or meta["vocab"]
    W = min(len(wins), meta["windows"])
    keys = ("kl", "rkl", "off", "top1")
    per = {n: {k: [] for k in keys} | {"kl_all": [], "flip": []} for n in names}
    conf_all = []
    files = {n: open(p, "rb") for n, p in cands}
    for w in range(W):
        ref = np.fromfile(os.path.join(ref_dir, f"ref-w{w:02d}.f16"), dtype=np.float16).reshape(rows, vocab).astype(np.float32)
        p = np.exp(ref)
        ref_top = ref.argmax(axis=1)
        confident = p.max(axis=1) > 0.9
        conf_all.append(confident)
        nuc = nucleus_mask(p)
        for n in names:
            q = logsoftmax_bf16(files[n].read(rows * vocab * 2), rows, vocab)
            eq = np.exp(q)
            kl = (p * (ref - q)).sum(axis=1)
            rkl = (eq * (q - ref)).sum(axis=1)
            off = (eq * ~nuc).sum(axis=1)
            agree = q.argmax(axis=1) == ref_top
            for k, v in (("kl", kl.mean()), ("rkl", rkl.mean()), ("off", off.mean()), ("top1", agree.mean())):
                per[n][k].append(float(v))
            per[n]["kl_all"].append(kl)
            per[n]["flip"].append(~agree & confident)
        print(f"w{w:02d} {wins[w]['domain']:5s} " + "  ".join(
            f"{n} KL {per[n]['kl'][-1]:.4f} rKL {per[n]['rkl'][-1]:.4f} top1 {100 * per[n]['top1'][-1]:.1f}%" for n in names), flush=True)
    doms = ["all"] + sorted({w["domain"] for w in wins[:W]})
    print("\nper candidate: KL(ref||cand) mean [paired-t 95% on window means] / median / p99 | KL(cand||ref) | mass outside"
          " ref's 0.95 nucleus | top-1 | flips where ref > 0.9")
    for d in doms:
        idx = np.array([i for i in range(W) if d == "all" or wins[i]["domain"] == d])
        conf = int(np.concatenate([conf_all[i] for i in idx]).sum())
        print(f"\n[{d}] {len(idx)} windows, {conf} confident positions")
        for n in names:
            k = np.array(per[n]["kl"])[idx]
            _, lo, hi = paired_ci(k)
            allk = np.concatenate([per[n]["kl_all"][i] for i in idx])
            flips = int(np.concatenate([per[n]["flip"][i] for i in idx]).sum())
            print(f"  {n:10s} KL {k.mean():.5f} [{lo:.5f}, {hi:.5f}] / {np.median(allk):.5f} / {np.percentile(allk, 99):.4f}"
                  f" | rKL {np.mean(np.array(per[n]['rkl'])[idx]):.5f} | off {np.mean(np.array(per[n]['off'])[idx]):.5f}"
                  f" | top-1 {100 * np.mean(np.array(per[n]['top1'])[idx]):.2f}% | conf flips {flips} ({100 * flips / max(conf, 1):.2f}%)")
        for n in names[1:]:
            b0 = names[0]
            out = []
            for key in ("kl", "rkl", "off"):
                diff = np.array(per[n][key])[idx] - np.array(per[b0][key])[idx]
                m, lo, hi = paired_ci(diff)
                base = np.mean(np.array(per[b0][key])[idx])
                out.append(f"{key} {m:+.5f} [{lo:+.5f}, {hi:+.5f}] ({100 * hi / max(base, 1e-12):+.1f}% upper)")
            dt = np.array(per[n]["top1"])[idx] - np.array(per[b0]["top1"])[idx]
            m, lo, hi = paired_ci(dt)
            out.append(f"top-1 {100 * m:+.2f} pts [{100 * lo:+.2f}, {100 * hi:+.2f}]")
            fn = np.concatenate([per[n]["flip"][i] for i in idx])
            fb = np.concatenate([per[b0]["flip"][i] for i in idx])
            b, c = int((fn & ~fb).sum()), int((fb & ~fn).sum())
            out.append(f"conf flips only-{n} {b} only-{b0} {c} one-sided p (candidate worse) {mcnemar_worse_p(b, c):.3f}")
            print(f"  {n} - {b0}: " + "; ".join(out))


if __name__ == "__main__":
    main()
