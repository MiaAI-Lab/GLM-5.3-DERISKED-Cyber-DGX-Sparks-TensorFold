#!/usr/bin/env python3
"""NumPy twin of the DSpark drafter (RedHatAI/GLM-5.3-speculator.dspark) on real GLM-5.3 trajectories.

The reference for the Zig drafter and the predictor for the port's gain (Mythos plan 4.7). NumPy only (no torch),
written from the drafter's config.json and the formulas of the owner's 3x recipe patch 0114 (dspark.py,
MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold), which adapted them from vllm-project/speculators and vLLM:

* taps: the target residual at the INPUT of layers aux_hidden_state_layer_ids, captured by the engine after 0-based
  layers 1, 19, 38, 57, 74 (dspark_traj.py);
* context row j: c_j = hidden_norm(fc(concat taps_j)); per layer K_j = RoPE(k_norm(k_proj c_j), j), V_j = v_proj c_j;
* block: [anchor, mask x 7] at positions p .. p + 7 with the verifier's embeddings; Qwen3 pre-norm layers (q_norm /
  k_norm, then neox RoPE, theta 8e6); slot m sees context rows [p - 2048, p) and block slots 0..m;
* h = norm(x); base logits = verifier lm_head(h); slot k predicts the token at p + k + 1;
* Markov: score = base + W2[c] . W1[prev], prev_0 = anchor, prev_k = draft k - 1, over the top-K base candidates;
* confidence: sigmoid(w_h . h_k + w_m . W1[prev_k] + b).

usage:
  dspark_twin.py selftest DRAFTER_DIR
  dspark_twin.py walk DRAFTER_DIR TARGET_DIR TRAJ_DIR OUT.json [--perturb NAME[,NAME]] [--limit N] [--fixture F]
    perturbations: rope+1 rope-1 noncausal win2047 win2049 nomarkov prevshift anchormask top64
"""
import json
import math
import os
import struct
import sys
import time

import numpy as np

MASK_DEFAULT = 154856


# -- safetensors without torch ------------------------------------------------------------------------------------
class St:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.head = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        h = self.head[name]
        a, b = h["data_offsets"]
        return self.mm[self.base + a:self.base + b], h["dtype"], h["shape"]

    def f32(self, name, rows=None):
        buf, dt, shape = self.raw(name)
        if dt not in ("BF16", "F16", "F32"):
            raise SystemExit(f"{name}: dtype {dt}")
        if dt == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(shape)
        u = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        if rows is not None:
            u = u[rows]
        if dt == "F16":
            return u.view(np.float16).astype(np.float32)
        return bf16(u)

    def u16(self, name):
        buf, dt, shape = self.raw(name)
        return np.frombuffer(buf, dtype=np.uint16).reshape(shape)


def bf16(u):
    return (np.asarray(u, dtype=np.uint32) << 16).view(np.float32)


def round_bf16(x):
    """fp32 -> nearest-even bf16 -> fp32 (the engine's activations are bf16 between kernels)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return r.view(np.float32)


def rms(x, w, eps):
    v = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
    return x / np.sqrt(v + eps) * w


# -- the drafter ----------------------------------------------------------------------------------------------------
class Drafter:
    def __init__(self, ddir):
        cfg = json.load(open(os.path.join(ddir, "config.json")))
        t = cfg["transformer_layer_config"]
        self.cfg = cfg
        self.D = int(t["hidden_size"])
        self.H = int(t["num_attention_heads"])
        self.hd = int(t.get("head_dim") or self.D // self.H)
        self.L = int(t["num_hidden_layers"])
        self.eps = float(t["rms_norm_eps"])
        self.theta = float((t.get("rope_parameters") or {}).get("rope_theta", t.get("rope_theta", 10000.0)))
        self.window = int(t["sliding_window"])
        self.block = int(cfg["block_size"])
        self.mask = int(cfg.get("mask_token_id", MASK_DEFAULT))
        self.aux = [int(i) for i in cfg["aux_hidden_state_layer_ids"]]
        st = St(os.path.join(ddir, "model.safetensors"))
        g = st.f32
        self.fc = np.ascontiguousarray(g("fc.weight").T)                  # [5D, D]
        self.hidden_norm = g("hidden_norm.weight")
        self.norm = g("norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"layers.{i}."
            self.layers.append({
                "in": g(p + "input_layernorm.weight"), "post": g(p + "post_attention_layernorm.weight"),
                "q": np.ascontiguousarray(g(p + "self_attn.q_proj.weight").T),
                "k": np.ascontiguousarray(g(p + "self_attn.k_proj.weight").T),
                "v": np.ascontiguousarray(g(p + "self_attn.v_proj.weight").T),
                "o": np.ascontiguousarray(g(p + "self_attn.o_proj.weight").T),
                "qn": g(p + "self_attn.q_norm.weight"), "kn": g(p + "self_attn.k_norm.weight"),
                "gate": np.ascontiguousarray(g(p + "mlp.gate_proj.weight").T),
                "up": np.ascontiguousarray(g(p + "mlp.up_proj.weight").T),
                "down": np.ascontiguousarray(g(p + "mlp.down_proj.weight").T)})
        self.w1 = st.u16("markov_head.markov_w1.weight")                  # bf16 bits [V, r]
        self.w2 = st.u16("markov_head.markov_w2.weight")
        cw = g("confidence_head.proj.weight")[0].astype(np.float64)
        self.conf_h = cw[:self.D].astype(np.float32)
        self.conf_m = cw[self.D:]
        self.conf_b = float(g("confidence_head.proj.bias")[0])
        self.inv = (1.0 / self.theta ** (np.arange(self.hd // 2, dtype=np.float64) * 2 / self.hd)).astype(np.float64)

    def rope(self, x, pos):
        """neox RoPE on x [..., n, H, hd] at positions pos [n]."""
        a = np.asarray(pos, dtype=np.float64)[:, None] * self.inv[None, :]
        c = np.cos(a).astype(np.float32)[:, None, :]
        s = np.sin(a).astype(np.float32)[:, None, :]
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)

    def context(self, taps):
        """taps [n][5][D] bf16 bits -> per layer (K [n,H,hd], V [n,H,hd]) at positions 0..n-1."""
        n = taps.shape[0]
        x = bf16(taps.reshape(n, -1))
        c = round_bf16(rms(round_bf16(x @ self.fc), self.hidden_norm, self.eps))
        out = []
        pos = np.arange(n)
        for Ly in self.layers:
            k = round_bf16(c @ Ly["k"]).reshape(n, self.H, self.hd)
            k = self.rope(rms(k, Ly["kn"], self.eps), pos)
            v = round_bf16(c @ Ly["v"]).reshape(n, self.H, self.hd)
            out.append((round_bf16(k), v))
        return out

    def block_pass(self, ctx, emb_rows, p, opts):
        """One anchor at position p: emb_rows [B][D] (the anchor's and masks' embeddings) -> h [B][D] (normed)."""
        B = emb_rows.shape[0]
        W = opts.get("window", self.window)
        shift = opts.get("rope", 0)
        causal = opts.get("causal", True)
        x = emb_rows.astype(np.float32).copy()
        bpos = p + shift + np.arange(B)
        lo = max(0, p - W)
        scale = self.hd ** -0.5
        for i, Ly in enumerate(self.layers):
            hN = round_bf16(rms(x, Ly["in"], self.eps))
            q = round_bf16(hN @ Ly["q"]).reshape(B, self.H, self.hd)
            k = round_bf16(hN @ Ly["k"]).reshape(B, self.H, self.hd)
            v = round_bf16(hN @ Ly["v"]).reshape(B, self.H, self.hd)
            q = round_bf16(self.rope(rms(q, Ly["qn"], self.eps), bpos))
            k = round_bf16(self.rope(rms(k, Ly["kn"], self.eps), bpos))
            K = np.concatenate([ctx[i][0][lo:p], k], axis=0)              # [n_ctx + B, H, hd]
            V = np.concatenate([ctx[i][1][lo:p], v], axis=0)
            nctx = p - lo
            s = np.einsum("bhd,khd->hbk", q, K) * scale                    # [H, B, n]
            if causal:
                m = np.zeros((B, nctx + B), dtype=bool)
                for b in range(B):
                    m[b, nctx + b + 1:] = True
                s[:, m] = -np.inf
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            a = e / e.sum(-1, keepdims=True)
            o = round_bf16(np.einsum("hbk,khd->bhd", a, V).reshape(B, -1))
            x = round_bf16(x + round_bf16(o @ Ly["o"]))
            hN = round_bf16(rms(x, Ly["post"], self.eps))
            gt = hN @ Ly["gate"]
            act = round_bf16(gt / (1.0 + np.exp(-gt)) * (hN @ Ly["up"]))
            x = round_bf16(x + round_bf16(act @ Ly["down"]))
        return round_bf16(rms(x, self.norm, self.eps))

    def chain(self, cand, vals, hconf, anchor, before, opts):
        """Sequential Markov chain over each slot's candidates (cand [B][K] ids, vals [B][K] base logits)."""
        markov = opts.get("markov", True)
        prevshift = opts.get("prevshift", False)
        out, conf = [], []
        prev, prev2 = int(anchor), int(before)
        for d in range(cand.shape[0]):
            use = prev2 if prevshift else prev
            emb = bf16(self.w1[use]).astype(np.float64)
            bias = (bf16(self.w2[cand[d]]).astype(np.float64) @ emb) if markov else 0.0
            score = vals[d].astype(np.float64) + bias
            j = int(np.argmax(score))      # ties: the first (candidates are value-descending, id-ascending)
            z = float(hconf[d]) + self.conf_b + float(self.conf_m @ emb)
            conf.append(1.0 / (1.0 + math.exp(-z)))
            prev2, prev = prev, int(cand[d, j])
            out.append(prev)
        return out, conf


class Target:
    def __init__(self, tdir):
        idx = json.load(open(os.path.join(tdir, "model.safetensors.index.json")))["weight_map"]
        e = St(os.path.join(tdir, idx["model.embed_tokens.weight"]))
        h = St(os.path.join(tdir, idx["lm_head.weight"]))
        self.embed = e.u16("model.embed_tokens.weight")                   # bf16 bits [V, D]
        t0 = time.time()
        self.head = np.ascontiguousarray(h.f32("lm_head.weight").T)       # [D, V]
        print(f"target head loaded in {time.time() - t0:.0f}s", flush=True)

    def emb(self, ids):
        return bf16(self.embed[np.asarray(ids, dtype=np.int64)])


def topk(logits, k):
    """Each row's k best: values descending, ids ascending on ties."""
    idx = np.argpartition(-logits, k, axis=-1)[:, :k]
    v = np.take_along_axis(logits, idx, -1)
    order = np.lexsort((idx, -v), axis=-1)
    return np.take_along_axis(idx, order, -1), np.take_along_axis(v, order, -1)


def auc(scores, labels):
    s = np.asarray(scores)
    y = np.asarray(labels, dtype=bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def walk(args):
    dr = Drafter(args.drafter)
    tg = Target(args.target)
    opts = {}
    for name in [x for x in (args.perturb or "").split(",") if x]:
        if name in ("rope+1", "rope-1"):
            opts["rope"] = 1 if name == "rope+1" else -1
        elif name == "noncausal":
            opts["causal"] = False
        elif name.startswith("win"):
            opts["window"] = int(name[3:])
        elif name == "nomarkov":
            opts["markov"] = False
        elif name == "prevshift":
            opts["prevshift"] = True
        elif name == "anchormask":
            opts["anchormask"] = True
        elif name == "top64":
            opts["topk"] = 64
        else:
            raise SystemExit(f"unknown perturbation {name}")
    K = opts.get("topk", 16)
    index = json.load(open(os.path.join(args.traj, "index.json")))
    if args.limit:
        index = index[:args.limit]
    B = dr.block
    per = []
    fixture = []
    for it in index:
        z = np.load(os.path.join(args.traj, f"{it['i']:02d}.npz"))
        ids, n0, taps = z["ids"], int(z["n_prompt"]), z["taps"]
        L = len(ids)
        t0 = time.time()
        ctx = dr.context(taps)
        acc, slot_ok, confs, labels, cut16 = [], np.zeros(B), [], [], 0
        slots_seen = np.zeros(B)
        drafts_at = {}
        for p in range(n0, L - 1):
            anchor = int(ids[p])
            first = dr.mask if opts.get("anchormask") else anchor
            emb = tg.emb([first] + [dr.mask] * (B - 1))
            h = dr.block_pass(ctx, emb, p, opts)
            logits = h @ tg.head
            cand, vals = topk(logits, K)
            hconf = h.astype(np.float64) @ dr.conf_h.astype(np.float64)
            out, conf = dr.chain(cand, vals, hconf, anchor, int(ids[p - 1]) if p > 0 else anchor, opts)
            if K == 16 and args.compare64:
                c64, v64 = topk(logits, 64)
                o64, _ = dr.chain(c64, v64, hconf, anchor, int(ids[p - 1]) if p > 0 else anchor, opts)
                cut16 += sum(1 for a_, b_ in zip(out, o64) if a_ != b_)
            a = 0
            for k in range(B):
                if p + k + 1 >= L:
                    break
                slots_seen[k] += 1
                ok = out[k] == int(ids[p + k + 1])
                slot_ok[k] += ok
                if a == k:      # the chain's prefix still holds: its confidence is that draft's survival
                    confs.append(conf[k])
                    labels.append(ok)
                if ok and a == k:
                    a += 1
            acc.append(a)
            drafts_at[p] = (a, out, conf)
            if args.fixture and len(fixture) < 64 and p % 7 == 0:
                fixture.append({"traj": it["i"], "p": p, "anchor": anchor, "drafts": out, "conf": conf,
                                "cand": cand.tolist(), "vals": vals.tolist(), "hconf": hconf.tolist(),
                                "h_sum": h.astype(np.float64).sum(1).tolist()})
        # the engine's walk: a round at p offers 7 drafts, keeps the accepted prefix and the bonus token
        rounds, tok, p = 0, 0, n0
        while p < L - 1:
            a = min(drafts_at[p][0], B - 1)
            rounds += 1
            tok += a + 1
            p += a + 1
        rec = {"i": it["i"], "kind": it["kind"], "anchors": len(acc), "mean_accept_8": float(np.mean(acc)) if acc else 0,
               "slot_agree": (slot_ok / np.maximum(slots_seen, 1)).tolist(), "rounds": rounds, "tokens": tok,
               "tokens_per_round": tok / max(rounds, 1), "auc": auc(confs, labels),
               "conf": confs, "labels": [bool(x) for x in labels], "top16_vs_64_diff": cut16,
               "accept_hist": np.bincount(np.asarray(acc, dtype=int), minlength=B + 1).tolist(),
               "n_prompt": n0, "length": L,
               "anchors_detail": [[int(q), int(v[0]), [round(float(c), 5) for c in v[2]]] for q, v in sorted(drafts_at.items())]}
        per.append(rec)
        print(f"{it['i']:02d} {it['kind']:12s} anchors {len(acc):4d} accept/8 {rec['mean_accept_8']:.2f} "
              f"slot0 {rec['slot_agree'][0]:.3f} tok/round {rec['tokens_per_round']:.2f} auc {rec['auc']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    # summaries per kind and pooled
    summary = {}
    kinds = sorted(set(r["kind"] for r in per)) + ["all"]
    for kind in kinds:
        rs = [r for r in per if kind == "all" or r["kind"] == kind]
        n = sum(r["anchors"] for r in rs)
        conf = [c for r in rs for c in r["conf"]]
        lab = [l_ for r in rs for l_ in r["labels"]]
        slot = np.sum([np.asarray(r["slot_agree"]) * r["anchors"] for r in rs], axis=0) / max(n, 1)
        bins = []
        for lo in np.arange(0, 1, 0.1):
            sel = [l_ for c, l_ in zip(conf, lab) if lo <= c < lo + 0.1]
            if sel:
                bins.append([round(float(lo), 1), len(sel), float(np.mean(sel))])
        summary[kind] = {"anchors": n, "mean_accept_8": sum(r["mean_accept_8"] * r["anchors"] for r in rs) / max(n, 1),
                         "slot_agree": slot.tolist(), "tokens_per_round": sum(r["tokens"] for r in rs) / max(sum(r["rounds"] for r in rs), 1),
                         "auc": auc(conf, lab), "calibration": bins,
                         "top16_vs_64_diff_slots": sum(r["top16_vs_64_diff"] for r in rs)}
    for kind, s in summary.items():
        print(f"SUMMARY {kind:12s} anchors {s['anchors']:5d} accept/8 {s['mean_accept_8']:.3f} slot0 {s['slot_agree'][0]:.3f} "
              f"tok/round {s['tokens_per_round']:.3f} auc {s['auc']:.3f} slots {' '.join(f'{x:.2f}' for x in s['slot_agree'])}")
    json.dump({"perturb": args.perturb, "summary": summary,
               "per": [{k: v for k, v in r.items() if k not in ("conf", "labels")} for r in per]},
              open(args.out, "w"), indent=1)
    if args.fixture:
        json.dump(fixture, open(args.fixture, "w"))


def points(args):
    """The twin's drafts and confidences at given (trajectory, anchor position) points: [{"traj", "p"}] -> the same
    with "drafts", "conf" (for comparing the engine's logged rounds, G5)."""
    dr = Drafter(args.drafter)
    tg = Target(args.target)
    pts = json.load(open(args.points))
    B = dr.block
    cache = {}
    for pt in pts:
        i, p = int(pt["traj"]), int(pt["p"])
        if i not in cache:
            z = np.load(os.path.join(args.traj, f"{i:02d}.npz"))
            cache = {i: (z["ids"], dr.context(z["taps"]))}
        ids, ctx = cache[i]
        anchor = int(pt.get("anchor", ids[p]))
        emb = tg.emb([anchor] + [dr.mask] * (B - 1))
        h = dr.block_pass(ctx, emb, p, {})
        logits = h @ tg.head
        cand, vals = topk(logits, 16)
        hconf = h.astype(np.float64) @ dr.conf_h.astype(np.float64)
        out, conf = dr.chain(cand, vals, hconf, anchor, int(ids[p - 1]) if p > 0 else anchor, {})
        pt["twin_drafts"], pt["twin_conf"] = out[:B - 1], [round(c, 5) for c in conf[:B - 1]]
    json.dump(pts, open(args.out, "w"))


def selftest(args):
    """Synthetic checks of the twin's own mechanics (random small weights, no checkpoint reads past the config)."""
    dr = Drafter.__new__(Drafter)
    rng = np.random.default_rng(0)
    dr.D, dr.H, dr.hd, dr.L, dr.eps, dr.theta, dr.window, dr.block, dr.mask = 64, 4, 16, 2, 1e-5, 8e6, 16, 8, 0
    dr.inv = 1.0 / dr.theta ** (np.arange(dr.hd // 2) * 2 / dr.hd)
    def m(a, b):
        return (rng.standard_normal((a, b)) * 0.2).astype(np.float32)
    dr.fc = m(5 * dr.D, dr.D)
    dr.hidden_norm = dr.norm = np.ones(dr.D, np.float32)
    dr.layers = [{"in": np.ones(dr.D, np.float32), "post": np.ones(dr.D, np.float32), "q": m(dr.D, dr.D), "k": m(dr.D, dr.D),
                  "v": m(dr.D, dr.D), "o": m(dr.D, dr.D), "qn": np.ones(dr.hd, np.float32), "kn": np.ones(dr.hd, np.float32),
                  "gate": m(dr.D, 128), "up": m(dr.D, 128), "down": m(128, dr.D)} for _ in range(dr.L)]
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'ok  ' if cond else 'FAIL'} {name}")
    # RoPE: q.k depends on the position difference only
    q = rng.standard_normal((1, 1, dr.hd)).astype(np.float32)
    k = rng.standard_normal((1, 1, dr.hd)).astype(np.float32)
    d1 = (dr.rope(q, [10]) * dr.rope(k, [7])).sum()
    d2 = (dr.rope(q, [110]) * dr.rope(k, [107])).sum()
    check("rope: score depends on the position difference only", abs(d1 - d2) < 1e-3)
    taps = rng.integers(0x3c00, 0x3f80, size=(40, 5, dr.D)).astype(np.uint16)
    ctx = dr.context(taps)
    emb = rng.standard_normal((dr.block, dr.D)).astype(np.float32)
    h = dr.block_pass(ctx, emb, 30, {})
    emb2 = emb.copy()
    emb2[5] += 1.0
    h2 = dr.block_pass(ctx, emb2, 30, {})
    check("causal: a later slot cannot change an earlier one", np.array_equal(h[:5], h2[:5]) and not np.array_equal(h[5:], h2[5:]))
    ctx2 = [(k_.copy(), v_.copy()) for k_, v_ in ctx]
    for k_, v_ in ctx2:
        k_[30 - dr.window - 1] += 5.0
        v_[30 - dr.window - 1] += 5.0
    h3 = dr.block_pass(ctx2, emb, 30, {})
    check("window: a context row at p - W - 1 has no effect", np.array_equal(h, h3))
    for k_, v_ in ctx2:
        k_[30 - dr.window] += 5.0
    h4 = dr.block_pass(ctx2, emb, 30, {})
    check("window: the row at p - W is seen", not np.array_equal(h, h4))
    # Markov alignment: prev_0 = anchor, prev_k = draft k - 1
    V, r = 50, 4
    dr.w1 = np.zeros((V, r), np.uint16)
    dr.w2 = np.zeros((V, r), np.uint16)
    one = np.uint16(0x3f80)
    dr.w1[7, 0] = one          # after token 7, candidate 9 gets +big
    dr.w2[9, 0] = np.uint16(0x4120)   # 10.0
    dr.w1[9, 1] = one          # after token 9, candidate 3 gets +big
    dr.w2[3, 1] = np.uint16(0x4120)
    dr.conf_m = np.zeros(r)
    dr.conf_b = 0.0
    cand = np.array([[1, 9, 3], [1, 9, 3]])
    vals = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], np.float32)
    out, _ = dr.chain(cand, vals, np.zeros(2), 7, 0, {})
    check("markov: prev_0 = anchor, prev_1 = draft 0", out == [9, 3])
    out, _ = dr.chain(cand, vals, np.zeros(2), 7, 0, {"markov": False})
    check("markov off: base argmax", out == [1, 1])
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("drafter", nargs="?")
    w = sub.add_parser("walk")
    w.add_argument("drafter")
    w.add_argument("target")
    w.add_argument("traj")
    w.add_argument("out")
    w.add_argument("--perturb", default="")
    w.add_argument("--limit", type=int, default=0)
    w.add_argument("--fixture", default="")
    w.add_argument("--compare64", action="store_true")
    pp = sub.add_parser("points")
    pp.add_argument("drafter")
    pp.add_argument("target")
    pp.add_argument("traj")
    pp.add_argument("points")
    pp.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "points":
        points(a)
    elif a.cmd == "selftest":
        selftest(a)
    else:
        walk(a)


if __name__ == "__main__":
    main()
