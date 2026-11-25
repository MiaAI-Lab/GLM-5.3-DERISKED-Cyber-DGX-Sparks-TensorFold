#!/usr/bin/env python3
"""X8 part 3: kindlingai's EXLR8 release (glm-5.3-exlr8-k2-k3-k4, exlr8-v2, CC BY 4.0) as the rest of a checkpoint the
TensorFold CUDA GLM engine reads beside the routed-expert rank pieces (exlr8_pieces.py), for one rank.

Each dense matrix (core/exlr8_dense.zig's records), by format:
  mxfp8 (q_a, kv_a, indexer wq_b / wk)        -> bf16, exact (e4m3 times a power of two; checked on every value)
  fp8_block (kv_b)                            -> weight F8_E4M3 + weight_scale_inv F32 [out/128, in/128], as stored (exact)
  bf16, fp32 (router, indexer weights_proj)   -> as stored (exact)
  trellis (q_b, o_proj, layers 0-2 MLP)       -> bf16 of the exact f64 value W = diag(suh) H T H diag(svh) / 128
  int4 g32 (q_b layers 0-1, o_proj 0-2)       -> bf16 of the exact (q - 8) s
  fp8_row (up1 embed_tokens, lm_head)       -> bf16 of the exact e4m3 x row scale
  bf16 (v0.7 embed_tokens, lm_head)         -> as stored (exact)
Trellis, int4 and FP8 row formats round once to BF16. Dense trellis uses mcg for both releases. The shared expert
uses the selected release's expert codebook and is permuted so this rank's halves (release rule
(16 * 256 + h + L) mod N) are rows [rank * 128 n, ...), with zeros elsewhere. Layer 78 (MTP) is skipped.

Usage: exlr8_dense.py RELEASE INDEX_DIR RANK RANKS OUT --manifest MANIFEST [--layers 0-77] [--vocab] [--check]
(needs numpy). Writes OUT/exlr8-dense-LLL.r{R}of{N}.safetensors and OUT/exlr8-vocab.safetensors."""
import argparse, json, os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dirstate import MCG, MUL1, check_expert_codebook, check_index, index_hashes, read_manifest, selected_release

HALVES, HIDDEN = 16, 6144
FORMATS = {1: 'trellis', 2: 'int4', 3: 'mxfp8', 4: 'fp8_block', 5: 'fp8_row', 6: 'bf16', 7: 'fp32',
           8: 'fp8_32x32'}  # format 8 exists in exlr8-v2 (FORMAT.md); GLM-5.3 dense never uses it — refused below
DT = {'I16': np.int16, 'F8_E4M3': np.uint8, 'F16': np.float16, 'F32': np.float32, 'BF16': np.uint16, 'U8': np.uint8, 'I8': np.int8, 'I64': np.int64, 'U64': np.uint64, 'I32': np.int32}


class ST:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        n = struct.unpack('<Q', self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def has(self, name):
        return name in self.h

    def arr(self, name):
        m = self.h[name]
        s, e = m['data_offsets']
        self.f.seek(self.base + s)
        b = self.f.read(e - s)
        return np.frombuffer(b, dtype=DT[m['dtype']]).reshape(m['shape']) if m['shape'] else np.frombuffer(b, dtype=DT[m['dtype']])

    def at(self, off, n):
        self.f.seek(off)
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit(f'{self.path}: short read at {off}')
        return b


# ---- E4M3 and the mcg codebook ------------------------------------------------------------------------------------------
def e4m3_values():
    c = np.arange(256)
    e, m = (c >> 3) & 15, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.ldexp(1.0, e - 10))
    v = np.where(c & 0x80, -v, v)
    v[(c & 0x7F) == 0x7F] = np.nan
    return v


E4M3 = e4m3_values()


def e4m3_codes(x):
    """Nearest-even E4M3 codes (saturating) of float64 values (core/exlr8_codec.e4m3FromF32 on exact fp16 inputs)."""
    s = np.signbit(x).astype(np.int64) << 7
    a = np.abs(x)
    out = np.zeros(x.shape, np.int64)
    sat = a >= 448
    norm = (a >= 2.0 ** -6) & ~sat
    mant, ex = np.frexp(a)  # a = mant 2^ex, mant in [0.5, 1)
    q = np.round(mant * 16)  # 4 significant bits, half to even
    ex = np.where(q == 16, ex + 1, ex)
    q = np.where(q == 16, 8, q)
    out = np.where(norm, ((ex - 1 + 7) << 3) | (q.astype(np.int64) - 8), out)
    sub = ~norm & ~sat
    out = np.where(sub, np.round(a / 2.0 ** -9).astype(np.int64), out)
    out = np.where(sat, 0x7E, out)
    return out | s


def mcg_book():
    st = np.arange(65536, dtype=np.uint64)
    x = (st * np.uint64(MCG)) & np.uint64(0xFFFFFFFF)
    x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
    lo = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (x >> np.uint64(16)).astype(np.uint16).view(np.float16)
    total = (lo + hi).astype(np.float16)  # the fp16 sum
    return E4M3[e4m3_codes(total.astype(np.float64))]


BOOK = mcg_book()


def mul1_book():
    """mul1 (FORMAT.md §Codebook, v0.7's expert codebook): x = state * 0x83DCD12D mod 2**32; v = fp16(1024 + sum of the
    four bytes of x) * c + d with c = fp16 0x1EEE, d = fp16 0xC931, the multiply-add rounded ONCE to fp16 (a fused
    multiply-add, as the kernel and kindlingai's reference loader loader/exlr8.py compute it: double, then .half()) — then
    the E4M3 round every codebook applies."""
    st = np.arange(65536, dtype=np.uint64)
    x = (st * np.uint64(MUL1)) & np.uint64(0xFFFFFFFF)
    total = ((x & np.uint64(0xFF)) + ((x >> np.uint64(8)) & np.uint64(0xFF))
             + ((x >> np.uint64(16)) & np.uint64(0xFF)) + (x >> np.uint64(24)))
    c = np.array([0x1EEE], np.uint16).view(np.float16)[0]  # 0.68310546875
    d = np.array([0xC931], np.uint16).view(np.float16)[0]  # -10.3828125
    v = ((1024 + total).astype(np.float64) * np.float64(c) + np.float64(d)).astype(np.float16)   # one fp16 rounding
    return E4M3[e4m3_codes(v.astype(np.float64))]


MUL1_BOOK = mul1_book()
EXPERT_BOOKS = {'mcg': BOOK, 'mul1': MUL1_BOOK}


def tile_states(tiles, k):
    """tiles: uint8 [T, 32 k] -> 16-bit states [T, 256] (stream value p; core/exlr8_codec.tileState)."""
    w = np.frombuffer(tiles.tobytes(), dtype='<u4').reshape(tiles.shape[0], 8 * k).astype(np.uint64)
    ring = 256 * k
    out = np.empty((tiles.shape[0], 256), np.int64)
    for p in range(256):
        start = ((p + 1) * k + ring - 16) % ring
        off, i0 = start % 32, start // 32
        if off <= 16:
            s = (w[:, i0] >> np.uint64(16 - off)) & np.uint64(0xFFFF)
        else:
            s = ((w[:, i0] << np.uint64(off - 16)) | (w[:, (i0 + 1) % (ring // 32)] >> np.uint64(48 - off))) & np.uint64(0xFFFF)
        out[:, p] = s.astype(np.int64)
    return out


P = np.arange(256)
ROW = 2 * ((P // 8) % 4) + ((P % 8) & 1) + 8 * (((P % 8) >> 1) & 1)
COL = (P // 8) // 4 + 8 * ((P % 8) >> 2)


def tile_values(tiles, k, book=None):
    """[T, 16, 16] float64 weights (rotated domain) of tiles at width k; book: the codebook (mcg default)."""
    v = (BOOK if book is None else book)[tile_states(tiles, k)]
    out = np.empty((tiles.shape[0], 16, 16))
    out[:, ROW, COL] = v
    return out


def run_matrix(run, k, order, rows, cols, book=None):
    """A run of tiles as a (rows, cols) matrix (core/exlr8_codec.decodeRun): out_major (cols 128) tile (tr, tc) at
    8 tr + tc; in_major (rows 128) tile (tr, tc) at 64 (tc / 8) + 8 tr + tc % 8."""
    t = tile_values(np.frombuffer(run, np.uint8).reshape(-1, 32 * k), k, book)
    if order == 'out':  # [tr][tc 8][16][16]
        return t.reshape(rows // 16, 8, 16, 16).transpose(0, 2, 1, 3).reshape(rows, cols)
    return t.reshape(cols // 128, 8, 8, 16, 16).transpose(1, 3, 0, 2, 4).reshape(rows, cols)  # [cb][tr][tcl][16][16]


def fwht(m, axis):
    """Unnormalized Hadamard in blocks of 128 along `axis` (f64, exact on these values)."""
    m = np.moveaxis(m, axis, -1)
    shp = m.shape
    v = m.reshape(-1, 128).copy()
    d = 1
    while d < 128:
        v = v.reshape(-1, 128 // (2 * d), 2, d)
        a, b = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
        v[:, :, 0, :], v[:, :, 1, :] = a + b, a - b
        v = v.reshape(-1, 128)
        d *= 2
    return np.moveaxis(v.reshape(shp), -1, axis)


def reconstruct(t, suh, svh):
    """W (in, out) = diag(suh) H T H diag(svh) / 128, f64."""
    w = fwht(fwht(t, 0), 1)
    return w * suh[:, None] * svh[None, :] / 128.0


def to_bf16(x):
    """bf16 bits of float64 values, rounded once to nearest even (8 significant bits)."""
    x = np.asarray(x, np.float64)
    m, e = np.frexp(x)
    q = np.round(m * 256) / 256
    y = np.ldexp(q, e).astype(np.float32)  # exact: 8 significant bits
    b = y.view(np.uint32)
    normal = np.isfinite(y) & (np.abs(y) >= 2.0 ** -126)
    if np.any((b[normal] & 0xFFFF) != 0):
        raise SystemExit('bf16 rounding left low bits')
    return (b >> 16).astype(np.uint16)


# ---- one dense matrix ---------------------------------------------------------------------------------------------------
def check_vocab(f, release):
    for name in ('embed_tokens', 'lm_head'):
        record = name + '.record'
        if not f.has(record) or int(f.arr(record)[0]) != release.vocab_format:
            raise ValueError(f'{name}: vocabulary format differs from {release.name} (expected {FORMATS[release.vocab_format]})')


def check_dense_trellis(f, name):
    declared = f.h.get('__metadata__', {}).get('codebook', '')
    if (declared.split() or ['mcg'])[0] != 'mcg' or name + '.trellis' not in f.h:
        raise ValueError(f'{name}: expected dense trellis with the mcg codebook and bare tensor names')


def dense_matrix(f, name, check):
    rec = f.arr(name + '.record').astype(np.int64)
    fmt, out_n, in_n, k, side = FORMATS[int(rec[0])], int(rec[1]), int(rec[2]), int(rec[3]), int(rec[4])
    if fmt == 'trellis':
        check_dense_trellis(f, name)
        tr = f.arr(name + '.trellis')
        suh = f.arr(name + '.suh').astype(np.float64)
        svh = f.arr(name + '.svh').astype(np.float64)
        tv = tile_values(np.ascontiguousarray(tr).reshape(-1, 32 * k), k)
        if side == 2:  # per 128 inputs a run of in_major tiles over every output: [G][OB][tr 8][tcl 8][16][16]
            t = tv.reshape(in_n // 128, out_n // 128, 8, 8, 16, 16).transpose(0, 2, 4, 1, 3, 5).reshape(in_n, out_n)
        else:  # per 128 outputs, per 128 inputs a run of 64 out_major tiles: [G][B][tr 8][tc 8][16][16]
            t = tv.reshape(out_n // 128, in_n // 128, 8, 8, 16, 16).transpose(1, 2, 4, 0, 3, 5).reshape(in_n, out_n)
        del tv
        return {'weight': ('BF16', to_bf16(reconstruct(t, suh, svh).T))}, 'trellis->bf16'
    if fmt == 'int4':
        q4, s4 = f.arr(name + '.q4'), f.arr(name + '.s4').astype(np.float64)
        if side == 2:  # [in/128][out][64] bytes, [in/128][out][4] scales
            lo, hi = (q4 & 15).astype(np.int64) - 8, (q4 >> 4).astype(np.int64) - 8
            vals = np.stack([lo, hi], -1).reshape(in_n // 128, out_n, 128)  # input i % 128 = 2 byte + nibble
            sc = np.repeat(s4, 32, axis=2)
            w = (vals * sc).transpose(1, 0, 2).reshape(out_n, in_n)
        else:  # [out][in/2], [out][in/32]
            lo, hi = (q4 & 15).astype(np.int64) - 8, (q4 >> 4).astype(np.int64) - 8
            vals = np.stack([lo, hi], -1).reshape(out_n, in_n)
            w = vals * np.repeat(s4, 32, axis=1)
        return {'weight': ('BF16', to_bf16(w))}, 'int4->bf16'
    if fmt == 'mxfp8':
        codes, e8 = f.arr(name + '.fp8'), f.arr(name + '.e8m0')
        w = E4M3[codes.astype(np.int64)] * np.ldexp(1.0, np.repeat(e8.astype(np.int64), 32, axis=1) - 127)
        b = to_bf16(w)
        if check and not np.array_equal((b.astype(np.uint32) << 16).view(np.float32).astype(np.float64), w):
            raise SystemExit(f'{name}: MXFP8 -> bf16 not exact')
        return {'weight': ('BF16', b)}, 'mxfp8->bf16 (exact)'
    if fmt == 'fp8_block':
        return {'weight': ('F8_E4M3', f.arr(name + '.fp8')), 'weight_scale_inv': ('F32', f.arr(name + '.scale'))}, 'fp8_block (as stored)'
    if fmt == 'fp8_row':
        codes, sc = f.arr(name + '.fp8'), f.arr(name + '.scale').astype(np.float64)
        out = np.empty((out_n, in_n), np.uint16)
        for r0 in range(0, out_n, 4096):  # in slices: the vocab tables are large
            out[r0:r0 + 4096] = to_bf16(E4M3[codes[r0:r0 + 4096].astype(np.int64)] * sc[r0:r0 + 4096, None])
        return {'weight': ('BF16', out)}, 'fp8_row->bf16'
    if fmt == 'bf16':
        return {'weight': ('BF16', f.arr(name + '.bf16'))}, 'bf16'
    if fmt == 'fp32':
        return {'weight': ('F32', f.arr(name + '.fp32'))}, 'fp32'
    raise SystemExit(f'{name}: format {fmt}')


# ---- the shared expert, from its halves ---------------------------------------------------------------------------------
def shared_expert(rel, idir, L, rank, ranks, shared_widths, release, hashes):
    index_path = os.path.join(idir, 'experts', f'exlr8-layer-{L:03d}.index.safetensors')
    check_index(index_path, hashes)
    idx = ST(index_path)
    book = EXPERT_BOOKS[release.codebook]
    signs_in = idx.arr('signs.in').astype(np.float64)
    signs_out = idx.arr('signs.out').astype(np.float64)
    mine = [h for h in range(HALVES) if (16 * 256 + h + L) % ranks == rank]
    nh = len(mine)
    out = {p: np.zeros((2048, HIDDEN) if p != 'down' else (HIDDEN, 2048), np.uint16) for p in ('gate', 'up', 'down')}
    files = {}
    # each half at the width the profile's manifest gives it (the release stores the shared expert at several widths;
    # a box holds only its profile's): gate/up a width a half, down one width for all its halves
    for pi, p in enumerate(('gate', 'up', 'down')):
        w = shared_widths[p]
        ks = list(w) if isinstance(w, list) else [w] * HALVES
        for q, h in enumerate(mine):
            k = int(ks[h])
            ix = idx.arr(f'shared.k{k}.index').reshape(3, 16, 2)
            o, n = int(ix[pi, h, 0]), int(ix[pi, h, 1])
            if n != 3072 * 32 * k:
                raise SystemExit(f'layer {L} shared {p} half {h}: index length {n} at K{k}')
            if k not in files:
                path = os.path.join(rel, 'experts', f'exlr8-layer-{L:03d}.k{k}.safetensors')
                wf = ST(path)
                check_expert_codebook(path, wf.h, release)
                files[k] = wf
            half = files[k].at(o, n)
            block = idx.arr(f'shared.k{k}.{p}.' + ('sv_block' if p == 'down' else 'su_block'))[0].astype(np.float64)
            chan = idx.arr(f'shared.k{k}.{p}.' + ('suh' if p == 'down' else 'svh'))[0, h * 128:(h + 1) * 128].astype(np.float64)
            hid = (signs_out if p == 'down' else signs_in) * np.repeat(block, 128)
            slot = rank * 128 * nh + q * 128  # this rank's rows of the permuted intermediate
            if p == 'down':  # (128 in, 6144 out), in_major
                wr = reconstruct(run_matrix(half, k, 'in', 128, HIDDEN, book), chan, hid)  # (in=128, out=6144)
                out[p][:, slot:slot + 128] = to_bf16(wr.T)
            else:  # (6144 in, 128 out), out_major
                wr = reconstruct(run_matrix(half, k, 'out', HIDDEN, 128, book), hid, chan)  # (in=6144, out=128)
                out[p][slot:slot + 128, :] = to_bf16(wr.T)
    return out


# ---- writer -------------------------------------------------------------------------------------------------------------
def write(path, tensors, meta):
    hdr, off, blobs = {'__metadata__': meta}, 0, []
    for name, (dt, a) in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        hdr[name] = {'dtype': dt, 'shape': list(a.shape), 'data_offsets': [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hb = json.dumps(hdr, separators=(',', ':')).encode()
    hb += b' ' * (-len(hb) % 8)
    with open(path + '.part', 'wb') as f:
        f.write(struct.pack('<Q', len(hb)) + hb)
        for b in blobs:
            f.write(b)
    os.replace(path + '.part', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('release'); ap.add_argument('index_dir'); ap.add_argument('rank', type=int); ap.add_argument('ranks', type=int); ap.add_argument('out')
    ap.add_argument('--manifest', required=True); ap.add_argument('--layers', default='0-77'); ap.add_argument('--vocab', action='store_true'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split('-'))
    release = selected_release()
    man = read_manifest(a.manifest, release)
    hashes = index_hashes(release)
    os.makedirs(a.out, exist_ok=True)
    wm_path = os.path.join(a.out, f'exlr8-dense.r{a.rank}of{a.ranks}.index.json')
    wm = json.load(open(wm_path))['weight_map'] if os.path.exists(wm_path) else {}
    jobs = [(L, os.path.join(a.release, 'dense', f'exlr8-dense-{L:03d}.safetensors'), f'exlr8-dense-{L:03d}.r{a.rank}of{a.ranks}.safetensors') for L in range(lo, hi + 1)]
    if a.vocab:
        jobs.append((None, os.path.join(a.release, 'dense', 'exlr8-vocab.safetensors'), 'exlr8-vocab.safetensors'))
    for L, src, dst in jobs:
        f = ST(src)
        if L is None:
            check_vocab(f, release)
        pre = 'model.' if L is None else f'model.layers.{L}.'
        recs = [k[:-len('.record')] for k in f.h if k.endswith('.record')]
        tensors, kinds = {}, {}
        for r in recs:
            got, kind = dense_matrix(f, r, a.check)
            for part, v in got.items():
                tensors[('lm_head' if r == 'lm_head' else pre + r) + '.' + part] = v
            kinds[kind] = kinds.get(kind, 0) + 1
        for k in f.h:
            if k.startswith('__') or any(k.startswith(r + '.') for r in recs):
                continue
            m = f.h[k]
            tensors[pre + k] = (m['dtype'], f.arr(k))
        for r in recs:  # a record's extra plain tensors (the router's e_score_correction_bias)
            for k in f.h:
                if k.startswith(r + '.') and k.split('.')[-1] not in ('record', 'checksum', 'trellis', 'suh', 'svh', 'q4', 's4', 'fp8', 'e8m0', 'scale', 'bf16', 'fp32'):
                    tensors[pre + k] = (f.h[k]['dtype'], f.arr(k))
        if L is not None and L >= 3:  # the MoE layers' shared expert
            se = shared_expert(a.release, a.index_dir, L, a.rank, a.ranks, man['layers'][str(L)]['shared'], release, hashes)
            for p, w in se.items():
                tensors[f'{pre}mlp.shared_experts.{p}_proj.weight'] = ('BF16', w)
            kinds['shared trellis->bf16 (permuted, rank rows)'] = 3
        write(os.path.join(a.out, dst), tensors, {'format': 'pt',
                                                  'source': release.receipt,
                                                  'rank': str(a.rank)})
        for name in tensors:
            wm[name] = dst
        json.dump({'weight_map': wm}, open(wm_path, 'w'))
        print(f"{dst}: {len(tensors)} tensors; {kinds}", flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError) as error:
        sys.exit(f'EXLR8: {error}')
