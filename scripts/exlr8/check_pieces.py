#!/usr/bin/env python3
"""X8 part 1 check: a repacked EXL3 rank piece, read the EXL3 way (tile (tk, tn) at tk * tiles_n + tn, W = diag(suh) H
Wq H diag(svh), as TensorFold's core/exl3_format.dequantize), equals the release's halves read the EXLR8 way (tile order
out_major / in_major per core/exlr8_codec.decodeRun, suh/svh per core/exlr8_experts.Half), block by block, exactly in
f64 (both are sums of the same E4M3 values with the same +-1 Hadamard and fp16 scales). Independent of the repacker:
this reads the release files and the index directly. Controls: a piece block compared with the wrong half, and a
piece with its suh and svh swapped, must differ.
Usage: check_pieces.py RELEASE INDEX_DIR PIECES_DIR LAYER RANK RANKS [blocks]"""
import json, os, random, struct, sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dirstate import MCG, MUL1, check_expert_codebook, check_index, index_hashes, selected_release

HALVES = 16


def st(path):
    f = open(path, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); return f, h, 8 + n


def raw(fhb, name):
    f, h, b = fhb; s, e = h[name]['data_offsets']; f.seek(b + s); return f.read(e - s)


def half_f16(b, i):
    return struct.unpack_from('<e', b, 2 * i)[0]


def e4m3_from(x):  # nearest even, saturating (core/exlr8_codec.e4m3FromF32 on the f32 of an fp16 value)
    import math
    if x != x: return 0x7F
    s = 0x80 if math.copysign(1.0, x) < 0 else 0
    a = abs(x)
    if a >= 448: return s | 0x7E
    if a == 0: return s
    if a >= 2 ** -6:
        m, e = math.frexp(a)  # a = m 2^e, m in [0.5, 1)
        q = round(m * 16) / 16  # 4 significant bits; round() is nearest even on exact ties in Python
        if q == 1.0: q, e = 0.5, e + 1
        exp = e - 1 + 7
        return s | (exp << 3) | (int(q * 16) - 8)
    n = round(a / 2 ** -9)
    return s | n


def e4m3_val(c):
    e, m = (c >> 3) & 15, c & 7
    v = m * 2 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10)
    return -v if c & 0x80 else v


def mcg(state):
    x = (state * MCG) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo, hi = struct.unpack('<ee', struct.pack('<I', x))
    return lo + hi


def mul1(state):
    x = (state * MUL1) & 0xFFFFFFFF
    total = sum(x.to_bytes(4, 'little'))
    c, d = struct.unpack('<ee', struct.pack('<HH', 0x1EEE, 0xC931))
    return (1024 + total) * c + d


def tile_values(tile, k, book):
    """256 values at (row, col) of a 16x16 tile (stream: u32 little-endian words, MSB first; core/exlr8_codec)."""
    ring = 256 * k
    words = struct.unpack(f'<{8 * k}I', tile)
    out = [[0.0] * 16 for _ in range(16)]
    for p in range(256):
        start = ((p + 1) * k + ring - 16) % ring
        off = start % 32
        lo = words[start // 32]
        if off <= 16:
            s = (lo >> (16 - off)) & 0xFFFF
        else:
            hi = words[(start // 32 + 1) % (ring // 32)]
            s = ((lo << (off - 16)) | (hi >> (48 - off))) & 0xFFFF
        l, j = p // 8, p % 8
        out[2 * (l % 4) + (j & 1) + 8 * ((j >> 1) & 1)][l // 4 + 8 * (j >> 2)] = book(s)
    return out


def fwht_cols(m):  # unnormalized Hadamard along rows index (axis 0) of a 128x128 list
    d = 1
    while d < 128:
        for base in range(0, 128, 2 * d):
            for i in range(base, base + d):
                a, b = m[i], m[i + d]
                m[i] = [x + y for x, y in zip(a, b)]
                m[i + d] = [x - y for x, y in zip(a, b)]
        d *= 2


def fwht_rows(m):
    for r in range(128):
        v = m[r]; d = 1
        while d < 128:
            for base in range(0, 128, 2 * d):
                for i in range(base, base + d):
                    v[i], v[i + d] = v[i] + v[i + d], v[i] - v[i + d]
            d *= 2


def block(get_tile, suh, svh, k, book):
    """W block (128 x 128) from a tile getter (tr, tc) in 0..7, suh/svh 128 fp16 values: diag(suh) H T H diag(svh) / 128."""
    t = [[0.0] * 128 for _ in range(128)]
    for tr in range(8):
        for tc in range(8):
            v = tile_values(get_tile(tr, tc), k, book)
            for r in range(16):
                t[16 * tr + r][16 * tc:16 * tc + 16] = v[r]
    fwht_cols(t); fwht_rows(t)
    return [[t[r][c] * suh[r] * svh[c] / 128.0 for c in range(128)] for r in range(128)]


def main():
    rel, idir, pdir, L, rank, ranks = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    nblocks = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    if nblocks < 1:
        raise SystemExit('check at least one block')
    release = selected_release()
    decode = {'mcg': mcg, 'mul1': mul1}[release.codebook]

    @lru_cache(maxsize=65536)
    def book(state):
        value, = struct.unpack('<e', struct.pack('<e', decode(state)))
        return e4m3_val(e4m3_from(value))

    pc = st(os.path.join(pdir, f'exlr8-exl3-layer-{L:03d}.r{rank}of{ranks}.safetensors'))
    decl = struct.unpack('<ii', raw(pc, f'model.layers.{L}.mlp.experts.pieces'))
    if decl != (rank, ranks):
        raise SystemExit(f'pieces file declares {decl}, not ({rank}, {ranks})')
    index_path = os.path.join(idir, 'experts', f'exlr8-layer-{L:03d}.index.safetensors')
    check_index(index_path, index_hashes(release))
    idx = st(index_path)
    wfs, ixc = {}, {}

    def width(k):  # the release's width file and index table of K (each piece is checked at its own width)
        if k not in wfs:
            path = os.path.join(rel, 'experts', f'exlr8-layer-{L:03d}.k{k}.safetensors')
            wt, header, _ = st(path)
            check_expert_codebook(path, header, release)
            wfs[k] = wt
            ixc[('', k)] = struct.unpack('<' + 'q' * (256 * 3 * 16 * 2), raw(idx, f'k{k}.index'))
            if f'shared.k{k}.index' in idx[1]:
                ixc[('shared.', k)] = struct.unpack('<' + 'q' * (3 * 16 * 2), raw(idx, f'shared.k{k}.index'))
        return wfs[k]
    signs = {'in': struct.unpack('<6144b', raw(idx, 'signs.in')), 'out': struct.unpack('<6144b', raw(idx, 'signs.out'))}
    rnd = random.Random(L * 100 + rank)
    passed = failed = 0
    for n in range(nblocks):
        e = 256 if n % 4 == 3 else rnd.randrange(256); pi = n % 3; proj = ('gate', 'up', 'down')[pi]
        pre, eb = ('shared.', 0) if e == 256 else ('', e)
        base = f'model.layers.{L}.mlp.shared_experts.{proj}_proj.rank{rank}' if e == 256 else f'model.layers.{L}.mlp.experts.{e}.{proj}_proj.rank{rank}'
        marker = base + '.' + release.marker
        if marker not in pc[1] or raw(pc, marker) != struct.pack('<I', release.multiplier):
            raise ValueError(f'{base}: missing or wrong {release.marker} marker')
        k = pc[1][base + '.trellis']['shape'][2] // 16  # this piece's width
        tb = 32 * k
        wf = width(k)
        ix = ixc[(pre, k)]
        mine = [h for h in range(HALVES) if (16 * e + h + L) % ranks == rank]
        q = rnd.randrange(len(mine)); h = mine[q]
        hb = rnd.randrange(48)  # which 128-block on the hidden side
        o, ln = ix[((eb * 3 + pi) * 16 + h) * 2], ix[((eb * 3 + pi) * 16 + h) * 2 + 1]
        wf.seek(o); half = wf.read(ln)
        blk_scale = raw(idx, f'{pre}k{k}.{proj}.' + ('sv_block' if proj == 'down' else 'su_block'))
        chan = raw(idx, f'{pre}k{k}.{proj}.' + ('suh' if proj == 'down' else 'svh'))
        hid = [signs['out' if proj == 'down' else 'in'][hb * 128 + i] * half_f16(blk_scale, eb * 48 + hb) for i in range(128)]
        ch = [half_f16(chan, eb * 2048 + h * 128 + i) for i in range(128)]
        # EXLR8 side: the half's own tile order and Half.suh/svh
        if proj == 'down':  # rows = 128 intermediate (the half), cols = hidden; tile (r, c) at 64 (c / 8) + 8 r + c % 8
            ref = block(lambda tr, tc: half[(64 * hb + 8 * tr + tc) * tb:][:tb], ch, hid, k, book)
        else:  # rows = hidden, cols = the half's 128; tile (r, c) at 8 r + c
            ref = block(lambda tr, tc: half[(8 * (8 * hb + tr) + tc) * tb:][:tb], hid, ch, k, book)
        # EXL3 side: the piece as an EXL3 matrix
        trel, psuh, psvh = raw(pc, base + '.trellis'), raw(pc, base + '.suh'), raw(pc, base + '.svh')
        shp = pc[1][base + '.trellis']['shape']; tiles_n = shp[1]
        if proj == 'down':  # K = intermediate (piece rows): block rows = half q, cols = hidden block hb
            got = block(lambda tr, tc: trel[((8 * q + tr) * tiles_n + 8 * hb + tc) * tb:][:tb],
                        [half_f16(psuh, 128 * q + i) for i in range(128)], [half_f16(psvh, 128 * hb + i) for i in range(128)], k, book)
        else:
            got = block(lambda tr, tc: trel[((8 * hb + tr) * tiles_n + 8 * q + tc) * tb:][:tb],
                        [half_f16(psuh, 128 * hb + i) for i in range(128)], [half_f16(psvh, 128 * q + i) for i in range(128)], k, book)
        same = got == ref
        print(f"{'PASS' if same else 'FAIL'} layer {L} {'shared expert' if e == 256 else f'expert {e}'} {proj} half {h} (piece slot {q}) hidden block {hb}", flush=True)
        passed += same; failed += not same
        if n == 0:  # controls on the first block: suh/svh swapped must differ; another half must differ when there is one
            sw = block(lambda tr, tc: trel[((8 * hb + tr) * tiles_n + 8 * q + tc) * tb:][:tb],
                       [half_f16(psvh, 128 * q + i) for i in range(128)], [half_f16(psuh, 128 * hb + i) for i in range(128)], k, book)
            c2 = sw != got
            c1 = None
            if len(mine) > 1 and proj != 'down':
                other = mine[(q + 1) % len(mine)]
                o2 = ix[((eb * 3 + pi) * 16 + other) * 2]; wf.seek(o2); half2 = wf.read(ln)
                c1 = block(lambda tr, tc: half2[(8 * (8 * hb + tr) + tc) * tb:][:tb], hid, ch, k, book) != got
            ok = c2 and c1 is not False
            print(f"{'PASS' if ok else 'FAIL'} controls: suh/svh swapped differs ({c2}), another half differs ({'n/a' if c1 is None else c1})")
            passed += ok; failed += not ok
    print(f'{passed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError) as error:
        sys.exit(f'EXLR8: {error}')
