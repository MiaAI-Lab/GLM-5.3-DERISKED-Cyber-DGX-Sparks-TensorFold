#!/usr/bin/env python3
"""X8 part 1: kindlingai's EXLR8 routed and shared experts (glm-5.3-exlr8-k2-k3-k4, format exlr8-v2, CC BY 4.0) as
TensorFold EXL3 rank pieces, one rank at a time, from the release files a rank already holds (gr8's TP4 caches follow
the release's sharding rule: half h of expert e in layer L on rank (16e + h + L) mod N).

Why it is exact: an EXLR8 half is an EXL3 16x16-tile trellis (same stream, lane placement and tile order; only the
codebook values are rounded to E4M3). The selected release determines the codebook marker and expected input
tensor names. A rank's piece of a projection is its
halves side by side on the intermediate axis, in increasing h: permuting intermediate channels the same way in gate, up
and down leaves the MoE output unchanged, and each 128-channel half keeps its own Had128 block, so the piece is one EXL3
matrix with suh/svh as TensorFold's exlr8_experts.Half.suh/svh define them (hidden side: layer signs x the expert's
128-block scale, exact in fp16; intermediate side: the half's 128 channel scales). Tiles are copied byte for byte:
gate/up halves keep tile (tr, c) at 8 tr + c (out_major), down halves tile (r, c) at 64 (c / 8) + 8 r + c % 8 (in_major).

Output: one safetensors file a layer holding only this rank's pieces (model.layers.L.mlp.experts.E.{gate,up,down}_proj.
rankR.{trellis,suh,svh,<codebook>_e4m3} and mlp.shared_experts.*.rankR.*) and the declaration model.layers.L.mlp.experts.pieces
(and mlp.shared_experts.pieces), int32 [rank, ranks]: the engine loads these pieces only on that rank and refuses them on
any other (patch 0205). The marker holds the selected codebook's multiplier. Widths come from the pinned manifest.
A piece whose halves differ in width is refused.

Usage: exlr8_pieces.py RELEASE MANIFEST RANK RANKS OUT [--index-dir DIR] [--layers 3-77] [--check N]
(--index-dir: where experts/exlr8-layer-LLL.index.safetensors are, when the release copy holds index slices only)
"""
import argparse, json, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dirstate import check_expert_codebook, check_index, index_hashes, read_manifest, selected_release

HALVES, ROUTED, SHARED, HIDDEN = 16, 256, 256, 6144


class ST:
    """A safetensors file's header and a mapped view of its data."""

    def __init__(self, path):
        self.f = open(path, 'rb')
        n = struct.unpack('<Q', self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n
        self.meta = self.h.get('__metadata__', {})

    def raw(self, name):
        s, e = self.h[name]['data_offsets']
        self.f.seek(self.base + s)
        return self.f.read(e - s)

    def at(self, off, n):
        self.f.seek(off)
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit(f'short read at {off} ({len(b)} of {n})')
        return b


def i64s(b):
    return struct.unpack(f'<{len(b) // 8}q', b)


def u64s(b):
    return struct.unpack(f'<{len(b) // 8}Q', b)


def rank_halves(e, layer, rank, ranks):
    return [h for h in range(HALVES) if (16 * e + h + layer) % ranks == rank]


def hidden_scales(signs, block_f16):
    """fp16 bytes of signs[i] * block[i // 128]: the block value with its sign bit flipped where the sign is -1."""
    rep = b''.join(block_f16[2 * j:2 * j + 2] * 128 for j in range(HIDDEN // 128))
    mask = b''.join(b'\x00\x80' if s < 0 else b'\x00\x00' for s in signs)
    return (int.from_bytes(rep, 'little') ^ int.from_bytes(mask, 'little')).to_bytes(len(rep), 'little')


def piece_trellis(halves_bytes, proj, k):
    """One EXL3 trellis [K/16][N/16][16 k] (int16 words, little-endian, as the release's u32 words) from the rank's halves."""
    tb = 32 * k
    nh = len(halves_bytes)
    out = bytearray(3072 * tb * nh)
    if proj in ('gate', 'up'):  # K = 6144 hidden rows (384 tile rows), N = 128 nh: half q's tiles 8 tr .. 8 tr + 7
        row = 8 * nh * tb
        for q, hb in enumerate(halves_bytes):
            for tr in range(384):
                out[tr * row + q * 8 * tb:tr * row + (q + 1) * 8 * tb] = hb[8 * tr * tb:(8 * tr + 8) * tb]
    else:  # down: K = 128 nh (8 tile rows a half), N = 6144 (384 tile columns): tile (r, c) at 64 (c / 8) + 8 r + c % 8
        row = 384 * tb
        for q, hb in enumerate(halves_bytes):
            for r in range(8):
                dst = (q * 8 + r) * row
                for cb in range(48):
                    src = (64 * cb + 8 * r) * tb
                    out[dst + cb * 8 * tb:dst + (cb + 1) * 8 * tb] = hb[src:src + 8 * tb]
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('release'); ap.add_argument('manifest'); ap.add_argument('rank', type=int); ap.add_argument('ranks', type=int)
    ap.add_argument('out'); ap.add_argument('--layers', default='3-77'); ap.add_argument('--check', type=int, default=0); ap.add_argument('--index-dir', default=None)
    a = ap.parse_args()
    if a.ranks not in (1, 2, 4, 8, 16) or not 0 <= a.rank < a.ranks:
        raise SystemExit('ranks must divide 16 and rank be below it')
    lo, hi = (int(x) for x in a.layers.split('-'))
    release = selected_release()
    man = read_manifest(a.manifest, release)
    hashes = index_hashes(release)
    marker = f'.{release.marker}'
    encoded_marker = struct.pack('<I', release.multiplier)
    os.makedirs(a.out, exist_ok=True)
    weight_map = {}
    for L in range(lo, hi + 1):
        lw = man['layers'][str(L)]
        index_path = os.path.join(a.index_dir or a.release, 'experts', f'exlr8-layer-{L:03d}.index.safetensors')
        check_index(index_path, hashes)
        idx = ST(index_path)
        if idx.meta.get('format') != 'exlr8-v2' or idx.meta.get('contents') != 'index':
            raise SystemExit(f'layer {L}: not an exlr8-v2 index')
        signs_in = struct.unpack('<6144b', idx.raw('signs.in'))
        signs_out = struct.unpack('<6144b', idx.raw('signs.out'))
        if set(signs_in) - {-1, 1} or set(signs_out) - {-1, 1}:
            raise SystemExit(f'layer {L}: layer signs other than +-1')
        files, tables = {}, {}

        def table(pre, k):
            key = (pre, k)
            if key not in tables:
                t = {'index': i64s(idx.raw(f'{pre}k{k}.index'))}
                for p, b, c in (('gate', 'su_block', 'svh'), ('up', 'su_block', 'svh'), ('down', 'sv_block', 'suh')):
                    t[p] = (idx.raw(f'{pre}k{k}.{p}.{b}'), idx.raw(f'{pre}k{k}.{p}.{c}'))
                tables[key] = t
            return tables[key]

        def width_file(k):
            if k not in files:
                path = os.path.join(a.release, 'experts', f'exlr8-layer-{L:03d}.k{k}.safetensors')
                wf = ST(path)
                check_expert_codebook(path, wf.h, release)
                files[k] = wf
            return files[k]

        units = [(e, f'mlp.experts.{e}') for e in range(ROUTED)] + [(SHARED, 'mlp.shared_experts')]
        tensors, plan = [], []  # header entries (name, dtype, shape, nbytes); this rank's writes (name, bytes-producer)
        for e, prefix in units:
            pre = 'shared.' if e == SHARED else ''
            mine = rank_halves(e, L, a.rank, a.ranks)
            for pi, proj in enumerate(('gate', 'up', 'down')):
                if e == SHARED:
                    w = lw['shared'][proj]
                    ks = list(w) if isinstance(w, list) else [w] * HALVES
                else:
                    ks = list(lw[proj][e]) if proj != 'down' else [lw['down'][e]] * HALVES
                for r in range(a.ranks):
                    rk = {ks[h] for h in rank_halves(e, L, r, a.ranks)}
                    if len(rk) != 1:
                        raise SystemExit(f'layer {L} {prefix}.{proj}: rank {r} halves of widths {sorted(rk)} (mixed widths: a later part)')
                    k = rk.pop()
                    nh = HALVES // a.ranks
                    shape_t = [384, 8 * nh, 16 * k] if proj != 'down' else [8 * nh, 384, 16 * k]
                    kdim, ndim = (HIDDEN, 128 * nh) if proj != 'down' else (128 * nh, HIDDEN)
                    if r != a.rank:
                        continue
                    base = f'model.layers.{L}.{prefix}.{proj}_proj.rank{r}'
                    tensors += [(base + '.trellis', 'I16', shape_t, 3072 * 32 * k * nh), (base + '.suh', 'F16', [kdim], 2 * kdim),
                                (base + '.svh', 'F16', [ndim], 2 * ndim), (base + marker, 'I32', [], 4)]
                    plan.append((base, e, pi, proj, k, mine, pre))
        decl = struct.pack('<ii', a.rank, a.ranks)
        decls = [(f'model.layers.{L}.mlp.experts.pieces', 'I32', [2], 8), (f'model.layers.{L}.mlp.shared_experts.pieces', 'I32', [2], 8)]
        tensors += decls
        # header with every rank's piece; data offsets in order
        hdr, off = {'__metadata__': {'format': 'pt', 'exlr8_pieces': f'rank {a.rank} of {a.ranks}',
                                     'source': idx.meta.get('format', ''), 'codebook': release.marker}}, 0
        for name, dt, shape, n in tensors:
            hdr[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + n]}
            off += n
        hb = json.dumps(hdr, separators=(',', ':')).encode()
        hb += b' ' * (-len(hb) % 8)
        path = os.path.join(a.out, f'exlr8-exl3-layer-{L:03d}.r{a.rank}of{a.ranks}.safetensors')
        tmp = path + '.part'
        with open(tmp, 'wb') as f:
            f.write(struct.pack('<Q', len(hb)) + hb)
            data0 = 8 + len(hb)
            f.truncate(data0 + off)
            for name, *_ in decls:
                f.seek(data0 + hdr[name]['data_offsets'][0]); f.write(decl)
            checked = 0
            for base, e, pi, proj, k, mine, pre in plan:
                t = table(pre, k)
                ix = t['index']
                halves = []
                for h in mine:
                    o, n = ix[((0 if e == SHARED else e) * 3 + pi) * 16 * 2 + h * 2], ix[((0 if e == SHARED else e) * 3 + pi) * 16 * 2 + h * 2 + 1]
                    if n != 3072 * 32 * k:
                        raise SystemExit(f'{base}: half {h} length {n}')
                    halves.append(width_file(k).at(o, n))
                trel = piece_trellis(halves, proj, k)
                block, chan = t[proj]
                eb = 0 if e == SHARED else e
                blk = block[eb * 96:(eb + 1) * 96]
                ch = b''.join(chan[(eb * 2048 + h * 128) * 2:(eb * 2048 + h * 128 + 128) * 2] for h in mine)
                if proj == 'down':
                    suh, svh = ch, hidden_scales(signs_out, blk)
                else:
                    suh, svh = hidden_scales(signs_in, blk), ch
                for suffix, data in (('.trellis', trel), ('.suh', suh), ('.svh', svh), (marker, encoded_marker)):
                    s, en = hdr[base + suffix]['data_offsets']
                    if len(data) != en - s:
                        raise SystemExit(f'{base}{suffix}: {len(data)} bytes, header says {en - s}')
                    f.seek(data0 + s)
                    f.write(data)
                if checked < a.check and proj != 'up':  # spot check: tile placement against the release, by index math
                    nh = len(mine)
                    tb = 32 * k
                    for q in range(nh):
                        for tr, c in ((0, 0), (5, 3), (7, 7)) if proj == 'down' else ((0, 0), (200, 5), (383, 7)):
                            src = halves[q][((64 * (c // 8) + 8 * tr + c % 8) if proj == 'down' else (8 * tr + c)) * tb:][:tb]
                            dst = (q * 8 + tr) * 384 + c if proj == 'down' else tr * 8 * nh + q * 8 + c
                            if trel[dst * tb:(dst + 1) * tb] != src:
                                raise SystemExit(f'{base}: tile placement check failed (half {q}, tile {tr},{c})')
                    checked += 1
        os.replace(tmp, path)
        for name, *_ in tensors:
            weight_map[name] = os.path.basename(path)
        print(f'layer {L}: {len(plan)} pieces of rank {a.rank} -> {path} ({off / 2**30:.2f} GiB)', flush=True)
    with open(os.path.join(a.out, f'exlr8-exl3-pieces.r{a.rank}of{a.ranks}.index.json'), 'w') as f:
        json.dump({'metadata': {'source': release.receipt,
                                'manifest': man.get('name'), 'rank': a.rank, 'ranks': a.ranks}, 'weight_map': weight_map}, f)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError) as error:
        sys.exit(f'EXLR8: {error}')
