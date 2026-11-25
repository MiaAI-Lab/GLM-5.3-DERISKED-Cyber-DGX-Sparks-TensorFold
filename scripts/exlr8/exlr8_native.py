#!/usr/bin/env python3
"""X8B (patch 0212): kindlingai's EXLR8 release (glm-5.3-exlr8-k2-k3-k4, format exlr8-v2 by kindlingai) as
the dense part of a rank's TensorFold checkpoint, with every trellis matrix kept AS STORED instead of rounded to bf16.

X8's night/gr8x/x8/exlr8_dense.py wrote q_b / o_proj / the layer 0-2 MLP and the shared expert as bf16 (rounded once).
This writes, for each trellis record (q_b layers 2-77, o_proj 3-77, mlp gate/up/down of layers 0-2), the matrix as one
EXL3 trellis, the layout patch 0205b's engine reads (cuda_x8.zig):
  model.layers.L.<module>.trellis   I16 [in/16][out/16][16 k]  tile (tk, tn) at tk * out/16 + tn, bytes as released
  model.layers.L.<module>.suh       F16 [in]                   the release's suh (input side)
  model.layers.L.<module>.svh       F16 [out]                  the release's svh (output side)
  model.layers.L.<module>.mcg_e4m3  I32 []                     0xCBAC1FED: EXLR8's marker (patch 0205)
The marker stays mcg_e4m3 / 0xCBAC1FED: v0.7 re-encoded only the EXPERTS in mul1 (k-file tensor names trellis.<proj>.mul1);
the dense trellis records keep the bare names and the mcg default (confirmed on disk), and the expert pieces carry
mul1_e4m3 / 0x83DCD12D instead (exlr8_pieces.py). The vocabulary uses the selected release's FP8 row or BF16 records.
Tiles move whole (no value changes): the release stores per 128 outputs (record side 1, out_major: [out/128][in/128][tr 8]
[tc 8]) or per 128 inputs (side 2, in_major: [in/128][out/128][tr 8][tc 8]) runs of 64 tiles; the tile (tr, tc) of run
(input block B, output block G) is EXL3 tile (8 B + tr, 8 G + tc). The shared expert is NOT written here: its rank pieces
(mlp.shared_experts.*.rankR.*, exlr8_pieces.py) are what 0205b serves. Every other record is as X8 wrote it (mxfp8 ->
bf16 exact, fp8_block as stored, bf16/fp32 as stored, int4 and fp8_row -> bf16 rounded once: still the interim).

Usage: exlr8_native.py RELEASE INDEX_DIR RANK RANKS OUT [--layers 0-77] [--vocab] [--check]
OUT must be an empty, private staging directory (e.g. mktemp -d)."""
import argparse
import fcntl
import os
import sys
import numpy as np

import exlr8_dense as xd  # X8's reader, codebook and reference reconstruction
from dirstate import MCG, SOURCE_LICENSING, protect_output, reject_mounts, release_identity, selected_release


def exl3_tiles(tr, k, side, out_n, in_n):
    """The release's trellis runs as an EXL3 trellis: uint8 [in/16, out/16, 32 k] (as int16 [.., 16 k] on disk)."""
    tb = 32 * k
    t = np.ascontiguousarray(tr).reshape(-1, tb)
    if t.shape[0] != (in_n // 16) * (out_n // 16):
        raise SystemExit(f'trellis holds {t.shape[0]} tiles, the record says {in_n} x {out_n}')
    if side == 2:  # [G = in block][OB][tr][tc]
        t = t.reshape(in_n // 128, out_n // 128, 8, 8, tb).transpose(0, 2, 1, 3, 4)
    else:  # [G = out block][B = in block][tr][tc]
        t = t.reshape(out_n // 128, in_n // 128, 8, 8, tb).transpose(1, 2, 0, 3, 4)
    return np.ascontiguousarray(t).reshape(in_n // 16, out_n // 16, tb)


def exl3_reconstruct(t3, k, suh, svh):
    """W (in, out) from the EXL3 layout, f64 (the check: must equal X8's run order reconstruction)."""
    in_n, out_n = t3.shape[0] * 16, t3.shape[1] * 16
    v = xd.tile_values(t3.reshape(-1, 32 * k), k).reshape(in_n // 16, out_n // 16, 16, 16)
    return xd.reconstruct(v.transpose(0, 2, 1, 3).reshape(in_n, out_n), suh, svh)


def trellis_native(f, name, check):
    xd.check_dense_trellis(f, name)
    rec = f.arr(name + '.record').astype(np.int64)
    out_n, in_n, k, side = int(rec[1]), int(rec[2]), int(rec[3]), int(rec[4])
    t3 = exl3_tiles(f.arr(name + '.trellis'), k, side, out_n, in_n)
    suh, svh = f.arr(name + '.suh'), f.arr(name + '.svh')
    if suh.shape != (in_n,) or svh.shape != (out_n,):
        raise SystemExit(f'{name}: suh {suh.shape} / svh {svh.shape} against {in_n} x {out_n}')
    if check:  # the EXL3 placement against X8's (reference-checked) run order, on the matrix's first 256 x 256
        got = exl3_reconstruct(t3[:16, :16], k, suh[:256].astype(np.float64), svh[:256].astype(np.float64))
        ref = xd.dense_matrix(f, name, False)[0]['weight'][1][:256, :256]  # bf16 [out][in]
        if not np.array_equal(xd.to_bf16(got.T), ref):
            raise SystemExit(f'{name}: EXL3 placement does not reproduce X8\'s matrix')
    return {'trellis': ('I16', t3.view(np.int16)), 'suh': ('F16', suh), 'svh': ('F16', svh),
            'mcg_e4m3': ('I32', np.array(MCG, dtype=np.uint32).view(np.int32).reshape(()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('release'); ap.add_argument('index_dir'); ap.add_argument('rank', type=int); ap.add_argument('ranks', type=int); ap.add_argument('out')
    ap.add_argument('--layers', default='0-77'); ap.add_argument('--vocab', action='store_true'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    try:
        out = protect_output(a.out)
        reject_mounts(out)
        fd = os.open(out, os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            st = os.fstat(fd)
            if st.st_uid != os.geteuid() or st.st_mode & 0o077 or os.listdir(fd):
                raise ValueError('use an empty, private staging directory owned by this user (mktemp -d)')
            convert(a, out)
        finally:
            os.close(fd)
    except (OSError, ValueError, RuntimeError) as error:
        ap.exit(1, f'exlr8_native.py: {error}\n')


def convert(a, out):
    release = selected_release()
    release_identity(a.release, a.ranks)
    lo, hi = (int(x) for x in a.layers.split('-'))
    if not 0 <= lo <= hi <= 77 or not 0 <= a.rank < a.ranks:
        raise ValueError('expected layers within 0-77 and a rank below ranks')
    jobs = [(L, os.path.join(a.release, 'dense', f'exlr8-dense-{L:03d}.safetensors'), f'exlr8-dense-{L:03d}.r{a.rank}of{a.ranks}.safetensors') for L in range(lo, hi + 1)]
    if a.vocab:
        jobs.append((None, os.path.join(a.release, 'dense', 'exlr8-vocab.safetensors'), 'exlr8-vocab.safetensors'))
    for L, src, dst in jobs:
        f = xd.ST(src)
        if L is None:
            xd.check_vocab(f, release)
        pre = 'model.' if L is None else f'model.layers.{L}.'
        recs = [k[:-len('.record')] for k in f.h if k.endswith('.record')]
        tensors, kinds = {}, {}
        for r in recs:
            fmt = xd.FORMATS[int(f.arr(r + '.record')[0])]
            if fmt == 'trellis':
                got, kind = trellis_native(f, r, a.check), 'trellis (as stored, EXL3 layout)'
            else:
                got, kind = xd.dense_matrix(f, r, a.check)
            for part, v in got.items():
                tensors[('lm_head' if r == 'lm_head' else pre + r) + '.' + part] = v
            kinds[kind] = kinds.get(kind, 0) + 1
        for k in f.h:  # plain tensors (norms, biases), as X8 copies them
            if k.startswith('__') or any(k.startswith(r + '.') for r in recs):
                continue
            tensors[pre + k] = (f.h[k]['dtype'], f.arr(k))
        for r in recs:
            for k in f.h:
                if k.startswith(r + '.') and k.split('.')[-1] not in ('record', 'checksum', 'trellis', 'suh', 'svh', 'q4', 's4', 'fp8', 'e8m0', 'scale', 'bf16', 'fp32'):
                    tensors[pre + k] = (f.h[k]['dtype'], f.arr(k))
        xd.write(str(out / dst), tensors, {'format': 'pt',
                                           'source': release.receipt,
                                           'source_licensing': SOURCE_LICENSING, 'rank': str(a.rank)})
        print(f'{dst}: {len(tensors)} tensors; {kinds}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
