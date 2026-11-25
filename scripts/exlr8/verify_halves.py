#!/usr/bin/env python3
"""X8: prove real_k4.bin is two whole release halves: each half's xxhash64 (seed 0) over its trellis bytes, its 128
intermediate-side fp16 values, then the projection's 48 hidden-side fp16 block scales equals the release index's
checksum (the format's rule, index metadata 'checksum'). Usage: verify_halves.py INDEX real_k4.bin
Halves (layer 3, K4): gate expert 0 half 1, down expert 7 half 5 (both rank 0 at TP4 by (16e + h + L) mod 4)."""
import array, json, struct, sys

P1, P2, P3, P4, P5 = 11400714785074694791, 14029467366897019727, 1609587929392839161, 9650029242287828579, 2870177450012600261
M = (1 << 64) - 1
rotl = lambda x, r: ((x << r) | (x >> (64 - r))) & M


def rnd(acc, lane):
    return (rotl((acc + lane * P2) & M, 31) * P1) & M


def merge(acc, v):
    return ((acc ^ rnd(0, v)) * P1 + P4) & M


def xxh64(b, seed=0):
    n, i = len(b), 0
    if n >= 32:
        v = [(seed + P1 + P2) & M, (seed + P2) & M, seed, (seed - P1) & M]
        while i + 32 <= n:
            for j in range(4):
                v[j] = rnd(v[j], struct.unpack_from('<Q', b, i + 8 * j)[0])
            i += 32
        h = (rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18)) & M
        for x in v:
            h = merge(h, x)
    else:
        h = (seed + P5) & M
    h = (h + n) & M
    while i + 8 <= n:
        h = (rotl(h ^ rnd(0, struct.unpack_from('<Q', b, i)[0]), 27) * P1 + P4) & M
        i += 8
    if i + 4 <= n:
        h = (rotl(h ^ (struct.unpack_from('<I', b, i)[0] * P1 & M), 23) * P2 + P3) & M
        i += 4
    while i < n:
        h = (rotl(h ^ (b[i] * P5 & M), 11) * P1) & M
        i += 1
    h ^= h >> 33; h = h * P2 & M; h ^= h >> 29; h = h * P3 & M; h ^= h >> 32
    return h


def main():
    idx, real = sys.argv[1], sys.argv[2]
    b = open(idx, 'rb').read()
    n = struct.unpack('<Q', b[:8])[0]
    h = json.loads(b[8:8 + n])
    base = 8 + n

    def raw(name):
        s, e = h[name]['data_offsets']
        return b[base + s:base + e]

    off = array.array('q'); off.frombytes(raw('k4.index'))
    cs = array.array('Q'); cs.frombytes(raw('k4.checksum'))
    data = open(real, 'rb').read()
    half = 3072 * 32 * 4
    want = [(0, 0, 1, 'gate', 'svh', 'su_block'), (7, 2, 5, 'down', 'suh', 'sv_block')]
    if len(data) != half * len(want):
        print(f'FAIL size {len(data)} != {half * len(want)}'); return 1
    ok = 0
    for i, (e, p, hf, proj, chan, block) in enumerate(want):
        o, ln = off[((e * 3 + p) * 16 + hf) * 2], off[((e * 3 + p) * 16 + hf) * 2 + 1]
        if ln != half:
            print(f'FAIL {proj} e{e} h{hf}: index length {ln}'); return 1
        trellis = data[i * half:(i + 1) * half]
        c = raw(f'k4.{proj}.{chan}')[(e * 2048 + hf * 128) * 2:(e * 2048 + hf * 128 + 128) * 2]
        bl = raw(f'k4.{proj}.{block}')[e * 48 * 2:(e + 1) * 48 * 2]
        got, exp = xxh64(trellis + c + bl), cs[(e * 3 + p) * 16 + hf]
        print(f"{'PASS' if got == exp else 'FAIL'} {proj} expert {e} half {hf} at {o}: xxhash64 {got:016x} index {exp:016x}")
        ok += got == exp
    # control: a flipped byte must not verify
    bad = bytearray(data[:half]); bad[1000] ^= 1
    e, p, hf, proj, chan, block = want[0]
    c = raw(f'k4.{proj}.{chan}')[(e * 2048 + hf * 128) * 2:(e * 2048 + hf * 128 + 128) * 2]
    bl = raw(f'k4.{proj}.{block}')[e * 48 * 2:(e + 1) * 48 * 2]
    ctrl = xxh64(bytes(bad) + c + bl) != cs[(e * 3 + p) * 16 + hf]
    print(f"{'PASS' if ctrl else 'FAIL'} control: a flipped byte does not verify")
    return 0 if ok == len(want) and ctrl else 1


if __name__ == '__main__':
    sys.exit(main())
