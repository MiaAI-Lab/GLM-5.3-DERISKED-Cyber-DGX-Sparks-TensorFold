#!/usr/bin/env python3
"""X8: finish a rank's converted EXLR8 checkpoint directory (exlr8_pieces.py + exlr8_dense.py outputs) so TensorFold
serves it. Writes the selected release's quantization config and set identity, the tokenizer and chat template,
and model.safetensors.index.json over every tensor of this rank's files.
Refuses when a layer's pieces or dense file is missing, or one name is in two files.
Usage: exlr8_assemble.py RELEASE OUT RANK RANKS [--layers 0-77] [--native]"""
import hashlib, json, os, shutil, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dirstate import (METADATA, SOURCE_LICENSING, check_counts, check_output_release, protect_output, read_json,
                      release_identity, selected_release, shard_names)


def names(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return [k for k in json.loads(f.read(n)) if k != '__metadata__']


def main():
    rel, out, rank, ranks = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    lo, hi = 0, 77
    if '--layers' in sys.argv:
        lo, hi = (int(x) for x in sys.argv[sys.argv.index('--layers') + 1].split('-'))
    if not 0 <= lo <= hi <= 77:
        raise SystemExit('expected layers within 0-77')
    release = selected_release()
    identity = release_identity(rel, ranks)
    check_output_release(out)
    files = shard_names(rank, ranks, lo, hi)
    wm = {}
    for fn in files:
        p = os.path.join(out, fn)
        if not os.path.exists(p):
            raise SystemExit(f'missing {fn}')
        tensors = names(p)
        if fn.startswith('exlr8-exl3-layer-'):
            markers = {name.rsplit('.', 1)[-1] for name in tensors if name.endswith('_e4m3')}
            if markers != {release.marker}:
                raise ValueError(f'{fn}: piece markers differ from {release.marker}')
        for n in tensors:
            if n in wm:
                raise SystemExit(f'{n} in {wm[n]} and {fn}')
            wm[n] = fn
    native = ('--native' in sys.argv or os.path.exists(os.path.join(out, '.x12-native'))
              or any(n.endswith('.trellis') and fn.startswith('exlr8-dense-') for n, fn in wm.items()))
    if native:
        protect_output(out)
        if os.path.lexists(os.path.join(out, '.x8-complete')):
            raise SystemExit('native assembly requires a staging directory without a completion marker')
    if (lo, hi) == (0, 77):
        check_counts(release, 'native' if native else 'interim', len(files), len(wm))

    def jdump(obj, p, **kw):
        with open(p, 'w') as f:
            json.dump(obj, f, **kw)
            f.flush()
            os.fsync(f.fileno())

    def put(name, write):  # every file through a temporary, synced, and a rename
        tmp = os.path.join(out, name + '.x8tmp')
        write(tmp)
        fd = os.open(tmp, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, os.path.join(out, name))

    cfg = read_json(Path(rel) / 'tokenizer' / 'config.json')
    # config.json is identical on every rank (the engine compares its sha256 across ranks): no rank in it
    cfg['quantization_config'] = {'quant_method': 'exl3', 'bits': 4, 'codebook': release.marker,
                                  'source': f'{release.receipt} (exlr8-v2, {release.profile}) converted for TensorFold',
                                  'source_licensing': SOURCE_LICENSING}
    # the conversion set every rank agrees on (the engine's checkpoint identity for rank-piece checkpoints, patch 0205):
    # release and revision, profile, ranks and the converter's own bytes; identical on every box by construction
    # provenance: the converters that wrote these weights (.x8-converted-by, written before converting), never the
    # copies present now; the release with its revision, and the profile's set_id, all required
    cb = os.path.join(out, '.x8-converted-by')
    if not os.path.exists(cb):
        raise SystemExit('no .x8-converted-by: which converters wrote these weights is unknown')
    tools = json.load(open(cb))
    if sorted(tools) != ['exlr8_dense.py', 'exlr8_pieces.py'] or not all(len(v) == 64 for v in tools.values()):
        raise SystemExit(f'.x8-converted-by: {tools}')
    the_set = {**identity, 'tools': tools}
    if native:
        the_set['dense'] = 'native'
    for t in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'):
        put(t, lambda p, t=t: shutil.copyfile(os.path.join(rel, 'tokenizer', t), p))
    put('config.json', lambda p: jdump(cfg, p, indent=1))
    put('exlr8-set.json', lambda p: jdump(the_set, p, sort_keys=True))
    put('model.safetensors.index.json', lambda p: jdump({'metadata': {'rank': rank, 'ranks': ranks}, 'weight_map': wm}, p))
    for fn in files:  # the weights on disk before the marker that vouches for them
        fd = os.open(os.path.join(out, fn), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    dfd = os.open(out, os.O_RDONLY)
    os.fsync(dfd)  # and their renames
    os.close(dfd)
    # the completion marker last: what a later run trusts before it reuses this directory
    every = files + list(METADATA)
    def sha(fn):
        h = hashlib.sha256()
        with open(os.path.join(out, fn), 'rb') as f:
            for blk in iter(lambda: f.read(1 << 24), b''):
                h.update(blk)
        return h.hexdigest()

    mark = {'rank': rank, 'ranks': ranks, 'files': len(files), 'tensors': len(wm), 'sizes': {fn: os.path.getsize(os.path.join(out, fn)) for fn in every},
            'sha256': {fn: sha(fn) for fn in every},  # the stage checks every byte against these before each boot
            'mtimes': {fn: os.stat(os.path.join(out, fn)).st_mtime_ns for fn in every}}  # a replaced file changes its mtime
    if native:
        mark['dense'] = 'native'
    put('.x8-complete.pending' if native else '.x8-complete', lambda p: jdump(mark, p))
    dfd = os.open(out, os.O_DIRECTORY)
    os.fsync(dfd)
    os.close(dfd)
    print(f'{out}: {len(wm)} tensors in {len(files)} files, config codebook {release.marker}, rank {rank} of {ranks}')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError) as error:
        sys.exit(f'EXLR8: {error}')
