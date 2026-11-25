#!/usr/bin/env python3
"""State of a rank's converted EXLR8 checkpoint directory (the marker files exlr8_assemble.py and box.sh write). Stdlib only.

  dirstate.py state  DIR RANK RANKS   new | foreign | partial | complete
        new: absent or empty. foreign: files without .x8-owned (never touched). complete: .x8-owned and an .x8-complete
        for this rank whose every file has its recorded size and mtime. partial: anything else this recipe owns.
  dirstate.py repair DIR              ok | repair: a complete directory from before the rank-free config.json and
        exlr8-set.json (or one whose set names other converter bytes) is assembled again in place, weights kept
  dirstate.py verify DIR RANK RANKS   sha256 of every file against the marker's: 'bytes ok N' (exit 0) or 'BAD ...' (exit 1)
  dirstate.py tools  TOOLS_DIR        the converters' provenance JSON ({exlr8_dense.py, exlr8_pieces.py: sha256}, sorted); with
        EXLR8_DENSE=native exlr8_dense.py's value is the sha256 of exlr8_native.py's bytes followed by exlr8_dense.py's
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

TOOLS = ('exlr8_dense.py', 'exlr8_pieces.py')
REPO = 'kindlingai/glm-5.3-exlr8-k2-k3-k4'
MCG = 0xCBAC1FED
MUL1 = 0x83DCD12D


@dataclass(frozen=True)
class Release:
    name: str
    revision: str
    profile: str
    set_id: str
    codebook: str
    multiplier: int
    vocab_format: int
    files: int
    interim_tensors: int
    native_tensors: int

    @property
    def receipt(self):
        return f'{REPO}@{self.revision}'

    @property
    def manifest(self):
        return f'manifests/exlr8-manifest-{self.profile}.json'

    @property
    def index_sha(self):
        return f'index-sha-{self.name}.json'

    @property
    def interim_leaf(self):
        return f'exlr8-{self.name}'

    @property
    def native_leaf(self):
        return f'{self.interim_leaf}-x8b'

    @property
    def served_name(self):
        return f'GLM-5.3-EXLR8-{self.profile}'

    @property
    def marker(self):
        return f'{self.codebook}_e4m3'


RELEASES = {
    'up1': Release('up1', 'a36ae4680871ffa6412061d5daf66c4d4ad5802c', 'up1',
                   '2011c0cedde962d19fc43e74c1d409279e398d40d27a6fdb4bcb7b49ea2ea6fb',
                   'mcg', MCG, vocab_format=5, files=154, interim_tensors=232722, native_tensors=232977),
    'v07': Release('v07', '9e25c28a2e8808d8b2599d780521fd6af24e7558', 'bit400',
                   '3d1a293399f0077e27acd18609ee67f33f8e3f8b5963a7c4b42b7dc16ecd1d98',
                   'mul1', MUL1, vocab_format=6, files=154, interim_tensors=232722, native_tensors=232977),
}
SET_FORMAT = 'tensorfold exlr8 rank pieces v1'
METADATA = ('config.json', 'exlr8-set.json', 'model.safetensors.index.json', 'tokenizer.json',
            'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')
SOURCE_LICENSING = ('CC BY 4.0 applies only to the packing, manifests and error tables. '
                    'The weights remain under the GLM-5.3 license. The EXLR8 format and reference loader '
                    'are separate works. The three ExLlamaV3-derived primitives retain their MIT terms.')


def read_json(path):
    with open(path) as file:
        return json.load(file)


def hex_digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(f'[0-9a-f]{{{length}}}', value) is not None


def shard_names(rank, ranks, lo=0, hi=77):
    suffix = f'r{rank}of{ranks}.safetensors'
    return ([f'exlr8-dense-{layer:03d}.{suffix}' for layer in range(lo, hi + 1)]
            + [f'exlr8-exl3-layer-{layer:03d}.{suffix}' for layer in range(max(lo, 3), hi + 1)]
            + ['exlr8-vocab.safetensors'])


def selected_release():
    name = os.environ.get('EXLR8_RELEASE') or 'up1'
    if name not in RELEASES:
        raise ValueError(f'EXLR8_RELEASE must be up1 or v07 (got {name!r})')
    release = RELEASES[name]
    require_revision(os.environ.get('EXLR8_REV'), release)
    profile = os.environ.get('EXLR8_PROFILE')
    if profile and profile != release.profile:
        raise ValueError(f'EXLR8_PROFILE must be {release.profile} for EXLR8_RELEASE={release.name}')
    return release


def require_revision(revision, release):
    if revision and revision != release.revision:
        raise ValueError(f'EXLR8_RELEASE={release.name} requires revision {release.revision}')


def read_manifest(path, release):
    manifest = read_json(path)
    if manifest['set_id'] != release.set_id:
        raise ValueError(f'{path}: manifest set ID differs from the pinned {release.profile} profile')
    return manifest


def index_hashes(release):
    return read_json(Path(__file__).resolve().with_name(release.index_sha))


def check_index(path, hashes):
    path = Path(path)
    if sha(path) != hashes[path.name]:
        raise ValueError(f'{path}: SHA-256 differs from the selected release')


def check_expert_codebook(path, header, release):
    declared = header.get('__metadata__', {}).get('codebook', '')
    codebook = (declared.split() or ['mcg'])[0]
    trellis = [m[3] for name in header if (m := re.fullmatch(r'(shared\.)?trellis\.(gate|up|down)(\.[^.]+)?', name))]
    suffix = None if release.codebook == 'mcg' else f'.{release.codebook}'
    if codebook != release.codebook or not trellis or any(name != suffix for name in trellis):
        raise ValueError(f'{path}: expert codebook or tensor names differ from {release.codebook} for {release.name}')


def release_identity(directory, ranks, revision=None):
    release = selected_release()
    require_revision(revision, release)
    directory = Path(directory)
    receipt = read_json(directory / 'meta.json')['release']
    if receipt != release.receipt:
        raise ValueError(f'{directory}: source revision differs from EXLR8_RELEASE={release.name}')
    read_manifest(directory / release.manifest, release)
    return dict(format=SET_FORMAT, release=release.receipt, profile=release.set_id, profile_name=release.profile, ranks=ranks)


def validate_identity(identity, ranks, release):
    if identity['format'] != SET_FORMAT or identity['ranks'] != ranks:
        raise ValueError('invalid EXLR8 set identity (format or topology)')
    if (identity['release'], identity['profile_name'], identity['profile']) != (release.receipt, release.profile, release.set_id):
        raise ValueError(f'checkpoint identity differs from EXLR8_RELEASE={release.name} (revision or profile)')


def check_output_release(directory):
    """Refuse existing outputs for another release before allowing recovery or writes."""
    release = selected_release()
    directory = Path(directory)
    other_leaves = {leaf for other in RELEASES.values() if other != release
                    for leaf in (other.interim_leaf, other.native_leaf)}
    if directory.name in other_leaves or directory.resolve().name in other_leaves:
        raise ValueError(f'{directory}: output path belongs to a different EXLR8 release')
    owned = directory / '.x8-owned'
    if owned.is_symlink():
        raise ValueError(f'{owned}: ownership marker is a symlink')
    if owned.is_file():
        receipt = owned.read_text().strip()
        if receipt and receipt != release.receipt:
            raise ValueError(f'{directory}: ownership marker differs from EXLR8_RELEASE={release.name}')
    set_path = directory / 'exlr8-set.json'
    if set_path.exists():
        identity = read_json(set_path)
        validate_identity(identity, identity['ranks'], release)


def claim_output(directory):
    check_output_release(directory)
    (Path(directory) / '.x8-owned').write_text(selected_release().receipt + '\n')


def check_counts(release, dense, files, tensors):
    expected = release.native_tensors if dense == 'native' else release.interim_tensors
    if files != release.files or tensors != expected:
        raise ValueError(f'{release.name} {dense}: expected {release.files} shard files and {expected} tensors, '
                         f'found {files} files and {tensors} tensors')


def protect_output(directory, *link_sources):
    """Reject production, its aliases and paths overlapping it, even for conversions without links."""
    output = Path(directory).absolute()
    resolved = output.resolve()
    interim = [Path.home() / release.interim_leaf for release in RELEASES.values()]
    for source in (*interim, *map(Path, filter(None, link_sources))):
        for protected in (source.absolute(), source.resolve()):
            if any(path == protected or path in protected.parents or protected in path.parents
                   for path in (output, resolved)):
                raise ValueError(f'output {output} overlaps protected directory {source}')
        if source.exists() and any(parent.exists() and parent.samefile(source)
                                   for parent in (resolved, *resolved.parents)):
            raise ValueError(f'output {output} aliases protected directory {source}')
    if output.is_symlink():
        raise ValueError(f'output {output} is a symlink')
    return resolved


def native_output(directory, *link_sources):
    output = protect_output(directory, *link_sources)
    if output.exists():
        st = output.stat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
            raise ValueError(f'{output} is not a directory owned by this user')
        names = {'.x8-owned', '.x12-native'}
        markers = {name for name in names if (output / name).is_file() and not (output / name).is_symlink()}
        initializing = not st.st_mode & 0o077 and {p.name for p in output.iterdir()} <= markers
        if markers != names and not initializing:
            raise ValueError(f'{output} is not an owned native conversion directory')
    check_output_release(output)
    reject_mounts(output)
    return output


def reject_mounts(output):
    with open('/proc/self/mountinfo') as mounts:
        for line in mounts:
            mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), line.split()[4]))
            if mount == output or output in mount.parents:
                raise ValueError(f'{output} is or contains a mount point ({mount})')


def reclaimable(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            st = (Path(root) / name).lstat()
            if stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
                total += st.st_blocks * 512
    return total


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()


def tools(d):
    t = {n: sha(os.path.join(d, n)) for n in TOOLS}
    if os.environ.get('EXLR8_DENSE') == 'native':  # patch 0212: exlr8_native.py (which imports exlr8_dense.py) wrote the dense files
        h = hashlib.sha256()
        for n in ('exlr8_native.py', 'exlr8_dense.py'):
            h.update(open(os.path.join(d, n), 'rb').read())
        t['exlr8_dense.py'] = h.hexdigest()  # the key exlr8_assemble.py requires; the value names both converters
    return t


def state(d, rank, ranks):
    check_output_release(d)
    if not os.path.exists(d) or (os.path.isdir(d) and not os.listdir(d)):
        return 'new'
    if not os.path.exists(os.path.join(d, '.x8-owned')):
        return 'foreign'
    mp = os.path.join(d, '.x8-complete')
    if not os.path.exists(mp):
        return 'partial'
    try:
        m = json.load(open(mp))
        if m['rank'] != rank or m['ranks'] != ranks:
            return 'partial'
        for f, size in m['sizes'].items():
            p = os.path.join(d, f)
            st = os.stat(p)
            if st.st_size != size or st.st_mtime_ns != m['mtimes'][f]:
                return 'partial'
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return 'partial'
    return 'complete'


def repair(d):
    c = json.load(open(os.path.join(d, 'config.json')))
    p = os.path.join(d, 'exlr8-set.json')
    s = json.load(open(p)) if os.path.exists(p) else None
    by = json.load(open(os.path.join(d, '.x8-converted-by')))
    quant = c.get('quantization_config', {})
    return ('repair' if 'rank' in quant or quant.get('source_licensing') != SOURCE_LICENSING
            or s is None or s.get('tools') != by else 'ok')


def verify(d, rank, ranks, marker='.x8-complete'):
    m = read_json(Path(d) / marker)
    if m['rank'] != rank or m['ranks'] != ranks:
        print('BAD marker rank')
        return 1
    if 'sha256' not in m:
        print('legacy marker: sizes and mtimes only')
        return 0
    bad = [f for f, v in m['sha256'].items() if sha(os.path.join(d, f)) != v]
    print('BAD', bad[:5]) if bad else print('bytes ok', len(m['sha256']))
    return 1 if bad else 0


def checkpoint(directory, rank, ranks, dense, *, pending_layers=None):
    """Check coverage and metadata against this rank's receipt. Weight bytes are hashed at conversion time."""
    release = selected_release()
    d = Path(directory)
    check_output_release(d)
    shards = set(shard_names(rank, ranks, *(pending_layers or (0, 77))))
    expected = shards | set(METADATA)
    marker = '.x8-complete.pending' if pending_layers is not None else '.x8-complete'
    if not (d / '.x8-owned').is_file():
        raise ValueError('missing .x8-owned')
    m = read_json(d / marker)
    if m['rank'] != rank or m['ranks'] != ranks or m['files'] != len(shards):
        raise ValueError('marker rank, topology or shard count differs')
    if dense == 'native' and pending_layers is None and m.get('verified') is not True:
        raise ValueError('native conversion needs a verified completion marker')
    if pending_layers is None and (d / '.x8-complete.pending').exists():
        raise ValueError('checkpoint has a pending marker')
    if m.get('dense', 'interim') != dense:
        raise ValueError(f'marker dense mode differs from {dense}')
    if any(m[name].keys() != expected for name in ('sizes', 'mtimes', 'sha256')):
        raise ValueError('marker file coverage differs from the expected shards and metadata')
    for name in expected:
        st = (d / name).stat()
        if st.st_size != m['sizes'][name] or st.st_mtime_ns != m['mtimes'][name] or not hex_digest(m['sha256'][name]):
            raise ValueError(f'{name} differs from the marker')
    for name in ('config.json', 'exlr8-set.json', 'model.safetensors.index.json'):
        if sha(d / name) != m['sha256'][name]:
            raise ValueError(f'{name} digest differs from the marker')
    identity = read_json(d / 'exlr8-set.json')
    validate_identity(identity, ranks, release)
    if read_json(d / 'config.json')['quantization_config']['codebook'] != release.marker:
        raise ValueError(f'config codebook differs from {release.marker}')
    if identity.get('dense', 'interim') != dense:
        raise ValueError(f'set dense mode differs from {dense}')
    provenance = identity['tools']
    if (set(provenance) != set(TOOLS) or not all(map(hex_digest, provenance.values()))
            or provenance != read_json(d / '.x8-converted-by')):
        raise ValueError('invalid converter provenance')
    index = read_json(d / 'model.safetensors.index.json')
    weights = index['weight_map']
    if pending_layers in (None, (0, 77)):
        check_counts(release, dense, m['files'], m['tensors'])
    if (index['metadata'] != {'rank': rank, 'ranks': ranks} or len(weights) != m['tensors']
            or set(weights.values()) != shards):
        raise ValueError('index rank, topology or shard coverage differs')
    if pending_layers is None or pending_layers[0] <= 3 <= pending_layers[1]:
        proj = 'model.layers.3.self_attn.q_b_proj'
        suffix = 'trellis' if dense == 'native' else 'weight'
        if weights.get(f'{proj}.{suffix}') != f'exlr8-dense-003.r{rank}of{ranks}.safetensors':
            raise ValueError(f'layer 3 q_b does not use {dense} storage')
    return m, identity


def match_release(identity, release, ranks):
    for field, value in release_identity(release, ranks).items():
        if identity.get(field) != value:
            raise ValueError(f'checkpoint {field} differs from the requested release')


def match_piece_source(identity, source):
    for key in ('format', 'release', 'profile', 'profile_name', 'ranks'):
        if identity[key] != source[key]:
            raise ValueError(f'interim checkpoint {key} differs from the native set')
    if source['tools']['exlr8_pieces.py'] != identity['tools']['exlr8_pieces.py']:
        raise ValueError('pieces converter differs from the interim marker')


def compare_pieces(marker, source):
    pieces = [name for name in marker['sha256'] if name.startswith('exlr8-exl3-layer-')]
    if any(marker['sha256'][name] != source['sha256'].get(name) for name in pieces):
        raise ValueError('linked piece digests differ from the interim marker')


def publish(directory, rank, ranks, release, link_source, lo, hi):
    d = Path(directory)
    if (d / '.x8-complete').exists():
        raise ValueError('a completion marker already exists')
    marker, identity = checkpoint(d, rank, ranks, 'native', pending_layers=(lo, hi))
    match_release(identity, release, ranks)
    if link_source:
        source, source_identity = checkpoint(link_source, rank, ranks, 'interim')
        match_piece_source(identity, source_identity)
        compare_pieces(marker, source)
    pending = d / '.x8-complete.pending'
    if verify(d, rank, ranks, pending.name):
        raise ValueError('native checkpoint bytes differ from its pending marker')
    if (lo, hi) != (0, 77):
        print(f'{d}: verified layers {lo}-{hi}; marker stays pending')
        return
    marker['verified'] = True
    tmp = d / '.x8-complete.tmp'
    with tmp.open('x') as file:
        json.dump(marker, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, pending)
    os.replace(pending, d / '.x8-complete')
    fd = os.open(d, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checkpoint_command(args):
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('rank', type=int)
    ap.add_argument('ranks', type=int)
    ap.add_argument('dense', choices=('interim', 'native'))
    ap.add_argument('--release')
    ap.add_argument('--revision')
    ap.add_argument('--tools')
    ap.add_argument('--pieces-tool')
    ap.add_argument('--link-source')
    a = ap.parse_args(args)
    marker, identity = checkpoint(a.directory, a.rank, a.ranks, a.dense)
    if a.release:
        match_release(identity, a.release, a.ranks)
    require_revision(a.revision, selected_release())
    if a.tools and identity['tools'] != json.loads(a.tools):
        raise ValueError('checkpoint was written by other converters')
    if a.pieces_tool and identity['tools']['exlr8_pieces.py'] != sha(a.pieces_tool):
        raise ValueError('pieces converter differs from the one that wrote the interim pieces')
    if a.link_source:
        source, source_identity = checkpoint(a.link_source, a.rank, a.ranks, 'interim')
        match_piece_source(identity, source_identity)
        compare_pieces(marker, source)
    print(marker['sha256']['config.json'], marker['sha256']['exlr8-set.json'])


def main():
    a = sys.argv[1:]
    if a == ['settings']:
        release = selected_release()
        print(release.name, release.revision, release.profile, release.manifest,
              release.interim_leaf, release.native_leaf, release.index_sha, release.served_name, REPO)
    elif len(a) == 2 and a[0] == 'manifest':
        read_manifest(a[1], selected_release())
    elif len(a) == 2 and a[0] == 'claim':
        claim_output(a[1])
    elif len(a) == 4 and a[0] == 'state':
        print(state(a[1], int(a[2]), int(a[3])))
    elif len(a) == 2 and a[0] == 'repair':
        print(repair(a[1]))
    elif len(a) == 4 and a[0] == 'verify':
        return verify(a[1], int(a[2]), int(a[3]))
    elif len(a) == 2 and a[0] == 'tools':
        print(json.dumps(tools(a[1]), sort_keys=True))
    elif len(a) >= 2 and a[0] == 'native-output':
        print(native_output(*a[1:]))
    elif len(a) == 2 and a[0] == 'native-reclaimable':
        print(reclaimable(native_output(a[1])))
    elif len(a) >= 2 and a[0] == 'protect-output':
        print(protect_output(*a[1:]))
    elif len(a) == 4 and a[0] == 'release':
        release_identity(a[1], int(a[2]), a[3])
    elif a and a[0] == 'checkpoint':
        checkpoint_command(a[1:])
    elif len(a) == 8 and a[0] == 'publish-native':
        d, rank, ranks, release, source, lo, hi = a[1:]
        publish(d, int(rank), int(ranks), release, source, int(lo), int(hi))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        sys.exit(f'EXLR8: {error}')
