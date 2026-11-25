#!/usr/bin/env python3
"""Fetch what one rank of kindlingai's EXLR8 release (kindlingai/glm-5.3-exlr8-k2-k3-k4, format exlr8-v2) needs
to be converted for TensorFold, and nothing else. Standard library only. The release's layout is kept, so the
converters (exlr8_pieces.py, exlr8_dense.py, exlr8_assemble.py) read it as they read any complete copy.

  fetch.py index   INDEX_DIR   REVISION                      the 76 experts/exlr8-layer-LLL.index.safetensors (13 MB each),
                                                             each checked against the selected release's hash list
  fetch.py small   RELEASE_DIR REVISION                      only meta.json, the manifest and the tokenizer files
  fetch.py release RELEASE_DIR INDEX_DIR RANK RANKS REVISION  meta.json, the selected release's manifest, the tokenizer
                                                             files, dense/ (layers 0-77 and the vocabulary, whole files)
                                                             and, in experts/, sparse copies of the K-width files holding
                                                             only the byte ranges of this rank's halves

A rank's halves follow the release's sharding rule: half h of expert e in layer L belongs to rank (16 e + h + L) mod
RANKS (the shared expert is expert 256). Each half is one range, whose offset and length the layer's index file gives;
the width of each half comes from the selected profile's manifest. The converters check the expert codebook before
reading the downloaded ranges.

Idempotent and resumable: every whole file is written as NAME.part, size-checked and renamed; every sparse expert
file's ranges are recorded in RELEASE_DIR/.fetch/ once all of them landed, and a rerun skips what is recorded. A rank's
fetch ends with RELEASE_DIR/.fetch/complete-rank{R}of{N}.json, which lists every file with its size. Nothing is written
outside RELEASE_DIR / INDEX_DIR. No token is sent (the repository is public).
"""
import hashlib, http.client, json, os, struct, sys, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exlr8_pieces import ST, i64s, rank_halves, HALVES, ROUTED, SHARED  # noqa: E402  (stdlib only; main() is guarded)

from dirstate import REPO, check_index, index_hashes, read_json, read_manifest, require_revision, selected_release
LAYERS = range(3, 78)
if os.environ.get('EXLR8_FETCH_LAYERS'):  # tests only: 'lo-hi' of the routed layers
    _lo, _hi = (int(x) for x in os.environ['EXLR8_FETCH_LAYERS'].split('-'))
    LAYERS = range(_lo, _hi + 1)
HOST = 'https://huggingface.co'
UA = 'glm53-recipe-exlr8-fetch/1'
THREADS = int(os.environ.get('EXLR8_FETCH_THREADS', '32'))  # connections; a range read answers in 0.3-1 s


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def api_tree(rev, path=''):
    url = f'{HOST}/api/models/{REPO}/tree/{rev}' + (f'/{path}' if path else '')
    out = {}
    while url:
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            for e in json.load(r):
                if e['type'] == 'file':
                    out[e['path']] = e['size']
            link = r.headers.get('Link', '')
        url = None
        for part in link.split(','):
            if 'rel="next"' in part:
                url = part.split(';')[0].strip().strip('<>')
    return out


def resolve_url(rev, path):
    return f'{HOST}/{REPO}/resolve/{rev}/{path}'


def retry(fn, what, tries=8):
    for i in range(tries):
        try:
            return fn()
        except (OSError, http.client.HTTPException) as e:
            if i == tries - 1:
                raise SystemExit(f'{what}: {e!r} after {tries} tries')
            time.sleep(min(60, 2 ** i))


def download_whole(rev, path, dest, size):
    """The repository file `path` as `dest`, `size` bytes: skipped when dest has that size; resumed from dest.part."""
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'

    def once():
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        if have < size:
            req = urllib.request.Request(resolve_url(rev, path), headers={'User-Agent': UA, 'Range': f'bytes={have}-'})
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:
                    os.remove(part)
                    raise OSError('server ignored the Range header')
                with open(part, 'ab') as f:
                    for blk in iter(lambda: r.read(1 << 20), b''):
                        f.write(blk)
        if os.path.getsize(part) != size:
            raise OSError(f'{path}: {os.path.getsize(part)} bytes, expected {size}')

    retry(once, path)
    os.replace(part, dest)
    return True


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()


# ---- the layer index files ------------------------------------------------------------------------------------------------
def cmd_index(index_dir, rev):
    release = selected_release()
    require_revision(rev, release)
    rev = release.revision
    want = index_hashes(release)
    sizes = api_tree(rev, 'experts')
    edir = os.path.join(index_dir, 'experts')
    os.makedirs(edir, exist_ok=True)

    def one(name):
        p = os.path.join(edir, name)
        if os.path.exists(p) and sha256_file(p) == want[name]:
            return 0
        if os.path.exists(p):  # a different file under the name: never overwritten here
            raise SystemExit(f'{p} exists and is not the hub\'s {name} (sha256): remove it to fetch again')
        download_whole(rev, f'experts/{name}', p, sizes[f'experts/{name}'])
        if sha256_file(p) != want[name]:
            os.rename(p, p + '.bad')
            raise SystemExit(f'{name}: sha256 differs from {release.index_sha} (kept as {p}.bad)')
        return 1

    with ThreadPoolExecutor(THREADS) as ex:
        n = sum(ex.map(one, sorted(want)))
    log(f'index: {len(want)} files in {edir} ({n} fetched), sha256 checked')


# ---- the rank's halves -------------------------------------------------------------------------------------------------------
def plan_layer(man, idx, L, rank, ranks):
    """{k: sorted [(offset, length)]} of the half ranges this rank reads from layer L's K-width files, routed and shared
    experts, by the same index arithmetic as exlr8_pieces.py (routed) and exlr8_dense.py (shared)."""
    lw = man['layers'][str(L)]
    need = {}
    tables = {}

    def index(pre, k):
        if (pre, k) not in tables:
            tables[(pre, k)] = i64s(idx.raw(f'{pre}k{k}.index'))
        return tables[(pre, k)]

    for e in range(ROUTED + 1):
        pre = 'shared.' if e == SHARED else ''
        mine = rank_halves(e, L, rank, ranks)
        for pi, proj in enumerate(('gate', 'up', 'down')):
            if e == SHARED:
                w = lw['shared'][proj]
                ks = list(w) if isinstance(w, list) else [w] * HALVES
            else:
                ks = list(lw[proj][e]) if proj != 'down' else [lw['down'][e]] * HALVES
            for h in mine:
                k = int(ks[h])
                ix = index(pre, k)
                at = ((0 if e == SHARED else e) * 3 + pi) * 16 * 2 + h * 2
                need.setdefault(k, set()).add((ix[at], ix[at + 1]))
    return {k: sorted(v) for k, v in need.items()}


class Ranged:
    """Range reads of one repository file over a kept-alive connection to the CDN host its resolve URL redirects to."""

    def __init__(self, rev, path):
        self.rev, self.path, self.conn, self.target = rev, path, None, None

    def _open(self):
        req = urllib.request.Request(resolve_url(self.rev, self.path), headers={'User-Agent': UA, 'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            final = r.geturl()
            r.read()
        u = urllib.parse.urlsplit(final)
        self.target = u.path + ('?' + u.query if u.query else '')
        self.conn = (http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection)(u.netloc, timeout=120)

    def read_into(self, f, off, n):
        """Bytes [off, off + n) of the file, written at the same offset in f; exactly n bytes or an error."""
        got = 0
        while got < n:
            try:
                if self.conn is None:
                    self._open()
                self.conn.request('GET', self.target, headers={'Range': f'bytes={off + got}-{off + n - 1}', 'User-Agent': UA})
                r = self.conn.getresponse()
                if r.status != 206:
                    r.read()
                    raise OSError(f'HTTP {r.status} for a range read')
                cr = r.headers.get('Content-Range', '')  # bytes a-b/total
                if not cr.startswith(f'bytes {off + got}-'):
                    raise OSError(f'Content-Range {cr!r} for a read at {off + got}')
                f.seek(off + got)
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    if got + len(blk) > n:
                        raise OSError('more bytes than asked for')
                    f.write(blk)
                    got += len(blk)
            except (OSError, http.client.HTTPException):
                try:
                    self.conn and self.conn.close()
                except OSError:
                    pass
                self.conn = None  # a new resolve (a presigned URL expires) and connection
                raise


class ExpertFile:
    """One sparse K-width file and the ranges of this rank's halves in it, fetched in chunks by the shared thread pool."""
    CHUNK = 48  # halves a task fetches over one connection

    def __init__(self, rev, rel, L, k, ranges, total, rank, ranks):
        self.rev, self.ranges, self.total = rev, ranges, total
        self.name = f'exlr8-layer-{L:03d}.k{k}.safetensors'
        self.dest = os.path.join(rel, 'experts', self.name)
        self.ledger = os.path.join(rel, '.fetch', f'{self.name}.r{rank}of{ranks}.json')
        self.want = {'ranges': [list(r) for r in ranges], 'size': total, 'header': True}
        self.bytes = sum(n for _, n in ranges)
        self.lock, self.left = threading.Lock(), 0

    def done(self):
        return (os.path.exists(self.ledger) and os.path.exists(self.dest) and os.path.getsize(self.dest) == self.total
                and json.load(open(self.ledger)) == self.want)

    def tasks(self):
        """Prepares the file (a ledger left by an earlier, different plan is dropped) and returns its chunk tasks."""
        os.makedirs(os.path.dirname(self.dest), exist_ok=True)
        os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
        if os.path.exists(self.ledger):
            os.remove(self.ledger)
        with open(self.dest, 'r+b' if os.path.exists(self.dest) else 'w+b') as f:
            if os.fstat(f.fileno()).st_size != self.total:
                f.truncate(self.total)  # sparse: holes where the other ranks' halves are
        chunks = [self.ranges[i:i + self.CHUNK] for i in range(0, len(self.ranges), self.CHUNK)]
        chunks.append('header')  # the safetensors header (8 bytes of length, then the JSON): the converters parse it
        self.left = len(chunks)
        return [(self, c) for c in chunks]

    def read(self, src, f, off, n):
        for attempt in range(8):
            try:
                return src.read_into(f, off, n)
            except (OSError, http.client.HTTPException) as e:
                if attempt == 7:
                    raise SystemExit(f'{self.name} range {off}+{n}: {e!r}')
                time.sleep(min(60, 2 ** attempt))

    def run(self, chunk):
        src = Ranged(self.rev, f'experts/{self.name}')
        with open(self.dest, 'r+b') as f:
            if chunk == 'header':
                self.read(src, f, 0, 8)
                f.flush()
                f.seek(0)
                n = struct.unpack('<Q', f.read(8))[0]
                if not 0 < n < 64 << 20:
                    raise SystemExit(f'{self.name}: header length {n}')
                self.read(src, f, 8, n)
                chunk = [(0, 8 + n)]
            else:
                for off, n in chunk:
                    self.read(src, f, off, n)
        with self.lock:
            self.left -= 1
            last = self.left == 0
        if last:  # every chunk landed: flush the file, then the ledger that vouches for it
            fd = os.open(self.dest, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
            tmp = self.ledger + '.tmp'
            json.dump(self.want, open(tmp, 'w'))
            os.replace(tmp, self.ledger)
        return sum(n for _, n in chunk)


def fetch_small(rel, rev, sizes, release):
    """meta.json, the profile's manifest and the tokenizer files: what exlr8_assemble.py reads. Returns the manifest."""
    tok = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')
    manp = release.manifest
    mp = os.path.join(rel, 'meta.json')
    if os.path.exists(mp) and read_json(mp)['release'] != release.receipt:
        raise ValueError(f'{rel}: cache belongs to a different EXLR8 revision')
    download_whole(rev, manp, os.path.join(rel, manp), sizes[manp])
    manifest = read_manifest(os.path.join(rel, manp), release)
    for t in tok:
        download_whole(rev, t, os.path.join(rel, 'tokenizer', t), sizes[t])
    meta = {'release': release.receipt, 'files': list(tok)}
    if not os.path.exists(mp) or read_json(mp) != meta:
        json.dump(meta, open(mp + '.tmp', 'w'), indent=1)
        os.replace(mp + '.tmp', mp)
    return manifest, meta


def cmd_small(rel, rev):
    release = selected_release()
    require_revision(rev, release)
    rev = release.revision
    os.makedirs(rel, exist_ok=True)
    sizes = api_tree(rev)
    sizes.update(api_tree(rev, 'manifests'))
    fetch_small(rel, rev, sizes, release)
    log(f'small files in {rel}')


def cmd_release(rel, index_dir, rank, ranks, rev):
    release = selected_release()
    require_revision(rev, release)
    rev = release.revision
    if ranks not in (1, 2, 4, 8, 16) or not 0 <= rank < ranks:
        raise SystemExit('ranks must divide 16 and rank be below it')
    os.makedirs(rel, exist_ok=True)
    root = api_tree(rev)
    sizes = dict(root)
    for d in ('manifests', 'dense', 'experts'):
        sizes.update(api_tree(rev, d))
    man, meta = fetch_small(rel, rev, sizes, release)
    hashes = index_hashes(release)
    # dense: layers 0-77 and the vocabulary (layer 78, the MTP layer, is not converted: DFlash2 drafts)
    dense = [f'dense/exlr8-dense-{L:03d}.safetensors' for L in range(78)] + ['dense/exlr8-vocab.safetensors']
    if os.environ.get('EXLR8_FETCH_LAYERS'):
        dense = dense[:0]  # tests only: no dense files
    with ThreadPoolExecutor(THREADS) as ex:
        n = sum(ex.map(lambda p: download_whole(rev, p, os.path.join(rel, p), sizes[p]), dense))
    log(f'dense: {len(dense)} files ({n} fetched)')
    # the rank's halves of the expert files
    jobs = []
    for L in LAYERS:
        ix = os.path.join(index_dir, 'experts', f'exlr8-layer-{L:03d}.index.safetensors')
        check_index(ix, hashes)
        idx = ST(ix)
        if idx.meta.get('format') != 'exlr8-v2' or idx.meta.get('contents') != 'index':
            raise SystemExit(f'{ix}: not an exlr8-v2 index')
        for k, ranges in plan_layer(man, idx, L, rank, ranks).items():
            jobs.append(ExpertFile(rev, rel, L, k, ranges, sizes[f'experts/exlr8-layer-{L:03d}.k{k}.safetensors'], rank, ranks))
    total_bytes = sum(j.bytes for j in jobs)
    log(f'experts: rank {rank} of {ranks}: {len(jobs)} files, {sum(len(j.ranges) for j in jobs)} halves, {total_bytes / 2**30:.1f} GiB')
    todo = [j for j in jobs if not j.done()]
    got, t0 = [sum(j.bytes for j in jobs if j not in todo)], time.time()
    tasks = [t for j in todo for t in j.tasks()]
    log(f'experts: {len(jobs) - len(todo)} files already complete, {len(tasks)} chunks to fetch on {THREADS} connections')
    lock, nlast = threading.Lock(), [0]

    def run(t):
        n = t[0].run(t[1])
        with lock:
            got[0] += n
            nlast[0] += 1
            if nlast[0] % 200 == 0 or nlast[0] == len(tasks):
                log(f'experts: {nlast[0]}/{len(tasks)} chunks, {got[0] / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB, '
                    f'{(got[0]) / max(time.time() - t0, 1) / 1e6:.0f} MB/s')

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(run, tasks))
    files = {p: os.path.getsize(os.path.join(rel, p)) for p in dense}
    files.update({f'experts/{j.name}': j.total for j in jobs})
    comp = os.path.join(rel, '.fetch', f'complete-rank{rank}of{ranks}.json')
    os.makedirs(os.path.dirname(comp), exist_ok=True)
    json.dump({'release': meta['release'], 'profile': release.profile, 'rank': rank, 'ranks': ranks, 'files': files}, open(comp + '.tmp', 'w'))
    os.replace(comp + '.tmp', comp)
    log(f'release: rank {rank} of {ranks} complete ({comp})')


def main():
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == 'index':
        return cmd_index(a[1], a[2])
    if len(a) == 3 and a[0] == 'small':
        return cmd_small(a[1], a[2])
    if len(a) == 6 and a[0] == 'release':
        return cmd_release(a[1], a[2], int(a[3]), int(a[4]), a[5])
    raise SystemExit(__doc__)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError) as error:
        sys.exit(f'EXLR8: {error}')
