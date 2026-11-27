#!/usr/bin/env bash
# Exercise EXLR8 conversion receipts and launch preflight with tiny checkpoint fixtures.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
tools=$PWD/scripts/exlr8
export PYTHONDONTWRITEBYTECODE=1
unset EXLR8_RELEASE EXLR8_REV EXLR8_PROFILE EXLR8_DENSE
python3 -B - "$tmp" "$tools" <<'PY'
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

root, tools = map(Path, sys.argv[1:])
sys.path.insert(0, str(tools))
import dirstate as ds

def dump(path, value):
    path.write_text(json.dumps(value))

def safetensor(path, name, data=b''):
    header = json.dumps({name: {'dtype': 'U8', 'shape': [len(data)], 'data_offsets': [0, len(data)]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + data)

def release(path, spec, *, set_id=None):
    (path / 'manifests').mkdir(parents=True)
    (path / 'tokenizer').mkdir()
    (path / 'dense').mkdir()
    dump(path / 'meta.json', {'release': spec.receipt})
    dump(path / spec.manifest, {'set_id': set_id or spec.set_id})
    for name in ds.METADATA:
        if name not in ('exlr8-set.json', 'model.safetensors.index.json'):
            (path / 'tokenizer' / name).write_text('{}')

def assemble(path, rank, mode, layers='0-77', *, source=None, key='up1', underfill=False):
    spec = ds.RELEASES[key]
    lo, hi = map(int, layers.split('-'))
    path.mkdir()
    (path / '.x8-owned').write_text(spec.receipt + '\n')
    if mode == 'native':
        (path / '.x12-native').touch()
    dump(path / '.x8-converted-by', ds.tools(tools))
    dense = f'exlr8-dense-003.r{rank}of4.safetensors'
    files = ds.shard_names(rank, 4, lo, hi)
    for i, name in enumerate(files):
        tensor = f'tensor_{i}'
        if name == dense:
            suffix = 'trellis' if mode == 'native' else 'weight'
            tensor = f'model.layers.3.self_attn.q_b_proj.{suffix}'
        if name.startswith('exlr8-exl3-layer-'):
            tensor += '.' + spec.marker
        safetensor(path / name, tensor)
    if layers == '0-77' and not underfill:
        # Assemble reads header names. Use zero-byte entries to exercise full index counts without model weights.
        count = (232977 if mode == 'native' else 232722) - len(files)
        header = json.dumps({**dict.fromkeys((f'padding_{i}' for i in range(count))),
                             f'tensor_{len(files) - 1}': None}).encode()
        (path / 'exlr8-vocab.safetensors').write_bytes(struct.pack('<Q', len(header)) + header)
    cmd = [sys.executable, '-IB', str(tools / 'exlr8_assemble.py'), str(source or rel),
           str(path), str(rank), '4', '--layers', layers]
    return subprocess.run(cmd, env={**os.environ, 'EXLR8_RELEASE': key}, check=not underfill,
                          capture_output=True, text=True)

def refused(fn, message):
    try:
        fn()
    except (ValueError, OSError) as error:
        assert message in str(error), error
    else:
        raise AssertionError(f'expected refusal containing {message!r}')

rel = root / 'release with spaces'
up1, v07 = (ds.RELEASES[key] for key in ('up1', 'v07'))
release(rel, up1)
other_rel = root / 'v07 release'
release(other_rel, v07)
other_profile = root / 'different profile'
release(other_profile, up1, set_id='c' * 64)

for r in range(4):
    assemble(root / f'interim {r}', r, 'interim')
    ds.checkpoint(root / f'interim {r}', r, 4, 'interim')
    assemble(root / f'v07 interim {r}', r, 'interim', source=other_rel, key='v07')
# Conversions from before the release selector have empty ownership markers.
(root / 'v07 interim 0' / '.x8-owned').write_text('')
short = assemble(root / 'too few tensors', 0, 'interim', underfill=True)
assert short.returncode and '232722 tensors' in short.stderr
assert not (root / 'too few tensors' / '.x8-complete').exists()

partial = root / 'partial'
assemble(partial, 0, 'native', '3-3')
assert not (partial / '.x8-complete').exists()
ds.publish(partial, 0, 4, rel, '', 3, 3)
assert (partial / '.x8-complete.pending').is_file()
assert not (partial / '.x8-complete').exists()
refused(lambda: ds.checkpoint(partial, 0, 4, 'native'), '.x8-complete')

native = root / 'native'
assemble(native, 0, 'native')
refused(lambda: ds.checkpoint(native, 0, 4, 'native'), '.x8-complete')
ds.publish(native, 0, 4, rel, root / 'interim 0', 0, 77)
_, identity = ds.checkpoint(native, 0, 4, 'native')
refused(lambda: ds.checkpoint(native, 0, 4, 'interim'), 'dense mode')
refused(lambda: ds.checkpoint(native, 1, 4, 'native'), 'marker rank')
refused(lambda: ds.match_release(identity, other_rel, 4), 'source revision differs')
refused(lambda: ds.match_release(identity, other_profile, 4), 'manifest set ID differs')
refused(lambda: ds.match_release(identity, rel, 8), 'ranks differs')
refused(lambda: ds.checkpoint(root / 'v07 interim 0', 0, 4, 'interim'), 'EXLR8_RELEASE=up1')
refused(lambda: ds.state(root / 'v07 interim 0', 0, 4), 'EXLR8_RELEASE=up1')

other_native = root / 'v07 native'
assemble(other_native, 0, 'native', source=other_rel, key='v07')
os.environ['EXLR8_RELEASE'] = 'v07'
refused(lambda: ds.publish(other_native, 0, 4, other_rel, root / 'interim 0', 0, 77), 'EXLR8_RELEASE=v07')
assert not (other_native / '.x8-complete').exists()
ds.publish(other_native, 0, 4, other_rel, root / 'v07 interim 0', 0, 77)
ds.checkpoint(other_native, 0, 4, 'native')
refused(lambda: ds.checkpoint(native, 0, 4, 'native'), 'EXLR8_RELEASE=v07')
refused(lambda: ds.state(partial, 0, 4), 'EXLR8_RELEASE=v07')
os.environ.pop('EXLR8_RELEASE')

corrupt = root / 'corrupt'
assemble(corrupt, 0, 'native')
fn = corrupt / 'exlr8-exl3-layer-003.r0of4.safetensors'
st = fn.stat()
fn.write_bytes(b'z' + fn.read_bytes()[1:])
os.utime(fn, ns=(st.st_atime_ns, st.st_mtime_ns))
refused(lambda: ds.publish(corrupt, 0, 4, rel, '', 0, 77), 'bytes differ')
assert not (corrupt / '.x8-complete').exists()

# Fake home and download inputs keep all tests independent of real checkpoints and releases.
home = root / 'fake home'
home.mkdir()
for key, source in (('up1', 'interim 0'), ('v07', 'v07 interim 0')):
    shutil.copytree(root / source, home / f'exlr8-{key}', copy_function=shutil.copy2)
(home / 'alias').symlink_to(home / 'exlr8-v07', target_is_directory=True)
index = root / 'index with spaces' / 'experts'
index.mkdir(parents=True)
(index / 'exlr8-layer-003.index.safetensors').touch()
for path, spec in ((rel, up1), (other_rel, v07)):
    safetensor(path / 'dense/exlr8-dense-002.safetensors', 'input_layernorm.weight', b'x')
    header, data = {}, bytearray()
    def tensor(name, dtype, shape, value):
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(data), len(data) + len(value)]}
        data.extend(value)

    for name in ('embed_tokens', 'lm_head'):
        tensor(name + '.record', 'I64', [8], struct.pack('<8q', spec.vocab_format, 1, 2, 0, 0, 0, 0, 0))
        if spec.vocab_format == 5:
            tensor(name + '.fp8', 'F8_E4M3', [1, 2], bytes((0x38, 0x40)))
            tensor(name + '.scale', 'F32', [1], struct.pack('<f', 2.0))
        else:
            tensor(name + '.bf16', 'BF16', [1, 2], struct.pack('<HH', 0x3f80, 0x4000))
    encoded = json.dumps(header).encode()
    (path / 'dense/exlr8-vocab.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)

# Same release, changed config. Its marker is valid in isolation, but differs from rank 0's.
(rel / 'tokenizer/config.json').write_text('{"different": true}')
assemble(root / 'wrong config 3', 3, 'interim')
(rel / 'tokenizer/config.json').write_text('{}')
print('exlr8-native: pending, full coverage, byte verification and source identity: pass')
PY

FIXTURES=$tmp bash -c '
  export MODEL_FORMAT=exlr8 EXLR8_DENSE=interim MODEL_DIR="$FIXTURES/interim 0"
  source scripts/prepare.sh
  source scripts/tp6.sh
  wssh_n() {
    local i=$1
    shift
    [[ "$1" == env && "$2" == EXLR8_RELEASE="$EXLR8_RELEASE" && "$3" == python3 ]] || return 1
    if [[ "${FAILED_RANK:-}" == "$i" ]]; then return 124; fi
    local folder="${PREFIX:-}interim $i"
    if [[ "${WRONG_RANK:-}" == "$i" ]]; then folder="wrong config $i"; fi
    if [[ "${OTHER_RELEASE_RANK:-}" == "$i" ]]; then folder="v07 interim $i"; fi
    env -i PATH="$PATH" "$2" python3 -IB - checkpoint "$FIXTURES/$folder" "$i" 4 "$EXLR8_DENSE"
  }
  tp6_check_exlr8
  [[ "$EXLR8_SET_ID" =~ ^[0-9a-f]{64}$ ]]
  ( EXLR8_RELEASE=v07 MODEL_DIR="$FIXTURES/v07 interim 0" PREFIX="v07 " tp6_check_exlr8 )
  ( FAILED_RANK=3 tp6_check_exlr8 ) > "$FIXTURES/failed-rank.log" 2>&1 && exit 1
  grep -q "Rank 3" "$FIXTURES/failed-rank.log"
  ( WRONG_RANK=3 tp6_check_exlr8 ) > "$FIXTURES/wrong-rank.log" 2>&1 && exit 1
  grep -q "different EXLR8 config or checkpoint set" "$FIXTURES/wrong-rank.log"
  ( OTHER_RELEASE_RANK=3 tp6_check_exlr8 ) > "$FIXTURES/other-release-rank.log" 2>&1 && exit 1
  grep -q "EXLR8_RELEASE=up1" "$FIXTURES/other-release-rank.log"
  ( EXLR8_DENSE=native tp6_check_exlr8 ) > "$FIXTURES/wrong-mode.log" 2>&1 && exit 1
  grep -q "Rank 0 needs a complete native" "$FIXTURES/wrong-mode.log"
'
echo 'exlr8-native: all four launch ranks must agree: pass'

home="$tmp/fake home"
wrapper=(bash "$tools/convert-native.sh" --rank 0 --tools "$tools" --release "$tmp/release with spaces" --index-dir "$tmp/index with spaces")
for out in "$home/exlr8-up1" "$home/exlr8-v07" "$home/alias" "$home/exlr8-v07/child"; do
  if env HOME="$home" "${wrapper[@]}" --out "$out" --dry-run > "$tmp/guard.log" 2>&1; then exit 1; fi
  grep -q 'overlaps protected directory' "$tmp/guard.log"
done
echo 'exlr8-native: wrapper protects interim without a link source: pass'
if env HOME="$home" "${wrapper[@]}" --out "$tmp/unused" --x8 "$home/exlr8-up1" --release "$tmp/different profile" \
   --dry-run > "$tmp/link.log" 2>&1; then exit 1; fi
grep -q 'manifest set ID differs from the pinned up1 profile' "$tmp/link.log"
test ! -e "$tmp/unused"
echo 'exlr8-native: wrapper rejects a different release profile before linking: pass'
if env HOME="$home" "${wrapper[@]}" --out "$tmp/unused" --x8 "$tmp/v07 interim 0" \
   --dry-run > "$tmp/link.log" 2>&1; then exit 1; fi
grep -q 'EXLR8_RELEASE=up1' "$tmp/link.log"
test ! -e "$tmp/unused"

mkdir "$tmp/bin"
cat > "$tmp/bin/df" <<'MOCK'
#!/usr/bin/env bash
printf 'Avail\n%d\n' "$(( 25 * 1073741824 ))"
MOCK
cat > "$tmp/bin/python3" <<'MOCK'
#!/usr/bin/env bash
for arg; do
  case "$arg" in
    */fetch.py) touch "$FETCH_ATTEMPT"; exit 99 ;;
  esac
done
exec "$REAL_PYTHON" "$@"
MOCK
chmod +x "$tmp/bin/"*
box=(env HOME="$home" PATH="$tmp/bin:$PATH" REAL_PYTHON="$(command -v python3)" FETCH_ATTEMPT="$tmp/fetch-attempt"
     EXLR8_MIN_AVAIL_GIB=1)
for release in up1 v07; do
  if "${box[@]}" EXLR8_RELEASE="$release" EXLR8_DENSE=native bash "$tools/box.sh" --rank 0 > "$tmp/disk.log" 2>&1; then exit 1; fi
  grep -q '25 GiB free, needs 122 GiB' "$tmp/disk.log"
done
test ! -e "$tmp/fetch-attempt"
for entry in 'up1:v07 interim 1' 'v07:interim 0' 'v07:partial'; do
  release=${entry%%:*}; output="$tmp/${entry#*:}"
  before=$(sha256sum -- "$output/.x8-owned")
  if "${box[@]}" EXLR8_RELEASE="$release" EXLR8_DIR="$output" bash "$tools/box.sh" --rank 0 > "$tmp/cross.log" 2>&1; then exit 1; fi
  grep -q 'ownership marker differs' "$tmp/cross.log"
  [[ "$(sha256sum -- "$output/.x8-owned")" == "$before" ]]
  test ! -e "$tmp/fetch-attempt"
done
echo 'exlr8-native: box refuses both cross-release directions and pending outputs: pass'
if env HOME="$home" PATH="$tmp/bin:$PATH" REAL_PYTHON="$(command -v python3)" EXLR8_MIN_FREE_GB=26 EXLR8_MIN_AVAIL_GIB=1 \
   "${wrapper[@]}" --out "$tmp/tiny output" --layers 2-2 > "$tmp/wrapper-disk.log" 2>&1; then exit 1; fi
grep -q 'needs 26 GiB' "$tmp/wrapper-disk.log"
echo 'exlr8-native: disk gates precede writes and downloads: pass'

if python3 -B -c 'import numpy' 2>/dev/null; then
  # NumPy can be installed in the real user's site directory. The guard uses a fake home.
  numpy_path=$(python3 -B -c 'import numpy; from pathlib import Path; print(Path(numpy.__file__).resolve().parent.parent)')
  writer=(python3 -B "$tools/exlr8_native.py" "$tmp/release with spaces" "$tmp/index with spaces" 0 4)
  for out in "$home/exlr8-up1" "$home/exlr8-v07" "$home/alias" "$home/exlr8-v07/child"; do
    if env HOME="$home" PYTHONPATH="$numpy_path" "${writer[@]}" "$out" --layers 2-2 > "$tmp/guard.log" 2>&1; then exit 1; fi
    grep -q 'overlaps protected directory' "$tmp/guard.log"
  done
  mkdir -m 755 "$tmp/public stage"
  if "${writer[@]}" "$tmp/public stage" --layers 2-2 > "$tmp/guard.log" 2>&1; then exit 1; fi
  grep -q 'empty, private staging directory' "$tmp/guard.log"
  for release in up1 v07; do
    source_dir="$tmp/release with spaces"
    [[ "$release" != v07 ]] || source_dir="$tmp/v07 release"
    "${box[@]}" PYTHONPATH="$numpy_path" EXLR8_RELEASE="$release" EXLR8_MIN_FREE_GB=1 \
      "${wrapper[@]}" --release "$source_dir" --out "$tmp/tiny $release" --layers 2-2
    test -f "$tmp/tiny $release/.x8-complete.pending"
    test ! -e "$tmp/tiny $release/.x8-complete"
  done
  PYTHONPATH="$tools:$numpy_path" python3 -B - "$tmp" <<'PY'
import struct
import sys
from pathlib import Path

import check_pieces as cp
from dirstate import RELEASES, check_expert_codebook
from exlr8_dense import EXPERT_BOOKS
from exlr8_pieces import ST

for key, words in (('up1', (0x4000, 0x4080)), ('v07', (0x3f80, 0x4000))):
    f = ST(Path(sys.argv[1]) / f'tiny {key}' / 'exlr8-vocab.safetensors')
    assert f.raw('model.embed_tokens.weight') == struct.pack('<HH', *words)
    assert f.raw('lm_head.weight') == struct.pack('<HH', *words)
    release = RELEASES[key]
    book = EXPERT_BOOKS[release.codebook]
    scalar = cp.mcg if release.codebook == 'mcg' else cp.mul1
    for state in range(65536):
        rounded, = struct.unpack('<e', struct.pack('<e', scalar(state)))
        assert book[state] == cp.e4m3_val(cp.e4m3_from(rounded))

headers = {
    'up1': {'trellis.gate': {}},
    'v07': {'trellis.gate.mul1': {}, '__metadata__': {'codebook': 'mul1 0x83DCD12D'}},
}
for key, release in RELEASES.items():
    for source, header in headers.items():
        try:
            check_expert_codebook('fixture', header, release)
        except ValueError:
            assert key != source
        else:
            assert key == source
print('exlr8-native: vocabulary values, expert codebook guards and all 65,536 states per book: pass')
PY
  echo 'exlr8-native: Python writer guard and wrapper staging: pass'
else
  echo 'exlr8-native: NumPy unavailable, skipped writer execution'
fi

request_fn=$(sed -n '/^request_desc() {/,/^}/p' start-tp6-main.sh)
request() {  # EXLR8's request; the recipe's default format is k4
  env -i PATH="$PATH" HOME="$HOME" TP=4 TP6_RANKS=4 GLM_TP6=1 MODEL_FORMAT=exlr8 "$@" bash -c '
    source scripts/prepare.sh
    validate_config
    eval "$1"
    request_desc
  ' _ "$request_fn"
}
default=$(request)
[[ "$default" == *'EXLR8_DENSE=native;'* ]]
[[ "$default" == *'PARALLEL=2;'* ]]
[[ "$default" == *'TF_GLM_SEQ_GROW=1;'* ]]
[[ "$default" == *'TF_GLM_SEQ_GROW_STREAMS=2;'* ]]
[[ "$default" == *'TF_GLM_SEQ_GROW_FLOOR_MIB=2560;'* ]]
[[ "$default" == *'TF_GLM_DSA_SHARED=1;'* ]]
[[ "$(request PARALLEL=1)" == *'PARALLEL=1;'* ]]
[[ "$(request TF_GLM_SEQ_GROW=0 TF_GLM_DSA_SHARED=0)" == *'TF_GLM_SEQ_GROW=0;'**'TF_GLM_DSA_SHARED=0;'* ]]
for assignment in TF_GLM_X8_PROMPT_Q8=0 TF_GLM_SEQ_GROW=0 TF_GLM_DSA_SHARED=0 EXLR8_RELEASE=v07 EXLR8_DENSE=interim WORKER_MODEL_DIR=/tmp/other; do
  [[ "$(request "$assignment")" != "$default" ]]
done
[[ "$(request TF_GLM_X8_PROMPT_Q8=0)" == "$(request TF_GLM_X8_PROMPT_Q8=1 RANK_EXTRA_ENV=TF_GLM_X8_PROMPT_Q8=0)" ]]
[[ "$(request TF_GLM_SEQ_GROW=1 KV_SPLIT=1 RANK_EXTRA_ENV=TF_GLM_SEQ_GROW=0)" != "$default" ]]
for assignment in KV_SPLIT=1 PROMPT_CACHE_GIB=0.25 HOST_CACHE_GIB=auto 'SERVE_EXTRA_ARGS=--prompt-cache-gib 1' 'RANK_EXTRA_ENV=TF_KV_PAGES=on'; do
  if request TF_GLM_SEQ_GROW=1 "$assignment" > "$tmp/reject.log" 2>&1; then exit 1; fi
  grep -q 'TF_GLM_SEQ_GROW=1 needs' "$tmp/reject.log"
done
if request 'RANK_EXTRA_ENV=TF_GLM_PF_QMMF2=0 TF_GLM_PF_QMMF2' > "$tmp/reject.log" 2>&1; then exit 1; fi
grep -q 'KEY=VALUE' "$tmp/reject.log"
# The effective switches must reach every rank, not just the descriptive request.
rank_env_fn=$(source scripts/tp6.sh; declare -f tp6_rank_env)
rank_env_test=$(env -i PATH="$PATH" HOME="$HOME" GLM_TP6=1 MODEL_FORMAT=exlr8 bash -c '
  source scripts/prepare.sh
  TP=4 TP6_RANKS=4  # prepare.sh initially sees the standalone layout, as start-tp6-main does not.
  validate_config
  NODE_DEV=(lo lo) NODE_HCAS=(dummy dummy) NODE_GID=("" "") LINK_HEAD_ADDR=(unused 127.0.0.1)
  RANK_EXTRA_ENV_ARGS=()
  eval "$1"
  tp6_rank_env 0
  printf "%s\n" "${RANK_ENV[@]}"
' _ "$rank_env_fn")
for assignment in TF_GLM_RANK=0 TF_GLM_SEQ_GROW=1 TF_GLM_SEQ_GROW_STREAMS=2 TF_GLM_SEQ_GROW_FLOOR_MIB=2560 TF_GLM_DSA_SHARED=1 TF_GLM_X8_PROMPT_Q8=1; do
  grep -qx "$assignment" <<< "$rank_env_test"
done
# Patch 0245: KV_SPLIT=1 and KV_SPLIT=2 allow LEARN_GIB > 0.
for split in 1 2; do
  request TF_GLM_SEQ_GROW=0 KV_SPLIT="$split" LEARN_GIB=100 >/dev/null
done
echo 'exlr8-native: effective request switches, rank environment and sequence-growth validation: pass'
