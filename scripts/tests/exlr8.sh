#!/usr/bin/env bash
# MODEL_FORMAT and both EXLR8 releases,
# with no Spark and no network:
#  1. a checkpoint named through HF_CACHE_DIR / MODEL_DIR gets no format: the variables resolve as they did before the
#     default changed (CONTEXT 212992, no MODEL_FORMAT), which is what every K4 start and the night stages' restores rely on;
#  2. MODEL_FORMAT=k4 resolves to the same values as that long form; unset means k4; exlr8 means EXLR8 up1
#     (196,608 tokens, check weights);
#  3. start-tp6-main.sh implies TP6_RANKS=4 for k4, derisked and exlr8 only when it is unset;
#  4. fetch.py's range reader, plan and sparse files against a mocked HTTP transport; dirstate.py's states.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
ok() { if [[ "$2" == "$3" ]]; then echo "exlr8: $1: pass"; else echo "exlr8: $1: FAIL (got '$2', want '$3')"; fails=$((fails + 1)); fi; }
K4=$HOME/.cache/huggingface/hub/models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-MC
vars() { # vars <env...>: the resolved values of the variables the formats decide, TP4 as start-tp6-main.sh runs them
  env -i PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 TP=4 GLM_TP6=1 "$@" bash -c '
    source scripts/config.sh || exit
    for k in HF_CACHE_DIR MODEL_DIR MODEL_SOURCE NFS_MODEL_VOLUME SERVED_NAME WORKER_WEIGHTS WORKER_MODEL_DIR CONTEXT SPLIT_QKVA DSA_INDEX_FAST KV KV8_CODES DRAFTER EXLR8_RELEASE EXLR8_PROFILE EXLR8_REV; do
      printf "%s=%s\n" "$k" "${!k-}"; done; printf "FORMAT=%s\n" "${MODEL_FORMAT:-}"'
}
long=$(vars TP6_RANKS=4 HF_CACHE_DIR="$K4" MODEL_REVISION=e28931c918fa89403b1cd247f89c7dc25eeb67ac MODEL_SOURCE=local NFS_MODEL_VOLUME=glm53-k4mc-model SERVED_NAME=GLM-5.3-EXL3-K4)
ok 'a named checkpoint: no format' "$(grep '^FORMAT=' <<< "$long")" 'FORMAT='
ok 'a named checkpoint: K4 window 212,992' "$(grep '^CONTEXT=' <<< "$long")" 'CONTEXT=212992'
k4=$(vars MODEL_FORMAT=k4)
ok 'MODEL_FORMAT=k4 equals the long form' "$(grep -v '^FORMAT=' <<< "$k4")" "$(grep -v '^FORMAT=' <<< "$long")"
ok 'default is k4' "$(grep '^FORMAT=' <<< "$(vars)")" 'FORMAT=k4'
ok 'the default equals MODEL_FORMAT=k4' "$(vars)" "$k4"
dflt=$(vars MODEL_FORMAT=derisked)
ok 'derisked is selectable' "$(grep '^FORMAT=' <<< "$dflt")" 'FORMAT=derisked'
ok 'derisked: served name and NFS weights' "$(grep -E '^(SERVED_NAME|WORKER_WEIGHTS|NFS_MODEL_VOLUME)=' <<< "$dflt" | tr '\n' ' ')" 'NFS_MODEL_VOLUME=glm53-derisked-exl3-4bpw SERVED_NAME=GLM-5.3-Derisked-EXL3-4bpw WORKER_WEIGHTS=nfs '
ok 'derisked: no copy means a pinned download' "$(vars MODEL_FORMAT=derisked DERISKED_DIR=/nonexistent/d | grep -E '^(MODEL_DIR|MODEL_SOURCE)=' | tr '\n' ' ')" 'MODEL_DIR=/nonexistent/d MODEL_SOURCE=download '
def=$(vars MODEL_FORMAT=exlr8)
ok 'exlr8 is selectable' "$(grep '^FORMAT=' <<< "$def")" 'FORMAT=exlr8'
ok 'exlr8 release is up1' "$(grep '^EXLR8_RELEASE=' <<< "$def")" 'EXLR8_RELEASE=up1'
for release in up1 v07; do
  if [[ "$release" == up1 ]]; then
    profile=up1; rev=a36ae4680871ffa6412061d5daf66c4d4ad5802c
  else
    profile=bit400; rev=9e25c28a2e8808d8b2599d780521fd6af24e7558
  fi
  for dense in interim native; do
    leaf=exlr8-$release
    [[ "$dense" != native ]] || leaf+=-x8b
    cfg=$(vars MODEL_FORMAT=exlr8 EXLR8_RELEASE="$release" EXLR8_DENSE="$dense" EXLR8_WORKER_HOME='/worker home')
    ok "$release $dense pins" "$(grep -E '^EXLR8_(RELEASE|PROFILE|REV)=' <<< "$cfg")" \
      "$(printf 'EXLR8_RELEASE=%s\nEXLR8_PROFILE=%s\nEXLR8_REV=%s' "$release" "$profile" "$rev")"
    ok "$release $dense model directory" "$(grep '^MODEL_DIR=' <<< "$cfg")" "MODEL_DIR=$HOME/$leaf"
    ok "$release $dense worker directory" "$(grep '^WORKER_MODEL_DIR=' <<< "$cfg")" "WORKER_MODEL_DIR=/worker home/$leaf"
    ok "$release $dense served name" "$(grep '^SERVED_NAME=' <<< "$cfg")" "SERVED_NAME=GLM-5.3-EXLR8-$profile"
    ok "$release $dense context" "$(grep '^CONTEXT=' <<< "$cfg")" 'CONTEXT=196608'
    plan=$(env -i PATH="$PATH" HOME="$HOME" EXLR8_WORKER_HOME='/worker home' WORKER=r1 WORKER2=r2 WORKER3=r3 \
      DRY_RUN=1 DRAFTER=none EXLR8_RELEASE="$release" EXLR8_DENSE="$dense" bash scripts/prepare-exlr8.sh)
    ok "$release $dense prepare plan includes selection" "$(grep -c "env MODEL_FORMAT=exlr8 EXLR8_RELEASE=$release EXLR8_DENSE=$dense ./start-tp6.sh" <<< "$plan")" 1
  done
done
for assignment in EXLR8_RELEASE=bogus EXLR8_PROFILE=bit400 EXLR8_REV=9e25c28a2e8808d8b2599d780521fd6af24e7558; do
  if vars MODEL_FORMAT=exlr8 "$assignment" >/dev/null 2>&1; then ok "$assignment is refused" accepted refused; else ok "$assignment is refused" refused refused; fi
done
ok 'default window' "$(grep '^CONTEXT=' <<< "$def")" 'CONTEXT=196608'
ok 'default workers read their own copy' "$(grep '^WORKER_WEIGHTS=' <<< "$def")" 'WORKER_WEIGHTS=check'
ok 'default keeps the knobs' "$(grep -E '^(SPLIT_QKVA|DSA_INDEX_FAST|KV|KV8_CODES|DRAFTER)=' <<< "$def" | tr '\n' ' ')" 'SPLIT_QKVA=1 DSA_INDEX_FAST=1 KV=fp8 KV8_CODES=int8 DRAFTER=dflash2 '
pf=$'PF_EXPERTS_ONCE=1 PF_QMMF2=1 PF_DOWN_FUSE=1 '
pfv() { env -i PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 "$@" bash -c 'source scripts/config.sh; printf "PF_EXPERTS_ONCE=%s PF_QMMF2=%s PF_DOWN_FUSE=%s " "$PF_EXPERTS_ONCE" "$PF_QMMF2" "$PF_DOWN_FUSE"'; }
ok 'prefill switches (0206-0208) on at TP4, exlr8' "$(pfv TP=4 GLM_TP6=1 MODEL_FORMAT=exlr8)" "$pf"
ok 'prefill switches on at TP4, a named K4 checkpoint' "$(pfv TP=4 GLM_TP6=1 TP6_RANKS=4 HF_CACHE_DIR="$K4")" "$pf"
ok 'prefill switches on at TP6 too (patch 0239)' "$(pfv TP=6 GLM_TP6=1 MODEL_FORMAT=nvfp4)" "$pf"
ok 'prefill switches off at TP2' "$(pfv TP=2 GLM_TP6=1 MODEL_FORMAT=nvfp4)" 'PF_EXPERTS_ONCE=0 PF_QMMF2=0 PF_DOWN_FUSE=0 '
ok 'a prefill switch opts out alone' "$(pfv TP=4 GLM_TP6=1 PF_QMMF2=0)" 'PF_EXPERTS_ONCE=1 PF_QMMF2=0 PF_DOWN_FUSE=1 '
ok 'an exported TF_GLM_PF_ name sets the default' "$(pfv TP=4 GLM_TP6=1 TF_GLM_PF_DOWN_FUSE=0)" 'PF_EXPERTS_ONCE=1 PF_QMMF2=1 PF_DOWN_FUSE=0 '
ok 'an exported CONTEXT wins' "$(vars CONTEXT=65536 | grep '^CONTEXT=')" 'CONTEXT=65536'
ok 'an exported MODEL_DIR wins, no format' "$(vars MODEL_DIR=/x/y | grep -E '^(FORMAT|MODEL_DIR)=' | tr '\n' ' ')" 'MODEL_DIR=/x/y FORMAT= '
skv() { env -i PATH="$PATH" HOME="$HOME" GLM_TP6=1 "$@" bash -c 'source scripts/config.sh; printf "%s %s" "$SHARED_KV" "$PARALLEL"'; }
ok 'shared KV, four streams at six Sparks (derisked)' "$(skv TP=6 MODEL_FORMAT=derisked)" '1 4'
ok 'shared KV, four streams at six Sparks (k4)' "$(skv TP=6 MODEL_FORMAT=k4)" '1 4'
ok 'one stream at four Sparks' "$(skv TP=4 MODEL_FORMAT=k4)" '0 1'
ok 'SHARED_KV=0 opts out at six Sparks' "$(skv TP=6 MODEL_FORMAT=derisked SHARED_KV=0)" '0 1'
ok 'nvfp4 is the six-Spark default' "$(env -i PATH="$PATH" HOME="$HOME" TP=6 GLM_TP6=1 MODEL_FORMAT=nvfp4 bash -c 'source scripts/config.sh; printf "%s %s" "$CONTEXT" "$WORKER_WEIGHTS"')" 'auto nfs'
if env -i PATH="$PATH" HOME="$HOME" MODEL_FORMAT=bogus bash -c 'source scripts/config.sh' >/dev/null 2>&1; then ok 'a bad format is refused' accepted refused; else ok 'a bad format is refused' refused refused; fi
snip=$(sed -n '/^# MODEL_FORMAT k4 (the default), derisked and exlr8/,/^export TP=/p' start-tp6-main.sh)
ranks() { env -i PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 "$@" bash -c "$snip"$'\nprintf "%s" "TP=$TP"'; }
ok 'start implies four ranks for k4 (the default)' "$(ranks)" 'TP=4'
ok 'start implies four ranks for derisked' "$(ranks MODEL_FORMAT=derisked)" 'TP=4'
ok 'start implies four ranks for exlr8' "$(ranks MODEL_FORMAT=exlr8)" 'TP=4'
ok 'start implies four ranks for k4' "$(ranks MODEL_FORMAT=k4)" 'TP=4'
ok 'start keeps six for a named checkpoint' "$(ranks HF_CACHE_DIR="$K4")" 'TP=6'
ok 'start keeps six for nvfp4' "$(ranks MODEL_FORMAT=nvfp4)" 'TP=6'
ok 'start keeps TP6_RANKS as given' "$(ranks TP6_RANKS=6)" 'TP=6'
# validate_config: exlr8 needs four ranks and its own copies
chk() { env -i PATH="$PATH" HOME="$HOME" LANG=C.UTF-8 "$@" bash -c 'source scripts/prepare.sh; validate_config' 2>&1 >/dev/null | head -1; }
ok 'exlr8 at six ranks is refused' "$(chk TP=6 GLM_TP6=1 MODEL_FORMAT=exlr8 | grep -c 'serves four Sparks')" 1
ok 'exlr8 over NFS is refused' "$(chk TP=4 GLM_TP6=1 TP6_RANKS=4 MODEL_FORMAT=exlr8 WORKER_WEIGHTS=nfs | grep -c 'WORKER_WEIGHTS must be check')" 1
ok 'exlr8 at four ranks is accepted' "$(chk TP=4 GLM_TP6=1 TP6_RANKS=4 MODEL_FORMAT=exlr8)" ''

tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
python3 -B -I - "$PWD/scripts/exlr8" "$tmp" <<'PY' || fails=$((fails + 1))
import io, json, os, struct, sys
tools, tmp = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools)
import fetch, dirstate

# Exercise redirect handling, HTTP range headers and sparse writes without opening a socket.
BLOB = bytes((i * 7 + 3) % 251 for i in range(1 << 20))
hdr = json.dumps({'a': 1}).encode(); hdr += b' ' * (-len(hdr) % 8)
BLOB = struct.pack('<Q', len(hdr)) + hdr + BLOB[8 + len(hdr):]

class Redirect(io.BytesIO):
    def geturl(self):
        return 'https://weights.example/model?ticket=local'

def resolve(request, timeout):
    assert request.get_header('Range') == 'bytes=0-0'
    return Redirect(BLOB[:1])

class Connection:
    def __init__(self, host, timeout):
        assert host == 'weights.example'

    def request(self, method, target, headers):
        assert method == 'GET' and target == '/model?ticket=local'
        a, b = map(int, headers['Range'].removeprefix('bytes=').split('-'))
        self.response = io.BytesIO(BLOB[a:b + 1])
        self.response.status = 206
        self.response.headers = {'Content-Range': f'bytes {a}-{b}/{len(BLOB)}'}

    def getresponse(self):
        return self.response

fetch.urllib.request.urlopen = resolve
fetch.http.client.HTTPSConnection = Connection
bad = []
def check(name, cond):
    print(f'exlr8: {name}: {"pass" if cond else "FAIL"}')
    if not cond: bad.append(name)

ranges = [(100000, 4096), (500000, 70000)]
ef = fetch.ExpertFile('rev', tmp, 3, 4, ranges, len(BLOB), 0, 4)
for t, c in ef.tasks(): t.run(c)
data = open(ef.dest, 'rb').read()
check('fetch: the ranges land at their offsets', all(data[o:o + n] == BLOB[o:o + n] for o, n in ranges))
check('fetch: the safetensors header lands', data[:8 + len(hdr)] == BLOB[:8 + len(hdr)])
check('fetch: nothing else is written (sparse holes)', data[8 + len(hdr):100000] == bytes(100000 - 8 - len(hdr)))
check('fetch: a ledger vouches for the file', ef.done())
ef2 = fetch.ExpertFile('rev', tmp, 3, 4, ranges + [(900000, 10)], len(BLOB), 0, 4)
check('fetch: another plan is not complete', not ef2.done())

# The two profiles pin distinct hashes at the same 76 basenames.
up1, v07 = (dirstate.index_hashes(dirstate.RELEASES[key]) for key in ('up1', 'v07'))
check('fetch: release-specific index tables', (set(up1) == set(v07) == {f'exlr8-layer-{L:03d}.index.safetensors'
                                                                     for L in range(3, 79)}
                                             and all(up1[key] != v07[key] for key in up1)))

# the rank rule and the plan: halves of expert e belong to rank (16 e + h + L) mod ranks; one range per half
import exlr8_pieces
check('plan: half ownership', [h for h in range(16) if (16 * 5 + h + 3) % 4 == 2] == exlr8_pieces.rank_halves(5, 3, 2, 4))
check('plan: ranks partition the halves', sorted(h for r in range(4) for h in exlr8_pieces.rank_halves(7, 40, r, 4)) == list(range(16)))

# dirstate: new / foreign / partial / complete
d = os.path.join(tmp, 'ck'); os.mkdir(d)
check('dirstate: empty is new', dirstate.state(d, 0, 4) == 'new')
open(os.path.join(d, 'w.safetensors'), 'w').write('x')
check('dirstate: files without the marker are foreign', dirstate.state(d, 0, 4) == 'foreign')
open(os.path.join(d, '.x8-owned'), 'w').close()
check('dirstate: owned without .x8-complete is partial', dirstate.state(d, 0, 4) == 'partial')
st = os.stat(os.path.join(d, 'w.safetensors'))
json.dump({'rank': 0, 'ranks': 4, 'sizes': {'w.safetensors': st.st_size}, 'mtimes': {'w.safetensors': st.st_mtime_ns}}, open(os.path.join(d, '.x8-complete'), 'w'))
check('dirstate: a matching marker is complete', dirstate.state(d, 0, 4) == 'complete')
check('dirstate: another rank is not', dirstate.state(d, 1, 4) == 'partial')
open(os.path.join(d, 'w.safetensors'), 'w').write('xy')
check('dirstate: a changed file is partial again', dirstate.state(d, 0, 4) == 'partial')
sys.exit(1 if bad else 0)
PY
(( fails == 0 )) || { echo "exlr8: $fails failure(s)" >&2; exit 1; }
echo 'exlr8: all pass'
