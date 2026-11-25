#!/usr/bin/env bash
# Convert one rank of kindlingai's EXLR8 checkpoint on this Spark. EXLR8_RELEASE=up1 (default) or v07
# selects the pinned release. Each box fetches its rank's expert halves and the shared dense files.
# Usage: box.sh --rank R [--ranks 4] [--status]
#
# EXLR8_DENSE=interim writes ~/exlr8-up1 or ~/exlr8-v07. EXLR8_DENSE=native writes the matching -x8b
# directory, reusing compatible pieces from EXLR8_X8_DIR when complete. EXLR8_DIR overrides the output.
# A complete directory is re-hashed. A partial directory owned by this release is converted again.
#
# Optional inputs: EXLR8_VENV (NumPy venv), EXLR8_INDEX_DIR (default ~/exlr8-index/$EXLR8_RELEASE),
# EXLR8_RELEASE_DIR (prepared release layout with meta.json), EXLR8_SRC (complete Hugging Face copy,
# default ~/exlr8-v07-src for v07 and unset for up1), EXLR8_KEEP_RELEASE=1 (keep fetched bytes).
# The server should be stopped. Defaults are 24 GiB available RAM and 135 GiB free for interim output
# or 116 GiB for native output (22 GiB with hard links). Fetching adds about 100 GiB per rank at TP4.
# EXLR8_MIN_AVAIL_GIB and EXLR8_MIN_FREE_GB override the output checks.
set -euo pipefail
T=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RANK=''; RANKS=4; STATUS=0
while (( $# )); do
  case "$1" in
    --rank) RANK=$2; shift 2 ;;
    --ranks) RANKS=$2; shift 2 ;;
    --status) STATUS=1; shift ;;
    -h|--help) sed -n '2,14s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "box.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$RANK" =~ ^(0|[1-9][0-9]?)$ && "$RANKS" =~ ^(1|2|4|8|16)$ ]] && (( RANK < RANKS )) || { echo 'box.sh: expected a rank below --ranks (1, 2, 4, 8 or 16)' >&2; exit 2; }

# shellcheck source=scripts/exlr8/release.sh
source "$T/release.sh"
DENSE=${EXLR8_DENSE:-interim}
case "$DENSE" in
  interim) leaf=$EXLR8_INTERIM_LEAF ;;
  native) leaf=$EXLR8_NATIVE_LEAF ;;
  *) echo "box.sh: EXLR8_DENSE is interim or native (got $DENSE)" >&2; exit 2 ;;
esac
export EXLR8_DENSE=$DENSE   # dirstate.py's provenance follows it
OUT=${EXLR8_DIR:-$HOME/$leaf}
X8DIR=${EXLR8_X8_DIR:-$HOME/$EXLR8_INTERIM_LEAF}
VENV=${EXLR8_VENV:-$HOME/exlr8-venv}; P=$VENV/bin/python
IDX=${EXLR8_INDEX_DIR:-$HOME/exlr8-index/$EXLR8_RELEASE}
SRC=${EXLR8_SRC:-}
if [[ -z "$SRC" && "$EXLR8_RELEASE" == v07 ]]; then SRC=$HOME/$EXLR8_INTERIM_LEAF-src; fi
MIN_AVAIL=${EXLR8_MIN_AVAIL_GIB:-24}
LINK_X8=''; [[ $DENSE == native && -f $X8DIR/.x8-complete ]] && LINK_X8=$X8DIR
default_free=135
if [[ $DENSE == native ]]; then
  default_free=116
  [[ -z $LINK_X8 ]] || default_free=22
fi
MIN_FREE=${EXLR8_MIN_FREE_GB:-$default_free}
say() { printf '%s rank %s: %s\n' "$(date +%T)" "$RANK" "$*"; }
fail() { printf '### exlr8 rank %s FAIL: %s\n' "$RANK" "$*" >&2; exit 1; }
ancestor() { local p=$1; while [[ ! -e "$p" ]]; do p=$(dirname -- "$p"); done; printf '%s\n' "$p"; }
free_bytes() { df -B1 --output=avail -- "$1" | awk 'END {print $1}'; }
check_disk() {
  local path=$1 need=$2 free
  free=$(free_bytes "$path")
  (( free >= need )) || fail "disk at $path has $(( free / 1073741824 )) GiB free, needs $(( (need + 1073741823) / 1073741824 )) GiB"
}
avail_gib() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
command -v python3 >/dev/null || fail 'python3 is missing'
[[ "$MIN_FREE" =~ ^(0|[1-9][0-9]{0,8})$ && "$MIN_AVAIL" =~ ^(0|[1-9][0-9]{0,8})$ ]] || fail 'EXLR8_MIN_FREE_GB and EXLR8_MIN_AVAIL_GIB must be non-negative integers'
state_tool() { python3 -IB "$T/dirstate.py" "$@"; }
state=$(state_tool state "$OUT" "$RANK" "$RANKS") || fail 'output release identity'
[[ $state != foreign ]] || fail "$OUT exists and is not this recipe's conversion (no .x8-owned)"

# ---- the release layout the converters read ---------------------------------------------------------------------------------
FETCHED=0; seed=0
TOK_FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja)
REL=${EXLR8_RELEASE_DIR:-$HOME/exlr8-release/$EXLR8_REV}
if [[ -z "${EXLR8_RELEASE_DIR:-}" ]]; then
  if [[ -n "$SRC" && -f "$SRC/$EXLR8_MANIFEST" && -f "$SRC/dense/exlr8-vocab.safetensors" &&
        $(find "$SRC/experts" -maxdepth 1 -name 'exlr8-layer-*.k4.safetensors' 2>/dev/null | wc -l) == 75 ]]; then
    state_tool manifest "$SRC/$EXLR8_MANIFEST" || fail 'source manifest identity'
    seed=1
  else
    FETCHED=1
  fi
fi
if [[ $DENSE == native ]]; then
  OUT=$(state_tool native-output "$OUT" "$X8DIR" "$REL" "$IDX" "$T" "$VENV") || fail 'unsafe native output directory'
  if (( FETCHED == 1 || seed == 1 )); then state_tool protect-output "$REL" "$X8DIR" >/dev/null || fail 'unsafe native release cache'; fi
elif [[ -e "$OUT/.x12-native" ]]; then
  fail "$OUT is a native conversion directory; use EXLR8_DENSE=native or another output path"
fi
if (( seed == 1 && STATUS == 0 )); then
  # Link a complete local source into the revision-specific cache, then record its receipt there.
  mkdir -p -- "$REL/tokenizer"
  for d in experts dense manifests; do [[ -e "$REL/$d" || -L "$REL/$d" ]] || ln -s -- "$SRC/$d" "$REL/$d"; done
  for f in "${TOK_FILES[@]}"; do
    [[ -e "$REL/tokenizer/$f" ]] && continue
    if [[ -f "$SRC/tokenizer/$f" ]]; then ln -s -- "$SRC/tokenizer/$f" "$REL/tokenizer/$f"; else ln -s -- "$SRC/$f" "$REL/tokenizer/$f"; fi
  done
  [[ -e "$REL/meta.json" ]] || printf '{"release": "%s@%s", "files": ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]}\n' "$EXLR8_REPO" "$EXLR8_REV" > "$REL/meta.json"
fi
TOOLS_JSON=$(state_tool tools "$T")
identity_args=(--revision "$EXLR8_REV")
if [[ -n ${EXLR8_RELEASE_DIR:-} || ( -f "$REL/meta.json" && -f "$REL/$EXLR8_MANIFEST" ) ]]; then
  identity_args+=(--release "$REL")
  state_tool release "$REL" "$RANKS" "$EXLR8_REV" || fail 'release identity'
fi
cleanup_release() {
  if (( FETCHED == 1 )) && [[ "${EXLR8_KEEP_RELEASE:-0}" != 1 && -d "$REL" ]]; then
    say "removing the fetched release $REL (EXLR8_KEEP_RELEASE=1 keeps it)"
    rm -rf -- "$REL"
  fi
}
if [[ $DENSE == native && $state == complete ]]; then
  reuse_args=()
  [[ -z "$LINK_X8" ]] || reuse_args=(--link-source "$LINK_X8")
  if state_tool checkpoint "$OUT" "$RANK" "$RANKS" native --tools "$TOOLS_JSON" "${identity_args[@]}" "${reuse_args[@]}" >/dev/null; then
    say "$OUT: complete"
    if [[ $STATUS == 0 ]]; then
      state_tool verify "$OUT" "$RANK" "$RANKS" || fail 'bytes differ from the marker'
      cleanup_release
    fi
    exit 0
  fi
  state=partial
fi
say "$OUT: $state"

if [[ $state == complete ]]; then
  had=$(python3 -I -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), sort_keys=True))' "$OUT/.x8-converted-by" 2>/dev/null || true)
  [[ "$had" == "$TOOLS_JSON" ]] || fail "$OUT was converted by other converters (${had:-no .x8-converted-by}): remove it to convert again"
  if [[ -f "$OUT/exlr8-set.json" ]]; then
    state_tool checkpoint "$OUT" "$RANK" "$RANKS" interim "${identity_args[@]}" >/dev/null || fail 'interim checkpoint identity'
  fi
  [[ $STATUS == 1 ]] && { say "complete (converters $(sha256sum "$T/exlr8_pieces.py" | cut -c1-12))"; exit 0; }
  need=$(state_tool repair "$OUT")
  if [[ $need == repair ]]; then  # refresh older metadata, keeping the weights
    say 're-assembling in place (the weights are re-hashed first)'
    python3 -I "$T/dirstate.py" verify "$OUT" "$RANK" "$RANKS" || fail 're-hash before re-assembling'
    [[ -f "$REL/$EXLR8_MANIFEST" && -f "$REL/meta.json" ]] || python3 -B "$T/fetch.py" small "$REL" "$EXLR8_REV"
    python3 -IB "$T/exlr8_assemble.py" "$REL" "$OUT" "$RANK" "$RANKS" || fail 'exlr8_assemble.py'
  fi
  python3 -I "$T/dirstate.py" verify "$OUT" "$RANK" "$RANKS" || fail "$OUT: bytes differ from its marker"
  cleanup_release
  say 'complete'; exit 0
fi
[[ $STATUS == 0 ]] || { say "not complete ($state)"; exit 3; }

# ---- checks, then the work -----------------------------------------------------------------------------------------------------
a=$(avail_gib); (( a >= MIN_AVAIL )) || fail "MemAvailable ${a} GiB, needs ${MIN_AVAIL}: stop the server first (./stop-tp6.sh); the dense converter holds ~8 GB"
# a partial directory is this recipe's own and is converted again, so its space counts as free: empty it first
if [[ $state == partial && $DENSE != native ]]; then   # native: convert-native.sh empties only a directory it made (.x12-native)
  [[ -f "$OUT/.x8-owned" ]] || fail "$OUT is partial but not owned"
  say "emptying its own partial $OUT"
  state_tool claim "$OUT" || fail 'cannot claim interim output'
  find "$OUT" -mindepth 1 -maxdepth 1 ! -name .x8-owned -exec rm -rf --one-file-system -- {} +
fi
out_fs=$(ancestor "$OUT")
output_need=$(( MIN_FREE * 1073741824 ))
if [[ $DENSE == native && $state == partial ]]; then
  reclaimed=$(state_tool native-reclaimable "$OUT")
  output_need=$(( output_need > reclaimed ? output_need - reclaimed : 0 ))
fi
if [[ -n "$LINK_X8" ]]; then
  state_tool checkpoint "$LINK_X8" "$RANK" "$RANKS" interim --pieces-tool "$T/exlr8_pieces.py" "${identity_args[@]}" >/dev/null \
    || fail "$LINK_X8 cannot supply pieces for this conversion"
  [[ $(stat -Lc %d -- "$LINK_X8") == "$(stat -Lc %d -- "$out_fs")" ]] || fail "$LINK_X8 and $OUT are on different filesystems"
fi
if (( FETCHED == 1 )); then
  cache_fs=$(ancestor "$REL")
  # About 100 GiB at TP4. Smaller worlds fetch more expert halves per rank.
  cache_need=$(( (16 + (336 + RANKS - 1) / RANKS) * 1073741824 ))
  if [[ -d "$REL" ]]; then
    read -r cached _ < <(du -sB1 -- "$REL")
    cache_need=$(( cache_need > cached ? cache_need - cached : 0 ))
  fi
  if [[ $(stat -Lc %d -- "$out_fs") == "$(stat -Lc %d -- "$cache_fs")" ]]; then
    output_need=$(( output_need + cache_need ))
  else
    check_disk "$cache_fs" "$cache_need"
  fi
fi
check_disk "$out_fs" "$output_need"
if [[ $DENSE != native ]]; then   # native: convert-native.sh makes and marks its directory
  [[ -e "$OUT" ]] || { mkdir -p "$OUT"; }
  state_tool claim "$OUT" || fail 'cannot claim interim output'
  rm -f "$OUT/.x8-complete"
else
  state_tool protect-output "$IDX" "$X8DIR" >/dev/null || fail 'unsafe native index cache'
  if [[ ! -x "$P" ]]; then state_tool protect-output "$VENV" "$X8DIR" >/dev/null || fail 'unsafe native venv'; fi
fi

if [[ ! -x $P ]]; then
  say "making $VENV (numpy from PyPI)"
  python3 -m venv "$VENV" && "$P" -m pip install --quiet numpy || fail "no numpy venv at $VENV"
fi
"$P" -c 'import numpy' || fail "numpy is missing in $VENV"

say 'the 76 layer index files (sha256-checked against the hub)'
python3 -B "$T/fetch.py" index "$IDX" "$EXLR8_REV" || fail 'index files'
if (( FETCHED == 1 )); then
  say "fetching this rank's halves, the dense files and the manifest into $REL"
  python3 -B "$T/fetch.py" release "$REL" "$IDX" "$RANK" "$RANKS" "$EXLR8_REV" || fail 'fetching the release (rerun: it resumes)'
fi
state_tool release "$REL" "$RANKS" "$EXLR8_REV" || fail 'release identity'

if [[ $DENSE == native ]]; then
  link_args=()
  [[ -z "$LINK_X8" ]] || link_args=(--x8 "$LINK_X8")
  EXLR8_MIN_FREE_GB=$MIN_FREE EXLR8_MIN_AVAIL_GIB=$MIN_AVAIL bash "$T/convert-native.sh" --rank "$RANK" --ranks "$RANKS" \
    --out "$OUT" --tools "$T" --release "$REL" --index-dir "$IDX" --py "$P" "${link_args[@]}" || fail 'native conversion'
else
  # Record the provenance of the converters that write the weights.
  printf '%s' "$TOOLS_JSON" > "$OUT/.x8-converted-by"
  MAN=$REL/$EXLR8_MANIFEST
  say 'routed and shared experts -> EXL3 rank pieces'
  python3 -IB "$T/exlr8_pieces.py" "$REL" "$MAN" "$RANK" "$RANKS" "$OUT" --index-dir "$IDX" --check 3 || fail 'exlr8_pieces.py'
  say 'dense layers, shared expert, vocabulary (each shared half at the manifest width)'
  "$P" -B "$T/exlr8_dense.py" "$REL" "$IDX" "$RANK" "$RANKS" "$OUT" --manifest "$MAN" --layers 0-77 --vocab --check || fail 'exlr8_dense.py'
  for L in 3 40 77; do
    python3 -IB "$T/check_pieces.py" "$REL" "$IDX" "$OUT" "$L" "$RANK" "$RANKS" 6 | tail -1 | grep -q ' 0 failed' || fail "check_pieces.py layer $L"
  done
  say 'assembling (config.json, exlr8-set.json, index, tokenizer, .x8-complete)'
  "$P" -I "$T/exlr8_assemble.py" "$REL" "$OUT" "$RANK" "$RANKS" || fail 'exlr8_assemble.py'
  python3 -I "$T/dirstate.py" verify "$OUT" "$RANK" "$RANKS" || fail 'bytes differ from the marker'
fi
cleanup_release
say "complete: $OUT"
