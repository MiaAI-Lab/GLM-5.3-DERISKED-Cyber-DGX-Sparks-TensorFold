#!/usr/bin/env bash
# Convert one rank of kindlingai's EXLR8 checkpoint for Ash Hart's TensorFold (EXLR8_RELEASE=up1 or v07).
# The dense trellis stays as stored. The recipe converter and patch 0212 are by Mia's AI Lab.
#
# Usage: convert-native.sh --rank R [--ranks 4] --out DIR --tools DIR [--release DIR]
#                          [--index-dir DIR] [--x8 DIR] [--py PYTHON] [--layers LO-HI] [--dry-run]
# --out must be absent or an owned native conversion directory. Dense writes use a private staging directory.
# --tools supplies the interim dense reader, pieces converter and checker, either flat or in x8/.
# --x8 reuses verified routed and shared-expert pieces from the same release, profile and rank topology.
# --layers defaults to 0-77. Any smaller range is a test conversion whose marker stays pending.
set -euo pipefail
T=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RANK=''; RANKS=4; OUT=''; TOOLS=''; REL=''; IDX=""; X8=''; PY=python3; LAYERS=0-77; DRY=0
while (( $# )); do
  case "$1" in
    --rank) RANK=$2; shift 2 ;; --ranks) RANKS=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --tools) TOOLS=$2; shift 2 ;;
    --release) REL=$2; shift 2 ;; --index-dir) IDX=$2; shift 2 ;; --x8) X8=$2; shift 2 ;; --py) PY=$2; shift 2 ;;
    --layers) LAYERS=$2; shift 2 ;; --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,10s/^# \{0,1\}//p' "$0"; exit 0 ;;
    *) echo "convert-native.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
say() { printf '%s rank %s: %s\n' "$(date +%T)" "$RANK" "$*"; }
fail() { printf '### x8b rank %s FAIL: %s\n' "$RANK" "$*" >&2; exit 1; }
[[ "$RANK" =~ ^(0|[1-9][0-9]?)$ && "$RANKS" =~ ^(1|2|4|8|16)$ ]] && (( RANK < RANKS )) || fail 'expected a rank below --ranks (1, 2, 4, 8 or 16)'
[[ -n "$OUT" && -d "$TOOLS" ]] || fail '--out and --tools are required'
[[ "$LAYERS" =~ ^(0|[1-9][0-9]?)-(0|[1-9][0-9]?)$ ]] || fail 'expected --layers LO-HI'
LO=${LAYERS%-*}; HI=${LAYERS#*-}
(( LO <= HI && HI <= 77 )) || fail 'expected layers within 0-77'
PIECE_LO=$(( LO < 3 ? 3 : LO ))
# shellcheck source=scripts/exlr8/release.sh
source "$T/release.sh"
IDX=${IDX:-${EXLR8_INDEX_DIR:-$HOME/exlr8-index/$EXLR8_RELEASE}}
if [[ -z "$REL" ]]; then
  REL=${EXLR8_RELEASE_DIR:-${EXLR8_SRC:-}}
  if [[ -z "$REL" && "$EXLR8_RELEASE" == v07 ]]; then REL=$HOME/$EXLR8_INTERIM_LEAF-src; fi
  [[ -n "$REL" && -f "$REL/meta.json" && -f "$REL/$EXLR8_MANIFEST" ]] || REL=$HOME/exlr8-release/$EXLR8_REV
fi

state_tool() { python3 -IB "$T/dirstate.py" "$@"; }
OUT=$(state_tool native-output "$OUT" "$X8" "$REL" "$IDX" "$TOOLS") || fail 'unsafe native output directory'
MAN=$REL/$EXLR8_MANIFEST
[[ -f "$MAN" && -f "$REL/meta.json" && -d "$REL/dense" ]] || fail "$REL is not an EXLR8 release layout"
state_tool release "$REL" "$RANKS" "$EXLR8_REV" || fail 'release identity'
shopt -s nullglob
indexes=("$IDX/experts/"exlr8-layer-*.index.safetensors)
(( ${#indexes[@]} )) || fail "no index files in $IDX/experts"
shopt -u nullglob
tf() {
  local p
  for p in "$TOOLS/x8/$1" "$TOOLS/$1"; do
    if [[ -f "$p" ]]; then printf '%s\n' "$p"; return; fi
  done
  fail "no $1 under $TOOLS"
}
DENSE=$(tf exlr8_dense.py); PIECES=$(tf exlr8_pieces.py); CHECKP=$(tf check_pieces.py); IDXSHA="$T/$EXLR8_INDEX_SHA"
TOOLS_JSON=$(python3 -IB - "$T/exlr8_native.py" "$DENSE" "$PIECES" <<'PY'
import hashlib, json, sys
from pathlib import Path

native, dense, pieces = map(Path, sys.argv[1:])
print(json.dumps({'exlr8_dense.py': hashlib.sha256(native.read_bytes() + dense.read_bytes()).hexdigest(),
                  'exlr8_pieces.py': hashlib.sha256(pieces.read_bytes()).hexdigest()}, sort_keys=True))
PY
)
MIN_FREE=${EXLR8_MIN_FREE_GB:-116}
[[ -z "$X8" || -n ${EXLR8_MIN_FREE_GB:-} ]] || MIN_FREE=22
[[ "$MIN_FREE" =~ ^(0|[1-9][0-9]{0,8})$ ]] || fail 'EXLR8_MIN_FREE_GB must be a non-negative integer'
parent=$(dirname -- "$OUT")
ancestor() { local p=$1; while [[ ! -e "$p" ]]; do p=$(dirname -- "$p"); done; printf '%s\n' "$p"; }

if [[ -n "$X8" ]]; then
  state_tool checkpoint "$X8" "$RANK" "$RANKS" interim --release "$REL" --pieces-tool "$PIECES" >/dev/null \
    || fail "$X8 cannot supply pieces for this conversion"
  [[ $(stat -Lc %d -- "$X8") == "$(stat -Lc %d -- "$(ancestor "$parent")")" ]] || fail "$X8 and $parent are on different filesystems"
fi

if [[ "$DRY" == 0 ]]; then
  umask 077
  mkdir -p -- "$parent"
  if [[ ! -e "$OUT" ]]; then
    mkdir -m 700 -- "$OUT"
  fi
  state_tool native-output "$OUT" "$X8" "$REL" "$IDX" "$TOOLS" >/dev/null || fail 'unsafe native output directory'
  exec {out_lock}<"$OUT"
  flock -n "$out_lock" || fail "another conversion is using $OUT"
  state_tool claim "$OUT" || fail 'cannot claim native output'
  touch "$OUT/.x12-native"
fi

link_args=()
[[ -z "$X8" ]] || link_args=(--link-source "$X8")
if [[ "$LAYERS" == 0-77 && -f "$OUT/.x8-complete" ]] &&
   state_tool checkpoint "$OUT" "$RANK" "$RANKS" native --release "$REL" --tools "$TOOLS_JSON" "${link_args[@]}" >/dev/null; then
  say "complete already: $OUT"
  if [[ "$DRY" == 0 ]]; then state_tool verify "$OUT" "$RANK" "$RANKS" || fail 'bytes differ from the marker'; fi
  exit 0
fi
if [[ "$LAYERS" != 0-77 && -e "$OUT/.x8-complete" ]]; then
  fail 'use a separate output directory for layer tests'
fi
if [[ "$DRY" == 1 ]]; then
  say "convert layers $LAYERS into $OUT (pieces from ${X8:-exlr8_pieces.py}, needs $MIN_FREE GiB free)"
  exit 0
fi

min_avail=${EXLR8_MIN_AVAIL_GIB:-${X8B_MIN_AVAIL_GIB:-14}}
[[ "$min_avail" =~ ^(0|[1-9][0-9]{0,8})$ ]] || fail 'EXLR8_MIN_AVAIL_GIB must be a non-negative integer'
a=$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)
(( a >= min_avail )) || fail "MemAvailable ${a} GiB, needs $min_avail GiB: stop the server first"
free_bytes() { df -B1 --output=avail -- "$OUT" | awk 'END {print $1}'; }
f=$(free_bytes)
reclaimed=$(state_tool native-reclaimable "$OUT")
(( f + reclaimed >= MIN_FREE * 1073741824 )) || fail "$OUT needs $MIN_FREE GiB for conversion"
chmod 700 -- "$OUT"
# Remove the previous marker first and keep the directory inode locked throughout recovery.
rm -f -- "$OUT/.x8-complete"
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .x8-owned ! -name .x12-native -exec rm -rf --one-file-system -- {} +
f=$(free_bytes)
(( f >= MIN_FREE * 1073741824 )) || fail "$(( f / 1073741824 )) GiB free at $OUT, needs $MIN_FREE GiB"

mkdir "$OUT/.x8-tools"
cp -p -- "$PIECES" "$DENSE" "$CHECKP" "$IDXSHA" "$T/"{exlr8_native,exlr8_assemble,dirstate}.py "$OUT/.x8-tools/"
printf '%s\n' "$TOOLS_JSON" > "$OUT/.x8-converted-by"
if (( PIECE_LO <= HI )); then
  if [[ -n "$X8" ]]; then
    say 'hard-linking interim piece files'
    for (( L = PIECE_LO; L <= HI; L++ )); do
      fn=$(printf 'exlr8-exl3-layer-%03d.r%dof%d.safetensors' "$L" "$RANK" "$RANKS")
      ln -L -- "$X8/$fn" "$OUT/$fn"
    done
  else
    say 'converting routed and shared-expert rank pieces'
    "$PY" -IB "$OUT/.x8-tools/exlr8_pieces.py" "$REL" "$MAN" "$RANK" "$RANKS" "$OUT" \
      --layers "$PIECE_LO-$HI" --index-dir "$IDX" --check 3 || fail 'exlr8_pieces.py'
  fi
fi
stage=$(mktemp -d -- "$OUT/.dense.XXXXXXXX")
say 'converting dense trellis and vocabulary'
"$PY" -B "$OUT/.x8-tools/exlr8_native.py" "$REL" "$IDX" "$RANK" "$RANKS" "$stage" \
  --layers "$LAYERS" --vocab --check || fail 'exlr8_native.py'
mv -- "$stage/"*.safetensors "$OUT/"
rmdir -- "$stage"
for L in 3 40 77; do
  (( L >= LO && L <= HI )) || continue
  python3 -IB "$OUT/.x8-tools/check_pieces.py" "$REL" "$IDX" "$OUT" "$L" "$RANK" "$RANKS" 6 | tail -1 | grep -q ' 0 failed' \
    || fail "check_pieces.py layer $L"
done
say 'assembling the pending checkpoint'
"$PY" -IB "$OUT/.x8-tools/exlr8_assemble.py" "$REL" "$OUT" "$RANK" "$RANKS" --layers "$LAYERS" --native || fail 'exlr8_assemble.py'
state_tool publish-native "$OUT" "$RANK" "$RANKS" "$REL" "$X8" "$LO" "$HI" || fail 'native checkpoint verification'
[[ "$LAYERS" != 0-77 ]] || say "complete: $OUT"
