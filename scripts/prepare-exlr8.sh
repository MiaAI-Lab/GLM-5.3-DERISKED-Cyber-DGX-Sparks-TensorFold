#!/usr/bin/env bash
# Prepare an EXLR8 checkpoint across four Sparks. EXLR8_RELEASE=up1 (default) or v07 chooses the release.
# Each Spark converts its own rank, then the four verified markers are compared before serving.
# Usage: scripts/prepare-exlr8.sh [--status | --convert]
#   --status    report each rank's state
#   --convert   (default) convert incomplete ranks and re-hash complete ones
#   DRY_RUN=1   print the plan
# Stop the server first (./stop.sh). Each rank needs about 135 GiB free for interim output
# or 116 GiB for native output (22 GiB with hard links), plus about 100 GiB to fetch the rank's data.
# EXLR8_SRC can specify a complete local release copy. v07 looks for ~/exlr8-v07-src on each Spark.
# Rank 0 runs here. Ranks 1-3 use WORKER, WORKER2 and WORKER3. Their homes must agree.
# Set EXLR8_WORKER_HOME in scripts/exlr8/hosts.env if the workers use a different home.
set -euo pipefail
export MODEL_FORMAT=${MODEL_FORMAT:-exlr8}  # this script prepares EXLR8 only; the recipe's default format is k4
source "$(dirname -- "${BASH_SOURCE[0]}")/prepare.sh"
mode=--convert
case "${1:-}" in
  ''|--convert) ;;
  --status) mode=--status ;;
  -h|--help) sed -n '2,12s/^# \{0,1\}//p' "$0"; exit 0 ;;
  *) echo 'Usage: scripts/prepare-exlr8.sh [--status|--convert]' >&2; exit 2 ;;
esac
[[ "${MODEL_FORMAT:-}" == exlr8 ]] || { echo "[E_CONFIG] MODEL_FORMAT is '${MODEL_FORMAT:-}', not exlr8: nothing to prepare here (MODEL_FORMAT=exlr8 $0)." >&2; exit 1; }
RANKS=4
TOOLS_SRC=$REPO_ROOT/scripts/exlr8
if [[ "$EXLR8_DENSE" == native ]]; then
  python3 -IB "$TOOLS_SRC/dirstate.py" native-output "$MODEL_DIR" "${EXLR8_X8_DIR:-}" >/dev/null \
    || fail E_CONFIG 'Choose a separate native checkpoint directory for rank 0.'
fi
LOGS=$LOG_DIR/exlr8/$EXLR8_RELEASE
if [[ "$DRY_RUN" != 1 && "$mode" == --convert ]]; then
  mkdir -p "$LOGS" "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
fi
declare -a HOSTS=(local "$WORKER" "$WORKER2" "$WORKER3") HOMES=() OUTS=()
on() {  # HOST COMMAND ARG...: a bounded call on a rank's box
  local h=$1; shift
  if [[ $h == local ]]; then timeout -k 10 "${ON_S:-120}" "$@"
  else
    # shellcheck disable=SC2086
    timeout -k 10 "${ON_S:-120}" ssh -n $SSH_OPTS -- "$h" "$(shq "$@")"
  fi
}

for r in 0 1 2 3; do
  [[ $r == 0 || -n "${HOSTS[r]}" ]] || fail E_CONFIG "rank $r: WORKER$([[ $r == 1 ]] || echo $r) is empty (scripts/local.sh)."
  if [[ "$DRY_RUN" == 1 ]]; then HOMES[r]=${EXLR8_WORKER_HOME:-$HOME}; [[ $r == 0 ]] && HOMES[r]=$HOME
  else HOMES[r]=$(on "${HOSTS[r]}" sh -c 'printf "%s\n" "$HOME"'); fi
  [[ "${HOMES[r]}" == /* ]] || fail E_WORKER "rank $r (${HOSTS[r]}): cannot read its home."
  if (( r == 0 )); then OUTS[r]=$MODEL_DIR; else OUTS[r]=${HOMES[r]}/$EXLR8_LEAF; fi
  (( r < 2 )) || [[ "${HOMES[r]}" == "${HOMES[1]}" ]] || fail E_WORKER "rank $r's home ${HOMES[r]} is not rank 1's ${HOMES[1]}: WORKER_MODEL_DIR is one path for the workers."
done
[[ "${WORKER_MODEL_DIR}" == "${OUTS[1]}" ]] || fail E_CONFIG "WORKER_MODEL_DIR=$WORKER_MODEL_DIR but the workers' directory is ${OUTS[1]}: set EXLR8_WORKER_HOME=${HOMES[1]} (scripts/exlr8/hosts.env)."
echo "ranks: 0 local ${OUTS[0]}; 1 ${HOSTS[1]} ${OUTS[1]}; 2 ${HOSTS[2]} ${OUTS[2]}; 3 ${HOSTS[3]} ${OUTS[3]}"

ship_tools() {  # r
  local r=$1 h=${HOSTS[$1]} dest
  dest="${HOMES[r]}/.cache/aevonix-glm53-nvfp4/exlr8-tools"
  if [[ "$DRY_RUN" == 1 ]]; then plan "copy scripts/exlr8/ to rank $r ($h): $dest"; return 0; fi
  if [[ $h == local ]]; then mkdir -p "$dest" && cp -p "$TOOLS_SRC"/*.py "$TOOLS_SRC"/*.sh "$TOOLS_SRC"/index-sha-*.json "$dest"/
  else
    on "$h" mkdir -p -- "$dest" && timeout -k 10 300 scp -q -p "$TOOLS_SRC"/*.py "$TOOLS_SRC"/*.sh "$TOOLS_SRC"/index-sha-*.json "$h:$dest/"
  fi
}
box() {  # r EXTRA...: box.sh on rank r's own box
  local r=$1 key; shift
  local -a settings=("EXLR8_RELEASE=$EXLR8_RELEASE" "EXLR8_DIR=${OUTS[r]}" "EXLR8_DENSE=$EXLR8_DENSE")
  for key in EXLR8_SRC EXLR8_MIN_FREE_GB EXLR8_MIN_AVAIL_GIB EXLR8_KEEP_RELEASE; do
    if [[ -n ${!key:-} ]]; then settings+=("$key=${!key}"); fi
  done
  on "${HOSTS[r]}" env "${settings[@]}" bash "${HOMES[r]}/.cache/aevonix-glm53-nvfp4/exlr8-tools/box.sh" --rank "$r" --ranks "$RANKS" "$@"
}

for r in 0 1 2 3; do ship_tools "$r" || fail E_WORKER "rank $r: copying the converters."; done
if [[ $mode == --status ]]; then
  for r in 0 1 2 3; do
    if [[ "$DRY_RUN" == 1 ]]; then plan "rank $r: box.sh --status"; else ON_S=300 box "$r" --status || true; fi
  done
  exit 0
fi

echo 'Converting each rank on its own box (logs in '"$LOGS"'):'
pids=()
for r in 0 1 2 3; do
  if [[ "$DRY_RUN" == 1 ]]; then plan "rank $r (${HOSTS[r]}): box.sh --rank $r  -> ${OUTS[r]}"; continue; fi
  ( rc=0; ON_S=${EXLR8_CONVERT_S:-21600} box "$r" || rc=$?; echo "rank $r exit $rc"; exit "$rc" ) > "$LOGS/rank$r.log" 2>&1 &
  pids+=($!)
done
if [[ "$DRY_RUN" != 1 ]]; then
  failed=()
  for r in 0 1 2 3; do
    wait "${pids[r]}" || failed+=("$r")
    tail -1 "$LOGS/rank$r.log"
  done
  (( ${#failed[@]} == 0 )) || fail E_CONVERT "Ranks ${failed[*]} did not finish. See $LOGS/rank*.log."
fi

identity=''
for r in 0 1 2 3; do
  if [[ "$DRY_RUN" == 1 ]]; then plan "validate rank $r's $EXLR8_DENSE EXLR8 marker and compare the config and set digests"; continue; fi
  digest=$(on "${HOSTS[r]}" env "EXLR8_RELEASE=$EXLR8_RELEASE" python3 -IB "${HOMES[r]}/.cache/aevonix-glm53-nvfp4/exlr8-tools/dirstate.py" \
    checkpoint "${OUTS[r]}" "$r" "$RANKS" "$EXLR8_DENSE") || fail E_CONVERT "Cannot check rank $r's checkpoint identity."
  [[ "$digest" =~ ^[0-9a-f]{64}\ [0-9a-f]{64}$ ]] || fail E_CONVERT "Rank $r did not return both checkpoint digests."
  [[ $r != 0 ]] || identity=$digest
  [[ "$digest" == "$identity" ]] || fail E_CONVERT "Rank $r's config or checkpoint set differs from rank 0."
done

# the DFlash2 drafter on the workers (WORKER_WEIGHTS=check: every worker reads its own copy, at WORKER_DFLASH2_DIR)
if [[ "$DRAFTER" == dflash2 ]]; then
  if [[ "$DRY_RUN" == 1 ]]; then plan "rsync $DFLASH2_DIR to each worker's $WORKER_DFLASH2_DIR (temporary name, then renamed; never into an existing one)"
  else
    [[ -f "$DFLASH2_DIR/config.json" ]] || fail E_WEIGHTS "no drafter in $DFLASH2_DIR on this Spark: scripts/prepare.sh --weights first."
    want=$(cd "$DFLASH2_DIR" && find -L . -type f -printf '%P %s\n' | sort)
    for r in 1 2 3; do
      h=${HOSTS[r]}
      if on "$h" test -e "$WORKER_DFLASH2_DIR"; then
        [[ "$(on "$h" bash -o pipefail -c 'cd -- "$1" && find -L . -type f -printf "%P %s\n" | sort' _ "$WORKER_DFLASH2_DIR")" == "$want" ]] \
          || fail E_WEIGHTS "$h: $WORKER_DFLASH2_DIR exists but differs from this Spark's drafter: not writing into it."
        continue
      fi
      on "$h" test -e "$WORKER_DFLASH2_DIR.x8part" && fail E_WEIGHTS "$h: $WORKER_DFLASH2_DIR.x8part exists: not writing into it."
      echo "drafter: copying DFlash2 to $h"
      timeout -k 30 1800 rsync -aLs --mkpath "$DFLASH2_DIR/" "$h:$WORKER_DFLASH2_DIR.x8part/" >/dev/null || fail E_WEIGHTS "copying the drafter to $h."
      on "$h" mv -T -- "$WORKER_DFLASH2_DIR.x8part" "$WORKER_DFLASH2_DIR" || fail E_WEIGHTS "placing the drafter on $h."
    done
  fi
fi
if [[ "$DRY_RUN" != 1 ]]; then echo "EXLR8 $EXLR8_RELEASE ($EXLR8_PROFILE, $EXLR8_DENSE) is ready on all four ranks."; fi
start_args=(MODEL_FORMAT=exlr8 "EXLR8_RELEASE=$EXLR8_RELEASE" "EXLR8_DENSE=$EXLR8_DENSE")
[[ "$MODEL_DIR" == "$HOME/$EXLR8_LEAF" ]] || start_args+=("MODEL_DIR=$MODEL_DIR")
echo 'Start with this command (TP6_RANKS=4 is implied).'
plan env "${start_args[@]}" ./start-tp6.sh
