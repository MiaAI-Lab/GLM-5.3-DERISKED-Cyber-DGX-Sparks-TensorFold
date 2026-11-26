#!/usr/bin/env bash
# Stop the ranks of ./start.sh (TP4, four Sparks) or ./start-tp6.sh / ./start-tp6-main.sh (TP6, six Sparks), save one
# gzipped archive with their logs and keep the newest LOG_KEEP (10). All six Sparks are checked, so one script stops
# either layout. Rank 0 goes first: it drains its requests and ends the followers' steps; then each worker that has one.
# A rank that still runs after the STOP_TIMEOUT stop gets docker kill; one still running after the kill fails the
# stop, so a start aborts for a human instead of layering a second container onto a wedged one.
# Usage: ./stop.sh; DRY_RUN=1 ./stop.sh prints the plan and changes nothing.
# Only this recipe's containers (label tf.recipe=glm53-nvfp4) are stopped; any other container is left alone.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TP=6 GLM_TP6=1   # TP=6 makes worker_ids cover all five workers; a TP4 run leaves spark5/6 empty
# shellcheck source=scripts/prepare.sh
source ./scripts/prepare.sh
# shellcheck source=scripts/tp6.sh
source ./scripts/tp6.sh
[[ "$STOP_TIMEOUT" =~ ^[1-9][0-9]*$ && "$LOG_KEEP" =~ ^[1-9][0-9]*$ ]] || fail E_CONFIG 'STOP_TIMEOUT and LOG_KEEP must be positive integers.'
[[ "$CONTAINER_NAME" != tensorfold-zig-r0 && "$CONTAINER_NAME" != tensorfold-zig-r1 && "$CONTAINER_NAME" != dsv41-1017-exl3-tf ]] \
  || fail E_CONFIG "$CONTAINER_NAME is a DeepSeek recipe's container on these Sparks; this script never stops it."
for key in WORKER WORKER2 WORKER3 WORKER4 WORKER5; do
  [[ -n "${!key}" && "${!key}" =~ ^[A-Za-z0-9_.@:-]+$ ]] || fail E_CONFIG "$key must be an ssh target such as user@host (scripts/local.sh)."
done
warn_open_requests() {
  local host="$HOST" open
  [[ "$host" != 0.0.0.0 && "$host" != :: ]] || host=127.0.0.1
  [[ "$host" != *:* ]] || host="[$host]"
  open=$(curl -s --max-time 3 -w '\n%{http_code}' "http://$host:$PORT/health" 2>/dev/null | python3 -c '
import json, sys
body, _, code = sys.stdin.read().rpartition("\n")
r = json.loads(body) if code == "200" else {}
n = r["live"]["connections"] if type(r) is dict and r.get("error") is None else None
sys.exit(1) if type(n) is not int or n < 0 else print(n)' 2>/dev/null) || open='?'
  if [[ "$open" == '?' ]]; then
    warn W_OPEN 'Could not read the open requests from /health; any still open get TF_DRAIN_S to finish, then are cut off.'
  elif (( open > 0 )); then
    warn W_OPEN "$open request(s) open; they get TF_DRAIN_S to finish, then are cut off."
  fi
}
if [[ "$DRY_RUN" == 1 ]]; then
  log "Stop each rank only if its container carries the label tf.recipe=glm53-nvfp4, rank 0 first (up to $STOP_TIMEOUT s each):"
  plan docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME"
  for i in $(worker_ids); do plan ssh "$(wtarget "$i")" docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME"; done
  log "  Save each stopped rank's Docker output as $LOG_DIR/glm53-nvfp4-tp6-<timestamp>.tar.gz; keep the newest $LOG_KEEP archives."
  plan docker rm "$CONTAINER_NAME"
  for i in $(worker_ids); do plan ssh "$(wtarget "$i")" docker rm "$CONTAINER_NAME"; done
  exit 0
fi
command -v docker >/dev/null || fail E_DEPENDENCY 'Missing command: docker.'
docker info >/dev/null 2>&1 || fail E_DOCKER 'Cannot talk to Docker. Start the daemon and check socket permissions.'
if [[ "${GLM_STOP_LOCK_HELD:-0}" != 1 ]]; then
  mkdir -p "$STATE_DIR"
  exec 8>"$STATE_DIR/start.lock"
  flock -n 8 || fail E_LOCK 'Another start, stop or preparation is in progress.'
fi

declare -a there=(0 0 0 0 0 0) dead=(0 0 0 0 0 0) foreign=(0 0 0 0 0 0)
docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && there[0]=1
# a worker that gives no answer within REMOTE_TIMEOUT (scripts/tp6.sh) is skipped, so a dead Spark costs one timeout
for i in $(worker_ids); do
  rc=0; wssh_n "$i" docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 || rc=$?
  if (( rc == 0 )); then there[i]=1
  elif (( rc == 124 || rc == 137 || rc == 255 )); then
    dead[i]=1; warn W_WORKER "Rank $i's Spark $(whost "$i") gave no answer within $REMOTE_TIMEOUT s (ssh/docker); it is skipped."
  fi
done
dead_ranks=$(for i in $(worker_ids); do (( dead[i] )) && printf '%s ' "$i"; done; true)
# all six Sparks counted: a TP4 run leaves spark5/6 empty, a TP6 leftover may sit there alone
if (( there[0] + there[1] + there[2] + there[3] + there[4] + there[5] == 0 )); then
  [[ -z "$dead_ranks" ]] || fail E_WORKER "Nothing to stop on the Sparks that answered; rank(s) ${dead_ranks% } did not answer. Check those Sparks (docker ps there)."
  log "No container named $CONTAINER_NAME on any of the six Sparks. Nothing to stop."; exit 0
fi
# Each Spark is checked for this recipe's label before anything is stopped.
if (( there[0] )); then
  [[ "$(docker inspect -f '{{index .Config.Labels "tf.recipe"}}' "$CONTAINER_NAME")" == glm53-nvfp4 ]] \
    || fail E_STOP "$CONTAINER_NAME here was not started by this recipe (no tf.recipe=glm53-nvfp4 label); not touching it."
fi
for i in $(worker_ids); do
  (( there[i] )) || continue
  if [[ "$(wssh_n "$i" docker inspect -f '{{index .Config.Labels "tf.recipe"}}' "$CONTAINER_NAME" 2>/dev/null || true)" != glm53-nvfp4 ]]; then
    # spark5/6 (ranks 4 and 5) are free at TP4 and may run another recipe's container of the same name: leave it alone
    # and stop this recipe's ranks; on ranks 1-3 a foreign container still stops everything
    if (( i >= 4 )); then
      warn W_FOREIGN "$CONTAINER_NAME on $(whost "$i") belongs to another recipe; it is left alone and not counted."
      there[i]=0; foreign[i]=1; continue
    fi
    fail E_STOP "$CONTAINER_NAME on $(whost "$i") was not started by this recipe; not touching it."
  fi
done
if (( there[0] + there[1] + there[2] + there[3] + there[4] + there[5] == 0 )); then
  log "No container of this recipe named $CONTAINER_NAME on the Sparks. Nothing to stop."; exit 0
fi

RUN_LOG_DIR=$(readlink -f "$LOG_DIR/current" 2>/dev/null || true)
[[ -n "$RUN_LOG_DIR" && -d "$RUN_LOG_DIR" ]] || { RUN_LOG_DIR="$LOG_DIR/run-tp6-stopped-$(date -u +%Y%m%dT%H%M%SZ)-$$"; mkdir -p "$RUN_LOG_DIR"; }

if (( there[0] )) && [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == true ]]; then
  warn_open_requests
  log "Stopping rank 0 here (up to $STOP_TIMEOUT s; requests in progress get TF_DRAIN_S to finish)."
  docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]; then
    warn W_STOP "Rank 0 still runs after the ${STOP_TIMEOUT}s stop; docker kill."
    docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]; then
    fail E_STOP "Rank 0 still runs after stop and kill. A human must clean it up (docker ps; docker rm -f $CONTAINER_NAME) before a start can be trusted."
  fi
fi
for i in $(worker_ids); do
  (( there[i] )) || continue
  log "Stopping rank $i on $(whost "$i")."
  wssh_n "$i" docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if [[ "$(wssh_n "$i" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]; then
    warn W_STOP "Rank $i still runs on $(whost "$i") after the ${STOP_TIMEOUT}s stop; docker kill."
    wssh_n "$i" docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  if [[ "$(wssh_n "$i" docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]]; then
    fail E_STOP "Rank $i on $(whost "$i") still runs after stop and kill. A human must clean it up there (docker ps; docker rm -f $CONTAINER_NAME)."
  fi
done

mkdir -p "$LOG_DIR"
(( there[0] == 0 )) || docker logs --timestamps "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank0.log" 2>&1 || warn W_LOG 'Could not save rank 0 output.'
(( there[0] == 0 )) || docker inspect "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank0.json" 2>/dev/null || true
for i in $(worker_ids); do
  (( there[i] )) || continue
  wssh_n "$i" docker logs --timestamps "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank$i.log" 2>&1 || warn W_LOG "Could not save rank $i output."
  wssh_n "$i" docker inspect "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank$i.json" 2>/dev/null || true
done

archive="$LOG_DIR/glm53-nvfp4-tp6-$(date -u +%Y%m%dT%H%M%SZ)-$$.tar.gz"
tar -czf "$archive.tmp" -C "$RUN_LOG_DIR" . || fail E_LOG 'Could not archive logs. Check LOG_DIR space and permissions; the stopped containers were retained.'
mv -- "$archive.tmp" "$archive"
(( there[0] == 0 )) || docker rm "$CONTAINER_NAME" >/dev/null || fail E_STOP 'Logs were saved, but Docker could not remove rank 0.'
for i in $(worker_ids); do
  (( there[i] )) || continue
  wssh_n "$i" docker rm "$CONTAINER_NAME" >/dev/null 2>&1 \
    || warn W_STOP "Logs were saved, but rank $i on $(whost "$i") could not be removed; remove it there with: docker rm $CONTAINER_NAME"
done
if [[ "$RUN_LOG_DIR" == "$LOG_DIR"/run-tp6-* && ! -L "$RUN_LOG_DIR" ]]; then
  rm -rf -- "$RUN_LOG_DIR"
  [[ ! -L "$LOG_DIR/current" || "$(readlink -- "$LOG_DIR/current")" != "$RUN_LOG_DIR" ]] || rm -- "$LOG_DIR/current"
fi
python3 - "$LOG_DIR" "$LOG_KEEP" <<'PY'
from pathlib import Path
import sys
archives = sorted(Path(sys.argv[1]).glob("glm53-nvfp4-tp6-*.tar.gz"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
for path in archives[int(sys.argv[2]):]:
    path.unlink()
PY
# A worker's env file can hold a key you exported, so it does not outlive the ranks.
for i in $(worker_ids); do wssh_n "$i" sh -c 'rm -f "$HOME/.cache/aevonix-glm53/zig-home/state/rank.env"' >/dev/null 2>&1 || true; done
# Verified, not assumed: no serving container of this recipe (by name, or by the tf.recipe=glm53-nvfp4 label) is left
# on a Spark that answered. A stop that leaves one exits non-zero, so the next start cannot mistake it for its own.
left=''
[[ -z "$( { docker ps -aq --filter "name=^${CONTAINER_NAME}\$"; docker ps -aq --filter label=tf.recipe=glm53-nvfp4; } 2>/dev/null)" ]] || left+="$(hostname -s) "
for i in $(worker_ids); do
  (( dead[i] )) && continue
  if (( ${foreign[i]:-0} )); then   # another recipe's container of the same name: only this recipe's label counts
    ids=$(wssh_n "$i" sh -c "docker ps -aq --filter label=tf.recipe=glm53-nvfp4" 2>/dev/null) \
      || { left+="$(whost "$i")(no answer) "; continue; }
    [[ -z "$ids" ]] || left+="$(whost "$i") "; continue
  fi
  ids=$(wssh_n "$i" sh -c "docker ps -aq --filter 'name=^${CONTAINER_NAME}\$'; docker ps -aq --filter label=tf.recipe=glm53-nvfp4" 2>/dev/null) \
    || { left+="$(whost "$i")(no answer) "; continue; }
  [[ -z "$ids" ]] || left+="$(whost "$i") "
done
[[ -z "$left" ]] || fail E_STOP "A serving container is still there after the stop on: ${left% }. Remove it there (docker rm -f $CONTAINER_NAME) before the next start."
log "Stopped every rank ($(( there[0] + there[1] + there[2] + there[3] + there[4] + there[5] )) Sparks). Logs saved to $archive (newest $LOG_KEEP kept)."
[[ -z "$dead_ranks" ]] || fail E_WORKER "Rank(s) ${dead_ranks% } did not answer and were not stopped or logged. Check those Sparks (docker ps; docker rm -f $CONTAINER_NAME there)."
