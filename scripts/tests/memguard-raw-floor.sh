#!/usr/bin/env bash
# The runtime memory guard must compare raw bytes against its floor, not a one-decimal GiB rounding of them (Codex
# review r2, finding 5): a sample just under the floor rounds up onto it (2 GiB - 2 KiB prints as "2.0"), so a guard
# that sampled mem_avail_gb read it as at-least-the-floor and left a Spark running below a raw 2 GiB MemAvailable.
# The test runs the real tp6_start_memguard with a fake /proc/meminfo that holds 2048 bytes less than the 2 GiB floor
# and fails unless the guard still stops the server; rounding may decide only what the log prints, never the trigger.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
guard=$(awk '/^tp6_start_memguard\(\) \{/ { on = 1 } on { print } on && /^\}/ { exit }' scripts/tp6.sh)
[[ -n "$guard" ]] || { echo 'tp6_start_memguard not found in scripts/tp6.sh' >&2; exit 1; }
counting=$(awk '/^mem_counting\(\) \{/ { on = 1 } on { print } on && /^\}/ { exit }' scripts/memavail.sh)
[[ -n "$counting" ]] || { echo 'mem_counting not found in scripts/memavail.sh' >&2; exit 1; }
printf '#!/usr/bin/env bash\ntouch "$(dirname "$0")/stopped"\n' > "$tmp/stop.sh"
chmod +x "$tmp/stop.sh"
mkdir -p "$tmp/log"
{ cat <<'SCRIPT'
set -u
docker() { echo true; }
at_least() { python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)' "$1" "$2"; }
whost() { echo worker; }
worker_ids() { echo 1 2 3 4 5; }
log() { :; }
SCRIPT
  printf '%s\n' "$counting" "$guard"
  cat <<'SCRIPT'
# Both samplers route mem_counting at the fake /proc/meminfo (no zoneinfo, so no parked pages): mem_avail_gb is what
# the pre-fix guard sampled, mem_avail_bytes what the raw-bytes guard samples. Whichever one the extracted guard
# calls decides whether this test sees the rounding or the raw bytes.
mem_avail_gb() { mem_counting "$MEMINFO" /dev/null | awk '{ print $1 }'; }
mem_avail_bytes() { mem_counting "$MEMINFO" /dev/null | awk '{ print $4 }'; }
tp6_worker_avail_gb() { mem_avail_gb; }
tp6_worker_avail_bytes() { mem_avail_bytes; }
tp6_start_memguard
for i in $(seq "$WAIT_S"); do [[ -f "$REPO_ROOT/stopped" ]] && break; sleep 1; done
if [[ -f "$REPO_ROOT/stopped" ]]; then echo stopped; exit 0; fi
kill "$MEMGUARD_PID" 2>/dev/null || true
echo never
exit 1
SCRIPT
} > "$tmp/run.sh"
# Under the floor by 2048 bytes: the one-decimal sample reads "2.0" GiB, so the pre-fix guard never stopped.
printf 'MemTotal:  33000000 kB\nMemAvailable:  2097100 kB\n' > "$tmp/under.meminfo"
if out=$(MEMINFO="$tmp/under.meminfo" WAIT_S=15 TP=6 MEMGUARD=1 DRY_RUN=0 MEMGUARD_INTERVAL_S=1 MEMGUARD_HOLD_S=4 \
    MEMGUARD_FLOOR_GIB=2 RUN_LOG_DIR="$tmp/log" REPO_ROOT="$tmp" CONTAINER_NAME=test bash "$tmp/run.sh"); then
  grep -q '2\.0 GiB available, under the 2 GiB floor' "$tmp/log/memguard.log" \
    || { echo 'memguard-raw-floor: the trigger line does not show the rounded sample' >&2; exit 1; }
else
  echo "memguard-raw-floor: the guard did not stop a Spark 2048 bytes under the floor (got: $out)" >&2; exit 1
fi
# Above the floor by 16640 bytes: the guard must keep the server up.
rm -f "$tmp/stopped"
: > "$tmp/log/memguard.log"
printf 'MemTotal:  33000000 kB\nMemAvailable:  2097200 kB\n' > "$tmp/over.meminfo"
if out=$(MEMINFO="$tmp/over.meminfo" WAIT_S=8 TP=6 MEMGUARD=1 DRY_RUN=0 MEMGUARD_INTERVAL_S=1 MEMGUARD_HOLD_S=4 \
    MEMGUARD_FLOOR_GIB=2 RUN_LOG_DIR="$tmp/log" REPO_ROOT="$tmp" CONTAINER_NAME=test bash "$tmp/run.sh"); then
  echo 'memguard-raw-floor: the guard stopped a Spark above the floor' >&2; exit 1
fi
echo 'memguard-raw-floor: pass'
