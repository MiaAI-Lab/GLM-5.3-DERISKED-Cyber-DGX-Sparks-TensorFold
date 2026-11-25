#!/usr/bin/env bash
# Test privileged compaction and read-only fabric reporting without a Spark cluster.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
fn=$(awk '/^compact_memory\(\) \{/ { on = 1 } on { print } on && /^\}/ { exit }' scripts/prepare.sh)
[[ -n "$fn" ]]
{
  cat <<'SCRIPT'
set -u
log() { printf '%s\n' "$*"; }
warn() { printf '[W_COMPACT] %s\n' "$*" >&2; }
fail() { printf '[E_COMPACT] %s\n' "$*" >&2; exit 1; }
sudo() { [[ "$1" == -n ]] && shift; "$@"; }
sh() { printf 'compaction attempted\n'; }
SCRIPT
  printf '%s\n' "$fn"
  cat <<'SCRIPT'
COMPACT_MEMORY=1 DRY_RUN=0
compact_memory
COMPACT_MEMORY=0 DRY_RUN=0
compact_memory
SCRIPT
} > "$tmp/compact.sh"
out=$(bash "$tmp/compact.sh")
[[ "$out" == *'compaction attempted'* ]]
[[ $(grep -c 'compaction attempted' <<< "$out") == 1 ]]

fn=$(awk '/^fabric_status\(\) \{/ { on = 1 } on { print } on && /^\}/ { exit }' scripts/tp6.sh)
[[ -n "$fn" ]]
{
  cat <<'SCRIPT'
set -u
KERNEL_RELEASE=test-kernel NVIDIA_DRIVER=test-driver NVIDIA_CLOCK=1800
log() { printf '%s\n' "$*"; }
warn() { printf '[W_FABRIC] %s\n' "$*" >&2; }
fail() { printf '[E_FABRIC] %s\n' "$*" >&2; exit 1; }
command() { [[ "$2" == ib_write_bw ]]; }
rdma() { printf 'link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev rocep1s0f0\n'; }
ib_write_bw() { printf -- '-d <device>, --report_gbits\n'; }
SCRIPT
  printf '%s\n' "$fn"
  cat <<'SCRIPT'
fabric_status
SCRIPT
} > "$tmp/fabric.sh"
out=$(bash "$tmp/fabric.sh" 2>&1)
grep -q 'test-kernel' <<< "$out"
grep -q 'test-driver' <<< "$out"
grep -q '1800' <<< "$out"
grep -q 'state ACTIVE' <<< "$out"
grep -q 'ib_write_bw' <<< "$out"
# Remote functions need every shell helper serialized by wfn_n.
grep -q 'wfn_n "$i" log compact_memory' start-tp6-main.sh
grep -q 'wfn_n "$i" log warn fail fabric_status' start-tp6-main.sh
echo 'startup-fabric-compaction: pass'
