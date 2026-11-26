#!/usr/bin/env bash
# pair_links' edge list, fabric_classify's verdicts (ring only when asked) and fabric_cycle's ring check, no Spark.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
pl=$(awk '/^pair_links\(\) \{/ { on = 1 } on { print } on && /^\}$/ { exit }' scripts/tp6.sh)
fc=$(awk '/^fabric_classify\(\) \{/ { on = 1 } on { print } on && /^\}$/ { exit }' scripts/tp6.sh)
fy=$(awk '/^fabric_cycle\(\) \{/ { on = 1 } on { print } on && /^\}$/ { exit }' scripts/tp6.sh)
[[ -n "$pl" && -n "$fc" && -n "$fy" ]]

# node_inventory lines as detect_links feeds them: "<rank> <netdev> <rdma> <cidr> <gid>", "<rank> default <netdev>".
# A ring: cable k joins rank k's port 0 to rank (k+1)%4's port 1, two subnets a cable, one a PCIe rail.
ring_inventory() {
  cat <<'INV'
0 default enP7s7
0 enp1s0f0np0 rocep1s0f0 10.20.0.11/24 1
0 enP2p1s0f0np0 roceP2p1s0f0 10.20.1.11/24 1
0 enp1s0f1np1 rocep1s0f1 10.20.6.11/24 1
0 enP2p1s0f1np1 roceP2p1s0f1 10.20.7.11/24 1
1 default enP7s7
1 enp1s0f1np1 rocep1s0f1 10.20.0.12/24 1
1 enP2p1s0f1np1 roceP2p1s0f1 10.20.1.12/24 1
1 enp1s0f0np0 rocep1s0f0 10.20.2.12/24 1
1 enP2p1s0f0np0 roceP2p1s0f0 10.20.3.12/24 1
2 default enP7s7
2 enp1s0f1np1 rocep1s0f1 10.20.2.13/24 1
2 enP2p1s0f1np1 roceP2p1s0f1 10.20.3.13/24 1
2 enp1s0f0np0 rocep1s0f0 10.20.4.13/24 1
2 enP2p1s0f0np0 roceP2p1s0f0 10.20.5.13/24 1
3 default enP7s7
3 enp1s0f1np1 rocep1s0f1 10.20.4.14/24 1
3 enP2p1s0f1np1 roceP2p1s0f1 10.20.5.14/24 1
3 enp1s0f0np0 rocep1s0f0 10.20.6.14/24 1
3 enP2p1s0f0np0 roceP2p1s0f0 10.20.7.14/24 1
INV
}
# a switch: every rank on one shared subnet pair
switch_inventory() {
  cat <<'INV'
0 default enP7s7
0 enp1s0f0np0 rocep1s0f0 10.30.0.11/24 1
0 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.11/24 1
1 default enP7s7
1 enp1s0f0np0 rocep1s0f0 10.30.0.12/24 1
1 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.12/24 1
2 default enP7s7
2 enp1s0f0np0 rocep1s0f0 10.30.0.13/24 1
2 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.13/24 1
3 default enP7s7
3 enp1s0f0np0 rocep1s0f0 10.30.0.14/24 1
3 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.14/24 1
INV
}

{
  printf '%s\n' "$pl" "$fc" "$fy"
  cat <<'SCRIPT'
set -euo pipefail
fail() { printf '[%s] %s\n' "$1" "$2" >&2; exit 1; }
TP=4
ring_inventory() {
  cat <<'INV2'
0 default enP7s7
0 enp1s0f0np0 rocep1s0f0 10.20.0.11/24 1
0 enP2p1s0f0np0 roceP2p1s0f0 10.20.1.11/24 1
0 enp1s0f1np1 rocep1s0f1 10.20.6.11/24 1
0 enP2p1s0f1np1 roceP2p1s0f1 10.20.7.11/24 1
1 default enP7s7
1 enp1s0f1np1 rocep1s0f1 10.20.0.12/24 1
1 enP2p1s0f1np1 roceP2p1s0f1 10.20.1.12/24 1
1 enp1s0f0np0 rocep1s0f0 10.20.2.12/24 1
1 enP2p1s0f0np0 roceP2p1s0f0 10.20.3.12/24 1
2 default enP7s7
2 enp1s0f1np1 rocep1s0f1 10.20.2.13/24 1
2 enP2p1s0f1np1 roceP2p1s0f1 10.20.3.13/24 1
2 enp1s0f0np0 rocep1s0f0 10.20.4.13/24 1
2 enP2p1s0f0np0 roceP2p1s0f0 10.20.5.13/24 1
3 default enP7s7
3 enp1s0f1np1 rocep1s0f1 10.20.4.14/24 1
3 enP2p1s0f1np1 roceP2p1s0f1 10.20.5.14/24 1
3 enp1s0f0np0 rocep1s0f0 10.20.6.14/24 1
3 enP2p1s0f0np0 roceP2p1s0f0 10.20.7.14/24 1
INV2
}
switch_inventory() {
  cat <<'INV2'
0 default enP7s7
0 enp1s0f0np0 rocep1s0f0 10.30.0.11/24 1
0 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.11/24 1
1 default enP7s7
1 enp1s0f0np0 rocep1s0f0 10.30.0.12/24 1
1 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.12/24 1
2 default enP7s7
2 enp1s0f0np0 rocep1s0f0 10.30.0.13/24 1
2 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.13/24 1
3 default enP7s7
3 enp1s0f0np0 rocep1s0f0 10.30.0.14/24 1
3 enP2p1s0f0np0 roceP2p1s0f0 10.30.1.14/24 1
INV2
}
ring_out=$(ring_inventory | pair_links 0 1 2 3)
# rank 0's neighbours on the ring are ranks 1 and 3; the diagonals are named as uncabled
grep -qx 'peer 1 10.20.0.12' <<<"$ring_out"
grep -qx 'peer 3 10.20.6.14' <<<"$ring_out"
grep -qx 'missing 0 2' <<<"$ring_out"
grep -qx 'missing 1 3' <<<"$ring_out"
# every rank carries all four of its RoCE devices (both cabled ports, both their rails)
grep -qx 'rank 0 enP7s7 rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1 1' <<<"$ring_out"
# the edges form exactly the closed ring 0-1-2-3-0
mapfile -t edges < <(grep '^edge' <<<"$ring_out" | cut -d' ' -f2,3)
(( ${#edges[@]} == 4 ))
fabric_cycle "${edges[@]}"
ring_missing=$(grep '^missing' <<<"$ring_out" | awk '{ printf " %s-%s", $2, $3 }')
# a ring classifies only when asked: FABRIC_TOPOLOGY=ring accepts it (dash-form pairs, as detect_links builds them)
FABRIC_TOPOLOGY=ring
[[ "$(fabric_classify "$ring_missing")" == ring ]]
# auto and switch refuse the ring by name; fail's exit 1 must not leave the test (a subshell contains it)
for t in auto switch; do
  if (FABRIC_TOPOLOGY=$t fabric_classify "$ring_missing") 2>/dev/null; then exit 1; fi
done
FABRIC_TOPOLOGY=auto
switch_out=$(switch_inventory | pair_links 0 1 2 3)
[[ -z "$(grep '^missing' <<<"$switch_out")" ]]
[[ "$(fabric_classify '')" == switch ]]
mapfile -t sedges < <(grep '^edge' <<<"$switch_out" | cut -d' ' -f2,3)
(( ${#sedges[@]} == 6 ))
if fabric_cycle "${sedges[@]}"; then exit 1; fi
# FABRIC_TOPOLOGY=ring on a switch is refused with the fix named
if (FABRIC_TOPOLOGY=ring fabric_classify '') 2>/dev/null; then exit 1; fi
# an open chain 0-1-2-3 is neither a switch nor a ring: each end has one neighbour
chain_missing=' 0-2 0-3 1-3'
FABRIC_TOPOLOGY=ring
[[ -z "$(fabric_classify "$chain_missing")" ]]
# the same chain under auto is empty too (a shape no layout names), not a ring diagnosis
[[ -z "$(FABRIC_TOPOLOGY=auto fabric_classify "$chain_missing")" ]]
if fabric_cycle '0 1' '1 2' '2 3'; then exit 1; fi
# edges in the wrong WORKER order are a closed ring the rank order does not walk: refused
if fabric_cycle '0 1' '1 3' '3 2' '2 0'; then exit 1; fi
# a 6-node cycle fabric_cycle refuses: the recipe's ring is four
if fabric_cycle '0 1' '1 2' '2 3' '3 4' '4 5' '5 0'; then exit 1; fi
SCRIPT
} > "$tmp/fabric.sh"
bash "$tmp/fabric.sh"
echo 'fabric-topology: pass'
