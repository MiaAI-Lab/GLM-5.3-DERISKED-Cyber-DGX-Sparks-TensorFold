#!/usr/bin/env bash
# Serve this recipe on four DGX Sparks cabled as a ring, without a switch (TP=4): each Spark's two QSFP ports cabled
# to its two neighbours, one subnet per cable. Rank 0 (the API) is this Spark; WORKER, WORKER2 and WORKER3 in
# scripts/local.sh go in the ring's order from here (WORKER and WORKER3 are this Spark's neighbours, WORKER2 is across
# the ring). Everything else is ./start.sh: the same checkpoints (MODEL_FORMAT k4 or derisked), image, patches and
# serving defaults, with the ring's NCCL settings from the 2x-pair recipe's four-Spark launcher
# (MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104, scripts/tp4.sh: NCCL_ALGO=Ring, per-rank graph files, subnet-aware routing).
#
# What differs from the switch: ROCE_MESH (one-shot RoCE writes to every peer, patch 0140) needs every Spark to reach
# every other directly, so it is off here. ROCE_RING=on (the default; patch 0108, the 1104 recipe's ring all-gather)
# writes the decode gathers to both neighbours and relays half to the Spark across; ROCE_RING=off leaves them to NCCL.
# Measured on a ring (K4, one stream): prose 29.9, code 46.3 tok/s (NCCL alone 27.8 / 42.6), prefill ~1,330 tok/s at 18K.
# Prompt chunks already use all-gathers (PREFILL_REDUCE=gather), which a ring carries.
#
# Weights: the Spark across the ring has no cable to this one, so NFS to it goes over the LAN (1 Gbit) or a route
# through a neighbour. Copying the checkpoint to every worker over the QSFP links and serving with
# WORKER_WEIGHTS=check (WORKER_MODEL_DIR, WORKER_DFLASH2_DIR) avoids that; see README "Four Sparks without a switch".
#
# Usage: ./start-switchless.sh [restart]   (./stop-tp6.sh stops it; DRY_RUN=1 ./start-switchless.sh prints the ring it
# finds and every rank's command, and changes nothing)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
[[ "${ROCE_MESH:-0}" == 0 ]] \
  || { echo "[E_CONFIG] ROCE_MESH=$ROCE_MESH: the mesh gather writes to every Spark directly and needs a switch. Unset it (or set 0) for a ring." >&2; exit 1; }
export FABRIC_EXPECT=ring ROCE_MESH=0
exec ./start.sh "$@"
