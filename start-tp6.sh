#!/usr/bin/env bash
# Serve nvidia/GLM-5.3-NVFP4 on six DGX Sparks on a switch at TP=6 (TensorFold native CUDA engine). This is a thin
# wrapper: ./start-tp6-main.sh does the work (checks, image, NFS weights from spark1, six ranks, smoke test). WORKER,
# WORKER2, WORKER3, WORKER4 and WORKER5 (scripts/local.sh) may be in any order; NCCL picks its own devices on the switch.
#
# Usage: ./start-tp6.sh [restart]   (./stop-tp6.sh stops it; DRY_RUN=1 ./start-tp6.sh prints the links it finds and every
# rank's command, and changes nothing)
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
# TP6 unified + neon stack (patches 0218-0241): policy mask 0x7fcffd1c (DSA_INDEX_FAST + index_scan_split + dflash_mesh
# + dsa_pdl + counts_local + absorb_one + scatter_inplace + L2PF bit 16) + DFlash2 4-bit down / split fc / split qkv+o
# + folded shared-down epilogue + balanced per-expert TP6 sharding (TF_GLM_TP6_BAL). NEON_STACK=0 disables the env block.
if [[ "${NEON_STACK:-1}" == 1 ]]; then
  neon_env="TF_GLM_POLICY_REQUESTS=1 TF_GLM_POLICY_DEFAULT=0x7fcffd1c TF_GLM_UNI_EY_BF16=1 TF_GLM_UNI_KSPLIT_ALL=1 TF_GLM_UNI_X3_ONCE=1"
  neon_env+=" TF_GLM_NEON_BUFS=1 TF_GLM_NEON_DF_DOWN=1 TF_GLM_NEON_DF_FC=1 TF_GLM_NEON_FOLD=1"
  export RANK_EXTRA_ENV="$neon_env${RANK_EXTRA_ENV:+ $RANK_EXTRA_ENV}"
fi
exec "$(dirname -- "$(readlink -f -- "$0")")/start-tp6-main.sh" "$@"
