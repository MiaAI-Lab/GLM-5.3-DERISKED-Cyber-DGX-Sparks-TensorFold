#!/usr/bin/env bash
# Serve this recipe on four DGX Sparks on a switch at TP=4: rank 0 (the API) on this Spark, ranks 1-3 on WORKER,
# WORKER2 and WORKER3 (scripts/local.sh). It is ./start-tp6.sh with TP6_RANKS=4 fixed: ./start-tp6-main.sh does the
# work (checks, image, weights, four ranks, smoke test). The checkpoint is wrldsuksgo2mars's K4 (EXL3 4-bit,
# wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 at 47af2334) in this Spark's Hugging Face cache; it is no longer on Hugging Face, so
# a local copy is needed. The workers mount it over NFS. MODEL_FORMAT=derisked (the DERISKED lane) serves Mia's AI Lab's
# GLM-5.3-DERISKED-Cyber-EXL3-4bpw (downloaded when missing) with the same serving defaults; another MODEL_FORMAT (or an
# exported MODEL_DIR / HF_CACHE_DIR) serves another checkpoint without them.
#
# Usage: ./start.sh [restart]   (./stop-tp6.sh stops it; DRY_RUN=1 ./start.sh prints the links it finds and every
# rank's command, and changes nothing)
set -euo pipefail
case "${1:-}" in
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac
[[ "${TP6_RANKS:-4}" == 4 ]] \
  || { echo "[E_CONFIG] ./start.sh runs four Sparks, and TP6_RANKS=$TP6_RANKS is exported: unset it, or use ./start-tp6.sh." >&2; exit 1; }
export TP6_RANKS=4 TP6_START=./start.sh
_start_fmt=${MODEL_FORMAT:-k4}
if [[ ( "$_start_fmt" == k4 || "$_start_fmt" == derisked ) && -z "${MODEL_DIR:-}" && -z "${HF_CACHE_DIR:-}" ]]; then
  # The checkpoint (scripts/config.sh): K4 by default, or the DERISKED lane (DERISKED_DIR when it holds it,
  # otherwise downloaded there at a pinned revision). The serving defaults below apply to both. K4 at 294,912 tokens and a
  # 3 GiB reserve: 0.3 GiB left on the tightest rank, needle at 265,420 passes, MemAvailable floor 2.81 GiB (2026-11-25).
  export MODEL_FORMAT=$_start_fmt
  # 109.77 GiB a rank at TP4: the unsplit 212,992 window left spark1 at 1.3 GiB and memguard stopped it (2026-11-21).
  # KV split serves 163,840 with 8.7 GiB available on spark1 (0.4 GiB left after the 8.5 GiB reserve), decode ~8% slower.
  # Owner 2026-11-21: reserve 4 GiB (the 8.5 was for TP6's uneven expert split; TP4 splits evenly) and no KV split
  # (split costs ~8% decode and the two-halves prefill schedule); unsplit fp8 KV is ~50 KB a token a rank.
  # Owner 2026-11-23: KV split back on for more context; the window is fitted to memory (auto: try CONTEXT_MAX, then
  # once the largest window the engine's memory refusal allows, remembered for later starts).
  # Owner 2026-11-24: reserve 3 GiB, no KV split again (the neon stack below was verified unsplit). The window is pinned,
  # not auto: unsplit, the engine gives no memory refusal to fit from, the kernel OOM-kills a rank at 1M and 524K (one
  # box fell to 0.14 GiB during that probe). 262,144 left 0.7 GiB over a 4 GiB reserve at ~50 KB a token a rank, so
  # 294,912 fits a 3 GiB reserve with ~0.14 GiB over it. CONTEXT=auto still probes.
  # Owner 2026-11-25: reserve 4 GiB (a 2.9 GiB process on spark1 stopped an idle server at 3), so back to 262,144.
  export KV_SPLIT=${KV_SPLIT:-0} CONTEXT=${CONTEXT:-262144} CONTEXT_MAX=${CONTEXT_MAX:-1048576} MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-4}
  # The unified + neon stack (patches 0218-0242; docs/unified-opt-results.md, docs/neon-opt-results.md): request policy
  # mask 0x7fcffd1c (adds dflash_mesh bit 17 for split DFlash2 qkv/o, dsa_pdl, counts_local, absorb_one, index_scan_split)
  # and their boot switches; every reply identical to the stack without them. NEON_STACK=0: none of it.
  # Placed before any RANK_EXTRA_ENV given, so a value set there wins.
  if [[ "${NEON_STACK:-1}" == 1 ]]; then
    neon_env="TF_GLM_POLICY_REQUESTS=1 TF_GLM_POLICY_DEFAULT=0x7fcffd1c TF_GLM_UNI_EY_BF16=1 TF_GLM_UNI_KSPLIT_ALL=1 TF_GLM_UNI_X3_ONCE=1"
    neon_env+=" TF_GLM_NEON_BUFS=1 TF_GLM_NEON_DF_DOWN=1 TF_GLM_NEON_DF_FC=1 TF_GLM_NEON_FOLD=1"
    export RANK_EXTRA_ENV="$neon_env${RANK_EXTRA_ENV:+ $RANK_EXTRA_ENV}"
  fi
fi
unset _start_fmt
exec "$(dirname -- "$(readlink -f -- "$0")")/start-tp6-main.sh" "$@"
