#!/usr/bin/env bash
# Settings in precedence order: exported environment, scripts/local.sh, .env, the defaults below.
# local.sh is trusted Bash. .env is read as literal KEY=value lines, never executed.
# Sourced by scripts/prepare.sh, start.sh and stop.sh; it runs no Docker, GPU, ssh or network command.
[[ "${_GLM_CONFIG_LOADED:-0}" == 1 ]] && return 0
_GLM_CONFIG_LOADED=1
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -f "$REPO_ROOT/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _cfg_key; do _cfg_env[$_cfg_key]=${!_cfg_key}; done < <(compgen -e)
  # shellcheck source=/dev/null
  source "$REPO_ROOT/scripts/local.sh"
  for _cfg_key in "${!_cfg_env[@]}"; do export "$_cfg_key=${_cfg_env[$_cfg_key]}"; done
  unset _cfg_env
fi
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r _cfg_line || [[ -n "$_cfg_line" ]]; do
    [[ "$_cfg_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _cfg_key=${BASH_REMATCH[2]}; _cfg_value=${BASH_REMATCH[3]}
    if [[ "$_cfg_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_cfg_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _cfg_value=${BASH_REMATCH[1]}
    else
      _cfg_value=${_cfg_value%%#*}; _cfg_value=${_cfg_value%"${_cfg_value##*[![:space:]]}"}
    fi
    [[ -n "${!_cfg_key+set}" ]] || export "$_cfg_key=$_cfg_value"
  done < "$REPO_ROOT/.env"
fi
unset _cfg_key _cfg_line _cfg_value

# --- the checkpoint: MODEL_FORMAT --------------------------------------------------------------------------------
#   k4     GLM-5.3-EXL3-K4 (the default): wrldsuksgo2mars's K4 quant, wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC at e28931c9,
#          its `original` view only (EXL3 MCG K4 experts, BF16 dense core packed to 8 bits at load; the k8/k6-compact
#          views' EXL3 cores are not loadable by this engine). TP4, NFS weights, in the HF cache's snapshot directory;
#          when it is missing the start downloads just that view's files (MODEL_INCLUDE, ~418 GB) at the pinned revision. ./start.sh adds its serving defaults (window, reserve, KV split, the neon stack).
#   derisked  Mia's AI Lab's GLM-5.3-DERISKED-Cyber-EXL3-4bpw (the DERISKED lane): TP4, NFS weights, served from DERISKED_DIR
#          ($HOME/models/GLM-5.3-Derisked-EXL3-4bpw) when it holds the checkpoint, otherwise downloaded there from
#          Hugging Face (Mia-AiLab/GLM-5.3-DERISKED-Cyber-EXL3-4bpw) at DERISKED_REVISION. ./start.sh adds its serving
#          defaults (window, reserve, KV split, the neon stack).
#   exlr8  a gr8 (kindlingai EXLR8) quant you prepared yourself; the recipe bundles none (README "gr8 (EXLR8) quants").
#          scripts/prepare-exlr8.sh converts one rank's pieces on each of spark1-4 (EXLR8_RELEASE names the release).
#          TP4, int8 KV, DFlash2, a 196,608-token default window. Each rank reads its own copy: WORKER_WEIGHTS=check.
#   nvfp4  nvidia/GLM-5.3-NVFP4 on six Sparks (everything below at its TP6 default).
# MODEL_FORMAT unset means k4, unless HF_CACHE_DIR or MODEL_DIR is already set: a checkpoint named that way is
# served exactly as before (no format default applies), which is how every K4 start and the night stages' restores run.
# An exported variable always outranks the format's default.
_fmt="${MODEL_FORMAT:-}"
if [[ -z "$_fmt" && -z "${HF_CACHE_DIR:-}" && -z "${MODEL_DIR:-}" ]]; then _fmt=k4; fi
case "$_fmt" in
  '') ;;
  nvfp4) ;;
  derisked)
    MODEL_FORMAT=derisked
    : "${DERISKED_DIR:=$HOME/models/GLM-5.3-Derisked-EXL3-4bpw}"
    : "${MODEL_DIR:=$DERISKED_DIR}"; : "${HF_CACHE_DIR:=$MODEL_DIR}"
    : "${SERVED_NAME:=GLM-5.3-Derisked-EXL3-4bpw}"
    : "${WORKER_WEIGHTS:=nfs}"; : "${NFS_MODEL_VOLUME:=glm53-derisked-exl3-4bpw}"
    : "${MODEL_ID:=Mia-AiLab/GLM-5.3-DERISKED-Cyber-EXL3-4bpw}"
    # A copy already in place (the lab's, converted on its own server) is served as it is; none: download it, pinned.
    if [[ -z "${MODEL_SOURCE:-}" ]]; then
      if [[ -f "$MODEL_DIR/config.json" ]]; then MODEL_SOURCE=local; else MODEL_SOURCE=download; fi
    fi
    if [[ "$MODEL_SOURCE" == download ]]; then : "${MODEL_REVISION:=${DERISKED_REVISION:-48aee9fbf16b7a18697298d90ee168310b501adc}}"; fi ;;
  k4)
    MODEL_FORMAT=k4
    : "${HF_CACHE_DIR:=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-MC}"
    : "${MODEL_REVISION:=e28931c918fa89403b1cd247f89c7dc25eeb67ac}"
    : "${MODEL_DIR:=$HF_CACHE_DIR/snapshots/$MODEL_REVISION}"
    : "${MODEL_ID:=wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC}"
    : "${NFS_MODEL_VOLUME:=glm53-k4mc-model}"
    : "${SERVED_NAME:=GLM-5.3-EXL3-K4}"
    # Only the `original` view: its 99 shards (model-*, original-*) and the files every view shares. Not the k8/k6 cores.
    : "${MODEL_INCLUDE:=config.json generation_config.json quantize_config.json model.safetensors.index.json tokenizer.json tokenizer_config.json chat_template.jinja LICENSE README.md model-*.safetensors original-*.safetensors}"
    if [[ -z "${MODEL_SOURCE:-}" ]]; then
      if [[ -f "$MODEL_DIR/config.json" ]]; then MODEL_SOURCE=local; else MODEL_SOURCE=download; fi
    fi ;;
  exlr8)
    MODEL_FORMAT=exlr8
    # shellcheck source=scripts/exlr8/release.sh
    source "$REPO_ROOT/scripts/exlr8/release.sh" || return 1
    # EXLR8_WORKER_HOME: the workers' home when it is not this one (the lab's spark2-4: /home/zurih); the converted
    # directory and the drafter copy the workers read sit under it. From the environment or scripts/exlr8/hosts.env.
    if [[ -z "${EXLR8_WORKER_HOME:-}" && -f "$REPO_ROOT/scripts/exlr8/hosts.env" ]]; then
      while IFS= read -r _cfg_line || [[ -n "$_cfg_line" ]]; do
        [[ "$_cfg_line" =~ ^[[:space:]]*(export[[:space:]]+)?EXLR8_WORKER_HOME=(.*)$ ]] || continue
        EXLR8_WORKER_HOME=${BASH_REMATCH[2]%%#*}; EXLR8_WORKER_HOME=${EXLR8_WORKER_HOME//[\"\' ]/}
      done < "$REPO_ROOT/scripts/exlr8/hosts.env"
    fi
    # Interim rounds dense trellis weights to BF16. Native keeps the stored trellis for decode.
    # Use the same dense mode and release for preparation and serving.
    EXLR8_DENSE="${EXLR8_DENSE:-native}"
    case "$EXLR8_DENSE" in
      interim) EXLR8_LEAF=$EXLR8_INTERIM_LEAF ;;
      native) EXLR8_LEAF=$EXLR8_NATIVE_LEAF ;;
      *) echo "[E_CONFIG] EXLR8_DENSE must be interim or native (got '$EXLR8_DENSE')." >&2; return 1 ;;
    esac
    : "${EXLR8_DIR:=$HOME/$EXLR8_LEAF}"
    if [[ -z "${HF_CACHE_DIR:-}" ]]; then : "${MODEL_DIR:=$EXLR8_DIR}"; HF_CACHE_DIR=$MODEL_DIR; fi
    : "${SERVED_NAME:=$EXLR8_SERVED_NAME}"
    : "${WORKER_WEIGHTS:=check}"
    if [[ -z "${WORKER_MODEL_DIR:-}" && -n "${EXLR8_WORKER_HOME:-}" ]]; then WORKER_MODEL_DIR="$EXLR8_WORKER_HOME/$EXLR8_LEAF"; fi
    [[ -n "${WORKER_DFLASH2_DIR:-}" ]] || _exlr8_wdf=1 ;;
  *) echo "[E_CONFIG] MODEL_FORMAT must be derisked, exlr8, k4 or nvfp4 (got '$_fmt')." >&2; return 1 ;;
esac
unset _fmt _cfg_line

# --- pinned engine build (hashed into the image tag with docker/ and patches/engine/*.patch) ----------------------
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-f350cd3ea253a042e205746928271f9d8f71a82d}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
# sha256 of zig-aarch64-linux-0.17.0.tar.xz (ziglang.org/download/index.json).
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
SM="${SM:-121}"                                  # DGX Spark GB10: sm_121 (the engine's own default -Dsm target)
ZIG_CPU="${ZIG_CPU:-native}"                     # the image is built on a Spark and runs on Sparks; baseline to move it
NCCL_WHEEL="${NCCL_WHEEL:-}"                     # non-empty: install this nvidia-nccl-cu13 version over the base image's
IMAGE="${IMAGE:-}"                               # empty: tensorfold-glm53nvfp4-spark:<PATCHES_HASH>
# WITH_VIDEO=1 builds libtfvideo.so (FFmpeg 9.0.2, static) into the image, so VISION=1 serves video_url parts too
# (issue 8). It adds a few minutes to a build and has its own image tag. 0 (default): video_url parts return HTTP 500.
WITH_VIDEO="${WITH_VIDEO:-0}"
IMAGE_REPO="${IMAGE_REPO:-tensorfold-glm53nvfp4-spark}"
DOCKER_BUILD_FLAGS="${DOCKER_BUILD_FLAGS:-}"     # extra docker build flags, word-split (e.g. --progress=plain)
TF_BIN="${TF_BIN:-/usr/local/bin/tensorfold-native}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-spark}"  # never a DeepSeek recipe's container on these Sparks
CONTAINER_USER="${CONTAINER_USER-$(id -u):$(id -g)}"   # empty or root: run the ranks as root

# --- the two Sparks ---------------------------------------------------------------------------------------------------
# This machine serves rank 0 and the HTTP API; WORKER (an ssh target, key-based, no password) runs rank 1.
# Both ranks run the same image and the same flags; before the weights load they compare a fingerprint of their
# settings, the checkpoint and the binary, and name the field that differs.
TP="${TP:-2}"                                    # this recipe wires two Sparks; the engine also does TP=3 (unqualified here)
WORKER="${WORKER:-}"                             # e.g. zurih@10.0.34.4 or spark4; set it in scripts/local.sh
# One multiplexed connection carries the many small calls to the worker (docker inspect, the memory samples): a new
# ssh per call would cost a handshake each time and fill its auth log.
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-glm53nvfp4-%C -o ControlPersist=300}"
# The ranks' link. Empty: start.sh asks the route to WORKER for this Spark's own ConnectX-7 netdev, the worker's address
# on it, the RoCE devices of that port (a Spark's one QSFP port reaches the GB10 over two PCIe links, so it shows up as
# two netdevs and two RDMA devices; both are used) and the RoCE v2 GID index that holds each side's IPv4 address.
FABRIC_PEER="${FABRIC_PEER:-}"                   # the worker's address on the link (auto)
FABRIC_SELF="${FABRIC_SELF:-}"                   # this Spark's address on the link (auto); rank 0's NCCL_COMM_ID address
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"     # this Spark's link netdev (auto); the worker's is found on the worker
TF_ROCE_HCA="${TF_ROCE_HCA:-}"                   # RDMA devices for the link, comma separated, at most two (auto)
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"       # the RoCE v2 IPv4 GID index (auto; it differs between hosts' tables)
NCCL_CHANNELS="${NCCL_CHANNELS:-4}"              # NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
PORT="${PORT:-8020}"
HOST="${HOST:-0.0.0.0}"
NCCL_PORT="${NCCL_PORT:-29583}"                  # NCCL_COMM_ID=FABRIC_SELF:NCCL_PORT; the ranks' watch line uses NCCL_PORT+1
SERVED_NAME="${SERVED_NAME:-GLM-5.3-NVFP4}"
# ROCE=on: the engine carries the decode and verify all-gathers up to ROCE_MAX_KB as one-shot RDMA writes over the
# Sparks' link (zig/src/cuda/roce.zig; 10.4 us a call at 4 KiB a rank against NCCL's 12.3, 17.4 against 51.5 at 64 KiB,
# measured upstream on two GB10s over one ConnectX-7 cable). Larger collectives stay on NCCL, and the size alone
# decides, so both ranks always take the same path. Same bytes in the same places as NCCL: the same bits.
# off (TF_ROCE=0 on either rank): NCCL carries everything.
ROCE="${ROCE:-on}"
ROCE_MAX_KB="${ROCE_MAX_KB:-4096}"               # TF_ROCE_MAX_KB, the largest shard the RDMA region holds
# ROCE_MESH (TP6): a decode window's two all-gathers a layer as one-shot RoCE writes to every peer (engine patch 0140,
# fabric/tp's multirank gather), NCCL for prompt chunks. Six GB10s on the switch, 16 / 32 / 64 KiB a rank: 23 / 28 /
# 36 us against NCCL's 52 / 80 / 86; decode prose 25.9 -> 28.1, code 36.3 -> 39.5 tok/s (2026-11-13). Every rank
# brings it up and passes a probe gather or all stay on NCCL. MESH_MAX_KB: the largest shard (TF_TP_MAILBOX_MAX_KB).
ROCE_MESH="${ROCE_MESH:-1}"
NSYS_DIR="${NSYS_DIR:-/opt/nvidia/nsight-systems/2025.3.2}" # RANK0_NSYS=1 mounts it into rank 0 (diagnosis only)
MESH_MAX_KB="${MESH_MAX_KB:-512}"
# MESH_FAST: the mesh kernel's fast path (TF_TP_ROCE_FAST: its own pollers, staggered posts). Bit-exact; decode
# +1.1% (two boots each, every sparkDash prompt faster: 33.3 / 43.2 against 32.9 / 42.7). Pinning the proxy thread
# (TF_TP_ROCE_CPU) measured nothing. 0: the classic path.
MESH_FAST="${MESH_FAST:-1}"
# SPLIT_QKVA (engine patch 0149, TF_GLM_DSA_SPLIT_QKVA): a decode window's [q_a | kv_a] split across the ranks plus one
# more mesh gather a layer. TP4 (K4, int8 KV, DFlash2; 2026-11-16, clocks capped at 1800 MHz): on by default, the
# replicated projection (request policy bit 5, same boot) decodes 0.95% slower over 36 code/tool/think pairs, replies
# bit-identical. TP6: off (measured neutral-to-negative, sparkDash 30.5 / 41.7 -> 30.1 / 41.2, before the faster gathers).
if [[ "${GLM_TP6:-0}" == 1 && ( "${TP:-6}" == 4 || "${TP:-6}" == 6 ) ]]; then SPLIT_QKVA="${SPLIT_QKVA:-1}"; else SPLIT_QKVA="${SPLIT_QKVA:-0}"; fi
# DSA_INDEX_FAST (engine patch 0192): the decode indexer's query RoPE once a row (TF_GLM_DSA_ROPE_HOIST=1) and its
# top-2048 over several CTAs (TF_GLM_DSA_TOPK_MULTI=1, patch 0067's scan/merge on per-token keys). Bit-identical picks
# (P12-fixed drafts on and off, every mask). TP4 (2026-11-16, 1800 MHz cap), decode against off, paired: +2.2% prose /
# +2.6% code at ~60K tokens of context, +13.0% / +14.5% at ~190K. On by default at TP4 and TP6 (0: off).
if [[ "${GLM_TP6:-0}" == 1 && ( "${TP:-6}" == 4 || "${TP:-6}" == 6 ) ]]; then DSA_INDEX_FAST="${DSA_INDEX_FAST:-1}"; else DSA_INDEX_FAST="${DSA_INDEX_FAST:-0}"; fi
# KV_SPLIT (engine patches 0187-0189, opt-in): each token's latent rows (both int8 scale planes) on one rank,
# TF_GLM_KV_SPLIT=1 with the history prefetch off (TF_GLM_KV_SPLIT_PREFETCH=0). Exact (P12-fixed reply hashes equal
# the unsplit boot's). TP4 (K4, int8 KV, DFlash2; 2026-11-16, 1800 MHz cap): a 540,672-token window against 212,992
# unsplit, or 245,760 with a 1.4 GiB prompt cache (cached turn TTFT 28.7 -> 4.3 s); decode and prefill ~10% slower.
# Needs KV_PAGES=off, no MTP drafter and LEARN_GIB=0 (prepare.sh refuses the rest). 0: off.
KV_SPLIT="${KV_SPLIT:-0}"
# SLOT_RESUME (engine patch 0195, TF_GLM_SLOT_RESUME): a later prompt on the same slot resumes where the slot's own
# prompt rows still match it (the longest common token prefix, cut back to a 4096-row chunk start, at least DFlash2's
# 2048-row window short of the end so the drafter re-taps it), instead of prefilling from 0. Nothing is copied and no
# memory is set aside; the reused rows are reported as cached_tokens. TP4 (K4, int8 KV, DFlash2, image night-sr2;
# 2026-11-17, 1800 MHz cap): a 12-turn agent loop resumed on 11 of 11 later turns, TTFT 3-7 s against 33-53 s for
# full re-prefills; the same 49,556-token prompt cold 54.7 s, resumed 6.3 s; cold / resumed / cold again gave
# identical generated token ids, and an 8-layer thin model on spark6 identical replies cold and resumed. 1: on
# (default); 0: every prompt prefills from row 0.
SLOT_RESUME="${SLOT_RESUME:-1}"
# PF_EXPERTS_ONCE, PF_QMMF2, PF_DOWN_FUSE (engine patches 0206, 0207, 0208; the prompt side's three switches, bit-exact):
#   PF_EXPERTS_ONCE (TF_GLM_PF_EXPERTS_ONCE): a prompt chunk's routed gate/up rotate each member row once in shared
#     memory (tf_exl3_prompt_gu_once) instead of copying and rotating a row per route.
#   PF_QMMF2 (TF_GLM_PF_QMMF2): block-FP8 prompt GEMMs on qmmf2's 128x64 tile; it takes effect only in the engine's
#     fp8g weight mode, so on EXLR8 up1 it changes nothing.
#   PF_DOWN_FUSE (TF_GLM_PF_DOWN_FUSE): the routed down kernel also sums each row's route slots (the fused combine).
# Measured with all three on, TP4 (1800 MHz cap): K4 prefill +18%, bit-exact; EXLR8 up1 1,029 / 1,039 / 1,005 tok/s
# against 929 / 933 / 905, exact on 18 of 18, decode 1.004x. TP6 (NVFP4) not measured. On by default at TP4 (exlr8, k4
# and a named K4 checkpoint alike, as DSA_INDEX_FAST); 0: off, the engine's own default. TP6: off. An exported
# TF_GLM_PF_* of the same name sets the default too (so it still wins), and PF_* itself wins over both.
if [[ "${GLM_TP6:-0}" == 1 && ( "${TP:-6}" == 4 || "${TP:-6}" == 6 ) ]]; then _pf=1; else _pf=0; fi
PF_EXPERTS_ONCE="${PF_EXPERTS_ONCE:-${TF_GLM_PF_EXPERTS_ONCE:-$_pf}}"
PF_QMMF2="${PF_QMMF2:-${TF_GLM_PF_QMMF2:-$_pf}}"
PF_DOWN_FUSE="${PF_DOWN_FUSE:-${TF_GLM_PF_DOWN_FUSE:-$_pf}}"
unset _pf

# --- six Sparks (./start-tp6.sh, ./start-tp6-main.sh, ./stop.sh; README "6 Sparks") ----------------------
# TP=6 over six Sparks: this Spark (rank 0, the API) and WORKER, WORKER2, WORKER3 (ranks 1-5). On a switch
# (./start-tp6.sh) every Spark's ports share the switch's subnet and the workers may be in any order. Cabled as a ring
# without a switch (./start-tp6-main.sh) they go in the ring's order, each cabled to the next and the last back to
# this one, one subnet per cable; the two Sparks across the ring from each other reach each other only through a
# neighbour (an address routed over the cables, or the LAN). start.sh never reads these; the TP=6 starts set TP=6.
WORKER2="${WORKER2:-}"                           # rank 2's ssh target (e.g. mia@spark3)
WORKER3="${WORKER3:-}"                           # rank 3's ssh target (e.g. mia@spark4)
WORKER4="${WORKER4:-}"                           # rank 4's ssh target (e.g. mia@spark5)
WORKER5="${WORKER5:-}"                           # rank 5's ssh target (e.g. mia@spark6)
# NCCL_COMM_ID's address at TP=6 and the ranks' lines to rank 0: an address of this Spark every worker reaches (auto:
# what this Spark's host name resolves to, else its first address). The Spark across the ring reaches it through a
# neighbour. SOCKET_IFNAME: NCCL's bootstrap netdev on every rank (auto: each Spark's default-route netdev, the LAN).
MASTER_ADDR="${MASTER_ADDR:-}"
SOCKET_IFNAME="${SOCKET_IFNAME:-}"
# TF_GLM_PREFILL_REDUCE at TP=6. The engine's default, split, sends every rank's prompt rows straight to every other
# rank. gather: the exchanges as all-gathers (the qualified mode on the switch).
# neighbours, no point-to-point sends (the qualified mode here); ring: split, its exchanges passed along the ring.
PREFILL_REDUCE="${PREFILL_REDUCE:-gather}"
# The checkpoint and drafter paths as the NFS server on this Spark exports them, when they differ from MODEL_DIR and
# DFLASH2_DIR (an NFS server in a container exports its own view: /huggingface/... for /home/you/.cache/huggingface/...).
# NFS_MODEL_PATH: the path the NFS server exports. Empty (default): MODEL_FS_ROOT -- the HF cache's
# models--nvidia--GLM-5.3-NVFP4 root, so the snapshot's symlinks into blobs/ resolve inside the mount.
# Set it when the server exports a different view of the same directory.
NFS_MODEL_PATH="${NFS_MODEL_PATH:-}"
NFS_DFLASH2_PATH="${NFS_DFLASH2_PATH:-}"
# NFS_SERVER1 / NFS_SERVER2 / NFS_SERVER3: the address each worker mounts from (auto: this Spark's address on its route
# to that worker; NFS_SERVER, when set, for all three).
NFS_SERVER1="${NFS_SERVER1:-}"; NFS_SERVER2="${NFS_SERVER2:-}"; NFS_SERVER3="${NFS_SERVER3:-}"
NFS_SERVER4="${NFS_SERVER4:-}"; NFS_SERVER5="${NFS_SERVER5:-}"

# --- weights ---------------------------------------------------------------------------------------------------------
# Both Sparks read the whole checkpoint at load: each rank keeps its half, but every rank reads every shard, so the
# worker needs the same bytes as this one. The ranks compare a fingerprint of the checkpoint before they load, so a
# different revision on the worker is refused rather than mixed.
#   WORKER_WEIGHTS=rsync  send this Spark's copy over the link when the worker's is missing or a different revision
#                         (resumable, ~164 GiB, about 5 minutes at 600 MB/s). The worker then loads from local disk.
#   WORKER_WEIGHTS=nfs    the worker mounts this Spark's copy over NFS, as a read-only Docker volume per directory,
#                         and no bytes are duplicated. GLM holds all its weights in memory once loaded, so NFS costs
#                         load time only, not serving speed. Needs an NFS server on this Spark exporting MODEL_DIR
#                         (and DFLASH2_DIR) to the worker's address on the link; ./start.sh checks the export and says
#                         what to add when it is missing.
#   WORKER_WEIGHTS=check  refuse a start unless the worker already has them (nothing is copied or mounted).
WORKER_WEIGHTS="${WORKER_WEIGHTS:-nfs}"
NFS_SERVER="${NFS_SERVER:-}"                     # this Spark's address on the link as the worker reaches it (auto: FABRIC_SELF)
NFS_MODEL_VOLUME="${NFS_MODEL_VOLUME:-glm53nvfp4-model}"      # the worker's Docker NFS volume for the checkpoint
NFS_DFLASH2_VOLUME="${NFS_DFLASH2_VOLUME:-glm53-dflash2full-8f02}"
NFS_OPTS="${NFS_OPTS:-nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600}"
DATA_DIR="${DATA_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/aevonix-glm53-nvfp4}"
MODEL_SOURCE="${MODEL_SOURCE:-local}"
# ABLIT=1 serves the Ablit weights (the same checkpoint, abliterated; README "Ablit weights") instead of the published
# one; 0 (default) the published one. The Ablit repository is gated: fetching it needs HF_TOKEN, from an account that
# accepted the terms on the model's page. It downloads beside the published checkpoint, which stays in the cache.
# ABLIT/FP8 select alternate checkpoints in the Flash recipe; this recipe serves only the NVFP4
# checkpoint, so both stay 0 (their checks in prepare.sh just validate the value).
ABLIT="${ABLIT:-0}"
FP8="${FP8:-0}"
# nvidia/GLM-5.3-NVFP4: ModelOpt MIXED_PRECISION -- NVFP4 (gs16) routed experts on layers 3-77, per-tensor FP8 dense
# MLPs on layers 0-2, BF16 elsewhere, an integrated MTP layer (78). ~420 GiB in 50 safetensor shards. It is fetched with
# `hf download nvidia/GLM-5.3-NVFP4` on this Spark (spark1) and lands in the default HF cache; every worker reads the same
# files over NFS (WORKER_WEIGHTS=nfs). The whole models--nvidia--GLM-5.3-NVFP4 cache directory must be exported --
# snapshot files are symlinks into blobs/ -- and mounted at the same absolute path on every Spark.
_id="nvidia/GLM-5.3-NVFP4"
MODEL_ID="${MODEL_ID:-$_id}"
# MODEL_DIR: a snapshot directory (the dir holding config.json). Default: the HF cache's snapshot at MODEL_REVISION, or
# the only/newest snapshot present. MODEL_REVISION="" resolves to refs/main's hash on disk when the cache is used.
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--GLM-5.3-NVFP4}"
if [[ -z "${MODEL_REVISION:-}" && -f "$HF_CACHE_DIR/refs/main" ]]; then _rev=$(cat "$HF_CACHE_DIR/refs/main"); else _rev="${MODEL_REVISION:-}"; fi
MODEL_REVISION="${MODEL_REVISION:-$_rev}"
if [[ -z "${MODEL_DIR:-}" ]]; then
  if [[ -n "$MODEL_REVISION" && -d "$HF_CACHE_DIR/snapshots/$MODEL_REVISION" ]]; then _dir="$HF_CACHE_DIR/snapshots/$MODEL_REVISION"
  elif [[ -d "$HF_CACHE_DIR/snapshots" ]]; then _dir=$(ls -dt "$HF_CACHE_DIR"/snapshots/*/ 2>/dev/null | head -1); _dir="${_dir%/}"
  else _dir="$HF_CACHE_DIR"; fi
else _dir="$MODEL_DIR"; fi
MODEL_DIR="${MODEL_DIR:-$_dir}"
unset _id _dir _rev
# MODEL_FS_ROOT: the ancestor directory every container mounts at its real absolute path. For an HF-cache
# MODEL_DIR (<cache>/snapshots/<rev>) the model files are relative symlinks into ../.. /blobs, so the whole
# models--nvidia--GLM-5.3-NVFP4 directory must be visible inside the mount -- exporting only the snapshot breaks
# every symlink on the clients. For a plain directory of real files it is MODEL_DIR itself.
case "$MODEL_DIR" in */snapshots/*) MODEL_FS_ROOT=$(dirname -- "$(dirname -- "$MODEL_DIR")") ;; *) MODEL_FS_ROOT=$MODEL_DIR ;; esac
unset _id _dir _rev
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-$MODEL_DIR}"
# HF_TOKEN from scripts/local.sh or .env reaches the hf CLI only when exported. It is never passed to the ranks.
[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
# DRAFTER=dflash2 (the default since 2026-11-16): Inco AI's DFlash2 for full GLM-5.3 (incoai/GLM-5.3-DFlash2 @ 8f02251d,
# Apache-2.0: D 6144, 6 layers, taps after layers 5/19/33/47/61/75; run on glm_moe_dsa by engine patches 0184-0185). The
# GLM-5.3-Flash DFlash2 (incoai/GLM-5.3-Flash-DFlash2) only fits a GLM-5.3-Flash target; the engine refuses it here.
DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-DFlash2}"
DFLASH2_REVISION="${DFLASH2_REVISION:-8f02251dc71a9da2053dfc15bfa293fbee5579b0}"
# the HF cache's snapshot at the pinned revision when it is there, else DATA_DIR/dflash2 (downloaded when missing)
_dflash2_hf="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${DFLASH2_ID//\//--}/snapshots/$DFLASH2_REVISION"
if [[ -f "$_dflash2_hf/config.json" ]]; then DFLASH2_DIR="${DFLASH2_DIR:-$_dflash2_hf}"; else DFLASH2_DIR="${DFLASH2_DIR:-$DATA_DIR/dflash2}"; fi
WORKER_DFLASH2_DIR="${WORKER_DFLASH2_DIR:-$DFLASH2_DIR}"          # the worker's copy (same path by default)
if [[ "${_exlr8_wdf:-}" == 1 && -n "${EXLR8_WORKER_HOME:-}" && "$DFLASH2_DIR" == "$HOME"/* ]]; then  # MODEL_FORMAT=exlr8: the workers' own copy
  WORKER_DFLASH2_DIR="$EXLR8_WORKER_HOME${DFLASH2_DIR#"$HOME"}"
fi
unset _exlr8_wdf
# WORKER_DFLASH2_LOCAL=1 (WORKER_WEIGHTS=nfs): the workers read the drafter from their own copy in WORKER_DFLASH2_DIR
# instead of an NFS volume of DFLASH2_DIR (a drafter folder this Spark does not export); 0 (default): over NFS.
WORKER_DFLASH2_LOCAL="${WORKER_DFLASH2_LOCAL:-0}"
# A mixed-width EXL3 quant names avg_bits, not bits; the stock GLM check wants quantization_config.bits == 4.
# auto: mount a copy of config.json with bits 4 added over /model/config.json (only when bits is missing); 0: never.
CONFIG_OVERLAY="${CONFIG_OVERLAY:-0}"
CACHE_DIR="${CACHE_DIR:-$DATA_DIR/zig-home}"     # the ranks' HOME: the engine's per-file checkpoint digest cache
WORKER_CACHE_DIR="${WORKER_CACHE_DIR:-$CACHE_DIR}"
# Where the worker keeps this recipe's own small files (the config.json overlay, when one is needed).
WORKER_STATE_DIR="${WORKER_STATE_DIR:-$WORKER_CACHE_DIR/state}"
WORKER_OVERLAY="${WORKER_OVERLAY:-$WORKER_STATE_DIR/model-config.json}"
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.state}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
LOG_KEEP="${LOG_KEEP:-10}"                       # newest compressed session logs kept

# --- serving --------------------------------------------------------------------------------------------------------
DRAFTER="${DRAFTER:-dflash2}"                # dflash2 (default) | dspark | mtp (the checkpoint's MTP head) | none (--no-drafts)
# DRAFTER=dflash2 is the default since 2026-11-16 (owner's rule: faster than DSpark on code, tool, think and sampled prose
# with thinking off and on): K4 TP4, image 3cde334f49ea, KV fp8 int8, GPU clocks capped at 1800 MHz: code 39.0 vs 36.2,
# tool 39.3 vs 34.1, think 39.6 vs 39.2, prose 26.3 vs 26.2 (thinking off) and 27.5 vs 27.3 (on) tok/s (README, CHANGELOG).
# DRAFTER=dspark (opt-in since 2026-11-16; the default 2026-11-15 to 2026-11-16): RedHatAI's DSpark speculator for full GLM-5.3 (engine patches 0165-0173; the glm-5.3
# license; downloaded, never redistributed), read through the drafter mount DFlash2 uses: its id, revision, folder and
# NFS volume take DFlash2's place, and the engine tells the two apart by the folder's config.json. Exact (0 of 54
# replies differ); TP4 against no drafts: code 1.92x, think 2.12x, tool 1.62x, copy 3.0-3.2x (CHANGELOG 0.3.1).
# TF_GLM_DSPARK_HEAD=full reads the full head rows, TF_GLM_DSPARK_TAGS=0 turns off the tool-call tag rules.
DSPARK_ID="${DSPARK_ID:-RedHatAI/GLM-5.3-speculator.dspark}"
DSPARK_REVISION="${DSPARK_REVISION:-b374b95663447ea0e935151be4f3d6666e36e6d7}"
# the HF cache's snapshot at the pinned revision when it is there, else DATA_DIR/dspark (downloaded when missing)
_dspark_hf="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--RedHatAI--GLM-5.3-speculator.dspark/snapshots/$DSPARK_REVISION"
if [[ -f "$_dspark_hf/config.json" ]]; then DSPARK_DIR="${DSPARK_DIR:-$_dspark_hf}"; else DSPARK_DIR="${DSPARK_DIR:-$DATA_DIR/dspark}"; fi
DSPARK_CONFIDENCE="${DSPARK_CONFIDENCE:-0.35}"   # TF_GLM_DRAFT_CONFIDENCE for DSpark: its chain stops under this chance product
DRAFTER_MODEL=dflash2
if [[ "$DRAFTER" == dspark ]]; then
  DRAFTER=dflash2
  DRAFTER_MODEL=dspark
  DFLASH2_ID=$DSPARK_ID
  DFLASH2_REVISION=$DSPARK_REVISION
  DFLASH2_DIR=$DSPARK_DIR
  WORKER_DFLASH2_DIR="${WORKER_DSPARK_DIR:-$DSPARK_DIR}"
  NFS_DFLASH2_VOLUME="${NFS_DSPARK_VOLUME:-glm53-dspark-b374}"
  [[ "${RANK_EXTRA_ENV:-}" == *TF_GLM_DRAFT_CONFIDENCE=* ]] || RANK_EXTRA_ENV="${RANK_EXTRA_ENV:+$RANK_EXTRA_ENV }TF_GLM_DRAFT_CONFIDENCE=$DSPARK_CONFIDENCE"
fi
DRAFTER_BITS="${DRAFTER_BITS:-4}"                # --drafter-bits: 4 (default) or 0 (bf16 drafter)
MTP_DRAFTS="${MTP_DRAFTS:-}"                     # --mtp-drafts N (empty: the engine's policy)
# MTP_CONFIDENCE: --mtp-confidence; a chain stops at the draft whose chained chance falls under it (engine patch 0146
# skips the head's later steps there). 0.5 at TP6 (knapcio's gr8 six-Spark sweep: 0 / 0.5 / 0.75 alike, +5% over none).
if [[ "${GLM_TP6:-0}" == 1 ]]; then MTP_CONFIDENCE="${MTP_CONFIDENCE-0.5}"; else MTP_CONFIDENCE="${MTP_CONFIDENCE:-}"; fi
# MTP_CAP (TF_GLM_MTP_CAP, engine patch 0153): the MTP chain's most drafts, 1..6; empty = the engine's 4. Under
# MTP_CONFIDENCE a deeper chain only pays where drafts stay likely: sparkDash code's most repetitive prompt went 48.8 ->
# 51.7 -> 55.2 tok/s at caps 4 / 5 / 6, the other code prompts and all prose lost 0-1%. 6 suits repetitive or
# structured output; 4 stays the default.
MTP_CAP="${MTP_CAP:-}"
# GLM-5.3 DSA keeps each stream's index-key ring in its fixed arena at full CONTEXT_MAX (patch 0132: dsa_select
# scores raw ik over every visible position, so the 8,192-row ring evicted earlier chunks). At 262,144 × 46 MLA
# caches that is ~3.0 GiB a stream — PARALLEL=4 OOM-killed rank 0 (2026-11-13). 1 stream until ik moves to pages.
# SHARED_KV (owner 2026-11-25): several streams on one shared KV pool (patch 0210, TF_GLM_SEQ_GROW: every stream's latent
# rows and index keys on granules of one pool the size of one whole window, so a lone stream still reaches the full
# window) with their decode rows in one verify window (patch 0211, TF_GLM_DSA_SHARED). Each stream past the first takes
# ~48 MiB of fixed parts. On by default at six Sparks for K4 and Derisked (11.3 GiB left at 655,360 tokens), with
# PARALLEL=4; 0: one stream. Needs KV_SPLIT=0, KV_PAGES=off, no prompt or host cache (prepare.sh checks).
if [[ "${GLM_TP6:-0}" == 1 && "${TP:-6}" == 6 && ( "${MODEL_FORMAT:-}" == k4 || "${MODEL_FORMAT:-}" == derisked ) ]]; then
  SHARED_KV="${SHARED_KV:-1}"; else SHARED_KV="${SHARED_KV:-0}"; fi
if [[ "$SHARED_KV" == 1 ]]; then PARALLEL="${PARALLEL:-4}"
elif [[ "${GLM_TP6:-0}" == 1 && "${TP:-6}" == 4 && "${MODEL_FORMAT:-}" == exlr8 ]]; then PARALLEL="${PARALLEL:-2}"; else PARALLEL="${PARALLEL:-1}"; fi # --parallel: streams at once (a number is required to fit; auto: up to 8)
# TENSORFOLD_PREFILL_SHARE: the share of wall time a long prompt's chunks (4,096 rows with patches 0094/0095, about
# 2.5 s each on this pair) get while other streams decode. 1: one decode round per chunk, so a stream decoding beside a
# long prefill gets one round every chunk. p: each chunk owes the decoders (1 - p) / p of its own time in decode rounds
# first (0.68: about 1.2 s, ~22 rounds; 0.5: as long as the chunk itself), and the long prompt's first token comes later
# by that much. Replies are the same at any value: it only moves when the next chunk runs. See README.
PREFILL_SHARE="${PREFILL_SHARE:-0.68}"
# TENSORFOLD_PREFILL_BUDGET: prompt rows filled in one visit while a stream waits to decode (empty: no cap)
PREFILL_BUDGET="${PREFILL_BUDGET:-}"
# auto: the largest window that fits. The first start tries CONTEXT_MAX (the model's 1,048,576), and when a rank
# refuses it with its memory figures, starts once more with the window they allow; the window that served is kept in
# STATE_DIR/context-fit.tsv for the next start with the same memory settings.
# TP4 (TP6_RANKS=4, K4, int8 KV, DFlash2; 2026-11-16, 1800 MHz cap): 212,992, the largest window that served at the
# 8.5 GiB reserve (passes the needle at 191,692; KV_SPLIT=1: 540,672). TP6: auto, as before.
# EXLR8 defaults to 196,608 for both releases. On spark1 a 229,376-token v0.7 boot tripped the
# TF_GLM_START_FLOOR_MIB floor (2304 MiB), with about 1.9 GiB available at the last arena.
if [[ "${GLM_TP6:-0}" == 1 && "${TP:-6}" == 4 && "${KV_SPLIT:-0}" == 1 ]]; then CONTEXT="${CONTEXT:-540672}"
elif [[ "${GLM_TP6:-0}" == 1 && "${TP:-6}" == 4 && "${MODEL_FORMAT:-}" == exlr8 ]]; then CONTEXT="${CONTEXT:-196608}"
elif [[ "${GLM_TP6:-0}" == 1 && "${TP:-6}" == 4 ]]; then CONTEXT="${CONTEXT:-212992}"
# Six Sparks, K4 or Derisked (owner 2026-11-25: as much KV as fits over the reserve): the shared KV pool is one window,
# so the window is the total KV. Derisked at 655,360 left 11.1 GiB after a 3 GiB reserve at ~49.8 KB a token a rank;
# 868,352 left 1.3-1.4 GiB at 3 GiB, and an idle server was stopped by memguard when a 2.9 GiB process ran on spark1.
# With the 4 GiB reserve, 847,872 keeps the same ~1.4 GiB left above it.
elif [[ "${GLM_TP6:-0}" == 1 && "${TP:-6}" == 6 && ( "${MODEL_FORMAT:-}" == k4 || "${MODEL_FORMAT:-}" == derisked ) ]]; then CONTEXT="${CONTEXT:-847872}"
else CONTEXT="${CONTEXT:-auto}"; fi
# At six Sparks the DSA index-key ring (patch 0132) sits outside the engine's memory budget: a 1,048,576 or 524,288
# window is not refused, rank 0 is OOM-killed (exit 137) after the ~5 min load, and every new patch hash probed it
# again (2026-11-13: 1M -> 512K -> 256K). 262,144 is the largest window that has served; raise it once ik is on pages.
if [[ "${GLM_TP6:-0}" == 1 ]]; then CONTEXT_MAX="${CONTEXT_MAX:-262144}"; else CONTEXT_MAX="${CONTEXT_MAX:-1048576}"; fi
CONTEXT_RETRY="${CONTEXT_RETRY:-1}"              # 1: one more start with a fitting window after a memory refusal
CONTEXT_FIT_MARGIN="${CONTEXT_FIT_MARGIN:-0.93}" # share of the computed window the retry asks for
# --kv-cache. bf16: the DSA latent rows 576-wide BF16 (1,152 B a token and layer). fp8 (engine patches 0162-0163, the
# format of the owner's 3x recipe patch 0038): E4M3 codes and one power-of-two fp32 scale a row (580 B), so TP4 serves
# 262,144 tokens without drafts (bf16: 131,072) and TP6 262,144 with 15 GiB to spare; prefill +7-10%. Opt-in: on the
# 8K agent panel it adds +0.03-0.06 nats of KL a window against BF16 KV (gate: +0.005), E4M3's 3-bit mantissa.
KV="${KV:-fp8}"                               # owner 2026-11-16: 8-bit KV is the default (KV=bf16 opts out)
# KV8_CODES (with KV=fp8): int8 = int8 codes, two power-of-two scales a row (engine patch 0164, TF_GLM_KV8_INT8=1;
# row RMS error 0.96% vs E4M3 2.67%); e4m3 = the E4M3 codes of patch 0162.
KV8_CODES="${KV8_CODES:-int8}"
# TF_KV_PAGES. off here: the DSA's latent cache keeps a BF16 window per stream — its kernels refuse paged rows
# outright (a pt table of any kind), so on and auto (which resolves to pages for bf16) fail at attention as well.
KV_PAGES="${KV_PAGES:-off}"
# q4 is rejected on this checkpoint, not merely unused: NLL 0.8005 against q8's 0.7535 (about 40x the
# ABSORB_MMA-vs-off difference), it fails the same spelling question, and at 1,747 tok/s with ABSORB_MMA it is
# slower than q8 + ABSORB_MMA + W8A8. Dense weights are only ~1.8 GiB of 81.8 here (first_k_dense_replace 3,
# 288 routed experts), so narrowing them buys little and costs measurably.
DENSE="${DENSE:-q8}"                             # TF_GLM_DENSE_BITS: bf16 (16) | q8 | q6 | q4 (symmetric int codes, groups of 64)
# HEAD (TF_GLM_HEAD_BITS, the LM head, same choices): q8 at TP6. Against bf16 (the engine is bit-deterministic, so
# the KL tool's floor is 0): mean KL 0.0004 / 0.0008 / 0.0003 and 0.12 / 0.43 / 0.18% top-1 flips on code, English
# prose and zh / ja / ru text, NLL unchanged (+-0.0001); sparkDash prose decode +4-5% (three q8 boots all above two
# interleaved bf16 boots), code +1%. DENSE=q6 by contrast was KL 0.020, 3.5% flips, NLL +0.006. HEAD=bf16 opts out.
if [[ "${GLM_TP6:-0}" == 1 ]]; then HEAD="${HEAD:-q8}"; else HEAD="${HEAD:-bf16}"; fi
# TENSORFOLD_MEMORY_RESERVE_GIB. GB10 is a unified-memory device: the engine budgets from /proc/meminfo's MemAvailable
# (plus the pages parked on the kernel's per-CPU free lists, which MemAvailable leaves out), not from a card's free
# memory, and everything the Spark runs besides the rank comes out of the same pool. The engine's own default here is
# max(4 GiB, MemTotal/10), about 11.9 GiB on a 128 GB Spark; its floor is 2. 6.5 everywhere since 2026-11-11 (it was 12;
# the pair ran 9.5). Raise it when a long session pushes a Spark toward its floor: rank 0 also carries the HTTP server
# and the vision tower.
# TP6: 8.5. Ranks 0 and 1 hold 64 more rows of every routed expert than ranks 2-5 (2,048 rows in 64-row units do
# not divide by six), ~12 GiB, and a 122K-token prefill took spark1 from 2.7 to 2.4 GiB available at 6.5: the memory
# guard stopped the server (2026-11-13). Back to 6.5 once the experts split evenly.
# Owner 2026-11-25: 4 GiB for every checkpoint and width (experts split evenly at TP6 since patch 0240). At 3 GiB an
# idle six-Spark server left spark1 2.7 GiB over the 2 GiB memory-guard floor, and a 2.9 GiB process there stopped it.
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-4}"
# CACHE_FIRST=1 (engine patch 0112, TF_PROMPT_CACHE_FIRST): PROMPT_CACHE_GIB is set aside before the room is counted
# in whole CONTEXT windows, so the PARALLEL streams share the pages left (one window at least) instead of the cache
# getting what whole windows leave. 0: windows first. On by default since 2026-11-11: at a 6.5 reserve windows-first
# would let the pair's page pool take a third 1M window and leave no prompt cache (the cliff RESULTS.md 2026-11-04
# records at 19.5 GiB of room); FP8 at six Sparks had none at 9.5.
CACHE_FIRST="${CACHE_FIRST:-0}" # 0 while PROMPT_CACHE_GIB=0 (0132's ik rings): no cache to fund first
# PROMPT_CACHE_GIB, device pages for kept prompt states: 5 on the pair (predicted to keep its two 1M windows, 4,096
# pages, beside it), 8 with EXL3 at six Sparks (measured: all eight windows, 16,392 pages, and an 8.0 GiB cache), 4
# with FP8=1 (measured: 6,147 pages and a 3.9 GiB cache).
if [[ "${GLM_TP6:-0}" == 1 && "$FP8" == 1 ]]; then PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-4}"
elif [[ "${GLM_TP6:-0}" == 1 ]]; then PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-0}" # 0132's capacity ik rings leave ~1 GiB of room; the cache's pages are the only GiB-scale slack
else PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-5}"; fi
# PROMPT_CACHE_MIN_GIB: the smallest prompt cache a start may serve with, in GiB. The engine funds the cache from the
# room each rank has at that start (it varies by a GiB or more between starts), and PROMPT_CACHE_GIB only caps it, even
# with CACHE_FIRST=1. With a value here, start.sh reads each rank's startup line and stops both ranks (E_CACHE) before
# LIVE when one funds less. Empty (default): no check.
PROMPT_CACHE_MIN_GIB="${PROMPT_CACHE_MIN_GIB:-}"
# TF_GLM_HOST_CACHE_GIB, the kept prompt states' pinned host tier. On a discrete card that memory is free; on GB10 it
# is the very memory the KV pool and the weights draw on, and a state in it costs a copy to restore, so the tier is off
# here and PROMPT_CACHE_GIB (device pages) is the one to raise. Unset or auto would pin a quarter of MemAvailable,
# up to 16 GiB over the two ranks.
HOST_CACHE_GIB="${HOST_CACHE_GIB:-0}"
# LEARN_GIB (engine patch 0113, --learn-conversations): kept prompt states also go to disk under CACHE_DIR/learn,
# every rank's part in rank 0's store, so a conversation resumes after its device copy is evicted or the pair
# restarts. The GiB of disk the states may take; 0 (default): off, as before. LEARN_MIN_FREE_GIB: no state is
# written while the disk has less free.
LEARN_GIB="${LEARN_GIB:-0}"
LEARN_MIN_FREE_GIB="${LEARN_MIN_FREE_GIB:-15}"
# KV_SHARD=1 (engine patch 0114, TF_GLM_KV_SHARD): each stream's FP8 latent rows live on one Spark (its owner) instead
# of both, so the same memory holds about twice the tokens; the other Spark's queries go to the owner for attention
# (1-4% decode). Needs KV=fp8, KV pages, DFlash2 or no drafter, PARALLEL above 1. The pool then takes most of the
# room: pair it with CACHE_FIRST=1 PROMPT_CACHE_GIB so long conversations stay kept. 0 (default): off, as before.
KV_SHARD="${KV_SHARD:-0}"
# L2PF=1 (engine patch 0117, TF_GLM_L2PF, ported from upstream): each decode window reads the next layer's mixer
# weights, this layer's router and KDA's o_proj into L2 on a side stream while the gathers and chains run (not the
# EXL3 routed experts). Upstream turns it on by default on GB10 at TP>=2; on this pair (2026-11-11) it gave +2% at one
# stream and nothing measurable at four or eight, token hashes identical. Both ranks must agree (the fingerprint
# carries it). 0 (default): off, as before.
# TP6: on. Its decode verify windows wait on two NCCL all-gathers a layer; reading the next weights into L2 meanwhile
# measured prose 25.4 -> 25.9, code 34.6 -> 36.3 tok/s (2026-11-13, with patch 0138).
if [[ "${GLM_TP6:-0}" == 1 ]]; then L2PF="${L2PF:-1}"; else L2PF="${L2PF:-0}"; fi
VISION="${VISION:-0}"                            # --vision: images (rank 0's tower)
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"     # --vision-max-images: images per request (engine default 4, at most 64)
# --chat-template: auto = with VISION=1, templates/glm-5.3-flash-vision.jinja (the checkpoint's own template, whose media
# branch writes GLM's image and video markers instead of a "cannot see" note); none = the checkpoint's; or a file path
CHAT_TEMPLATE="${CHAT_TEMPLATE:-auto}"
# The template's "Reasoning Effort" line. The checkpoint's template (and the vision copy) sets it to Max unless a
# request's effort is low or high, so every prompt carries `<|system|>Reasoning Effort: Max`, thinking off included,
# and a thinking-off reply then reasons in its answer (a 64-token "digits only" question runs out of tokens mid-sum).
# TensorFold v0.6.0's recipe wrote the line with thinking on only. thinking (default): the same here -- the served
# template is written to STATE_DIR/chat_template.jinja from the chosen one, its line taken from reasoning_effort,
# which the engine passes with thinking on only; always: the template as it is. CHAT_TEMPLATE=auto only; a template
# of your own decides for itself.
EFFORT_LINE="${EFFORT_LINE:-thinking}"
# EXTRA_EOS (engine patch 0120, TF_EXTRA_EOS): tokens that end a reply besides the checkpoint's EOS ids
# (<|endoftext|>, <|user|>, <|observation|>), comma-separated. GLM-5.3 can close a reply with <|assistant|>, which
# its generation_config leaves out: the token then reaches the reply as text and the model writes a second turn of
# its own until MAX_TOKENS. none: the checkpoint's list only.
EXTRA_EOS="${EXTRA_EOS:-<|assistant|>}"
# THINK_OFF_SPLIT (engine patch 0121, TF_THINK_OFF_SPLIT): with thinking off, GLM-5.3 sometimes drafts in its reasoning
# voice anyway and writes </think> before its answer. A chat reply that writes that token within its first
# THINK_OFF_SPLIT tokens gets the text before it as reasoning_content and the rest as content, as a thinking reply
# does. While a reply streams, its content waits until the token, the end or THINK_OFF_SPLIT tokens (whichever comes
# first), so a long thinking-off reply's first content arrives up to that many tokens later. 0 (default): off, since
# that hold delays every streamed thinking-off reply; 1024 is the hold its author measured.
THINK_OFF_SPLIT="${THINK_OFF_SPLIT:-0}"
THINKING="${THINKING:-0}"                        # --thinking / --no-thinking default; requests override
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"    # --reasoning-effort low|medium|high|xhigh (empty: omit); xhigh renders the checkpoint's trained "Reasoning Effort: Max" line
# The reply limit of a request that sends none. The server caps it to the room the prompt leaves in the window, so the
# model window (1048576) means "to the end of the context": a long thinking reply is not cut at a fixed count (32768
# before 2026-11-20 ended effort-max replies mid-thinking, content empty). A request's own max_tokens still wins.
MAX_TOKENS="${MAX_TOKENS:-1048576}"              # --max-tokens default reply limit
CLEAR_THINKING="${CLEAR_THINKING:-1}"            # TF_GLM_CLEAR_THINKING: 1 drops earlier turns' reasoning, 0 keeps it
# TF_GLM_CE_EXCHANGE moves a prompt chunk's split exchanges by copy engines into the peer's registered rows, which
# needs CUDA IPC between the ranks: that exists between two GPUs in one box, not between two Sparks. The engine falls
# back to NCCL by itself when it cannot map the peer, so 1 would only log that it did; off here.
CE_EXCHANGE="${CE_EXCHANGE:-0}"
# Prompt numerics (engine patches 0041 on): faster prefill, not bit-identical to stock. Both carried +6.6% and
# +10% claims from sm_120; measured on a Spark pair on 2026-11-02 (3 cold trials a point, fresh prompts so no
# prefix-cache hit, NLL over four held passages plus ten greedy answers):
#   ABSORB_MMA=1 (absorb_q/expand_v on tensor cores, an fp32 summation reorder): prefill 1,503 -> 1,623 tok/s at
#     23.5K and 1,530 -> 1,652 at 95K, +8.0%. All ten greedy answers byte-identical to it off, and NLL 0.7523
#     against 0.7535 -- equal or a shade better. Nothing is traded for it, so it is on by default here.
#   W8A8=1 (int8 activations in the prompt dense GEMMs): a further +10.5% (1,789 / 1,828 tok/s), which is the
#     v1.0.7 Spark build's prefill. It rounds activations: per-passage NLL moves +-0.03 (its total matching
#     baseline is cancellation across passages, not equivalence) and it answered "spell necessary backwards"
#     as yrassen for baseline's correct yrassecen. Off; turn it on only behind your own eval.
ABSORB_MMA="${ABSORB_MMA:-1}"                    # TF_GLM_ABSORB_MMA: measured free on GB10 (see above)
# Re-measured at TP6 on 2026-11-14 with the deterministic engine's KL gate (top-20, vs W8A8=0): prefill only
# 1,219 / 1,200 -> 1,241 / 1,238 tok/s (+2-3%) for KL 0.018 (code), 0.022 (prose), 0.014 (zh/ja/ru) and 2.8-4.7%
# top-1 flips -- the same loss as DENSE=q6, which was rejected. Stays off.
W8A8="${W8A8:-0}"                                # TF_GLM_W8A8: stashed (see above); a real rounding of activations
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"         # more tensorfold-native serve flags, word-split, given to every rank

# --- memory guard ----------------------------------------------------------------------------------------------------
# A Spark's weights, KV, prompt cache and everything else it runs share one 128 GB pool, and a box driven into the
# kernel's reclaim path can stop answering ssh long before the OOM killer picks a victim. MEMGUARD=1 samples
# MemAvailable on both Sparks and, when one stays under MEMGUARD_FLOOR_GIB for MEMGUARD_HOLD_S, stops the server (the
# same way ./stop.sh does) and says which Spark and when in the run's log. 0 turns it off.
MEMGUARD="${MEMGUARD:-1}"
MEMGUARD_FLOOR_GIB="${MEMGUARD_FLOOR_GIB:-2}"  # owner 2026-11-16: 2 GiB (was 2.5)
MEMGUARD_HOLD_S="${MEMGUARD_HOLD_S:-4}"
MEMGUARD_INTERVAL_S="${MEMGUARD_INTERVAL_S:-2}"

# --- checks and timeouts -----------------------------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"                          # 1: print the plan; no Docker, ssh, GPU, download or file change
WAIT_TIMEOUT="${WAIT_TIMEOUT:-2400}"             # seconds for the API to come up (digest, load, graphs)
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"
STOP_TIMEOUT="${STOP_TIMEOUT:-45}"               # docker stop grace: rank 0 drains for TF_DRAIN_S (20 s) first
# Memory a Spark must have available before a start, counted as the engine counts it (MemAvailable plus the parked
# per-CPU pages; scripts/memavail.sh). The model is about 82 GiB a rank, and the reserve and the KV pool come on top.
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-100}"
# Compact the GB10's shared host-memory pool on every rank before admission. Needs passwordless sudo.
COMPACT_MEMORY="${COMPACT_MEMORY:-1}"
IMAGE_FREE_GIB="${IMAGE_FREE_GIB:-40}"
MODEL_FREE_GIB="${MODEL_FREE_GIB:-180}"
DFLASH2_FREE_GIB="${DFLASH2_FREE_GIB:-10}"
