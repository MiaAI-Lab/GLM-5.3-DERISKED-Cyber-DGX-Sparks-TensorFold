# Six Sparks at TP=6 on a switch (./start-tp6.sh to start, ./stop-tp6.sh to stop): reaching the five workers,
# finding the links and the RDMA devices each rank runs on, and the image and the weights on every Spark.
# Sourced after scripts/prepare.sh (its helpers, config.sh, nodes.sh).
#
# The link discovery (node_inventory, pair_links) is adapted from the lab's 2x DGX Sparks recipe's scripts/nodes.sh
# (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, branch zig-engine). The lessons are that
# recipe's: with one GPU a node NCCL takes one RoCE device a channel, and a DGX Spark's single QSFP port reaches the
# GB10 over two PCIe links, so one port shows up as two netdevs and two RDMA devices and both have to be given to
# NCCL for the port's full width. On the switch every pair shares one RoCE subnet, so NCCL picks its own
# algorithms and devices (no graph files, no NCCL_ALGO, no subnet-aware routing).
[[ "${_GLM_TP6_LOADED:-0}" == 1 ]] && return 0
_GLM_TP6_LOADED=1

# --- the five workers -------------------------------------------------------------------------------------------------
worker_ids() { seq 1 $(( ${TP:-6} - 1 )); }  # TP6_RANKS=4 (start-tp6-main.sh) runs ranks 0-3 on this Spark and WORKER..WORKER3
wvar() { if (( $2 == 1 )); then printf '%s' "$1"; else printf '%s%s' "$1" "$2"; fi; }   # WORKER, WORKER2, WORKER3, WORKER4, WORKER5
wtarget() { local v; v=$(wvar WORKER "$1"); printf '%s' "${!v}"; }
whost() { local t; t=$(wtarget "$1"); printf '%s' "${t#*@}"; }
# Every remote call is bounded: a dead or memory-starved worker (17:54 2026-11-15: spark3 out of memory, a
# `docker logs` over ssh hung 38 min) fails the call instead of hanging the start or stop. One-shot calls get
# REMOTE_TIMEOUT seconds (default: the docker stop grace + 75); ssh drops a connection whose peer stops answering for
# 60 s (ServerAliveInterval 15 x CountMax 4). timeout exits 124 and ssh 255 when a worker cannot be reached.
REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-$(( ${STOP_TIMEOUT:-45} + 75 ))}"
SHIP_TIMEOUT="${SHIP_TIMEOUT:-3600}"   # docker save | ssh docker load of the image to one worker
SSH_ALIVE='-o ServerAliveInterval=15 -o ServerAliveCountMax=4'
# wssh_n <rank> <words...>: a command on worker <rank>'s host, the words quoted for its shell ($HOME left to it).
wssh_n() {
  local i=$1 a remote=''
  shift
  for a in "$@"; do
    case "$a" in '$HOME'|'$HOME'/*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
  done
  # shellcheck disable=SC2086
  timeout -k 10 "$REMOTE_TIMEOUT" ssh $SSH_OPTS $SSH_ALIVE -- "$(wtarget "$i")" "$remote"
}
# shellcheck disable=SC2086
wsh_n() { timeout -k 10 "$REMOTE_TIMEOUT" ssh $SSH_OPTS $SSH_ALIVE -- "$(wtarget "$1")" bash -s <<< "$2"; }
wfn_n() {
  local i=$1
  shift
  local -a names=()
  while (( $# )) && [[ "$1" != -- ]]; do names+=("$1"); shift; done
  shift || true
  wsh_n "$i" "$(declare -f "${names[@]}"); $*"
}

# Filled by tp6_probe_workers: each worker's home (its cache and state live under it: the workers may be other logins
# than this Spark's), uid:gid and host name.
declare -a WORKER_HOME=() WORKER_UG=() WORKER_NAME=() WORKER_CACHE=() WORKER_STATE=() WORKER_VERSION=()
tp6_nvidia_ready() {
  local runtimes
  runtimes=$(wssh_n "$1" docker info --format '{{json .Runtimes}}' 2>/dev/null || true)
  [[ "$runtimes" != *'"nvidia"'* ]] || return 0
  wssh_n "$1" test -x /usr/bin/nvidia-container-runtime-hook >/dev/null 2>&1 && return 0
  wssh_n "$1" test -x /usr/bin/nvidia-ctk >/dev/null 2>&1 && return 0
  return 1
}
tp6_probe_workers() {
  local i out m ug home name
  for i in $(worker_ids); do
    out=$(wssh_n "$i" sh -c 'printf "%s %s:%s %s %s\n" "$(uname -m)" "$(id -u)" "$(id -g)" "$HOME" "$(hostname)"' 2>/dev/null) \
      || fail E_WORKER "Cannot reach $(wtarget "$i") (rank $i) over ssh with $SSH_OPTS. Set up key-based login (ssh-copy-id) and check $(wvar WORKER "$i") in scripts/local.sh."
    read -r m ug home name <<< "$out"
    [[ "$m" == aarch64 ]] || fail E_WORKER "$(whost "$i") reports $m, not aarch64: every rank runs the same aarch64 image."
    [[ "$home" == /* ]] || fail E_WORKER "Could not read the login's home on $(whost "$i")."
    wssh_n "$i" docker info >/dev/null 2>&1 || fail E_WORKER "$(whost "$i") cannot talk to Docker (is the login there in the docker group?)."
    tp6_nvidia_ready "$i" || fail E_WORKER "$(whost "$i") cannot give a container the GPU: no nvidia runtime registered and no NVIDIA Container Toolkit hook. Install NVIDIA Container Toolkit there."
    WORKER_HOME[i]=$home; WORKER_UG[i]=$ug; WORKER_NAME[i]=$name
    WORKER_VERSION[i]=$(wssh_n "$i" sh -c 'uname -r; nvidia-smi --query-gpu=driver_version,clocks.max.sm --format=csv,noheader,nounits' 2>/dev/null) \
      || fail E_FABRIC "Could not read the kernel and GPU driver on $(whost "$i")."
    WORKER_CACHE[i]="$home/.cache/aevonix-glm53/zig-home"; WORKER_STATE[i]="${WORKER_CACHE[i]}/state"
    log "[I_WORKER] rank $i: $name ($(whost "$i")) reachable over ssh, aarch64, Docker with the NVIDIA runtime."
  done
}
# The user the ranks run as on worker i: the worker's own login when CONTAINER_USER is this Spark's login (the default),
# else CONTAINER_USER as given (root, or a fixed UID:GID).
tp6_container_user() {
  if [[ "$CONTAINER_USER" == "$(id -u):$(id -g)" ]]; then printf '%s' "${WORKER_UG[$1]}"; else printf '%s' "$CONTAINER_USER"; fi
}
tp6_worker_gpu() {
  local i names
  for i in $(worker_ids); do
    names=$(wssh_n "$i" nvidia-smi -L 2>/dev/null) || fail E_GPU "Cannot list GPUs on $(whost "$i"). Check its driver."
    grep -q 'GB10' <<< "$names" || warn W_GPU "$(whost "$i")'s GPU is not a GB10: $(head -1 <<< "$names")."
    log "[I_GPU] rank $i: $(sed -n '1s/ (UUID.*//p' <<< "$names") on $(whost "$i")."
  done
}
tp6_worker_avail_gib() { wfn_n "$1" mem_counting mem_avail_gb -- 'mem_avail_gb' 2>/dev/null | awk '{print $1}'; }
tp6_worker_avail_bytes() { wfn_n "$1" mem_counting mem_avail_bytes -- 'mem_avail_bytes' 2>/dev/null; }
tp6_worker_memory() {
  local i theirs
  for i in $(worker_ids); do
    theirs=$(tp6_worker_avail_gib "$i")
    if [[ -z "$theirs" ]]; then warn W_RAM "Cannot read $(whost "$i")'s available memory; the start goes on."; continue; fi
    log "[I_RAM] rank $i: $theirs GiB available on $(whost "$i")."
    at_least "$theirs" "$MIN_AVAIL_GIB" \
      || fail E_MEMORY "$(whost "$i") has $theirs GiB available; MIN_AVAIL_GIB=$MIN_AVAIL_GIB is required. Stop whatever else holds its memory and try again."
  done
}
tp6_worker_disk() {
  local i root free
  for i in $(worker_ids); do
    if wssh_n "$i" docker image inspect "$IMAGE" >/dev/null 2>&1; then continue; fi
    root=$(wssh_n "$i" docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
    free=$(wssh_n "$i" df -Pk -- "$root" 2>/dev/null | awk 'END {printf "%d\n", $4 / 1048576}')
    [[ "$free" =~ ^[0-9]+$ ]] || { warn W_DISK "Cannot read $(whost "$i")'s free disk space; the start goes on."; continue; }
    (( free >= IMAGE_FREE_GIB )) || fail E_DISK "$(whost "$i") needs about $IMAGE_FREE_GIB GiB free for the image; it has $free GiB on $root."
    log "[I_DISK] $(whost "$i") has $free GiB free for the image."
  done
}

# --- the image on every Spark --------------------------------------------------------------------------------------------
tp6_image_ident() { local s; s=$(wssh_n "$1" docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
# Every worker runs the very image this Spark holds; a missing or different copy is streamed there (docker save | ssh
# docker load).
tp6_ship_image() {
  local i here there
  if [[ "$DRY_RUN" == 1 ]]; then
    log 'Compare the image on every Spark by content; where a worker'"'"'s copy differs or is missing:'
    for i in $(worker_ids); do plan docker save "$IMAGE" '|' ssh "$(wtarget "$i")" docker load; done
    return 0
  fi
  here=$(image_ident "$IMAGE")
  [[ "$here" != missing ]] || fail E_IMAGE "The image $IMAGE is not on this Spark. Run scripts/prepare.sh --image first."
  for i in $(worker_ids); do
    there=$(tp6_image_ident "$i" "$IMAGE")
    if [[ "$here" == "$there" ]]; then log "[I_IMAGE] $(whost "$i") already runs this image ($IMAGE, ${here:0:12})."; continue; fi
    log "Sending $IMAGE to $(whost "$i") (its copy is ${there:0:12}); this takes a few minutes."
    # shellcheck disable=SC2086
    docker save "$IMAGE" | timeout -k 30 "$SHIP_TIMEOUT" ssh $SSH_OPTS $SSH_ALIVE -o Compression=no -- "$(wtarget "$i")" docker load >/dev/null \
      || fail E_IMAGE "Could not load $IMAGE on $(whost "$i"). Check its disk space and Docker access."
    there=$(tp6_image_ident "$i" "$IMAGE")
    [[ "$here" == "$there" ]] || fail E_IMAGE "$(whost "$i") still reports another image for $IMAGE (${there:0:12} against ${here:0:12})."
    log "[I_IMAGE] $(whost "$i") now runs this image ($IMAGE, ${here:0:12})."
  done
}

# --- the fabric --------------------------------------------------------------------------------------------------------------
# node_inventory [master]: this node's default-route netdev ("default <netdev>"), whether it routes to <master>
# ("master ok|none"), and one line per RoCE netdev that is up with an IPv4 address: "<netdev> <rdma device>
# <address/prefix> <RoCE v2 GID index of that address, or ->". Read-only (ip and sysfs). The netdevs come in uverbs
# order (rocep1s0f0, rocep1s0f1, then the PCIe twins), which is the order NCCL 2.30 numbers the devices in on DGX
# Sparks, so a graph file's "net dev N" names the device meant. Self-contained: it is sent to each worker as text.
node_inventory() {
  local n dev hca cidr want idx f u
  echo "default $(ip -o -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
  [[ -z "${1:-}" ]] || { ip -o -4 route get "$1" >/dev/null 2>&1 && echo "master ok" || echo "master none"; }
  for n in $(for n in /sys/class/net/*; do
               [[ -d $n/device/infiniband ]] || continue
               u=$(ls "$n/device/infiniband_verbs" 2>/dev/null | head -1); u=${u#uverbs}
               [[ "$u" =~ ^[0-9]+$ ]] || u=9999; echo "$u $n"
             done | sort -n | cut -d' ' -f2); do
    dev=${n##*/}
    [[ -d $n/device/infiniband && "$(cat "$n/operstate" 2>/dev/null)" == up ]] || continue
    hca=$(ls "$n/device/infiniband" | head -1)
    for cidr in $(ip -o -4 addr show dev "$dev" | awk '{print $4}'); do
      # shellcheck disable=SC2046
      want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"${cidr%/*}"))
      idx=-
      for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
        [[ "$(cat "$f" 2>/dev/null)" == "$want" ]] || continue
        [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)" == *v2* ]] && { idx=${f##*/}; break; }
      done
      echo "$dev $hca $cidr $idx"
    done
  done
}

# pair_links <ranks...> (stdin: node_inventory lines, each prefixed with its rank): two nodes are linked where they
# have addresses in one subnet (a direct cable). Prints "rank <r> <default netdev> <hcas> <gid or ->" (a node's RoCE
# devices toward all its peers, in inventory order; the GID index when it is one for all of them), "peer <i> <worker
# i's lowest address on its link to rank 0>", "missing <a> <b>" for two ranks without a common subnet, "ring
# <ranks...>" when the links form one ring through all the ranks (every rank linked to two), in its order from rank 0
# toward its lower-numbered neighbour, "toward <r> <s> <hcas>" for every linked pair (r's devices on the subnets it
# shares with s), and "prefix <len>": the shortest prefix length (longer than the links' own) at which, on every link,
# each device's address and exactly one of the peer's share a subnet and no two of a node's devices do (a port and its
# PCIe twin addressed .N and .10N in one /24: 26), so NCCL's subnet-aware routing can tell the twins apart.
pair_links() {
  python3 -c '
import ipaddress, sys
nodes, default = {}, {}
for line in sys.stdin:
    f = line.split()
    if len(f) == 3 and f[1] == "default":
        default[int(f[0])] = f[2]
    elif len(f) == 5:
        nodes.setdefault(int(f[0]), []).append((f[1], f[2], ipaddress.ip_interface(f[3]), f[4]))
ranks = [int(x) for x in sys.argv[1:]]
linked = lambda e, others: any(e[2].network == o[2].network for o in others)
links = {r: set() for r in ranks}
for r in ranks:
    used = set()
    for s in ranks:
        if s == r:
            continue
        mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
        if not mine:
            if r < s:
                print("missing", r, s)
            continue
        links[r].add(s)
        used.update(id(e) for e in mine)
        print("toward", r, s, ",".join(dict.fromkeys(e[1] for e in mine)))
        if r == 0:
            print("peer", s, min(o[2].ip for o in nodes[s] if linked(o, mine)))
    hcas, gids = [], set()
    for e in nodes.get(r, []):
        if id(e) in used and e[1] not in hcas:
            hcas.append(e[1]); gids.add(e[3])
    print("rank", r, default.get(r) or "-", ",".join(hcas) or "-", gids.pop() if len(gids) == 1 else "-")
def splits(n):
    for r in ranks:
        for s in links[r]:
            mine = [e for e in nodes.get(r, []) if linked(e, nodes.get(s, []))]
            theirs = [o for o in nodes.get(s, []) if linked(o, mine)]
            nets = [ipaddress.ip_interface(f"{e[2].ip}/{n}").network for e in mine]
            if len(set(nets)) != len(nets) or any(sum(o[2].ip in net for o in theirs) != 1 for net in nets):
                return False
    return True
own = max((e[2].network.prefixlen for r in ranks for e in nodes.get(r, [])), default=32)
pre = next((n for n in range(own, 31) if splits(n)), None)
if pre is not None:
    print("prefix", pre)
if len(ranks) > 3 and all(len(links[r]) == 2 for r in ranks):
    order = [ranks[0], min(links[ranks[0]])]
    while len(order) <= len(ranks):
        nxt = (links[order[-1]] - {order[-2]}).pop()
        if nxt == order[0]:
            break
        order.append(nxt)
    if len(order) == len(ranks):
        print("ring", *order)
' "$@"
}

# Filled by detect_links: each rank's bootstrap netdev, RoCE devices and GID index; the head's address toward each
# worker (this Spark's address on its route there) and the worker's own address on the shared subnet.
declare -a NODE_DEV=() NODE_HCAS=() NODE_GID=() NODE_GRAPH=() LINK_HEAD_ADDR=() LINK_WORKER_ADDR=()
# Ring only (FABRIC_EXPECT=ring, ./start-switchless.sh): each rank's devices toward each neighbour, and the subnet prefix
# that keeps a port and its PCIe twin apart (pair_links, from the 1104 recipe's scripts/tp4.sh).
declare -A TOWARD=()
RING_PREFIX=''
detect_links() {
  local i r k kind a b c d pairs missing='' ring='' order='' peer want
  local -a inv=()
  inv[0]=$(node_inventory)
  for i in $(worker_ids); do
    inv[i]=$(wfn_n "$i" node_inventory -- "node_inventory $(printf '%q' "$MASTER_ADDR")") \
      || fail E_FABRIC "Could not read rank $i's ($(whost "$i")) ConnectX-7 ports over ssh."
    [[ "${inv[i]}" != *'master none'* ]] \
      || fail E_FABRIC "$(whost "$i") has no route to MASTER_ADDR $MASTER_ADDR. Set MASTER_ADDR to an address of this Spark every worker reaches."
  done
  pairs=$(for r in $(seq 0 $(( TP - 1 ))); do sed "s/^/$r /" <<<"${inv[r]}"; done | pair_links $(seq 0 $(( TP - 1 )))) \
    || fail E_FABRIC 'Could not pair the Sparks'"'"' links (python3 and ip are needed on every Spark).'
  while read -r kind a b c d; do
    case "$kind" in
      rank) NODE_DEV[a]=${SOCKET_IFNAME:-$b}; NODE_HCAS[a]=$c; NODE_GID[a]=$d; [[ "$d" != - ]] || NODE_GID[a]='' ;;
      peer) LINK_WORKER_ADDR[a]=$b ;;
      missing) missing+=" $a-$b" ;;
      ring) ring="$a $b $c $d" ;;
      toward) TOWARD["$a $b"]=$c ;;
      prefix) RING_PREFIX=$a ;;
    esac
  done <<<"$pairs"
  ring=$(xargs <<<"$ring")
  if [[ "${FABRIC_EXPECT:-mesh}" == ring ]]; then
    # a ring without a switch (./start-switchless.sh): every Spark cabled to the next, one subnet per cable, the
    # workers listed in the ring's order from this Spark. pair_links reports a ring for four ranks only (its "ring"
    # line carries four), so the ring start runs TP=4.
    (( TP == 4 )) || fail E_CONFIG "FABRIC_EXPECT=ring runs four Sparks (TP=4), not $TP."
    [[ -n "$missing" ]] || fail E_FABRIC 'Every pair of these four Sparks shares a RoCE subnet (a switch), not a ring: use ./start.sh.'
    [[ -n "$ring" ]] || fail E_FABRIC "The four Sparks' links do not form one ring (no common subnet between ranks$missing): cable each Spark to two others, one subnet per cable, and give both ends addresses on it."
    want=$(seq -s ' ' 0 $(( TP - 1 )))
    if [[ "$ring" != "$want" ]]; then
      for k in $(cut -d' ' -f2- <<<"$ring"); do order+=", $(whost "$k")"; done
      fail E_FABRIC "The Sparks are cabled as a ring, but WORKER, WORKER2 and WORKER3 are not in its order (from this Spark$order, and back): set them in that order in scripts/local.sh."
    fi
  fi
  # a switch: every pair of Sparks on one RoCE subnet, so NCCL reaches every rank directly, in any order
  if [[ "${FABRIC_EXPECT:-mesh}" != ring && -n "$missing" && -n "$ring" ]]; then
    fail E_FABRIC "These Sparks are cabled as a ring without a switch (no RoCE subnet between ranks$missing): use ./start-switchless.sh."
  fi
  [[ "${FABRIC_EXPECT:-mesh}" == ring || -z "$missing" ]] || fail E_FABRIC "Some of these six Sparks share no RoCE subnet (ranks$missing): on a switch every Spark's ConnectX-7 ports need an address on the switch's subnet (ip -4 addr)."
  for i in $(worker_ids); do
    peer=${LINK_WORKER_ADDR[i]:-}
    if [[ -z "$peer" ]]; then
      peer=$(whost "$i")
      if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        c=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
        [[ -n "$c" ]] || fail E_FABRIC "Cannot resolve $peer ($(wvar WORKER "$i")) to an IPv4 address."
        peer=$c
      fi
      LINK_WORKER_ADDR[i]=$peer
    fi
    read -r LINK_HEAD_ADDR[i] _ <<<"$(link_info "$peer")" || true
    [[ -n "${LINK_HEAD_ADDR[i]:-}" ]] || fail E_FABRIC "No route from this Spark to $peer (rank $i, $(whost "$i"))."
  done
  for r in $(seq 0 $(( TP - 1 ))); do
    [[ -n "${NODE_DEV[r]:-}" && "${NODE_DEV[r]}" != - ]] \
      || fail E_FABRIC "rank $r has no default route: set SOCKET_IFNAME to the netdev of the network all six Sparks share (NCCL's bootstrap socket)."
    [[ -n "${NODE_HCAS[r]:-}" && "${NODE_HCAS[r]}" != - ]] \
      || fail E_FABRIC "rank $r shares no RoCE subnet with any other rank: check the ConnectX-7 cabling and addresses (ip -4 addr)."
  done
  if [[ "${FABRIC_EXPECT:-mesh}" == ring ]]; then
    log "[I_FABRIC] A ring: this Spark -> $(whost 1) -> $(whost 2) -> $(whost 3) -> back here; RoCE v2 GID index ${NODE_GID[0]:-per device}/${NODE_GID[1]:-per device}/${NODE_GID[2]:-per device}/${NODE_GID[3]:-per device}; subnet prefix ${RING_PREFIX:-of the links}."
    for r in $(seq 0 $(( TP - 1 ))); do
      a=$(( (r + 1) % TP )); b=$(( (r + TP - 1) % TP ))
      log "[I_FABRIC] rank $r: NCCL on ${NODE_HCAS[r]} (bootstrap ${NODE_DEV[r]}); toward the next rank ${TOWARD[$r $a]:-?}, toward the previous ${TOWARD[$r $b]:-?}."
    done
    log "[I_FABRIC] Rank 2 ($(whost 2)) is across the ring from this Spark: its line to rank 0 and any NFS go to ${LINK_HEAD_ADDR[2]} (through a neighbour or the LAN)."
    return 0
  fi
  log "[I_FABRIC] A switch: every pair of the six Sparks shares a RoCE subnet; RoCE v2 GID index ${NODE_GID[0]:-per device}/${NODE_GID[1]:-per device}/${NODE_GID[2]:-per device}/${NODE_GID[3]:-per device}/${NODE_GID[4]:-per device}/${NODE_GID[5]:-per device}."
  for r in $(seq 0 $(( TP - 1 ))); do log "[I_FABRIC] rank $r: NCCL on ${NODE_HCAS[r]} (bootstrap ${NODE_DEV[r]})."; done
}

# --- a ring without a switch (FABRIC_EXPECT=ring), from the 1104 recipe's scripts/tp4.sh ---------------------------------
# ring_graphs: an NCCL graph file for each rank, so that every ring channel runs over a cable: channel by channel a
# device toward the previous rank (where the node listens), a port and its PCIe twin in turn. Written under this
# Spark's STATE_DIR for rank 0 and under each worker's state directory for the others; mounted read-only.
ring_graphs() {
  local r prev path
  for r in $(seq 0 $(( TP - 1 ))); do
    prev=$(( (r + TP - 1) % TP ))
    if (( r == 0 )); then
      path="$STATE_DIR/nccl-graph-rank0.xml"
      if [[ "$DRY_RUN" == 1 ]]; then plan "write $path"; else mkdir -p "$STATE_DIR"; graph_xml "$r" "$prev" > "$path"; fi
    else
      path="${WORKER_STATE[r]}/nccl-graph-rank$r.xml"
      if [[ "$DRY_RUN" == 1 ]]; then plan "write $path on $(whost "$r")"
      else
        wssh_n "$r" mkdir -p "${WORKER_STATE[r]}" || fail E_FABRIC "Could not create ${WORKER_STATE[r]} on $(whost "$r")."
        graph_xml "$r" "$prev" | wssh_n "$r" sh -c "cat > $(printf '%q' "$path")" || fail E_FABRIC "Could not write $path on $(whost "$r")."
      fi
    fi
    NODE_GRAPH[r]=$path
  done
  log "[I_FABRIC] NCCL ring graphs written: $NCCL_CHANNELS channels a rank, each on a device toward the previous rank (NCCL_ALGO=$RING_NCCL_ALGO)."
}
# graph_xml <rank> <previous rank>: an NCCL graph whose channels listen on the rank's devices toward the previous rank.
graph_xml() {
  local r=$1 peer=$2 k n d idx ch='' xml
  local -a all=() toward=()
  IFS=, read -r -a all <<<"${NODE_HCAS[r]}"
  IFS=, read -r -a toward <<<"${TOWARD[$r $peer]:-}"
  (( ${#toward[@]} )) || fail E_FABRIC "rank $r has no RoCE device toward rank $peer."
  for (( k = 0; k < NCCL_CHANNELS; k++ )); do
    d=${toward[k % ${#toward[@]}]}; idx=-1
    for (( n = 0; n < ${#all[@]}; n++ )); do [[ "${all[n]}" == "$d" ]] && idx=$n; done
    ch+="    <channel><net dev=\"$idx\"/><gpu dev=\"0\" rank=\"$r\"/><net dev=\"$idx\"/></channel>"$'\n'
  done
  xml="<graphs version=\"1\">"$'\n'
  for k in "0 4" "1 3"; do
    xml+="  <graph id=\"${k% *}\" pattern=\"${k#* }\" crossnic=\"1\" nchannels=\"$NCCL_CHANNELS\" speedintra=\"12\" speedinter=\"12\" latencyinter=\"0\" typeintra=\"LOC\" typeinter=\"P2C\" samechannels=\"1\">"$'\n'"$ch  </graph>"$'\n'
  done
  for k in "4 6" "2 3" "3 5"; do
    xml+="  <graph id=\"${k% *}\" pattern=\"${k#* }\" crossnic=\"0\" nchannels=\"0\" speedintra=\"0\" speedinter=\"0\" latencyinter=\"0\" typeintra=\"LOC\" typeinter=\"LOC\" samechannels=\"0\"/>"$'\n'
  done
  printf '%s</graphs>\n' "$xml"
}

# A read-only per-boot fabric health record; no benchmark can displace serving traffic.
fabric_status() {
  local links tool
  links=$(rdma link 2>/dev/null) || fail E_FABRIC 'rdma link failed: install rdma-core and check RDMA permissions.'
  log "[I_FABRIC] kernel=$KERNEL_RELEASE driver=$NVIDIA_DRIVER sm_clock_cap=$NVIDIA_CLOCK MHz"
  log "$links"
  if ! grep -q 'state ACTIVE.*physical_state LINK_UP' <<< "$links"; then
    fail E_FABRIC 'No RDMA link is ACTIVE/LINK_UP. Check ConnectX-7 cabling and addresses.'
  fi
  if command -v ib_write_bw >/dev/null 2>&1; then
    tool=$(ib_write_bw --help 2>&1 || true)
    if grep -q -- '--report_gbits' <<< "$tool"; then log '[I_FABRIC] ib_write_bw available (RDMA write bandwidth; --report_gbits).'; else
      log '[I_FABRIC] ib_write_bw available; no --report_gbits support reported.'
    fi
  else warn W_FABRIC 'ib_write_bw unavailable: install perftest for an on-demand fabric bandwidth test.'; fi
}

# Rank r's placement and NCCL settings (RANK_ENV) and its mounts (RANK_MOUNTS). The bootstrap socket on the
# network every Spark shares; every RoCE device the rank has on the switch's subnet; NCCL picks its own algorithms
# and devices; no P2P or SHM transport (one GPU a node).
# NCCL_RUNTIME_CONNECT=1: the engine otherwise asks NCCL to connect every algorithm's channels at init (so a full GB10
# never registers them later); connected at first use instead, before the weights load. NCCL needs cuMem for that
# (NCCL_CUMEM_ENABLE=1); the host-side cuMem buffers stay off as the engine sets them.
# NCCL_SOCKET_IFNAME goes to rank 0 only: a follower that names one makes the engine unset NCCL_COMM_ID before NCCL
# (cuda_native.zig), and the engine's std library then stops at the first stderr lock after that. Without it a
# follower keeps NCCL_COMM_ID, and NCCL finds its bootstrap interface by that address's subnet: MASTER_ADDR is
# therefore this Spark's address on the network every Spark shares (the LAN), not an alias only this Spark holds.
tp6_rank_env() {
  local r=$1
  RANK_ENV=(-e "TF_GLM_RANK=$r" -e "NCCL_IB_HCA=${NODE_HCAS[r]}")
  local kv
  [[ "${KV:-}" == fp8 && "${KV8_CODES:-int8}" == int8 ]] && RANK_ENV+=(-e TF_GLM_KV8_INT8=1)
  # REQUEST includes these switches, so each rank must receive them; explicit user rank overrides remain last.
  for kv in "${!RANK_SWITCHES[@]}"; do RANK_ENV+=(-e "$kv=${RANK_SWITCHES[$kv]}"); done
  for kv in "${RANK_EXTRA_ENV_ARGS[@]}"; do RANK_ENV+=(-e "$kv"); done
  (( r != 0 )) || RANK_ENV+=(-e "NCCL_SOCKET_IFNAME=${NODE_DEV[r]}")
  [[ -z "${NODE_GID[r]}" ]] || RANK_ENV+=(-e "NCCL_IB_GID_INDEX=${NODE_GID[r]}")
  # ROCE_MESH: fabric/tp's bootstrap on rank 0's address on the switch's subnet, each rank's HCAs with their GID
  if [[ "${ROCE_MESH:-0}" == 1 ]]; then
    local hcas='' h
    for h in ${NODE_HCAS[r]//,/ }; do hcas+="${hcas:+,}$h${NODE_GID[r]:+:${NODE_GID[r]}}"; done
    RANK_ENV+=(-e TF_GLM_ROCE_MESH=1 -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/tp_mailbox.fatbin -e "TF_TP_MASTER=${LINK_HEAD_ADDR[1]}"
               -e "TF_TP_PORT=$((NCCL_PORT + 20))" -e "TF_TP_ROCE_HCA=$hcas" -e "TF_TP_MAILBOX_MAX_KB=$MESH_MAX_KB"
               -e "TF_TP_ROCE_FAST=${MESH_FAST:-1}")
  fi
  RANK_ENV+=(-e "TF_GLM_DSA_SPLIT_QKVA=${SPLIT_QKVA:-0}")
  # DSA_INDEX_FAST=1 (config.sh): the decode indexer's RoPE hoist and multi-CTA top-2048 (patch 0192), on every rank
  [[ "${DSA_INDEX_FAST:-0}" != 1 ]] || RANK_ENV+=(-e TF_GLM_DSA_ROPE_HOIST=1 -e TF_GLM_DSA_TOPK_MULTI=1)
  # KV_SPLIT=1 (config.sh): each token's latent rows on one rank (patches 0187-0189), the history prefetch off
  [[ "${KV_SPLIT:-0}" != 1 ]] || RANK_ENV+=(-e TF_GLM_KV_SPLIT=1 -e TF_GLM_KV_SPLIT_PREFETCH=0)
  # SLOT_RESUME (config.sh): a later prompt resumes on its slot's own prompt rows (patch 0195); every rank alike
  RANK_ENV+=(-e "TF_GLM_SLOT_RESUME=${SLOT_RESUME:-1}")
  # PF_EXPERTS_ONCE / PF_QMMF2 / PF_DOWN_FUSE (config.sh; patches 0206-0208): a 1 reaches every rank alike, a 0 passes nothing;
  # a TF_GLM_PF_<name>= already in RANK_EXTRA_ENV is left alone
  local pf pv kvw has
  for pf in EXPERTS_ONCE QMMF2 DOWN_FUSE; do
    pv=PF_$pf; has=0
    for kvw in "${RANK_EXTRA_ENV_ARGS[@]}"; do [[ "$kvw" == "TF_GLM_PF_$pf="* ]] && has=1; done
    [[ "${!pv:-0}" != 1 || $has == 1 ]] || RANK_ENV+=(-e "TF_GLM_PF_$pf=1")
  done
  [[ -z "${MTP_CAP:-}" ]] || RANK_ENV+=(-e "TF_GLM_MTP_CAP=$MTP_CAP")
  RANK_ENV+=(-e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_RUNTIME_CONNECT=1 -e NCCL_CUMEM_ENABLE=1 -e NCCL_CUMEM_HOST_ENABLE=0
             -e "NCCL_MIN_NCHANNELS=$NCCL_CHANNELS" -e "NCCL_MAX_NCHANNELS=$NCCL_CHANNELS")
  RANK_MOUNTS=()
  if [[ "${FABRIC_EXPECT:-mesh}" == ring ]]; then
    # A ring (the 1104 recipe's tp4_rank_env): NCCL_ALGO=Ring with a graph file whose channels run over the cables,
    # crossnic and subnet-aware routing (each cable joins port 0 of one Spark to port 1 of the next), no NIC fusion, and
    # the prefix that keeps a port and its PCIe twin apart. NCCL_RUNTIME_CONNECT=1 (above) matters more here: NCCL's
    # tree for four ranks joins rank 0 to rank 2, which no cable reaches, so only the ring's channels may ever connect.
    # ROCE_MESH writes to every peer directly and cannot run without a switch (start-switchless.sh refuses it).
    [[ "${ROCE_MESH:-0}" != 1 ]] || fail E_CONFIG 'ROCE_MESH=1 needs every Spark to reach every other directly (a switch): set ROCE_MESH=0 on a ring.'
    RANK_ENV+=(-e "NCCL_ALGO=$RING_NCCL_ALGO" -e NCCL_GRAPH_FILE=/nccl-graph.xml -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1
               -e NCCL_IB_MERGE_NICS=0)
    [[ -z "$RING_PREFIX" ]] || RANK_ENV+=(-e "NCCL_IB_SUBNET_PREFIX_LEN=$RING_PREFIX")
    [[ -n "${NODE_GRAPH[r]:-}" ]] || fail E_FABRIC "rank $r has no NCCL ring graph (ring_graphs did not run)."
    RANK_MOUNTS=(-v "${NODE_GRAPH[r]}:/nccl-graph.xml:ro")
  fi
}

# --- the workers' weights ------------------------------------------------------------------------------------------------
# NFS_SERVER<i> names the address worker i mounts from (NFS_SERVER for all); else the head's address on its route to
# that worker (this Spark's address on the shared subnet, the LINK_HEAD_ADDR the link discovery found).
tp6_nfs_server() { local v="NFS_SERVER$1"; printf '%s' "${!v:-${NFS_SERVER:-${LINK_HEAD_ADDR[$1]}}}"; }
# tp6_nfs_volume <rank> <volume> <exported path> <container path>: a read-only NFS volume on that worker, created when
# missing and refused when it points somewhere else; proven by a read before any rank starts (a hard mount that does
# not reach the server would wedge a load).
tp6_nfs_volume() {
  local i=$1 vol=$2 path=$3 at=$4 probe=${5:-config.json} server have
  server=$(tp6_nfs_server "$i")
  [[ -n "$server" ]] || fail E_WEIGHTS "No NFS server address for rank $i: set NFS_SERVER$i to this Spark's address as $(whost "$i") reaches it."
  if [[ "$DRY_RUN" == 1 ]]; then
    plan ssh "$(wtarget "$i")" docker volume create --driver local --opt type=nfs --opt "device=:$path" --opt "o=addr=$server,$NFS_OPTS" "$vol"
    return 0
  fi
  have=$(wssh_n "$i" docker volume inspect -f '{{index .Options "device"}} {{index .Options "o"}}' "$vol" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    wssh_n "$i" docker volume create --driver local --opt type=nfs --opt "device=:$path" --opt "o=addr=$server,$NFS_OPTS" "$vol" >/dev/null \
      || fail E_WEIGHTS "Could not create the Docker volume $vol on $(whost "$i")."
  elif [[ "${have%% *}" != ":$path" || "$have" != *"addr=$server,"* ]]; then
    fail E_WEIGHTS "$(whost "$i")'s Docker volume $vol is ${have%% *} from ${have#* }, not :$path from $server. Remove it there (docker rm -f $vol-pin; docker volume rm $vol), or set NFS_MODEL_VOLUME / NFS_DFLASH2_VOLUME / NFS_SERVER$i."
  fi
  wssh_n "$i" docker run --rm --network host -v "$vol:/w:ro" --entrypoint /bin/sh "$IMAGE" -c "test -f /w/$probe" >/dev/null 2>&1 \
    || fail E_WEIGHTS "$(whost "$i") cannot read $path over NFS from $server (volume $vol). The NFS server on this Spark must export that path, read-only, to $(whost "$i")'s address on its route here (${LINK_WORKER_ADDR[i]:-its link address}); see README \"6 Sparks\"."
  tp6_pin_volume "$i" "$vol"
  log "[I_WEIGHTS] rank $i ($(whost "$i")) reads $at over NFS from $server:$path (volume $vol)."
}
# tp6_pin_volume <rank> <volume>: a sleeping container on that worker that keeps the NFS volume mounted between starts.
# The engine's checkpoint-digest cache keys each file on (st_dev, inode, size, times), and an NFS mount's st_dev is an
# anonymous device number handed out when Docker mounts the volume for a rank. On a worker that starts and stops other
# containers between two of our starts (spark5) the number moved every time (its cache held devs 234, 118, then 120),
# so that rank hashed all 432 GiB again (~2 min, 0/50 cached) while the other five waited. A volume mounted the whole
# time keeps one superblock, so one st_dev, and the cache hits. TP6_PIN_WEIGHTS=0 turns it off.
tp6_pin_volume() {
  local i=$1 vol=$2 pin="$2-pin" state
  [[ "${TP6_PIN_WEIGHTS:-1}" == 1 ]] || return 0
  state=$(wssh_n "$i" docker inspect -f '{{.State.Running}}' "$pin" 2>/dev/null || true)
  [[ "$state" != true ]] || return 0
  [[ -z "$state" ]] || wssh_n "$i" docker rm -f "$pin" >/dev/null 2>&1 || true
  wssh_n "$i" docker run -d --name "$pin" --label tf.recipe=glm53-nvfp4-pin --restart unless-stopped --network none \
    --memory 64m -v "$vol:/w:ro" --entrypoint /usr/bin/sleep "$IMAGE" infinity >/dev/null \
    || { warn W_WEIGHTS "Could not start $pin on $(whost "$i"); its digest cache may miss and re-hash the checkpoint."; return 0; }
  log "[I_WEIGHTS] $(whost "$i"): $pin keeps $vol mounted between starts (a stable st_dev for the digest cache)."
}
# 0 when any rank's container was OOM-killed by the kernel (exit 137, no engine refusal in its log).
tp6_any_oom_killed() {
  local i
  [[ "$(docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER_NAME" 2>/dev/null || true)" != true ]] || return 0
  for i in $(worker_ids); do
    [[ "$(wssh_n "$i" docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER_NAME" 2>/dev/null || true)" != true ]] || return 0
  done
  return 1
}
tp6_check_exlr8() {
  [[ "${MODEL_FORMAT:-}" == exlr8 && "$DRY_RUN" != 1 ]] || return 0
  local i identity digest
  local checker="$REPO_ROOT/scripts/exlr8/dirstate.py"
  identity=$(exlr8_checkpoint "$MODEL_DIR" 0) \
    || fail E_WEIGHTS "Rank 0 needs a complete $EXLR8_DENSE EXLR8 checkpoint in $MODEL_DIR. Run EXLR8_RELEASE=$EXLR8_RELEASE EXLR8_DENSE=$EXLR8_DENSE scripts/prepare-exlr8.sh."
  for i in 1 2 3; do
    digest=$(wssh_n "$i" env "EXLR8_RELEASE=$EXLR8_RELEASE" python3 -IB - checkpoint "$WORKER_MODEL_DIR" "$i" 4 "$EXLR8_DENSE" < "$checker") \
      || fail E_WEIGHTS "Rank $i ($(whost "$i")) needs a complete $EXLR8_DENSE EXLR8 checkpoint in $WORKER_MODEL_DIR. Run EXLR8_RELEASE=$EXLR8_RELEASE EXLR8_DENSE=$EXLR8_DENSE scripts/prepare-exlr8.sh."
    [[ "$digest" == "$identity" ]] \
      || fail E_WEIGHTS "Rank $i ($(whost "$i")) has a different EXLR8 config or checkpoint set from rank 0."
  done
  EXLR8_SET_ID=${identity##* }
}

# tp6_prepare_worker_weights: the volumes on every worker (WORKER_WEIGHTS=nfs), or a check that each worker holds its
# own copy (check). rsync is not offered at six: five copies of ~420 GiB is what NFS is for.
tp6_prepare_worker_weights() {
  local i mpath dpath
  mpath=${NFS_MODEL_PATH:-$MODEL_FS_ROOT}; dpath=${NFS_DFLASH2_PATH:-$DFLASH2_DIR}
  local rel=${MODEL_DIR#"$MODEL_FS_ROOT"}; rel=${rel#/}; local mprobe="${rel:+$rel/}config.json"
  for i in $(worker_ids); do
    case "$WORKER_WEIGHTS" in
      nfs)
        tp6_nfs_volume "$i" "$NFS_MODEL_VOLUME" "$mpath" "$MODEL_FS_ROOT" "$mprobe"
        if [[ "$DRAFTER" == dflash2 && "${WORKER_DFLASH2_LOCAL:-0}" == 1 ]]; then
          # the drafter from the worker's own copy in WORKER_DFLASH2_DIR (a folder this Spark does not export)
          [[ "$DRY_RUN" == 1 ]] || wssh_n "$i" test -r "$WORKER_DFLASH2_DIR/config.json" >/dev/null 2>&1 \
            || fail E_WEIGHTS "$(whost "$i") has no readable drafter in $WORKER_DFLASH2_DIR (WORKER_DFLASH2_LOCAL=1)."
          log "[I_WEIGHTS] rank $i ($(whost "$i")) reads the drafter from its own copy (WORKER_DFLASH2_LOCAL=1)."
        else
          [[ "$DRAFTER" != dflash2 ]] || tp6_nfs_volume "$i" "$NFS_DFLASH2_VOLUME" "$dpath" /dflash2
        fi ;;
      check)
        [[ "$DRY_RUN" == 1 ]] && continue
        wssh_n "$i" test -f "$WORKER_MODEL_DIR/config.json" >/dev/null 2>&1 \
          || fail E_WEIGHTS "$(whost "$i") has no checkpoint in $WORKER_MODEL_DIR (WORKER_WEIGHTS=check). Set WORKER_WEIGHTS=nfs to mount this Spark's copy."
        [[ -z "$MODEL_REVISION" || "$(wssh_n "$i" cat "$WORKER_MODEL_DIR/.revision" 2>/dev/null || true)" == "$MODEL_REVISION" ]] \
          || fail E_WEIGHTS "$(whost "$i")'s checkpoint in $WORKER_MODEL_DIR is not revision ${MODEL_REVISION:0:12} (WORKER_WEIGHTS=check)."
        [[ "$DRAFTER" != dflash2 ]] || wssh_n "$i" test -f "$WORKER_DFLASH2_DIR/config.json" >/dev/null 2>&1 \
          || fail E_WEIGHTS "$(whost "$i") has no DFlash2 weights in $WORKER_DFLASH2_DIR (WORKER_WEIGHTS=check)."
        log "[I_WEIGHTS] rank $i ($(whost "$i")) holds the checkpoint and the drafter." ;;
      *) fail E_CONFIG "WORKER_WEIGHTS=$WORKER_WEIGHTS: at six Sparks the workers read the checkpoint over NFS (nfs) or hold their own copy (check)." ;;
    esac
  done
}
# tp6_worker_mounts <rank>: WMOUNTS, that worker's -v arguments: the weights, the config.json overlay (placed there),
# the chat template (placed there), the rank's HOME (its digest cache) and its NCCL graph.
tp6_worker_mounts() {
  local i=$1
  WMOUNTS=()
  if [[ "$WORKER_WEIGHTS" == nfs ]]; then
    WMOUNTS+=(-v "$NFS_MODEL_VOLUME:$MODEL_FS_ROOT:ro")
    if [[ "${WORKER_DFLASH2_LOCAL:-0}" == 1 ]]; then [[ "$DRAFTER" != dflash2 ]] || WMOUNTS+=(-v "$WORKER_DFLASH2_DIR:/dflash2:ro")
    else [[ "$DRAFTER" != dflash2 ]] || WMOUNTS+=(-v "$NFS_DFLASH2_VOLUME:/dflash2:ro"); fi
  else
    WMOUNTS+=(-v "$WORKER_MODEL_DIR:$MODEL_DIR:ro")
    [[ "$DRAFTER" != dflash2 ]] || WMOUNTS+=(-v "$WORKER_DFLASH2_DIR:/dflash2:ro")
  fi
  if (( ${#OVERLAY_MOUNT[@]} )); then
    [[ "$DRY_RUN" == 1 ]] || wssh_n "$i" tee "${WORKER_STATE[i]}/model-config.json" < "$STATE_DIR/model-config.json" >/dev/null \
      || fail E_WEIGHTS "Could not place the config.json overlay on $(whost "$i")."
    WMOUNTS+=(-v "${WORKER_STATE[i]}/model-config.json:$MODEL_DIR/config.json:ro")
  fi
  if [[ -n "$TEMPLATE_FILE" ]]; then
    [[ "$DRY_RUN" == 1 ]] || wssh_n "$i" tee "${WORKER_STATE[i]}/chat_template.jinja" < "$TEMPLATE_FILE" >/dev/null \
      || fail E_START "Could not place the chat template on $(whost "$i")."
    WMOUNTS+=(-v "${WORKER_STATE[i]}/chat_template.jinja:/templates/chat_template.jinja:ro")
  fi
  WMOUNTS+=(-v "${WORKER_CACHE[i]}:/state" "${RANK_MOUNTS[@]}")
}

# --- logs and the memory guard over six Sparks ------------------------------------------------------------------------------
tp6_collect_logs() {
  local i
  [[ -n "${RUN_LOG_DIR:-}" && -d "$RUN_LOG_DIR" ]] || return 0
  docker logs "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank0.log" 2>&1 || true
  for i in $(worker_ids); do wssh_n "$i" docker logs "$CONTAINER_NAME" > "$RUN_LOG_DIR/rank$i.log" 2>&1 || true; done
}
# Every Spark's weights, KV pool and everything else it runs share one pool, and a box driven into reclaim can stop
# answering long before the OOM killer picks a victim. The guard samples this Spark every tick and one worker a tick
# in turn (an ssh round trip each), and stops the server when any Spark stays under the floor.
tp6_start_memguard() {
  [[ "$MEMGUARD" == 1 ]] || return 0
  if [[ "$DRY_RUN" == 1 ]]; then
    log "Memory guard: sample MemAvailable (plus parked pages) on every Spark (this one each ${MEMGUARD_INTERVAL_S} s, a worker"
    log "in turn) and run ./stop-tp6.sh when one stays under ${MEMGUARD_FLOOR_GIB} GiB for ${MEMGUARD_HOLD_S} s."
    return 0
  fi
  ( exec 8>&-
    declare -a under=(0 0 0 0 0 0) sample=('' '' '' '' '' '')
    tick=0
    # The floor comparison runs on raw bytes (mem_avail_bytes), not the one-decimal GiB value: a sample just under the
    # floor rounds up onto it ("2 GiB - 2 KiB" prints as "2.0"), and a rounded comparison left a Spark running below a
    # raw 2 GiB MemAvailable. Rounding decides only what the log prints.
    floor_bytes=$(awk -v g="$MEMGUARD_FLOOR_GIB" 'BEGIN { printf "%.0f", g * 1073741824 }')
    while sleep "$MEMGUARD_INTERVAL_S"; do
      [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == true ]] || exit 0
      sample[0]=$(mem_avail_bytes)
      w=$(( tick % (TP - 1) + 1 )); sample[w]=$(tp6_worker_avail_bytes "$w" || true)
      tick=$(( tick + 1 ))
      for r in $(seq 0 $(( TP - 1 ))); do
        if [[ -n "${sample[r]}" ]] && ! at_least "${sample[r]}" "$floor_bytes"; then
          under[r]=$(( under[r] + MEMGUARD_INTERVAL_S ))
        else under[r]=0; fi
        if (( under[r] >= MEMGUARD_HOLD_S )); then
          who=$(hostname); (( r == 0 )) || who=$(whost "$r")
          printf '%s [memguard] %s: %s GiB available, under the %s GiB floor for %s s; stopping the server\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$who" \
            "$(awk -v b="${sample[r]}" 'BEGIN { printf "%.1f", b / 1073741824 }')" \
            "$MEMGUARD_FLOOR_GIB" "$MEMGUARD_HOLD_S" | tee -a "$RUN_LOG_DIR/memguard.log" >&2
          MEMGUARD=0 "$REPO_ROOT/stop-tp6.sh" >> "$RUN_LOG_DIR/memguard.log" 2>&1 || true
          exit 0
        fi
      done
    done ) </dev/null >>"$RUN_LOG_DIR/memguard.log" 2>&1 &
  MEMGUARD_PID=$!
  log "[I_MEMGUARD] Watching the $TP Sparks that serve: a stop below ${MEMGUARD_FLOOR_GIB} GiB held ${MEMGUARD_HOLD_S} s (pid $MEMGUARD_PID; MEMGUARD=0 turns it off)."
}
