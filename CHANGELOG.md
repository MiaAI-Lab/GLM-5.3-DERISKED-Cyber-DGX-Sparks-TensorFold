# Changelog

- **2026-11-26** — **`./stop.sh` replaces `./stop-tp6.sh`** and stops either layout: `./start.sh` (TP4) or
  `./start-tp6.sh` (TP6). It already checked all six Sparks; only the name and its messages change.
- **2026-11-25** — **Four Sparks without a switch: the cabled ring (`FABRIC_TOPOLOGY=ring`).**
  - Four Sparks cabled in a closed ring (each Spark's port 0 to the next Spark's port 1, the last back to the
    first; two subnets a cable, one per PCIe rail). `FABRIC_TOPOLOGY=ring` is the only required ring setting: the
    ring also defaults `WORKER_WEIGHTS=check` (local copies, no NFS; the exlr8 lane's mode) and `ROCE_MESH=0`.
    `detect_links` classifies the cabling and refuses any shape that is neither a switch (every pair on a shared
    RoCE subnet) nor a closed ring over the four ranks in WORKER order — including a ring under `auto`, which is
    refused rather than half-configured, and a rank whose two cabled ports carry different RoCE v2 GID indices
    (one `NCCL_IB_GID_INDEX` is pinned a rank). Ring ranks get `NCCL_ALGO=Ring`, `NCCL_IB_MERGE_NICS=0`,
    `NCCL_IB_SUBNET_AWARE_ROUTING=1` (stock NCCL algorithm selection opens channels between ranks no cable
    reaches) and `PREFILL_REDUCE=ring` (the exchanges pass rank to rank, engine patch 0106). `auto` keeps the
    switch path unchanged. Decode on the ring is unmeasured end to end on this engine; published four-node
    ring-vs-switch comparisons span about 1-10% slower decode.
  - Tests: `scripts/tests/fabric-topology.sh` (edge extraction, classification with its refusals, ring-order
    check, no synthetic Sparks needed).
- **2026-11-25** — **K4 with the unified and neon stacks; gr8 quants supported, not bundled; 3 GiB reserve.**
  - **K4 is now `wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC` @ `e28931c9`** (K4-v1 was removed from Hugging Face), served as
    `GLM-5.3-EXL3-K4`: its `original` view (EXL3 MCG K4 experts, byte-identical in size to K4-v1's, BF16 core packed to
    8 bits at load). The start downloads only that view's 99 shards and config (`MODEL_INCLUDE`, ~418 GB); the
    k8/k6-compact cores are skipped because the engine's GLM loader cannot read an EXL3 dense core. Live at TP6: model
    67.88 GiB, 0.9 GiB left at 847,872 tokens and a 4 GiB reserve.
  - **K4 is the default checkpoint; Derisked is the DERISKED lane** (`MODEL_FORMAT=derisked`). `MODEL_FORMAT=k4` was
    `wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1` at `47af2334` in the HF cache (NFS volume `glm53-k4v1-model`). It is no longer on
    Hugging Face, so a local copy is needed. `./start.sh` gives K4 and Derisked the same serving defaults: 294,912 tokens,
    3 GiB reserve, no KV split, the neon stack. sparkDash at TP4 (~2,190 MHz): K4 prefill 1,571 (2K) to 1,301 (128K) tok/s,
    decode prose 38.8, at 64K 33.3, code 52.5; Derisked prefill 1,582 (2K) to 1,233 (256K), decode prose 38.9, at 64K
    35.6, code 45.8 (temperature 0.7, thinking on).
  - **0238–0242** (`30e4c73`): `TF_GLM_NEON_FOLD=1` (a fused chunk's second half folds its shared slot in the shared
    down epilogue) and the six-Spark waves (PF_EXPERTS_ONCE at widths 384/256, per-expert balanced EXL3 sharding,
    DFlash2 qkv/o_proj split across ranks). Policy mask `0x7fcffd1c`. `SPLIT_QKVA`, `DSA_INDEX_FAST` and the `PF_*`
    switches are on at TP6 as at TP4.
  - **K4 measured with the stack** (`wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1` @ `47af2334`, TP4, image `4e6cb710b154`,
    212,992 window, GPU clocks ~2,190 MHz), stack off then on: prefill 1,176 / 1,197 / 1,128 -> 1,282 / 1,321 / 1,264
    tok/s at 9K / 37K / 150K (+9.0 / +10.3 / +12.0%), decode prose 26.1 -> 28.7 (+10.0%), code 40.4 -> 43.5 (+7.7%);
    the policy bits alone in one boot +7.0 / +6.9 / +8.3 / +11.9% decode (prose / code / 61K / 190K). 24 of 24 replies
    identical to the stack-off boot, drafts on and off; the needle at 191,692 tokens passes. At a 3 GiB reserve K4 serves
    294,912 tokens (0.3 GiB left; needle at 265,420 passes, MemAvailable floor 2.81 GiB).
  - `MEMORY_RESERVE_GIB` defaults to **4** for every checkpoint and width (owner). At 3 an idle six-Spark server at
    868,352 tokens left spark1 2.7 GiB over the memory guard's 2 GiB floor, and a 2.9 GiB process there stopped it.
    The windows shrink to match: four Sparks 262,144 (0.7 GiB left), six Sparks 847,872 (1.3 GiB left; idle
    MemAvailable 4.6-6.0 GiB on every Spark). The NVFP4 checkpoint's credit entries are removed (not used).
  - **`SHARED_KV`** (owner request): four streams on one shared KV pool (patches 0210-0211, `TF_GLM_SEQ_GROW=1`,
    `TF_GLM_DSA_SHARED=1`), on by default at six Sparks for K4 and Derisked with `PARALLEL=4`, and the six-Spark window
    (the shared pool) raised to 868,352 tokens: Derisked at TP6 leaves 1.3 GiB after the 3 GiB reserve (11.1 GiB at
    655,360), 48 MiB a further stream; the needle in an 800,672-token prompt passes with every Spark at 3.48 GiB available
    or more; four greedy 384-token prompts 46.4 tok/s a stream alone, 64.5 tok/s aggregate together, 4 of 4 replies
    identical to solo.
  - The README no longer offers EXLR8 up1: gr8 (EXLR8) quants can be prepared and served (`MODEL_FORMAT=exlr8`) but none
    is bundled or served by default. The README was rewritten and ends with a Credits section; CREDITS and NOTICE credit
    patch 0214 and the Derisked and NVIDIA checkpoints.

- **2026-11-24** — **Derisked EXL3 4 bpw: the unified and neon stacks by default, no KV split, 294,912 tokens.**
  Image `tensorfold-glm53nvfp4-spark:99841b4cfc0b` (patches 0001–0237). Patch 0133 no longer carries the 114 build
  artifacts (`prefix/`, `.src/prefix/` binaries, 144 MB) it was generated with; its one source change is unchanged, the
  patched source tree is identical, and the image is the built `0c348c513364` relabelled (same layers).
  - **The default checkpoint is Mia's AI Lab's quant** (`MODEL_FORMAT=derisked`, was kindlingai's EXLR8 up1):
    `DERISKED_DIR` (`~/models/GLM-5.3-Derisked-EXL3-4bpw`) when it holds it, otherwise downloaded there from
    [Mia-AiLab/GLM-5.3-DERISKED-Cyber-EXL3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-DERISKED-Cyber-EXL3-4bpw) at
    revision `48aee9fb` (`DERISKED_REVISION` pins another). `MODEL_FORMAT=exlr8`, `k4` or `nvfp4` selects the others;
    `scripts/prepare-exlr8.sh` defaults to exlr8. A download start no longer stops on the Flash recipe's unset
    `ABLIT_ID`.
  - **0218–0234** (unified and neon optimization runs): the
    start sets request policy mask `0x7f4d7c10` and `TF_GLM_UNI_EY_BF16`, `TF_GLM_UNI_KSPLIT_ALL`, `TF_GLM_UNI_X3_ONCE`,
    `TF_GLM_NEON_BUFS`, `TF_GLM_NEON_DF_DOWN`, `TF_GLM_NEON_DF_FC`. Every reply is token-identical to the stack
    without them (24 of 24, drafts on and off). `NEON_STACK=0` leaves them out.
  - **Defaults for the derisked start:** `KV_SPLIT=0`, `MEMORY_RESERVE_GIB=3`, `CONTEXT=294912` (owner decision). The
    window is pinned: unsplit, the engine gives no memory refusal to fit from, and `CONTEXT=auto` lets the kernel
    OOM-kill ranks at 1,048,576 and 524,288 tokens (one Spark fell to 0.14 GiB free during that probe). 294,912 is
    262,144's measured 0.7 GiB spare over a 4 GiB reserve plus the freed 1 GiB, at about 50 KB a token a rank.
  - **0235–0237**, ported from closed pull requests of the
    [GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104) recipe (authors in
    `CREDITS.md`): 0235, a resumed turn that has not started is ranked by its uncached tail (#20, upstream
    TensorFold #2276); 0236, `--learn-conversations` stages the longest state whole instead of skipping it with
    `StagingFull` (#49); 0237, a refused disk read tries one shorter disk state before a cold prefill (#41, upstream
    TensorFold #2442). 0236 and 0237 are regenerated against this stack; their changed lines are the originals'. The
    engine's unit tests fail the same 6 tests with and without them; their 6 new tests pass. 0235 acts at two or more
    streams, 0236 and 0237 with `LEARN_GIB` above 0.
  - **Measured** (`scripts/tests/bench.py`, one request, fp8 KV with int8 codes, DFlash2):

    | | clock | prefill 9k / 37k / 150k tok/s | decode prose / code tok/s |
    | --- | --- | --- | --- |
    | unified stack only (N0), 196,608 | 1,800 MHz | 1,063 / 1,077 / 1,019 | 25.4 / 39.8 |
    | this stack, 262,144, 4 GiB reserve | 1,800 MHz | 1,140 / 1,191 / 1,134 | 25.9 / 40.6 |
    | **this stack, 294,912, 3 GiB reserve** | **~2,190 MHz (cap lifted)** | **1,255 / 1,263 / 1,234** | **27.0 / 42.0** |

    A needle at 265,420 tokens passed. Lowest MemAvailable during the gates at 294,912: spark1 2.53 GiB, spark3
    2.77, spark2 3.46, spark4 4.04 (memguard floor 2 GiB). The 3 GiB reserve is the engine's budget, so the free
    memory a Spark reports sits about 0.5 GiB lower; `CONTEXT=286720` keeps every Spark at 3 GiB or more.

- **2026-11-21** — **Derisked EXL3 4 bpw: mul1 prompt and decode kernels (patches 0215–0216), no KV split, 196,608
  tokens.** `./start.sh` serves `GLM-5.3-Derisked-EXL3-4bpw` (EXL3 4-bit routed experts on the mul1 codebook, BF16
  dense packed to 8 bits at load). Every mcg-only fast path skipped it: all 75 MoE layers prefilled through the
  generic `runSlots` path, so `PF_EXPERTS_ONCE` and `PF_DOWN_FUSE` did nothing.
  - **0215** gives the prompt expert kernels (gate/up, down, down_b16, fused down, gate/up once) a mul1 form, for
    4-bit gate/up and down. One MoE layer, 4,096 rows: 33.5 ms against 285 ms on `runSlots` (mcg: 33.0 ms). Within
    1e-3 RMS of `runSlots`; the once and fused-down forms are byte-identical to the plain ones. The mcg kernels'
    SASS is unchanged.
  - **0216** gives the decode grouped kernels' window forms (nt4, r8, read-once, PDL) a mul1 form. Byte-identical
    to the generic grouped kernel at 1–32 rows; `TF_EXL3_MUL1_DECODE=0` turns it off.
  - **Defaults for the derisked start:** `KV_SPLIT=0`, `CONTEXT=196608`, `MEMORY_RESERVE_GIB=4` (owner decision). The
    8.5 GiB reserve was set for TP6's uneven expert split; at TP4 the experts split evenly. The KV split costs the
    two-halves prefill schedule and about 10% decode.
  - **Measured** (`scripts/tests/bench.py`, one request, fp8 KV with int8 codes, DFlash2, 1800 MHz cap):

    | | prefill 9k / 37k / 150k tok/s | decode prose / code tok/s |
    | --- | --- | --- |
    | image `fa6428300d98` (before), KV split, 163,840 | 340 / 337 / 326 | 21.5 / 30.7 |
    | image `130f753ae9b4` (0215–0216), KV split, 163,840 | 949 / 943 / 869 | 21.3 / 34.7 |
    | **`130f753ae9b4`, no KV split, 196,608, 4 GiB reserve** | **1,052 / 1,073 / 1,010** | **23.9 / 37.8** |

    A needle at 176,947 tokens passed. Lowest MemAvailable during the needle and the bench: spark1 3.1 GiB (memguard
    floor 2 GiB), spark2–4 5.7 GiB or more. The engine reports 0.0 GiB left above the reserve, so spark1 has about
    1 GiB of headroom: other work on spark1 can trip memguard.
  - Prose decode is bound by DFlash2's acceptance: 1.9 tokens a round on prose against 3.6 on code.

- **2026-11-20** — **No fixed default reply limit.** `MAX_TOKENS` now defaults to 1,048,576 (was 32,768): a request
  that sends no `max_tokens` runs to the end of the served window, since the server caps the default to the room the
  prompt leaves. Two effort-max chat replies had ended at 32,768 tokens still thinking, with empty content. A
  request's own `max_tokens` still wins, so a client with its own cap (Pi's `maxTokens`) has to raise it too.

- **2026-11-20** — **EXLR8 up1 TP4 native dense and shared-window startup executed.** The measured C=2 configuration is
  now the default only for up1 TP4: native stored-trellis decode, 196,608 tokens, sequence growth on both streams,
  2,560 MiB fixed-part admission floor, and `TF_GLM_DSA_SHARED=1`. Startup compacts each host after stop and before
  memory checks; every rank records the kernel, driver, clock cap and active RoCE links. A bounded bidirectional
  `ib_write_bw` probe on spark1–spark2 measured 194.74 Gbit/s at 4,096 bytes over `rocep1s0f1` GID 3.
  Native/C=2 measured prefill 979.9 / 1,006.5 / 949.4 tok/s at 9,385 / 37,421 / 149,529 tokens and sparkDash decode
  26.8 prose / 33.3 code tok/s. Greedy code, greedy prose and seeded sampled prose all matched serial generation
  token-for-token. Two concurrent 512-token requests completed and a 184,952-token needle passed at depth 0.85.
  The request and rank switch forwarding are regression-tested; NVFP4 TP6 and K4 defaults remain unchanged.

- **2026-11-19** — **EXLR8 up1 is the recipe default again.** `EXLR8_RELEASE=up1|v07` selects one of kindlingai's
  pinned EXLR8 releases. up1 uses the mcg codebook with E4M3 rounding and FP8 row-scaled vocabulary. v0.7 remains
  available as profile `bit400` with the mul1 codebook, E4M3 rounding and BF16 vocabulary. v0.7 matched up1's measured
  quality and reached about 348 tok/s prefill, compared with about 1,030 tok/s on up1.
  - Each rank fetches its own expert ranges using the selected manifest and SHA-256 index list. Completion validates the
    pinned revision, set ID, codebook and coverage. Native conversion keeps its staged publication and validates the
    source before linking interim pieces. Launch checks all four ranks before replacing a running service.
  - Both releases default to 196,608 tokens. spark1 hit the 2304 MiB `TF_GLM_START_FLOOR_MIB` floor at 229,376 on v0.7.
- **2026-11-19** — **EXLR8 v0.7 support** added `mul1_e4m3` expert pieces and engine routing (patch 0214).
  kindlingai authored the format, the reference loader and both releases of the checkpoint. v0.7 was briefly the
  recipe default and is selectable as `EXLR8_RELEASE=v07`. It pins commit `9e25c28a2e8808d8b2599d780521fd6af24e7558`
  and set ID `3d1a293399f0077e27acd18609ee67f33f8e3f8b5963a7c4b42b7dc16ecd1d98`.
  [Model repository](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) · [Reference loader](https://github.com/kindlingai/exlr8)

- **2026-11-18** -- **EXLR8 native dense (0212–0213)** is opt-in with `EXLR8_DENSE=native`. Prepare and start with that mode.
  Native dense leaves the stored trellis intact for decode. Compatible interim pieces can be hard-linked into `~/exlr8-up1-x8b`.
  The converter writes dense shards in a private staging directory and publishes a completion marker only after verifying the full
  checkpoint. Layer tests stay pending. A start validates the mode and the conversion set on all four ranks before replacing a
  running service. See the native dense measurements below.

- **2026-11-18** -- **Concurrency patches 0210 and 0211** default to the sequence-growth pool off and serialized DSA verification.
  **0210** allocates a startup word buffer and runs an NCCL all-gather even with the pool off. With `TF_GLM_SEQ_GROW=1`, each rank
  physically allocates a full context window's KV pool at startup, then streams map 2 MiB granules as their rows grow. Set
  `PARALLEL` and `TF_GLM_SEQ_GROW_STREAMS` together. Two streams use 2 for both. Every rank checks the requested number of additional
  streams above `TF_GLM_SEQ_GROW_FLOOR_MIB` (default 3072 MiB). Sequence growth requires unsplit, unsharded, unpaged KV and zero
  device and host snapshot budgets. **0211** runs one DSA verification window at a time by default. `TF_GLM_DSA_SHARED=1` lets
  several streams share a decode window (`cuda_shared.dsaSegs`). A plain start requests the current recipe image and replaces an
  older running image for any model format. They are the 0203 and 0204 of the K4 stages rebased onto 0001-0195 + 0205-0209 (patch-by-patch `git apply` in
  filename order; the resulting `zig/` tree is the built tree's, identical tree hash). **Measured (K4, TP4, 212,992, 1800 MHz cap; stage
  x6g-1117-163858, i.e. x7):** one stream alone unchanged (+0.9 to +2.0%), 23 of 23 replies identical to the release; shared window:
  aggregate 36.7 tok/s at C=1, **44.7 at C=2 (+22%), 54.8 at C=4 (+49%)**, every concurrent reply equal to its solo reply (36/36, 32/32,
  long 2/2, big-together 2/2). Per-stream throughput was 24.2 and 15.3 tok/s at C=2 and C=4, with TTFT of 0.39 and 0.63 s. Fallback (rounds one at a time): 36.2 / 36.5, no gain. **Measured here, offline (spark6 GB10, 2026-11-18):**
  the patched tree builds (spark6, the release's builder container, release flags; binary b19059aac4fc4e76); host tests 117 of 117 in 22 steps (test-glm-seq-grow-host 17, test-glm-dsa-shared-host 12, equal to the K4 runs so not vacuous; policy 24, slot-resume 13, kv-split 10, fp8-kv 31, halves 10); four mutations of the floor code (floor ignored, default 0, minimum 1024, ranks not compared) each fail a named test; GPU test-glm-seq-grow-gpu 8 of 8; one real up1 layer (39, rank 0, E4M3 image) with its real tap rows and routes: every row's decode outputs byte-identical alone and inside shared windows of 4 to 32 rows in 8 window shapes (tf-cuda-test glm-conc-rows-e4m3, 11 of 11 cases incl. 3 controls; also with TF_EXL3_SHFL=0 and TF_EXL3_PDL=0; the K4 layer 11 of 11). **Not measured:** anything on EXLR8 up1 (window 229,376) end to end: stage `night/stages/x11-conc-up1.sh` is written, not run;
  the up1 memory margin is derived, not measured (spark1 idles at 2.74-3.08 GiB, a 4-stream pool adds ~0.3 GiB: expect ~2.4 GiB against
  MEMGUARD's 2 GiB). TP6 concurrency remains unmeasured.

- **2026-11-18** -- **Prefill patches 0206-0208 on by default at TP4** (owner decision: "include 0206-0208 in up1 too"). Engine patch
  **0206** (`TF_GLM_PF_EXPERTS_ONCE`: prompt gate/up rotate member rows once in shared memory), **0207** (`TF_GLM_PF_QMMF2`: block-FP8
  prompt GEMMs on qmmf2's 128x64 tile) and **0208** (`TF_GLM_PF_DOWN_FUSE`: the routed down sums the route slots), and **0209**, the GPU
  tests for the E4M3 image (no engine behaviour). They apply in filename order on 0001-0195 + 0205, and the resulting `zig/` tree is the
  reviewed stack tree's (`diff -r` empty). New settings `PF_EXPERTS_ONCE`, `PF_QMMF2`, `PF_DOWN_FUSE`: `1` at TP4 (EXLR8 up1, K4,
  a named K4 checkpoint), `0` at TP6 and for everything else; `=0` opts out, an exported `TF_GLM_PF_*` sets the default too. Measured
  with all three on (1800 MHz cap): EXLR8 up1 prefill 1,029 / 1,039 / 1,005 tok/s at 9K / 37K / 100K against 929 / 933 / 905
  (+11%), 18 of 18 replies identical, decode 1.004x (stage x9u-1118-004526); K4 1,082 / 1,095 / 1,059 (+18%), 18 of 18 identical, decode
  1.022x (stage x9ab-1117-225930). TP6 remains unmeasured. `PF_QMMF2` changes nothing on
  up1 (fp8g weight mode only), and `PF_EXPERTS_ONCE` is TP4 only (its kernel is written for width 512).

- **2026-11-17** -- **EXLR8 up1 is the default model** (owner decision: "make up1 the default, I see no reason to use K4").
  kindlingai's EXLR8 release (`kindlingai/glm-5.3-exlr8-k2-k3-k4` @ a36ae468, exlr8-v2, profile `up1`: every routed expert at
  4 bits), TP4 on spark1-4 (`TP6_RANKS=4` implied), 8-bit KV (int8 codes), DFlash2, `SPLIT_QKVA=1`, `DSA_INDEX_FAST=1`,
  `SLOT_RESUME=1` as before, and **`CONTEXT=229376`** (K4: 212,992). `MODEL_FORMAT=exlr8|k4|nvfp4` selects the checkpoint; unset it
  means exlr8 unless `HF_CACHE_DIR` or `MODEL_DIR` is already set (a checkpoint named that way is served exactly as before, so
  every K4 start with its settings spelled out is unchanged); an exported variable always outranks a format's default.
  `MODEL_FORMAT=k4` is K4 as it was the default 2026-11-16 to 2026-11-17 (same variables as spelling them out).
  - Engine patch **0205** (nine commits on TensorFold at 0195, plus a tenth that makes `tf-cuda-test` compile: the recipe image builds it,
    and the first image build of 0205 failed on two test compile errors, `glm_exl3_layer_tests` and `glm_loader_gate`: `CB_MCG_E4M3`, the E4M3 expert image, EXLR8 loading, rank-piece
    agreement through `exlr8-set.json`); it applies on 0001-0195 and its tree is the one the validated binary (9e47d6aa6d5e33da)
    was built from. Patch 0205b was the prototype for native dense, which later shipped as patch 0212.
  - `scripts/prepare-exlr8.sh` and `scripts/exlr8/`: each rank's pieces are converted on its own box (`box.sh`: fetch the rank's
    halves by range requests, `exlr8_pieces.py`, `exlr8_dense.py`, `check_pieces.py`, `exlr8_assemble.py`, `.x8-complete` marker,
    every byte re-hashed), ~125 GB a box, idempotent; a gr8 TP4 up1 cache already on the box is read in place. The converters are
    interim dense and pieces writers used for the checkpoints already on spark1-4. `scripts/exlr8/hosts.env` carries the workers' home
    (`EXLR8_WORKER_HOME`, /home/zurih on this cluster); only `MODEL_FORMAT=exlr8` reads it.
  - Measured (image night-x8 = night-sr2 + patch 0205's binary; 1800 MHz cap): against K4 on the 72-window panel, full-vocabulary KL
    0.0630 (K4 0.0694), top-20 KL 0.0553 (0.0612), top-1 agreement 93.42% (93.20%); speed equal to K4 (sparkDash prose / code
    24.5 / 32.5 against 25.3 / 31.1 tok/s, prefill 921 / 934 against 914 / 922 at 9K / 37K, decode at 8K context 41.3 tok/s); the
    unsplit window 229,376 passed the needle at 206,438 tokens (K4: 212,992 and 191,692). This run used 0205 on the night image.
    The recipe's own image build of the amended 0205 then passed (image 4ace70f225b4 with 0206-0209) and TP4 interim booted from it at the defaults
    (see the README's up1 section for what that run measured). Codex review of the recipe edits found, and this fixes, a RANK_EXTRA_ENV tokenization
    mismatch (a tab- or newline-separated explicit `TF_GLM_PF_<name>=0` was not seen as an override, and the forwarding loop expanded globs) and an
    unvalidated explicit PF value.
  - Switching back: `MODEL_FORMAT=k4 ./start-tp6.sh` (stop first; README "Switching back to K4").

- **2026-11-17** -- Patch 0195 and `SLOT_RESUME=1` (default): a later prompt on the same slot resumes where the
  slot's own prompt rows still match it (the longest common prefix, cut back to the 4,096-row chunk grid, DFlash2's
  2,048-row window short of the end), instead of prefilling from row 0; nothing copied, no memory set aside, the
  reused rows reported as `cached_tokens`. K4 at TP4 (1800 MHz cap): a 12-turn agent loop resumed on 11 of 11 turns,
  time to first token 3-7 s against 33-53 s; a 49,556-token prompt 54.7 s cold, 6.3 s resumed; generated token ids
  identical cold, resumed and cold again. `SLOT_RESUME=0` turns it off.

- **2026-11-16** -- K4 at TP4 is the release configuration: engine patches 0001-0194 and, at four Sparks
  (`TP6_RANKS=4`), `CONTEXT=212992`, `SPLIT_QKVA=1` and `DSA_INDEX_FAST=1` by default (TP6 unchanged: `CONTEXT=auto`,
  both off), with 8-bit KV (int8 codes) and DFlash2. Decode with DFlash2 ~39 / 39 / 40 tok/s (code / tool / think),
  sparkDash prose ~25.5 and code ~31, no drafts ~18, prefill ~915 / 920 at 9K / 37K; parity 54 of 54, needle at
  191,692, 30-minute soak with 0 errors. All measured at the 1800 MHz GPU clock cap.

- **2026-11-16** -- `SPLIT_QKVA=1` is the TP4 default (patch 0149): with it off, the replicated [q_a | kv_a]
  projection decodes 0.95% slower over 36 paired code / tool / think replies in one boot, bit-identical. TP6 keeps 0.

- **2026-11-16** -- Patch 0192, `DSA_INDEX_FAST=1` (TP4 default): the decode indexer's query RoPE once a row and its
  top-2048 on several CTAs (patch 0067's scan/merge scheme on per-token keys). Bit-identical picks; decode +2.2% /
  +2.6% (prose / code) at ~60K tokens of context and +13.0% / +14.5% at ~190K.

- **2026-11-16** -- Patches 0187-0189 and the opt-in `KV_SPLIT=1` (`TF_GLM_KV_SPLIT=1`, prefetch off): each token's
  latent rows on one rank, prompt-cache states kept per rank. Exact; 540,672 tokens at TP4 (unsplit 212,992), or
  245,760 with a 1.4 GiB prompt cache (cached turn TTFT 28.7 -> 4.3 s); decode and prefill ~10% slower. `prepare.sh`
  refuses it with `KV_PAGES` on, the MTP drafter or `LEARN_GIB` > 0.

- **2026-11-16** -- Patch 0186 fixes int8 KV prompt-cache states: the kept state's plan and disk image counted one
  FP32 scale a row where int8 rows have two, so a restored state lost the scales from row at/2 on. Found in Codex's
  analysis report (section 6.3); a poison-and-restore host test covers int8 and E4M3, arena
  and pages.

- **2026-11-16** -- Patch 0190 fixes `ScatterRoom`: since patch 0141 a prompt chunk of 128k + 1 rows gave its
  second half one row more than its exchange span, which stopped the server on about 1 prompt in 256 at TP4
  (2,049, 3,841 and 28,417 tokens among them).

- **2026-11-16** -- Patch 0191, per-request experiment policy (`TF_GLM_POLICY_REQUESTS=1`, a request's
  `"tf_policy"` mask, named on the done line; off by default), the tool the switches above were measured with in one
  boot. Patches 0193-0194 run a prompt chunk's [q_a | kv_a] and indexer queries on their row owners
  (`TF_GLM_PROMPT_OWNER_PROJ`, policy bit 4); off by default, measured at no gain (-0.07% prefill over 14 pairs).

- **2026-11-16** -- **DFlash2 is the default drafter** (`DRAFTER=dflash2`; `DRAFTER=dspark` selects DSpark, now
  opt-in). The owner's rule was that DFlash2 replaces DSpark only if it is faster on code, tool, think and sampled
  prose with thinking off and on; it is. Gate on the recipe image 3cde334f49ea (patches 0001-0185), K4 TP4, KV `fp8`
  with int8 codes, context 65,536, GPU clocks capped at 1,800 MHz, DSpark at cut 0.35; 12-prompt greedy set at 512
  tokens, prose 6 prompts x 1,024 tokens sampled (temperature 1.0, top_p 0.95, seed 1234), thinking off, and on with
  `THINKING=1`. DFlash2 against DSpark, tok/s: code 39.0 / 36.2 (1.075x), tool 39.3 / 34.1 (1.153x), think 39.6 /
  39.2 (1.011x), prose thinking off 26.3 / 26.2 (1.002x), prose thinking on 27.5 / 27.3 (1.008x); weighted 1.093x.
  Reply hashes identical with drafts on and off for both drafters; prefill unchanged. Launcher only: the image is
  unchanged.

- **2026-11-16** -- Owner decision 2026-11-16: 8-bit KV is the default. `KV=fp8` with the new `KV8_CODES=int8`
  (default) serves the DSA latent rows as int8 codes with two power-of-two scales a row (engine patch 0164;
  `scripts/tp6.sh` passes `TF_GLM_KV8_INT8=1` to every rank). int8 codes were chosen over E4M3 for their lower row
  error (RMS 0.96% against 2.67%); `KV8_CODES=e4m3` selects E4M3 (patch 0162) and `KV=bf16` opts out. Measured
  2026-11-15 before patches 0175-0185: twice the BF16 window at TP4 with E4M3 codes (262,144 against 131,072 tokens
  without drafts; 98,304 against 65,536 with MTP; int8 rows are 4 bytes larger), prefill with int8 codes 899 / 894
  against 849 / 829 tok/s at 9K / 37K. The earlier agent-panel KL gate results (int8 +0.0082, E4M3 +0.0375 nats
  against the +0.005 gate) were measured before the SwiGLU fix (patch 0183). Launcher only: the image is unchanged.

- **2026-11-16** -- `DRAFTER=dflash2`: Inco AI's DFlash2 for full GLM-5.3
  ([incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) @ 8f02251d, Apache-2.0), opt-in. Engine patch
  0184 runs it on glm_moe_dsa (the DFlash2 drafter's shape from its config: D 6144, 6 layers, 64 query heads, taps after
  layers 5/19/33/47/61/75; its width-dependent kernels as templates, the GLM-5.3-Flash shape bit-identical); 0185 lets
  its chain take the tool-list tokens of 0181. The recipe pins and downloads the original weights (the HF cache's
  snapshot when present); ranks quantize them in device memory, no converted copy is written. `WORKER_DFLASH2_LOCAL=1`
  reads a worker's own copy instead of NFS. K4 TP4 on one image (0001-0182 + the DFlash2 engine), 12-prompt greedy set at
  512 tokens: code 37.0 / tool 40.8 / think 42.2 tok/s against DSpark's 34.2 / 35.3 / 40.4 (weighted 1.10x), reply hashes
  identical, full parity 0 of 54, needle 58,982 passes, prefill unchanged.
- **2026-11-16** -- Quality fix for full GLM-5.3: engine patch 0183. Our glm_moe_dsa port (patch 0122) read
  `swiglu_limit` only when `config.json` has it, and otherwise kept the engine's default of 10, which is
  GLM-5.3-Flash's value. GLM-5.3's config has no such field and its MLPs have no clamp (transformers'
  `GlmMoeDsaMLP`), but every dense, shared and routed expert here clamped its gate at 10 and its up projection to
  [-10, 10]. That cut the massive activations of the attention-sink rows in half (layer 9's shared expert on
  `[gMASK]<sop>`: gate 19, up 17), and every later token reads those rows: distributions came out flatter than
  BF16's (mean entropy 0.77 against 0.67), the second prompt row was broken (KL 4 nats), and chat NLL fell below
  BF16's. Upstream TensorFold requires the field (GLM-5.3-Flash only), so upstream is unaffected. With 0183 a
  glm_moe_dsa config without the field gets no clamp; GLM-5.3-Flash is unchanged; `RANK_EXTRA_ENV=
  TF_GLM_SWIGLU_LIMIT=10` restores the old clamp for an A/B. K4 at TP4 on the 72-window panel against BF16:
  full-vocabulary KL 0.126 -> 0.069, top-20 KL (eval suite v1.1, masked) 0.112 -> 0.063 (EXLR8 up1 0.059),
  top-1 90.5 -> 93.1%, confident flips 1.02 -> 0.47%, chat NLL 1.395 -> 1.473 (BF16 1.463). Prefill and decode
  paths agree before and after (KL 0.002), so the fault was never prefill-only. Gates on K4 at TP4 (same stage, old
  vs fixed image): draft parity 0 of 54, needle at 58,982 tokens passes, prefill 928 / 936 -> 926 / 933 tok/s (8K /
  32K), sparkDash prose 24.3 -> 24.3 and code 31.4 -> 32.6 tok/s. Every build served by this recipe (NVFP4, K4, the
  other EXL3 quants) changes output.
- **2026-11-15** -- K4 decode and memory at TP4: engine patches 0181-0182. 0181: DSpark drafts what a request's tool
  list forces inside a tool call (a name's rest once one listed name fits, its keys, the tags between); tool 30.8 ->
  35.0 tok/s, tool sampled 31.2 -> 35.7 (2.44 -> 2.93 tokens a round), code 36.4 -> 36.1, think 40.3 -> 40.0, copy
  unchanged; draft parity 0 of 54. `TF_GLM_TOOL_DRAFTS=0`: off. 0182: the embedding table as column slices across
  the ranks, rows gathered (same values; greedy and drafted replies identical): 105.08 -> 103.75 GiB a rank at TP4;
  prefill 986 / 1012 tok/s, needle at 58,982 tokens passes. `TF_GLM_EMBED_SHARD=0`: whole table.

- **2026-11-15** -- Launcher text: the memory guard's stop line starts with a UTC timestamp, as in Mia's AI Lab's
  [GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104) recipe (its
  `start.sh` memory guard), so a stop can be matched to the request that caused it. `[I_MEMGUARD]`, the LIVE banner
  and `stop-tp6.sh`'s last line give the real number of Sparks (a `TP6_RANKS=4` run said "six"): seen live as
  "Watching the 4 Sparks that serve" and "LIVE! ... (4 Sparks, TP=4)". Messages only; serving is unchanged.

- **2026-11-15** -- Launcher: every remote call is bounded, so a dead worker can no longer hang a start or stop
  (a Spark that ran out of memory during the weight load left `ssh ... docker logs` hanging for 38 minutes).
  `scripts/tp6.sh` runs each one-shot ssh/docker call on a worker under `timeout` (`REMOTE_TIMEOUT`, default the
  docker stop grace + 75 = 120 s) and the image copy under `SHIP_TIMEOUT` (3600 s), with ssh keepalives
  (`ServerAliveInterval=15`, `ServerAliveCountMax=4`). While the ranks load, a worker that stops answering ends the
  start with `E_WORKER` (the other ranks are stopped first). `stop-tp6.sh` skips a worker that does not answer,
  stops and logs the rest, then reports it with `E_WORKER`. Models, defaults and serving are unchanged.
  A running server is reused only for the same request: every rank's container carries the settings it was
  started with (`tf.request`: image, window, KV and `KV8_CODES`, drafter and its folder, `RANK_EXTRA_ENV`, ...) and
  the image ID; when a start asks for anything else, it says what differs (`W_RUNNING`), stops that server and starts
  fresh (a boot asking for DFlash2 had been answered by a server left over from another stage). `stop-tp6.sh` checks
  afterwards that no container of this recipe is left on any Spark and exits with `E_STOP` if one is.
  Image tags: a start never moves or rebuilds a tag that already exists. `IMAGE=` naming an existing image (an older
  recipe image, or a `night-*` test image) boots that image as it is with `W_IMAGE`; the default tag existing with
  another label fails with `E_IMAGE`. Only a missing tag is filled, from an image with the current label (none
  tagged `night-*`) or by a build.

- **2026-11-15** -- 0.3.1: 8-bit KV caches for glm_moe_dsa (opt-in). Engine patches 0162 (`KV=fp8`: the DSA latent
  rows as E4M3 codes and one power-of-two scale a row, the format of the owner's 3x recipe patch 0038; 580 B a token
  and layer instead of 1,152), 0163 (the msa2 prompt kernels over the E4M3 cache), 0164 (`TF_GLM_KV8_INT8=1`: int8
  codes, separate scales for the latent and the rope plane, 584 B). With K4 at TP4, `KV=fp8` serves 262,144 tokens
  without drafts (BF16: 131,072) and 98,304 with MTP (BF16: 65,536); TP6 holds 262,144 with 15 GiB to spare. Prefill
  +7-10%, decode within 3%, drafting exact (0 of 54 replies differ). Quality against BF16 KV on the 8K panel: E4M3
  code +0.0038 / agent +0.0375 nats, INT8 code +0.0006 / agent +0.0082; the gate (+0.005 on both) keeps them opt-in
  and `KV=bf16` the default. Recipe: MTP options are passed only with `DRAFTER=mtp`; `stop-tp6.sh` counts all six
  Sparks' containers; `scripts/tests/draft_parity.py` (drafted vs `"draft": false` replies, one boot).
  **DSpark is the default drafter** (`DRAFTER=dspark`, the owner's decision): engine patches 0165-0173 port
  RedHatAI/GLM-5.3-speculator.dspark to the Zig engine on DFlash2's drafter (DSA taps after layers 1/19/38/57/74,
  3 Qwen3 layers at 4 bits, 64 candidates a slot, rank 0's Markov + confidence chain, a chance-product cut at 0.35,
  GLM's fixed tool-call tag successors drafted); the MTP layer is no longer loaded. The pass went from ~13 to ~5 ms
  a round (staged window attention, fc and down at 4 bits). Exact (0 of 54 replies differ, greedy, seeded sampling
  and copy-heavy). TP4 against no drafts (19.5 tok/s): code 37.5 (1.92x), think 41.0 (2.12x), tool 32.0 (1.62x),
  copy 58-63 (3.0-3.2x); against MTP code 1.04x, think 1.10x, tool 0.83x. Tools: `dspark_traj.py` (`TF_GLM_TAP_DUMP` trajectories),
  `dspark_twin.py` (NumPy reference), `g3.py` (predicted speed), `dspark_cmp.py` (live drafts vs the twin).

- **2026-11-15** -- K4 prefill at TP4: engine patches 0175-0178. 0177: the two-halves prompt path runs an EXL3
  layer's routed experts once over the chunk, not once a half (each expert's trellis read once; same bits, greedy 24
  of 24 identical): 856 / 844 -> 875 / 869 tok/s at 9K / 37K without drafts. 0176: DSpark's taps inside the
  two-halves path (tapped chunks ran whole): 802 / 800 -> 867 / 867 with DSpark, draft parity 0 of 54, decode
  unchanged. 0175 + 0178: EXL3 prompt chunks write bf16 slot rows by default (`TF_GLM_EXL3_BF16_SLOTS=0`: fp32):
  -> 937 / 928 tok/s; 8K agent+code panel KL vs BF16 -0.0018 [-0.0044, +0.0008] nats against fp32 slots.
  0179: msa2 (prompt sparse attention) reads its latent keys coalesced, same bits: with DSpark 926 / 925 ->
  997 / 1011 tok/s (greedy replies identical, draft parity 0 differ).

- **2026-11-14** -- 0.3.0: prefill past the old TP4 recipe, deterministic long prompts. Prefill 966 / 953 / 903 ->
  1,224 / 1,200 / 1,125 tok/s at 10k / 38k / 150k; decode 33.3 / 43.2 tok/s prose / code on sparkDash's prompts.
  Engine patches 0142 (index keys only on the indexer layers + MTP; no `ig` / `pk` for DSA), 0143-0144 (MTP prompt
  pass writes caches only), 0145 (msa2 sparse attention on 576-wide BF16 rows, prompts), 0146 (`--mtp-confidence`
  stops an MTP chain early, cap 4; `MTP_CONFIDENCE=0.5` at TP6), 0147 (o_proj L2 prefetch during attention), 0148
  (decode attention in 128-key chunks), 0149 (`SPLIT_QKVA=1`: decode [q_a | kv_a] split across ranks; off, neutral),
  0150-0151 (the indexer's query rope: read before write, and the second half to element 32 + i -- a base-engine
  bug sent it two heads down, so the index picks past 2,048 tokens were wrong and differed run to run; now two runs
  are bit-identical, NLL 0.934 -> 0.920 on a 6.6k-token code text), 0152 (the NVFP4 decode gate-up prefetches 8
  K groups ahead into L2; bit-exact; decode 30.3 / 41.5 -> 31.5 / 42.3), 0153 (`TF_GLM_MTP_CAP`: the MTP cap at
  run time, 1..6, default 4; recipe `MTP_CAP`). Recipe: `MESH_FAST=1` (the mesh gather's fast path, +1.1%, bit-exact); `HEAD=q8` at TP6 (KL 0.0003-0.0008
  on code, prose and zh / ja / ru, NLL unchanged; decode -> 33.2 / 42.9; `HEAD=bf16` opts out); `RANK0_NSYS=1` profiles rank 0 under
  nsys; `scripts/tests/nll.py --save` and `kl.py` compare two runs' top-20 distributions.
- **2026-11-14** -- 0.2.0: speed. Prefill 497 / 505 -> 966 / 953 tok/s at 10k / 38k (903 at 150k), decode
  25.4 / 34.6 -> 28.3 / 39.6 tok/s prose / code. Engine patches 0137 (NVFP4 experts in 32-row units: 352 x 4 /
  320 x 2), 0138 (prompt exchanges by reduce-scatter + bf16 all-gather), 0139 (prompt index selection split across
  ranks), 0140 (`TF_GLM_ROCE_MESH`: decode all-gathers by fabric/tp's multirank RoCE gather; the image carries
  `tp_mailbox.fatbin`), 0141 (prompt chunks in two halves, exchanges on a side stream and a second communicator).
  Recipe: `ROCE_MESH=1`, `L2PF=1`, `MEMORY_RESERVE_GIB=8.5` at TP6 (a 122k prefill took spark1 under the memory
  guard's floor at 6.5), `scripts/tests/bench.py`.
- **2026-11-13** -- 0.1.1: long-context retrieval works. Engine patch 0134: under `TF_GLM_ABSORB_MMA=1`
  the tensor-core absorb placed each head at the Flash strides (q 192, out 512) instead of the DSA's
  (256, 576), so every prompt of 64+ tokens ran with scrambled queries (the old needle failures, and
  replies that called the prompt "garbled"); one MMA launch a head at the real strides. 0135: the BF16
  latent cache is sized at its 576-wide rows (it was 512, and the last ninth of a full window wrote into
  `ik`). 0136: `TF_GLM_DSA_DUMP` probes a select call's top scores and pick spans (`TF_GLM_DSA_WATCH`).
  Start: `CONTEXT_MAX` defaults to 262144 at TP6 (1M and 512K were OOM-killed after the ~5 min load on
  every new patch hash, then retried), an OOM-killed rank also halves the window on retry, the workers
  keep their NFS weights volume mounted between starts (`<volume>-pin`) so the digest cache's st_dev
  stays put (spark5 re-hashed 432 GiB every start), the plan lists all six ranks, the LIVE line names the
  served model, and `start-tp6.sh`'s help text is readable again. `scripts/tests/needle.py` talks to the
  chat API (`--raw` keeps the old untemplated completion).
- **2026-11-13** -- 0.1.0: GLM-5.3-NVFP4 (`nvidia/GLM-5.3-NVFP4`, ModelOpt MIXED_PRECISION -- NVFP4
  experts + per-tensor FP8 dense MLPs) on six DGX Sparks over a switch at TP=6, adapted from the
  GLM-5.3-Flash EXL3 recipe. Weights download once into spark1's default HF cache and reach every
  worker over read-only NFS Docker volumes (`WORKER_WEIGHTS=nfs` is the default); `MODEL_DIR` resolves
  the HF-cache snapshot at `refs/main`. DRAFTER=mtp uses the checkpoint's integrated layer-78 head.
  Engine patches 0122-0124 (over the Flash stack's 0121) add the `glm_moe_dsa` backbone graph, the DSA
  rope table, `model.layers.*` prefix detection, per-tensor FP8 dequantization and `modelopt-mixed`
  /`modelopt-nvfp4` registry entries. TP6 lands on legal shard boundaries by construction (MLA heads
  11/11/11/11/10/10, expert rows 384/384/320/320/320/320); it is not an upstream-advertised width, and
  a live six-rank decode has not yet run. DSA piecewise (shared-window) prompts are refused; KV-shard
  stays TP2-only; the TP2 pair scripts were removed.

### 2026-11-18: EXLR8 native dense, prompt chunks on int8 tiles (patches 0212, 0213; `EXLR8_DENSE=native`, opt-in)

- Patch **0212** keeps q_b, o_proj, the layer 0-2 MLP and the shared expert as kindlingai's EXL3 E4M3 trellis as stored; decode rows use the trellis
  kernel. Its first run (stage x12-lite) served a 245,760-token window (model 108.38 GiB against 109.06) with a better decode (prose 26.5 against
  24.4 tok/s, code 33.4 against 32.3) but a 21% slower prefill (810 / 839 against 1,022 / 1,041 tok/s): each 4,096-row chunk runs as two 2,048-row
  halves and the rebuild cache held one projection, so all 307 projections were rebuilt twice a chunk (about 874 ms of rebuilds plus a slower bf16 GEMM).
- Patch **0213** rebuilds projections into the interim dense lane's int8 weight tiles. X18 measured 122.1 versus 433.4 ms per set.
  Both prompt halves share the per-layer rebuild. The rebuilt tiles matched the interim tiles in 169 byte comparisons, and all
  11 kernel mutations failed their checks. The kernel takes BF16 activations and int8 weights, as traced in x13.
  `TF_GLM_X8_PROMPT_Q8=0` selects the BF16 rebuild. With q8 prompt mode on, short chunks also use q8 after the short-chunk fix.
  The measured image sent 1–16-row tails through trellis, so the prompt equality result covers the tested int8 path and panels.
- **x12-lite2** at 245,760 tokens logged 108.33 GiB native versus 109.06 GiB interim. Prefill was 989 / 1,011 tok/s versus
  1,022 / 1,041 (0.97x). Decode was 26.4 versus 24.4 tok/s for prose and 32.7 versus 32.3 for code. The needle passed.
  Code parity (4.17% versus 3.05%, limit +1.0 point) and the code decode floor (32.8 tok/s) missed their gates narrowly.
- Recipe image `249b937e54cb` built with patches 0210–0213 and booted at TP4. The TP4 interim save matched all 18 baseline reply hashes.
  **x19** found 96/96 prompt windows and 12/12 controls bit-identical on the int8 path. Native-minus-interim decode KL against BF16
  was +0.00028 (paired 95% interval [−0.00110, +0.00165]) across 12 code windows. HumanEval greedy pass@1 was 159/164 native versus
  160/164 interim. Native ran alone in a single stream with DFlash2. Interim ran on the restored production service, where other
  traffic could share decode rounds. x19 B1 logged 107.51 GiB of native model memory, about 1.6 GB below the 109.06 GiB interim
  comparison. EXLR8 up1 concurrency remains unmeasured end to end.
