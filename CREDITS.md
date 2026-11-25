# Credits

This recipe is by [Mia's AI Lab](https://huggingface.co/Mia-AiLab) ([MiaAI-Lab](https://github.com/MiaAI-Lab)). It builds on the
work below. Licence terms and required notices are in [NOTICE.md](NOTICE.md); this repository's own licence is in [LICENSE](LICENSE).

## The engine

**[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and its contributors: the inference engine, native Zig/CUDA
server, EXL3 kernels, speculative verification and API. This repository does not contain the engine. Its build fetches TensorFold at
commit `f350cd3e` and applies `patches/engine/` to it. TensorFold's own
[README](https://github.com/ashhart/TensorFold#readme) and
[third-party notices](https://github.com/ashhart/TensorFold/blob/main/THIRD_PARTY_NOTICES.md) list its contributors and sources.
The native EXL3 experts that several patches here change build on [@justincasey](https://github.com/justincasey)'s
[#837](https://github.com/ashhart/TensorFold/pull/837).

## Patches by other authors

These are by people other than Mia's AI Lab, most of them upstream TensorFold pull requests, carried with their authors' names
in each patch's `From:` line. Their patches remain theirs.

| Patches | Author | Upstream PR | What |
|---|---|---|---|
| 0013 | Adam Hillson ([@BadAd84](https://github.com/BadAd84)) | [#1623](https://github.com/ashhart/TensorFold/pull/1623) | DFlash2 reads complete prompt taps at TP>1 |
| 0026-0027 | [@ajensenwaud](https://github.com/ajensenwaud) | [#1505](https://github.com/ashhart/TensorFold/pull/1505) | Load-ahead in the grouped EXL3 experts kernel |
| 0028-0029 | Adam Hillson ([@BadAd84](https://github.com/BadAd84)) | [#1534](https://github.com/ashhart/TensorFold/pull/1534) | EXL3 decode reads the trellis through `ld.global.nc.L1::no_allocate` |
| 0049-0056 | [@Bizuayeu](https://github.com/Bizuayeu) | [#1423](https://github.com/ashhart/TensorFold/pull/1423) | Paired prompt chunks |
| 0109-0110 | Cameron Basham | [#1823](https://github.com/ashhart/TensorFold/pull/1823) (0110 ports it) | The DFlash2 drafter's MLP split across every rank, partials summed in rank order (0109); `TF_GLM_CHUNK_SOLO`, a prompt step beside a decoding stream takes part of a chunk, a port of upstream's adaptive prompt chunk (0110) |
| 0113 | Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)) | [#1411](https://github.com/ashhart/TensorFold/pull/1411) | GLM conversation states on disk, every rank's part (`--learn-conversations`), backported to this stack by its author |
| 0114 | Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)) | [#1431](https://github.com/ashhart/TensorFold/issues/1431) | KV sharding at TP=2: each stream's latent rows on its owner rank, with 0113's disk states under it |
| 0120-0121 | Peter Jurik ([@wagzee](https://github.com/wagzee)) | | `TF_EXTRA_EOS`: named tokens end a reply like the checkpoint's EOS (0120); `TF_THINK_OFF_SPLIT`: a thinking-off reply's text before its close token is reasoning (0121) |
| 0124 | Devin (Cognition's coding agent) | | `glm_moe_dsa`'s backbone graph, rope table and NVIDIA NVFP4/FP8 load |
| 0235 | [@devopsfortunato](https://github.com/devopsfortunato) | [#2276](https://github.com/ashhart/TensorFold/pull/2276) | A resumed turn that has not started is ranked by its uncached tail, not as a whole cold prompt (fixes [#2256](https://github.com/ashhart/TensorFold/issues/2256)); backported to `f350cd3e` by [@laidick](https://github.com/laidick) in [GLM-5.3-Flash-DGX-Sparks-1104#20](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/pull/20) |
| 0236 | Robert Wojciechowski ([@wojo](https://github.com/wojo), X [@rwojo](https://x.com/rwojo)) | [#1411](https://github.com/ashhart/TensorFold/pull/1411) | `--learn-conversations` stages the longest state whole instead of skipping states past the 1 GiB default permit (`StagingFull`); authored by him for [GLM-5.3-Flash-DGX-Sparks-1104#49](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/pull/49) |
| 0237 | Ash Hart ([@ashhart](https://github.com/ashhart)) | [#2442](https://github.com/ashhart/TensorFold/pull/2442), commit [12d61cb6](https://github.com/ashhart/TensorFold/commit/12d61cb6b96c968aab76b8c53444380e66a76f53) | A refused disk read of a learned state tries one shorter disk state before a cold prefill; backported to `f350cd3e` by [@wangtianwen07](https://github.com/wangtianwen07) in [GLM-5.3-Flash-DGX-Sparks-1104#41](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/pull/41) |

Patches 0057-0063 are Mia's AI Lab's extension of #1423 to DFlash2.

## The port to 2x DGX Spark

This recipe is the lab's [2x RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold) moved to GB10. The engine, the first 93 patches and the serving shape are
that recipe's (0103 and 0104 were ported from it later, by the lab, 0103 with a guard); 0094-0102 are this pair's own; the two-host launch, the ConnectX-7 link discovery and the unified-memory accounting are adapted from Mia's
AI Lab's **[GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)**
(branch `zig-engine`): `scripts/nodes.sh`'s `link_info`, `rails`, `rail_info`, `rail_ping` and `cx7_peer`, and the
memory rule in `scripts/memavail.sh`. The hard-won parts of that code are its: that the GID index must be the RoCE v2
entry holding the node's own IPv4 address, and that a Spark's single QSFP port reaches the GB10 over two PCIe links and
so needs both of its RDMA devices given to NCCL.

The engine's own RoCE all-gather, which this recipe turns on for the pair, is the GPU side of b12x's "RoCEnante"
exchange ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), `b12x/comm/roce`, Apache
License 2.0), carried upstream in TensorFold as `zig/src/cuda/roce.zig`.

## Work this recipe adapts

- **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** by Z.ai (MIT): `templates/glm-5.3-flash-vision.jinja` is the
  checkpoint's own chat template with its media branch changed.
- **[DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold)**
  pull requests: patch 0092 adapts [@SvangenStudios](https://github.com/SvangenStudios)'s
  [#31](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/31) and patch 0093 part of their
  [#28](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/28), carried with their name in each
  patch's `From:` line. `stop.sh`'s warning about open requests and `start.sh`'s stricter smoke test follow their
  [#24](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/24), and `start.sh`'s log follower closing
  the lock's descriptor follows [@tfolkman](https://github.com/tfolkman)'s
  [#22](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/22).

## The six-Spark (TP6) work

- **The memory guard's timestamped stop line** (`scripts/tp6.sh`) is taken from Mia's AI Lab's
  [GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104) recipe, which the lab
  authored (its `start.sh` memory guard).
- **Patch 0145** ports Mia's AI Lab's own **msa2** sparse-attention kernel, from the lab's
  [GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold) recipe
  (its patches 0150, 0158 and 0162; Apache-2.0), to this engine's BF16 576-wide latent rows.
- **Patch 0140** turns TensorFold's own `zig/src/fabric/tp` multirank RoCE gather, by Ash Hart
  ([@ashhart](https://github.com/ashhart)), on for the GLM family's decode all-gathers.
- **Patches 0146, 0147, 0149 and 0152** follow the mechanisms authored by knapcio ([@knapcio](https://github.com/knapcio))
  in his [GLM-5.3-gr8-6x-DGX-Spark-TP6](https://github.com/knapcio/GLM-5.3-gr8-6x-DGX-Spark-TP6) recipe (MIT):
  - stopping an MTP chain at a confidence threshold, the 0.5 threshold (`MTP_CONFIDENCE`), and the six-Spark sweep of
    fixed draft depths that set the cap at 4 (0146);
  - prefetching the out-projection into L2 through the attention (0147), after his gr8 patch 0009, which names
    blaesus/dsgx10 as the origin of the paced prefetch;
  - splitting the replicated [q_a | kv_a] projection across the six ranks in decode (0149), after his gr8 PR
    [#84](https://github.com/kindlingai/gr8/pull/84).
  - prefetching the routed experts' weights ahead of the decode expert kernel's loads (0152), after his gr8 patch
    0010's translation prefetch at the start of each expert's fc1 (`G6_K1_TLBW`); 0152 prefetches along each warp's
    stream into L2 instead.
  No code was taken from that recipe; the patches are written against this engine.
- **Patches 0162 and 0163** (`KV=fp8` on glm_moe_dsa's 576-wide latent rows) port the FP8 cache format that Mia's AI
  Lab authored in its own [GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)
  recipe (its patch 0038, `0038-glm-kv-fp8`, `kv8.py`: E4M3 codes and one power-of-two scale a row; Apache-2.0) to
  this engine's chunk and msa2 kernels.
- **Patch 0164** (`TF_GLM_KV8_INT8=1`) is written for this recipe on 0162's layout.
- **Patches 0165-0173** (`DRAFTER=dspark`, the default drafter 2026-11-15 to 2026-11-16, now opt-in) port the DSpark drafter to this engine's Zig code:
  - the Python DSpark port that Mia's AI Lab authored in its
    [GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold) recipe
    (its patches 0114, 0116, 0117, 0123 and 0125: `dspark.py`'s model, chain, depth rule and dump) is the source of
    the port's math, and `scripts/tests/dspark_twin.py` is written from it;
  - the DSpark model definition (`MarkovHead`, `ConfidenceHead`, the anchored-block forward and mask) was authored by
    [vllm-project/speculators](https://github.com/vllm-project/speculators) (commit 36a19ca, Apache 2.0) and the
    sequential top-k Markov sampling by [vLLM](https://github.com/vllm-project/vllm) (commit 5f30fc7, Apache 2.0),
    from which that Python port adapted them (its `dspark.py` header lists the files);
  - the engine side is the Zig DFlash2 drafter authored by Ash Hart ([@ashhart](https://github.com/ashhart)) and the
    TensorFold contributors (`families/glm/dflash*.zig`, `glm_dflash.cu`, whose top-16 kernels the 64-candidate ones
    copy), after Z Lab's DFlash architecture ([z-lab/dflash](https://github.com/z-lab/dflash), MIT), as TensorFold's
    notices say.
- **Patches 0175-0178** (K4 prefill) are written for this recipe. They change the EXL3 prompt experts that
  TensorFold's `exl3_prompt.cuh` / `exl3_experts.cu` carry, which their headers credit to @BobClawblaw's design
  ([#976](https://github.com/ashhart/TensorFold/pull/976)) and to the transposed prompt kernels Mia's AI Lab authored
  in its [GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)
  recipe (its patches 0004 and 0009); 0176 and 0177 change the two-halves prompt path of patch 0141.
- **Patch 0179** changes how msa2 loads its keys. msa2, the prompt sparse-attention kernel of patch 0145, was
  authored by Mia's AI Lab in its GLM-5.3 recipe (its patches 0150, 0158 and 0162; Apache-2.0), as the kernel's
  header says; the kernel's math and its bits are unchanged.
- **Patches 0181 and 0182** are written for this recipe. 0181 extends the DSpark draft chain of patches 0165-0173
  (and the tool-call tag successors of patch 0173) and reads the tool list on rank 0 with TensorFold's tokenizer and
  per-request copy proposer, which Ash Hart ([@ashhart](https://github.com/ashhart)) and the TensorFold contributors
  authored. 0182 gathers the embedding rows through TensorFold's decode mesh and NCCL routes and reuses the
  `tf_glm_unblock` kernel of patch 0149.
- **Patch 0183** is written for this recipe. It corrects this recipe's own patch 0122 (the glm_moe_dsa config),
  which kept GLM-5.3-Flash's `swiglu_limit` for full GLM-5.3. The SwiGLU kernels it configures (`swiglu1` in
  `glm_glue_moe.cuh`, the clamp in the EXL3 and NVFP4 expert kernels) and the `swiglu_limit` setting are TensorFold's,
  by Ash Hart ([@ashhart](https://github.com/ashhart)) and the TensorFold contributors; upstream TensorFold reads the
  field from the config and is not affected. The diagnosis compared against the BF16 model run through Hugging Face
  transformers' `glm_moe_dsa` modelling code, and decoded the K4 experts with an EXL3 reader derived from TensorFold's
  own kernels.
- **Patches 0184 and 0185** (`DRAFTER=dflash2`) run Inco AI's DFlash2 drafter for full GLM-5.3,
  [`incoai/GLM-5.3-DFlash2`](https://huggingface.co/incoai/GLM-5.3-DFlash2/tree/8f02251dc71a9da2053dfc15bfa293fbee5579b0)
  (revision 8f02251d, Apache-2.0), on glm_moe_dsa. Inco AI authored the drafter and its weights; the DFlash
  architecture it extends was authored by Z Lab ([z-lab/dflash](https://github.com/z-lab/dflash), MIT). The engine side
  extends the Zig DFlash2 drafter authored by Ash Hart ([@ashhart](https://github.com/ashhart)) and the TensorFold
  contributors (`families/glm/dflash*.zig`, `glm_dflash.cu`): 0184 reads the drafter's shape from its config (D 6144,
  6 layers, 64 query heads, taps after layers 5/19/33/47/61/75 of the one residual stream) and turns its width-dependent
  kernels into templates; 0185 lets its chain take the tool-list tokens of patch 0181. The weights are read unmodified
  and quantized in device memory at load; no converted copy is written or shipped.
- **Patches 0186-0194** (2026-11-16, the TP4 release) are written for this recipe on TensorFold's glm_moe_dsa and
  DFlash2 code by Ash Hart ([@ashhart](https://github.com/ashhart)) and the TensorFold contributors:
  - **Codex's analysis report** (written by Codex, OpenAI's coding
    agent, for this recipe) found the int8 prompt-cache bug that patch 0186 fixes (its section 6.3), and proposed the
    decode indexer's RoPE hoist and multi-CTA top-2048 of patch 0192 (section 4.4), the per-request experiment policy
    of patch 0191 ("one immutable request policy") and the prompt owner rows of patches 0193-0194 (section 5.1).
  - **Patch 0192**'s multi-CTA top-2048 adapts the scan/merge scheme of patch 0067 (`tf_glm_dsel_scan` /
    `tf_glm_dsel_merge`), which Mia's AI Lab ([MiaAI-Lab](https://github.com/MiaAI-Lab)) wrote for this stack, as
    0192's commit message says.
  - **Patch 0191**'s bit 5 switches the [q_a | kv_a] split of patch 0149 to the split authored by knapcio in gr8 PR
    [#84](https://github.com/kindlingai/gr8/pull/84) (above).
  - Patches 0187-0189 (the KV split) and 0190 (the ScatterRoom fix to patch 0141's two-halves prompt path) are
    written for this recipe.
- **Patches 0210** (`TF_GLM_SEQ_GROW`, one pool of KV granules for every stream) **and 0211** (`TF_GLM_DSA_SHARED`, several glm_moe_dsa
  streams in one decode window) are new code for TensorFold's CUDA GLM path, written by Mia's AI Lab
  ([MiaAI-Lab](https://github.com/MiaAI-Lab)). The pooled-KV scheme of several sessions sharing decode steps was
  authored by kindlingai's gr8 ([kindlingai/gr8](https://github.com/kindlingai/gr8), MIT; authors Matt Mastracci ([@kindlingai](https://github.com/kindlingai)),
  adapt-ai-systems and the gr8 contributors); no gr8 code is used.
- **Patch 0195** (`SLOT_RESUME`, slot resume for agent loops) is new code for TensorFold's CUDA GLM path, written by
  Mia's AI Lab ([MiaAI-Lab](https://github.com/MiaAI-Lab)). The mechanism follows the Metal slot resume for
  Qwen3.8-27B (`zig/src/native/qwen27_resume.zig`, `qwen27_slots.zig`), authored by Ash Hart
  ([@ashhart](https://github.com/ashhart)) in commit
  [1407ad28](https://github.com/ashhart/TensorFold/commit/1407ad28bec9742e5e4d69e016b1c7ca7bf9af30) with the TensorFold
  contributors, after the proposal in issue [#1197](https://github.com/ashhart/TensorFold/issues/1197) by RS91Dev
  ([@RS91Dev](https://github.com/RS91Dev)).
- The TP6 plan was reviewed by Codex and by Claude Mythos 5.2.

## Ideas and formats

- **EXLR8** (format exlr8-v2), its MIT-licensed reference loader, and both releases of the GLM-5.3 EXLR8 checkpoint
  were authored by **kindlingai** (Matt Mastracci). The recipe can convert and serve either release that a user downloads
  (`up1` at commit `a36ae468`, or v0.7, profile `bit400`, through `EXLR8_RELEASE=v07`); it bundles neither and serves
  neither by default.
  [@kindlingai](https://github.com/kindlingai) · [Reference loader](https://github.com/kindlingai/exlr8) · [up1](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/tree/a36ae4680871ffa6412061d5daf66c4d4ad5802c) · [v0.7](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/tree/9e25c28a2e8808d8b2599d780521fd6af24e7558)
  Patch 0205 and `scripts/exlr8/` are written for this recipe by Mia's AI Lab ([MiaAI-Lab](https://github.com/MiaAI-Lab)).
  They read kindlingai's format
  (their tile layout, sharding rule and index files) and repack it into TensorFold's EXL3 rank pieces. Patch 0212 and
  `scripts/exlr8/exlr8_native.py` (also by Mia's AI Lab) keep kindlingai's trellis matrices (q_b, o_proj, the dense MLP and the shared
  expert) as kindlingai stored them, tile for tile, and decode them with TensorFold's EXL3 linear kernel on the E4M3 values. The E4M3 tile decode patch 0205
  adds to the EXL3 kernels follows `kernels/cuda/exlr8.cu` and `core/exlr8_*`, TensorFold's own EXLR8 code by Ash Hart
  ([@ashhart](https://github.com/ashhart)) and its contributors, against which the converter's output is checked bit for bit.
  Patch 0214 (also by Mia's AI Lab) decodes v0.7's experts on kindlingai's `mul1_e4m3` scheme, checked bit for bit against a host
  reference that follows kindlingai's reference loader.
- **[kindlingai/glm-5.3-exlr8-k4-4x-spark](https://github.com/kindlingai/glm-5.3-exlr8-k4-4x-spark)** (MIT) was authored by
  kindlingai (Matt Mastracci): their four-Spark engine and settings for the same checkpoint, the one this recipe's EXLR8 results were
  compared against. No code of it is used here.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp (MIT): the EXL3 format, and an experts-kernel technique that
  patch 0026 cites.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: PR #22522's results on this card class were the evidence for trying programmatic
  dependent launches. No code was taken.

## Models (not in this repository)

- **[Mia's AI Lab](https://huggingface.co/Mia-AiLab)**: authored
  [`Mia-AiLab/GLM-5.3-DERISKED-Cyber-EXL3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-DERISKED-Cyber-EXL3-4bpw), the EXL3
  4 bpw checkpoint of the DERISKED lane (`MODEL_FORMAT=derisked`), a quantization of Z.ai's GLM-5.3.
- **kindlingai** (Matt Mastracci) authored both releases of `kindlingai/glm-5.3-exlr8-k2-k3-k4` used here.
  CC BY 4.0 applies only to the packing, manifests and error tables. The weights remain under the GLM-5.3 license.
  The EXLR8 format and reference loader are separate works. The three ExLlamaV3-derived primitives retain their MIT terms.
  Users download and convert the files per rank; the recipe bundles neither release. [kindlingai](https://huggingface.co/kindlingai) · [EXLR8 model repository](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) · [EXLR8 model card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/blob/main/README.md#licence) · [Conversion changes](NOTICE.md)
- **[wrldsuksgo2mars](https://huggingface.co/wrldsuksgo2mars)** (Hugging Face account): authored the K4 EXL3 4-bit quantization of GLM-5.3,
  cached locally as [`wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1-b`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1-b) until
  2026-11-25, and as `wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1` at revision `47af2334` until that was removed from Hugging Face
  the same day. Since then the recipe serves its K4-MC release,
  [`wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC) at revision
  [`e28931c9`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC/commit/e28931c918fa89403b1cd247f89c7dc25eeb67ac) (its `original`
  view, served as `GLM-5.3-EXL3-K4`).
  The checkpoint is under the GLM-5.3 license. It was this recipe's default 2026-11-16 to 2026-11-17 and is again since 2026-11-25
  (`MODEL_FORMAT=k4`). [Model card](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1) · [Revision `47af2334`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1/commit/47af23347db743b4666d952e2eb48f2b01c3fede)
- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3)**: authors of GLM-5.3 (`zai-org/GLM-5.3`), the base model of the K4 and EXLR8 quantizations
  (the glm-5.3 licence, as the K4 model card states).
- **[Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash)**: GLM-5.3-Flash, its architecture and base weights (MIT).
- **[Mia's AI Lab](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)**: the EXL3 checkpoints, including the gated
  [Ablit weights](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit).
- **[Red Hat AI](https://huggingface.co/RedHatAI)**: authors of the DSpark speculator
  [`RedHatAI/GLM-5.3-speculator.dspark`](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) (glm-5.3 licence, as
  its model card states), `DRAFTER=dspark` (opt-in). Downloaded by the user, never redistributed here.
- **[Inco AI](https://huggingface.co/incoai/GLM-5.3-DFlash2)**: authors of the DFlash2 drafter for full GLM-5.3,
  [`incoai/GLM-5.3-DFlash2`](https://huggingface.co/incoai/GLM-5.3-DFlash2/tree/8f02251dc71a9da2053dfc15bfa293fbee5579b0)
  at revision 8f02251d (Apache-2.0, as its model card states), `DRAFTER=dflash2`, the default drafter. Downloaded by the user at that
  revision, never redistributed here.
- **[Inco AI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)**: DFlash2 drafter weights, **CC BY-NC-ND 4.0, for research and
  evaluation**. Downloaded by the user, never redistributed here.

## Build and runtime

[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) and its container images, [Zig](https://ziglang.org),
[xgrammar](https://github.com/mlc-ai/xgrammar), [libjpeg-turbo](https://libjpeg-turbo.org) and
[libwebp](https://developers.google.com/speed/webp), under their own licences.

## How it was made

Most of Mia's AI Lab's patches were written by AI coding agents under the lab's direction, with reviews by Codex, DeepSeek
and GLM-5.3-Flash.
