# NOTICE

GLM-5.3-Flash EXL3 on 2x DGX Spark with TensorFold's native engine
Copyright 2026 Mia's AI Lab (MiaAI-Lab). Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).

This recipe builds on third-party work that keeps its own licence. If you redistribute it or a modified version, keep this
notice and the notices below, and state what you changed (Apache-2.0, section 4). People and projects are credited in
[CREDITS.md](CREDITS.md).

## What is in this repository, and what is not

This repository holds a recipe: scripts, a Dockerfile, a chat template and patches. It does not contain TensorFold, model weights
or binaries. The LICENSE above does not cover third-party work this recipe downloads, or change that work's terms.

## TensorFold

https://github.com/ashhart/TensorFold, by Ash Hart and the TensorFold contributors.

`patches/engine/*.patch` modify TensorFold's source at commit `f350cd3ea253a042e205746928271f9d8f71a82d` and contain parts of it. The
Docker build fetches that commit, applies them, and copies TensorFold's `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/` into
the image.

TensorFold's NOTICE reads:

    TensorFold
    Copyright 2026 TensorFold contributors
    https://github.com/ashhart/TensorFold

    TensorFold is licensed under the Apache License, Version 2.0 (LICENSE) from 0.6.0.
    Releases up to 0.5.0 were MIT-licensed, and code written before 0.6.0 keeps its
    MIT notice in LICENSES/MIT.txt. Third-party code and its licenses are listed in
    THIRD_PARTY_NOTICES.md.

TensorFold's MIT notice, for its code written before 0.6.0:

    MIT License

    Copyright (c) 2026 TensorFold contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Patches by other authors

Patches 0013, 0026-0029 and 0049-0056 are upstream TensorFold pull requests, carried with their authors' names in each patch's `From:`
line (see [CREDITS.md](CREDITS.md)). Those names must stay on them. Some were rebased to apply on this stack.

## DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold

Pull requests #22, #24, #28 and #31 on that repository, by their authors (see [CREDITS.md](CREDITS.md)). Licensed under the Apache
License, Version 2.0.

**Changes:** patches 0092 and 0093 adapt its patches 0042 and 0039 to this repository's engine commit; the names in their `From:` lines
must stay on them. The checks in `start.sh` and `stop.sh` were rewritten for this recipe's scripts.

## GLM-5.3-Flash chat template

`templates/glm-5.3-flash-vision.jinja` is adapted from the GLM-5.3-Flash checkpoint's `chat_template.jinja`, under its MIT licence.

**Changes:** its media branch writes GLM's image and video markers instead of a note that it cannot see them.

    MIT License

    Copyright (c) 2026 Z.AI Co., Ltd

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## ExLlamaV3

This recipe's EXL3 code reads the EXL3 format of ExLlamaV3 (https://github.com/turboderp-org/exllamav3), MIT License,
Copyright (c) 2025 Turboderp. It contains no ExLlamaV3 code.

## kindlingai's EXLR8 (gr8 quants: supported, not bundled)

kindlingai (Matt Mastracci) authored the EXLR8 format (exlr8-v2), the MIT-licensed reference loader, and both
releases of `kindlingai/glm-5.3-exlr8-k2-k3-k4` used here. The recipe bundles neither and serves neither by default.
`MODEL_FORMAT=exlr8` serves a release the user prepared: commit `a36ae4680871ffa6412061d5daf66c4d4ad5802c` (profile `up1`,
the converter's default) or, with `EXLR8_RELEASE=v07`, tag v0.7 at commit `9e25c28a2e8808d8b2599d780521fd6af24e7558`
(profile `bit400`).
[@kindlingai](https://github.com/kindlingai) · [EXLR8 model repository](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4) · [Reference loader](https://github.com/kindlingai/exlr8)

CC BY 4.0 applies only to the packing, manifests and error tables. The weights remain under Z.ai's GLM-5.3 license.
The EXLR8 format and reference loader are separate works.
The three ExLlamaV3-derived primitives retain their MIT terms. [EXLR8 model card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4/blob/main/README.md#licence)

When the user runs `scripts/prepare-exlr8.sh`, it downloads the selected release per rank and converts it on each Spark. No checkpoint is
redistributed with this repository. The conversion repacks each rank's routed-expert and shared-expert tiles byte for
byte into EXL3 rank pieces and recomputes their scale vectors exactly in FP16. The interim converter rounds trellis
matrices, int4, up1's FP8 row-scaled vocabulary and the shared expert once to BF16. It retains kv_b in its stored form,
converts MXFP8 to BF16 exactly, and copies v0.7's BF16 vocabulary as stored. With `EXLR8_DENSE=native` (patch 0212),
the trellis and shared-expert tiles move whole into EXL3's layout with their values intact. The converter adds an EXL3
`quantization_config` to the release's config.json. The converter reads kindlingai's format and contains none of their code.
[`kindlingai/glm-5.3-exlr8-k4-4x-spark`](https://github.com/kindlingai/glm-5.3-exlr8-k4-4x-spark) (MIT), their four-Spark engine,
is where the settings this recipe compared against come from; no code of it is used here.

## Downloaded at build or first start, under their own terms

| What | Licence | Note |
|---|---|---|
| GLM-5.3-Flash EXL3 checkpoint, Mia's AI Lab, from Z.ai's GLM-5.3-Flash | Apache-2.0 for the quantization; Z.ai's MIT for the base model | Not in this repository |
| Ablit weights (`ABLIT=1`), Mia's AI Lab, from that checkpoint | MIT | Gated on Hugging Face; its terms apply. Not in this repository |
| DFlash2 drafter weights for full GLM-5.3 (`DRAFTER=dflash2`), Inco AI, `incoai/GLM-5.3-DFlash2` @ 8f02251d | Apache-2.0, as its model card states | Downloaded by the user at the pinned revision; read unmodified, no converted copy written; not redistributed here |
| **DFlash2 drafter weights for GLM-5.3-Flash, Inco AI (`incoai/GLM-5.3-Flash-DFlash2`)** | **CC BY-NC-ND 4.0** | **Research and evaluation only; commercial use needs a licence from Inco AI. Not redistributed here** |
| DSpark speculator weights (`DRAFTER=dspark`), Red Hat AI, `RedHatAI/GLM-5.3-speculator.dspark` @ b374b956 | The glm-5.3 licence, as its model card states | Downloaded by the user at the pinned revision; not redistributed here |
| K4 checkpoint (`MODEL_FORMAT=k4`), authored by [wrldsuksgo2mars](https://huggingface.co/wrldsuksgo2mars), a 4-bit EXL3 quantization of Z.ai's GLM-5.3. [Model card](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1) · [Revision `47af2334`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1/commit/47af23347db743b4666d952e2eb48f2b01c3fede) | GLM-5.3 license | Downloaded at the pinned revision by the start: [`wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC`](https://huggingface.co/wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC) @ e28931c9, only its `original` view's files; earlier `GLM-5.3-EXL3-K4-v1-b` and `-v1` @ 47af2334 (both removed from Hugging Face). Not redistributed here. |
| EXLR8 weights (`MODEL_FORMAT=exlr8`), kindlingai, `kindlingai/glm-5.3-exlr8-k2-k3-k4` @ a36ae468 (up1) or tag v0.7 @ 9e25c28a (bit400) | GLM-5.3 license for the weights. CC BY 4.0 only for packing, manifests and error tables. | Downloaded by the user and converted per rank on each Spark. See the EXLR8 section above for the conversion changes and other terms. |
| xgrammar v0.2.8 | Apache-2.0 | Fetched and built into the image |
| libjpeg-turbo 3.1.4.1 | Its own BSD-style and zlib terms | Fetched and built into the image |
| libwebp 1.6.0 | BSD-3-Clause | Fetched and built into the image |
| Zig 0.17.0 | MIT | Build tool |
| NVIDIA CUDA images | NVIDIA's own terms | Deep Learning Container License; see each image's `/NGC-DL-CONTAINER-LICENSE` |

The image built by this recipe contains third-party software. If you redistribute that image, its terms apply as well as this notice.
