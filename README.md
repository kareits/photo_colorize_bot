# Photo Colorize Bot

A Telegram bot that colourises old black-and-white photographs, restores faces, and
upscales the result. Runs on CPU. All models are open source and permissively
licensed, so the bot may be used commercially.

## Pipeline

```
original (full resolution)
  → desaturate      strip a sepia/toned cast, keep luminance
  → colourise       DDColor predicts Lab chroma at 512×512
  → white balance   shades-of-grey
  → restore faces   YuNet finds them, RestoreFormer++ repairs them
  → upscale         Real-ESRGAN ×2, only for small photos
```

Two design decisions are worth knowing about, because they are not obvious.

**Luminance is never resampled.** DDColor is an `L → ab` model: it predicts only the
two Lab chroma channels and never reproduces brightness. So the original
full-resolution luminance is kept exactly as scanned, and only the model's `ab` — which
is low-frequency and survives it — is upsampled onto it. A 2000 px scan keeps its own
grain and sharpness. (The previous version shrank every input to 768 px before
colourising, then leaned on Real-ESRGAN to hallucinate the lost detail back.)

**Faces are restored after colourising, not before.** Face restorers are trained on
colour faces. Feeding one a greyscale image puts it outside its training distribution,
which is what produces the glassy, over-contrasted eyes that generative restorers are
notorious for. Colourising first keeps the restorer in-distribution.

| Stage | Model | Licence |
|---|---|---|
| Colourise | DDColor | Apache-2.0 |
| Face detection | YuNet (bundled in OpenCV) | Apache-2.0 |
| Face restoration | RestoreFormer++ | Apache-2.0 |
| Upscale | Real-ESRGAN x2plus | BSD-3 |

CodeFormer and GPEN are better known but are licensed for non-commercial use only
(S-Lab 1.0), which rules them out here.

## Architecture

The models run as **ONNX**, exported ahead of time. The running bot never imports
torch — that boundary is the point of the design.

```
tools/export_onnx.py   the ONLY file allowed to import torch; build-time only
    ↓ produces
onnx/*.onnx
    ↓ consumed by
models.py     ONNX sessions: threads, lazy loading, eviction
imaging.py    pure ndarray → ndarray operations; no models, no I/O
pipeline.py   stage orchestration
bot.py        Telegram transport
```

Dropping torch took an entire pin cascade with it — `numpy<2`, `timm<1.0`,
`datasets<3`, `torchvision==0.16.2` and a Python 3.10 ceiling. That is what lets the
bot be installed next to another service without fighting it over versions.

`imaging.py` holds no models on purpose: the stages are pure functions, so they are
testable without weights and the pipeline's behaviour can be compared by swapping an
argument.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env            # then put your @BotFather token in it
python bot.py
```

This needs the exported models in `onnx/`. To produce them, see below.

## Exporting the models

Done once. It needs torch, which is why it lives in a separate environment — nothing
here is installed in production.

```bash
python -m venv .venv-export
.venv-export/Scripts/pip install -r requirements-export.txt

git clone --depth 1 https://github.com/piddnad/DDColor.git vendor_repos/DDColor
git clone --depth 1 https://github.com/wzhouxiff/RestoreFormerPlusPlus.git vendor_repos/RestoreFormerPlusPlus

# weights → weights/ : RealESRGAN_x2plus.pth, RestoreFormer++.ckpt,
#                      ddcolor_paper_tiny.pth, ddcolor_modelscope.pth
# detector → onnx/face_detection_yunet.onnx (from opencv_zoo; no export needed)

python -m tools.export_onnx --weights-dir weights --out-dir onnx --repos-dir vendor_repos
```

Every export is checked against the original torch model and is **discarded if it
does not reproduce it** within 0.1% of the output range — a silently wrong model
would be far worse than a failed build.

## Performance and memory

Measured on 2 CPU threads:

| Stage | Time | Peak RAM |
|---|---|---|
| Colourise (DDColor large) | ~6 s | ~1.0 GB |
| Colourise (DDColor tiny) | ~3 s | ~0.3 GB |
| Restore faces | ~6 s | ~0.4 GB |
| Upscale | ~27 s **per megapixel of output** | ~0.3 GB |

Two things follow, and both are load-bearing:

**`enable_cpu_mem_arena = False`** in `models.py`. onnxruntime's arena allocator
reserves memory far beyond what the graph needs and dominates peak usage — it cost
DDColor-large 2.3 GB instead of 1.0 GB. Without that one line the large model does
not fit on a small server at all.

**Upscaling is conditional.** At ~27 s per output megapixel, upscaling a large scan
would take minutes and blow through the job timeout, so it is skipped unless the photo
is small (`UPSCALE_TARGET_SIDE`, `UPSCALE_MAX_INPUT_PIXELS`). Since the pipeline no
longer discards the original resolution, most photos do not need it.

Models are loaded per stage and evicted after (`KEEP_MODELS_LOADED=false`), so peak
memory is the largest single model rather than the sum of all of them.

## Configuration

Everything in `config.py`, all overridable via environment (see `.env.example`), so a
container can be retuned without a rebuild. Notably `NUM_THREADS` — onnxruntime and
OpenCV would otherwise take every core and starve whatever else shares the machine.

## Tests

```bash
pytest
```

The suite covers `imaging.py` without needing models or a network. Tests that need real
photographs (face detection) skip unless you drop some into `tests/fixtures/photos/`.
