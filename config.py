"""Settings, read from the environment so compose can retune limits without a rebuild.

Defaults here are sized for the target server: 4 vCPU, 3.8 GB RAM, roughly 1 GB of
which the neighbouring TTS service already holds. They are deliberately
conservative — the bot sharing a box with a live service must not be the reason
that service gets OOM-killed.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# A real environment variable wins over .env, which is what lets compose override.
load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Telegram ---
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = Path(os.environ.get("TMP_DIR", BASE_DIR / "tmp"))
ONNX_DIR = Path(os.environ.get("ONNX_DIR", BASE_DIR / "onnx"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Resources ---
# onnxruntime and OpenCV both grab every core by default. On a box shared with a busy
# neighbour this must be capped below the core count; on this server the TTS service is
# effectively idle, so the bot is allowed all 4. Lower it via env if that changes.
NUM_THREADS = _int("NUM_THREADS", 4)

# False: load a model before its stage, evict it after. Peak memory becomes the
# largest single model instead of the sum of all three — which is the difference
# between fitting in the server's free ~2.3 GB and being OOM-killed. True is for
# roomy machines, where keeping models resident is simply faster.
KEEP_MODELS_LOADED = _bool("KEEP_MODELS_LOADED", False)

# --- Processing limits ---
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 20)

# A file-size limit does not bound memory: compression means a small file can expand
# into an enormous array (a "decompression bomb"). This caps the decoded pixel count,
# checked from the image header *before* anything is decoded.
MAX_INPUT_PIXELS = _int("MAX_INPUT_PIXELS", 50_000_000)

# Safety cap on the input's long side, not a quality knob: the colouriser works at
# 512 regardless, and full-resolution luminance is what carries the detail.
MAX_INPUT_SIDE = _int("MAX_INPUT_SIDE", 2400)
JOB_TIMEOUT_SEC = _int("JOB_TIMEOUT_SEC", 300)
MAX_QUEUE_SIZE = _int("MAX_QUEUE_SIZE", 20)
USER_COOLDOWN_SEC = _int("USER_COOLDOWN_SEC", 5)

# --- Stages ---
ENABLE_COLORIZE = _bool("ENABLE_COLORIZE", True)
ENABLE_FACE_RESTORE = _bool("ENABLE_FACE_RESTORE", True)
ENABLE_UPSCALE = _bool("ENABLE_UPSCALE", True)

# Off by default. Side-by-side on real photos it hurt as often as it helped: on a
# portrait it pushed the background blue and cooled the skin, which reads as *more*
# saturated by any metric while being less true. Where it does help — an old print
# with a green-olive cast — it is a per-photo judgement, not something to impose on
# everyone. Best exposed as a user-facing option later rather than defaulted on.
ENABLE_WHITE_BALANCE = _bool("ENABLE_WHITE_BALANCE", False)

# Which DDColor export to use: ddcolor_large.onnx | ddcolor_tiny.onnx
#
# 'large' is ddcolor_modelscope — the same weights the original bot pulled through
# modelscope, and the reason its colours were vivid. 'tiny' is a different, weaker
# model: markedly less saturated, and swapping it in was a visible regression.
#
# It fits: with the arena allocator disabled (see models.py) large peaks at ~1.0 GB,
# well inside the server's free memory. It costs ~3 s more per photo than tiny, which
# is nothing next to what upscaling costs.
COLORIZER_MODEL = os.environ.get("COLORIZER_MODEL", "ddcolor_large.onnx")

# Stage order. Faces after colourising means the restorer sees a *colour* face,
# which is the distribution it was trained on — the old pipeline fed it greyscale
# and got the glassy, over-contrasted eyes that led to the stage being switched
# off. Set true to restore the old order and compare.
FACE_RESTORE_BEFORE_COLORIZE = _bool("FACE_RESTORE_BEFORE_COLORIZE", False)

# 0.0 = original, 1.0 = full restoration. Kept low on purpose: RestoreFormer++, like
# every generative face restorer, redraws eyes that were already fine — at 0.7 the
# gaze visibly changes and skin goes plastic; at 0.3 it barely helps. 0.5 is the
# point where it sharpens a soft face without inventing a new one. It only decides
# how much of the model's output is blended over the original, so the safe failure
# mode is "looks like the original", not "looks wrong".
FACE_RESTORE_STRENGTH = _float("FACE_RESTORE_STRENGTH", 0.5)

# Cost is linear in the number of faces — each one is a separate 512x512 pass, about
# 5 s. A group photo triggered 20 detections and spent 103 s on this stage alone; on
# the server, with half the cores, that would run past JOB_TIMEOUT_SEC and pin the
# single worker. So restore the largest faces and leave the rest: the small ones in
# the back row gain little from restoration anyway, and are the likeliest to be false
# detections in the first place.
MAX_FACES = _int("MAX_FACES", 10)

# YuNet's confidence threshold. Raised from the 0.7 default because a busy old photo
# produces spurious detections, and every one of them costs 5 s and paints a
# hallucinated face onto the picture.
FACE_DETECT_THRESHOLD = _float("FACE_DETECT_THRESHOLD", 0.8)

# Shades-of-grey white balance: 1 is the old grey-world behaviour, higher norms
# weight bright pixels more and stop washing out honestly dominant colours.
WHITE_BALANCE_STRENGTH = _float("WHITE_BALANCE_STRENGTH", 0.6)
WHITE_BALANCE_NORM = _int("WHITE_BALANCE_NORM", 6)

# Upscaling is conditional. The old pipeline always upscaled because it had shrunk
# the input to 768 first and needed to claw the resolution back; keeping the original
# resolution means most photos do not need it at all.
#
# The trigger is file size: a small file means a small or heavily compressed photo,
# which is exactly what benefits from upscaling. A large file is already detailed and
# only gets slower.
UPSCALE_MAX_FILE_BYTES = _int("UPSCALE_MAX_FILE_BYTES", 1_000_000)

# And a hard safety net on top, because file size does not bound work: a 900 KB JPEG
# can decode to 3000x2000, whose upscale is 24 MP of output. At the measured ~27 s per
# output megapixel that is ten minutes — well past JOB_TIMEOUT_SEC, with the single
# worker pinned the whole time. This caps the stage to roughly half a minute.
UPSCALE_MAX_INPUT_PIXELS = _int("UPSCALE_MAX_INPUT_PIXELS", 1_200_000)

# Tile size bounds peak memory during upscaling but barely affects speed (128 px ->
# 226 MB, 384 px -> 502 MB, both ~26s). 256 is the middle of that trade.
UPSCALE_TILE = _int("UPSCALE_TILE", 256)

# Which upscaler: realesrgan_x2plus.onnx | realesrgan_compact_x4.onnx
#
# x2plus is the heavy one (23 residual blocks, 67 MB) and stays the default because
# it looks right: the compact model (5 MB) is 3.2x faster — 13 s against 42 s — but
# over-sharpens, turning hair into contrasty painted strokes. Note that a sharpness
# metric *rises* on that output, which is precisely why the choice was made by eye.
#
# The compact model is exported and ready, so switching is a config change if the
# server turns out to need the speed more than the fidelity.
UPSCALER_MODEL = os.environ.get("UPSCALER_MODEL", "realesrgan_x2plus.onnx")
UPSCALER_NATIVE_SCALE = _int("UPSCALER_NATIVE_SCALE", 2)  # compact is 4; x2plus is 2
