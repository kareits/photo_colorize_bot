"""Pure image operations. No models, no I/O, no torch — just ndarray in, ndarray out.

Keeping these free of the models is what makes the phase-5 quality review cheap:
stage order, face model and colouriser variant can all be swapped by passing a
different callable, without touching this file.

Everything here speaks OpenCV's convention: BGR, uint8, HxWx3, unless a
docstring says otherwise.
"""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pillow_heif

# Teach Pillow to open HEIC/HEIF. This is what iPhones shoot by default, so users send
# it constantly — but neither Pillow nor OpenCV reads it without this. Registered here
# because every entry point imports this module, and it must happen before any file is
# opened. Idempotent, so importing this module twice is harmless.
pillow_heif.register_heif_opener()

# The 5-point template ArcFace/FFHQ-style aligners use for a 512x512 face crop:
# left eye, right eye, nose, left mouth corner, right mouth corner. RestoreFormer++
# (like GFPGAN and CodeFormer) was trained on crops aligned to exactly this layout,
# so the numbers are not ours to tune — they come with the model.
FACE_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

FACE_SIZE = 512


def limit_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the long side is at most max_side. Bounds peak memory.

    Note this is a *safety* cap, not a quality knob. The old pipeline set it to
    768 and fed that to the colouriser, which is what actually threw away the
    scan's detail — DDColor only ever sees 512 anyway, and the full-resolution
    luminance is what carries the sharpness.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def is_already_colour(
    img: np.ndarray,
    spread_threshold: float = 6.0,
    cast_threshold: float = 30.0,
) -> bool:
    """True if the image carries real colour, so colourising it would repaint it.

    Users do send colour photos to a colouriser bot, and repainting one is a bug.
    But the obvious test — "is there much colour?" — gets the important case
    backwards: a sepia print is heavily tinted (saturation ~50) yet is exactly the
    photo this bot exists for. Thresholding on the amount of colour would reject
    the old prints and accept the modern snapshots.

    Two signals together, because neither alone is enough:

    * **Spread** of Lab's a/b channels. A tone is one hue laid evenly across the
      frame, so a/b barely vary; a real photo has different hues in different
      places. This is what tells sepia apart from a landscape.
    * **Cast**, i.e. how far a/b sit from neutral. Spread alone would call a
      uniformly red image monochrome, since an even hue has no variation at all.
      Sepia's cast is mild; a saturated colour's is not.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a, b = lab[:, :, 1].astype(np.float32), lab[:, :, 2].astype(np.float32)

    spread = max(float(a.std()), float(b.std()))
    # OpenCV packs Lab into uint8 with 128 as neutral for a and b.
    cast = max(abs(float(a.mean()) - 128.0), abs(float(b.mean()) - 128.0))

    return spread > spread_threshold or cast > cast_threshold


def desaturate(img: np.ndarray) -> np.ndarray:
    """Strip an old print's tone, keeping its luminance. Sepia in, neutral grey out.

    Runs before anything else on a monochrome photo. The colouriser would ignore the
    tone anyway — it reads only Lab's L and predicts fresh chroma — but relying on
    that leaves the tone alive everywhere else in the pipeline: white balance would
    try to "correct" it, and the face restorer would get a sepia face, which is not
    a face it was ever trained on.

    Neutralising once, up front, means every later stage sees the same thing no
    matter what tone the print happened to carry, so the result no longer depends on
    how brown the scan was.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab[:, :, 1] = 128   # a -> neutral
    lab[:, :, 2] = 128   # b -> neutral (128 is neutral in OpenCV's uint8 Lab)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


# --------------------------------------------------------------------------- #
# Colourisation glue.
#
# DDColor is an L -> ab model: it takes luminance and predicts only the two Lab
# chroma channels, never luminance itself. So the original full-resolution L can
# be kept verbatim and only ab upsampled from the model's 512x512 output. Chroma
# is low-frequency and survives that; the scan's grain and sharpness are untouched.
#
# This mirrors DDColor's own ColorizationPipeline.process — it is how the model is
# meant to be used, not a trick of ours.
# --------------------------------------------------------------------------- #
def colorize_input(img: np.ndarray, size: int = 512) -> np.ndarray:
    """BGR uint8 -> the (1, 3, size, size) float32 RGB tensor DDColor expects."""
    img_f = (img / 255.0).astype(np.float32)
    resized = cv2.resize(img_f, (size, size))

    # Strip chroma via Lab, then hand the model a grey image carried as RGB.
    lightness = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    grey_lab = np.concatenate([lightness, np.zeros_like(lightness), np.zeros_like(lightness)], axis=-1)
    grey_rgb = cv2.cvtColor(grey_lab, cv2.COLOR_Lab2RGB)

    return grey_rgb.transpose(2, 0, 1)[None].astype(np.float32)


def colorize_output(img: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Attach the model's ab to the original image's full-resolution L.

    ab arrives as (1, 2, 512, 512) in Lab units; img is the untouched original.
    """
    h, w = img.shape[:2]
    img_f = (img / 255.0).astype(np.float32)
    lightness = cv2.cvtColor(img_f, cv2.COLOR_BGR2Lab)[:, :, :1]

    ab_hw = ab[0].transpose(1, 2, 0)                        # (512, 512, 2)
    ab_full = cv2.resize(ab_hw, (w, h), interpolation=cv2.INTER_LINEAR)

    lab = np.concatenate([lightness, ab_full], axis=-1)
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return (np.clip(bgr, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# White balance.
# --------------------------------------------------------------------------- #
def white_balance(img: np.ndarray, strength: float = 0.7, norm: int = 6) -> np.ndarray:
    """Shades-of-grey white balance (Minkowski norm), blended by strength.

    The old code used grey-world (norm=1), which assumes the scene's *mean* is
    neutral. That is false for any photo with an honestly dominant colour — a
    sunset, foliage, the sea — and it duly washed those out, treating real colour
    as a cast to be removed. A higher Minkowski norm weights bright pixels more
    and behaves far better on exactly those images; norm=6 is the usual choice.
    """
    if strength <= 0.0:
        return img

    f = img.astype(np.float32)
    powered = np.power(f.reshape(-1, 3), norm).mean(axis=0)
    illuminant = np.power(powered, 1.0 / norm)

    # Scale each channel toward the mean illuminant, i.e. toward neutral.
    scales = illuminant.mean() / np.clip(illuminant, 1e-6, None)
    corrected = f * scales

    out = f * (1.0 - strength) + corrected * strength
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Face alignment and paste-back.
#
# The restorer only understands a 512x512 crop aligned to FACE_TEMPLATE_512, so
# each face is warped into that frame, restored, and warped back.
# --------------------------------------------------------------------------- #
def align_face(img: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp one face into the 512x512 aligned frame.

    Returns the crop and the affine matrix used, which paste_face needs to invert.
    """
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), FACE_TEMPLATE_512, method=cv2.LMEDS
    )
    if matrix is None:
        raise ValueError("could not fit an affine transform to the landmarks")

    crop = cv2.warpAffine(
        img, matrix, (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    return crop, matrix


def _face_blend_mask(feather: int) -> np.ndarray:
    """A feathered oval over the 512 crop, used to blend a restored face back in.

    Not a rectangle. The crop is square but a face is not, and the restorer shifts
    the face's colour and contrast relative to its surroundings — so a square mask
    blends along the edge of the *crop*, leaving a faint but real rectangular seam in
    the background beside the face. An oval keeps the blend on facial skin, where the
    two images already agree, and the seam disappears. Confirmed by eye on a real
    photo; the square version left a visible border, the oval does not.
    """
    mask = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    centre = FACE_SIZE // 2
    # Axes chosen to cover the face generously while staying clear of the crop edge.
    cv2.ellipse(mask, (centre, centre), (185, 225), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)


def paste_face(img: np.ndarray, restored: np.ndarray, matrix: np.ndarray, feather: int = 41) -> np.ndarray:
    """Warp a restored crop back and blend it in under a feathered oval mask."""
    h, w = img.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)

    warped = cv2.warpAffine(restored, inverse, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(_face_blend_mask(feather), inverse, (w, h), flags=cv2.INTER_LINEAR)

    alpha = mask[:, :, None]
    blended = warped.astype(np.float32) * alpha + img.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Upscaling.
# --------------------------------------------------------------------------- #
def upscale_tiled(
    img: np.ndarray,
    run: Callable[[np.ndarray], np.ndarray],
    tile: int = 256,
    pad: int = 16,
    scale: int = 2,
) -> np.ndarray:
    """Run a super-resolution model tile by tile.

    Feeding a whole photo to Real-ESRGAN allocates activations proportional to
    its area, which on a 3.8 GB server is how you get OOM-killed. Tiling bounds
    that to one tile at a time.

    Tiles are cut with an overlap of `pad` and the padding is trimmed off the
    model's output, so no seam appears where neighbouring tiles meet.
    """
    h, w = img.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            # The tile itself, then the same tile grown by `pad` on every side.
            y0, y1 = y, min(y + tile, h)
            x0, x1 = x, min(x + tile, w)
            py0, py1 = max(y0 - pad, 0), min(y1 + pad, h)
            px0, px1 = max(x0 - pad, 0), min(x1 + pad, w)

            patch = img[py0:py1, px0:px1]

            # Real-ESRGAN x2 begins with a pixel_unshuffle, which can only fold an
            # image whose height and width are even. Edge tiles are whatever is left
            # over and are frequently odd — an 883px side leaves a 115px remainder —
            # and the model fails outright on those. Pad to even, then take the
            # padding back out of the output.
            odd_h, odd_w = patch.shape[0] % 2, patch.shape[1] % 2
            if odd_h or odd_w:
                patch = cv2.copyMakeBorder(patch, 0, odd_h, 0, odd_w, cv2.BORDER_REFLECT_101)

            tensor = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            result = run(tensor.transpose(2, 0, 1)[None])

            upscaled = result[0].transpose(1, 2, 0)
            upscaled = np.clip(upscaled, 0.0, 1.0) * 255.0
            upscaled = cv2.cvtColor(upscaled.round().astype(np.uint8), cv2.COLOR_RGB2BGR)

            if odd_h or odd_w:
                upscaled = upscaled[: upscaled.shape[0] - odd_h * scale,
                                    : upscaled.shape[1] - odd_w * scale]

            # Drop the padding we added, in output coordinates.
            trim_top = (y0 - py0) * scale
            trim_left = (x0 - px0) * scale
            keep_h = (y1 - y0) * scale
            keep_w = (x1 - x0) * scale

            out[y0 * scale : y1 * scale, x0 * scale : x1 * scale] = upscaled[
                trim_top : trim_top + keep_h, trim_left : trim_left + keep_w
            ]

    return out


def face_restore_input(crop: np.ndarray) -> np.ndarray:
    """512x512 BGR crop -> the (1, 3, 512, 512) float32 RGB tensor in [-1, 1]."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - 0.5) / 0.5
    return normalized.transpose(2, 0, 1)[None]


def face_restore_output(tensor: np.ndarray) -> np.ndarray:
    """(1, 3, 512, 512) float32 RGB in [-1, 1] -> 512x512 BGR uint8."""
    rgb = tensor[0].transpose(1, 2, 0)
    rgb = (np.clip(rgb, -1.0, 1.0) + 1.0) / 2.0
    bgr = cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr
