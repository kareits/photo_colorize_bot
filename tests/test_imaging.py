"""Tests for the pure image operations.

These need no models and no network, which is the payoff for keeping imaging.py
free of both.
"""
import cv2
import numpy as np
import pytest
from PIL import Image

import imaging
from pipeline import Pipeline


def test_limit_long_side_shrinks_only_when_needed():
    big = np.zeros((1000, 2000, 3), dtype=np.uint8)
    assert imaging.limit_long_side(big, 1000).shape[:2] == (500, 1000)

    small = np.zeros((100, 200, 3), dtype=np.uint8)
    # Under the cap it must be left alone, not upscaled to meet it.
    assert imaging.limit_long_side(small, 1000).shape[:2] == (100, 200)


def test_desaturate_neutralises_the_tone_but_keeps_luminance():
    """Sepia in, neutral grey out — with the brightness structure intact.

    Losing the tone is the point; losing the luminance would destroy the photo,
    since luminance is the only thing the colouriser has to work from.
    """
    rng = np.random.default_rng(47)
    grey = rng.integers(40, 220, (70, 70)).astype(np.float32)
    sepia = cv2.merge([
        (grey * 0.80).astype(np.uint8),
        (grey * 0.95).astype(np.uint8),
        grey.astype(np.uint8),
    ])

    out = imaging.desaturate(sepia)

    # Neutral: the three channels now agree.
    assert np.abs(out[:, :, 0].astype(int) - out[:, :, 2].astype(int)).max() <= 2

    # Luminance survives: L before and after must track each other.
    before = cv2.cvtColor(sepia, cv2.COLOR_BGR2Lab)[:, :, 0].astype(float)
    after = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[:, :, 0].astype(float)
    assert np.abs(before - after).mean() < 2.0


def test_desaturate_flattens_a_colour_photo_too():
    """Colour in, neutral grey out — every input is neutralised, not just sepia."""
    rng = np.random.default_rng(43)
    photo = np.zeros((60, 60, 3), dtype=np.uint8)
    photo[:20] = [200, 40, 40]
    photo[20:40] = [40, 160, 40]
    photo[40:] = [40, 40, 190]
    photo = np.clip(photo.astype(int) + rng.integers(-15, 15, photo.shape), 0, 255).astype(np.uint8)

    out = imaging.desaturate(photo)
    assert np.abs(out[:, :, 0].astype(int) - out[:, :, 2].astype(int)).max() <= 2
    assert np.abs(out[:, :, 1].astype(int) - out[:, :, 2].astype(int)).max() <= 2


# --------------------------------------------------------------------------- #
# The heart of the refactor.
# --------------------------------------------------------------------------- #
def test_colorize_preserves_original_luminance_at_full_resolution():
    """Chroma comes from the model at 512; luminance must survive untouched.

    This is the property the whole redesign rests on. The old pipeline shrank the
    photo to 768 before colourising and then had Real-ESRGAN invent the detail
    back. Here the original L is carried through exactly, so a 2000px scan keeps
    its own grain and sharpness — only the low-frequency ab is upsampled.
    """
    rng = np.random.default_rng(11)
    # Detailed, high-frequency greyscale — the kind of texture a resize destroys.
    detail = rng.integers(0, 255, (800, 1200), dtype=np.uint8)
    img = np.repeat(detail[:, :, None], 3, axis=2)

    # Pretend the model returned some plausible chroma at its native 512x512.
    ab = rng.uniform(-40, 40, (1, 2, 512, 512)).astype(np.float32)

    out = imaging.colorize_output(img, ab)

    assert out.shape == img.shape, "result must keep the original resolution"

    original_l = cv2.cvtColor((img / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)[:, :, 0]
    result_l = cv2.cvtColor((out / 255.0).astype(np.float32), cv2.COLOR_BGR2Lab)[:, :, 0]

    # Lab L runs 0..100. Round-tripping through uint8 BGR costs a little, but the
    # luminance must not be meaningfully rebuilt or blurred.
    assert np.abs(original_l - result_l).mean() < 1.0


def test_colorize_input_hands_the_model_a_grey_image():
    """DDColor expects luminance carried as RGB, with chroma stripped."""
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)  # arbitrary colour

    tensor = imaging.colorize_input(img, size=512)

    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == np.float32
    # Chroma stripped => the three channels must agree at every pixel.
    r, g, b = tensor[0]
    assert np.abs(r - g).max() < 0.02
    assert np.abs(g - b).max() < 0.02


# --------------------------------------------------------------------------- #
# White balance.
# --------------------------------------------------------------------------- #
def test_white_balance_removes_a_colour_cast():
    rng = np.random.default_rng(3)
    neutral = rng.integers(60, 200, (100, 100, 3), dtype=np.uint8)

    cast = neutral.astype(np.float32)
    cast[:, :, 0] *= 1.4          # heavy blue cast
    cast = np.clip(cast, 0, 255).astype(np.uint8)

    fixed = imaging.white_balance(cast, strength=1.0)

    def imbalance(x):
        means = x.reshape(-1, 3).mean(axis=0)
        return means.max() - means.min()

    assert imbalance(fixed) < imbalance(cast)


def test_white_balance_at_zero_strength_changes_nothing(synthetic_gray):
    out = imaging.white_balance(synthetic_gray, strength=0.0)
    assert np.array_equal(out, synthetic_gray)


def test_white_balance_spares_a_genuinely_dominant_colour():
    """A sunset is orange on purpose; balancing must not launder that away.

    Grey-world (norm=1) assumes the average pixel is neutral and duly wrecks such
    images — which is why the old config carried a warning telling users to turn
    the strength down for them. A higher Minkowski norm is far gentler here, and
    this test pins that difference down.
    """
    rng = np.random.default_rng(9)
    sunset = np.stack([
        rng.integers(20, 60, (80, 80)),     # B low
        rng.integers(80, 130, (80, 80)),    # G mid
        rng.integers(180, 250, (80, 80)),   # R high — the actual sunset
    ], axis=-1).astype(np.uint8)

    def warmth(x):
        m = x.reshape(-1, 3).mean(axis=0)
        return m[2] - m[0]   # how much red exceeds blue

    grey_world = imaging.white_balance(sunset, strength=1.0, norm=1)
    shades_of_grey = imaging.white_balance(sunset, strength=1.0, norm=6)

    assert warmth(shades_of_grey) > warmth(grey_world), (
        "shades-of-grey should preserve more of the real colour than grey-world"
    )


# --------------------------------------------------------------------------- #
# Faces.
# --------------------------------------------------------------------------- #
def test_align_and_paste_round_trip_leaves_the_image_alone():
    """Aligning a face and pasting it straight back must be near-identity.

    If the affine transform and its inverse do not agree, faces would land skewed
    or offset — and on a portrait that is the most visible failure there is.
    """
    # A smooth image, not noise: warping resamples with bilinear interpolation, and
    # white noise cannot survive a resample-and-resample-back by construction. Any
    # error left on a gradient is the transform's fault, which is what we are testing.
    yy, xx = np.mgrid[0:600, 0:800].astype(np.float32)
    gradient = (xx / 800 * 180 + yy / 600 * 60).astype(np.uint8)
    img = cv2.merge([gradient, (gradient * 0.7).astype(np.uint8), (gradient * 0.4).astype(np.uint8)])

    # Landmarks roughly where a face would sit, in the template's own proportions.
    landmarks = np.array([
        [300.0, 250.0],   # left eye
        [400.0, 250.0],   # right eye
        [350.0, 310.0],   # nose
        [310.0, 360.0],   # left mouth corner
        [390.0, 360.0],   # right mouth corner
    ], dtype=np.float32)

    crop, matrix = imaging.align_face(img, landmarks)
    assert crop.shape == (512, 512, 3)

    out = imaging.paste_face(img, crop, matrix)
    assert out.shape == img.shape

    # Compare only where the mask actually blends, i.e. the face's neighbourhood.
    region = (slice(240, 380), slice(280, 420))
    assert np.abs(out[region].astype(int) - img[region].astype(int)).mean() < 12


def test_face_restore_tensor_round_trip():
    rng = np.random.default_rng(23)
    crop = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)

    tensor = imaging.face_restore_input(crop)
    assert tensor.shape == (1, 3, 512, 512)
    assert -1.01 <= tensor.min() and tensor.max() <= 1.01, "model expects [-1, 1]"

    back = imaging.face_restore_output(tensor)
    assert np.abs(back.astype(int) - crop.astype(int)).max() <= 1


# --------------------------------------------------------------------------- #
# Tiling.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("height", "width", "tile"),
    [
        (150, 230, 64),     # every tile comes out even
        (883, 720, 256),    # 883 = 3*256 + 115 -> an odd edge tile
        (131, 271, 128),    # odd on both axes, and an odd total
        (1, 1, 64),         # degenerate
    ],
)
def test_upscale_tiled_handles_odd_tile_sizes(height, width, tile):
    """Edge tiles are usually odd-sized, and the model cannot accept those.

    Real-ESRGAN x2 opens with a pixel_unshuffle, which requires even dimensions. An
    883px side leaves a 115px remainder, so the edge tile is odd and inference fails
    with a reshape error — which is exactly how this broke on the first real photo
    anyone sent. The earlier test only passed because 150x230 happens to divide
    evenly at every tile size tried.
    """
    rng = np.random.default_rng(31)
    img = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

    seen: list[tuple[int, int]] = []

    def double_but_demand_even(tensor: np.ndarray) -> np.ndarray:
        h, w = tensor.shape[2], tensor.shape[3]
        seen.append((h, w))
        assert h % 2 == 0 and w % 2 == 0, f"model was handed an odd tile: {h}x{w}"
        return np.repeat(np.repeat(tensor, 2, axis=2), 2, axis=3)

    out = imaging.upscale_tiled(img, run=double_but_demand_even, tile=tile, scale=2)

    assert out.shape == (height * 2, width * 2, 3)
    assert seen, "no tiles were processed"


@pytest.mark.parametrize("tile", [64, 128, 256])
def test_upscale_tiled_is_seamless(tile):
    """With a model that merely doubles the image, tiling must be invisible.

    Substituting a known-exact 'upscaler' isolates the tiling itself: any residue
    is a seam, an off-by-one in the padding trim, or a misplaced tile — not the
    network. Seams are exactly the artefact tiling tends to introduce.
    """
    rng = np.random.default_rng(29)
    img = rng.integers(0, 255, (150, 230, 3), dtype=np.uint8)

    def exact_double(tensor: np.ndarray) -> np.ndarray:
        # (1, 3, h, w) float RGB in [0,1] -> the same, doubled by pixel repetition.
        return np.repeat(np.repeat(tensor, 2, axis=2), 2, axis=3)

    out = imaging.upscale_tiled(img, run=exact_double, tile=tile, scale=2)
    expected = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)

    assert out.shape == expected.shape
    # Allow 1 unit for the uint8 <-> float round trip, but no seams.
    assert np.abs(out.astype(int) - expected.astype(int)).max() <= 1


# --------------------------------------------------------------------------- #
# HEIC.
# --------------------------------------------------------------------------- #
def test_pipeline_loads_heic(tmp_path):
    """HEIC is what iPhones shoot by default and OpenCV cannot read it.

    _load must fall through to Pillow (with pillow-heif registered in imaging). If
    the opener were not registered, or the fallback missing, this raises.
    """
    rgb = np.zeros((120, 80, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.linspace(20, 200, 80, dtype=np.uint8)[None, :]
    heic = tmp_path / "photo.heic"
    Image.fromarray(rgb).save(heic, format="HEIF")

    # OpenCV alone cannot — this is the reason the fallback exists.
    assert cv2.imread(str(heic)) is None

    img = Pipeline._load(heic)
    assert img.shape == (120, 80, 3)


def test_load_raises_on_a_non_image(tmp_path):
    junk = tmp_path / "notimage.png"
    junk.write_bytes(b"definitely not an image")
    with pytest.raises(ValueError):
        Pipeline._load(junk)
