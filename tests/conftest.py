"""Shared fixtures.

Two distinct needs, worth keeping apart:

* Verifying the ONNX export (phase 1) does not care what the image depicts — it
  compares torch against onnxruntime numerically, so a deterministic synthetic
  input is enough. No network, no downloads.
* Face detection and quality review need real photographs. Those are dropped by
  hand into tests/fixtures/photos/ (gitignored — other people's photos do not
  belong in the repo). When the directory is empty such tests skip rather than
  fail.
"""
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PHOTOS_DIR = FIXTURES_DIR / "photos"

# Formats the bot accepts at all.
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@pytest.fixture
def synthetic_gray() -> np.ndarray:
    """A deterministic 'black-and-white photo' as BGR uint8: smooth ramp plus grain.

    The ramp supplies low frequencies and the grain high ones, so this image
    reveals both a colour cast and a loss of detail. That is all the numerical
    torch-vs-onnxruntime comparison needs.
    """
    rng = np.random.default_rng(1234)
    h, w = 480, 640

    ramp = np.linspace(0, 255, w, dtype=np.float32)
    img = np.repeat(ramp[None, :], h, axis=0)
    img += rng.normal(0.0, 8.0, size=(h, w)).astype(np.float32)

    gray = np.clip(img, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)  # grey, but carried in 3 BGR channels


@pytest.fixture
def real_photos() -> list[Path]:
    """Real photos from tests/fixtures/photos/. Skips the test if there are none."""
    if not PHOTOS_DIR.is_dir():
        pytest.skip(f"no photo fixtures: drop some images into {PHOTOS_DIR}")

    photos = sorted(p for p in PHOTOS_DIR.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not photos:
        pytest.skip(f"no photo fixtures: drop some images into {PHOTOS_DIR}")
    return photos
