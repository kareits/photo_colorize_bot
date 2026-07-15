"""Tests that stage planning honours the user's settings.

_plan only decides which stages run; it does not invoke any model, so it can be
tested with no weights and a dummy Models. That is the seam that makes per-user
settings verifiable at all.
"""
import numpy as np
import pytest

from pipeline import Pipeline
from settings import PRESETS, Settings

SMALL_FILE = 10_000          # under UPSCALE_MAX_FILE_BYTES, so upscale is eligible
LARGE_FILE = 5_000_000       # over it, so upscale is refused regardless of preset


@pytest.fixture
def pipe():
    # _plan never touches self.models, so None is a fine stand-in.
    return Pipeline(models=None)


def stage_names(pipe, settings, file_bytes=SMALL_FILE, shape=(400, 300)):
    img = np.zeros((*shape, 3), dtype=np.uint8)
    return [name for name, _ in pipe._plan(img, file_bytes, settings)]


def test_desaturate_always_runs_first(pipe):
    # Every input is neutralised, colour or not — it must lead every plan.
    for preset in PRESETS.values():
        assert stage_names(pipe, preset)[0] == "desaturate"


def test_speed_preset_omits_upscale(pipe):
    assert "upscale" not in stage_names(pipe, PRESETS["speed"])


def test_quality_preset_includes_upscale_for_a_small_file(pipe):
    assert "upscale" in stage_names(pipe, PRESETS["quality"], file_bytes=SMALL_FILE)


def test_a_large_file_is_never_upscaled_even_on_quality(pipe):
    # The area/size guard overrides the preset — this is what protects the timeout.
    assert "upscale" not in stage_names(pipe, PRESETS["quality"], file_bytes=LARGE_FILE)


def test_face_strength_zero_drops_the_face_stage(pipe):
    off = Settings(upscale=False, white_balance=False, face_restore_strength=0.0)
    assert "faces" not in stage_names(pipe, off)

    on = Settings(upscale=False, white_balance=False, face_restore_strength=0.5)
    assert "faces" in stage_names(pipe, on)


def test_white_balance_toggles_with_the_setting(pipe):
    without = Settings(upscale=False, white_balance=False, face_restore_strength=0.5)
    assert "white_balance" not in stage_names(pipe, without)

    with_wb = Settings(upscale=False, white_balance=True, face_restore_strength=0.5)
    assert "white_balance" in stage_names(pipe, with_wb)


def test_faces_run_after_colorize_by_default(pipe):
    # The order fix: the restorer should see a colour face, not greyscale.
    names = stage_names(pipe, Settings(upscale=False, white_balance=False, face_restore_strength=0.5))
    assert names.index("faces") > names.index("colorize")
