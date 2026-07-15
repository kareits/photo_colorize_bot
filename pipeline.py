"""Stage orchestration. Decides what runs, in what order, and on what.

Pure orchestration: the pixel work lives in imaging.py and the sessions in
models.py. No torch anywhere in this process — see tools/export_onnx.py for the
one place it is allowed.

The order below is the substance of the refactor:

    original (full resolution)
      -> colourise    DDColor predicts ab at 512; the original full-resolution L
                      is kept verbatim and only ab is upsampled onto it
      -> white balance
      -> restore faces  after colourising, so the restorer sees a colour face
      -> upscale        only if the result is still small

The old pipeline shrank the input to 768, restored faces on a *greyscale* image,
and then leaned on Real-ESRGAN to invent back the detail it had just discarded.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
import imaging
from models import Models

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, models: Models):
        self.models = models

    def process(self, in_path: Path, out_path: Path, on_stage=None, deadline: float | None = None) -> Path:
        """Run the full pipeline. Blocking and CPU-heavy — call it off the event loop.

        on_stage(name, index, total) is called before each stage so the bot can show
        progress; a minute of silence feels broken.

        deadline is a time.monotonic() value past which we give up. It is checked here,
        between stages, rather than left to the caller — asyncio.wait_for around a
        thread-pool job does *not* stop the thread, so a runaway image would keep
        burning CPU and holding the only worker long after the user was told it failed.
        Cooperative checks are the only way to actually stop.
        """
        img = self._load(in_path)
        img = imaging.limit_long_side(img, config.MAX_INPUT_SIDE)

        stages = self._plan(img, in_path.stat().st_size)
        for index, (name, run) in enumerate(stages, start=1):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"ran out of time before stage '{name}'")
            if on_stage:
                on_stage(name, index, len(stages))

            started = time.monotonic()
            before = img.shape[:2]
            img = run(img)
            # Free this stage's model before the next one loads its own, so peak
            # memory is the largest single model rather than the sum of all three.
            self.models.end_of_stage()
            logger.info(
                "stage %s: %sx%s -> %sx%s in %.1fs",
                name, before[1], before[0], img.shape[1], img.shape[0], time.monotonic() - started,
            )

        if not cv2.imwrite(str(out_path), img):
            raise RuntimeError(f"could not write the result to {out_path}")
        return out_path

    def _plan(self, img: np.ndarray, file_bytes: int) -> list[tuple[str, object]]:
        """Decide which stages actually apply to *this* image."""
        stages: list[tuple[str, object]] = []

        # Neutralise every input to greyscale first, colour or not. This drops the old
        # "is this already colour?" check, which rested on a brittle threshold — it
        # once mistook a sepia print for a colour photo and refused to colourise it.
        # A colour photo now gets recoloured by DDColor rather than kept, which is the
        # honest behaviour for a colourising bot and removes a whole class of misfires.
        stages.append(("desaturate", lambda i: imaging.desaturate(i)))

        if config.ENABLE_FACE_RESTORE and config.FACE_RESTORE_BEFORE_COLORIZE:
            stages.append(("faces", self._restore_faces))
        if config.ENABLE_COLORIZE:
            stages.append(("colorize", self._colorize))
        if config.ENABLE_WHITE_BALANCE:
            stages.append(("white_balance", self._white_balance))
        if config.ENABLE_FACE_RESTORE and not config.FACE_RESTORE_BEFORE_COLORIZE:
            stages.append(("faces", self._restore_faces))
        if config.ENABLE_UPSCALE and self._worth_upscaling(img, file_bytes):
            stages.append(("upscale", self._upscale))
        return stages

    @staticmethod
    def _worth_upscaling(img: np.ndarray, file_bytes: int) -> bool:
        """Upscale small files only, and never when it would take minutes.

        File size is the signal: a small file means a small or heavily compressed
        photo, which is what upscaling actually helps. A large one is already detailed.

        The area check is a safety net rather than a nicety — file size does not bound
        the work. At ~27 s per output megapixel, a compact-but-large scan would run
        past the job timeout with the only worker pinned.
        """
        h, w = img.shape[:2]

        if file_bytes >= config.UPSCALE_MAX_FILE_BYTES:
            logger.info("skipping upscale: %.1f MB file is already detailed", file_bytes / 1e6)
            return False

        if h * w > config.UPSCALE_MAX_INPUT_PIXELS:
            logger.info("skipping upscale: %dx%d would take too long", w, h)
            return False

        return True

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        # IMREAD_COLOR gives 3-channel BGR and honours EXIF orientation, so phone
        # photos do not come out rotated.
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img

        # OpenCV cannot read HEIC/HEIF — the format iPhones shoot by default, so this
        # is a common case rather than an exotic one. Pillow can, via pillow-heif
        # (registered in imaging), but it does not apply EXIF rotation on its own, so
        # do that explicitly or portrait photos arrive sideways.
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"could not read the image: {path}") from exc

    # ---------------------------------------------------------------- stages --
    def _colorize(self, img: np.ndarray) -> np.ndarray:
        tensor = imaging.colorize_input(img)
        ab = self.models.colorizer.run(tensor)
        return imaging.colorize_output(img, ab)

    def _white_balance(self, img: np.ndarray) -> np.ndarray:
        return imaging.white_balance(
            img, strength=config.WHITE_BALANCE_STRENGTH, norm=config.WHITE_BALANCE_NORM
        )

    def _restore_faces(self, img: np.ndarray) -> np.ndarray:
        faces = self.models.face_detector.detect(img, max_faces=config.MAX_FACES)
        if not faces:
            logger.info("no faces found; leaving the image alone")
            return img

        strength = config.FACE_RESTORE_STRENGTH
        out = img
        for landmarks in faces:
            try:
                crop, matrix = imaging.align_face(out, landmarks)
            except ValueError:
                # A face too distorted to align is a face we leave untouched,
                # rather than pasting a mangled crop over the photo.
                logger.warning("could not align a face; skipping it")
                continue

            restored = imaging.face_restore_output(
                self.models.face_restorer.run(imaging.face_restore_input(crop))
            )
            if strength < 1.0:
                restored = cv2.addWeighted(restored, strength, crop, 1.0 - strength, 0.0)

            out = imaging.paste_face(out, restored, matrix)

        logger.info("restored %d face(s)", len(faces))
        return out

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        native = config.UPSCALER_NATIVE_SCALE
        out = imaging.upscale_tiled(
            img, run=self.models.upscaler.run, tile=config.UPSCALE_TILE, scale=native
        )

        # The compact model only comes at x4, but we always want x2, so resample back
        # down. That supersamples rather than discards — the extra detail is averaged
        # in, not thrown away.
        if native != 2:
            h, w = img.shape[:2]
            out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
        return out
