"""Build a contact sheet comparing colouriser settings. Development aid, not runtime.

Colour is decided by the colouriser and the white balance; faces and upscaling do not
touch it and cost a minute each, so both are skipped here.

    python -m tools.compare_colour
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config
import imaging
from models import Models

PHOTOS = Path("tests/fixtures/photos")
OUT = Path("compare_out")

# (label, colouriser, white balance strength, Minkowski norm)
# norm=1 is grey-world — what the original bot used, at strength 0.7. Worth
# reproducing exactly: it may be the source of the colours the old version got.
VARIANTS = [
    ("large, no WB", "ddcolor_large.onnx", 0.0, 6),
    ("OLD: greyworld 0.7", "ddcolor_large.onnx", 0.7, 1),
    ("large, WB 0.3 n6", "ddcolor_large.onnx", 0.3, 6),
    ("large, WB 0.6 n6", "ddcolor_large.onnx", 0.6, 6),
]


def saturation(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean())


def label_strip(width: int, text: str, height: int = 34) -> np.ndarray:
    strip = np.full((height, width, 3), 32, dtype=np.uint8)
    cv2.putText(strip, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    OUT.mkdir(exist_ok=True)
    originals = sorted(
        p for p in PHOTOS.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "colorized" not in p.name
    )
    if not originals:
        raise SystemExit(f"no source photos in {PHOTOS}")

    # One session per colouriser, reused across photos — loading DDColor-large twice
    # per image would dominate the runtime.
    sessions = {
        name: Models(config.ONNX_DIR, name, config.NUM_THREADS, keep_loaded=True)
        for name in {v[1] for v in VARIANTS}
    }

    for photo in originals:
        img = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        img = imaging.limit_long_side(img, config.MAX_INPUT_SIDE)
        grey = imaging.desaturate(img)

        columns = []
        thumb_w = 420

        def add(image: np.ndarray, text: str) -> None:
            scale = thumb_w / image.shape[1]
            thumb = cv2.resize(image, (thumb_w, int(image.shape[0] * scale)))
            columns.append(np.vstack([label_strip(thumb_w, text), thumb]))

        add(grey, "original (b/w)")

        for label, model_name, wb, norm in VARIANTS:
            ab = sessions[model_name].colorizer.run(imaging.colorize_input(grey))
            out = imaging.colorize_output(grey, ab)
            if wb > 0:
                out = imaging.white_balance(out, strength=wb, norm=norm)

            sat = saturation(out)
            add(out, f"{label} | sat {sat:.0f}")
            print(f"{photo.name:20s} {label:16s} saturation {sat:5.1f}")

            cv2.imwrite(str(OUT / f"{photo.stem}__{label.replace(', ', '_').replace(' ', '')}.png"), out)

        height = max(c.shape[0] for c in columns)
        padded = [
            np.vstack([c, np.full((height - c.shape[0], c.shape[1], 3), 32, dtype=np.uint8)])
            for c in columns
        ]
        sheet = np.hstack(padded)
        sheet_path = OUT / f"sheet_{photo.stem}.png"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"  -> {sheet_path}\n")


if __name__ == "__main__":
    main()
