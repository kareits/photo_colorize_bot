"""ONNX session lifecycle: thread limits, lazy loading, eviction, warm-up.

Separate from imaging.py on purpose. This file worries about memory and threads;
that one worries about pixels.

Two constraints from the target server shape everything here (4 vCPU, 3.8 GB RAM,
~1 GB of which the neighbouring TTS service already holds):

* **Threads are capped.** onnxruntime and OpenCV both grab every core by default,
  which would starve the TTS service sharing the box.
* **Models are loaded one at a time.** Held resident, the three models sum to more
  memory than is free. Loaded per stage and evicted after, peak memory is the
  *largest* model rather than their sum — the difference between fitting and an
  OOM kill that could take the neighbour down with it.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def configure_threads(num_threads: int) -> None:
    """Stop OpenCV from spreading across every core. Call once at startup."""
    cv2.setNumThreads(num_threads)


class OnnxModel:
    """A lazily-loaded ONNX session that can be evicted again.

    Sessions are cheap to recreate relative to how expensive it is to hold three
    of them at once on a 3.8 GB box, so `release()` is the normal path, not an
    emergency measure.
    """

    def __init__(self, path: Path, num_threads: int, keep_loaded: bool, use_arena: bool = False):
        self.path = path
        self.num_threads = num_threads
        self.keep_loaded = keep_loaded
        self.use_arena = use_arena
        self._session: ort.InferenceSession | None = None

    def _load(self) -> ort.InferenceSession:
        if self._session is None:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"missing model: {self.path}. Export it with tools/export_onnx.py "
                    f"(see docker/Dockerfile.export)."
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.num_threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # The arena allocator is a genuine trade, and the right answer differs
            # per model — which is why this is a flag rather than a constant.
            #
            # It reserves memory well beyond what the graph needs, and on the big
            # colouriser that is fatal: DDColor-large peaks at 2325 MB with the arena
            # versus 1006 MB without, and only the latter fits the server's free RAM.
            #
            # But the arena is also what makes multithreading work. Without it every
            # allocation goes to the system malloc, which serialises the graph: the
            # upscaler takes ~62 s on both 2 and 8 threads. With it, 8 threads bring
            # that to ~42 s — for 786 MB, which is affordable because peak memory is
            # the largest single model, not the sum.
            #
            # So: off for the colouriser (memory), on for the upscaler (speed).
            options.enable_cpu_mem_arena = self.use_arena

            logger.info("loading %s", self.path.name)
            self._session = ort.InferenceSession(
                str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        return self._session

    def run(self, tensor: np.ndarray) -> np.ndarray:
        session = self._load()
        name = session.get_inputs()[0].name
        return session.run(None, {name: tensor})[0]

    def release(self) -> None:
        if self._session is not None:
            self._session = None
            gc.collect()


class FaceDetector:
    """YuNet, via OpenCV's built-in FaceDetectorYN.

    Chosen over facexlib's RetinaFace for one decisive reason: it ships inside
    OpenCV, so it needs no torch. RetinaFace would have dragged torch, basicsr and
    facexlib back into the runtime for the sake of a detector — which is most of
    what this refactor is trying to shed. YuNet also returns the 5 landmarks the
    aligner needs, and the weights are 233 KB.
    """

    def __init__(self, path: Path, score_threshold: float = 0.8):
        if not path.is_file():
            raise FileNotFoundError(f"missing YuNet weights: {path}")
        self.path = path
        self.score_threshold = score_threshold

    def detect(self, img: np.ndarray, max_faces: int | None = None) -> list[np.ndarray]:
        """Return 5-point landmarks (as (5, 2) float32) per face, biggest first.

        Capped at max_faces because restoration cost is linear in face count: a group
        photo yielded 20 detections and 103 s in one stage. Sorting by size before
        truncating means we keep the faces that matter — a face large in frame is both
        the one a viewer looks at and the one restoration visibly improves.
        """
        h, w = img.shape[:2]
        # YuNet bakes the input size into the detector, so it is created per image.
        detector = cv2.FaceDetectorYN.create(
            str(self.path), "", (w, h), score_threshold=self.score_threshold
        )
        _, faces = detector.detect(img)
        if faces is None:
            return []

        # Each row is [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
        #              x_rcm, y_rcm, x_lcm, y_lcm, score] — landmarks live at 4:14,
        #              and the box's area is faces[2] * faces[3].
        ordered = sorted(faces, key=lambda f: float(f[2]) * float(f[3]), reverse=True)
        if max_faces is not None:
            ordered = ordered[:max_faces]

        return [face[4:14].reshape(5, 2).astype(np.float32) for face in ordered]


class Models:
    """The three ONNX models plus the detector, wired to the config."""

    def __init__(
        self,
        onnx_dir: Path,
        colorizer_name: str,
        num_threads: int,
        keep_loaded: bool,
        upscaler_name: str = "realesrgan_x2plus.onnx",
        detect_threshold: float = 0.8,
    ):
        self.keep_loaded = keep_loaded

        # Arena off where the model is big and memory is the binding constraint;
        # on where the model is small and speed is. See OnnxModel._load.
        self.colorizer = OnnxModel(onnx_dir / colorizer_name, num_threads, keep_loaded, use_arena=False)
        self.face_restorer = OnnxModel(
            onnx_dir / "restoreformer.onnx", num_threads, keep_loaded, use_arena=False
        )
        self.upscaler = OnnxModel(
            onnx_dir / upscaler_name, num_threads, keep_loaded, use_arena=True
        )
        self.face_detector = FaceDetector(
            onnx_dir / "face_detection_yunet.onnx", score_threshold=detect_threshold
        )

    def end_of_stage(self) -> None:
        """Evict models between stages, when they are not being kept resident.

        Eviction belongs here rather than in OnnxModel.run(): a single stage may
        invoke its model many times — the upscaler runs once per tile — and
        dropping the session after every call would reload it per tile, which
        turned a 6-second upscale into 38.
        """
        if not self.keep_loaded:
            self.release_all()

    def warm_up(self) -> None:
        """Touch each model once so the first user does not pay the load cost.

        Skipped when models are evicted after every stage — warming a session we
        are about to throw away would only waste time.
        """
        if not self.colorizer.keep_loaded:
            logger.info("sequential loading is on; skipping warm-up")
            return

        logger.info("warming up models")
        self.colorizer.run(np.zeros((1, 3, 512, 512), dtype=np.float32))
        self.face_restorer.run(np.zeros((1, 3, 512, 512), dtype=np.float32))
        self.upscaler.run(np.zeros((1, 3, 64, 64), dtype=np.float32))
        logger.info("models ready")

    def release_all(self) -> None:
        self.colorizer.release()
        self.face_restorer.release()
        self.upscaler.release()
