"""
Пайплайн обработки фото на CPU.

Порядок стадий (важен!):
  1. Реставрация лиц  — GFPGAN        (Apache-2.0)
  2. Раскрашивание    — DDColor       (Apache-2.0, через modelscope)
  3. Апскейл          — Real-ESRGAN   (BSD-3)

Между стадиями изображение живёт как BGR uint8 ndarray (формат OpenCV).
Модели грузятся лениво. При SEQUENTIAL_MODEL_LOADING=True каждая стадия
выгружает свою модель после работы, чтобы не держать всё в памяти разом.
"""
import gc
import logging

import cv2
import numpy as np
import torch

import config

logger = logging.getLogger(__name__)

# CPU-режим жёстко: никаких попыток уехать на GPU.
torch.set_grad_enabled(False)
_DEVICE = torch.device("cpu")


class RestorePipeline:
    def __init__(self):
        self._gfpgan = None
        self._colorizer = None
        self._upsampler = None

    # ------------------------------------------------------------------ #
    # Публичный метод: полный прогон одного файла.
    # Блокирующий и CPU-тяжёлый — в боте вызывается в отдельном потоке.
    # ------------------------------------------------------------------ #
    def process(self, in_path, out_path):
        img = self._load_bgr(in_path)
        img = self._limit_size(img, config.MAX_INPUT_SIDE)

        if config.ENABLE_FACE_RESTORE:
            img = self._stage(self._restore_faces, img, name="face_restore")
        if config.ENABLE_COLORIZE:
            img = self._stage(self._colorize, img, name="colorize")
        if config.ENABLE_AUTO_WHITE_BALANCE:
            img = self._stage(self._auto_white_balance, img, name="white_balance")
        if config.ENABLE_UPSCALE:
            img = self._stage(self._upscale, img, name="upscale")

        cv2.imwrite(str(out_path), img)
        return out_path

    # ------------------------------------------------------------------ #
    # Обёртка стадии: логирование + опциональная выгрузка модели после.
    # ------------------------------------------------------------------ #
    def _stage(self, fn, img, name):
        logger.info("Стадия %s: старт (вход %sx%s)", name, img.shape[1], img.shape[0])
        out = fn(img)
        if config.SEQUENTIAL_MODEL_LOADING:
            self._release_all()
        logger.info("Стадия %s: готово (выход %sx%s)", name, out.shape[1], out.shape[0])
        return out

    # ------------------------------------------------------------------ #
    # Вспомогательное: загрузка и нормализация входа.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_bgr(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # всегда 3 канала BGR
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")
        return img

    @staticmethod
    def _limit_size(img, max_side):
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side <= max_side:
            return img
        scale = max_side / long_side
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ #
    # Стадия 1: реставрация лиц (GFPGAN).
    # ------------------------------------------------------------------ #
    def _restore_faces(self, img):
        gfpgan = self._get_gfpgan()
        # upscale=1: апскейлом займётся Real-ESRGAN в самом конце.
        _, _, restored = gfpgan.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        if restored is None:
            return img

        strength = config.FACE_RESTORE_STRENGTH
        if strength >= 0.999:
            return restored

        # Подмешиваем оригинал, чтобы убрать пере-резкость GFPGAN
        # ("стеклянные"/контрастные глаза). strength=0.5 — половина эффекта.
        if restored.shape != img.shape:
            restored = cv2.resize(restored, (img.shape[1], img.shape[0]))
        return cv2.addWeighted(restored, strength, img, 1.0 - strength, 0.0)

    def _get_gfpgan(self):
        if self._gfpgan is None:
            from gfpgan import GFPGANer  # импорт здесь — чтобы тяжёлые либы грузились лениво
            logger.info("Загружаю GFPGAN...")
            self._gfpgan = GFPGANer(
                model_path=config.GFPGAN_MODEL_URL,  # скачается и закэшируется при первом запуске
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=_DEVICE,
            )
        return self._gfpgan

    # ------------------------------------------------------------------ #
    # Стадия 2: раскрашивание (DDColor через modelscope).
    # ------------------------------------------------------------------ #
    def _colorize(self, img):
        colorizer = self._get_colorizer()
        from modelscope.outputs import OutputKeys
        result = colorizer(img)              # принимает BGR ndarray
        out = result[OutputKeys.OUTPUT_IMG]  # BGR uint8 ndarray
        return np.ascontiguousarray(out)

    def _get_colorizer(self):
        if self._colorizer is None:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            logger.info("Загружаю DDColor (modelscope)...")
            self._colorizer = pipeline(
                Tasks.image_colorization,
                model=config.DDCOLOR_MODELSCOPE_ID,
                device="cpu",
            )
        return self._colorizer

    # ------------------------------------------------------------------ #
    # Постобработка: авто-баланс белого ("серый мир").
    # Предполагаем, что средний цвет сцены должен быть нейтрально-серым,
    # и масштабируем каналы так, чтобы убрать общий цветовой сдвиг.
    # Модели не требует — чистый numpy.
    # ------------------------------------------------------------------ #
    def _auto_white_balance(self, img):
        strength = config.AUTO_WHITE_BALANCE_STRENGTH
        if strength <= 0.001:
            return img

        f = img.astype(np.float32)
        means = f.reshape(-1, 3).mean(axis=0)          # средние по каналам [B, G, R]
        gray = float(means.mean())
        scales = gray / np.clip(means, 1e-6, None)     # коэффициенты к серому
        corrected = f * scales                         # broadcast по каналам

        # Смешиваем с оригиналом по силе, чтобы коррекцию можно было ослабить.
        out = f * (1.0 - strength) + corrected * strength
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Стадия 3: апскейл (Real-ESRGAN x2).
    # ------------------------------------------------------------------ #
    def _upscale(self, img):
        upsampler = self._get_upsampler()
        out, _ = upsampler.enhance(img, outscale=config.UPSCALE_FACTOR)
        return out

    def _get_upsampler(self):
        if self._upsampler is None:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            logger.info("Загружаю Real-ESRGAN...")
            arch = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            )
            self._upsampler = RealESRGANer(
                scale=2,
                model_path=config.REALESRGAN_MODEL_URL,  # скачается при первом запуске
                model=arch,
                tile=config.REALESRGAN_TILE,
                tile_pad=10,
                pre_pad=0,
                half=False,          # на CPU только full precision
                device=_DEVICE,
            )
        return self._upsampler

    # ------------------------------------------------------------------ #
    # Выгрузка моделей (режим экономии памяти).
    # ------------------------------------------------------------------ #
    def _release_all(self):
        self._gfpgan = None
        self._colorizer = None
        self._upsampler = None
        gc.collect()