"""Telegram transport: takes a photo, queues it, returns the result.

Everything CPU-heavy happens in a single worker thread — the models saturate the
cores they are given, so running two jobs at once would only make both slower while
doubling peak memory. Other requests wait in the queue, and the event loop stays
responsive.

Run:
    python bot.py
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from PIL import Image

import config
from models import Models, configure_threads
from pipeline import Pipeline
from settings import PRESET_LABELS, PRESETS, SettingsStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

router = Router()

# Extensions we accept. A whitelist, not a parse of the user's filename: the
# filename is attacker-controlled, and the old code built the temp path from
# whatever followed its last dot — so a name like "x.png/../../etc/authorized_keys"
# wrote the download wherever the sender liked.
# .heic/.heif are what iPhones shoot by default, so they turn up constantly. OpenCV
# cannot decode them; pipeline._load falls back to Pillow + pillow-heif for those.
ALLOWED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
}

# Pillow's own decompression-bomb guard; ours is stricter and reported nicely.
Image.MAX_IMAGE_PIXELS = config.MAX_INPUT_PIXELS

_models: Models | None = None
_pipeline: Pipeline | None = None
_settings = SettingsStore(config.DATA_DIR / "settings.json")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")

# Jobs queued or running. Guarded by the event loop being single-threaded: every
# read-modify-write below happens without an await in between.
_pending = 0
_last_job_at: dict[int, float] = {}


class RejectedError(Exception):
    """The request is refused, with a message meant for the user."""


def _suffix_for(file_name: str | None, default: str) -> str:
    """Pick a safe extension. Never trust the sender's filename."""
    if not file_name:
        return default
    suffix = Path(file_name).suffix.lower()
    return suffix if suffix in ALLOWED_SUFFIXES else default


def _check_decoded_size(path: Path) -> None:
    """Reject images that would explode in memory once decoded.

    A size limit on the *file* proves nothing: compression means a few hundred KB can
    unpack into hundreds of megapixels. Pillow reads the header without decoding the
    pixels, so this costs nothing and runs before any allocation.
    """
    try:
        with Image.open(path) as im:
            width, height = im.size
    except Exception as exc:
        raise RejectedError("Не удалось прочитать это изображение. Попробуйте другой файл.") from exc

    if width * height > config.MAX_INPUT_PIXELS:
        megapixels = width * height / 1e6
        raise RejectedError(
            f"Слишком большое изображение — {width}×{height} ({megapixels:.0f} Мпикс). "
            f"Максимум {config.MAX_INPUT_PIXELS / 1e6:.0f} Мпикс."
        )


def _claim_slot(user_id: int) -> int:
    """Take a place in the queue, or refuse. Returns the position (1 = next up)."""
    global _pending

    if _pending >= config.MAX_QUEUE_SIZE:
        raise RejectedError("Сейчас слишком много задач в очереди. Попробуйте через пару минут.")

    now = time.monotonic()
    last = _last_job_at.get(user_id)
    if last is not None and now - last < config.USER_COOLDOWN_SEC:
        wait = config.USER_COOLDOWN_SEC - (now - last)
        raise RejectedError(f"Подождите {wait:.0f} с перед следующим фото.")

    _last_job_at[user_id] = now
    _pending += 1
    return _pending


def _release_slot() -> None:
    global _pending
    _pending = max(0, _pending - 1)


@router.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "Пришлите старое чёрно-белое фото — я раскрашу его, восстановлю лица "
        "и повышу качество.\n\n"
        "Совет: отправляйте фото <b>как файл</b> (документ), а не как картинку — "
        "тогда Telegram не сожмёт его, и результат будет заметно лучше.",
        parse_mode="HTML",
    )


@router.message(Command("help"))
async def on_help(message: Message):
    await message.answer(
        "Что я делаю с фотографией:\n"
        "1. Убираю жёлтый налёт старого отпечатка\n"
        "2. Раскрашиваю (нейросеть DDColor)\n"
        "3. Восстанавливаю лица (RestoreFormer++)\n"
        "4. Повышаю разрешение, если фото маленькое\n\n"
        f"Ограничения: до {config.MAX_UPLOAD_MB} МБ, "
        f"до {config.MAX_INPUT_PIXELS / 1e6:.0f} Мпикс.\n"
        "Режим обработки — команда /settings.\n\n"
        "Отправляйте <b>файлом</b>, чтобы Telegram не сжимал фото."
    , parse_mode="HTML")


def _settings_keyboard(active: str) -> InlineKeyboardMarkup:
    """Preset buttons, with a check mark on the one currently in effect."""
    row = [
        InlineKeyboardButton(
            text=("✓ " if name == active else "") + PRESET_LABELS[name],
            callback_data=f"preset:{name}",
        )
        for name in PRESETS
    ]
    return InlineKeyboardMarkup(inline_keyboard=[row])


@router.message(Command("settings"))
async def on_settings(message: Message):
    active = _settings.preset_name(message.from_user.id)
    await message.answer(
        "Режим обработки:\n\n"
        "⚡ <b>Скорость</b> — раскрашивание и лица, без апскейла. ~10–20 секунд.\n"
        "💎 <b>Качество</b> — то же плюс повышение разрешения для мелких фото. Дольше.",
        reply_markup=_settings_keyboard(active),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("preset:"))
async def on_preset_chosen(callback: CallbackQuery):
    preset = callback.data.split(":", 1)[1]
    if preset not in PRESETS:
        await callback.answer("Неизвестный режим.")
        return

    _settings.set_preset(callback.from_user.id, preset)
    await callback.answer(f"Режим: {PRESET_LABELS[preset]}")
    # Re-render so the check mark moves to the chosen preset.
    with contextlib.suppress(Exception):
        await callback.message.edit_reply_markup(reply_markup=_settings_keyboard(preset))


@router.message(F.photo)
async def on_photo(message: Message):
    # Telegram sends several sizes; the last is the largest. A compressed photo
    # carries no filename, hence None.
    await _handle(message, file=message.photo[-1], suffix=".jpg", original_name=None)


@router.message(F.document)
async def on_document(message: Message):
    doc = message.document

    # Trust either signal. Telegram often labels HEIC as application/octet-stream
    # rather than image/heic, so demanding an image MIME type would reject the format
    # iPhones shoot by default. The extension is checked against a whitelist anyway,
    # and the file itself is validated before it is decoded.
    looks_like_image = (doc.mime_type or "").startswith("image/")
    has_image_suffix = Path(doc.file_name or "").suffix.lower() in ALLOWED_SUFFIXES

    if not (looks_like_image or has_image_suffix):
        await message.answer("Это не похоже на изображение. Пришлите фото или картинку-файлом.")
        return

    if doc.file_size and doc.file_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        await message.answer(
            f"Файл слишком большой ({doc.file_size / 1e6:.0f} МБ). "
            f"Максимум {config.MAX_UPLOAD_MB} МБ."
        )
        return

    await _handle(
        message,
        file=doc,
        suffix=_suffix_for(doc.file_name, ".png"),
        original_name=doc.file_name,
    )


@router.message()
async def on_other(message: Message):
    await message.answer("Пришлите изображение — фото или файлом.")


async def _handle(message: Message, file, suffix: str, original_name: str | None) -> None:
    user_id = message.from_user.id
    job_id = uuid.uuid4().hex[:8]
    in_path = config.TMP_DIR / f"{job_id}_in{suffix}"
    out_path = config.TMP_DIR / f"{job_id}_out.png"

    try:
        position = _claim_slot(user_id)
    except RejectedError as exc:
        await message.answer(str(exc))
        return

    waiting = position - 1
    status = await message.answer(
        f"Принял, вы {waiting}-й в очереди. Начну, как освободится очередь…"
        if waiting
        else "Принял. Обрабатываю…"
    )

    loop = asyncio.get_running_loop()

    def report(stage: str, index: int, total: int) -> None:
        """Called from the worker thread — hop back to the loop to touch Telegram."""
        text = _STAGE_LABELS.get(stage, stage)
        asyncio.run_coroutine_threadsafe(
            _edit(status, f"{text}… ({index}/{total})"), loop
        )

    settings = _settings.settings_for(user_id)

    try:
        await message.bot.download(file, destination=in_path)
        _check_decoded_size(in_path)

        started = time.monotonic()
        deadline = started + config.JOB_TIMEOUT_SEC

        await asyncio.wait_for(
            loop.run_in_executor(
                _executor,
                lambda: _pipeline.process(in_path, out_path, settings, report, deadline),
            ),
            # A little slack over the pipeline's own deadline: let it stop itself
            # cleanly rather than abandoning a thread that keeps running regardless.
            timeout=config.JOB_TIMEOUT_SEC + 30,
        )
        elapsed = time.monotonic() - started

        await _send_result(message, out_path, elapsed, original_name)

    except RejectedError as exc:
        await message.answer(str(exc))
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("job %s timed out", job_id)
        await message.answer(
            "Не успел обработать это фото за отведённое время. "
            "Попробуйте фото поменьше."
        )
    except Exception:
        logger.exception("job %s failed", job_id)
        await message.answer("Не получилось обработать это изображение. Попробуйте другое фото.")
    finally:
        _release_slot()
        for path in (in_path, out_path):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            await status.delete()


_STAGE_LABELS = {
    "desaturate": "Убираю налёт старого отпечатка",
    "colorize": "Раскрашиваю",
    "white_balance": "Выравниваю цвета",
    "faces": "Восстанавливаю лица",
    "upscale": "Повышаю разрешение",
}


async def _edit(status: Message, text: str) -> None:
    # The user may have deleted the status message, and the text may be unchanged —
    # neither is worth failing a job over.
    with contextlib.suppress(Exception):
        await status.edit_text(text)


def _result_name(original_name: str | None) -> str:
    """Name the result after the photo it came from, so a batch stays sortable.

    Returning "colorized.png" for everything means a user who sends ten photos ends
    up with ten identically-named files and no way to tell which is which.
    """
    stem = Path(original_name).stem if original_name else "photo"
    stem = stem.strip() or "photo"
    return f"{stem}_colorized.png"


async def _send_result(
    message: Message, out_path: Path, elapsed: float, original_name: str | None
) -> None:
    """Send a preview plus the full-quality file.

    Sending only a document (as the old bot did) leaves the user looking at a file
    icon, having to download it to see what they got. Sending only a photo would let
    Telegram re-compress it and throw away the quality we just spent a minute
    producing. So: both.
    """
    filename = _result_name(original_name)
    caption = f"Готово за {elapsed:.0f} с."

    data = out_path.read_bytes()
    with contextlib.suppress(Exception):
        await message.answer_photo(
            BufferedInputFile(data, filename=filename),
            caption=caption + " Ниже — файл без сжатия.",
        )

    await message.answer_document(FSInputFile(out_path, filename=filename))


def _clean_tmp() -> None:
    """Clear leftovers from a previous run that died mid-job."""
    removed = 0
    for path in config.TMP_DIR.glob("*"):
        if path.is_file():
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
    if removed:
        logger.info("cleared %d stale temp file(s)", removed)


async def main() -> None:
    global _models, _pipeline

    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is not set (put it in .env).")

    _clean_tmp()
    configure_threads(config.NUM_THREADS)

    _models = Models(
        onnx_dir=config.ONNX_DIR,
        colorizer_name=config.COLORIZER_MODEL,
        num_threads=config.NUM_THREADS,
        keep_loaded=config.KEEP_MODELS_LOADED,
        upscaler_name=config.UPSCALER_MODEL,
    )
    _models.warm_up()
    _pipeline = Pipeline(_models)

    bot = Bot(config.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    logger.info(
        "bot up: colorizer=%s threads=%d keep_loaded=%s",
        config.COLORIZER_MODEL, config.NUM_THREADS, config.KEEP_MODELS_LOADED,
    )
    try:
        await dp.start_polling(bot)
    except TelegramUnauthorizedError:
        # By far the most common first-run mistake; a stack trace helps nobody.
        raise SystemExit("Telegram rejected BOT_TOKEN. Check the value in .env.") from None
    finally:
        # Let an in-flight job finish rather than killing it half-written.
        logger.info("shutting down; waiting for the current job")
        _executor.shutdown(wait=True)
        await bot.session.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
