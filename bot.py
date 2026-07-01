"""
Telegram-бот: принимает чёрно-белое фото, реставрирует/раскрашивает/апскейлит
и возвращает результат.

Запуск:
    export BOT_TOKEN="123:ABC..."
    python bot.py
"""
import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message

import config
from pipeline import RestorePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

router = Router()

# Один пайплайн на процесс (модели держатся в памяти и переиспользуются).
_pipeline = RestorePipeline()

# Один воркер: CPU тянет одно фото за раз, остальные запросы ждут в очереди.
_executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS)


@router.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "Пришлите чёрно-белое фото — я отреставрирую, раскрашу и повышу качество.\n\n"
        "Совет: отправляйте фото как файл (документ), а не как картинку, "
        "чтобы Telegram не сжимал его и результат был чётче."
    )


@router.message(F.photo)
async def on_photo(message: Message):
    # Telegram присылает несколько размеров; берём самый большой.
    await _handle(message, file=message.photo[-1], suffix=".jpg")


@router.message(F.document)
async def on_document(message: Message):
    doc = message.document
    if not (doc.mime_type or "").startswith("image/"):
        await message.answer("Это не похоже на изображение. Пришлите фото или картинку-файл.")
        return
    suffix = "." + (doc.file_name.rsplit(".", 1)[-1] if doc.file_name and "." in doc.file_name else "png")
    await _handle(message, file=doc, suffix=suffix)


@router.message()
async def on_other(message: Message):
    await message.answer("Пришлите изображение (фото или файл), и я его обработаю.")


async def _handle(message: Message, file, suffix: str):
    job_id = uuid.uuid4().hex[:8]
    in_path = config.TMP_DIR / f"{job_id}_in{suffix}"
    out_path = config.TMP_DIR / f"{job_id}_out.png"

    status = await message.answer("Принял. Обрабатываю — это может занять до пары минут...")

    try:
        bot = message.bot
        await bot.download(file, destination=in_path)

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        # Тяжёлая блокирующая работа — в пуле потоков, чтобы не вешать event loop.
        await loop.run_in_executor(_executor, _pipeline.process, in_path, out_path)
        dt = time.monotonic() - t0

        await message.answer_document(
            FSInputFile(out_path),
            caption=f"Готово за {dt:.0f} с.",
        )
    except Exception as exc:  # noqa: BLE001 — для MVP отдаём пользователю мягкую ошибку
        logger.exception("Job %s упал", job_id)
        await message.answer("Не получилось обработать это изображение. Попробуйте другое фото.")
    finally:
        # Чистим временные файлы.
        for p in (in_path, out_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            await status.delete()
        except Exception:
            pass


async def main():
    if not config.BOT_TOKEN:
        raise SystemExit("Не задан BOT_TOKEN (export BOT_TOKEN=...).")
    bot = Bot(config.BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    logger.info("Бот запущен. Ожидаю сообщения...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
