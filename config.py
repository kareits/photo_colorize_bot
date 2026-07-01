"""Настройки приложения. Меняйте значения здесь, не в коде пайплайна."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Подхватываем переменные из файла .env (если он есть) в окружение.
# Реальная переменная окружения, заданная вручную, имеет приоритет над .env.
load_dotenv()

# --- Telegram ---
# Токен берётся из .env или переменной окружения BOT_TOKEN (получить у @BotFather).
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# --- Директории ---
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"          # входные/выходные файлы, чистятся после отправки
WEIGHTS_DIR = BASE_DIR / "weights"  # сюда кэшируются веса моделей
TMP_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# --- Параметры обработки ---
# Ограничиваем длинную сторону ДО реставрации/раскрашивания, чтобы CPU не задыхался.
# Финальный апскейл всё равно поднимет разрешение в конце.
MAX_INPUT_SIDE = 768

# Финальный апскейл Real-ESRGAN. На CPU x2 в разы быстрее, чем x4.
UPSCALE_FACTOR = 2

# Тайлинг Real-ESRGAN: режет картинку на плитки, чтобы ограничить пик памяти.
# 0 = без тайлинга. На 8 ГБ ОЗУ держите 200-256.
REALESRGAN_TILE = 256

# CPU обрабатывает одно фото за раз. Один воркер = бот остаётся отзывчивым,
# остальные запросы спокойно ждут в очереди исполнителя.
MAX_CONCURRENT_JOBS = 1

# Включение/выключение стадий (удобно для отладки и сравнения).
ENABLE_FACE_RESTORE = True

# Сила реставрации лиц: 0.0 = оригинал без изменений, 1.0 = полный GFPGAN.
# GFPGAN склонен пере-усиливать лица — контрастные, "стеклянные" глаза и зубы.
# 0.4-0.6 обычно даёт естественный результат. Уменьшайте, если лица выглядят
# слишком резкими/нарисованными.
FACE_RESTORE_STRENGTH = 0.4

ENABLE_COLORIZE = True
ENABLE_UPSCALE = True

# Если RAM совсем мало (<= 8 ГБ): True = модель грузится перед стадией и
# выгружается после неё (экономит память ценой времени на загрузку каждый раз).
# False = модели грузятся один раз и остаются в памяти (быстрее на запрос).
SEQUENTIAL_MODEL_LOADING = False

# URL весов (скачиваются автоматически при первом запуске, если их нет).
GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
REALESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
DDCOLOR_MODELSCOPE_ID = "damo/cv_ddcolor_image-colorization"
