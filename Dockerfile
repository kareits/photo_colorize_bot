# Runtime image. Torch-free: onnxruntime + opencv + aiogram, nothing more.
#
# The models are NOT baked in — they are ~1.5 GB of .onnx and change independently of
# the code. They come in through a volume (see docker-compose.yml), copied to the
# server once. This keeps the image small and a code change a small push.
#
# There is no build stage here. Exporting the ONNX needs torch and is done on a
# workstation (see requirements-export.txt); the server only ever runs inference.
FROM python:3.11-slim

# libglib2.0-0 is the one system library opencv-python-headless still needs; the
# headless build spares us libGL and the rest of the GUI stack. pillow-heif bundles
# its own libheif, so HEIC support needs nothing installed here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code edit does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# tmp/ holds in/out files mid-job; onnx/ is the mounted model volume; data/ is the
# mounted volume for persistent state (user settings).
RUN mkdir -p tmp onnx data

# Unbuffered so logs reach docker as they happen, not in blocks.
ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
