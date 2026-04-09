FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install chatterbox WITH all its dependencies
RUN pip install --no-cache-dir chatterbox-tts

# Force correct torch versions back (chatterbox may have downgraded them)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install runpod and boto3
RUN pip install --no-cache-dir runpod boto3

# Copy handler
COPY rp_handler.py /

CMD ["python", "rp_handler.py"]
