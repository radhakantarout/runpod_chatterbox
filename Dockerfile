FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Install system deps
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install chatterbox without overriding torch
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install remaining deps manually
RUN pip install --no-cache-dir \
    numpy scipy soundfile tokenizers conformer einops \
    encodec s3tokenizer resemble-perth pyyaml safetensors \
    huggingface_hub transformers diffusers runpod boto3 \
    librosa audioread

# Copy handler
COPY rp_handler.py /

# Start — model downloads on first run, not during build
CMD ["python", "rp_handler.py"]
