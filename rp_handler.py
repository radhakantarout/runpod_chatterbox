import runpod
import torch
import torchaudio
import base64
import io
import os
import re
import tempfile
import numpy as np

print("Loading Chatterbox model...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Model ready on {'cuda' if torch.cuda.is_available() else 'cpu'}!")

EMOTION_PRESETS = {
    "calm":     {"exaggeration": 0.3, "cfg_weight": 0.3},
    "warm":     {"exaggeration": 0.6, "cfg_weight": 0.4},
    "neutral":  {"exaggeration": 0.5, "cfg_weight": 0.5},
    "happy":    {"exaggeration": 0.8, "cfg_weight": 0.5},
    "dramatic": {"exaggeration": 1.0, "cfg_weight": 0.3},
    "whisper":  {"exaggeration": 0.2, "cfg_weight": 0.2},
    "excited":  {"exaggeration": 0.9, "cfg_weight": 0.4},
    "sad":      {"exaggeration": 0.4, "cfg_weight": 0.6},
}

MAX_CHUNK_CHARS = 250  # safe limit per generation


def split_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split text into chunks at sentence boundaries"""
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If single sentence exceeds limit, split at commas/clauses
        if len(sentence) > max_chars:
            sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in sub_parts:
                if len(current) + len(part) + 1 <= max_chars:
                    current = (current + " " + part).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = part
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def add_silence(duration_ms, sample_rate):
    """Create silence tensor"""
    samples = int(sample_rate * duration_ms / 1000)
    return torch.zeros(1, samples)


def generate_chunk(text, ref_path, exaggeration, cfg_weight):
    """Generate audio for a single chunk"""
    wav = model.generate(
        text,
        audio_prompt_path=ref_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    # Ensure shape is (1, N)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("prompt", job_input.get("text", ""))
    reference_audio_base64 = job_input.get("reference_audio_base64", None)
    emotion = job_input.get("emotion", "warm")
    pause_between_chunks_ms = int(job_input.get("pause_ms", 300))

    if not text:
        return {"error": "text or prompt is required"}

    # Get emotion settings
    if emotion in EMOTION_PRESETS:
        preset = EMOTION_PRESETS[emotion]
        exaggeration = preset["exaggeration"]
        cfg_weight = preset["cfg_weight"]
    else:
        exaggeration = float(job_input.get("exaggeration", 0.5))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))

    ref_path = None
    try:
        # Save reference audio if provided
        if reference_audio_base64:
            audio_bytes = base64.b64decode(reference_audio_base64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_path = tmp.name
            print(f"Voice clone reference saved: {ref_path}")

        # Split text into chunks
        chunks = split_text(text)
        total = len(chunks)
        print(f"Split into {total} chunks for long text generation")

        if total == 0:
            return {"error": "No text to process"}

        # Generate audio for each chunk
        audio_parts = []
        silence = add_silence(pause_between_chunks_ms, model.sr)

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{total}: {chunk[:50]}...")
            try:
                wav = generate_chunk(chunk, ref_path, exaggeration, cfg_weight)
                audio_parts.append(wav)
                # Add pause between chunks (not after last)
                if i < total - 1:
                    audio_parts.append(silence)
            except Exception as e:
                print(f"Chunk {i+1} failed: {e} — skipping")
                continue

        if not audio_parts:
            return {"error": "All chunks failed to generate"}

        # Stitch all chunks together
        print("Stitching audio chunks together...")
        full_audio = torch.cat(audio_parts, dim=1)

        # Normalize to prevent clipping
        peak = full_audio.abs().max()
        if peak > 1.0:
            full_audio = full_audio / peak * 0.95

        # Save to buffer
        buf = io.BytesIO()
        torchaudio.save(buf, full_audio, model.sr, format="wav")
        buf.seek(0)

        audio_base64 = base64.b64encode(buf.read()).decode("utf-8")
        duration_sec = full_audio.shape[1] / model.sr

        print(f"Done! Total duration: {duration_sec:.1f}s from {total} chunks")

        return {
            "audio_base64": audio_base64,
            "sample_rate": model.sr,
            "duration_seconds": round(duration_sec, 1),
            "chunks_processed": total,
            "voice_cloned": reference_audio_base64 is not None,
            "emotion": emotion,
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


runpod.serverless.start({"handler": handler})
