import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile

print("Loading Chatterbox model...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Model ready on {'cuda' if torch.cuda.is_available() else 'cpu'}!")

# Emotion presets for bedtime story app
EMOTION_PRESETS = {
    "calm":       {"exaggeration": 0.3, "cfg_weight": 0.3},  # soothing bedtime
    "warm":       {"exaggeration": 0.6, "cfg_weight": 0.4},  # warm storyteller
    "neutral":    {"exaggeration": 0.5, "cfg_weight": 0.5},  # normal narration
    "happy":      {"exaggeration": 0.8, "cfg_weight": 0.5},  # excited/joyful
    "dramatic":   {"exaggeration": 1.0, "cfg_weight": 0.3},  # big story moments
    "whisper":    {"exaggeration": 0.2, "cfg_weight": 0.2},  # soft/sleepy ending
    "excited":    {"exaggeration": 0.9, "cfg_weight": 0.4},  # adventure scenes
    "sad":        {"exaggeration": 0.4, "cfg_weight": 0.6},  # emotional moments
}


def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("prompt", job_input.get("text", ""))
    reference_audio_base64 = job_input.get("reference_audio_base64", None)
    emotion = job_input.get("emotion", None)

    # Use emotion preset if provided, otherwise use manual values
    if emotion and emotion in EMOTION_PRESETS:
        preset = EMOTION_PRESETS[emotion]
        exaggeration = preset["exaggeration"]
        cfg_weight = preset["cfg_weight"]
        print(f"Using emotion preset: {emotion}")
    else:
        exaggeration = float(job_input.get("exaggeration", 0.5))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))

    if not text:
        return {"error": "text or prompt is required"}

    print(f"Text: {text[:60]}...")
    print(f"Emotion: {emotion or 'custom'} | exaggeration: {exaggeration} | cfg_weight: {cfg_weight}")
    print(f"Voice cloning: {'YES' if reference_audio_base64 else 'NO (default voice)'}")

    ref_path = None
    try:
        if reference_audio_base64:
            audio_bytes = base64.b64decode(reference_audio_base64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_path = tmp.name

        wav = model.generate(
            text,
            audio_prompt_path=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        buf = io.BytesIO()
        torchaudio.save(buf, wav, model.sr, format="wav")
        buf.seek(0)

        audio_base64 = base64.b64encode(buf.read()).decode("utf-8")
        print("Done!")

        return {
            "audio_base64": audio_base64,
            "sample_rate": model.sr,
            "voice_cloned": reference_audio_base64 is not None,
            "emotion": emotion or "custom",
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
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
