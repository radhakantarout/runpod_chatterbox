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


def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("prompt", job_input.get("text", ""))
    reference_audio_base64 = job_input.get("reference_audio_base64", None)
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))

    if not text:
        return {"error": "text or prompt is required"}

    print(f"Generating: {text[:60]}...")

    ref_path = None
    try:
        # Save reference audio if provided
        if reference_audio_base64:
            audio_bytes = base64.b64decode(reference_audio_base64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            ref_path = tmp.name

        # Generate audio
        wav = model.generate(
            text,
            audio_prompt_path=ref_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        # Convert to WAV bytes
        buf = io.BytesIO()
        torchaudio.save(buf, wav, model.sr, format="wav")
        buf.seek(0)

        audio_base64 = base64.b64encode(buf.read()).decode("utf-8")
        print("Done!")

        return {
            "audio_base64": audio_base64,
            "sample_rate": model.sr,
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


runpod.serverless.start({"handler": handler})
