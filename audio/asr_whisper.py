import os
import tempfile
from datetime import datetime
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY is picked up
load_dotenv()

# Single global OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()


def _record_to_wav(path: str, duration_sec: float = 8.0, sample_rate: int = 16000):
    """
    Record from default system microphone into a mono WAV file.

    This function is BLOCKING – safe for your current turn-based loop.
    """
    print("[ASR] Speak now...")
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # wait until recording is finished

    # Normalize to int16 PCM
    audio_int16 = np.int16(audio * 32767)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def _transcribe_file(path: str, language: str = "en") -> str:
    """
    Send WAV file to OpenAI Whisper and return transcript text.
    """
    with open(path, "rb") as audio_file:
        # Model name can be "whisper-1" or newer audio models.
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="json",
        )

    text = (response.text or "").strip()
    return text


def transcribe_from_mic(duration_sec: float = 8.0, language: str = "en") -> str:
    """
    Public function used by the conversation loop.

    1. Records audio from mic to a temp WAV.
    2. Sends file to Whisper.
    3. Returns recognized text (string).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"robot_input_{timestamp}.wav")

    try:
        # 1) Record
        _record_to_wav(tmp_path, duration_sec=duration_sec)

        # 2) Transcribe
        transcript = _transcribe_file(tmp_path, language=language)

        print(f"[ASR] Recognized: {transcript}")
        return transcript
    finally:
        # 3) Cleanup temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            # If deletion fails, not critical for the POC
            pass
