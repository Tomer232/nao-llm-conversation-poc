# audio/tts.py
#
# Text-to-Speech using OpenAI TTS (gpt-4o-mini-tts).
# Voice is read from settings.json ("tts_voice"); falls back to "onyx".
# Audio is requested as raw 16-bit PCM and wrapped into a proper WAV
# header to avoid the broken WAV header bug in the OpenAI response.
# Playback is done via Windows winsound.

import os
import io
import json
import tempfile
import wave
import winsound  # Windows-native WAV playback

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables so OPENAI_API_KEY is available
load_dotenv()

# Single global OpenAI client (reads OPENAI_API_KEY from env)
client = OpenAI()

# Project root and settings file (same settings.json used everywhere else)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

ALLOWED_VOICES = {"alloy", "verse", "onyx", "copper", "opal", "amber"}
DEFAULT_VOICE = "onyx"  # default male, natural-sounding

# OpenAI TTS PCM format: 24 kHz, 16-bit, mono
PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)


def _get_configured_voice() -> str:
    """
    Read the TTS voice from settings.json ("tts_voice").
    If missing/invalid, return DEFAULT_VOICE.
    """
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = data.get("tts_voice", DEFAULT_VOICE)
        if v in ALLOWED_VOICES:
            return v
    except Exception:
        pass
    return DEFAULT_VOICE


def _save_wav_to_temp(wav_bytes: bytes) -> str:
    """
    Save the WAV bytes to a temporary .wav file and return the path.
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)  # we will reopen with normal write
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


def _pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    """
    Wrap raw 16-bit PCM bytes in a valid WAV container.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(PCM_CHANNELS)
        wf.setsampwidth(PCM_SAMPLE_WIDTH)
        wf.setframerate(PCM_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def synthesize_and_play(text: str, voice: str = None) -> None:
    """
    Synthesize the given text with OpenAI TTS and play it out loud.

    :param text: Text to speak.
    :param voice: Optional explicit voice name; if None, use settings.json.
    """
    if not text:
        return

    if voice is None:
        voice = _get_configured_voice()
    elif voice not in ALLOWED_VOICES:
        voice = DEFAULT_VOICE

    try:
        # Request raw PCM output from OpenAI TTS (24kHz, mono, 16-bit).
        # We then wrap it into a proper WAV to avoid the broken WAV header
        # that we saw when using response_format="wav".
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="pcm",
        )

        pcm_bytes = response.read()

        if not pcm_bytes:
            print("[TTS] Empty audio response from OpenAI.")
            print("[TTS] Fallback, text was:", text)
            return

        # Build a correct WAV container
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes)

        # Optional debug: inspect the WAV header
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                ch = wf.getnchannels()
                rate = wf.getframerate()
                width = wf.getsampwidth()
                frames = wf.getnframes()
            print(f"[TTS] Voice={voice}, WAV: channels={ch}, rate={rate}, "
                  f"width={width}, frames={frames}")
        except Exception as e:
            print("[TTS] Warning: WAV sanity check failed:", e)

        tmp_path = _save_wav_to_temp(wav_bytes)
        print(f"[TTS] Saved WAV to: {tmp_path}")
        print(f"[TTS] Playing via winsound.PlaySound: {tmp_path}")

        try:
            # Blocking playback; this should now play correctly if your
            # Windows sound device is working.
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
        finally:
            print("[TTS] PlaySound finished")
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    except Exception as e:
        # TTS failure should not crash the session
        print("[TTS] Error during synthesis:", e)
        print("[TTS] Fallback, text was:", text)
