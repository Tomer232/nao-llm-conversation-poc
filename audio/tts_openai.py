import io
import os
import sys
import tempfile
import traceback

from openai import OpenAI

try:
    import winsound
except ImportError:
    winsound = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Single source of truth for allowed voices
SUPPORTED_TTS_VOICES = {
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
    "coral",
    "verse",
    "ballad",
    "ash",
    "sage",
    "marin",
    "cedar",
}

DEFAULT_TTS_VOICE = "onyx"
TTS_MODEL = "gpt-4o-mini-tts"


def _normalize_voice(raw_voice: str) -> str:
    """
    Normalize a requested voice to one of the supported OpenAI TTS voices.

    - If raw_voice is already valid, return it.
    - Otherwise, log a warning and return DEFAULT_TTS_VOICE.
    """
    if not raw_voice:
        return DEFAULT_TTS_VOICE

    v = str(raw_voice).strip()
    if v in SUPPORTED_TTS_VOICES:
        return v

    print(
        f"[TTS] Invalid voice '{v}'. Falling back to default '{DEFAULT_TTS_VOICE}'. "
        f"Supported voices: {sorted(SUPPORTED_TTS_VOICES)}"
    )
    return DEFAULT_TTS_VOICE


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        # Uses OPENAI_API_KEY env var
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak_text(text: str, voice: str = None) -> None:
    """
    Synthesize speech using OpenAI TTS and play it on the local machine.

    In this POC we:
      - always run in "PC speakers" mode (simulation)
      - do not stream, just fetch the full audio and then play

    If anything fails, we log the error and return without crashing.
    """
    if not text:
        return

    # Normalize the requested voice to a supported one
    voice = _normalize_voice(voice)

    client = get_client()

    try:
        print(f"[TTS] Synthesizing with voice='{voice}' and model='{TTS_MODEL}'")
        # Using with_raw_response so we can handle the binary body easily
        with client.audio.speech.with_raw_response.create(
            model=TTS_MODEL,
            voice=voice,
            input=text,
            format="wav",
        ) as resp:
            if resp.status_code != 200:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.read()
                print(f"[TTS] Error during synthesis: Error code: {resp.status_code} - {err_body}")
                return

            audio_bytes = resp.read()

        _play_wav_bytes(audio_bytes)
    except Exception as e:
        print(f"[TTS] Exception in speak_text: {e}")
        traceback.print_exc()
        # Do not raise further; conversation loop should continue


def _play_wav_bytes(wav_bytes: bytes) -> None:
    """
    Play WAV bytes on Windows using winsound. If winsound is unavailable,
    writes the audio to a temp file as a fallback.
    """
    if not wav_bytes:
        return

    if winsound is not None and sys.platform.startswith("win"):
        try:
            # Play from memory
            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_SYNC)
            return
        except Exception as e:
            print(f"[TTS] winsound memory playback failed: {e}")

    # Fallback: write to a temp file and do nothing more (or you could
    # hook in another player if you want).
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(wav_bytes)
            temp_path = f.name
        print(f"[TTS] Audio written to temp file: {temp_path}")
    except Exception as e:
        print(f"[TTS] Failed to write temp WAV file: {e}")
