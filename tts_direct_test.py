from dotenv import load_dotenv
from openai import OpenAI
import os
import tempfile
import winsound

load_dotenv()  # loads OPENAI_API_KEY from .env

print("OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))

client = OpenAI()

try:
    print("Requesting OpenAI TTS...")
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input="Hello Tomer, this is a direct OpenAI TTS test.",
        format="wav",
    )

    audio_bytes = resp.read()
    print("Bytes received:", len(audio_bytes))

    if not audio_bytes:
        print("No audio bytes returned.")
    else:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(audio_bytes)

        print("Playing:", path)
        winsound.PlaySound(path, winsound.SND_FILENAME)
        print("Playback finished.")
        winsound.PlaySound(None, 0)

        os.remove(path)

except Exception as e:
    print("TTS ERROR:", repr(e))
