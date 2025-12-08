# audio/tts.py

import pyttsx3


def synthesize_and_play(text: str) -> None:
    """
    Synthesize the given text and play it through the default audio device.
    We re-initialize the engine on every call to avoid Windows/pyttsx3
    issues where only the first utterance is spoken.
    """
    if not text:
        return

    engine = pyttsx3.init()

    # (Optional) you can tweak voice/rate/volume here if you want:
    # voices = engine.getProperty("voices")
    # engine.setProperty("voice", voices[0].id)
    # engine.setProperty("rate", 180)
    # engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()
    engine.stop()
