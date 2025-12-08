# audio/asr.py

import speech_recognition as sr


def transcribe_from_mic():
    r = sr.Recognizer()

    # Allow slightly longer pauses before it decides the phrase is over
    # (default is ~0.8; increasing gives you more time to think)
    r.pause_threshold = 1.2          # seconds of silence allowed mid-sentence
    r.non_speaking_duration = 0.5    # how long silence is considered "non speaking"

    with sr.Microphone() as source:
        # Calibrate to room noise for half a second
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("[ASR] Speak now...")

        try:
            # You have up to 10 seconds of speaking per turn.
            # Within that window, short pauses are allowed.
            audio = r.listen(source, timeout=None, phrase_time_limit=10)

            text = r.recognize_google(audio, language="en-US")
            print(f"[ASR] Recognized: {text}")
            return text

        except sr.UnknownValueError:
            print("[ASR] Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"[ASR] API error: {e}")
            return ""
