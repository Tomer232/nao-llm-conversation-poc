# nao_interface.py

class NaoInterface(object):
    def speak(self, text):
        """
        Speak text out loud.
        Simulation: uses PC TTS.
        Real robot: uses NAOqi ALTextToSpeech.
        """
        raise NotImplementedError()

    def listen(self):
        """
        Listen to user speech and return recognized text.
        Simulation: PC microphone + Whisper/Groq.
        Real robot: NAO mics (later), for now PC mic.
        """
        raise NotImplementedError()

    def shutdown(self):
        """Optional cleanup."""
        pass
