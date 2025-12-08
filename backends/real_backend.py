# backends/real_backend.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nao_interface import NaoInterface
from audio.asr import transcribe_from_mic


class RealNao(NaoInterface):
    def __init__(self, robot_ip, port=9559):
        # LAZY IMPORT so Python 3 never loads NAOqi
        try:
            from naoqi import ALProxy
        except Exception:
            raise RuntimeError(
                "NAOqi can only be used in Python 2.7 environment. "
                "Switch MODE='simulation' when running under Python 3."
            )

        self.tts = ALProxy("ALTextToSpeech", robot_ip, port)

    def speak(self, text):
        print("[REAL ROBOT] Speaking:", text)
        self.tts.say(text)

    def listen(self):
        # For now still using PC mic for real robot mode
        print("[REAL ROBOT] Listening via PC microphone (temporary)...")
        return transcribe_from_mic()
