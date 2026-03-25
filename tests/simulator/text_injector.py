"""Text-only conversation injector that bypasses all audio hardware.

Subclasses ConversationManager to provide run_text_turn(), which accepts
plain text instead of recording from a microphone.  TTS and audio output
are replaced with silent no-op stubs so the test suite never touches
speakers or the NAO robot.
"""

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from antagonist_robot.conversation.avct_manager import AvctManager
from antagonist_robot.conversation.manager import ConversationManager, extract_end_signal
from antagonist_robot.logging.session_logger import SessionLogger
from antagonist_robot.nao.base import NAOAdapter
from antagonist_robot.pipeline.audio_output import AudioOutputBase
from antagonist_robot.pipeline.llm import LLMEngine
from antagonist_robot.pipeline.types import ASRResult, LLMResult, TTSResult, TurnResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal stubs — callers never instantiate these directly
# ---------------------------------------------------------------------------

class _SilentAudioOutput(AudioOutputBase):
    """No-op audio output that never plays sound."""

    def play_audio(self, tts_result: TTSResult) -> None:
        """Do nothing — audio is suppressed in test mode."""

    def speak_text(self, text: str) -> None:
        """Do nothing — TTS is suppressed in test mode."""

    def stop(self) -> None:
        """Do nothing — nothing to stop."""


class _NullNAO(NAOAdapter):
    """No-op NAO adapter that satisfies the abstract interface."""

    def connect(self) -> None:
        """No-op."""

    def disconnect(self) -> None:
        """No-op."""

    def on_response(self, text: str, hostility_level: int) -> None:
        """No-op."""

    def on_listening(self) -> None:
        """No-op."""

    def on_idle(self) -> None:
        """No-op."""

    def is_connected(self) -> bool:
        """Always disconnected."""
        return False


# ---------------------------------------------------------------------------
# TextInjector
# ---------------------------------------------------------------------------

class TextInjector(ConversationManager):
    """ConversationManager subclass that accepts text instead of audio.

    Bypasses AudioCapture, ASR, TTS, and AudioOutput entirely.  The real
    LLM, AVCT prompt assembly, conversation history, end-signal detection,
    and session logging all work identically to production.
    """

    def __init__(
        self,
        llm: LLMEngine,
        avct_manager: AvctManager,
        session_logger: SessionLogger,
    ) -> None:
        """Initialise with only the components needed for text-mode turns.

        Args:
            llm: Real LLM engine for generating responses.
            avct_manager: AVCT prompt assembler and risk rater.
            session_logger: SQLite logger for sessions and turns.
        """
        super().__init__(
            audio_capture=None,  # type: ignore[arg-type]
            asr=None,            # type: ignore[arg-type]
            llm=llm,
            tts=None,            # type: ignore[arg-type]
            audio_output=_SilentAudioOutput(),
            avct_manager=avct_manager,
            session_logger=session_logger,
            nao_adapter=_NullNAO(),
        )

    # ---- public API -------------------------------------------------------

    def run_text_turn(self, text: str) -> TurnResult:
        """Execute one conversation turn using *text* as the user utterance.

        Mirrors the production run_turn() pipeline (manager.py lines 144-234)
        but skips audio capture, ASR, TTS, and audio playback.

        Args:
            text: The user utterance to inject.

        Returns:
            A fully populated TurnResult with user_audio=None and
            tts_result=None.
        """
        self._turn_count += 1
        latency: dict[str, int] = {}

        # ---- LLM generation (matches manager.py lines 164-176) -----------
        system_prompt = self._avct.get_system_prompt(
            self._session_id,
            self._polar_level,
            self._category,
            self._subtype,
            self._modifiers,
        )

        self._history.add_user_message(text)

        # Snapshot history BEFORE the assistant reply is appended
        conversation_history = self._history.get_messages()

        t0 = time.monotonic()
        try:
            llm_result = self._llm.generate(system_prompt, conversation_history)
        except Exception as exc:
            log.warning("LLM error during text turn: %s", exc)
            llm_result = LLMResult(
                text="I see. Go on.",
                model="fallback",
                total_tokens=0,
                generation_time_seconds=time.monotonic() - t0,
            )
        llm_ms = round((time.monotonic() - t0) * 1000)

        # ---- End signal & risk (matches manager.py lines 182-186) ---------
        response_text, end_detected = extract_end_signal(llm_result.text)
        if end_detected:
            self._end_requested = True

        risk_rating = self._avct.get_risk_rating(
            self._polar_level, self._category, self._subtype, self._modifiers,
        )

        # ---- History update (matches manager.py line 206) -----------------
        self._history.add_assistant_message(response_text)

        # ---- Build synthetic ASR result for the logger --------------------
        asr_result = ASRResult(
            text=text,
            language="en",
            confidence=1.0,
            transcription_time_seconds=0.0,
        )

        # ---- Latency dict -------------------------------------------------
        latency["vad_ms"] = 0
        latency["asr_ms"] = 0
        latency["llm_ms"] = llm_ms
        latency["tts_ms"] = 0
        latency["total_ms"] = llm_ms

        # ---- TurnResult (user_audio=None, tts_result=None) ----------------
        timestamp = datetime.now(timezone.utc).isoformat()
        turn_result = TurnResult(
            turn_number=self._turn_count,
            user_audio=None,
            transcript=text,
            llm_response=response_text,
            tts_result=None,
            polar_level=self._polar_level,
            category=self._category,
            subtype=self._subtype,
            modifiers=list(self._modifiers),
            risk_rating=risk_rating,
            latency=latency,
            timestamp=timestamp,
        )

        # ---- Log identically to production (manager.py lines 224-231) -----
        self._logger.log_turn(
            session_id=self._session_id,
            turn=turn_result,
            asr_result=asr_result,
            llm_result=llm_result,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
        )

        return turn_result

    def update_avct(
        self,
        polar_level: int,
        category: str,
        subtype: int,
        modifiers: list,
    ) -> None:
        """Update AVCT parameters mid-session.

        Delegates to the existing set_avct() on ConversationManager which
        handles clamping polar_level to [-3, 3].

        Args:
            polar_level: Antagonism intensity (-3 to +3).
            category: Behavioral category (B-G).
            subtype: Intensity variant (1-3).
            modifiers: Behavioral overlays (M1-M6).
        """
        self.set_avct(polar_level, category, subtype, modifiers)
