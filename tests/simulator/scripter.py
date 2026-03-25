"""Scripted conversation runner that feeds pre-written turns into TextInjector.

Drives a multi-turn conversation by iterating over a list of ScriptedTurns,
optionally overriding AVCT parameters between turns, and collecting the
resulting TurnResults into a SimulatedSession.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from antagonist_robot.pipeline.types import TurnResult
from tests.simulator.text_injector import TextInjector


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScriptedTurn:
    """A single scripted user utterance with optional AVCT overrides.

    Attributes:
        user_text: The text to inject as the user utterance.
        avct_override: Optional dict with keys polar_level, category,
            subtype, modifiers.  Only provided keys are changed; the rest
            keep their current session values.  None means keep everything.
    """
    user_text: str
    avct_override: Optional[Dict] = None


@dataclass
class SimulatedSession:
    """Complete result of running a scripted conversation.

    Attributes:
        case_id: Identifier of the test case that produced this session.
        session_id: UUID assigned by ConversationManager.start_session().
        turns: Ordered list of TurnResults from each scripted turn.
        avct_params: The initial AVCT parameters used for the session.
        elapsed_seconds: Wall-clock duration of the full session.
    """
    case_id: str
    session_id: str
    turns: List[TurnResult]
    avct_params: Dict
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# ConversationScripter
# ---------------------------------------------------------------------------

class ConversationScripter:
    """Runs a list of scripted turns through a TextInjector.

    Usage::

        scripter = ConversationScripter(injector)
        session = scripter.run(
            case_id="B1_polar1",
            polar_level=1, category="B", subtype=1, modifiers=[],
            script=[ScriptedTurn("Hello"), ScriptedTurn("How are you?")],
        )
    """

    def __init__(self, injector: TextInjector) -> None:
        """Initialise with a TextInjector instance.

        Args:
            injector: The text-mode conversation manager to drive.
        """
        self._injector = injector

    def run(
        self,
        case_id: str,
        polar_level: int,
        category: str,
        subtype: int,
        modifiers: list,
        script: List[ScriptedTurn],
        participant_id: str = "simulator",
    ) -> SimulatedSession:
        """Execute a full scripted conversation and return the results.

        Args:
            case_id: Test case identifier (for traceability).
            polar_level: Initial antagonism intensity (-3 to +3).
            category: Initial behavioral category (B-G).
            subtype: Initial intensity variant (1-3).
            modifiers: Initial behavioral overlays (M1-M6).
            script: Ordered list of ScriptedTurns to inject.
            participant_id: Identifier for the simulated participant.

        Returns:
            A SimulatedSession containing all collected TurnResults.
        """
        t_start = time.monotonic()

        session_id = self._injector.start_session(
            polar_level=polar_level,
            category=category,
            subtype=subtype,
            modifiers=modifiers,
            participant_id=participant_id,
        )

        avct_params = {
            "polar_level": polar_level,
            "category": category,
            "subtype": subtype,
            "modifiers": list(modifiers),
        }

        turns: List[TurnResult] = []

        for scripted_turn in script:
            # Apply mid-conversation AVCT overrides if specified
            if scripted_turn.avct_override is not None:
                merged_polar = scripted_turn.avct_override.get(
                    "polar_level", self._injector.polar_level,
                )
                merged_category = scripted_turn.avct_override.get(
                    "category", self._injector._category,
                )
                merged_subtype = scripted_turn.avct_override.get(
                    "subtype", self._injector._subtype,
                )
                merged_modifiers = scripted_turn.avct_override.get(
                    "modifiers", list(self._injector._modifiers),
                )
                self._injector.update_avct(
                    polar_level=merged_polar,
                    category=merged_category,
                    subtype=merged_subtype,
                    modifiers=merged_modifiers,
                )

            turn_result = self._injector.run_text_turn(scripted_turn.user_text)
            turns.append(turn_result)

        self._injector.end_session()

        elapsed = round(time.monotonic() - t_start, 3)

        return SimulatedSession(
            case_id=case_id,
            session_id=session_id,
            turns=turns,
            avct_params=avct_params,
            elapsed_seconds=elapsed,
        )
