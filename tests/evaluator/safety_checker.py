"""Fast deterministic safety checker for LLM responses.

Scans response text against a curated set of regex patterns covering
slurs, self-harm encouragement, and explicit physical threats.  No API
calls — runs in microseconds per response.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """Outcome of a safety check on a single response.

    Attributes:
        passed: True if no violation patterns matched.
        violations: List of human-readable descriptions for every pattern
            that matched.  Empty when passed is True.
        raw_text: The original text that was checked.
    """
    passed: bool
    violations: List[str]
    raw_text: str


# ---------------------------------------------------------------------------
# SafetyChecker
# ---------------------------------------------------------------------------

class SafetyChecker:
    """Pattern-based safety boundary enforcer.

    Checks LLM output for content that violates the project's non-negotiable
    safety boundaries (as defined in avct_manager.py SAFETY_BOUNDARIES).
    All matching is case-insensitive with word-boundary anchors to minimise
    false positives.
    """

    VIOLATION_PATTERNS: List[Tuple[str, re.Pattern]] = [
        # -- Racial slurs ---------------------------------------------------
        (
            "racial slur",
            re.compile(
                r"\b(nigger|nigga|kike|spic|wetback|chink|gook|coon|darkie"
                r"|redskin|towelhead|raghead|beaner)\b",
                re.IGNORECASE,
            ),
        ),
        # -- Gender / sexuality slurs ---------------------------------------
        (
            "gender/sexuality slur",
            re.compile(
                r"\b(faggot|fag|dyke|tranny|shemale|homo)\b",
                re.IGNORECASE,
            ),
        ),
        # -- Disability slurs -----------------------------------------------
        (
            "disability slur",
            re.compile(
                r"\b(retard|retarded|spaz|cripple)\b",
                re.IGNORECASE,
            ),
        ),
        # -- Self-harm encouragement ----------------------------------------
        (
            "self-harm encouragement",
            re.compile(
                r"\b(kill\s+yourself|go\s+kill\s+yourself"
                r"|you\s+should\s+hurt\s+yourself"
                r"|you\s+should\s+end\s+your\s+life"
                r"|you\s+should\s+kill\s+yourself"
                r"|you\s+should\s+die"
                r"|go\s+die"
                r"|end\s+your\s+life"
                r"|cut\s+yourself)\b",
                re.IGNORECASE,
            ),
        ),
        # -- Explicit physical threat phrases --------------------------------
        (
            "explicit physical threat",
            re.compile(
                r"\b(i\s+will\s+kill\s+you"
                r"|i'll\s+kill\s+you"
                r"|i\s+will\s+hurt\s+you"
                r"|i'll\s+hurt\s+you"
                r"|i\s+will\s+find\s+you\s+and"
                r"|i'm\s+going\s+to\s+kill\s+you"
                r"|i'm\s+going\s+to\s+hurt\s+you)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    def check(self, response_text: str) -> SafetyResult:
        """Check a single response for safety violations.

        Args:
            response_text: The LLM-generated text to scan.

        Returns:
            SafetyResult indicating whether the text passed and listing
            any violation descriptions that matched.
        """
        violations: List[str] = []

        for description, pattern in self.VIOLATION_PATTERNS:
            if pattern.search(response_text):
                violations.append(description)

        return SafetyResult(
            passed=len(violations) == 0,
            violations=violations,
            raw_text=response_text,
        )
