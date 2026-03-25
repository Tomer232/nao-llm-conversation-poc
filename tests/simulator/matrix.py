"""Test matrix loader — reads test_cases.yaml and provides filtered access.

Loads the YAML file containing reusable participant dialogue scripts and
hand-crafted test cases with AVCT parameters, behavioral expectations,
and rubrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Contradictory modifier pairs
# ---------------------------------------------------------------------------

CONTRADICTORY_MODIFIER_PAIRS = {
    frozenset({"M1", "M6"}),   # Interrupting + Silent Treatment:
                               # M1 requires taking the floor to speak;
                               # M6 requires near-silence. Mutually exclusive.
}

_MODIFIER_NAMES = {
    "M1": "Interrupting",
    "M2": "Gaslighting",
    "M3": "Deflecting",
    "M4": "Condescending",
    "M5": "Threatening",
    "M6": "Silent Treatment",
}

_CONTRADICTION_REASONS = {
    frozenset({"M1", "M6"}): (
        "contradictory — interrupting requires taking the "
        "conversational floor; silent treatment requires near-silence."
    ),
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single test case from the YAML matrix.

    Attributes:
        case_id: Unique identifier for this test case.
        polar_level: Antagonism intensity (-3 to +3).
        category: Behavioral category (B-G).
        subtype: Intensity variant within category (1-3).
        modifiers: Behavioral overlays (M1-M6).
        script_key: Key into the scripts section of the YAML.
        tier: Testing depth — "smoke", "deep", or "both".
        description: Human-readable one-line summary.
        behavioural_expectation: Prose describing the expected behavior.
        rubric: Evaluation criteria with must_show, must_not_show,
            and intensity_range keys.
    """
    case_id: str
    polar_level: int
    category: str
    subtype: int
    modifiers: List[str]
    script_key: str
    tier: str
    description: str
    behavioural_expectation: str
    rubric: Dict


# ---------------------------------------------------------------------------
# TestMatrix
# ---------------------------------------------------------------------------

class TestMatrix:
    """Loads and provides access to test cases and scripts from YAML.

    Usage::

        matrix = TestMatrix()
        scripts = matrix.get_scripts()         # {key: [turn_text, ...]}
        cases  = matrix.get_cases(tier="smoke") # filtered TestCase list
    """

    def __init__(self, yaml_path: Optional[Path] = None) -> None:
        """Load the test matrix from a YAML file.

        Args:
            yaml_path: Path to the YAML file.  Defaults to
                ``tests/test_cases.yaml`` resolved relative to this file.
        """
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / "test_cases.yaml"
        else:
            yaml_path = Path(yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        self._scripts: Dict[str, Dict] = raw.get("scripts", {})
        self._raw_cases: List[Dict] = raw.get("test_cases", [])

        # Pre-parse all cases into dataclass instances
        self._cases: List[TestCase] = [
            TestCase(
                case_id=c["case_id"],
                polar_level=c["polar_level"],
                category=c["category"],
                subtype=c["subtype"],
                modifiers=c.get("modifiers", []),
                script_key=c["script"],
                tier=c["tier"],
                description=c["description"],
                behavioural_expectation=c["behavioural_expectation"],
                rubric=c["rubric"],
            )
            for c in self._raw_cases
        ]

    def get_scripts(self) -> Dict[str, List[str]]:
        """Return all scripts as ``{script_key: [turn_text, ...]}``.

        Each script entry in the YAML has a ``turns`` list — this method
        returns just those turn strings, keyed by script name.
        """
        return {
            key: script_data["turns"]
            for key, script_data in self._scripts.items()
        }

    def get_cases(self, tier: Optional[str] = None) -> List[TestCase]:
        """Return test cases, optionally filtered by tier.

        Args:
            tier: If ``"smoke"``, return cases where tier is "smoke" or
                "both".  If ``"deep"``, return cases where tier is "deep"
                or "both".  If ``None``, return all cases.

        Returns:
            A list of matching TestCase instances.
        """
        if tier is None:
            return list(self._cases)

        if tier == "smoke":
            return [c for c in self._cases if c.tier in ("smoke", "both")]

        if tier == "deep":
            return [c for c in self._cases if c.tier in ("deep", "both")]

        # Unknown tier — return empty rather than crash
        return []

    def get_contradictions(self, case: TestCase) -> List[str]:
        """Return human-readable conflict descriptions for a case.

        Checks the case's modifiers against CONTRADICTORY_MODIFIER_PAIRS.
        Returns an empty list if no conflicts exist.

        Args:
            case: The TestCase to check for modifier contradictions.

        Returns:
            A list of description strings, one per conflicting pair found.
        """
        if len(case.modifiers) < 2:
            return []

        conflicts: List[str] = []
        modifier_set = set(case.modifiers)

        for pair in CONTRADICTORY_MODIFIER_PAIRS:
            if pair.issubset(modifier_set):
                sorted_pair = sorted(pair)
                m_a, m_b = sorted_pair
                name_a = _MODIFIER_NAMES.get(m_a, m_a)
                name_b = _MODIFIER_NAMES.get(m_b, m_b)
                reason = _CONTRADICTION_REASONS.get(pair, "contradictory.")
                conflicts.append(
                    f"{m_a} ({name_a}) + {m_b} ({name_b}): {reason}"
                )

        return conflicts
