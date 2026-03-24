import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from antagonist_robot.config.settings import AvctConfig
from antagonist_robot.conversation.avct_manager import AvctManager

def test_avct_safety():
    cfg = AvctConfig()
    mgr = AvctManager(cfg)

    prompt = mgr.get_system_prompt("session_123", polar_level=3, category="G", subtype=3, modifiers=["M4"])

    # Safety boundaries must always be present
    assert "MANDATORY SAFETY BOUNDARIES" in prompt, "Missing safety boundaries!"
    # Verify actual behavioral content — not just labels
    assert "Extreme" in prompt, "Missing category name for G!"
    assert "Maximum possible antagonistic intensity" in prompt, "Missing G3 subtype behavioral description!"
    assert "CONDESCENDING" in prompt, "Missing M4 modifier behavioral description!"
    assert "Maximally antagonistic" in prompt, "Missing polar level 3 description!"

    print("test_avct_safety passed!")

def test_avct_risk():
    cfg = AvctConfig()
    mgr = AvctManager(cfg)

    assert mgr.get_risk_rating(3, "B", 1, []) == "Red", "Risk rating failed for 3B1"
    assert mgr.get_risk_rating(2, "G", 1, []) == "Red", "Risk rating failed for 2G1"
    assert mgr.get_risk_rating(2, "C", 1, []) == "Amber", "Risk rating failed for 2C1"
    assert mgr.get_risk_rating(0, "D", 1, []) == "Green", "Risk rating failed for 0D1"

    print("test_avct_risk passed!")

def test_avct_negative_polar():
    """Verify anti-polar (supportive) prompts contain supportive content, not category behavior."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)
    prompt = mgr.get_system_prompt("session_neg", polar_level=-2, category="D", subtype=2, modifiers=[])

    assert "MANDATORY SAFETY BOUNDARIES" in prompt, "Missing safety boundaries!"
    # Anti-polar supportive behavior should appear
    assert "supportive" in prompt.lower(), "Missing anti-polar supportive description!"
    assert "Moderately supportive" in prompt, "Missing polar level -2 description!"
    # Category antagonistic descriptions should NOT appear for negative polar
    assert "Confrontational" not in prompt, "Antagonistic category should not appear in anti-polar mode!"

    print("test_avct_negative_polar passed!")

def test_avct_modifier_content():
    """Verify that full modifier behavioral descriptions appear in the prompt."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)
    prompt = mgr.get_system_prompt("session_mod", polar_level=1, category="C", subtype=1, modifiers=["M1", "M6"])

    assert "INTERRUPTING" in prompt, "M1 modifier description missing!"
    assert "Cut the speaker off" in prompt, "M1 behavioral detail missing!"
    assert "SILENT TREATMENT" in prompt, "M6 modifier description missing!"
    assert "1 to 5 words" in prompt, "M6 behavioral detail missing!"
    # Modifiers NOT in the list should NOT appear
    assert "CONDESCENDING" not in prompt, "M4 should not appear when not in modifiers list!"

    print("test_avct_modifier_content passed!")

def test_avct_neutral_polar():
    """Verify polar_level=0 produces neutral prompt without category behavior."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)
    prompt = mgr.get_system_prompt("session_zero", polar_level=0, category="D", subtype=2, modifiers=[])

    assert "MANDATORY SAFETY BOUNDARIES" in prompt, "Missing safety boundaries!"
    assert "Neutral" in prompt, "Missing neutral polar description!"
    # Slot 2 should be omitted — no category behavior at neutral
    assert "Confrontational" not in prompt, "Category should not appear at neutral polar level!"

    print("test_avct_neutral_polar passed!")

def test_avct_slot6_persona():
    """Verify Slot 6 persona/voice instructions are present in every prompt."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)
    prompt = mgr.get_system_prompt("session_voice", polar_level=1, category="B", subtype=1, modifiers=[])

    assert "spoken dialogue" in prompt.lower(), "Missing persona voice instructions!"
    assert "markdown" in prompt.lower(), "Persona should instruct against markdown!"
    assert "Slot 6" in prompt, "Slot 6 marker missing from prompt!"

    print("test_avct_slot6_persona passed!")

def test_avct_polar_minus3():
    """Verify polar_level=-3 produces maximum supportive content with no antagonistic language."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)
    prompt = mgr.get_system_prompt("session_m3", polar_level=-3, category="D", subtype=2, modifiers=[])

    assert "MANDATORY SAFETY BOUNDARIES" in prompt, "Missing safety boundaries!"
    assert "Maximally supportive" in prompt, "Missing POLAR_DESCRIPTIONS[-3] content!"
    assert "maximally supportive and affirming" in prompt.lower(), "Missing ANTI_POLAR_DEFINITIONS[-3] content!"
    assert "Confrontational" not in prompt, "Antagonistic category should not appear at polar -3!"

    print("test_avct_polar_minus3 passed!")

def test_avct_polar_minus1():
    """Verify polar_level=-1 produces mildly supportive content."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)
    prompt = mgr.get_system_prompt("session_m1", polar_level=-1, category="F", subtype=1, modifiers=[])

    assert "Mildly supportive" in prompt, "Missing POLAR_DESCRIPTIONS[-1] content!"
    assert "gently supportive" in prompt.lower(), "Missing ANTI_POLAR_DEFINITIONS[-1] content!"
    assert "Aggressive" not in prompt, "Antagonistic category should not appear at polar -1!"

    print("test_avct_polar_minus1 passed!")

def test_avct_risk_negative_polar():
    """Verify get_risk_rating() returns Green for all negative polar levels."""
    cfg = AvctConfig()
    mgr = AvctManager(cfg)

    # Even with extreme categories and subtypes, negative polar is always Green
    assert mgr.get_risk_rating(-3, "G", 3, []) == "Green", "Risk should be Green at polar -3 even with category G"
    assert mgr.get_risk_rating(-2, "F", 3, []) == "Green", "Risk should be Green at polar -2"
    assert mgr.get_risk_rating(-1, "D", 1, []) == "Green", "Risk should be Green at polar -1"

    print("test_avct_risk_negative_polar passed!")

if __name__ == "__main__":
    test_avct_safety()
    test_avct_risk()
    test_avct_negative_polar()
    test_avct_modifier_content()
    test_avct_neutral_polar()
    test_avct_slot6_persona()
    test_avct_polar_minus3()
    test_avct_polar_minus1()
    test_avct_risk_negative_polar()
    print("All AVCT tests passed!")
