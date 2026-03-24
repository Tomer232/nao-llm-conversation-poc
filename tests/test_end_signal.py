import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from antagonist_robot.conversation.manager import extract_end_signal

def test_end_signal_detected():
    """[END] on its own line after text returns cleaned text and True."""
    text = "Goodbye.\n[END]"
    cleaned, detected = extract_end_signal(text)
    assert detected is True, "Should detect [END]"
    assert cleaned == "Goodbye.", f"Expected 'Goodbye.', got '{cleaned}'"
    print("test_end_signal_detected passed!")

def test_end_signal_not_detected():
    """Text without [END] returns original text and False."""
    text = "Hello there."
    cleaned, detected = extract_end_signal(text)
    assert detected is False, "Should not detect [END]"
    assert cleaned == "Hello there.", f"Expected 'Hello there.', got '{cleaned}'"
    print("test_end_signal_not_detected passed!")

def test_end_signal_case_insensitive():
    """[end], [End], and [END] all trigger detection."""
    for token in ["[end]", "[End]", "[END]", "[eNd]"]:
        text = f"Done. {token}"
        cleaned, detected = extract_end_signal(text)
        assert detected is True, f"Should detect {token}"
        assert "[" not in cleaned, f"Token remnant in cleaned text for {token}"
    print("test_end_signal_case_insensitive passed!")

def test_end_signal_strips_whitespace():
    """Trailing whitespace and newlines around [END] are stripped from result."""
    text = "Goodbye.\n[END]\n"
    cleaned, detected = extract_end_signal(text)
    assert detected is True, "Should detect [END]"
    assert cleaned == "Goodbye.", f"Expected 'Goodbye.', got '{cleaned}'"
    print("test_end_signal_strips_whitespace passed!")

def test_end_signal_mid_text():
    """[END] embedded mid-sentence is still detected and removed."""
    text = "I said [END] too soon"
    cleaned, detected = extract_end_signal(text)
    assert detected is True, "Should detect [END] even mid-text"
    assert "[END]" not in cleaned, "Token should be removed"
    assert "said" in cleaned and "too soon" in cleaned, f"Surrounding text should remain, got '{cleaned}'"
    print("test_end_signal_mid_text passed!")

if __name__ == "__main__":
    test_end_signal_detected()
    test_end_signal_not_detected()
    test_end_signal_case_insensitive()
    test_end_signal_strips_whitespace()
    test_end_signal_mid_text()
    print("All end signal tests passed!")
