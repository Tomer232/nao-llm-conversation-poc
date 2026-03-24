"""Tests for conversation history management."""

import pytest

from antagonist_robot.conversation.history import ConversationHistory


class TestConversationHistory:
    """Tests for the ConversationHistory class."""

    def test_add_user_message(self):
        """Adding a user message appends to history."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_add_assistant_message(self):
        """Adding an assistant message appends to history."""
        history = ConversationHistory()
        history.add_assistant_message("Hi there")
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0] == {"role": "assistant", "content": "Hi there"}

    def test_multi_turn_history(self):
        """Multiple turns build up correctly."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi")
        history.add_user_message("How are you?")
        history.add_assistant_message("Fine")

        messages = history.get_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_clear(self):
        """clear() empties the history."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi")
        history.clear()
        assert history.get_messages() == []

    def test_get_messages_returns_copy(self):
        """get_messages returns a copy, not a reference."""
        history = ConversationHistory()
        history.add_user_message("Hello")
        messages = history.get_messages()
        messages.append({"role": "user", "content": "Injected"})
        assert len(history.get_messages()) == 1  # original unchanged

    def test_truncation_drops_oldest(self):
        """When token limit exceeded, oldest turns are dropped."""
        # Use a very small token limit
        history = ConversationHistory(max_tokens=20)

        # Add enough messages to exceed the limit
        history.add_user_message("a " * 10)  # ~13 tokens
        history.add_assistant_message("b " * 10)  # ~13 tokens
        history.add_user_message("c " * 10)  # ~13 tokens
        history.add_assistant_message("d " * 10)  # ~13 tokens

        messages = history.get_messages()
        # Should have truncated some older messages
        assert len(messages) < 4

    def test_first_turn_preserved_during_truncation(self):
        """The first message is always preserved during truncation."""
        history = ConversationHistory(max_tokens=30)

        history.add_user_message("first message here")
        history.add_assistant_message("first response")
        history.add_user_message("second " * 20)
        history.add_assistant_message("second response " * 20)
        history.add_user_message("third " * 20)

        messages = history.get_messages()
        # First message should still be there
        assert messages[0]["content"] == "first message here"

    def test_token_estimation_reasonable(self):
        """Token estimation is in a reasonable range."""
        history = ConversationHistory()
        # "hello world" = 2 words * 1.3 = ~2.6 tokens
        history.add_user_message("hello world")
        tokens = history._estimate_tokens()
        assert 2 <= tokens <= 4

    def test_truncation_non_alternating(self):
        """Non-alternating messages (two user msgs in a row) don't break truncation."""
        history = ConversationHistory(max_tokens=20)

        history.add_user_message("first " * 5)
        # Two consecutive user messages (simulates ASR fallback)
        history.add_user_message("second " * 5)
        history.add_user_message("third " * 5)
        history.add_assistant_message("response " * 5)
        history.add_user_message("fourth " * 5)

        messages = history.get_messages()
        # Should not crash or infinite-loop, and first message preserved
        assert len(messages) >= 1
        assert messages[0]["content"].startswith("first")

    def test_truncation_preserves_minimum(self):
        """A single oversized message doesn't reduce the list below 1 item."""
        history = ConversationHistory(max_tokens=5)

        # One message that far exceeds the token limit
        history.add_user_message("word " * 100)

        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "word " * 100

    def test_truncation_only_complete_pairs_dropped(self):
        """After truncation, messages[0] always has the original first message."""
        history = ConversationHistory(max_tokens=30)

        history.add_user_message("original first message")
        history.add_assistant_message("first reply")
        history.add_user_message("big message " * 20)
        history.add_assistant_message("big reply " * 20)
        history.add_user_message("latest question")

        messages = history.get_messages()
        assert messages[0]["content"] == "original first message"

    def test_empty_history_returns_empty_list(self):
        """A brand-new history returns an empty list."""
        history = ConversationHistory()
        assert history.get_messages() == []

    def test_truncation_on_empty_no_crash(self):
        """Calling _truncate_if_needed on an empty history does not crash."""
        history = ConversationHistory(max_tokens=5)
        history._truncate_if_needed()
        assert history.get_messages() == []
