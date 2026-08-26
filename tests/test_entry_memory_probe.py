"""Tests for entry_fact_text.

Covers:
- entry_fact_text: SPO assembly at the render boundary — the stored predicate
  identity form is already space-form prose ('_' was folded to a space by
  canonical() upstream; '-' survives). No further substitution happens here.

All tests are CPU-only.
The QA shape lives in ``archive/legacy_qa.py``; its probe_key format is not
covered here.
``build_memory_source``'s mode → MemorySource selection contract is covered
in ``tests/test_mode_fork_guard.py`` and ``tests/test_server.py``, not here.
"""

from __future__ import annotations

from paramem.memory.entry import entry_fact_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_pair(
    key: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    speaker_id: str = "spk1",
) -> dict:
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
    }


# ---------------------------------------------------------------------------
# entry_fact_text
# ---------------------------------------------------------------------------


class TestEntryFactText:
    def test_basic_assembly(self) -> None:
        """The stored identity predicate (already space-form) is used as-is."""
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "lives in", "object": "Heilbronn"}
        )
        assert result == "Alex lives in Heilbronn"

    def test_single_word_predicate(self) -> None:
        result = entry_fact_text({"subject": "Alex", "predicate": "knows", "object": "Bob"})
        assert result == "Alex knows Bob"

    def test_multi_word_predicate_passthrough(self) -> None:
        """A multi-word identity-form predicate passes through unchanged."""
        result = entry_fact_text(
            {"subject": "Alice", "predicate": "works at company", "object": "Acme"}
        )
        assert result == "Alice works at company Acme"

    def test_hyphenated_predicate_preserved(self) -> None:
        """``-`` is not a blank in the identity form, so it survives the render."""
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "has sister-in-law", "object": "Mia"}
        )
        assert result == "Alex has sister-in-law Mia"

    def test_non_ascii_object_rendered(self) -> None:
        """Non-ASCII object surfaces pass through the render untouched."""
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "has key achievement", "object": "€4.5B sales"}
        )
        assert result == "Alex has key achievement €4.5B sales"

    def test_render_is_prose_not_canonical(self) -> None:
        """The render boundary is a distinct layer from the identity form."""
        from paramem.utils.identity import canonical

        pred = canonical("has hobby")
        assert pred == "has hobby"
        result = entry_fact_text({"subject": "Alex", "predicate": pred, "object": "chess"})
        assert result == "Alex has hobby chess"

    # --- Speaker tokens render verbatim: resolution moved to the reply
    # boundary (paramem.server.speaker.resolve_speaker_tokens); entry_fact_text
    # takes no resolver of its own — every model-facing surface stays in
    # token space. ---

    def test_speaker_token_subject_rendered_verbatim(self) -> None:
        """A speaker{N} token in subject position is emitted as-is — no resolution
        happens at the fact-render boundary."""
        result = entry_fact_text(
            {"subject": "speaker9", "predicate": "lives in", "object": "Berlin"}
        )
        assert result == "speaker9 lives in Berlin"

    def test_speaker_token_object_rendered_verbatim(self) -> None:
        """A speaker{N} token in object position is emitted as-is."""
        result = entry_fact_text(
            {"subject": "speaker0", "predicate": "knows", "object": "speaker9"}
        )
        assert result == "speaker0 knows speaker9"

    def test_no_resolve_parameter_accepted(self) -> None:
        """entry_fact_text takes no resolver parameter — passing one raises."""
        import pytest

        with pytest.raises(TypeError):
            entry_fact_text(
                {"subject": "Alex", "predicate": "knows", "object": "Bob"},
                resolve=lambda t: t,
            )
