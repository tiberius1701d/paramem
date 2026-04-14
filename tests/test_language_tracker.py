"""Tests for `paramem/server/language_tracker.py` (F5.7)."""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.server.language_tracker import LanguageTracker


def test_record_below_threshold_ignored(tmp_path):
    tracker = LanguageTracker(tmp_path / "lang.json", min_prob=0.7)
    assert not tracker.record("en", 0.4)
    assert tracker.languages == []


def test_record_accepts_high_confidence(tmp_path):
    tracker = LanguageTracker(tmp_path / "lang.json", min_prob=0.7)
    assert tracker.record("en", 0.95)
    assert "en" in tracker.languages


def test_record_dedup(tmp_path):
    tracker = LanguageTracker(tmp_path / "lang.json", min_prob=0.7)
    tracker.record("en", 0.95)
    assert not tracker.record("en", 0.99)  # already in set
    assert tracker.record("de", 0.85)
    assert tracker.languages == ["de", "en"]


def test_persists_across_instances(tmp_path):
    store = tmp_path / "lang.json"
    tracker_a = LanguageTracker(store, min_prob=0.7)
    tracker_a.record("fr", 0.9)
    tracker_b = LanguageTracker(store, min_prob=0.7)
    assert "fr" in tracker_b.languages


def test_publishes_to_ha_client_on_change(tmp_path):
    client = MagicMock()
    tracker = LanguageTracker(
        tmp_path / "lang.json", ha_client=client, ha_entity_id="input_text.langs"
    )
    tracker.record("en", 0.9)
    client.set_state.assert_called_with("input_text.langs", "en")
    tracker.record("de", 0.9)
    client.set_state.assert_called_with("input_text.langs", "de,en")


def test_republishes_on_load(tmp_path):
    store = tmp_path / "lang.json"
    LanguageTracker(store).record("en", 0.9)
    client = MagicMock()
    # Second instance loads persisted state and republishes on init.
    LanguageTracker(store, ha_client=client)
    client.set_state.assert_called_with("input_text.voice_observed_languages", "en")


def test_empty_language_not_recorded(tmp_path):
    tracker = LanguageTracker(tmp_path / "lang.json")
    assert not tracker.record(None, 0.99)
    assert not tracker.record("", 0.99)
