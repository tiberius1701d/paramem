"""Unit tests for the keyed_pairs.json I/O facade.

Covers the round-trip contract, schema enforcement, encryption-awareness,
and edge cases (empty list, missing file, non-list payload).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paramem.backup.age_envelope import AGE_MAGIC
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.training.keyed_pairs_io import (
    KEYED_PAIR_FIELDS,
    _normalise_pair,
    read_keyed_pairs,
    write_keyed_pairs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_PAIR: dict = {
    "key": "graph1",
    "question": "Where does Alice live?",
    "answer": "Alice lives in Berlin.",
    "source_subject": "Alice",
    "source_predicate": "lives_in",
    "source_object": "Berlin",
    "speaker_id": "Speaker0",
    "first_seen_cycle": 1,
}


def _make_pair(**overrides) -> dict:
    """Return a copy of _FULL_PAIR with the given fields overridden."""
    p = dict(_FULL_PAIR)
    p.update(overrides)
    return p


def _setup_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point env + module default at it."""
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    """Isolate daily identity cache per test so encryption state is predictable."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# _normalise_pair
# ---------------------------------------------------------------------------


class TestNormalisePair:
    def test_returns_all_canonical_fields(self):
        result = _normalise_pair(_FULL_PAIR)
        assert list(result.keys()) == list(KEYED_PAIR_FIELDS)

    def test_drops_extra_field(self):
        src = dict(_FULL_PAIR)
        src["extra_debug_field"] = "should not appear"
        result = _normalise_pair(src)
        assert "extra_debug_field" not in result

    def test_raises_key_error_on_missing_field(self):
        src = dict(_FULL_PAIR)
        del src["source_predicate"]
        with pytest.raises(KeyError):
            _normalise_pair(src)

    def test_raises_key_error_on_missing_first_seen_cycle(self):
        src = dict(_FULL_PAIR)
        del src["first_seen_cycle"]
        with pytest.raises(KeyError):
            _normalise_pair(src)


# ---------------------------------------------------------------------------
# write_keyed_pairs / read_keyed_pairs — round-trip
# ---------------------------------------------------------------------------


class TestWriteReadRoundTrip:
    def test_round_trip_single_pair(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])
        result = read_keyed_pairs(path)
        assert len(result) == 1
        assert result[0] == dict(_FULL_PAIR)

    def test_round_trip_multiple_pairs(self, tmp_path):
        pairs = [
            _make_pair(key="graph1", source_predicate="lives_in", first_seen_cycle=1),
            _make_pair(
                key="graph2",
                question="What is Bob's job?",
                answer="Bob is an engineer.",
                source_subject="Bob",
                source_predicate="has_job",
                source_object="engineer",
                speaker_id="Speaker1",
                first_seen_cycle=2,
            ),
        ]
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, pairs)
        result = read_keyed_pairs(path)
        assert len(result) == 2
        assert result[0]["key"] == "graph1"
        assert result[1]["key"] == "graph2"

    def test_round_trip_empty_list(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [])
        result = read_keyed_pairs(path)
        assert result == []

    def test_write_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])
        assert path.exists()

    def test_extra_fields_stripped_on_write(self, tmp_path):
        pair = dict(_FULL_PAIR)
        pair["debug_tag"] = "should_not_appear"
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [pair])
        result = read_keyed_pairs(path)
        assert "debug_tag" not in result[0]

    def test_field_values_preserved_exactly(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])
        result = read_keyed_pairs(path)
        for field in KEYED_PAIR_FIELDS:
            assert result[0][field] == _FULL_PAIR[field]

    def test_first_seen_cycle_is_int(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])
        result = read_keyed_pairs(path)
        assert isinstance(result[0]["first_seen_cycle"], int)


# ---------------------------------------------------------------------------
# write_keyed_pairs — schema enforcement
# ---------------------------------------------------------------------------


class TestWriteSchemaEnforcement:
    @pytest.mark.parametrize("missing_field", KEYED_PAIR_FIELDS)
    def test_raises_key_error_for_missing_field(self, tmp_path, missing_field):
        pair = dict(_FULL_PAIR)
        del pair[missing_field]
        path = tmp_path / "keyed_pairs.json"
        with pytest.raises(KeyError):
            write_keyed_pairs(path, [pair])

    def test_raises_key_error_on_empty_dict(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        with pytest.raises(KeyError):
            write_keyed_pairs(path, [{}])

    def test_accepts_iterable_not_list(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, iter([_FULL_PAIR]))
        assert read_keyed_pairs(path)[0]["key"] == "graph1"


# ---------------------------------------------------------------------------
# read_keyed_pairs — edge cases
# ---------------------------------------------------------------------------


class TestReadEdgeCases:
    def test_missing_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "nonexistent_keyed_pairs.json"
        result = read_keyed_pairs(path)
        assert result == []

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        path.write_bytes(b"")
        result = read_keyed_pairs(path)
        assert result == []

    def test_non_list_payload_raises_value_error(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        path.write_bytes(json.dumps({"foo": 1}).encode("utf-8"))
        with pytest.raises(ValueError, match="expected JSON array"):
            read_keyed_pairs(path)

    def test_partial_entry_raises_value_error(self, tmp_path):
        """An entry on disk missing a canonical field raises ValueError."""
        path = tmp_path / "keyed_pairs.json"
        # Write directly (bypassing facade) to simulate a pre-facade file
        partial = [{"key": "graph1", "question": "Q?", "answer": "A."}]
        path.write_bytes(json.dumps(partial).encode("utf-8"))
        with pytest.raises(ValueError, match="missing required fields"):
            read_keyed_pairs(path)

    def test_invalid_json_raises_json_decode_error(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        path.write_bytes(b"{not valid json}")
        with pytest.raises(Exception):  # json.JSONDecodeError is a subclass of ValueError
            read_keyed_pairs(path)


# ---------------------------------------------------------------------------
# Encryption awareness
# ---------------------------------------------------------------------------


class TestEncryptionAwareness:
    def test_plaintext_when_no_key_loaded(self, tmp_path, monkeypatch):
        """When no daily identity is loaded, file is written as plaintext."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])

        raw = path.read_bytes()
        assert not raw.startswith(AGE_MAGIC), "expected plaintext, got age envelope"

        # round-trip still works
        result = read_keyed_pairs(path)
        assert result[0]["key"] == "graph1"

    def test_age_envelope_when_daily_loaded(self, tmp_path, monkeypatch):
        """When daily identity is loaded, file is written as age ciphertext."""
        _setup_daily(tmp_path, monkeypatch)

        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])

        raw = path.read_bytes()
        assert raw.startswith(AGE_MAGIC), f"expected age envelope magic, got {raw[:40]!r}"

    def test_encrypted_file_round_trips_correctly(self, tmp_path, monkeypatch):
        """Write with key → read decrypts transparently and returns full schema."""
        _setup_daily(tmp_path, monkeypatch)

        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_PAIR])

        result = read_keyed_pairs(path)
        assert len(result) == 1
        for field in KEYED_PAIR_FIELDS:
            assert result[0][field] == _FULL_PAIR[field]
