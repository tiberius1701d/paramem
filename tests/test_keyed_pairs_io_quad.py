"""Unit tests for the quadruple-format keyed_pairs.json I/O facade.

Mirrors the structure of ``tests/test_keyed_pairs_io.py`` for the quad schema.
Covers round-trip contract, schema enforcement, back-compat legacy-QA projection,
encryption awareness, and edge cases.
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
    KEYED_PAIR_FIELDS_QUAD,
    _normalise_pair_quad,
    read_keyed_pairs_quad,
    write_keyed_pairs,
    write_keyed_pairs_quad,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_QUAD: dict = {
    "key": "graph1",
    "subject": "Alice",
    "predicate": "lives_in",
    "object": "Berlin",
    "speaker_id": "Speaker0",
    "first_seen_cycle": 1,
}

# A legacy QA-format pair (8-field QA schema) used for back-compat tests.
_FULL_QA_PAIR: dict = {
    "key": "graph1",
    "question": "Where does Alice live?",
    "answer": "Alice lives in Berlin.",
    "source_subject": "Alice",
    "source_predicate": "lives_in",
    "source_object": "Berlin",
    "speaker_id": "Speaker0",
    "first_seen_cycle": 1,
}


def _make_quad(**overrides) -> dict:
    """Return a copy of _FULL_QUAD with the given fields overridden."""
    p = dict(_FULL_QUAD)
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
# _normalise_pair_quad
# ---------------------------------------------------------------------------


class TestNormalisePairQuad:
    def test_returns_all_canonical_fields(self):
        result = _normalise_pair_quad(_FULL_QUAD)
        assert list(result.keys()) == list(KEYED_PAIR_FIELDS_QUAD)

    def test_drops_extra_field(self):
        src = dict(_FULL_QUAD)
        src["extra_debug_field"] = "should not appear"
        result = _normalise_pair_quad(src)
        assert "extra_debug_field" not in result

    def test_raises_key_error_on_missing_subject(self):
        src = dict(_FULL_QUAD)
        del src["subject"]
        with pytest.raises(KeyError):
            _normalise_pair_quad(src)

    def test_raises_key_error_on_missing_first_seen_cycle(self):
        src = dict(_FULL_QUAD)
        del src["first_seen_cycle"]
        with pytest.raises(KeyError):
            _normalise_pair_quad(src)

    def test_raises_key_error_on_missing_object(self):
        src = dict(_FULL_QUAD)
        del src["object"]
        with pytest.raises(KeyError):
            _normalise_pair_quad(src)


# ---------------------------------------------------------------------------
# write_keyed_pairs_quad / read_keyed_pairs_quad — round-trip
# ---------------------------------------------------------------------------


class TestWriteReadQuadRoundTrip:
    def test_round_trip_single_quad(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])
        result = read_keyed_pairs_quad(path)
        assert len(result) == 1
        assert result[0] == dict(_FULL_QUAD)

    def test_round_trip_multiple_quads(self, tmp_path):
        quads = [
            _make_quad(key="graph1", subject="Alice", predicate="lives_in", object="Berlin"),
            _make_quad(
                key="graph2",
                subject="Bob",
                predicate="has_job",
                object="engineer",
                speaker_id="Speaker1",
                first_seen_cycle=2,
            ),
        ]
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, quads)
        result = read_keyed_pairs_quad(path)
        assert len(result) == 2
        assert result[0]["key"] == "graph1"
        assert result[1]["key"] == "graph2"

    def test_round_trip_empty_list(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [])
        result = read_keyed_pairs_quad(path)
        assert result == []

    def test_write_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])
        assert path.exists()

    def test_extra_fields_stripped_on_write(self, tmp_path):
        quad = dict(_FULL_QUAD)
        quad["debug_tag"] = "should_not_appear"
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [quad])
        result = read_keyed_pairs_quad(path)
        assert "debug_tag" not in result[0]

    def test_field_values_preserved_exactly(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])
        result = read_keyed_pairs_quad(path)
        for field in KEYED_PAIR_FIELDS_QUAD:
            assert result[0][field] == _FULL_QUAD[field]

    def test_first_seen_cycle_is_int(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])
        result = read_keyed_pairs_quad(path)
        assert isinstance(result[0]["first_seen_cycle"], int)

    def test_accepts_iterable_not_list(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, iter([_FULL_QUAD]))
        assert read_keyed_pairs_quad(path)[0]["key"] == "graph1"


# ---------------------------------------------------------------------------
# write_keyed_pairs_quad — schema enforcement
# ---------------------------------------------------------------------------


class TestWriteQuadSchemaEnforcement:
    @pytest.mark.parametrize("missing_field", KEYED_PAIR_FIELDS_QUAD)
    def test_raises_key_error_for_missing_field(self, tmp_path, missing_field):
        quad = dict(_FULL_QUAD)
        del quad[missing_field]
        path = tmp_path / "keyed_pairs.json"
        with pytest.raises(KeyError):
            write_keyed_pairs_quad(path, [quad])

    def test_raises_key_error_on_empty_dict(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        with pytest.raises(KeyError):
            write_keyed_pairs_quad(path, [{}])


# ---------------------------------------------------------------------------
# read_keyed_pairs_quad — edge cases
# ---------------------------------------------------------------------------


class TestReadQuadEdgeCases:
    def test_missing_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "nonexistent_keyed_pairs.json"
        result = read_keyed_pairs_quad(path)
        assert result == []

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        path.write_bytes(b"")
        result = read_keyed_pairs_quad(path)
        assert result == []

    def test_non_list_payload_raises_value_error(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        path.write_bytes(json.dumps({"foo": 1}).encode("utf-8"))
        with pytest.raises(ValueError, match="expected JSON array"):
            read_keyed_pairs_quad(path)

    def test_partial_quad_entry_raises_value_error(self, tmp_path):
        """An on-disk quad entry missing a required field raises ValueError."""
        path = tmp_path / "keyed_pairs.json"
        partial = [{"key": "graph1", "subject": "Alice", "predicate": "lives_in"}]
        path.write_bytes(json.dumps(partial).encode("utf-8"))
        with pytest.raises(ValueError, match="missing required fields"):
            read_keyed_pairs_quad(path)

    def test_invalid_json_raises_error(self, tmp_path):
        path = tmp_path / "keyed_pairs.json"
        path.write_bytes(b"{not valid json}")
        with pytest.raises(Exception):
            read_keyed_pairs_quad(path)


# ---------------------------------------------------------------------------
# Back-compat: legacy QA-format → quad projection
# ---------------------------------------------------------------------------


class TestReadQuadBackCompat:
    def test_legacy_qa_file_projects_to_quad(self, tmp_path):
        """write_keyed_pairs (QA, 8-field) then read_keyed_pairs_quad should
        project source_* fields into subject/predicate/object."""
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_QA_PAIR])
        result = read_keyed_pairs_quad(path)
        assert len(result) == 1
        entry = result[0]
        assert entry["subject"] == _FULL_QA_PAIR["source_subject"]
        assert entry["predicate"] == _FULL_QA_PAIR["source_predicate"]
        assert entry["object"] == _FULL_QA_PAIR["source_object"]

    def test_legacy_qa_projection_drops_qa_fields(self, tmp_path):
        """Projected quad entries must not carry question/answer/source_* fields."""
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_QA_PAIR])
        result = read_keyed_pairs_quad(path)
        entry = result[0]
        dropped_fields = (
            "question",
            "answer",
            "source_subject",
            "source_predicate",
            "source_object",
        )
        for dropped_field in dropped_fields:
            assert dropped_field not in entry

    def test_legacy_qa_projection_keeps_metadata(self, tmp_path):
        """key / speaker_id / first_seen_cycle must survive the projection."""
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_QA_PAIR])
        result = read_keyed_pairs_quad(path)
        entry = result[0]
        assert entry["key"] == _FULL_QA_PAIR["key"]
        assert entry["speaker_id"] == _FULL_QA_PAIR["speaker_id"]
        assert entry["first_seen_cycle"] == _FULL_QA_PAIR["first_seen_cycle"]

    def test_legacy_qa_projection_result_satisfies_quad_fields(self, tmp_path):
        """After projection the result must contain exactly KEYED_PAIR_FIELDS_QUAD."""
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_QA_PAIR])
        result = read_keyed_pairs_quad(path)
        entry = result[0]
        assert set(entry.keys()) == set(KEYED_PAIR_FIELDS_QUAD)

    def test_legacy_qa_multiple_entries(self, tmp_path):
        """Multiple legacy entries all get projected correctly."""
        qa_pair_2 = dict(_FULL_QA_PAIR)
        qa_pair_2.update(
            key="graph2",
            question="What is Bob's job?",
            answer="Bob is an engineer.",
            source_subject="Bob",
            source_predicate="has_job",
            source_object="engineer",
            speaker_id="Speaker1",
            first_seen_cycle=2,
        )
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs(path, [_FULL_QA_PAIR, qa_pair_2])
        result = read_keyed_pairs_quad(path)
        assert len(result) == 2
        assert result[1]["subject"] == "Bob"
        assert result[1]["predicate"] == "has_job"

    def test_native_quad_file_not_projected(self, tmp_path):
        """A quad-format file must not be modified by the back-compat path."""
        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])
        result = read_keyed_pairs_quad(path)
        assert result[0]["subject"] == _FULL_QUAD["subject"]
        # No QA fields should appear
        for qa_field in ("question", "answer"):
            assert qa_field not in result[0]


# ---------------------------------------------------------------------------
# Quad schema does not collide with QA schema
# ---------------------------------------------------------------------------


class TestQuadVsQaSchemas:
    def test_quad_fields_differ_from_qa_fields(self):
        """The two schemas must be distinct — they are parallel, not nested."""
        assert set(KEYED_PAIR_FIELDS_QUAD) != set(KEYED_PAIR_FIELDS)

    def test_quad_lacks_question_answer(self):
        assert "question" not in KEYED_PAIR_FIELDS_QUAD
        assert "answer" not in KEYED_PAIR_FIELDS_QUAD

    def test_quad_has_subject_predicate_object(self):
        assert "subject" in KEYED_PAIR_FIELDS_QUAD
        assert "predicate" in KEYED_PAIR_FIELDS_QUAD
        assert "object" in KEYED_PAIR_FIELDS_QUAD

    def test_quad_lacks_source_fields(self):
        for f in ("source_subject", "source_predicate", "source_object"):
            assert f not in KEYED_PAIR_FIELDS_QUAD


# ---------------------------------------------------------------------------
# Encryption awareness
# ---------------------------------------------------------------------------


class TestEncryptionAwarenessQuad:
    def test_plaintext_when_no_key_loaded(self, tmp_path, monkeypatch):
        """When no daily identity is loaded, file is written as plaintext."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        _clear_daily_identity_cache()

        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])

        raw = path.read_bytes()
        assert not raw.startswith(AGE_MAGIC), "expected plaintext, got age envelope"

        result = read_keyed_pairs_quad(path)
        assert result[0]["key"] == "graph1"

    def test_age_envelope_when_daily_loaded(self, tmp_path, monkeypatch):
        """When daily identity is loaded, file is written as age ciphertext."""
        _setup_daily(tmp_path, monkeypatch)

        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])

        raw = path.read_bytes()
        assert raw.startswith(AGE_MAGIC), f"expected age envelope magic, got {raw[:40]!r}"

    def test_encrypted_file_round_trips_correctly(self, tmp_path, monkeypatch):
        """Write with key → read decrypts transparently and returns full schema."""
        _setup_daily(tmp_path, monkeypatch)

        path = tmp_path / "keyed_pairs.json"
        write_keyed_pairs_quad(path, [_FULL_QUAD])

        result = read_keyed_pairs_quad(path)
        assert len(result) == 1
        for field in KEYED_PAIR_FIELDS_QUAD:
            assert result[0][field] == _FULL_QUAD[field]
