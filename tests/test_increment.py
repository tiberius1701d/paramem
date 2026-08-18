"""Unit tests for paramem.memory.increment.build_tier_increment.

Writes a shadow directory by hand (the same shapes phase 1 would produce —
KeyRegistry.save for the registry, write_infra_json for rows and the keyed
list) and asserts the assembled TierIncrement matches. No GPU, no live store,
no consolidation loop.
"""

import pytest

from paramem.memory.increment import TierIncrement, build_tier_increment
from paramem.training.key_registry import KeyRegistry


@pytest.fixture
def shadow_dir(tmp_path):
    d = tmp_path / "shadow" / "episodic"
    d.mkdir(parents=True)
    return d


def _write_registry(shadow_dir, *, active=(), stale=None, simhash=None):
    reg = KeyRegistry()
    for k in active:
        reg.add(k)
    if stale:
        reg._stale = dict(stale)
    if simhash:
        for k, fp in simhash.items():
            reg.set_simhash(k, fp)
    reg.save(shadow_dir / "indexed_key_registry.json")
    return reg


def _write_rows(shadow_dir, keys_payload, *, tier_cycle=3):
    from paramem.backup.encryption import write_infra_json

    write_infra_json(
        shadow_dir / "key_metadata.json", {"tier_cycle": tier_cycle, "keys": keys_payload}
    )


def _write_keyed(shadow_dir, keyed_list):
    from paramem.backup.encryption import write_infra_json

    write_infra_json(shadow_dir / "keyed.json", keyed_list)


def _row(**overrides):
    base = {
        "speaker_id": "speaker0",
        "relation_type": "factual",
        "reinforcement_count": 1,
        "last_reinforced_cycle": 1,
        "last_seen": "2026-01-01T00:00:00Z",
        "first_seen": "2026-01-01T00:00:00Z",
        "promoted": False,
    }
    base.update(overrides)
    return base


class TestMissingShadowArtifacts:
    def test_raises_when_no_registry(self, shadow_dir):
        _write_rows(shadow_dir, {})
        with pytest.raises(FileNotFoundError):
            build_tier_increment(
                tier="episodic", adapter_name="episodic", pre_sha="", shadow_dir=shadow_dir
            )

    def test_raises_when_no_rows(self, shadow_dir):
        _write_registry(shadow_dir)
        with pytest.raises(FileNotFoundError):
            build_tier_increment(
                tier="episodic", adapter_name="episodic", pre_sha="", shadow_dir=shadow_dir
            )


class TestRowsOnlyMember:
    def test_rebuilt_false_and_empty_entries_when_no_keyed_json(self, shadow_dir):
        _write_registry(shadow_dir, active=["graph1"], simhash={"graph1": 12345})
        _write_rows(shadow_dir, {"graph1": _row()})

        inc = build_tier_increment(
            tier="episodic", adapter_name="episodic", pre_sha="prior-digest", shadow_dir=shadow_dir
        )

        assert isinstance(inc, TierIncrement)
        assert inc.rebuilt is False
        assert inc.keyed == []
        assert inc.entries == {}
        assert inc.has_payload is False
        assert inc.bookkeeping == {"graph1": _row()}
        assert "graph1" in inc.registry
        assert inc.pre_sha == "prior-digest"
        assert inc.adapter_name == "episodic"
        assert inc.tier == "episodic"


class TestRebuiltMember:
    def test_rebuilt_true_projects_entries_from_keyed(self, shadow_dir):
        _write_registry(shadow_dir, active=["graph1", "graph2"], simhash={"graph1": 1, "graph2": 2})
        _write_rows(shadow_dir, {"graph1": _row(), "graph2": _row()})
        keyed = [
            {
                "key": "graph1",
                "subject": "speaker0",
                "predicate": "likes",
                "object": "tea",
                "speaker_id": "speaker0",
                "relation_type": "preference",
            },
            {
                "key": "graph2",
                "subject": "speaker0",
                "predicate": "works at",
                "object": "acme",
                "speaker_id": "speaker0",
                "relation_type": "factual",
            },
        ]
        _write_keyed(shadow_dir, keyed)

        inc = build_tier_increment(
            tier="episodic", adapter_name="episodic", pre_sha="d1", shadow_dir=shadow_dir
        )

        assert inc.rebuilt is True
        assert inc.has_payload is True
        assert inc.keyed == keyed
        assert inc.entries == {
            "graph1": {
                "key": "graph1",
                "subject": "speaker0",
                "predicate": "likes",
                "object": "tea",
            },
            "graph2": {
                "key": "graph2",
                "subject": "speaker0",
                "predicate": "works at",
                "object": "acme",
            },
        }

    def test_rebuilt_to_zero_keys_has_no_payload(self, shadow_dir):
        """A tier rebuilt to zero keys: keyed.json exists (empty list) —
        rebuilt=True, but has_payload is False (nothing to train)."""
        _write_registry(shadow_dir)
        _write_rows(shadow_dir, {})
        _write_keyed(shadow_dir, [])

        inc = build_tier_increment(
            tier="episodic", adapter_name="episodic", pre_sha="", shadow_dir=shadow_dir
        )

        assert inc.rebuilt is True
        assert inc.keyed == []
        assert inc.entries == {}
        assert inc.has_payload is False


class TestBytesFidelity:
    def test_registry_bytes_are_exact_plaintext_of_shadow_file(self, shadow_dir):
        from paramem.backup.encryption import read_maybe_encrypted

        _write_registry(shadow_dir, active=["graph1"], simhash={"graph1": 7})
        _write_rows(shadow_dir, {"graph1": _row()})

        inc = build_tier_increment(
            tier="episodic", adapter_name="episodic", pre_sha="", shadow_dir=shadow_dir
        )

        on_disk = read_maybe_encrypted(shadow_dir / "indexed_key_registry.json")
        assert inc.registry_bytes == on_disk

    def test_rows_bytes_are_exact_plaintext_of_shadow_file(self, shadow_dir):
        from paramem.backup.encryption import read_maybe_encrypted

        _write_registry(shadow_dir)
        _write_rows(shadow_dir, {"graph1": _row()})

        inc = build_tier_increment(
            tier="episodic", adapter_name="episodic", pre_sha="", shadow_dir=shadow_dir
        )

        on_disk = read_maybe_encrypted(shadow_dir / "key_metadata.json")
        assert inc.rows_bytes == on_disk

    def test_registry_object_is_parse_of_registry_bytes(self, shadow_dir):
        _write_registry(shadow_dir, active=["a", "b"], simhash={"a": 1, "b": 2})
        _write_rows(shadow_dir, {})

        inc = build_tier_increment(
            tier="episodic", adapter_name="episodic", pre_sha="", shadow_dir=shadow_dir
        )

        assert set(inc.registry.list_active()) == {"a", "b"}
        assert len(inc.registry) == 2
