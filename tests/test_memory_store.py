"""Unit tests for :class:`paramem.memory.store.MemoryStore`.

Locks the per-tier {quads, simhash, registry} contract that replaced the
flat ``indexed_key_cache`` + three flat ``*_simhash`` dicts on
:class:`ConsolidationLoop`.
"""

from __future__ import annotations

import threading

import pytest

from paramem.memory.store import MemoryStore
from paramem.training.key_registry import KeyRegistry


def _entry(
    key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"
) -> dict:
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": "spk-alice",
    }


# ---------------------------------------------------------------------------
# Quad payload — read / write / membership
# ---------------------------------------------------------------------------


class TestQuadPayload:
    def test_put_then_get_round_trip(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert s.get("graph1") == _entry("graph1")

    def test_get_miss_returns_none(self):
        assert MemoryStore().get("graph999") is None

    def test_has_membership(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert s.has("graph1")
        assert "graph1" in s
        assert not s.has("graph999")

    def test_tier_of_returns_owning_tier(self):
        s = MemoryStore()
        s.put("semantic", "graph42", _entry("graph42"))
        assert s.tier_of("graph42") == "semantic"
        assert s.tier_of("graph999") is None

    def test_quads_in_tier_returns_only_that_tier(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("semantic", "graph2", _entry("graph2"))
        assert set(s.entries_in_tier("episodic")) == {"graph1"}
        assert set(s.entries_in_tier("semantic")) == {"graph2"}
        assert s.entries_in_tier("procedural") == {}

    def test_iter_quads_yields_tier_key_quad(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("semantic", "graph2", _entry("graph2"))
        out = sorted((t, k) for t, k, _ in s.iter_entries())
        assert out == [("episodic", "graph1"), ("semantic", "graph2")]

    def test_len_counts_all_tiers(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("episodic", "graph2", _entry("graph2"))
        s.put("semantic", "graph3", _entry("graph3"))
        assert len(s) == 3

    def test_bookkeeping_round_trips_speaker_id(self):
        """set_bookkeeping then bookkeeping_for_key returns the correct fields.

        Replaces the deleted setdefault_entry test — bookkeeping is now the
        canonical owner of speaker_id/relation_type/reinforcement_count/last_seen."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph1", speaker_id="spk-a", relation_type="factual", first_seen="", promoted=False
        )
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "spk-a"
        assert bk["relation_type"] == "factual"
        # Second call must return updated values (idempotent overwrite).
        s.set_bookkeeping(
            "graph1", speaker_id="spk-b", relation_type="preference", first_seen="", promoted=False
        )
        bk2 = s.bookkeeping_for_key("graph1")
        assert bk2["speaker_id"] == "spk-b"
        assert bk2["relation_type"] == "preference"


# ---------------------------------------------------------------------------
# SimHash fingerprints — separated from the quad
# ---------------------------------------------------------------------------


class TestSimHash:
    def test_put_via_put_writes_simhash_when_supplied(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=0xCAFE)
        assert s.simhash("episodic", "graph1") == 0xCAFE

    def test_put_simhash_independently(self):
        s = MemoryStore()
        s.put_simhash("episodic", "graph1", 0xBEEF)
        assert s.simhash("episodic", "graph1") == 0xBEEF
        assert s.has_simhash("episodic", "graph1")

    def test_simhash_miss_returns_none(self):
        assert MemoryStore().simhash("episodic", "graph1") is None
        assert not MemoryStore().has_simhash("episodic", "graph1")

    def test_delete_simhash_only(self):
        s = MemoryStore()
        s.put_simhash("episodic", "graph1", 1)
        s.delete_simhash("episodic", "graph1")
        assert not s.has_simhash("episodic", "graph1")

    def test_replace_simhashes_in_tier_swaps_the_whole_map(self):
        s = MemoryStore()
        s.registry("episodic").add("graph1")
        s.registry("episodic").add("graph2")
        s.replace_simhashes_in_tier("episodic", {"graph1": 1, "graph2": 2})

        s.replace_simhashes_in_tier("episodic", {"graph1": 99})

        assert s.simhash("episodic", "graph1") == 99
        assert s.simhash("episodic", "graph2") is None

    def test_replace_simhashes_in_tier_refuses_a_withheld_id_without_truncating_the_map(self):
        """A naive clear-then-set loop that raises partway through leaves the
        fingerprint map truncated. Validation must run BEFORE any mutation
        so a refusal leaves the ORIGINAL map fully intact."""
        from paramem.memory.store import BookkeepingInvariantViolation

        s = MemoryStore()
        s.registry("episodic").add("graph1")
        s.registry("episodic").add("graph2")
        s.registry("episodic").stale("graph2")
        s.replace_simhashes_in_tier("episodic", {"graph1": 1})

        with pytest.raises(BookkeepingInvariantViolation):
            s.replace_simhashes_in_tier("episodic", {"graph1": 10, "graph2": 20})

        # The original map is untouched -- not truncated, not partially
        # overwritten.
        assert s.simhash("episodic", "graph1") == 1
        assert s.simhash("episodic", "graph2") is None


# ---------------------------------------------------------------------------
# KeyRegistry simhash methods — unit tests for the simhash primitives
# ---------------------------------------------------------------------------


class TestKeyRegistrySimhash:
    """Unit tests for the KeyRegistry simhash primitives.

    Locks the contract: set_simhash, drop_simhash, simhash_for,
    has_simhash, _simhashes (the tier's one fingerprint map), stale
    drops the fingerprint, remove auto-drop.
    """

    def test_set_and_simhash_for_active(self):
        """set_simhash on an active key is returned by simhash_for."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xCAFE)
        assert reg.simhash_for("graph1") == 0xCAFE

    def test_remove_drops_simhash(self):
        """remove() erases the fingerprint from both active and stale partitions."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xAAAA)
        reg.remove("graph1")
        assert reg.simhash_for("graph1") is None
        assert not reg.has_simhash("graph1")

    def test_drop_simhash_clears_both_partitions(self):
        """drop_simhash removes the fingerprint regardless of partition."""
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xCAFE)
        reg.drop_simhash("graph1")
        assert not reg.has_simhash("graph1")
        assert reg.simhash_for("graph1") is None

    def test_load_reads_simhash_from_new_schema(self, tmp_path):
        """KeyRegistry.load reads the 'simhash' field from the new schema (no .get fallback).

        A registry file without 'simhash' raises ValueError — see
        tests/test_key_registry.py::TestStrictLoadShape for the shape-refusal
        pins (single home, not duplicated here).
        """
        import json

        path = tmp_path / "indexed_key_registry.json"
        # Current schema: "stale" is a bare list of withheld ids.
        data = {
            "active_keys": ["graph1"],
            "stale": [],
            "simhash": {"graph1": 0xABCDEF},
        }
        path.write_text(json.dumps(data))

        reg = KeyRegistry.load(path)
        assert reg.simhash_for("graph1") == 0xABCDEF


# ---------------------------------------------------------------------------
# Lifecycle registry — Optional gate
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_always_present(self):
        s = MemoryStore()
        assert s.registry("episodic") is not None
        assert isinstance(s.registry("episodic"), KeyRegistry)

    def test_put_registers_by_default(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        assert "graph1" in s.registry("episodic")

    def test_register_false_skips_registry(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), register=False)
        assert "graph1" not in s.registry("episodic")
        assert s.get("graph1") is not None

    def test_load_registry_installs_preloaded_instance(self):
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph_preloaded")
        s.load_registry("episodic", reg)
        assert s.registry("episodic") is reg
        assert "graph_preloaded" in s.registry("episodic")

    def test_active_keys_in_tier(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("episodic", "graph2", _entry("graph2"))
        s.put("semantic", "graph3", _entry("graph3"))
        assert sorted(s.active_keys_in_tier("episodic")) == ["graph1", "graph2"]
        assert s.active_keys_in_tier("procedural") == []

    def test_all_active_keys(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("semantic", "graph2", _entry("graph2"))
        s.put("procedural", "proc1", _entry("proc1"))
        assert sorted(s.all_active_keys()) == ["graph1", "graph2", "proc1"]

    def test_active_keys_in_tier_excludes_withheld_ids(self):
        """The consolidation summary counts a tier's ACTIVE keys
        (``len(store.active_keys_in_tier(tier))`` — the honest source both
        ``_run_extraction_phase`` and ``/status`` read, see
        ``paramem/server/app.py``) — a withheld id must not inflate it, the
        same way it is excluded from ``__contains__``/``list_active``."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.put("episodic", "graph2", _entry("graph2"))
        s.registry("episodic").stale("graph2")

        assert s.active_keys_in_tier("episodic") == ["graph1"]
        assert len(s.active_keys_in_tier("episodic")) == 1
        # The withheld id is still KNOWN (retained for bookkeeping), just not
        # counted as active.
        assert s.registry("episodic").knows("graph2")


# ---------------------------------------------------------------------------
# drop_tier — tier-granularity rollback primitive (Part B)
# ---------------------------------------------------------------------------


class TestDropTier:
    def test_drop_tier_removes_entries_registry_and_bookkeeping(self):
        s = MemoryStore()
        s.put("episodic_interim_20260417T0000", "graph1", _entry("graph1"), simhash=0xCAFE)
        s.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="t0",
            promoted=False,
        )
        s.drop_tier("episodic_interim_20260417T0000")
        assert s.entries_in_tier("episodic_interim_20260417T0000") == {}
        assert s.has_registry("episodic_interim_20260417T0000") is False
        assert s.bookkeeping_for_key("graph1") is None

    def test_drop_tier_does_not_touch_other_tiers(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"), simhash=1)
        s.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="t0",
            promoted=False,
        )
        s.put("episodic_interim_20260417T0000", "graph2", _entry("graph2"), simhash=2)
        s.drop_tier("episodic_interim_20260417T0000")
        assert s.get("graph1") == _entry("graph1")
        assert s.bookkeeping_for_key("graph1") is not None
        assert s.entries_in_tier("episodic_interim_20260417T0000") == {}

    def test_drop_tier_unknown_tier_is_noop(self):
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        s.drop_tier("does_not_exist")  # must not raise
        assert s.get("graph1") == _entry("graph1")


# ---------------------------------------------------------------------------
# Bookkeeping — speaker_id / relation_type / reinforcement_count / last_seen
# ---------------------------------------------------------------------------


class TestBookkeeping:
    def test_bookkeeping_does_not_create_content_hit(self):
        """REGRESSION LOCK: set_bookkeeping must NOT create a content cache hit.

        The cache door answers None for an active key with no entry — the
        same no-fact shape as a miss — so bookkeeping presence alone must
        never mask an empty mirror."""
        s = MemoryStore()
        s.registry("episodic").add("k")
        s.set_bookkeeping(
            "k", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        # No put — _entries is empty.
        assert s.get("k") is None
        results = s.probe_cache({"episodic": ["k"]})
        assert results["k"] is None

    def test_new_key_write_back_round_trips_speaker_id(self):
        """set_bookkeeping + bookkeeping_for_key round-trips all fields."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph1", speaker_id="bob", relation_type="factual", first_seen="", promoted=False
        )
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "bob"
        assert bk["relation_type"] == "factual"

    def test_bookkeeping_absent_returns_none(self):
        s = MemoryStore()
        assert s.bookkeeping_for_key("nonexistent") is None

    def test_iter_bookkeeping_yields_all_keys(self):
        s = MemoryStore()
        s.set_bookkeeping(
            "k1", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        s.set_bookkeeping(
            "k2", speaker_id="bob", relation_type="preference", first_seen="", promoted=False
        )
        items = dict(s.iter_bookkeeping())
        assert items["k1"]["speaker_id"] == "alice"
        assert items["k2"]["speaker_id"] == "bob"
        assert s.bookkeeping_count() == 2

    def test_probe_cold_disk_source_key_does_not_backfill_bookkeeping(self):
        """A cold (previously-unbookkept) key served by a disk source on a
        cache miss must NOT gain a bookkeeping record from that probe — the
        fold publishes bookkeeping durably itself, and probe is no longer a
        bookkeeping writer."""
        from paramem.memory.entry import entry_simhash

        entry = {"key": "graph9", "subject": "Dana", "predicate": "lives_in", "object": "Oslo"}

        class _DiskLikeSource:
            def probe(self, keys_by_tier):
                # Content-only — matches DiskMemorySource's post-fix contract.
                return {"graph9": dict(entry)}

        s = MemoryStore()
        s.put_simhash("episodic", "graph9", entry_simhash(entry))
        assert s.bookkeeping_for_key("graph9") is None

        results = s.probe_source({"episodic": ["graph9"]}, source=_DiskLikeSource())

        assert results["graph9"]["subject"] == "Dana"
        assert s.bookkeeping_for_key("graph9") is None  # no backfill

    def test_probe_source_answers_none_for_an_off_contract_source_result(self, caplog):
        """A source returning something other than a dict for a key is a
        contract violation of ``source.probe()``'s shape -- normalized to
        ``None`` at the door, loudly (warning level), same no-fact shape as
        a miss or a gate-drop.  The conversation continues; it never sees
        the off-contract value."""
        import logging

        class _OffContractSource:
            def probe(self, keys_by_tier):
                return {"graph9": "not-a-dict"}

        s = MemoryStore()
        with caplog.at_level(logging.WARNING):
            results = s.probe_source({"episodic": ["graph9"]}, source=_OffContractSource())

        assert results["graph9"] is None
        assert "graph9" in results  # explicit None, not merely absent
        assert any(
            record.levelno == logging.WARNING and "graph9" in record.getMessage()
            for record in caplog.records
        )

    def test_probe_has_no_speaker_id_parameter(self):
        """Neither read door takes a speaker_id kwarg — passing one raises."""

        class _NoOpSource:
            def probe(self, keys_by_tier):
                return {}

        s = MemoryStore()
        with pytest.raises(TypeError):
            s.probe_cache({"episodic": []}, speaker_id="x")
        with pytest.raises(TypeError):
            s.probe_source({"episodic": []}, source=_NoOpSource(), speaker_id="x")

    def test_cache_off_empty_store_always_probes(self):
        """Under cache-off the store has no entries.  Every key must miss
        and be delegated to the source — restores preload_cache=False contract."""
        from paramem.memory.entry import entry_simhash

        probed: list = []

        class _FakeSource:
            def probe(self, keys_by_tier):
                probed.extend(k for keys in keys_by_tier.values() for k in keys)
                return {
                    k: {"key": k, "subject": "X", "predicate": "p", "object": "Y"}
                    for keys in keys_by_tier.values()
                    for k in keys
                }

        s = MemoryStore()
        # Bookkeeping loaded but NO entries (cache-off scenario).
        s.set_bookkeeping(
            "k1", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        s.set_bookkeeping(
            "k2", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        # Register keys in the registry so tier resolution works, with
        # fingerprints matching what _FakeSource returns so the live door's
        # confidence gate passes (a fingerprint-less key is dropped there).
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        for k in ("k1", "k2"):
            reg.add(k)
            reg.set_simhash(
                k, entry_simhash({"key": k, "subject": "X", "predicate": "p", "object": "Y"})
            )
        s.load_registry("episodic", reg)
        results = s.probe_source({"episodic": ["k1", "k2"]}, source=_FakeSource())
        assert set(probed) == {"k1", "k2"}
        assert results["k1"] is not None
        assert results["k2"] is not None

    # -- relation_type round-trip tests --

    def test_relation_type_round_trips(self):
        """set_bookkeeping with relation_type='preference' → bookkeeping_for_key returns it."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph1", speaker_id="alice", relation_type="preference", first_seen="", promoted=False
        )
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_relation_type_overwrite_idempotent(self):
        """A second set_bookkeeping call updates relation_type in place."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph1", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        s.set_bookkeeping(
            "graph1", speaker_id="alice", relation_type="preference", first_seen="", promoted=False
        )
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"


# ---------------------------------------------------------------------------
# reinforce — MemoryStore's own absorbing-key count resolution
# ---------------------------------------------------------------------------


class TestReinforce:
    """MemoryStore.reinforce keeps its key-shaped ``absorbing`` parameter and
    resolves counts against its own bookkeeping dict, under its own lock,
    before delegating to credit_reinforcement.  Since ``_bookkeeping`` is one
    flat dict spanning every tier, this is the "same-tier absorb" case: any
    key known to the store resolves regardless of which tier owns it."""

    def _set(self, s, key, count, **overrides):
        s.set_bookkeeping(
            key,
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            promoted=False,
            reinforcement_count=count,
            **overrides,
        )

    def test_absorbing_a_higher_count_key_from_another_tier_yields_that_count(self):
        s = MemoryStore()
        self._set(s, "survivor", 1)
        self._set(s, "absorbed", 4)
        s.reinforce(
            "survivor",
            cycle=1,
            first_seen="",
            timestamp="2026-01-01T00:00:00Z",
            absorbing=["absorbed"],
            reobserved=True,
        )
        assert s.bookkeeping_for_key("survivor")["reinforcement_count"] == 5

    def test_same_tier_absorb_keeps_max_of_absorbed(self):
        s = MemoryStore()
        self._set(s, "survivor", 1)
        self._set(s, "absorbed", 4)
        s.reinforce("survivor", cycle=1, first_seen="", absorbing=["absorbed"])
        assert s.bookkeeping_for_key("survivor")["reinforcement_count"] == 4

    def test_unknown_absorbing_key_contributes_nothing(self):
        s = MemoryStore()
        self._set(s, "survivor", 1)
        s.reinforce("survivor", cycle=1, first_seen="", absorbing=["ghost"])
        assert s.bookkeeping_for_key("survivor")["reinforcement_count"] == 1


# ---------------------------------------------------------------------------
# SimHash confidence gate — Bug-B read-time gate tests
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    """Locks the SimHash confidence gate the live door applies to every
    source result: a key whose fingerprint matches its content is served
    with the real computed confidence, never a hardcoded 1.0.  The cache
    door performs no fingerprint work of its own — its content was gated
    once at admission — so this gate is exercised through probe_source."""

    def _spo_entry(
        self,
        key: str,
        subject: str = "Alice",
        predicate: str = "lives_in",
        obj: str = "Berlin",
    ) -> dict:
        return {"key": key, "subject": subject, "predicate": predicate, "object": obj}

    def _correct_fingerprint(self, entry: dict) -> int:
        """Compute the expected SimHash fingerprint for *entry*.

        Routed through :func:`entry_simhash` — the production registration
        primitive — so the space/underscore fold it applies matches what
        :func:`verify_confidence` recomputes at recall time.
        """
        from paramem.memory.entry import entry_simhash

        return entry_simhash(entry)

    def _mismatched_fingerprint(self, entry: dict) -> int:
        """Return a fingerprint that will NOT match *entry* (different content)."""
        from paramem.memory.entry import compute_simhash

        # Compute for a completely different triple so the Hamming distance is large.
        return compute_simhash("wrong_key", "Eve", "hates", "Brussels")

    def test_confident_key_served(self):
        """A key with a matching fingerprint passes the live door's gate and
        is served with the real computed confidence."""
        entry = self._spo_entry("graph1")
        fp = self._correct_fingerprint(entry)

        class _Source:
            def probe(self, keys_by_tier):
                return {"graph1": dict(entry)}

        s = MemoryStore()
        s.put_simhash("episodic", "graph1", fp)
        results = s.probe_source({"episodic": ["graph1"]}, source=_Source())
        assert results["graph1"] is not None, "correctly-fingerprinted key must be served"
        # Confidence must be exactly 1.0 since the fingerprint was computed from
        # the same content (identical simhash → Hamming distance 0).
        assert results["graph1"]["confidence"] == 1.0, (
            f"matching fingerprint must yield confidence 1.0, got {results['graph1']['confidence']}"
        )

    def test_fingerprint_less_active_key_dropped_at_live_door(self):
        """A key active in the registry but carrying no registered
        fingerprint (e.g. a fresh tier before first consolidation) is
        treated as a failed gate at the live door, never served
        pass-through — a trained tier must always be provable."""
        entry = self._spo_entry("graph1")

        class _Source:
            def probe(self, keys_by_tier):
                return {"graph1": dict(entry)}

        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")  # active, but no simhash set for it
        s.load_registry("episodic", reg)
        results = s.probe_source({"episodic": ["graph1"]}, source=_Source())
        assert results["graph1"] is None, (
            "a fingerprint-less active key must answer None at the live door"
        )


# ---------------------------------------------------------------------------
# load_bookkeeping_from_disk — the sole boot loader for per-tier bookkeeping
# ---------------------------------------------------------------------------


def _bk_row(speaker_id: str = "speaker0", **overrides) -> dict:
    """A complete seven-field bookkeeping row, as written by
    ``_write_tier_key_metadata`` and read back by ``load_bookkeeping_from_disk``."""
    row = {
        "speaker_id": speaker_id,
        "relation_type": "factual",
        "reinforcement_count": 1,
        "last_reinforced_cycle": 0,
        "last_seen": "",
        "first_seen": "2026-01-01",
        "promoted": False,
    }
    row.update(overrides)
    return row


def _write_tier_key_metadata_file(tier_root, keys: dict, *, tier_cycle: int = 0) -> None:
    from paramem.backup.encryption import write_infra_json

    tier_root.mkdir(parents=True, exist_ok=True)
    write_infra_json(tier_root / "key_metadata.json", {"tier_cycle": tier_cycle, "keys": keys})


class TestLoadBookkeepingFromDisk:
    """MemoryStore.load_bookkeeping_from_disk(adapter_dir) — walks
    iter_tier_roots and splats each tier's own key_metadata.json rows into
    _bookkeeping via set_bookkeeping."""

    def test_relation_type_propagates_from_disk(self, tmp_path):
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        _write_tier_key_metadata_file(
            tmp_path / "episodic",
            {"graph1": _bk_row(relation_type="preference")},
        )

        result = s.load_bookkeeping_from_disk(tmp_path)

        assert result == {"loaded": 1, "orphaned": 0}
        bk = s.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_incomplete_record_raises_instead_of_silently_defaulting(self, tmp_path):
        """A row missing a mandatory field (``promoted``, the newest of the
        seven) fails loud from the splat into set_bookkeeping — no compat
        shim fills it in."""
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        incomplete_row = _bk_row()
        del incomplete_row["promoted"]
        _write_tier_key_metadata_file(tmp_path / "episodic", {"graph1": incomplete_row})

        with pytest.raises(TypeError):
            s.load_bookkeeping_from_disk(tmp_path)

    def test_raises_when_registry_has_known_keys_but_no_row_file(self, tmp_path):
        """A tier whose registry (already loaded) reports a known key but has
        no ``key_metadata.json`` at all is a violation of the
        every-known-key-has-a-row invariant -- raised, never silently
        skipped past."""
        from paramem.memory.store import BookkeepingInvariantViolation

        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        # No key_metadata.json written at all under tmp_path/episodic/.

        with pytest.raises(BookkeepingInvariantViolation):
            s.load_bookkeeping_from_disk(tmp_path)

    def test_no_raise_when_registry_has_zero_known_keys_and_no_row_file(self, tmp_path):
        """A tier with a registry and ZERO known keys and no row file is the
        ordinary empty case, not a violation."""
        s = MemoryStore()
        s.load_registry("episodic", KeyRegistry())

        result = s.load_bookkeeping_from_disk(tmp_path)

        assert result == {"loaded": 0, "orphaned": 0}

    def test_legacy_cased_speaker_id_coerced_to_lowercase(self, tmp_path):
        """A cased ``Speaker0`` left over from a legacy key_metadata.json is
        silently coerced to ``speaker0`` at boot via set_bookkeeping's own
        canonicalization — self-healing on the next save."""
        s = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        s.load_registry("episodic", reg)
        _write_tier_key_metadata_file(
            tmp_path / "episodic",
            {"graph1": _bk_row(speaker_id="Speaker0")},
        )

        s.load_bookkeeping_from_disk(tmp_path)

        assert s.bookkeeping_for_key("graph1")["speaker_id"] == "speaker0"


class TestLoadBookkeepingMergeConflictRule:
    """When a key's row appears in more than one tier's key_metadata.json —
    a stale leftover from a tier the key no longer belongs to — the row
    from the tier whose registry CURRENTLY owns the key wins. A key active
    in two tiers' registries at once has no producing path under the
    single-tier-ownership invariant, so that shape is not modeled here."""

    _INTERIM_STAMP_DIR = "interim_20260101T0000"
    _INTERIM_TIER_NAME = "episodic_interim_20260101T0000"

    def test_owner_tier_row_wins_over_stale_interim_leftover(self, tmp_path):
        """graph1 is known (active) only to the main episodic registry; a
        stale leftover row for it also sits in an interim slot's file (the
        interim registry does NOT know it). The main tier's own row wins,
        and the interim's stale row is never applied."""
        s = MemoryStore()
        main_reg = KeyRegistry()
        main_reg.add("graph1")
        s.load_registry("episodic", main_reg)
        s.load_registry(self._INTERIM_TIER_NAME, KeyRegistry())  # does not know graph1

        _write_tier_key_metadata_file(
            tmp_path / "episodic", {"graph1": _bk_row(speaker_id="speaker_main")}
        )
        _write_tier_key_metadata_file(
            tmp_path / "episodic" / self._INTERIM_STAMP_DIR,
            {"graph1": _bk_row(speaker_id="speaker_interim")},
        )

        result = s.load_bookkeeping_from_disk(tmp_path)

        assert s.bookkeeping_for_key("graph1")["speaker_id"] == "speaker_main"
        assert result["loaded"] == 1


# ---------------------------------------------------------------------------
# Thread-safety concurrency contract
# ---------------------------------------------------------------------------


class TestConcurrencyContract:
    """Verify the RLock concurrency contract.

    These tests do NOT test for race conditions (that would require a stress
    harness with tight timing).  Instead they verify the observable contract:
    - iter_entries / iter_bookkeeping snapshot under the lock → their result
      is unaffected by a concurrent (but serialised) mutation.
    - swap() atomically rebinds all three structures.
    - entries_in_tier() returns a copy, not a live internal dict.
    - The store exposes an RLock (not plain Lock) on ``_lock``.
    """

    def test_rlock_is_reentrant(self):
        """_lock is an RLock; the same thread can acquire it twice without deadlock."""
        s = MemoryStore()
        # RLock allows re-entry from the same thread.
        acquired_inner = False
        with s._lock:
            with s._lock:
                acquired_inner = True
        assert acquired_inner, "_lock must be an RLock (re-entrant)"

    def test_entries_in_tier_returns_copy_not_live(self):
        """entries_in_tier must return a snapshot copy.

        Mutations to the returned dict must NOT propagate to the store."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        copy_dict = s.entries_in_tier("episodic")
        # Mutate the returned copy.
        copy_dict["injected"] = {"key": "injected"}
        # Store must be unaffected.
        assert "injected" not in s.entries_in_tier("episodic")
        assert s.get("injected") is None

    def test_iter_entries_snapshot_unaffected_by_subsequent_put(self):
        """iter_entries snapshot is taken before yielding.

        A put() that happens after the iterator is created must NOT appear
        in the iteration (because the snapshot is fixed at creation time)."""
        s = MemoryStore()
        s.put("episodic", "graph1", _entry("graph1"))
        it = s.iter_entries()
        # Materialise the generator to exhaust the snapshot.
        snap = list(it)
        # Now add a second key.
        s.put("episodic", "graph2", _entry("graph2"))
        # The snapshot must only contain graph1 (the generator was already
        # exhausted before graph2 was inserted).
        keys_in_snap = {k for _, k, _ in snap}
        assert "graph1" in keys_in_snap
        assert "graph2" not in keys_in_snap

    def test_iter_bookkeeping_snapshot_unaffected_by_subsequent_set(self):
        """iter_bookkeeping snapshot is taken before yielding."""
        s = MemoryStore()
        s.set_bookkeeping(
            "k1", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        it = s.iter_bookkeeping()
        snap = list(it)
        # Add a second key after the iterator is exhausted.
        s.set_bookkeeping(
            "k2", speaker_id="bob", relation_type="factual", first_seen="", promoted=False
        )
        keys_in_snap = {k for k, _ in snap}
        assert "k1" in keys_in_snap
        assert "k2" not in keys_in_snap

    def test_swap_rebinds_all_three_structures_atomically(self):
        """swap() replaces entries, registry, and bookkeeping in one operation.

        After swap, get/iter_entries/bookkeeping_for_key all reflect the new
        state and the old keys are gone."""
        s = MemoryStore()
        s.put("episodic", "old_key", _entry("old_key"))
        s.set_bookkeeping(
            "old_key", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )

        new_reg = KeyRegistry()
        new_reg.add("new_key")
        new_entries: dict[str, dict[str, dict]] = {"semantic": {"new_key": _entry("new_key")}}
        new_bookkeeping: dict[str, dict] = {
            "new_key": {
                "speaker_id": "bob",
                "relation_type": "preference",
                "reinforcement_count": 1,
                "last_reinforced_cycle": 0,
                "last_seen": "",
            }
        }
        s.swap(new_entries, {"semantic": new_reg}, new_bookkeeping)

        # Old key is gone.
        assert s.get("old_key") is None
        assert s.bookkeeping_for_key("old_key") is None
        # New key is present.
        assert s.get("new_key") == _entry("new_key")
        assert s.tier_of("new_key") == "semantic"
        bk = s.bookkeeping_for_key("new_key")
        assert bk is not None
        assert bk["speaker_id"] == "bob"
        # Registry reflects the swap.
        assert "new_key" in s.registry("semantic")
        assert not s.has_registry("episodic")

    def test_swap_visible_to_concurrent_reader_after_release(self):
        """A thread that acquires the lock after swap() sees the new state.

        This is a serialised (not truly concurrent) test: it proves the lock
        is not bypassed by swap() — the new state is visible to the next
        acquirer."""
        s = MemoryStore()
        s.put("episodic", "before", _entry("before"))

        results: list = []

        def reader():
            results.append(s.get("before"))
            results.append(s.get("after"))

        new_entries: dict[str, dict[str, dict]] = {"episodic": {"after": _entry("after")}}
        s.swap(new_entries, {}, {})

        t = threading.Thread(target=reader)
        t.start()
        t.join()

        assert results[0] is None, "old key must be gone after swap"
        assert results[1] == _entry("after"), "new key must be visible after swap"


# ---------------------------------------------------------------------------
# read_registries_from_disk — store-free static builder (phase-2)
# ---------------------------------------------------------------------------


class TestReadRegistriesFromDisk:
    """read_registries_from_disk returns a fresh dict without touching any store."""

    def test_returns_dict_with_main_tiers_on_empty_dir(self, tmp_path) -> None:
        """All three main tiers appear even when their registry files are absent.

        KeyRegistry.load on a missing file returns an empty registry — so the
        returned dict always has "episodic", "semantic", "procedural" keys."""
        # Create the adapter dir structure but leave registry files absent.
        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        result = MemoryStore.read_registries_from_disk(tmp_path)

        assert set(result.keys()) == {"episodic", "semantic", "procedural"}
        for tier in ("episodic", "semantic", "procedural"):
            assert result[tier].list_active() == []

    def test_does_not_touch_any_live_store(self, tmp_path) -> None:
        """read_registries_from_disk is store-free — the live store is unchanged."""
        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        live = MemoryStore()
        live.put("episodic", "live_key", _entry("live_key"))

        # Call the static method — should not touch the live store.
        MemoryStore.read_registries_from_disk(tmp_path)

        # live store must be unmodified.
        assert live.get("live_key") == _entry("live_key")

    def test_load_registries_from_disk_delegates_to_static(self, tmp_path) -> None:
        """load_registries_from_disk (instance method) produces the same registry
        as read_registries_from_disk (static) on the same adapter dir."""
        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        static_result = MemoryStore.read_registries_from_disk(tmp_path)

        store = MemoryStore()
        store.load_registries_from_disk(tmp_path)

        for tier in ("episodic", "semantic", "procedural"):
            assert list(static_result[tier].list_active()) == list(
                store.registry(tier).list_active()
            ), f"Tier '{tier}' active keys differ between static and instance method"


# ---------------------------------------------------------------------------
# set_bookkeeping guard — no-unattributed-keys invariant
# ---------------------------------------------------------------------------


class TestSetBookkeepingGuard:
    """set_bookkeeping raises on an empty speaker_id unless the caller opts out.

    ``MemoryStore.set_bookkeeping`` rejects ``speaker_id=""`` with ValueError
    (no-unattributed-keys invariant) unless ``allow_empty_speaker=True`` is
    passed — the carve-out reserved for reload paths and keyless concept-node
    edges that genuinely have no speaker.
    """

    def test_empty_speaker_id_raises_without_allow_flag(self):
        """set_bookkeeping(speaker_id='') raises ValueError by default."""
        s = MemoryStore()
        with pytest.raises(ValueError, match="no-unattributed-keys invariant"):
            s.set_bookkeeping(
                "graph1",
                speaker_id="",
                relation_type="factual",
                first_seen="",
                promoted=False,
            )

    def test_empty_speaker_id_allowed_with_flag(self):
        """set_bookkeeping(speaker_id='', allow_empty_speaker=True) succeeds."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph2",
            speaker_id="",
            relation_type="factual",
            allow_empty_speaker=True,
            first_seen="",
            promoted=False,
        )
        bk = s.bookkeeping_for_key("graph2")
        assert bk is not None
        assert bk["speaker_id"] == ""

    def test_nonempty_speaker_id_succeeds_without_flag(self):
        """Non-empty speaker_id always succeeds (no flag needed).

        A cased ``Speaker0`` is accepted and normalized to lowercase ``speaker0``
        by :meth:`set_bookkeeping`'s ``is_speaker_id`` gate."""
        s = MemoryStore()
        s.set_bookkeeping(
            "graph3",
            speaker_id="Speaker0",
            relation_type="factual",
            first_seen="",
            promoted=False,
        )
        bk = s.bookkeeping_for_key("graph3")
        assert bk is not None
        assert bk["speaker_id"] == "speaker0"

    def test_cased_speaker_id_normalized_to_lowercase(self):
        """set_bookkeeping normalizes is_speaker_id values to lowercase.

        Cased ``Speaker0`` is coerced to ``speaker0``; the router's
        ``_speaker_key_index`` receives the normalized form, eliminating the
        silent-drop regression where legacy key_metadata.json held cased ids."""
        s = MemoryStore()
        s.set_bookkeeping(
            "g1", speaker_id="Speaker0", relation_type="factual", first_seen="", promoted=False
        )
        bk = s.bookkeeping_for_key("g1")
        assert bk is not None
        assert bk["speaker_id"] == "speaker0", (
            "set_bookkeeping must lowercase Speaker0 → speaker0 to match the router index."
        )

    def test_empty_speaker_id_passes_through_with_flag(self):
        """Empty speaker_id passes through (allow_empty_speaker=True); not lowercased."""
        s = MemoryStore()
        s.set_bookkeeping(
            "g2",
            speaker_id="",
            relation_type="factual",
            allow_empty_speaker=True,
            first_seen="",
            promoted=False,
        )
        bk = s.bookkeeping_for_key("g2")
        assert bk is not None
        assert bk["speaker_id"] == ""

    def test_non_speaker_value_passes_through_unchanged(self):
        """A non-speaker_id value that is non-empty passes through without lowercasing."""
        s = MemoryStore()
        s.set_bookkeeping(
            "g3", speaker_id="alice", relation_type="factual", first_seen="", promoted=False
        )
        bk = s.bookkeeping_for_key("g3")
        assert bk is not None
        assert bk["speaker_id"] == "alice"

    def test_router_routes_by_legacy_cased_speaker_id_after_normalization(self, monkeypatch):
        """Cased-bookkeeping → normalized → routable: ``QueryRouter``'s
        ``_speaker_key_index`` (sole privacy boundary) is built from
        normalized bookkeeping, so ``route(speaker_id="speaker0")`` finds a
        key bookkept as ``Speaker0``."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from paramem.server.router import Intent, QueryRouter

        s = MemoryStore()
        s.put(
            "episodic",
            "graph42",
            {"key": "graph42", "subject": "speaker0", "predicate": "lives_in", "object": "London"},
        )
        s.set_bookkeeping(
            "graph42", speaker_id="Speaker0", relation_type="factual", first_seen="", promoted=False
        )

        monkeypatch.setattr(
            "paramem.server.intent.classify_intent", MagicMock(return_value=Intent.PERSONAL)
        )
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=s)
        plan = router.route("Where do I live?", speaker_id="speaker0")

        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph42" in all_keys, (
            "route(speaker_id='speaker0') must find a key bookkept under cased "
            "'Speaker0' — the router index is the sole privacy boundary."
        )


# ---------------------------------------------------------------------------
# probe renders speaker{N} tokens verbatim — no resolver in probe
# ---------------------------------------------------------------------------


class TestProbeRendersTokensVerbatim:
    """Both read doors render ``fact_text`` with raw ``speaker{N}`` tokens —
    the cache door's own renderer and the live door's source passthrough
    alike.  Neither door takes a resolver parameter — display-name
    resolution happens exactly once, at the reply boundary, via
    :func:`paramem.server.speaker.resolve_speaker_tokens`, never inside a
    probe door.
    """

    def test_cache_hit_subject_token_verbatim(self) -> None:
        """Cache door: fact_text carries the raw subject token as-is."""
        s = MemoryStore()
        s.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "speaker0", "predicate": "lives_in", "object": "Berlin"},
        )
        s.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )
        results = s.probe_cache({"episodic": ["graph1"]})
        assert results["graph1"]["fact_text"] == "speaker0 lives_in Berlin"

    def test_cache_hit_object_token_verbatim(self) -> None:
        """Cache door: fact_text carries the raw object token as-is."""
        s = MemoryStore()
        s.put(
            "episodic",
            "graph2",
            {"key": "graph2", "subject": "speaker0", "predicate": "knows", "object": "speaker9"},
        )
        s.set_bookkeeping(
            "graph2", speaker_id="speaker0", relation_type="factual", first_seen="", promoted=False
        )
        results = s.probe_cache({"episodic": ["graph2"]})
        assert results["graph2"]["fact_text"] == "speaker0 knows speaker9"

    def test_source_miss_path_verbatim(self) -> None:
        """Live door: the source's own fact_text (already rendered verbatim
        by the source layer) passes through unmodified — the door does not
        re-render or resolve it.  Registers a matching fingerprint so the
        confidence gate passes (a fingerprint-less key is dropped there)."""
        from paramem.memory.entry import entry_simhash

        entry = {
            "key": "graph9",
            "subject": "speaker9",
            "predicate": "lives_in",
            "object": "Paris",
        }

        class _FakeSource:
            def probe(self, keys_by_tier):
                return {"graph9": {**entry, "fact_text": "speaker9 lives_in Paris"}}

        s = MemoryStore()
        s.put_simhash("episodic", "graph9", entry_simhash(entry))
        results = s.probe_source(
            {"episodic": ["graph9"]},
            source=_FakeSource(),
        )
        assert results["graph9"]["fact_text"] == "speaker9 lives_in Paris"

    def test_probe_has_no_resolver_parameter(self) -> None:
        """Neither read door takes a speaker_resolver kwarg — passing one raises."""
        import pytest

        class _NoOpSource:
            def probe(self, keys_by_tier):
                return {}

        s = MemoryStore()
        with pytest.raises(TypeError):
            s.probe_cache({"episodic": []}, speaker_resolver=lambda t: t)
        with pytest.raises(TypeError):
            s.probe_source({"episodic": []}, source=_NoOpSource(), speaker_resolver=lambda t: t)


# ---------------------------------------------------------------------------
# read_simhash_registry_from_disk — opt-in cache
# ---------------------------------------------------------------------------


def _write_episodic_registry(adapter_dir, fingerprints: dict[str, int]) -> None:
    """Write a minimal ``episodic/indexed_key_registry.json`` under *adapter_dir*."""
    reg = KeyRegistry()
    for key, fp in fingerprints.items():
        reg.add(key)
        reg.set_simhash(key, fp)
    tier_dir = adapter_dir / "episodic"
    tier_dir.mkdir(parents=True, exist_ok=True)
    (tier_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())


@pytest.fixture(autouse=True)
def _clear_simhash_registry_cache():
    """Every test in this module gets a clean process-wide cache — a hit
    left behind by one test must never leak into the next."""
    from paramem.memory.store import invalidate_simhash_registry_cache

    invalidate_simhash_registry_cache()
    yield
    invalidate_simhash_registry_cache()


class TestFingerprintChainEndToEnd:
    """The SimHash fingerprint gate fires exactly once per fact, at every
    boundary source output crosses — asserted as one chain: a fold
    registers a key's fingerprint through ``entry_simhash``; the recall
    gate (``build_registry`` + ``verify_confidence``, the same two
    primitives :class:`~paramem.training.early_stop.RecallEarlyStopCallback`
    calls) accepts a matching entry and refuses a mismatched one below
    threshold; and a hit whose stored triple does not match its stored
    fingerprint is dropped on BOTH read doors — the live door via its own
    confidence gate, the cache door because a mismatched entry never gets
    admitted into the mirror in the first place (the boot fill's own gate,
    performed by the source before content_only projection)."""

    def test_register_gate_verify_and_both_doors_drop_a_mismatched_entry(self, tmp_path):
        from paramem.memory.entry import (
            build_registry,
            content_only_entry,
            entry_simhash,
            is_admissible_probe_result,
        )
        from paramem.memory.source import DiskMemorySource
        from tests._fold_fixtures import _write_graph

        good_entry = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
        }
        # Registered under graph1's own (correct) fingerprint, but the
        # ON-DISK content is a different triple entirely — a fold that
        # somehow wrote mismatched content, or on-disk tampering.
        mismatched_key = "graph2"
        mismatched_registered_entry = {
            "key": mismatched_key,
            "subject": "Bob",
            "predicate": "works_at",
            "object": "Acme",
        }
        mismatched_disk_entry = {
            "key": mismatched_key,
            "subject": "Eve",
            "predicate": "hates",
            "object": "Brussels",
        }

        # 1. Registration: a fold registers each key's fingerprint through
        #    entry_simhash — the production registration primitive.
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", entry_simhash(good_entry))
        registry.add(mismatched_key)
        registry.set_simhash(mismatched_key, entry_simhash(mismatched_registered_entry))
        registry.save(tmp_path / "episodic" / "indexed_key_registry.json")

        # 2. Recall gate: build_registry + verify_confidence are the exact
        #    two primitives RecallEarlyStopCallback uses to verify staged
        #    weights against the registered fingerprints.
        gate_registry = build_registry([good_entry, mismatched_disk_entry])
        from paramem.memory.entry import DEFAULT_CONFIDENCE_THRESHOLD, verify_confidence

        # A staged reconstruction that recalls the CORRECT content for
        # graph1 verifies against its own just-built registry.
        assert (
            verify_confidence(good_entry, {"graph1": gate_registry["graph1"]})
            >= DEFAULT_CONFIDENCE_THRESHOLD
        )
        # A staged reconstruction that recalls the WRONG content for
        # mismatched_key, verified against the REGISTERED (correct)
        # fingerprint, refuses below threshold.
        mismatched_registered_fp = {mismatched_key: registry.simhash_for(mismatched_key)}
        assert (
            verify_confidence(mismatched_disk_entry, mismatched_registered_fp)
            < DEFAULT_CONFIDENCE_THRESHOLD
        )

        # 3. Live door: DiskMemorySource gates its own results against the
        #    on-disk registry; a hit whose stored triple does not match its
        #    stored fingerprint answers None at the door.
        _write_graph(
            tmp_path / "episodic",
            [
                {**good_entry, "speaker_id": "speaker0"},
                {**mismatched_disk_entry, "speaker_id": "speaker0"},
            ],
        )
        disk_registry = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        source = DiskMemorySource(tmp_path, registry=disk_registry)
        source_results = source.probe({"episodic": ["graph1", mismatched_key]})

        store = MemoryStore()
        # The store's own confidence gate (probe_source) needs the registry
        # loaded too — a fingerprint-less key is dropped there, so graph1's
        # correct fingerprint must be on record for it to be served.
        store.load_registry("episodic", registry)
        results = store.probe_source(
            {"episodic": ["graph1", mismatched_key]},
            source=source,
        )
        assert results["graph1"] is not None, "the correctly-fingerprinted key must be served"
        assert results[mismatched_key] is None, (
            "a hit whose triple does not match its fingerprint is dropped at the live door"
        )

        # 4. Cache door: the mismatch is caught at ADMISSION (the source's
        #    own gate, before content_only projection) — a mismatched entry
        #    never enters the mirror, so it answers None there too, purely
        #    because it was never admitted, not via a second read-time gate.
        new_entries: dict = {}
        for key, result in source_results.items():
            if not is_admissible_probe_result(result):
                continue
            new_entries.setdefault("episodic", {})[key] = content_only_entry(result)
        assert mismatched_key not in new_entries.get("episodic", {}), (
            "the mismatched entry must never be admitted into the mirror"
        )
        assert new_entries["episodic"]["graph1"] == good_entry

        new_registry = {
            "episodic": KeyRegistry.load(tmp_path / "episodic" / "indexed_key_registry.json")
        }
        store.swap(new_entries, new_registry, {})
        cache_results = store.probe_cache({"episodic": ["graph1", mismatched_key]})
        assert cache_results["graph1"] is not None
        assert cache_results[mismatched_key] is None, (
            "never-admitted content answers None at the cache door, with no "
            "fingerprint work performed at read time"
        )


class TestDiskMemorySourceBoundSlot:
    """DiskMemorySource resolves each tier's LIVE slot
    (``find_live_slot`` against the tier's own registry hash) and reads
    ``graph.json`` from INSIDE that slot -- never a tier-root ``graph.json``
    fallback in either direction."""

    def test_disk_source_reads_the_bound_slots_graph(self, tmp_path) -> None:
        from paramem.memory.entry import entry_simhash
        from paramem.memory.source import DiskMemorySource
        from tests._fold_fixtures import _write_graph

        entry = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
        }

        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", entry_simhash(entry))
        registry.save(tmp_path / "episodic" / "indexed_key_registry.json")

        _write_graph(tmp_path / "episodic", [{**entry, "speaker_id": "speaker0"}])

        disk_registry = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        source = DiskMemorySource(tmp_path, registry=disk_registry)

        results = source.probe({"episodic": ["graph1"]})

        assert results["graph1"] is not None
        assert results["graph1"]["subject"] == "Alice"
        assert results["graph1"]["object"] == "Berlin"

    def test_disk_source_returns_misses_for_a_tier_with_no_bound_slot(self, tmp_path) -> None:
        """A tier whose registry knows a key but carries no bound slot (never
        written, or the registry moved on past whatever slot is on disk)
        answers ``None`` for every one of its keys -- the ordinary per-key
        miss shape, never a silently empty graph and never a tier-root
        fallback read."""
        from paramem.memory.source import DiskMemorySource

        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 12345)
        registry.save(tmp_path / "episodic" / "indexed_key_registry.json")
        # No slot written under episodic/ at all.

        disk_registry = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        source = DiskMemorySource(tmp_path, registry=disk_registry)

        results = source.probe({"episodic": ["graph1"]})

        assert results == {"graph1": None}


def _write_unpublished_increment(tier_root, *, keys: "list[dict]") -> None:
    """Write (never publish) one simulate-venue slot carrying *keys*.

    Uses the production two-phase primitive
    (:func:`~paramem.memory.persistence.write_tier_slot`) rather than
    :func:`tests._fold_fixtures._write_graph` -- the written slot's
    manifest is stamped with the digest of registry bytes that never land
    at *tier_root*, so it stays unbound: ``find_live_slot`` cannot resolve
    it until a matching ``publish_tier_registry`` call lands those exact
    bytes.

    *keys*: ``[{"key", "subject", "predicate", "object"}, ...]``.
    """
    import json as _json

    from paramem.memory.increment import TierIncrement, TierWriteContext
    from paramem.memory.persistence import write_tier_slot
    from paramem.training.key_registry import KeyRegistry

    registry = KeyRegistry()
    for spec in keys:
        registry.add(spec["key"])
        registry.set_simhash(spec["key"], 1)
    registry_bytes = registry.save_bytes()
    rows_bytes = _json.dumps({"tier_cycle": 0, "keys": {}}).encode("utf-8")

    increment = TierIncrement(
        tier="episodic",
        adapter_name="episodic",
        registry=registry,
        registry_bytes=registry_bytes,
        rows_bytes=rows_bytes,
        entries={},
        bookkeeping={},
        keyed=[
            {
                "key": spec["key"],
                "subject": spec["subject"],
                "predicate": spec["predicate"],
                "object": spec["object"],
                "speaker_id": "speaker0",
            }
            for spec in keys
        ],
        rebuilt=True,
        pre_sha="",
    )
    ctx = TierWriteContext(
        model=None,
        tokenizer=None,
        fingerprint_cache={},
        output_dir=tier_root.parent,
        tier_configs={},
        store=None,
        keep_prior_slots=1,
    )
    write_tier_slot(ctx=ctx, increment=increment, stamp="20260102T0000", mode="simulate")
    # Deliberately no publish_tier_registry call -- this slot's manifest
    # digest never lands at tier_root/indexed_key_registry.json.


class TestDiskMemorySourceMidWindowWrite:
    """A boot (or any live-door probe) landing between a tier's WRITE and
    its PUBLISH resolves the OLD bound slot -- ``find_live_slot`` binds by
    the registry hash actually on disk, and the written-but-unpublished
    slot's manifest is stamped with a digest that has not landed there
    yet."""

    def test_probe_source_serves_the_published_value_and_misses_the_written_only_key(
        self, tmp_path
    ) -> None:
        from paramem.memory.entry import entry_simhash
        from paramem.memory.source import DiskMemorySource
        from tests._fold_fixtures import _write_graph

        v1_entry = {
            "key": "graph1",
            "subject": "alice",
            "predicate": "lives_in",
            "object": "berlin",
        }

        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", entry_simhash(v1_entry))
        registry.save(tmp_path / "episodic" / "indexed_key_registry.json")

        # Published baseline: graph1 -> V1, bound to the registry above.
        _write_graph(tmp_path / "episodic", [{**v1_entry, "speaker_id": "speaker0"}])

        # Written but never published: graph1 -> V2 plus a brand-new graph2.
        _write_unpublished_increment(
            tmp_path / "episodic",
            keys=[
                {
                    "key": "graph1",
                    "subject": "alice",
                    "predicate": "lives_in",
                    "object": "hamburg",
                },
                {
                    "key": "graph2",
                    "subject": "bob",
                    "predicate": "lives_in",
                    "object": "munich",
                },
            ],
        )

        disk_registry = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        source = DiskMemorySource(tmp_path, registry=disk_registry)

        store = MemoryStore()
        store.load_registry("episodic", registry)
        results = store.probe_source({"episodic": ["graph1", "graph2"]}, source=source)

        assert results["graph1"] is not None
        assert results["graph1"]["object"] == "berlin", (
            "the live door must resolve the OLD bound slot -- the written-"
            "only increment's value must never be served before its publish"
        )
        assert results["graph2"] is None, (
            "a key that exists only in the unpublished slot is unreachable "
            "-- it is not enumerable via the live registry and its slot "
            "does not bind"
        )


class TestReadSimhashRegistryFromDiskCache:
    """``cached=True`` is opt-in: default behaviour is unchanged (always
    disk-truth), and the cache is fresh until explicitly invalidated."""

    def test_default_uncached_reflects_disk_every_call(self, tmp_path) -> None:
        _write_episodic_registry(tmp_path, {"graph1": 111})
        first = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        assert first == {"graph1": 111}

        _write_episodic_registry(tmp_path, {"graph1": 222})
        second = MemoryStore.read_simhash_registry_from_disk(tmp_path)
        assert second == {"graph1": 222}

    def test_cached_read_matches_fresh_read(self, tmp_path) -> None:
        _write_episodic_registry(tmp_path, {"graph1": 111, "graph2": 222})
        fresh = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=False)
        cached = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert cached == fresh == {"graph1": 111, "graph2": 222}

    def test_cached_read_survives_disk_change_until_invalidated(self, tmp_path) -> None:
        from paramem.memory.store import invalidate_simhash_registry_cache

        _write_episodic_registry(tmp_path, {"graph1": 111})
        first = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert first == {"graph1": 111}

        # Mutate the registry file on disk without invalidating the cache.
        _write_episodic_registry(tmp_path, {"graph1": 999})
        stale = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert stale == {"graph1": 111}, "cache must serve the old mapping until invalidated"

        invalidate_simhash_registry_cache()
        fresh = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert fresh == {"graph1": 999}

    def test_invalidate_clears_whole_cache(self, tmp_path) -> None:
        from paramem.memory.store import invalidate_simhash_registry_cache

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        _write_episodic_registry(dir_a, {"graph1": 1})
        _write_episodic_registry(dir_b, {"graph2": 2})
        MemoryStore.read_simhash_registry_from_disk(dir_a, cached=True)
        MemoryStore.read_simhash_registry_from_disk(dir_b, cached=True)

        _write_episodic_registry(dir_a, {"graph1": 999})
        _write_episodic_registry(dir_b, {"graph2": 999})
        invalidate_simhash_registry_cache()

        # invalidate_simhash_registry_cache() takes no argument — it always
        # clears the whole process-wide cache, every adapter_dir at once.
        assert MemoryStore.read_simhash_registry_from_disk(dir_a, cached=True) == {"graph1": 999}
        assert MemoryStore.read_simhash_registry_from_disk(dir_b, cached=True) == {"graph2": 999}

    def test_invalidate_takes_no_argument(self) -> None:
        """Signature guard: passing an argument is a TypeError, not a
        per-adapter_dir invalidation — there is no such arm any more."""
        from paramem.memory.store import invalidate_simhash_registry_cache

        with pytest.raises(TypeError):
            invalidate_simhash_registry_cache("/some/adapter/dir")

    def test_invalidation_mid_walk_is_returned_but_not_published(
        self, tmp_path, monkeypatch
    ) -> None:
        """An invalidation landing while a disk walk is in flight must not
        corrupt the cache: the walked result still reaches this call's own
        caller (a walk in progress owes its result to whoever started it),
        but the generation guard discards the publish — the next cached
        read re-walks instead of ever serving that now-possibly-stale
        value from the cache."""
        from paramem.memory.store import invalidate_simhash_registry_cache

        _write_episodic_registry(tmp_path, {"graph1": 111})

        # Captured before monkeypatching — stays bound to the real
        # implementation regardless of the patch below.
        original_load_simhashes = KeyRegistry.load_simhashes
        calls = {"n": 0}

        def _racing_load_simhashes(cls, path):
            calls["n"] += 1
            if calls["n"] == 1:
                # A concurrent registry mutation + invalidation landing
                # while this walk is still in flight.
                invalidate_simhash_registry_cache()
            return original_load_simhashes(path)

        monkeypatch.setattr(KeyRegistry, "load_simhashes", classmethod(_racing_load_simhashes))

        result = MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert result == {"graph1": 111}, "the walked result is still returned to the caller"

        # The race must have discarded the publish: a second cached read
        # re-walks (observable via the call counter) rather than serving a
        # cache hit from the raced walk.
        calls["n"] = 0
        MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)
        assert calls["n"] > 0, "cache entry was not published after the race — must re-walk"


class TestRouterReloadInvalidatesSimhashRegistryCache:
    """``QueryRouter.reload()`` invalidates the process-wide simhash-registry
    cache so a stale opt-in read can never survive past the reload every
    registry-mutating server path ends in."""

    def test_reload_calls_invalidate_simhash_registry_cache(self, tmp_path, monkeypatch) -> None:
        import paramem.server.router as router_module

        calls: list[object] = []
        monkeypatch.setattr(
            router_module,
            "invalidate_simhash_registry_cache",
            lambda *a, **k: calls.append((a, k)),
        )

        router = router_module.QueryRouter(adapter_dir=tmp_path, memory_store=None)
        assert len(calls) == 1, "__init__ calls reload() once"

        calls.clear()
        router.reload()
        assert len(calls) == 1

    def test_reload_actually_clears_a_populated_cache(self, tmp_path) -> None:
        import paramem.server.router as router_module

        _write_episodic_registry(tmp_path, {"graph1": 111})
        MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True)

        _write_episodic_registry(tmp_path, {"graph1": 999})
        router = router_module.QueryRouter(adapter_dir=tmp_path, memory_store=None)
        router.reload()

        assert MemoryStore.read_simhash_registry_from_disk(tmp_path, cached=True) == {"graph1": 999}
