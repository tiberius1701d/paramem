"""Tests for the key registry."""

import hashlib
import inspect
import json as _json

import pytest

from paramem.memory.store import BookkeepingInvariantViolation, MemoryStore
from paramem.training.key_registry import KeyRegistry


class TestKeyRegistry:
    def test_add_and_list(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        assert reg.list_active() == ["session_001", "session_002"]

    def test_add_duplicate_ignored(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_001")
        assert len(reg) == 1

    def test_remove(self):
        reg = KeyRegistry()
        reg.add("session_001")
        reg.add("session_002")
        reg.remove("session_001")
        assert reg.list_active() == ["session_002"]
        assert "session_001" not in reg

    def test_remove_nonexistent(self):
        reg = KeyRegistry()
        reg.remove("nonexistent")  # should not raise

    def test_contains(self):
        reg = KeyRegistry()
        reg.add("session_001")
        assert "session_001" in reg
        assert "session_002" not in reg

    def test_len(self):
        reg = KeyRegistry()
        assert len(reg) == 0
        reg.add("a")
        reg.add("b")
        assert len(reg) == 2


class TestPersistence:
    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = KeyRegistry.load(path)
        assert len(loaded) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "registry.json"
        reg = KeyRegistry()
        reg.add("key")
        reg.save(path)
        assert path.exists()

    def test_roundtrip_preserves_order(self, tmp_path):
        path = tmp_path / "registry.json"
        reg = KeyRegistry()
        for i in range(10):
            reg.add(f"key_{i:02d}")
        reg.save(path)
        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == [f"key_{i:02d}" for i in range(10)]


class TestStrictLoadShape:
    """``KeyRegistry.load`` is the single shape predicate for
    ``indexed_key_registry.json``: an ABSENT file loads empty (fresh-install
    contract — pinned by ``TestPersistence.test_load_missing_file`` above,
    not duplicated here), but an EXISTING file must be a dict with a
    list-valued ``"active_keys"`` AND a dict-valued ``"simhash"`` or it is
    refused.
    """

    def test_refuses_non_dict_payload(self, tmp_path):
        """A JSON document that is not an object at all is not a registry file."""
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(_json.dumps(["graph1", "graph2"]))
        with pytest.raises(ValueError, match="not a JSON object"):
            KeyRegistry.load(path)

    def test_refuses_existing_null_payload(self, tmp_path):
        """A file containing the bare JSON literal ``null`` must NOT collapse
        onto the absent-file sentinel.

        ``json.loads("null")`` parses to Python ``None`` — the same value an
        absent file's read short-circuits to before this check ever runs.
        ``load`` distinguishes the two via ``path.exists()`` (checked before
        any parse), so an EXISTING ``null`` file is refused like any other
        non-dict payload rather than silently treated as fresh.
        """
        path = tmp_path / "indexed_key_registry.json"
        path.write_text("null")
        with pytest.raises(ValueError, match="not a JSON object"):
            KeyRegistry.load(path)

    def test_refuses_missing_active_keys(self, tmp_path):
        """A dict payload with no ``"active_keys"`` at all is refused."""
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(_json.dumps({"simhash": {}}))
        with pytest.raises(ValueError, match="active_keys"):
            KeyRegistry.load(path)

    def test_refuses_non_list_active_keys(self, tmp_path):
        """A non-list ``"active_keys"`` is a schema fault, not tolerable input."""
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(_json.dumps({"active_keys": {"graph1": True}, "simhash": {}}))
        with pytest.raises(ValueError, match="active_keys"):
            KeyRegistry.load(path)

    def test_refuses_missing_simhash(self, tmp_path):
        """A dict payload with ``"active_keys"`` but no ``"simhash"`` is refused."""
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(_json.dumps({"active_keys": ["graph1"]}))
        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load(path)

    def test_refuses_non_dict_simhash(self, tmp_path):
        """A ``"simhash"`` field that is not a map is a schema fault, not empty."""
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(_json.dumps({"active_keys": ["graph1"], "simhash": 5}))
        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load(path)

    def test_load_simhashes_delegates_to_load(self, tmp_path):
        """``load_simhashes`` shares the same shape check — no second gate.

        A payload carrying only ``"simhash"`` (no ``"active_keys"``) used to
        pass ``load_simhashes``'s own standalone guard before the collapse;
        it now raises exactly like ``load`` does on the identical file.
        """
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(_json.dumps({"simhash": {"graph1": 1}}))
        with pytest.raises(ValueError, match="active_keys"):
            KeyRegistry.load_simhashes(path)


class TestLoadFromBytes:
    """``KeyRegistry.load_from_bytes`` is ``load``'s parse-and-shape-check
    half, split out so a caller already holding a shadow file's decrypted
    bytes (:func:`~paramem.memory.increment.build_tier_increment`) parses
    them once instead of ``load`` re-reading and re-decrypting the file.
    Same shape predicate, same error shape as ``load`` — no second gate.
    """

    def test_parses_valid_payload(self):
        payload = _json.dumps(
            {"active_keys": ["graph1", "graph2"], "stale": [], "simhash": {"graph1": 7}}
        )
        reg = KeyRegistry.load_from_bytes(payload.encode("utf-8"))
        assert reg.list_active() == ["graph1", "graph2"]
        assert len(reg) == 2

    def test_no_file_io(self, tmp_path, monkeypatch):
        """``load_from_bytes`` touches no filesystem at all — patching
        ``read_maybe_encrypted`` to explode must not affect it."""
        import paramem.backup.encryption as _enc

        def _boom(*a, **kw):
            raise AssertionError("load_from_bytes must not read from disk")

        monkeypatch.setattr(_enc, "read_maybe_encrypted", _boom)
        payload = _json.dumps({"active_keys": [], "stale": [], "simhash": {}})
        reg = KeyRegistry.load_from_bytes(payload.encode("utf-8"))
        assert len(reg) == 0

    def test_refuses_non_dict_payload(self):
        with pytest.raises(ValueError, match="not a JSON object"):
            KeyRegistry.load_from_bytes(_json.dumps(["graph1"]).encode("utf-8"))

    def test_refuses_missing_active_keys(self):
        with pytest.raises(ValueError, match="active_keys"):
            KeyRegistry.load_from_bytes(_json.dumps({"simhash": {}}).encode("utf-8"))

    def test_refuses_missing_simhash(self):
        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load_from_bytes(_json.dumps({"active_keys": []}).encode("utf-8"))

    def test_error_names_path_when_given(self):
        with pytest.raises(ValueError, match="registry.json"):
            KeyRegistry.load_from_bytes(
                _json.dumps({"simhash": {}}).encode("utf-8"), path="registry.json"
            )

    def test_load_delegates_to_load_from_bytes(self, tmp_path):
        """``load`` reads the file once, then hands the bytes to
        ``load_from_bytes`` for parsing — same outcome as before the split."""
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(
            _json.dumps({"active_keys": ["graph1"], "stale": [], "simhash": {"graph1": 3}})
        )
        reg = KeyRegistry.load(path)
        assert reg.list_active() == ["graph1"]


class TestPerTierSchema:
    """Per-tier KeyRegistry: each registry owns one tier's keys.

    The adapter_id concept is now encoded by the tier name in the store's
    per-tier registries (``MemoryStore.registry(tier)``), not by a field
    on the registry record.  These tests verify the single-tier registry
    behaviours that the per-tier pattern relies on.
    """

    def test_add_no_adapter_id_kwarg(self):
        """add(key) with no kwargs works; old adapter_id kwarg is gone."""
        reg = KeyRegistry()
        reg.add("k1")
        assert "k1" in reg

    def test_add_accepts_only_key_positional(self):
        """Positional-only add(key) matches every production call site."""
        reg = KeyRegistry()
        reg.add("graph1")
        assert "graph1" in reg
        # Duplicate add must remain idempotent.
        reg.add("graph1")
        assert len(reg) == 1

    def test_list_active_scoped_to_this_tier(self):
        """list_active() returns only the keys in THIS tier's registry."""
        ep_reg = KeyRegistry()
        ep_reg.add("graph1")
        ep_reg.add("graph2")

        sem_reg = KeyRegistry()
        sem_reg.add("graph3")

        # Each registry is isolated.
        assert ep_reg.list_active() == ["graph1", "graph2"]
        assert sem_reg.list_active() == ["graph3"]

    def test_contains_scoped_to_tier(self):
        """Membership check is local to this tier's registry."""
        ep_reg = KeyRegistry()
        ep_reg.add("graph1")

        sem_reg = KeyRegistry()
        sem_reg.add("graph2")

        assert "graph1" in ep_reg
        assert "graph1" not in sem_reg
        assert "graph2" not in ep_reg
        assert "graph2" in sem_reg

    def test_load_tolerates_unknown_keys(self, tmp_path):
        """Load ignores any on-disk key outside the current schema.

        Forward-tolerant by construction — ``_from_payload`` only reads the
        keys it knows about (``active_keys``, ``stale``, ``simhash``);
        anything else on disk is silently dropped rather than requiring an
        explicit migration whenever the schema changes.
        """
        path = tmp_path / "registry_with_unknown_key.json"
        payload = {
            "active_keys": ["graph1", "graph2", "graph3"],
            "stale": [],
            "simhash": {},
            "some_future_field": {"anything": True},
        }
        path.write_text(_json.dumps(payload))

        loaded = KeyRegistry.load(path)
        assert loaded.list_active() == ["graph1", "graph2", "graph3"]


class TestLoadSimhashes:
    """``KeyRegistry.load_simhashes`` — the single on-disk fingerprint reader.

    It is the leaf both the trial-consolidation gates (one handed-in path) and
    ``MemoryStore.read_simhash_registry_from_disk`` (the adapter-tree walk)
    call, so the encryption read, the file shape and the wrong-file guard live
    here and nowhere else.
    """

    def test_missing_file_returns_empty(self, tmp_path):
        """Absent file = fresh install / untrained tier, not an error."""
        assert KeyRegistry.load_simhashes(tmp_path / "missing.json") == {}

    def test_non_integer_fingerprint_raises(self, tmp_path):
        """A non-int fingerprint is corruption, not a variant — it must raise,
        not be silently filtered.

        Superseded tolerance: this used to silently drop a non-int
        fingerprint and return the rest. Under the single-shape read
        (``KeyRegistry.load_from_bytes``), a non-int ``"simhash"`` value
        fails the whole file's shape check with a loud ``ValueError``
        instead — see ``KeyRegistry._from_payload``'s "no filter" contract.
        """
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(
            _json.dumps(
                {
                    "active_keys": ["graph1", "graph2"],
                    "stale": [],
                    "simhash": {"graph1": 1, "graph2": "x"},
                }
            )
        )

        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load_simhashes(path)

    def test_absent_simhash_section_raises(self, tmp_path):
        """A file with no ``"simhash"`` section cannot answer — it must raise.

        Returning ``{}`` here would un-gate every key of that tier.  Every
        registry ``save_bytes`` writes carries the section, so its absence
        means the caller was handed a foreign file.
        """
        path = tmp_path / "indexed_key_registry.json"
        path.write_text(
            _json.dumps(
                {
                    "active_keys": ["graph1"],
                    "stale": [],
                }
            )
        )

        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load_simhashes(path)

    def test_empty_simhash_section_returns_empty(self, tmp_path):
        """An untrained tier serialises ``"simhash": {}`` — a truthful empty map.

        ``KeyRegistry()`` with no keys is what a registered-but-never-trained
        tier writes, and reading it back is not an error.
        """
        path = tmp_path / "indexed_key_registry.json"
        path.write_bytes(KeyRegistry().save_bytes())

        assert KeyRegistry.load_simhashes(path) == {}

    def test_key_metadata_shape_raises(self, tmp_path):
        """key_metadata.json carries bookkeeping, never a fingerprint.

        Pointing the reader at it must fail loudly — silently returning ``{}``
        would un-gate every key the caller meant to verify.
        """
        path = tmp_path / "key_metadata.json"
        path.write_text(
            _json.dumps(
                {
                    "cycle_count": 3,
                    "promoted_keys": [],
                    "keys": {"graph1": {"speaker_id": "speaker0", "relation_type": "unknown"}},
                }
            )
        )

        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load_simhashes(path)


class TestSaveFromBytesSerializationBarrier:
    """save_bytes() -> digest -> save_from_bytes(payload, path) is THE
    serialization barrier a manifest's registry_sha256 binds against: the
    bytes written to disk must be byte-identical to whatever was hashed.

    Pinned WITHOUT the removed ``_require_consolidating``/``consolidating``
    keyword pair -- collapsed out of the signature because the guard they
    gated was unreachable in production (the ``RuntimeError`` check fired
    only on an explicit falsy ``consolidating``, and no production caller
    ever passed one) and the v5 slot-manifest migration script is a caller
    that cannot answer either parameter honestly (it runs offline, outside
    any consolidation window).
    """

    def test_registry_bytes_written_are_byte_identical_to_the_bytes_hashed(self, tmp_path):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")

        payload = reg.save_bytes()
        digest = hashlib.sha256(payload).hexdigest()

        path = tmp_path / "indexed_key_registry.json"
        reg.save_from_bytes(payload, path)  # positional call, no kwargs

        on_disk = path.read_bytes()
        assert on_disk == payload, "bytes on disk must be byte-identical to the hashed bytes"
        assert hashlib.sha256(on_disk).hexdigest() == digest

    def test_save_from_bytes_signature_has_no_consolidating_kwarg(self):
        params = inspect.signature(KeyRegistry.save_from_bytes).parameters
        assert set(params) == {"self", "payload", "path"}


# ---------------------------------------------------------------------------
# The marker record — a withheld id holds only its id: no timestamp, no
# fingerprint, no other field. Excluded from active enumeration and the
# fingerprint map; retained in list_known() so its bookkeeping row survives
# beside it.
# ---------------------------------------------------------------------------


class TestMarkerRecord:
    def test_marking_a_key_withholds_its_id_and_drops_its_fingerprint(self):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 0xCAFE)

        reg.stale("graph1")

        assert "graph1" not in reg  # active-only __contains__
        assert reg.simhash_for("graph1") is None
        assert not reg.has_simhash("graph1")

    def test_marking_an_already_withheld_key_changes_nothing(self):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")
        before_known = reg.list_known()
        before_stale = reg.list_stale()

        reg.stale("graph1")  # idempotent no-op

        assert reg.list_known() == before_known
        assert reg.list_stale() == before_stale

    def test_marking_an_absent_key_is_a_noop(self):
        reg = KeyRegistry()
        reg.stale("never_added")  # must not raise
        assert reg.list_known() == []

    def test_a_withheld_id_is_known_but_not_active(self):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")

        assert reg.knows("graph1") is True
        assert "graph1" not in reg
        assert "graph1" not in reg.list_active()
        assert "graph1" in reg.list_stale()

    def test_registering_a_withheld_id_as_active_is_refused(self):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")

        with pytest.raises(BookkeepingInvariantViolation):
            reg.add("graph1")

        # The refusal must not have half-mutated anything.
        assert "graph1" not in reg
        assert reg.knows("graph1")

    def test_attaching_a_fingerprint_to_a_withheld_id_is_refused(self):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.stale("graph1")

        with pytest.raises(BookkeepingInvariantViolation):
            reg.set_simhash("graph1", 0xBEEF)

        assert reg.simhash_for("graph1") is None

    def test_the_fingerprint_map_is_exactly_the_active_keys_fingerprints(self):
        reg = KeyRegistry()
        reg.add("with_fp")
        reg.set_simhash("with_fp", 111)
        reg.add("without_fp")  # active, never fingerprinted
        reg.add("was_active_then_staled")
        reg.set_simhash("was_active_then_staled", 222)
        reg.stale("was_active_then_staled")  # fingerprint dropped on transition

        store = MemoryStore()
        store.load_registry("episodic", reg)

        assert store.tier_simhashes("episodic") == {"with_fp": 111}

    def test_known_keys_list_active_in_registration_order_then_withheld_ids_sorted(self):
        reg = KeyRegistry()
        reg.add("z_first_registered")
        reg.add("a_second_registered")
        reg.stale("z_first_registered")
        reg.add("b_third_registered")
        # Withheld ids appear sorted, independent of the order they were staled.
        reg.add("m_to_be_staled")
        reg.stale("m_to_be_staled")

        assert reg.list_active() == ["a_second_registered", "b_third_registered"]
        assert reg.list_stale() == ["m_to_be_staled", "z_first_registered"]
        assert reg.list_known() == [
            "a_second_registered",
            "b_third_registered",
            "m_to_be_staled",
            "z_first_registered",
        ]

    def test_the_serialized_marker_section_is_a_sorted_id_list(self):
        reg = KeyRegistry()
        for key in ("graph9", "graph1", "graph5"):
            reg.add(key)
            reg.stale(key)

        data = _json.loads(reg.save_bytes())

        assert data["stale"] == ["graph1", "graph5", "graph9"]
        assert isinstance(data["stale"], list)

    def test_serialization_of_the_same_registry_is_byte_identical(self):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 42)
        reg.add("graph2")
        reg.stale("graph2")

        first = reg.save_bytes()
        second = reg.save_bytes()

        assert first == second


class TestSingleShapeReadAndItsRaises:
    """``KeyRegistry.load_from_bytes`` refuses any shape but the current one —
    no coercion, no default, no second reading of a foreign schema."""

    def test_an_old_shape_stale_section_fails_the_registry_parse(self):
        """A pre-migration 'stale' section (a dict of per-id records) is
        refused, not coerced into a marker-only set."""
        payload = _json.dumps(
            {
                "active_keys": ["graph1"],
                "stale": {"graph7": {"stale_since": "2026-01-01T00:00:00Z", "simhash": 999}},
                "simhash": {"graph1": 1},
            }
        )
        with pytest.raises(ValueError, match="graph7|stale"):
            KeyRegistry.load_from_bytes(payload.encode("utf-8"))

    def test_a_registry_missing_its_stale_section_fails_the_parse(self):
        payload = _json.dumps({"active_keys": ["graph1"], "simhash": {"graph1": 1}})
        with pytest.raises(ValueError, match="stale"):
            KeyRegistry.load_from_bytes(payload.encode("utf-8"))

    def test_a_fingerprint_naming_a_withheld_id_fails_the_parse(self):
        """A 'simhash' entry naming a withheld id is the persisted form of
        the in-memory refusal (add()/set_simhash() refuse a withheld id) —
        the boundary that meets persisted data enforces the same invariant."""
        payload = _json.dumps(
            {
                "active_keys": ["graph1"],
                "stale": ["graph2"],
                "simhash": {"graph1": 1, "graph2": 2},
            }
        )
        with pytest.raises(ValueError, match="withheld"):
            KeyRegistry.load_from_bytes(payload.encode("utf-8"))

    def test_an_id_in_both_active_and_stale_fails_the_parse(self):
        payload = _json.dumps({"active_keys": ["graph1"], "stale": ["graph1"], "simhash": {}})
        with pytest.raises(ValueError, match="graph1"):
            KeyRegistry.load_from_bytes(payload.encode("utf-8"))

    def test_a_non_hashable_active_keys_member_fails_the_parse_not_a_bare_typeerror(self):
        """A non-string 'active_keys' member (e.g. a dict) used to reach the
        overlap check's `set(active_keys) & set(stale_ids)` unguarded and
        raise a bare, uncaught TypeError -- the documented contract is a
        ValueError naming the path, the same as every other shape defect."""
        payload = _json.dumps({"active_keys": [{"k": 1}], "stale": [], "simhash": {}})
        with pytest.raises(ValueError, match="active_keys"):
            KeyRegistry.load_from_bytes(payload.encode("utf-8"))

    def test_a_bool_fingerprint_is_refused_not_admitted_as_an_int(self):
        """`bool` is a subtype of `int` in Python, so a bare `isinstance(fp,
        int)` admits `True`/`False` as fingerprints -- excluded explicitly."""
        payload = _json.dumps({"active_keys": ["graph1"], "stale": [], "simhash": {"graph1": True}})
        with pytest.raises(ValueError, match="simhash"):
            KeyRegistry.load_from_bytes(payload.encode("utf-8"))


class TestAdoptKeyFrom:
    """``adopt_key_from`` moves an ACTIVE key's membership and fingerprint out
    of one tier's registry into another — the registry-layer half of a key
    changing tier."""

    def test_adopting_a_key_moves_it_active_with_its_fingerprint(self):
        source = KeyRegistry()
        source.add("graph1")
        source.set_simhash("graph1", 0xCAFE)
        dest = KeyRegistry()

        dest.adopt_key_from(source, "graph1")

        assert "graph1" in dest
        assert dest.simhash_for("graph1") == 0xCAFE
        assert not source.knows("graph1")

    def test_adopting_a_key_with_no_fingerprint_moves_it_fingerprint_free(self):
        source = KeyRegistry()
        source.add("graph1")
        dest = KeyRegistry()

        dest.adopt_key_from(source, "graph1")

        assert "graph1" in dest
        assert dest.simhash_for("graph1") is None

    def test_adopting_a_key_the_destination_already_knows_is_refused(self):
        source = KeyRegistry()
        source.add("graph1")
        dest = KeyRegistry()
        dest.add("graph1")  # dest already knows this id

        with pytest.raises(BookkeepingInvariantViolation):
            dest.adopt_key_from(source, "graph1")

        # Neither registry mutated by the failed adoption.
        assert "graph1" in source
        assert "graph1" in dest

    def test_adopting_a_key_the_source_does_not_know_is_refused(self):
        source = KeyRegistry()
        dest = KeyRegistry()

        with pytest.raises(BookkeepingInvariantViolation):
            dest.adopt_key_from(source, "graph1")

        assert not dest.knows("graph1")


class TestWorkingCopyIsolation:
    """``KeyRegistry.working_copy`` -- the registry-layer seed for one
    fold's working universe (``ConsolidationLoop._recall_working_tiers``).
    Pinned directly at the ``KeyRegistry`` level, the exact mechanism the
    fold uses: independence (no shared container, either direction) and the
    ``active_only`` marker-retention contract."""

    def test_a_working_copy_shares_no_container_with_the_live_registry(self):
        live = KeyRegistry()
        live.add("graph1")
        live.set_simhash("graph1", 1)
        live.add("graph2")
        live.stale("graph2")

        working = live.working_copy(active_only=False)

        # Mutate every container on the working copy.
        working.add("graph3")
        working.stale("graph1")
        working.add("graph4")
        working.set_simhash("graph4", 999)

        # The live registry must be untouched.
        assert live.list_active() == ["graph1"]
        assert live.list_stale() == ["graph2"]
        assert live.simhash_for("graph1") == 1
        assert "graph3" not in live
        assert "graph4" not in live

        # And the reverse direction: mutating the live registry after the
        # copy was taken must not reach the working copy either.
        live.add("graph5")
        assert "graph5" not in working

    def test_active_only_true_drops_every_marker(self):
        """A rebuilt (primary) tier's working copy: markers end at this
        tier's own rebuild, so `active_only=True` carries active keys and
        their fingerprints only -- no stale ids survive into the copy at
        all, not even as an empty-fingerprint marker."""
        live = KeyRegistry()
        live.add("graph1")
        live.set_simhash("graph1", 1)
        live.add("graph2")
        live.stale("graph2")

        working = live.working_copy(active_only=True)

        assert working.list_active() == ["graph1"]
        assert working.list_stale() == []
        assert working.list_known() == ["graph1"]
        assert working.simhash_for("graph1") == 1
        assert not working.knows("graph2")

        # And the live registry is untouched by taking the copy.
        assert live.list_stale() == ["graph2"]

    def test_active_only_false_carries_markers(self):
        """A candidate (dedup-only) tier's working copy: this event does
        not rebuild it, so its markers must survive into the working copy
        -- `active_only=False` is the full active-union-stale universe."""
        live = KeyRegistry()
        live.add("graph1")
        live.add("graph2")
        live.stale("graph2")

        working = live.working_copy(active_only=False)

        assert working.list_active() == ["graph1"]
        assert working.list_stale() == ["graph2"]
        assert working.knows("graph2")
