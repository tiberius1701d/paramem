"""Tests for the post-fold entry-cache refill: the materialized medium on
``_build_store_contents`` and its wiring from ``_run_full_cycle``.

Splits out one boundary from the three existing modules that already touch
``_build_store_contents`` (``test_preload_failfast.py`` — CUDA fail-fast
containment; ``test_cooldown_gate.py`` — gate ordering; and
``test_live_reload_base_model.py`` — the in-process reload primitive, which
has already drifted into hosting builder and ``_finalize_full`` tests).

Unlike those modules' ``_verified_bindings`` (MagicMock registries with no
fingerprints), this module builds REAL ``KeyRegistry`` objects with real
``set_simhash`` fingerprints — the materialized medium's acceptance gate
(``MemoryStore.probe``'s SimHash check) is a no-op against a fingerprint-free
registry, so a MagicMock-based fixture cannot exercise it.  This is a
different fixture, not a fourth copy of the existing helper.

Design context: a weights-venue fold's training gate probe already proves
every rebuilt tier's content, all-or-refuse, before promoting the staged
weights (``ConsolidationLoop._assert_tier_recall``); the fold's own
main-tier state rebuild (``_rebuild_main_tier_state``) writes that SAME
verified content into the live store's entry cache.  So a post-fold refill
that already holds a ``entries_gate_attested=True`` result can hydrate the
store's entry cache from a plain ``loop.store.iter_entries()`` snapshot
instead of re-probing the promoted weights a second time over the same
content — see ``_build_store_contents``'s ``materialized_entries`` keyword.

The wiring helpers (``_make_dispatch_state``, ``_run_sync``) are imported
from ``test_consolidate_dispatch.py`` rather than re-implemented here — that
module already owns the ``_run_full_consolidation_sync`` driving harness.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from paramem.adapters.registry_binding import VERIFIED, TierBinding
from paramem.memory.entry import entry_simhash
from paramem.memory.store import MemoryStore
from paramem.server.config import load_server_config
from paramem.training.key_registry import KeyRegistry
from tests.server.test_consolidate_dispatch import _make_dispatch_state, _run_sync

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tier_binding(tier: str, registry: KeyRegistry) -> TierBinding:
    """Wrap *registry* into the VERIFIED, publishable ``TierBinding`` shape
    ``verify_adapter_tree`` returns."""
    from pathlib import Path

    return TierBinding(
        tier=tier,
        tier_root=Path(f"/fake/{tier}"),
        status=VERIFIED,
        registry=registry,
        registry_present=True,
        slot=None,
        manifest=None,
        candidate_count=0,
        detail="",
    )


def _registry_with(entries_by_key: dict) -> KeyRegistry:
    """Build a REAL ``KeyRegistry`` with every key in *entries_by_key* active,
    each fingerprinted from its OWN content via :func:`entry_simhash` — the
    same primitive :func:`~paramem.memory.entry.build_registry` uses, so a
    materialized entry with content identical to what minted the fingerprint
    verifies, and a mismatched one is genuinely rejected."""
    reg = KeyRegistry()
    for key, entry in entries_by_key.items():
        reg.add(key)
        reg.set_simhash(key, entry_simhash(entry))
    return reg


def _server_config(tmp_path, *, preload_cache: bool = True):
    """The project's canonical test config, scoped to *tmp_path* so
    ``key_metadata.json`` reads/incident writes land in the pytest tmp tree
    rather than the operator's real data dir."""
    cfg = load_server_config("tests/fixtures/server.yaml")
    cfg.inference.preload_cache = preload_cache
    cfg.paths.data = tmp_path
    return cfg


def _entry(key: str, subject: str = "Alice", predicate: str = "lives_in", obj: str = "Berlin"):
    return {"key": key, "subject": subject, "predicate": predicate, "object": obj}


# ---------------------------------------------------------------------------
# Builder-medium contract — _build_store_contents(materialized_entries=...)
# ---------------------------------------------------------------------------


class TestMaterializedMedium:
    def test_seeded_tier_publishes_from_map_no_source_built(self, tmp_path):
        """Every enumerated active key resolves from the map, and neither
        source class is ever constructed -- no GPU work, no cooldown wait."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.memory.source as source_mod
        import paramem.server.app as app_module

        entry = _entry("graph1")
        registry = _registry_with({"graph1": entry})
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)

        class _RaisingSource:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "no MemorySource may be constructed on the materialized medium"
                )

        cooldown_calls: list = []

        with (
            patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings),
            patch.object(source_mod, "WeightMemorySource", _RaisingSource),
            patch.object(source_mod, "DiskMemorySource", _RaisingSource),
            patch.object(
                app_module,
                "wait_for_cooldown",
                side_effect=lambda *a, **kw: cooldown_calls.append((a, kw)),
            ),
        ):
            new_entries, _new_registry, _new_bookkeeping, stats = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": entry},
            )

        assert cooldown_calls == [], "wait_for_cooldown must not run on the materialized medium"
        assert new_entries["episodic"]["graph1"] == entry
        assert stats["boot_degraded"] is None

    def test_tier_comes_from_staged_registry_not_from_the_input(self, tmp_path):
        """A key whose live entry sat under episodic but whose staged
        registry lists it under semantic publishes under semantic --
        promotion re-bucketing falls out of the enumeration, never the
        input map's shape."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        entry = _entry("graph1", predicate="prefers", obj="tea")
        registry = _registry_with({"graph1": entry})
        # graph1's fingerprint is staged under semantic ONLY.
        bindings = {"semantic": _tier_binding("semantic", registry)}
        cfg = _server_config(tmp_path)

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, *_ = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": entry},
            )

        assert new_entries.get("episodic", {}).get("graph1") is None
        assert new_entries["semantic"]["graph1"] == entry

    def test_erased_key_drops(self, tmp_path):
        """A key present in the materialized map but absent from every
        registry's list_active() is absent from new_entries -- it is never
        even enumerated."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        surviving = _entry("graph1")
        registry = _registry_with({"graph1": surviving})  # graph_erased NOT registered
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)
        materialized = {"graph1": surviving, "graph_erased": _entry("graph_erased")}

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, *_ = app_module._build_store_contents(
                cfg, model=MagicMock(), tokenizer=MagicMock(), materialized_entries=materialized
            )

        assert set(new_entries["episodic"]) == {"graph1"}

    def test_residual_miss_sets_boot_degraded(self, tmp_path):
        """An active key with no entry in the map is a miss: hits < total,
        boot_degraded['reason'] == 'preload_partial', and the key is listed
        in missed_by_tier."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        present = _entry("graph1")
        missing = _entry("graph2", subject="Bob", predicate="likes", obj="tea")
        registry = _registry_with({"graph1": present, "graph2": missing})
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, _reg, _bk, stats = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": present},
            )

        assert set(new_entries["episodic"]) == {"graph1"}
        degraded = stats["boot_degraded"]
        assert degraded is not None
        assert degraded["reason"] == "preload_partial"
        assert degraded["hits"] == 1
        assert degraded["total"] == 2
        assert "graph2" in degraded["missed_by_tier"]["episodic"]
        assert degraded["source"] == "fold_entries"

    def test_simhash_gate_rejects_mismatched_carried_forward_entry(self, tmp_path):
        """A live entry whose content disagrees with the registry's
        fingerprint is dropped to a miss -- not published, and no exception
        propagates (drop-to-miss, never fail-loud)."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        registered = _entry("graph1", subject="Alice")
        registry = _registry_with({"graph1": registered})
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)
        # Carried-forward content disagrees with the fingerprint minted above.
        mismatched = _entry("graph1", subject="Someone Else Entirely")

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, _reg, _bk, stats = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": mismatched},
            )

        assert "graph1" not in new_entries.get("episodic", {})
        assert stats["boot_degraded"]["reason"] == "preload_partial"

    def test_content_only_projection(self, tmp_path):
        """A materialized entry carrying speaker_id (or any other
        provenance field) publishes as exactly {key, subject, predicate,
        object} -- content_only_entry applied at both the write-in and the
        read-out side of the transient store."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        entry = _entry("graph1")
        registry = _registry_with({"graph1": entry})
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)
        carrying_provenance = {**entry, "speaker_id": "speaker0", "confidence": 0.99}

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, *_ = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": carrying_provenance},
            )

        published = new_entries["episodic"]["graph1"]
        assert published == entry
        assert set(published) == {"key", "subject", "predicate", "object"}

    def test_preload_cache_false_ignores_the_map(self, tmp_path):
        """preload_cache=False stays entry-empty regardless of the
        materialized medium -- the opt-out is medium-agnostic."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        entry = _entry("graph1")
        registry = _registry_with({"graph1": entry})
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path, preload_cache=False)

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, _reg, _bk, stats = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": entry},
            )

        assert new_entries == {}
        assert stats["boot_degraded"] is None

    def test_malformed_candidate_is_skipped_not_put(self, tmp_path):
        """A materialized candidate missing a content field is skipped at
        write-in -- never handed to content_only_entry (which would raise
        KeyError) -- and comes back a clean miss, exactly like a candidate
        absent from the map entirely."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        present = _entry("graph1")
        registry = _registry_with({"graph1": present})
        registry.add("graph2")  # active, but the candidate below is malformed
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)
        malformed = {"key": "graph2", "subject": "Bob"}  # missing predicate/object

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, _reg, _bk, stats = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries={"graph1": present, "graph2": malformed},
            )

        assert set(new_entries["episodic"]) == {"graph1"}
        assert "graph2" in stats["boot_degraded"]["missed_by_tier"]["episodic"]


class TestDuplicateEntryDefense:
    """Defense test, not contract: with the store-consistency fix in place
    (``_rebuild_main_tier_state`` writes every key's content into its NEW
    tier and removes any stale copy under its OLD tier;
    ``drop_registry_and_entries`` pops the retired interim tier's
    ``_entries`` bucket alongside its registry), a live key can no longer
    sit in two ``_entries`` buckets at once.  If that invariant were ever
    violated again, this pins the outcome as deterministic rather than
    silently ambiguous."""

    def test_duplicate_key_resolves_to_the_last_iter_entries_copy_and_registry_tier(self, tmp_path):
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        store = MemoryStore(replay_enabled=True)
        stale_copy = _entry("graph1", subject="Stale Old Content")
        fresh_copy = _entry("graph1", subject="Alice")
        # Simulate a duplicate: the same key's content sitting under two
        # _entries buckets at once (an interim tier's stale leftover plus
        # the main tier the rebuild wrote it into).  Direct _entries
        # mutation because the store's own public API (put/move) enforces
        # single-tier ownership and cannot construct this state -- it is
        # reachable only by the invariant violation this test defends
        # against, never through the store's contract.
        store._entries.setdefault("episodic_interim_20260101T0000", {})["graph1"] = stale_copy
        store._entries.setdefault("episodic", {})["graph1"] = fresh_copy

        # Constructed exactly as _run_full_cycle does: a flat {key: entry}
        # map over iter_entries() -- insertion order of _entries determines
        # which copy wins a duplicate key (episodic was inserted second
        # above, so its copy is the last one the comprehension writes).
        materialized = {key: entry for _tier, key, entry in store.iter_entries()}
        assert materialized["graph1"] == fresh_copy, (
            "the flat map must keep the LAST iter_entries() copy, deterministically"
        )

        registry = _registry_with({"graph1": materialized["graph1"]})
        bindings = {"episodic": _tier_binding("episodic", registry)}
        cfg = _server_config(tmp_path)

        with patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings):
            new_entries, *_ = app_module._build_store_contents(
                cfg,
                model=MagicMock(),
                tokenizer=MagicMock(),
                materialized_entries=materialized,
            )

        # The builder never enumerates the interim tier at all -- it has no
        # registry binding, so it contributes nothing regardless of what
        # its stale _entries bucket still holds.
        assert "episodic_interim_20260101T0000" not in new_entries
        assert new_entries["episodic"]["graph1"] == fresh_copy


# ---------------------------------------------------------------------------
# Wiring — _run_full_cycle's materialized-map construction and its all-
# or-nothing propagation through _finalize_full
# ---------------------------------------------------------------------------


class TestRunFullCycleWiring:
    def test_attested_fold_materialized_equals_the_store_snapshot(self, monkeypatch, tmp_path):
        """entries_gate_attested=True -> materialized_entries is exactly the
        flat {key: entry} snapshot of loop.store.iter_entries() -- no
        overlay, no filtering."""
        import paramem.server.app as app_module

        store = MemoryStore(replay_enabled=True)
        store.put("episodic", "graph1", _entry("graph1"), register=True)
        store.put("semantic", "graph2", _entry("graph2", subject="Bob"), register=True)

        consolidate_return = {
            "tiers_rebuilt": ["episodic"],
            "entries_gate_attested": True,
            "graph_drift_count": 0,
            "rolled_back": False,
        }
        # Snapshot BEFORE running the cycle -- _finalize_full's store.swap()
        # (fed the fake builder's empty return below) rebinds the store's
        # live entries, so an expected-value snapshot taken afterward would
        # observe the swapped-in (empty) state instead of the pre-cycle one.
        expected = {key: entry for _tier, key, entry in store.iter_entries()}
        state = _make_dispatch_state(
            tmp_path=tmp_path, store=store, consolidate_return=consolidate_return
        )

        captured: dict = {}

        def _fake_build(config, *, model, tokenizer, should_abort=None, materialized_entries=None):
            captured["materialized_entries"] = materialized_entries
            return ({}, {}, {}, {"boot_degraded": None, "tier_bindings": {}})

        with (
            patch.object(app_module, "_build_store_contents", side_effect=_fake_build),
            patch.object(app_module, "_revalidate_adapter_manifests"),
        ):
            _run_sync(state, monkeypatch)

        assert captured["materialized_entries"] == expected
        assert captured["materialized_entries"] is not None

    def test_unattested_fold_materialized_is_none(self, monkeypatch, tmp_path):
        """entries_gate_attested False (or absent, the disk-venue shape) ->
        materialized_entries is None -> the source-medium path is preserved."""
        import paramem.server.app as app_module

        store = MemoryStore(replay_enabled=True)
        store.put("episodic", "graph1", _entry("graph1"), register=True)

        consolidate_return = {
            "tiers_rebuilt": ["episodic"],
            "entries_gate_attested": False,
            "graph_drift_count": 0,
            "rolled_back": False,
        }
        state = _make_dispatch_state(
            tmp_path=tmp_path, store=store, consolidate_return=consolidate_return
        )

        captured: dict = {}

        def _fake_build(config, *, model, tokenizer, should_abort=None, materialized_entries=None):
            captured["materialized_entries"] = materialized_entries
            return ({}, {}, {}, {"boot_degraded": None, "tier_bindings": {}})

        with (
            patch.object(app_module, "_build_store_contents", side_effect=_fake_build),
            patch.object(app_module, "_revalidate_adapter_manifests"),
        ):
            _run_sync(state, monkeypatch)

        assert captured["materialized_entries"] is None

    def test_all_or_nothing_preserved_when_the_builder_raises(self, monkeypatch, tmp_path):
        """_build_store_contents raising leaves staged=None, so the
        finalizer performs no swap -- the caller's try/except + staged=None
        contract is untouched by the new kwarg."""
        import paramem.server.app as app_module

        store = MagicMock()
        store.replay_enabled = True
        store.iter_entries.return_value = []
        store.all_active_keys.return_value = []

        consolidate_return = {
            "tiers_rebuilt": ["episodic"],
            "entries_gate_attested": True,
            "graph_drift_count": 0,
            "rolled_back": False,
        }
        state = _make_dispatch_state(
            tmp_path=tmp_path, store=store, consolidate_return=consolidate_return
        )

        with (
            patch.object(
                app_module, "_build_store_contents", side_effect=RuntimeError("builder exploded")
            ),
            patch.object(app_module, "_revalidate_adapter_manifests"),
        ):
            _run_sync(state, monkeypatch)

        store.swap.assert_not_called()

    def test_end_to_end_materialized_medium_through_real_build_store_contents(
        self, monkeypatch, tmp_path
    ):
        """No stubbing of the builder: a real MemoryStore, real KeyRegistry
        fingerprints, and the real _build_store_contents invoked by the
        real _run_full_cycle wiring path.  A malformed live entry (missing
        content fields) must resolve as a clean miss end to end, never an
        unhandled exception that aborts the whole publish."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.server.app as app_module

        store = MemoryStore(replay_enabled=True)
        good = _entry("graph1")
        store.put("episodic", "graph1", good, register=True)
        # Malformed content reachable only via a bug elsewhere in the
        # pipeline (never through the store's own put() contract) -- the
        # materialized medium must survive it as a miss, not a crash.
        store._entries["episodic"]["graph2"] = {"key": "graph2"}

        registry = _registry_with({"graph1": good})
        registry.add("graph2")
        bindings = {"episodic": _tier_binding("episodic", registry)}

        consolidate_return = {
            "tiers_rebuilt": ["episodic"],
            "entries_gate_attested": True,
            "graph_drift_count": 0,
            "rolled_back": False,
        }
        state = _make_dispatch_state(
            tmp_path=tmp_path, store=store, consolidate_return=consolidate_return
        )
        state["config"].inference.preload_cache = True
        state["config"].key_metadata_path = tmp_path / "key_metadata.json"

        with (
            patch.object(registry_binding_mod, "verify_adapter_tree", return_value=bindings),
            patch.object(app_module, "_revalidate_adapter_manifests"),
        ):
            _run_sync(state, monkeypatch)

        # _finalize_full's swap() rebound the store from the real builder's
        # output -- the well-formed key survived end to end, and the
        # malformed one resolved as a miss rather than aborting the publish.
        assert store.get("graph1") == good
        assert store.get("graph2") is None
