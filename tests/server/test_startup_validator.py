"""Tests for eager consolidation-loop creation at boot, and the boot binding
verification suite rebuilt against the venue-uniform target design.

``TestValidateAdapterSlotBindingDecisions`` covers ``_validate_adapter_slot``
directly -- the per-tier mount decision built on a real, already-resolved
``TierBinding`` (no model needed for the branches exercised here: the
KEYS_WITHOUT_SLOT unpublishable case, the PAYLOAD_MISMATCH unpublishable
case, and the VERIFIED-simulate never-mount all return before the fingerprint check that
needs one). ``TestBootLevelQuarantine``
drives the whole boot/lift store step (``_hydrate_memory_store_in_place``,
``model=None, tokenizer=None`` -- registry and bookkeeping hydration need no
model) against synthetic on-disk trees, proving the quarantine loudly names
the offending tier(s) and (for the unmigrated-tree case) that no
consolidation action dispatches while a main tier's binding stays
unverified.

What remains from the original suite is orthogonal: eager
``ConsolidationLoop`` creation at boot (``_eager_create_consolidation_loop``)
and the lifespan wiring that calls it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _build_content_increment(*, tier: str, keys: "list[dict]", pre_sha: str = "") -> object:
    """One ``TierIncrement`` carrying *keys* (each ``{key, subject,
    predicate, object}``) as a simulate payload, plus a real bookkeeping row
    for every key.

    Distinct from ``tests.test_memory_persistence._build_simulate_increment``
    (whose ``rows_bytes`` carries an empty ``keys`` map, fine for its own
    binding-only assertions): the boot store step
    (``app._build_store_contents``) also loads bookkeeping and raises
    ``BookkeepingInvariantViolation`` for any active key with no row, so
    every key here gets one.
    """
    import json as _json

    from paramem.memory.entry import entry_simhash
    from paramem.memory.increment import TierIncrement
    from paramem.training.key_registry import KeyRegistry

    registry = KeyRegistry()
    rows: dict = {}
    keyed: list[dict] = []
    for spec in keys:
        registry.add(spec["key"])
        registry.set_simhash(spec["key"], entry_simhash(spec))
        rows[spec["key"]] = {
            "speaker_id": "speaker0",
            "relation_type": "factual",
            "first_seen": "2026-01-01T00:00:00Z",
            "promoted": False,
            "reinforcement_count": 1,
            "last_reinforced_cycle": 0,
            "last_seen": "2026-01-01T00:00:00Z",
        }
        keyed.append(
            {
                "key": spec["key"],
                "subject": spec["subject"],
                "predicate": spec["predicate"],
                "object": spec["object"],
                "speaker_id": "speaker0",
            }
        )
    registry_bytes = registry.save_bytes()
    rows_bytes = _json.dumps({"tier_cycle": 0, "keys": rows}).encode("utf-8")

    return TierIncrement(
        tier=tier,
        adapter_name=tier,
        registry=registry,
        registry_bytes=registry_bytes,
        rows_bytes=rows_bytes,
        entries={},
        bookkeeping={},
        keyed=keyed,
        rebuilt=True,
        pre_sha=pre_sha,
    )


def _build_write_context(output_dir) -> object:
    from paramem.memory.increment import TierWriteContext

    return TierWriteContext(
        model=None,
        tokenizer=None,
        fingerprint_cache={},
        output_dir=output_dir,
        tier_configs={},
        store=None,
        keep_prior_slots=1,
    )


class TestEagerConsolidationLoopCreation:
    def test_creates_loop_when_model_tokenizer_and_store_present(self) -> None:
        """Local-mode state (model + tokenizer + memory_store all present)
        creates the consolidation loop via the shared get-or-create.

        ``_eager_create_consolidation_loop`` is zero-arg — it reads
        ``_state`` directly rather than taking a ``config`` parameter."""
        from paramem.server import app as app_module

        state = {
            "config": MagicMock(name="config"),
            "model": MagicMock(name="model"),
            "tokenizer": MagicMock(name="tokenizer"),
            "memory_store": MagicMock(name="memory_store"),
            "consolidation_loop": None,
        }

        with (
            patch.object(app_module, "_state", state),
            patch.object(app_module, "get_or_create_consolidation_loop") as mock_get_or_create,
        ):
            app_module._eager_create_consolidation_loop()

        mock_get_or_create.assert_called_once_with(state)

    def test_noop_in_cloud_only_mode(self) -> None:
        """No model resident (cloud-only) — the loop is never created."""
        from paramem.server import app as app_module

        state = {
            "config": MagicMock(name="config"),
            "model": None,
            "tokenizer": None,
            "memory_store": None,
            "consolidation_loop": None,
        }

        with (
            patch.object(app_module, "_state", state),
            patch.object(app_module, "get_or_create_consolidation_loop") as mock_get_or_create,
        ):
            app_module._eager_create_consolidation_loop()

        mock_get_or_create.assert_not_called()

    def test_noop_when_loop_already_exists(self) -> None:
        """Idempotent: a second call (loop already resident) does not build
        another one — proven through the real create_consolidation_loop
        factory, not just a mocked get-or-create.

        ``create_consolidation_loop`` is patched at its DEFINING module
        (``paramem.server.consolidation``), not at ``app_module`` — the
        get-or-create it backs resolves the name via its own
        ``__globals__`` when it runs, so a patch placed on the importing
        module (``app_module``) has no effect."""
        from paramem.server import app as app_module
        from paramem.server import consolidation as consolidation_module

        state = {
            "config": MagicMock(name="config"),
            "model": MagicMock(name="model"),
            "tokenizer": MagicMock(name="tokenizer"),
            "memory_store": MagicMock(name="memory_store"),
            "consolidation_loop": None,
        }
        fake_loop = MagicMock(name="loop")
        fake_loop.model = state["model"]

        with (
            patch.object(app_module, "_state", state),
            patch.object(
                consolidation_module, "create_consolidation_loop", return_value=fake_loop
            ) as mock_create,
        ):
            app_module._eager_create_consolidation_loop()
            app_module._eager_create_consolidation_loop()

        mock_create.assert_called_once()


class TestLifespanEagerLoopWiring:
    def test_lifespan_invokes_eager_create_consolidation_loop(self) -> None:
        """The lifespan boot path must call _eager_create_consolidation_loop
        after the memory store is built, so a refactor that drops the call
        fails here rather than silently."""
        import inspect

        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert "_eager_create_consolidation_loop(" in source, (
            "lifespan must call _eager_create_consolidation_loop so "
            "adapter_loaded is symmetric across a restart"
        )


class TestValidateAdapterSlotBindingDecisions:
    """``_validate_adapter_slot``'s per-tier mount decision, driven by a
    real ``TierBinding`` -- no model needed for either branch here, since
    both return before the fingerprint check that requires a loaded model."""

    def test_a_simulate_tier_with_active_keys_and_no_manifest_is_withheld_and_named(
        self, tmp_path
    ) -> None:
        """A tier whose registry holds an active key but carries no
        weight-slot candidate at all (the pre-upgrade shape, or a slot
        lost between the last two store steps) resolves KEYS_WITHOUT_SLOT --
        held back from mount, and named in ``manifest_status`` for the
        operator, not silently dropped."""
        from paramem.adapters.registry_binding import verify_tier_binding
        from paramem.server.app import _validate_adapter_slot
        from paramem.training.key_registry import KeyRegistry

        tier_root = tmp_path / "adapters" / "episodic"
        tier_root.mkdir(parents=True)
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        registry.save(tier_root / "indexed_key_registry.json")
        # No weight-slot candidate on disk at all.

        binding = verify_tier_binding("episodic", tier_root)
        assert binding.status == "keys_without_slot"

        manifest_status: dict = {}
        slot, manifest, should_mount = _validate_adapter_slot(
            "episodic", MagicMock(), MagicMock(), tier_root, binding, manifest_status
        )

        assert should_mount is False
        assert slot is None
        assert manifest is None
        row = manifest_status["episodic"]
        assert row["status"] == "keys_without_slot"
        assert row["severity"] == "red"  # episodic is the primary tier

    def test_a_graph_payload_verifies_and_is_never_mounted(self, tmp_path) -> None:
        """A bound, VERIFIED simulate slot is healthy -- ``slot``/``manifest``
        are returned -- but there are no PEFT weights to mount, so
        ``should_mount`` is False and no manifest_status row is minted
        (a healthy simulate slot is not a problem to report)."""
        from paramem.adapters.registry_binding import VERIFIED, verify_tier_binding
        from paramem.server.app import _validate_adapter_slot
        from paramem.training.key_registry import KeyRegistry
        from tests._fold_fixtures import _write_graph

        tier_root = tmp_path / "adapters" / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")
        _write_graph(
            tier_root,
            [
                {
                    "key": "graph1",
                    "subject": "alice",
                    "predicate": "lives_in",
                    "object": "berlin",
                    "speaker_id": "speaker0",
                }
            ],
        )

        binding = verify_tier_binding("episodic", tier_root)
        assert binding.status == VERIFIED
        assert binding.manifest.payload.kind == "simulate"

        manifest_status: dict = {}
        slot, manifest, should_mount = _validate_adapter_slot(
            "episodic", MagicMock(), MagicMock(), tier_root, binding, manifest_status
        )

        assert should_mount is False
        assert slot == binding.slot
        assert manifest is binding.manifest
        assert "episodic" not in manifest_status

    def test_a_payload_mismatch_binding_is_withheld_and_named(self, tmp_path) -> None:
        """A bound train slot whose payload bytes no longer hash to the
        manifest's stamped digest resolves PAYLOAD_MISMATCH -- held back
        from mount, and named in ``manifest_status`` with the binding's own
        detail. Driven through the REAL ``_validate_adapter_slot`` against a
        real ``TierBinding``, never a hand-built manifest_status row."""
        from paramem.adapters.manifest import tier_registry_sha256
        from paramem.adapters.registry_binding import PAYLOAD_MISMATCH, verify_tier_binding
        from paramem.adapters.slot import write_slot
        from paramem.server.app import _validate_adapter_slot
        from paramem.training.key_registry import KeyRegistry
        from tests._manifest_fixtures import make_train_manifest, write_slot_files

        tier_root = tmp_path / "adapters" / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")
        registry_hash = tier_registry_sha256(tier_root)
        manifest = make_train_manifest(name="episodic", registry_sha256=registry_hash, key_count=1)
        slot = write_slot(
            tier_root,
            manifest=manifest,
            write_payload=lambda pending: write_slot_files(pending),
        )
        # Corrupt the written payload bytes in place -- the digest stamped
        # into the manifest at write time no longer matches.
        (slot / "adapter_model.safetensors").write_bytes(b"corrupted-not-what-was-written")

        binding = verify_tier_binding("episodic", tier_root)
        assert binding.status == PAYLOAD_MISMATCH

        manifest_status: dict = {}
        result_slot, result_manifest, should_mount = _validate_adapter_slot(
            "episodic", MagicMock(), MagicMock(), tier_root, binding, manifest_status
        )

        assert should_mount is False
        assert result_slot is None
        assert result_manifest is None
        row = manifest_status["episodic"]
        assert row["status"] == "payload_mismatch"
        assert row["severity"] == "red"  # episodic is the primary tier
        assert row["slot_path"] == slot.name

    def test_an_unrecognized_verdict_is_withheld_loudly_never_mounted(self, tmp_path) -> None:
        """A ``TierBinding`` carrying a verdict string outside today's
        closed vocabulary (simulating a future verdict added without a
        matching branch here) must never fall through into the VERIFIED
        handling and dereference ``binding.manifest`` (``None`` here) --
        it is left unpublishable and named in ``manifest_status`` like every
        other non-VERIFIED verdict, and mounts nothing."""
        from paramem.adapters.registry_binding import TierBinding
        from paramem.server.app import _validate_adapter_slot

        tier_root = tmp_path / "adapters" / "episodic"
        tier_root.mkdir(parents=True)
        binding = TierBinding(
            tier="episodic",
            tier_root=tier_root,
            status="some_future_verdict",
            registry=None,
            registry_present=False,
            slot=None,
            manifest=None,
            candidate_count=0,
            detail="synthetic verdict for the totality test",
        )

        manifest_status: dict = {}
        slot, manifest, should_mount = _validate_adapter_slot(
            "episodic", MagicMock(), MagicMock(), tier_root, binding, manifest_status
        )

        assert should_mount is False
        assert slot is None
        assert manifest is None
        row = manifest_status["episodic"]
        assert row["status"] == "unrecognized_verdict"
        assert row["severity"] == "red"  # episodic is the primary tier


class TestSweepKeylessTierArtifactsKeyedNoCandidateArm:
    """``_sweep_keyless_tier_artifacts``'s KEYS_WITHOUT_SLOT arm: a tier
    whose registry holds a known key but has zero slot candidates at all,
    in either venue, is PRESERVED (never reaped) -- this shape can be a
    crash-interrupted erase or a torn training write, and deleting facts
    never folded anywhere else would be silent data loss."""

    def test_a_keyed_tier_with_no_candidate_at_all_is_preserved_not_reaped(self, tmp_path) -> None:
        from paramem.server.app import _sweep_keyless_tier_artifacts
        from paramem.training.key_registry import KeyRegistry

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")
        # No slot candidate on disk at all, in either venue.

        config = MagicMock()
        config.adapter_dir = adapter_dir
        state: dict = {}

        reaped = _sweep_keyless_tier_artifacts(config, state)

        assert "episodic" not in reaped
        # The registry file is untouched -- still on disk with its key.
        registry_after = KeyRegistry.load(tier_root / "indexed_key_registry.json")
        assert registry_after.list_active() == ["graph1"]

    def test_a_tier_holding_only_markers_is_preserved_by_the_boot_sweep(
        self, tmp_path, caplog
    ) -> None:
        """A registry whose only known keys are WITHHELD (list_known()
        non-empty via markers alone, list_active() empty) with zero slot
        candidates is preserved SILENTLY -- distinct from the active-key
        sibling above, which logs an ERROR naming the known-key count. A
        marker-only tier is not this sweep's to report: the mount stage
        handles whatever binding status a keyed tier resolves to."""
        import logging

        from paramem.server.app import _sweep_keyless_tier_artifacts
        from paramem.training.key_registry import KeyRegistry

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.stale("graph1")
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")
        # No slot candidate on disk at all, in either venue.

        config = MagicMock()
        config.adapter_dir = adapter_dir
        state: dict = {}

        with caplog.at_level(logging.ERROR):
            reaped = _sweep_keyless_tier_artifacts(config, state)

        assert "episodic" not in reaped
        registry_after = KeyRegistry.load(tier_root / "indexed_key_registry.json")
        assert registry_after.list_known() == ["graph1"]
        assert registry_after.list_active() == []
        # Silent -- no ERROR naming this tier (unlike the active-key case).
        assert not any(
            record.levelno == logging.ERROR and "episodic" in record.getMessage()
            for record in caplog.records
        )


class TestBootLevelQuarantine:
    """The whole boot/lift store step (``_hydrate_memory_store_in_place``)
    against synthetic on-disk trees -- no model, no GPU: a train-venue call
    with no resident model still hydrates registries/bookkeeping fine, and
    the failure path (an unpublishable tier) never reaches the entry-preload
    section that would need one."""

    def test_a_pre_upgrade_simulate_tree_boots_quarantined_with_its_tiers_named(
        self, tmp_path, monkeypatch
    ) -> None:
        """A pre-upgrade simulate tier -- content landed as a bare tier-root
        ``graph.json`` with no accompanying slot ``meta.json`` -- is reaped
        by boot's own cleanup_partial_slots sweep as scratch, leaving an
        active-keyed registry with zero slot candidates. The whole store
        boots quarantined, and the cause names the tier."""
        import paramem.server.app as app_module
        from paramem.backup.integrity import cleanup_partial_slots
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        tier_root.mkdir(parents=True)
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        registry.save(tier_root / "indexed_key_registry.json")
        # The pre-upgrade shape: graph.json lands inside a freshly allocated
        # slot dir with no meta.json -- a candidate by neither name.
        stray_slot = tier_root / "20260101-000000"
        stray_slot.mkdir()
        (stray_slot / "graph.json").write_text('{"directed": true}')

        removed = cleanup_partial_slots(adapter_dir)
        assert len(removed) == 1
        assert not stray_slot.exists()

        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.paths.data = tmp_path / "data"
        cfg.paths.data.mkdir(parents=True, exist_ok=True)

        state = {"store_quarantine": None}
        monkeypatch.setattr(app_module, "_state", state)

        store = MemoryStore()
        published = app_module._hydrate_memory_store_in_place(
            store, cfg, model=None, tokenizer=None
        )

        assert published is False
        assert state["store_quarantine"] is not None
        cause = state["store_quarantine"]["cause"]
        assert cause["exception_type"] == "TierBindingUnpublishable"
        assert "episodic" in cause["message"]
        assert "keys_without_slot" in cause["message"]

    def test_an_unmigrated_tree_is_withheld_and_quarantined_loudly(
        self, tmp_path, monkeypatch
    ) -> None:
        """Two main tiers carrying prior-schema (v4) manifests -- unreadable
        by the current-schema-only reader -- resolve NO_MATCHING_SLOT (their
        registry hash matches no readable candidate). The store boots
        quarantined naming both tiers, and no consolidation action
        dispatches while that drift stands."""
        import paramem.server.app as app_module
        from paramem.adapters.manifest import tier_registry_sha256
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry
        from tests._manifest_fixtures import (
            v4_train_meta_dict,
            write_raw_meta,
            write_slot_files,
        )

        adapter_dir = tmp_path / "adapters"
        for tier_name in ("episodic", "semantic"):
            tier_root = adapter_dir / tier_name
            key = f"{tier_name[:3]}1"
            registry = KeyRegistry()
            registry.add(key)
            registry.set_simhash(key, 1)
            tier_root.mkdir(parents=True)
            registry.save(tier_root / "indexed_key_registry.json")
            reg_hash = tier_registry_sha256(tier_root)
            slot = tier_root / "20260101-000000"
            write_slot_files(slot)
            write_raw_meta(
                slot, v4_train_meta_dict(name=tier_name, registry_sha256=reg_hash, key_count=1)
            )

        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.paths.data = tmp_path / "data"
        cfg.paths.data.mkdir(parents=True, exist_ok=True)

        state = {"store_quarantine": None}
        monkeypatch.setattr(app_module, "_state", state)

        store = MemoryStore()
        published = app_module._hydrate_memory_store_in_place(
            store, cfg, model=None, tokenizer=None
        )

        assert published is False
        cause = state["store_quarantine"]["cause"]
        assert cause["exception_type"] == "TierBindingUnpublishable"
        assert "episodic" in cause["message"]
        assert "semantic" in cause["message"]
        assert "no_matching_slot" in cause["message"]

        # No consolidation action dispatches while a main tier's binding is
        # unverified -- the deferred_tier_unverified arm of the arbitrator.
        from paramem.server.app import ConsolidationAction

        dispatch_state = {
            "config": cfg,
            "adapter_manifest_status": {"episodic": {"status": "no_matching_slot"}},
            "store_quarantine": None,
            "consolidating": False,
            "mode": "local",
            "migration": {},
            "background_trainer": None,
        }
        monkeypatch.setattr(app_module, "_state", dispatch_state)

        for action in (
            ConsolidationAction.AUTO,
            ConsolidationAction.FULL,
            ConsolidationAction.INTERIM,
            ConsolidationAction.RECONCILE,
        ):
            status, resolved = app_module._dispatch_consolidation(action)
            assert status == "deferred_tier_unverified"
            assert resolved is action


class TestBootMidWindowServesPreEventContent:
    """A boot landing between a tier's write and its publish resolves the
    OLD bound slot -- ``find_live_slot`` binds by the registry hash actually
    on disk, and the written-but-unpublished slot's manifest is stamped with
    a digest of registry bytes that never landed there. Under
    ``preload_cache: true`` the cache fills from that old bound slot, so a
    key that exists only in the unpublished slot is never enumerated and
    unreachable."""

    def test_cache_holds_the_published_value_and_omits_the_written_only_key(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.memory.persistence import publish_tier_registry, write_tier_slot
        from paramem.memory.store import MemoryStore

        adapter_dir = tmp_path / "adapters"
        ctx = _build_write_context(adapter_dir)

        baseline = _build_content_increment(
            tier="episodic",
            keys=[
                {
                    "key": "graph1",
                    "subject": "alice",
                    "predicate": "lives_in",
                    "object": "berlin",
                }
            ],
        )
        baseline_slot = write_tier_slot(
            ctx=ctx, increment=baseline, stamp="20260101T0000", mode="simulate"
        )
        publish_tier_registry(increment=baseline, ctx=ctx, written_slot=baseline_slot)

        increment = _build_content_increment(
            tier="episodic",
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
        write_tier_slot(ctx=ctx, increment=increment, stamp="20260102T0000", mode="simulate")
        # Deliberately NOT published -- the second slot's manifest digest
        # never lands at episodic/indexed_key_registry.json.

        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.paths.data = tmp_path / "data"
        cfg.paths.data.mkdir(parents=True, exist_ok=True)
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 4
        cfg.inference.preload_cache = True

        state = {"store_quarantine": None}
        monkeypatch.setattr(app_module, "_state", state)

        store = MemoryStore()
        published = app_module._hydrate_memory_store_in_place(
            store, cfg, model=None, tokenizer=None
        )

        assert published is True
        assert state["store_quarantine"] is None

        assert store.active_keys_in_tier("episodic") == ["graph1"]
        cache_results = store.probe_cache({"episodic": ["graph1", "graph2"]})
        assert cache_results["graph1"] is not None
        assert cache_results["graph1"]["object"] == "berlin", (
            "the mirror must fill from the OLD bound slot -- the written-"
            "only increment's value must never be served before its publish"
        )
        assert cache_results["graph2"] is None, (
            "a key that exists only in the unpublished slot is never "
            "enumerated (the bound registry does not know it) and answers "
            "nothing at the cache door"
        )


class TestBootMidWindowFirstContentQuarantines:
    """A tier receiving content for the FIRST time (no previously published
    registry) cannot fall back: a written-but-unpublished slot with no
    registry ever published for the tier resolves
    ``REGISTRY_ABSENT_WITH_SLOTS``, which is unpublishable, and the boot
    quarantines the whole store."""

    def test_a_tier_whose_only_content_is_a_written_unpublished_slot_quarantines(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module
        from paramem.adapters.registry_binding import REGISTRY_ABSENT_WITH_SLOTS
        from paramem.memory.persistence import write_tier_slot
        from paramem.memory.store import MemoryStore

        adapter_dir = tmp_path / "adapters"
        ctx = _build_write_context(adapter_dir)

        increment = _build_content_increment(
            tier="episodic",
            keys=[
                {
                    "key": "graph1",
                    "subject": "alice",
                    "predicate": "lives_in",
                    "object": "berlin",
                }
            ],
        )
        write_tier_slot(ctx=ctx, increment=increment, stamp="20260101T0000", mode="simulate")
        # No indexed_key_registry.json has EVER been published for this
        # tier -- the write is the only content on disk.

        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.paths.data = tmp_path / "data"
        cfg.paths.data.mkdir(parents=True, exist_ok=True)
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 4
        cfg.inference.preload_cache = True

        state = {"store_quarantine": None}
        monkeypatch.setattr(app_module, "_state", state)

        store = MemoryStore()
        published = app_module._hydrate_memory_store_in_place(
            store, cfg, model=None, tokenizer=None
        )

        assert published is False
        assert state["store_quarantine"] is not None
        cause = state["store_quarantine"]["cause"]
        assert cause["exception_type"] == "TierBindingUnpublishable"
        assert "episodic" in cause["message"]
        assert REGISTRY_ABSENT_WITH_SLOTS in cause["message"]
