"""Tests for the operator erase doors' shared post-mutation sequence,
``paramem.server.app._stale_mark_keys``.

``_stale_mark_keys`` is the ONE implementation ``POST /speaker/forget`` and
``POST /debug/erase-keys`` both compose (see ``_stale_mark_keys``'s own
docstring) -- an erase door never refuses: every affected tier's registry
mutation lands, every rebind is attempted, and the outcome is reported.
Driven directly (no ``TestClient``, no model) -- the two endpoints differ
only in how they resolve ``staled_keys``/``store``, never in this sequence,
and a direct-function test avoids the endpoint scaffolding a model would
otherwise require.

Also covers ``_record_or_resolve_tier_health``'s sticky-payload-status
invariant directly: a limited-authority caller (a restamp-only erase door,
``resolves_payload_status=False``) may neither overwrite nor resolve a
``PAYLOAD_MISMATCH`` marker a full-authority caller (the drift sweep,
``resolves_payload_status=True``) recorded -- only another full-authority
caller can.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import paramem.server.app as app_module
from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
from paramem.memory.increment import TierIncrement, TierWriteContext
from paramem.memory.persistence import publish_tier_registry, write_tier_slot
from paramem.memory.store import MemoryStore
from paramem.server.app import DebugEraseKeysResponse, SpeakerForgetResponse
from paramem.server.incidents import read_incidents
from paramem.training.key_registry import KeyRegistry
from paramem.training.stage_ledger import data_state_dir


def _snapshot_tree(root: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every file under *root*.

    A pure read used to prove a refused call wrote zero bytes anywhere
    under *root* -- comparing two snapshots catches both a changed file
    (digest differs) and a written/removed one (path set differs) in one
    assertion, rather than checking a single named file and assuming the
    rest of the tree is untouched. Mirrors
    ``tests/test_publish_bundle_resume.py::_snapshot_tree``.
    """
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _config(tmp_path: Path) -> MagicMock:
    """Minimal config stand-in -- ``_stale_mark_keys`` and its collaborators
    read only ``adapter_dir`` and ``paths.data``."""
    cfg = MagicMock()
    cfg.adapter_dir = tmp_path / "adapters"
    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.data = tmp_path / "data"
    cfg.paths.data.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_bound_tier(
    adapter_dir: Path,
    tier: str,
    keys: "list[str]",
    *,
    simhashes: "dict[str, int] | None" = None,
) -> Path:
    """Write and publish a bound simulate-venue slot for *tier* carrying
    every key in *keys*, active. Returns the bound slot path.

    *simhashes* optionally attaches a fingerprint per key (default: none) --
    needed by tests that assert the fingerprint is gone from the on-disk
    registry after an erase.
    """
    registry = KeyRegistry()
    for key in keys:
        registry.add(key)
        if simhashes and key in simhashes:
            registry.set_simhash(key, simhashes[key])
    registry_bytes = registry.save_bytes()
    increment = TierIncrement(
        tier=tier,
        adapter_name=tier,
        registry=registry,
        registry_bytes=registry_bytes,
        rows_bytes=b'{"tier_cycle": 0, "keys": {}}',
        entries={},
        bookkeeping={},
        keyed=[
            {
                "key": key,
                "subject": "alice",
                "predicate": "lives_in",
                "object": "berlin",
                "speaker_id": "speaker0",
            }
            for key in keys
        ],
        rebuilt=True,
        pre_sha="",
    )
    ctx = TierWriteContext(
        model=None,
        tokenizer=None,
        fingerprint_cache={},
        output_dir=adapter_dir,
        tier_configs={},
        store=None,
        keep_prior_slots=1,
    )
    slot = write_tier_slot(ctx=ctx, increment=increment, stamp="20260101T0000", mode="simulate")
    publish_tier_registry(increment=increment, ctx=ctx, written_slot=slot)
    return slot


def _memory_store_with_key(tier: str, key: str, *, simhash: int) -> MemoryStore:
    """A RAM-resident :class:`MemoryStore` carrying *key* active in *tier*,
    with a fingerprint, bookkeeping row and entry -- the shape a healthy,
    non-quarantined server holds for the same key an erase door is about to
    stale-mark on disk."""
    store = MemoryStore()
    registry = KeyRegistry()
    registry.add(key)
    registry.set_simhash(key, simhash)
    store.load_registry(tier, registry)
    store.set_bookkeeping(
        key,
        speaker_id="speaker0",
        relation_type="factual",
        reinforcement_count=1,
        last_reinforced_cycle=0,
        last_seen="2026-01-01T00:00:00Z",
        first_seen="2026-01-01T00:00:00Z",
        promoted=False,
    )
    store.put(
        tier,
        key,
        {"key": key, "subject": "alice", "predicate": "lives_in", "object": "berlin"},
        register=False,
    )
    return store


class TestEraseDoorLandsTheRegistryMutationAndRebinds:
    def test_the_erase_door_lands_the_registry_mutation_and_rebinds(self, tmp_path):
        cfg = _config(tmp_path)
        slot = _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"])

        result = app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test")

        assert result["staled_keys"] == ["graph1"]
        assert result["unbound_tiers"] == []
        assert result["tiers"] == [
            {"tier": "episodic", "outcome": "rebound", "slot": str(slot), "reason": None}
        ]

        # The slot manifest was rebound to match the mutated registry.
        tier_root = cfg.adapter_dir / "episodic"
        assert find_live_slot(tier_root, tier_registry_sha256(tier_root)) == slot

        # A rebound tier resolves (never records) a tier_registry_unverified
        # incident.
        state_dir = data_state_dir(cfg.paths.data)
        assert read_incidents(state_dir) == []

    def test_a_key_no_tier_knows_is_a_no_op(self, tmp_path):
        """Sanity counterpart: a key not present in any tier's registry
        touches nothing -- the erase door only acts on affected tiers."""
        cfg = _config(tmp_path)
        _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"])

        result = app_module._stale_mark_keys(
            config=cfg, staled_keys=["nobody_knows_this"], label="t"
        )

        assert result == {"staled_keys": ["nobody_knows_this"], "tiers": [], "unbound_tiers": []}
        registry_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        assert "graph1" in KeyRegistry.load(registry_path)


class TestEraseDoorOnATierWithKeysAndNoSlot:
    def test_the_erase_door_on_a_tier_with_keys_and_no_slot_lands_and_reports_unbound(
        self, tmp_path, caplog
    ):
        cfg = _config(tmp_path)

        # A tier with two active keys and NO on-disk slot at all -- stale-
        # marking one still leaves an active key behind, so plan_restamp
        # resolves KEYS_WITHOUT_SLOT rather than NOTHING_TO_BIND.
        registry = KeyRegistry()
        registry.add("graph1")
        registry.add("graph2")
        registry.save(cfg.adapter_dir / "episodic" / "indexed_key_registry.json")

        with caplog.at_level(logging.ERROR, logger="paramem.server.app"):
            result = app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test")

        assert result["unbound_tiers"] == ["episodic"]
        assert result["tiers"] == [
            {
                "tier": "episodic",
                "outcome": "unbound",
                "slot": None,
                "reason": "keys_without_slot",
            }
        ]

        # The untouched key in the same tier is unaffected.
        registry_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        reloaded = KeyRegistry.load(registry_path)
        assert "graph2" in reloaded

        # An ERROR was logged naming the tier.
        assert any(
            record.levelno == logging.ERROR and "episodic" in record.getMessage()
            for record in caplog.records
        )

        # A tier_registry_unverified incident was recorded for episodic,
        # named/failed since episodic is the primary tier.
        state_dir = data_state_dir(cfg.paths.data)
        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        incident = incidents[0]
        assert incident.type == "tier_registry_unverified"
        assert incident.id == "tier_registry_unverified:episodic"
        assert incident.severity == "failed"
        assert incident.status == "active"


class TestOneTierRebindIOFailureIsRecordedAndTheRestStillCommit:
    def test_an_io_failure_rebinding_one_tier_does_not_abort_the_others(
        self, tmp_path, monkeypatch
    ):
        """An ``OSError`` raised while re-stamping ONE tier's slot manifest --
        the actual write_manifest call inside restamp_tier_manifest, patched
        selectively so the real commit path still runs for both tiers -- is
        caught at that tier: outcome ``"rebind_failed"`` with the exception's
        message, while the OTHER affected tier's mutation and rebind land for
        real (registry write + manifest re-stamp), and both erased keys are
        immediately unservable in the RAM mirror. No exception propagates --
        the caller (an endpoint handler) would return 200."""
        import paramem.adapters.manifest as manifest_module

        cfg = _config(tmp_path)
        episodic_slot = _write_bound_tier(
            cfg.adapter_dir, "episodic", ["graph1"], simhashes={"graph1": 111}
        )
        semantic_slot = _write_bound_tier(
            cfg.adapter_dir, "semantic", ["graph2"], simhashes={"graph2": 222}
        )

        store = MemoryStore()
        for tier, key, simhash in (("episodic", "graph1", 111), ("semantic", "graph2", 222)):
            registry = KeyRegistry()
            registry.add(key)
            registry.set_simhash(key, simhash)
            store.load_registry(tier, registry)
            store.set_bookkeeping(
                key,
                speaker_id="speaker0",
                relation_type="factual",
                reinforcement_count=1,
                last_reinforced_cycle=0,
                last_seen="2026-01-01T00:00:00Z",
                first_seen="2026-01-01T00:00:00Z",
                promoted=False,
            )
            store.put(
                tier,
                key,
                {"key": key, "subject": "alice", "predicate": "lives_in", "object": "berlin"},
                register=False,
            )

        real_write_manifest = manifest_module.write_manifest

        def _flaky_write_manifest(slot, manifest):
            # Selective failure at the exact seam restamp_tier_manifest
            # documents as propagating OSError -- episodic's slot fails,
            # semantic's own write_manifest call (a separate invocation)
            # still runs for real.
            if slot == episodic_slot:
                raise OSError("simulated disk failure writing episodic manifest")
            return real_write_manifest(slot, manifest)

        monkeypatch.setattr(manifest_module, "write_manifest", _flaky_write_manifest)

        result = app_module._stale_mark_keys(
            config=cfg, staled_keys=["graph1", "graph2"], label="test", store=store
        )

        assert result["staled_keys"] == ["graph1", "graph2"]

        tiers_by_name = {t["tier"]: t for t in result["tiers"]}
        assert set(tiers_by_name) == {"episodic", "semantic"}

        episodic_outcome = tiers_by_name["episodic"]
        assert episodic_outcome["outcome"] == "rebind_failed"
        assert episodic_outcome["slot"] is None
        assert "simulated disk failure writing episodic manifest" in episodic_outcome["reason"]

        semantic_outcome = tiers_by_name["semantic"]
        assert semantic_outcome["outcome"] == "rebound"
        assert semantic_outcome["slot"] == str(semantic_slot)
        assert semantic_outcome["reason"] is None

        assert result["unbound_tiers"] == ["episodic"]

        # The SECOND tier committed for real on disk -- registry bytes
        # changed and the slot manifest was actually rebound.
        semantic_root = cfg.adapter_dir / "semantic"
        assert find_live_slot(semantic_root, tier_registry_sha256(semantic_root)) == semantic_slot
        semantic_registry = KeyRegistry.load(semantic_root / "indexed_key_registry.json")
        assert "graph2" not in semantic_registry
        assert semantic_registry.knows("graph2")

        # The FIRST (failed-rebind) tier's registry mutation still landed --
        # only the slot rebind failed, per restamp_tier_manifest's own
        # write-then-rebind ordering.
        episodic_registry = KeyRegistry.load(
            cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        )
        assert "graph1" not in episodic_registry
        assert episodic_registry.knows("graph1")

        # RAM store no longer serves either erased key, in either tier.
        assert store.active_keys_in_tier("episodic") == []
        assert store.active_keys_in_tier("semantic") == []
        assert "graph1" not in store.registry("episodic")
        assert "graph2" not in store.registry("semantic")

        # The result constructs into the response model without error --
        # the shape a 200 response requires.
        response = SpeakerForgetResponse(**result, removed_speaker=True, discarded_sessions=[])
        assert response.unbound_tiers == ["episodic"]
        outcomes = {t.tier: t.outcome for t in response.tiers}
        assert outcomes == {"episodic": "rebind_failed", "semantic": "rebound"}


class TestOneTierRebindNonIOFailurePropagates:
    def test_a_non_io_exception_from_the_restamp_still_propagates(self, tmp_path, monkeypatch):
        """Only ``OSError`` and ``ManifestError`` are caught per-tier -- any
        other exception raised inside restamp_tier_manifest still propagates
        uncaught out of erase_keys_and_restamp_manifest (fail-loud
        preserved)."""
        cfg = _config(tmp_path)
        _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"])

        import paramem.memory.persistence as persistence_module

        def _boom(*args, **kwargs):
            raise RuntimeError("unexpected failure, not an I/O shape")

        monkeypatch.setattr(persistence_module, "restamp_tier_manifest", _boom)

        with pytest.raises(RuntimeError, match="unexpected failure"):
            app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test")


class TestUnreadableTierRegistryAbortsBeforeAnyMutation:
    def test_an_unreadable_tier_registry_fails_the_erase_before_any_mutation(self, tmp_path):
        """A tier whose on-disk registry exists but is not KeyRegistry-shaped
        aborts the whole call with a 500 -- ``KeyRegistry.load`` raises for
        every tier under ``adapter_dir`` (iter_tier_roots's own loop, before
        the mutation loop even starts, see
        ``erase_keys_and_restamp_manifest``'s own docstring), so an affected
        tier ahead of the broken one in iteration order is left byte-identical,
        never half-mutated. Verified across the WHOLE adapter tree, not one
        named file -- a snapshot (path set + per-file sha256) taken before
        the call must match one taken after, so a stray write anywhere else
        under ``adapter_dir`` would also fail this test."""
        cfg = _config(tmp_path)
        # episodic is affected (holds the requested key, active, bound) and
        # would normally be mutated first -- its bytes must survive the abort.
        _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"], simhashes={"graph1": 111})
        episodic_registry_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"

        # semantic exists but is not KeyRegistry-shaped (foreign schema).
        semantic_dir = cfg.adapter_dir / "semantic"
        semantic_dir.mkdir(parents=True)
        (semantic_dir / "indexed_key_registry.json").write_text(
            json.dumps({"active_keys": ["ghost_key_0"]})
        )

        before = _snapshot_tree(cfg.adapter_dir)

        with pytest.raises(HTTPException) as exc_info:
            app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test")

        assert exc_info.value.status_code == 500

        # The whole adapter tree is untouched -- whole-tree byte identity,
        # not just episodic's registry file.
        assert _snapshot_tree(cfg.adapter_dir) == before
        on_disk = KeyRegistry.load(episodic_registry_path)
        assert "graph1" in on_disk


class TestEraseDoorResponseModelConstruction:
    """Pins the handler-to-model field wiring: ``_stale_mark_keys``'s real
    return dict, fed straight into each endpoint's response model, exactly
    as ``POST /speaker/forget`` (``SpeakerForgetResponse(**result)``) and
    ``POST /debug/erase-keys`` (``DebugEraseKeysResponse(..., tiers=...,
    unbound_tiers=...)``) build them -- no ``TestClient``, per this file's
    own direct-function-test allowance, but a real construction rather than
    a hand-built dict standing in for ``_stale_mark_keys``'s shape."""

    def test_speaker_forget_response_constructs_from_a_rebound_result(self, tmp_path):
        cfg = _config(tmp_path)
        slot = _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"])

        result = app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test")
        # The two fields the handler's no-await tail fills in after
        # _stale_mark_keys returns -- see speaker_forget's own body.
        result["removed_speaker"] = True
        result["discarded_sessions"] = ["conv1"]

        response = SpeakerForgetResponse(**result)

        assert response.removed_speaker is True
        assert response.staled_keys == ["graph1"]
        assert response.discarded_sessions == ["conv1"]
        assert response.unbound_tiers == []
        assert len(response.tiers) == 1
        assert response.tiers[0].tier == "episodic"
        assert response.tiers[0].outcome == "rebound"
        assert response.tiers[0].slot == str(slot)
        assert response.tiers[0].reason is None

    def test_debug_erase_keys_response_constructs_from_an_unbound_result(self, tmp_path):
        cfg = _config(tmp_path)
        registry = KeyRegistry()
        registry.add("graph1")
        registry.add("graph2")
        registry.save(cfg.adapter_dir / "episodic" / "indexed_key_registry.json")

        result = app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test")
        # The two fields debug_erase_keys' own _erase_sync fills in --
        # "unknown" (requested keys no tier recognised) and "lifted" (quarantine
        # lift outcome, None when no lift was attempted) -- see its own body.
        result["unknown"] = ["ghost_key"]
        result["lifted"] = None

        response = DebugEraseKeysResponse(
            staled=result["staled_keys"],
            unknown=result["unknown"],
            lifted=result["lifted"],
            tiers=result["tiers"],
            unbound_tiers=result["unbound_tiers"],
        )

        assert response.staled == ["graph1"]
        assert response.unknown == ["ghost_key"]
        assert response.lifted is None
        assert response.unbound_tiers == ["episodic"]
        assert len(response.tiers) == 1
        assert response.tiers[0].tier == "episodic"
        assert response.tiers[0].outcome == "unbound"
        assert response.tiers[0].slot is None
        assert response.tiers[0].reason == "keys_without_slot"


class TestStickyPayloadStatusAcrossAuthorityLevels:
    """``_record_or_resolve_tier_health``'s sticky-payload-status invariant,
    driven directly: a payload-level ``PAYLOAD_MISMATCH`` marker recorded by
    a full-authority caller (``resolves_payload_status=True``) can be
    OVERWRITTEN or RESOLVED only by another full-authority caller -- a
    limited-authority record must preserve the marker, and a
    limited-authority resolve must refuse while it is present."""

    def test_a_limited_authority_record_does_not_erase_the_payload_marker(self, tmp_path):
        cfg = _config(tmp_path)
        state_dir = data_state_dir(cfg.paths.data)

        # A full-authority caller (the post-fold drift sweep) already
        # recorded PAYLOAD_MISMATCH for this tier.
        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status="payload_mismatch",
            detail="payload digest disagreement",
            candidate_count=1,
            resolves_payload_status=True,
        )

        # A limited-authority caller (a restamp-only erase door) now
        # records its OWN, unrelated failure for the same tier.
        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status="no_matching_slot",
            detail="test: registry mutation landed, manifest re-stamp no_matching_slot",
            candidate_count=0,
            resolves_payload_status=False,
        )

        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        incident = incidents[0]
        # The payload-level marker is PINNED -- never overwritten by the
        # limited-authority caller's own (weaker) status.
        assert incident.detail["status"] == "payload_mismatch"
        # The limited-authority caller's own failure text still lands, for
        # observability.
        assert "no_matching_slot" in incident.detail["detail"]

    def test_a_limited_authority_resolve_refuses_while_the_marker_is_present(self, tmp_path):
        cfg = _config(tmp_path)
        state_dir = data_state_dir(cfg.paths.data)

        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status="payload_mismatch",
            detail="payload digest disagreement",
            candidate_count=1,
            resolves_payload_status=True,
        )

        # A limited-authority caller's own healthy signal (e.g. a
        # successful restamp) must not clear the marker.
        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status=None,
            detail="",
            candidate_count=0,
            resolves_payload_status=False,
        )

        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        assert incidents[0].status != "resolved"
        assert incidents[0].detail["status"] == "payload_mismatch"

    def test_a_full_authority_verified_resolve_clears_the_marker(self, tmp_path):
        cfg = _config(tmp_path)
        state_dir = data_state_dir(cfg.paths.data)

        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status="payload_mismatch",
            detail="payload digest disagreement",
            candidate_count=1,
            resolves_payload_status=True,
        )

        # The drift sweep recomputed the full binding and found it
        # VERIFIED -- a full-authority healthy signal.
        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status=None,
            detail="",
            candidate_count=0,
            resolves_payload_status=True,
        )

        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        assert incidents[0].status == "resolved"

    def test_a_plain_erase_door_record_and_resolve_cycle_with_no_marker_is_unaffected(
        self, tmp_path
    ):
        """No payload-level marker in play -- a restamp-only caller's own
        record and resolve behave exactly as before this fix."""
        cfg = _config(tmp_path)
        state_dir = data_state_dir(cfg.paths.data)

        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status="keys_without_slot",
            detail="test: registry mutation landed, manifest re-stamp keys_without_slot",
            candidate_count=0,
            resolves_payload_status=False,
        )
        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        assert incidents[0].detail["status"] == "keys_without_slot"
        assert incidents[0].status == "active"

        app_module._record_or_resolve_tier_health(
            cfg,
            tier="episodic",
            unhealthy_status=None,
            detail="",
            candidate_count=0,
            resolves_payload_status=False,
        )
        incidents = read_incidents(state_dir)
        assert len(incidents) == 1
        assert incidents[0].status == "resolved"


class TestErasedKeyIsUnservableImmediately:
    def test_an_erased_key_is_unservable_from_the_moment_the_door_returns(self, tmp_path):
        """The door's registry mutation lands the withhold AND drops the
        fingerprint in the same write -- both on disk and in the caller's
        RAM mirror, before this call returns."""
        cfg = _config(tmp_path)
        _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"], simhashes={"graph1": 111})
        store = _memory_store_with_key("episodic", "graph1", simhash=111)

        result = app_module._stale_mark_keys(
            config=cfg, staled_keys=["graph1"], label="test", store=store
        )

        assert result["staled_keys"] == ["graph1"]
        assert result["unbound_tiers"] == []

        # Unservable on disk immediately -- excluded from __contains__ and
        # list_active, but still known (withheld, not erased), and its
        # fingerprint is gone in the same write.
        registry_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        on_disk = KeyRegistry.load(registry_path)
        assert "graph1" not in on_disk
        assert on_disk.list_active() == []
        assert on_disk.knows("graph1") is True
        assert on_disk.simhash_for("graph1") is None

        # Unservable in RAM immediately -- the router's probe intersection
        # (allowed_keys & active_keys_in_tier) reads exactly this.
        assert store.active_keys_in_tier("episodic") == []
        assert "graph1" not in store.registry("episodic")


class TestErasedKeyKeepsItsRowUntilItsTierRebuilds:
    def test_an_erased_key_keeps_its_row_and_its_id_until_its_tier_rebuilds(self, tmp_path):
        """This door writes no ``key_metadata.json`` -- the row and entry
        are untouched, and the id stays reserved (known) on disk, until the
        owning tier's own rebuild retires them."""
        cfg = _config(tmp_path)
        _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"], simhashes={"graph1": 111})
        store = _memory_store_with_key("episodic", "graph1", simhash=111)

        app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test", store=store)

        registry_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        on_disk = KeyRegistry.load(registry_path)
        assert on_disk.list_known() == ["graph1"]

        assert store.bookkeeping_for_key("graph1") is not None
        assert store.get("graph1") is not None


class TestReErasingAWithheldKeyIsIdempotent:
    def test_erasing_an_already_withheld_key_writes_no_bytes_and_reports_it_staled(self, tmp_path):
        cfg = _config(tmp_path)
        _write_bound_tier(cfg.adapter_dir, "episodic", ["graph1"], simhashes={"graph1": 111})
        store = _memory_store_with_key("episodic", "graph1", simhash=111)

        app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="first", store=store)
        registry_path = cfg.adapter_dir / "episodic" / "indexed_key_registry.json"
        before = registry_path.read_bytes()

        result = app_module._stale_mark_keys(
            config=cfg, staled_keys=["graph1"], label="second", store=store
        )

        # No tier is affected -- the key is already withheld everywhere, so
        # nothing is written and no rebind is attempted (designed
        # idempotence, per erase_keys_and_restamp_manifest's own docstring).
        assert result == {"staled_keys": ["graph1"], "tiers": [], "unbound_tiers": []}
        assert registry_path.read_bytes() == before
        # Still reported as staled, not silently dropped.
        assert result["staled_keys"] == ["graph1"]


class TestEraseDoorAndRamMirrorWithholdTheSameKeys:
    def test_the_erase_door_and_the_ram_mirror_withhold_the_same_keys(self, tmp_path):
        """The disk door (active-membership guard) and the RAM mirror
        (``MemoryStore.discard_keys``) agree on exactly which keys are
        withheld and which stay active -- one predicate, two call sites."""
        cfg = _config(tmp_path)
        _write_bound_tier(
            cfg.adapter_dir, "episodic", ["graph1", "graph2"], simhashes={"graph1": 1, "graph2": 2}
        )
        store = MemoryStore()
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 1)
        registry.add("graph2")
        registry.set_simhash("graph2", 2)
        store.load_registry("episodic", registry)
        for key in ("graph1", "graph2"):
            store.set_bookkeeping(
                key,
                speaker_id="speaker0",
                relation_type="factual",
                reinforcement_count=1,
                last_reinforced_cycle=0,
                last_seen="2026-01-01T00:00:00Z",
                first_seen="2026-01-01T00:00:00Z",
                promoted=False,
            )
            store.put(
                "episodic",
                key,
                {"key": key, "subject": "a", "predicate": "p", "object": "o"},
                register=False,
            )

        app_module._stale_mark_keys(config=cfg, staled_keys=["graph1"], label="test", store=store)

        on_disk = KeyRegistry.load(cfg.adapter_dir / "episodic" / "indexed_key_registry.json")
        ram = store.registry("episodic")

        assert on_disk.list_active() == ram.list_active() == ["graph2"]
        assert on_disk.list_stale() == ram.list_stale() == ["graph1"]
        assert on_disk.knows("graph1") and ram.knows("graph1")


class TestEraseRoundTripFromDoorToRebuild:
    def test_the_erase_round_trip_from_door_to_rebuild_leaves_no_trace_of_the_key(
        self, tmp_path, monkeypatch
    ):
        """Erase (disk-only door) -> unservable, row present, id reserved ->
        the owning tier's real rebuild (the fold, via the shared driver
        fixtures) -> registry, rows, RAM mirror all free of the key."""
        from tests._fold_fixtures import _make_loop, _recalled_entries_from_store, _rel, _wire_fakes

        loop = _make_loop(tmp_path, resident_tiers=["episodic"])
        _wire_fakes(loop, monkeypatch)

        loop.store.registry("episodic").add("graph1")
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
            promoted=False,
        )
        loop.store.put(
            "episodic",
            "graph1",
            {"key": "graph1", "subject": "alex", "predicate": "lives in", "object": "berlin"},
            register=False,
        )
        # _make_loop bypasses __init__ (no _derive_key_counters call) -- do
        # it now so the rebuild's own mint does not collide with "graph1"'s
        # own numeric suffix.
        loop._derive_key_counters()
        # The door reads/writes disk, not RAM -- persist the matching
        # on-disk registry before calling it.
        (loop.output_dir / "episodic").mkdir(parents=True, exist_ok=True)
        loop.store.registry("episodic").save(
            loop.output_dir / "episodic" / "indexed_key_registry.json"
        )

        cfg = MagicMock()
        cfg.adapter_dir = loop.output_dir
        cfg.paths.data = tmp_path / "data"
        cfg.paths.data.mkdir(parents=True, exist_ok=True)

        app_module._stale_mark_keys(
            config=cfg, staled_keys=["graph1"], label="round-trip", store=loop.store
        )

        # Immediately after the door: unservable, row present, id reserved.
        assert loop.store.registry("episodic").knows("graph1") is True
        assert "graph1" not in loop.store.registry("episodic")
        assert loop.store.bookkeeping_for_key("graph1") is not None
        assert loop.store.get("graph1") is not None

        # New material drives the tier's own rebuild -- a withheld key with
        # no rebuild-triggering content anywhere else would otherwise noop.
        staged = loop.stage_event(
            recalled_entries=_recalled_entries_from_store(loop),
            event="full",
            venue="weights",
            stamp="s_rebuild",
            primary_tiers={"episodic": "episodic"},
            episodic_rels=[_rel("sam", "works_at", "acme")],
        )
        assert staged is not None

        summary = loop.run_build_and_publish(staged, router=None)
        assert summary["all_live"] is True

        # No trace of the erased key anywhere after the rebuild.
        registry = loop.store.registry("episodic")
        assert not registry.knows("graph1")
        assert loop.store.bookkeeping_for_key("graph1") is None
        assert loop.store.get("graph1") is None
        on_disk_after = KeyRegistry.load(loop.output_dir / "episodic" / "indexed_key_registry.json")
        assert not on_disk_after.knows("graph1")
