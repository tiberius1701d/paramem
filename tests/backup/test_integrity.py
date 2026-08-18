"""Unit tests for paramem.backup.integrity.

Covers the generic ``FileCheck`` / ``IntegrityReport`` dataclass
serialization contract (does not depend on manifest shape or slot layout),
plus the rebuilt store-level integrity report coverage for the venue-uniform
written-payload design: a written+bound simulate slot's ``payload`` row reads
the BOUND slot's ``graph.json`` (never a tier-root path — nothing writes one
any more) via the same :func:`~paramem.adapters.registry_binding.verify_tier_binding`
resolution the ``manifest`` row uses, and ``cleanup_partial_slots``'s
fail-loud handling of a present-but-unparseable ``meta.json``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from paramem.adapters.manifest import tier_registry_sha256
from paramem.adapters.slot import write_slot
from paramem.backup.integrity import (
    _INCONSISTENT,
    _OK,
    _PARSE_ERROR,
    _SKIPPED,
    FileCheck,
    IntegrityReport,
    cleanup_partial_slots,
    verify_infrastructure_integrity,
)
from paramem.training.key_registry import KeyRegistry
from tests._fold_fixtures import _write_graph
from tests._manifest_fixtures import (
    make_train_manifest,
    v4_train_meta_dict,
    write_raw_meta,
    write_slot_files,
)


class TestDataModel:
    def test_filecheck_to_dict(self):
        """FileCheck.to_dict() returns all fields."""
        fc = FileCheck("/a/b/c.json", "registry", "episodic", _OK, "")
        d = fc.to_dict()
        assert d == {
            "path": "/a/b/c.json",
            "category": "registry",
            "tier": "episodic",
            "status": "ok",
            "detail": "",
        }

    def test_integrity_report_ok_false_when_failures(self):
        """IntegrityReport.ok is False when failures list is non-empty."""
        bad = FileCheck("/bad.json", "registry", "episodic", _PARSE_ERROR, "bad json")
        report = IntegrityReport(ok=False, checks=[bad], failures=[bad])
        assert report.ok is False
        d = report.to_dict()
        expected_check = {
            "path": "/bad.json",
            "category": "registry",
            "tier": "episodic",
            "status": _PARSE_ERROR,
            "detail": "bad json",
        }
        assert d == {
            "ok": False,
            "checks": [expected_check],
            "failures": [expected_check],
        }

    def test_integrity_report_ok_true_when_all_skipped(self):
        """IntegrityReport.ok True when only skipped entries."""
        skipped = FileCheck("/f.json", "registry", "episodic", _SKIPPED, "")
        report = IntegrityReport(ok=True, checks=[skipped], failures=[])
        assert report.ok is True


# ---------------------------------------------------------------------------
# cleanup_partial_slots: fail-loud on a present-but-unparseable meta.json
# ---------------------------------------------------------------------------


class TestCleanupPartialSlotsUnparseableManifest:
    """Per the ruled fail-loud design: only a slot with meta.json ABSENT (or
    a readable manifest missing its own kind's payload file) is scratch.  A
    slot whose meta.json is PRESENT but fails to parse (e.g. prior schema)
    is NOT scratch -- nothing here repairs or deletes it; it is left in
    place for registry↔slot binding verification to leave it unpublishable
    loudly."""

    def test_complete_prior_schema_slot_survives_untouched(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        write_raw_meta(slot, v4_train_meta_dict(name="episodic"))
        meta_bytes_before = (slot / "meta.json").read_bytes()

        removed = cleanup_partial_slots(adapter_dir)

        assert removed == []
        assert slot.exists()
        assert (slot / "meta.json").read_bytes() == meta_bytes_before
        assert (slot / "adapter_model.safetensors").exists()

    def test_slot_with_no_meta_json_is_removed(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        # No meta.json written -- this is the scratch case, unlike the
        # present-but-unparseable case above.

        removed = cleanup_partial_slots(adapter_dir)

        assert len(removed) == 1
        assert removed[0]["tier"] == "episodic"
        assert removed[0]["missing"] == ["meta.json"]
        assert not slot.exists()

    def test_binary_garbage_meta_json_survives_untouched(self, tmp_path: Path, caplog) -> None:
        """A meta.json that is not valid UTF-8 (binary garbage) is present-
        but-unparseable, same as a prior-schema meta.json -- read_manifest
        raises ManifestSchemaError (not an uncaught UnicodeDecodeError), and
        this fail-loud sweep retains the slot with a WARNING rather than
        deleting it or crashing the boot sweep."""
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        slot = tier_root / "20260101-000000"
        write_slot_files(slot)
        (slot / "meta.json").write_bytes(b"\xff\xfe\x00garbage-not-utf8\x80\x81")
        meta_bytes_before = (slot / "meta.json").read_bytes()

        with caplog.at_level("WARNING"):
            removed = cleanup_partial_slots(adapter_dir)

        assert removed == []
        assert slot.exists()
        assert (slot / "meta.json").read_bytes() == meta_bytes_before
        assert (slot / "adapter_model.safetensors").exists()
        assert any(
            record.levelname == "WARNING" and str(slot) in record.getMessage()
            for record in caplog.records
        )

    def test_slot_missing_its_own_kind_payload_file_is_removed(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        slot = tier_root / "20260101-000000"
        slot.mkdir(parents=True)
        # A current-schema, READABLE manifest -- but no payload file at all
        # (train kind requires adapter_model.safetensors + adapter_config.json).
        from paramem.adapters.manifest import write_manifest

        write_manifest(slot, make_train_manifest(name="episodic"))

        removed = cleanup_partial_slots(adapter_dir)

        assert len(removed) == 1
        assert removed[0]["tier"] == "episodic"
        assert "adapter_model.safetensors" in removed[0]["missing"]
        assert not slot.exists()


# ---------------------------------------------------------------------------
# verify_infrastructure_integrity: venue-uniform written-payload rows
# ---------------------------------------------------------------------------


def _write_bound_simulate_slot(tier_root: Path, key: str) -> None:
    """Registry with one active key + simhash, then a REAL written simulate
    slot bound to it -- the exact shape a healthy written simulate tier has
    on disk (through the same envelope production uses)."""
    registry = KeyRegistry()
    registry.add(key)
    registry.set_simhash(key, 12345)
    tier_root.mkdir(parents=True, exist_ok=True)
    registry.save(tier_root / "indexed_key_registry.json")
    entry = {
        "key": key,
        "subject": "alice",
        "predicate": "lives_in",
        "object": "berlin",
        "speaker_id": "speaker0",
    }
    _write_graph(tier_root, [entry])


def _write_bound_train_slot(tier_root: Path, key: str, *, name: str) -> None:
    """Registry with one active key + simhash, then a REAL written train
    (weights) slot bound to it."""
    registry = KeyRegistry()
    registry.add(key)
    registry.set_simhash(key, 54321)
    tier_root.mkdir(parents=True, exist_ok=True)
    registry.save(tier_root / "indexed_key_registry.json")
    registry_hash = tier_registry_sha256(tier_root)
    manifest = make_train_manifest(name=name, registry_sha256=registry_hash, key_count=1)
    write_slot(
        tier_root, manifest=manifest, write_payload=lambda pending: write_slot_files(pending)
    )


def _make_integrity_config(adapter_dir: Path, data_dir: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.adapter_dir = adapter_dir
    cfg.paths.data = data_dir
    return cfg


class TestIntegrityReportWrittenSimulateSlot:
    """A healthy written+bound simulate slot reports ok=True. The graph check
    reads the BOUND slot's graph.json -- nothing writes a tier-root
    graph.json any more -- via the same verify_tier_binding resolution the
    manifest row uses."""

    def test_written_simulate_slot_reports_ok_true(self, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapters"
        _write_bound_simulate_slot(adapter_dir / "episodic", "graph1")
        cfg = _make_integrity_config(adapter_dir, tmp_path / "data")

        report = verify_infrastructure_integrity(cfg, store=None, daily_loadable=False)

        assert report.ok is True, report.failures

        episodic_checks = {c.category: c for c in report.checks if c.tier == "episodic"}
        assert episodic_checks["manifest"].status == _OK
        assert episodic_checks["payload"].status == _OK
        # The payload row reads the bound SLOT's graph.json, never a
        # tier-root path -- nothing writes one any more.
        payload_path = Path(episodic_checks["payload"].path)
        assert payload_path.name == "graph.json"
        assert payload_path.parent != adapter_dir / "episodic"


class TestIntegrityReportsOnePayloadRowPerTierBothVenues:
    """Both venues resolve through the SAME venue-blind verify_tier_binding
    call: exactly one manifest row and one payload row per tier, no venue
    fork, no second independently-derived row for either category."""

    def test_integrity_reports_one_payload_row_per_tier_in_both_venues(
        self, tmp_path: Path
    ) -> None:
        adapter_dir = tmp_path / "adapters"
        _write_bound_simulate_slot(adapter_dir / "episodic", "graph1")
        _write_bound_train_slot(adapter_dir / "semantic", "sem1", name="semantic")
        cfg = _make_integrity_config(adapter_dir, tmp_path / "data")

        report = verify_infrastructure_integrity(cfg, store=None, daily_loadable=False)

        assert report.ok is True, report.failures

        for tier in ("episodic", "semantic"):
            tier_checks = [c for c in report.checks if c.tier == tier]
            manifest_rows = [c for c in tier_checks if c.category == "manifest"]
            payload_rows = [c for c in tier_checks if c.category == "payload"]
            assert len(manifest_rows) == 1, f"{tier}: expected exactly one manifest row"
            assert len(payload_rows) == 1, f"{tier}: expected exactly one payload row"
            # No separate "graph"-category row -- the payload row IS the
            # graph check for a simulate-kind tier.
            assert not [c for c in tier_checks if c.category == "graph"]

        episodic_payload = next(
            c for c in report.checks if c.tier == "episodic" and c.category == "payload"
        )
        assert episodic_payload.status == _OK

        # Train-kind payload byte verification already ran once, inside
        # verify_tier_binding's own step 7 -- this row surfaces that SAME
        # VERIFIED verdict rather than a placeholder skip.
        semantic_payload = next(
            c for c in report.checks if c.tier == "semantic" and c.category == "payload"
        )
        assert semantic_payload.status == _OK


class TestPayloadRowConsultsTheBindingFirstInBothVenues:
    """A non-VERIFIED binding must report ``payload: inconsistent`` in
    EITHER venue -- before this fix the simulate arm called ``_check_graph``
    unconditionally, so a corrupted-but-still-PARSEABLE graph.json on a tier
    whose binding was PAYLOAD_MISMATCH (digest disagreement) reported
    ``payload: ok``, while the train-venue equivalent correctly reported
    ``inconsistent`` for the identical binding status."""

    def test_a_payload_mismatch_binding_reports_inconsistent_in_the_simulate_venue(
        self, tmp_path: Path
    ) -> None:
        import json

        from paramem.adapters.manifest import find_live_slot, tier_registry_sha256

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        _write_bound_simulate_slot(tier_root, "graph1")

        slot = find_live_slot(tier_root, tier_registry_sha256(tier_root))
        assert slot is not None
        graph_path = slot / "graph.json"
        # Mutate the written payload bytes in place -- still perfectly
        # well-formed, parseable node-link JSON (so an unconditional
        # _check_graph call would report "ok"), but the digest stamped into
        # the manifest at write time no longer matches.
        data = json.loads(graph_path.read_text())
        data["nodes"].append({"id": "injected-not-what-was-written"})
        graph_path.write_text(json.dumps(data, indent=2))

        cfg = _make_integrity_config(adapter_dir, tmp_path / "data")
        report = verify_infrastructure_integrity(cfg, store=None, daily_loadable=False)

        payload_row = next(
            c for c in report.checks if c.tier == "episodic" and c.category == "payload"
        )
        assert payload_row.status == _INCONSISTENT
        assert payload_row.detail  # the binding's own PAYLOAD_MISMATCH detail
        assert report.ok is False


class TestKeyMetadataToleratesAWithheldIdsRow:
    """The per-tier ``key_metadata.json`` orphan check
    (``paramem.backup.integrity._check_registry``'s ``known_keys`` -- active
    ∪ stale) must not flag a withheld id's row as an orphan: a marker's
    bookkeeping row is designed to survive until its own tier's next
    rebuild (``paramem.memory.persistence._write_tier_key_metadata`` scopes
    to ``list_known()``, not ``list_active()``, for exactly this reason)."""

    def test_integrity_accepts_a_row_for_a_withheld_id(self, tmp_path: Path) -> None:
        import json

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        tier_root.mkdir(parents=True)

        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 12345)
        registry.add("graph2")
        registry.stale("graph2")
        registry.save(tier_root / "indexed_key_registry.json")

        entry = {
            "key": "graph1",
            "subject": "alice",
            "predicate": "lives_in",
            "object": "berlin",
            "speaker_id": "speaker0",
        }
        _write_graph(tier_root, [entry])

        bk_row = {
            "speaker_id": "speaker0",
            "relation_type": "factual",
            "reinforcement_count": 1,
            "last_reinforced_cycle": 0,
            "last_seen": "2026-01-01T00:00:00Z",
            "first_seen": "2026-01-01T00:00:00Z",
            "promoted": False,
        }
        key_metadata = {
            "tier_cycle": 0,
            "keys": {
                "graph1": bk_row,
                # graph2 is withheld (stale) -- list_known() (active ∪
                # stale) still covers it, so its row is NOT an orphan.
                "graph2": bk_row,
            },
        }
        (tier_root / "key_metadata.json").write_text(json.dumps(key_metadata))

        cfg = _make_integrity_config(adapter_dir, tmp_path / "data")
        report = verify_infrastructure_integrity(cfg, store=None, daily_loadable=False)

        assert report.ok is True, report.failures
        km_rows = [
            c for c in report.checks if c.tier == "episodic" and c.category == "key_metadata"
        ]
        assert km_rows and all(c.status != _INCONSISTENT for c in km_rows)
