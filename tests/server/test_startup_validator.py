"""Integration tests for _mount_adapters_from_slots startup validator.

Exercises _mount_adapters_from_slots directly (bypasses full lifespan).
All assertions operate on state["adapter_manifest_status"].

Covers:
- Fresh install → no manifest rows.
- Healthy mount → no rows.
- Episodic wrong base_model.sha → red fingerprint_mismatch, not loaded.
- Semantic wrong lora.rank → yellow mismatch.
- Corrupt meta.json (registry present) → no_matching_slot (find_live_slot
  skips it internally); registry absent → registry_unverified instead (see
  TestNoMatchingSlot).
- Weights + no meta.json → fresh (no row); manifest_missing is now only
  minted by _load_one's load-failure fallback, not by this decision tree.
- enabled=False → no row.
- Registry hash mismatch → no_matching_slot.
- Multiple rows render independently.
- Migration-script slot (synthesized=True, UNKNOWN fields) → yellow even for episodic.
- Fresh-built manifest with UNKNOWN fields (synthesized=False) → red for the
  primary tier (episodic), yellow for a non-primary tier (semantic) — a red
  row renders as a failed-level "PA routing DISABLED" attention item, which
  is only true for the primary tier.
- episodic_interim_* routes through the same _validate_adapter_slot decision
  tree as main tiers (fingerprints compared against config.adapters.episodic):
  no-weight-slot-candidate → INFO fresh install, no row; a real weight-slot
  candidate with a hash mismatch → WARNING, yellow no_matching_slot row;
  healthy slot → mounted, no row (and clears a stale pre-existing row);
  fingerprint mismatch → not mounted, yellow mismatch row with field set.
- A keyless tier (main or interim) is reaped before any slot is resolved:
  interim-child survival, stale-only preservation with the reworded ERROR,
  an unreadable registry preserved and logged, an absent registry left
  untouched, a reaped tier's fresh-install classification, and the
  keyless-tier sweep running before any slot is resolved (mock order).
- _revalidate_adapter_manifests: revalidates BOTH main tiers and every live
  interim dir (healthy clears a stale row, torn gets an accurate row), plus
  pruning stale adapter_manifest_status rows for interim tiers whose dir is
  gone entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.adapters.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNKNOWN,
    AdapterManifest,
    BaseModelFingerprint,
    LoRAShape,
    TokenizerFingerprint,
    write_manifest,
)
from paramem.server.app import _mount_adapters_from_slots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, adapter_names=("episodic",), enabled_names=None):
    """Build a minimal ServerConfig-like object."""
    if enabled_names is None:
        enabled_names = adapter_names

    config = MagicMock()
    config.adapter_dir = tmp_path / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)

    def _make_adapter_cfg(name):
        cfg = MagicMock()
        cfg.enabled = name in enabled_names
        cfg.rank = 8
        cfg.alpha = 16
        cfg.dropout = 0.0
        cfg.target_modules = ["q_proj", "v_proj"]
        return cfg

    config.adapters.episodic = _make_adapter_cfg("episodic")
    config.adapters.semantic = _make_adapter_cfg("semantic")
    config.adapters.procedural = _make_adapter_cfg("procedural")
    return config


def _make_model(name_or_path: str = "hf/model", commit_hash: str = "abc123"):
    model = MagicMock()
    model.config._name_or_path = name_or_path
    model.config._commit_hash = commit_hash
    model.peft_config = {}
    return model


def _make_tokenizer():
    tok = MagicMock()
    tok.name_or_path = "hf/model"
    return tok


def _write_slot(
    adapter_kind_dir: Path,
    ts: str = "20260421-000000",
    registry_sha256: str = "",
    sha: str = "abc123",
    rank: int = 8,
    synthesized: bool = False,
    key_count: "int | str" = 0,
) -> Path:
    """Write a slot manifest.  ``key_count`` defaults to 0 — matching the
    active-key count of a genuinely absent registry (the common shape most
    fingerprint/manifest tests in this file use), so an int default stays
    a non-mismatch against verify_tier_binding's key-count check without
    tripping app.py's separate ``_first_unknown_field`` UNKNOWN-sentinel
    check (which an UNKNOWN default WOULD trip). Tests writing a real
    registry with active keys, or specifically exercising
    KEY_COUNT_MISMATCH, pass an explicit key_count that matches (or
    deliberately disagrees with) their own registry.
    """
    slot = adapter_kind_dir / ts
    slot.mkdir(parents=True)
    # Write minimal adapter files so load won't fail on file-not-found
    (slot / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "hf/model"}))
    (slot / "adapter_model.safetensors").write_bytes(b"weights")
    m = AdapterManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        name="episodic",
        trained_at="2026-04-21T00:00:00Z",
        base_model=BaseModelFingerprint(repo="hf/model", sha=sha, hash="sha256:dead"),
        tokenizer=TokenizerFingerprint(
            name_or_path="hf/model", vocab_size=32000, merges_hash="cafe"
        ),
        lora=LoRAShape(rank=rank, alpha=rank * 2, dropout=0.0, target_modules=("q_proj", "v_proj")),
        registry_sha256=registry_sha256,
        key_count=key_count,
        synthesized=synthesized,
    )
    write_manifest(slot, m)
    return slot


def _run(config, model=None, tokenizer=None, state=None):
    """Execute _mount_adapters_from_slots and return (model, state)."""
    if model is None:
        model = _make_model()
    if tokenizer is None:
        tokenizer = _make_tokenizer()
    if state is None:
        state = {"adapter_manifest_status": {}, "base_model_hash_cache": {}}
    result_model = _mount_adapters_from_slots(model, tokenizer, config, state)
    return result_model, state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFreshInstall:
    def test_no_rows_on_empty_adapter_dir(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _, state = _run(config)
        assert state["adapter_manifest_status"] == {}


class TestHealthyMount:
    def test_healthy_mount_produces_no_row(self, tmp_path: Path) -> None:
        """A healthy slot with matching fingerprints must produce no manifest row.

        We patch PeftModel.from_pretrained so the test doesn't need real weights.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        _write_slot(kind_dir, registry_sha256="")

        from peft import PeftModel

        with patch.object(PeftModel, "from_pretrained", return_value=MagicMock(spec=PeftModel)):
            _, state = _run(config)
        assert "episodic" not in state["adapter_manifest_status"]


class TestFingerprintMismatch:
    def test_episodic_wrong_sha_gives_red_and_not_loaded(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # Slot has sha=different_sha; model has sha=abc123
        _write_slot(kind_dir, sha="different_sha")

        model = _make_model(commit_hash="abc123")
        _, state = _run(config, model=model)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None, "Expected a row for episodic fingerprint mismatch"
        assert row["severity"] == "red"
        assert row["reason"] == "fingerprint_mismatch"
        # Model must NOT have episodic loaded
        assert "episodic" not in getattr(model, "peft_config", {})

    def test_semantic_wrong_rank_gives_yellow(self, tmp_path: Path) -> None:
        config = _make_config(
            tmp_path, adapter_names=("episodic", "semantic"), enabled_names=("semantic",)
        )
        kind_dir = config.adapter_dir / "semantic"
        kind_dir.mkdir()
        # Slot has rank=4; config has rank=8
        _write_slot(kind_dir, rank=4)
        config.adapters.semantic.rank = 8

        model = _make_model()
        _, state = _run(config, model=model)

        row = state["adapter_manifest_status"].get("semantic")
        assert row is not None
        assert row["severity"] == "yellow"


class TestCorruptManifest:
    def test_corrupt_meta_json_without_patch_gives_no_matching_slot(self, tmp_path: Path) -> None:
        """Corrupt meta.json causes find_live_slot to skip the slot → no_matching_slot.

        A real registry is present (registry_present=True) so the binding
        distinguishes this from the absent-registry shape — the point under
        test is the corrupt manifest, not the registry's presence.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        (kind_dir / "indexed_key_registry.json").write_text('{"active_keys": []}')
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        (slot / "meta.json").write_text("{bad json")

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        # find_live_slot skips unreadable meta.json and returns None → no_matching_slot
        assert row is not None
        assert row["status"] == "no_matching_slot"


class TestManifestMissing:
    def test_weights_present_no_meta_classified_as_fresh(self, tmp_path: Path) -> None:
        """A subdir with adapter weights but no meta.json is not a real slot.

        meta.json is the authoritative marker of a committed adapter slot (see
        find_live_slot in manifest.py, which gates slot discovery on a readable
        meta.json).  A subdir lacking meta.json — whether it holds weights, only
        progress.json, or any other content — is treated as a non-slot and the
        tier is classified as fresh (no manifest row emitted).

        The ``manifest_missing`` status now only ever comes from
        ``_load_one``'s load-failure fallback (an actual PEFT mount
        failure, app.py) — not from ``_validate_adapter_slot``, which no
        longer re-reads a matched slot's manifest (it reads
        ``binding.manifest``, already parsed inside
        ``verify_tier_binding``). A slot whose manifest disappears between
        ``verify_tier_binding``'s own match-and-parse and a caller using
        the result is therefore a race ``verify_tier_binding`` itself
        resolves to ``NO_MATCHING_SLOT`` (see
        ``tests/adapters/test_registry_binding.py``'s race-arm test) — it
        never reaches this function as ``manifest_missing``.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        # No meta.json written — subdir is not a real slot.

        _, state = _run(config)
        # No real slot found; classifier treats the tier as fresh — no row.
        row = state["adapter_manifest_status"].get("episodic")
        assert row is None


class TestEnabledFalse:
    def test_disabled_adapter_has_no_row(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, adapter_names=("episodic",), enabled_names=())
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        _write_slot(kind_dir)

        _, state = _run(config)
        assert "episodic" not in state["adapter_manifest_status"]


class TestNoMatchingSlot:
    def test_registry_hash_mismatch_gives_no_matching_slot(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # A real, readable registry so the binding is REGISTRY-present — the
        # slot's stamped hash simply does not match it.
        (kind_dir / "indexed_key_registry.json").write_text('{"active_keys": ["k1"]}')
        _write_slot(kind_dir, registry_sha256="old_hash")

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "red"  # episodic is primary

    def test_absent_registry_with_slot_gives_registry_unverified(self, tmp_path: Path) -> None:
        """A real weight-slot candidate but NO indexed_key_registry.json at all
        (distinct from "registry present, hash mismatch") is a different,
        more severe verdict — the registry itself cannot be corroborated,
        not just unmatched."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # No indexed_key_registry.json — registry_present=False.
        _write_slot(kind_dir, registry_sha256="old_hash")

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "registry_unverified"
        assert row["reason"] == "registry_absent_with_slots"
        assert row["severity"] == "red"  # episodic is primary

    def test_progress_only_stub_dir_classified_as_fresh(self, tmp_path: Path) -> None:
        """A subdir containing only progress.json (aborted training run) must NOT
        trigger no_matching_slot.  The tier has no real weight-bearing slot, so
        the classifier must report fresh install (no manifest row, info log only).

        Regression guard: before the fix, has_slots counted any non-hidden subdir,
        which caused the stub to be mistaken for a real slot and produced a false
        red adapter_no_matching_slot_primary incident even though boot-time integrity
        passed and the registry was genuinely absent.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # Replicate the on-disk shape of an aborted interim training run:
        # a timestamped subdir with only progress.json, no meta.json or weights.
        stub = kind_dir / "interim_20260619T1200"
        stub.mkdir()
        (stub / "progress.json").write_text('{"phase": "phase4-episodic", "epoch": 0}')

        _, state = _run(config)

        # No manifest row — tier must be treated as fresh/empty, not corrupt.
        assert "episodic" not in state["adapter_manifest_status"]

    def test_empty_tier_dir_classified_as_fresh(self, tmp_path: Path) -> None:
        """A tier directory with no subdirs at all is fresh install — no row emitted."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # No subdirs at all.

        _, state = _run(config)

        assert "episodic" not in state["adapter_manifest_status"]

    def test_real_slot_wrong_hash_still_gives_no_matching_slot(self, tmp_path: Path) -> None:
        """A directory that contains meta.json (a real slot) but whose registry hash
        does not match the live registry hash must STILL produce no_matching_slot.

        This is the genuine-mismatch / possible-corruption case that warrants the
        red incident and investigate/restore guidance.
        """
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        # A real, readable registry so the binding is REGISTRY-present.
        (kind_dir / "indexed_key_registry.json").write_text('{"active_keys": ["k1"]}')
        # Real slot: meta.json written with a non-matching registry_sha256.
        _write_slot(kind_dir, registry_sha256="stale_hash_from_old_training_run")
        # Also place a progress.json stub alongside it — must not affect outcome.
        stub = kind_dir / "interim_20260619T1200"
        stub.mkdir()
        (stub / "progress.json").write_text('{"phase": "phase4-episodic", "epoch": 0}')

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None, "Real weight-bearing slot with hash mismatch must produce a row"
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "red"  # episodic is primary


class TestInterimNoMatchingSlotLogLevel:
    """Interim tiers now route through the same _validate_adapter_slot
    decision tree as main tiers, so their log level / row shape follows the
    shared classifier exactly: INFO + no row for the benign "no weight-slot
    candidate at all" shape (simulate mode never creates a timestamped
    weight slot); WARNING + a yellow no_matching_slot row when a real
    weight-slot candidate (a subdir with meta.json) exists but its hash
    doesn't match — the genuinely torn case.
    """

    def test_no_weight_slot_candidate_is_info_fresh_install_no_row(
        self, tmp_path: Path, caplog
    ) -> None:
        """Simulate-mode shape (graph.json + registry, no meta.json anywhere)
        → nothing mounted, INFO-level fresh-install log, no manifest row.

        The registry carries a known key so the boot-time keyless-tier sweep
        preserves this slot instead of reaping it before mounting — an empty
        registry here would be reaped pre-mount and the mount-loop
        classifier under test would never run.
        """
        import logging

        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260619T1200"
        interim_dir.mkdir(parents=True)
        (interim_dir / "graph.json").write_text("{}")
        (interim_dir / "indexed_key_registry.json").write_text('{"active_keys": ["k1"]}')

        caplog.set_level(logging.INFO, logger="paramem.server.app")
        model, state = _run(config)

        assert "episodic_interim_20260619T1200" not in getattr(model, "peft_config", {})
        assert "episodic_interim_20260619T1200" not in state["adapter_manifest_status"]

        warning_or_error = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert not any("episodic_interim_20260619T1200" in msg for msg in warning_or_error), (
            f"Benign simulate-mode shape must not log WARNING/ERROR, got: {warning_or_error}"
        )
        assert any(
            "episodic_interim_20260619T1200" in msg and "fresh install" in msg
            for msg in info_messages
        ), f"Expected an INFO fresh-install log naming the interim adapter, got: {info_messages}"

    def test_weight_slot_candidate_with_hash_mismatch_is_warning_yellow_row(
        self, tmp_path: Path, caplog
    ) -> None:
        """A real weight-slot candidate (meta.json) whose hash doesn't match
        → nothing mounted, WARNING log, yellow no_matching_slot row keyed by
        the interim adapter name.

        The registry carries a known key so the boot-time keyless-tier sweep
        preserves this slot instead of reaping it before mounting — an empty
        registry here would be reaped pre-mount and the mount-loop
        classifier under test would never run.
        """
        import logging

        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260619T1200"
        interim_dir.mkdir(parents=True)
        (interim_dir / "indexed_key_registry.json").write_text('{"active_keys": ["k1"]}')
        # Real weight-slot candidate whose registry_sha256 will not match the
        # live (non-empty, drifted) registry's hash.
        _write_slot(interim_dir, registry_sha256="stale_hash_from_old_training_run")

        caplog.set_level(logging.WARNING, logger="paramem.server.app")
        model, state = _run(config)

        assert "episodic_interim_20260619T1200" not in getattr(model, "peft_config", {})
        row = state["adapter_manifest_status"].get("episodic_interim_20260619T1200")
        assert row is not None
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "yellow"

        warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert not any("episodic_interim_20260619T1200" in msg for msg in error_messages), (
            f"Torn interim shape must not log ERROR, got: {error_messages}"
        )
        assert any("episodic_interim_20260619T1200" in msg for msg in warning_messages), (
            f"Expected a WARNING naming the interim adapter, got: {warning_messages}"
        )


class TestInterimHealthyMountAndFingerprintMismatch:
    """Interim tiers get the full :func:`_validate_adapter_slot` treatment:
    a healthy slot mounts with no row (and clears any stale pre-existing
    row for that name); a fingerprint mismatch against
    ``config.adapters.episodic`` (interim slots are episodic-shaped) skips
    the mount and writes a yellow ``mismatch`` row naming the field.
    """

    def test_healthy_interim_slot_mounts_and_clears_stale_row(self, tmp_path: Path) -> None:
        """Binds the PeftModel.from_pretrained patch and asserts the actual
        mount call fired — a pin on only the popped stale row would stay
        green even if the mount itself silently regressed, since the
        fresh-install branch also pops a stale row without mounting."""
        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260619T1200"
        interim_dir.mkdir(parents=True)
        _write_slot(interim_dir, registry_sha256="")

        state = {
            "adapter_manifest_status": {
                "episodic_interim_20260619T1200": {
                    "status": "no_matching_slot",
                    "reason": "no_matching_slot",
                    "field": None,
                    "severity": "yellow",
                    "slot_path": None,
                    "checked_at": "2026-06-19T12:00:00Z",
                }
            },
            "base_model_hash_cache": {},
        }

        from peft import PeftModel

        with patch.object(
            PeftModel, "from_pretrained", return_value=MagicMock(spec=PeftModel)
        ) as mock_from_pretrained:
            _, state = _run(config, state=state)

        mock_from_pretrained.assert_called_once()
        assert (
            mock_from_pretrained.call_args.kwargs.get("adapter_name")
            == "episodic_interim_20260619T1200"
        ), "Expected the interim slot to actually be mounted, not just the row cleared"
        assert "episodic_interim_20260619T1200" not in state["adapter_manifest_status"], (
            "Healthy interim mount must clear a pre-existing stale row"
        )

    def test_interim_fingerprint_mismatch_not_mounted_yellow_row(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260619T1200"
        interim_dir.mkdir(parents=True)
        # config.adapters.episodic.rank is 8 (see _make_config); slot rank=4 mismatches.
        _write_slot(interim_dir, registry_sha256="", rank=4)

        model, state = _run(config)

        assert "episodic_interim_20260619T1200" not in getattr(model, "peft_config", {})
        row = state["adapter_manifest_status"].get("episodic_interim_20260619T1200")
        assert row is not None
        assert row["status"] == "mismatch"
        assert row["severity"] == "yellow"
        assert row["field"] == "lora.rank"

    def test_interim_mounts_even_when_all_main_tiers_disabled(self, tmp_path: Path) -> None:
        """No ``.enabled`` gate on the interim loop: interims stay
        episodic-shaped and mountable even when episodic/semantic/procedural
        are all disabled in config."""
        config = _make_config(tmp_path, enabled_names=())
        interim_dir = config.adapter_dir / "episodic" / "interim_20260619T1200"
        interim_dir.mkdir(parents=True)
        _write_slot(interim_dir, registry_sha256="")

        from peft import PeftModel

        with patch.object(
            PeftModel, "from_pretrained", return_value=MagicMock(spec=PeftModel)
        ) as mock_from_pretrained:
            _run(config)

        mock_from_pretrained.assert_called_once()
        assert (
            mock_from_pretrained.call_args.kwargs.get("adapter_name")
            == "episodic_interim_20260619T1200"
        ), "Interim slot must mount even with every main tier disabled"


class TestMalformedInterimStampSurvivesMount:
    """A stray ``episodic/interim_<malformed-stamp>/`` directory (e.g.
    ``interim_garbage/``) is a shape :func:`iter_interim_dirs` yields
    without validating the stamp (pinned separately at
    ``tests/test_interim_adapter_lifecycle.py::test_whole_ring_reap_removes_stray_malformed_stamp_dir``).
    ``_validate_adapter_slot`` must not crash boot on it: it receives
    ``kind_dir`` already resolved from its caller and never re-derives it
    from ``name`` via ``adapter_slot_root_for_name``/``interim_dir_for_name``
    (which would raise ``ValueError`` on the malformed stamp).
    """

    def test_stray_malformed_stamp_dir_mounts_nothing_no_row_no_raise(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        malformed_dir = config.adapter_dir / "episodic" / "interim_garbage"
        malformed_dir.mkdir(parents=True)
        # Flat adapter files directly under the malformed dir — NOT inside a
        # meta.json-bearing slot subdirectory — so this has no weight-slot
        # candidate and lands in the fresh-install branch (INFO, no row).
        (malformed_dir / "adapter_config.json").write_text("{}")
        (malformed_dir / "adapter_model.safetensors").write_bytes(b"w")

        model, state = _run(config)  # must not raise ValueError

        assert "episodic_interim_garbage" not in getattr(model, "peft_config", {})
        assert "episodic_interim_garbage" not in state["adapter_manifest_status"]


class TestEmptiedInterimSlotReapedAtBoot:
    """An interim simulate slot emptied by ``/speaker/forget`` (graph.json
    rewritten to zero edges, registry rewritten to zero known keys) is
    removed by the boot-time keyless-tier sweep before mounting is
    attempted.  This is the self-heal for a crash between the forget
    handler's registry write (a durable hard erase) and its own on-disk
    reap: ``POST /speaker/forget`` persists the emptied registry to disk
    before it unmounts the tier and deletes its artifacts, so a kill in
    that window leaves a stale, keyless slot directory behind whose
    manifest still carries the pre-erase hash.  Sweeping any tier whose
    registry legitimately tracks zero keys, unconditionally, at every boot,
    means that window never needs a dedicated recovery path.
    """

    def test_zero_edge_zero_key_interim_slot_dir_is_removed(self, tmp_path: Path) -> None:
        """graph.json with no edges + registry with zero known keys → the
        interim directory (and both files) are gone after mounting."""
        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260803T1200"
        interim_dir.mkdir(parents=True)
        (interim_dir / "graph.json").write_text('{"directed": true, "nodes": [], "links": []}')
        (interim_dir / "indexed_key_registry.json").write_text(
            '{"active_keys": [], "fidelity_history": {}, "stale": {}, "simhash": {}}'
        )

        _run(config)

        assert not interim_dir.exists(), "a keyless interim slot must be reaped at boot"


class TestMultipleRows:
    def test_multiple_adapter_rows_independent(self, tmp_path: Path) -> None:
        config = _make_config(
            tmp_path, adapter_names=("episodic", "semantic"), enabled_names=("episodic", "semantic")
        )
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        # episodic: hash mismatch → row
        ep_dir = config.adapter_dir / "episodic"
        ep_dir.mkdir()
        _write_slot(ep_dir, sha="wrong_sha")

        # semantic: no slot at all → no row (fresh install)
        sem_dir = config.adapter_dir / "semantic"
        sem_dir.mkdir()

        model = _make_model(commit_hash="abc123")
        _, state = _run(config, model=model)

        assert "episodic" in state["adapter_manifest_status"]
        assert "semantic" not in state["adapter_manifest_status"]


class TestSynthesizedUnknown:
    def test_synthesized_true_unknown_episodic_is_yellow(self, tmp_path: Path) -> None:
        """synthesized=True + UNKNOWN fields → yellow even for episodic."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo=UNKNOWN, sha=UNKNOWN, hash=UNKNOWN),
            tokenizer=TokenizerFingerprint(
                name_or_path=UNKNOWN, vocab_size=UNKNOWN, merges_hash=UNKNOWN
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
            registry_sha256="",
            key_count=UNKNOWN,
            synthesized=True,
        )
        write_manifest(slot, m)

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["severity"] == "yellow"
        assert row["status"] == "migrated_unverified"

    def test_synthesized_false_unknown_episodic_is_red(self, tmp_path: Path) -> None:
        """synthesized=False + UNKNOWN fields → red (fresh-built manifest failure)."""
        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo=UNKNOWN, sha=UNKNOWN, hash=UNKNOWN),
            tokenizer=TokenizerFingerprint(
                name_or_path=UNKNOWN, vocab_size=UNKNOWN, merges_hash=UNKNOWN
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
            registry_sha256="",
            key_count=UNKNOWN,
            synthesized=False,
        )
        write_manifest(slot, m)

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["severity"] == "red"
        assert row["status"] == "migrated_unverified"

    def test_synthesized_false_unknown_semantic_is_yellow(self, tmp_path: Path) -> None:
        """synthesized=False + UNKNOWN fields on a NON-primary tier (semantic)
        → yellow, not red. Severity red is reserved for the primary tier —
        it is what makes _collect_adapter_fingerprint_items render a
        failed-level "PA routing DISABLED" item (attention.py:550-560), and
        PA routing is only disabled by a primary-tier problem."""
        config = _make_config(
            tmp_path, adapter_names=("episodic", "semantic"), enabled_names=("semantic",)
        )
        kind_dir = config.adapter_dir / "semantic"
        kind_dir.mkdir()
        slot = kind_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"w")
        m = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="semantic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo=UNKNOWN, sha=UNKNOWN, hash=UNKNOWN),
            tokenizer=TokenizerFingerprint(
                name_or_path=UNKNOWN, vocab_size=UNKNOWN, merges_hash=UNKNOWN
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
            registry_sha256="",
            key_count=UNKNOWN,
            synthesized=False,
        )
        write_manifest(slot, m)

        _, state = _run(config)

        row = state["adapter_manifest_status"].get("semantic")
        assert row is not None
        assert row["severity"] == "yellow"
        assert row["status"] == "migrated_unverified"


class TestRevalidateAdapterManifests:
    """Post-cycle revalidation shares the same per-tier decision tree as the
    boot validator (_validate_adapter_slot) — for BOTH main and interim
    tiers, the single row-freshness owner for the whole
    ``adapter_manifest_status`` dict. These tests exercise
    _revalidate_adapter_manifests directly to verify:

    1. Stale RED rows from the boot snapshot are CLEARED when on-disk slots
       are now healthy (the bug this function exists to fix).
    2. Slots that genuinely became unhealthy after boot get a fresh row
       written, with current ``checked_at``.
    3. Stale ``adapter_manifest_status`` rows keyed by an interim adapter
       name whose on-disk directory is gone entirely are pruned.
    4. A live, on-disk interim dir is actively revalidated (not merely left
       alone): a healthy interim clears a stale problematic row; a torn
       interim gets an accurate row reflecting its current on-disk state.
    """

    def _state_from_config(self, config, model=None, tokenizer=None):
        return {
            "config": config,
            "model": model if model is not None else _make_model(),
            "tokenizer": tokenizer if tokenizer is not None else _make_tokenizer(),
            "adapter_manifest_status": {},
            "base_model_hash_cache": {},
        }

    def test_clears_stale_red_row_when_slot_now_healthy(self, tmp_path: Path) -> None:
        """A boot-time row exists for episodic; on-disk state is healthy.
        Revalidation removes the row.
        """
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path)
        # Healthy slot on disk with empty registry hash (matches "no registry").
        episodic_dir = config.adapter_dir / "episodic"
        _write_slot(episodic_dir, ts="20260427-105338", registry_sha256="")

        state = self._state_from_config(config)
        # Inject a stale boot-time RED row.
        state["adapter_manifest_status"]["episodic"] = {
            "status": "no_matching_slot",
            "reason": "no_matching_slot",
            "field": None,
            "severity": "red",
            "slot_path": None,
            "checked_at": "2026-04-27T11:43:44Z",
        }

        _revalidate_adapter_manifests(state)

        assert "episodic" not in state["adapter_manifest_status"], (
            "Stale RED row must be cleared once slot is healthy"
        )

    def test_writes_red_row_when_no_matching_slot(self, tmp_path: Path) -> None:
        """No matching slot on disk → revalidation writes a no_matching_slot row."""
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True, exist_ok=True)
        # A real, readable registry so the binding is REGISTRY-present — the
        # slot's stamped hash simply does not match it.
        (episodic_dir / "indexed_key_registry.json").write_text('{"active_keys": []}')
        _write_slot(episodic_dir, ts="20260427-105338", registry_sha256="stale_hash_123")

        state = self._state_from_config(config)
        # Start with no row (post-boot default for a healthy slot).
        _revalidate_adapter_manifests(state)

        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None, "Mismatch must produce a row"
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "red"  # episodic is primary

    def test_disabled_adapter_pops_any_existing_row(self, tmp_path: Path) -> None:
        """If a tier is disabled in config, its row is removed regardless of state."""
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path, enabled_names=())  # all tiers disabled
        state = self._state_from_config(config)
        state["adapter_manifest_status"]["episodic"] = {
            "status": "mismatch",
            "severity": "red",
            "reason": "fingerprint_mismatch",
            "field": None,
            "slot_path": None,
            "checked_at": "2026-04-27T11:00:00Z",
        }

        _revalidate_adapter_manifests(state)

        assert "episodic" not in state["adapter_manifest_status"]

    def test_noop_when_state_missing_model(self, tmp_path: Path) -> None:
        """Defensive: missing model in state → silently no-op (no exception)."""
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path)
        state = {"config": config, "tokenizer": _make_tokenizer()}  # no "model" key
        _revalidate_adapter_manifests(state)  # must not raise
        assert state.get("adapter_manifest_status", {}) == {}

    def test_prunes_gone_interim_row_and_clears_stale_main_row(self, tmp_path: Path) -> None:
        """A full cycle can retire an interim slot entirely (dir removed from
        disk) — the stale row for that now-gone interim must be popped, same
        as before this dir ever gets a chance to be revalidated. Main-tier
        rows behave exactly as the pins above (stale RED cleared once
        healthy)."""
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path)
        # Healthy main episodic slot — stale RED row must be cleared.
        episodic_dir = config.adapter_dir / "episodic"
        _write_slot(episodic_dir, ts="20260427-105338", registry_sha256="")

        state = self._state_from_config(config)
        state["adapter_manifest_status"]["episodic"] = {
            "status": "no_matching_slot",
            "reason": "no_matching_slot",
            "field": None,
            "severity": "red",
            "slot_path": None,
            "checked_at": "2026-04-27T11:43:44Z",
        }
        # Row for an interim dir that no longer exists on disk at all — must
        # be popped by the gone-dir prune (it never reaches _validate_adapter_slot).
        state["adapter_manifest_status"]["episodic_interim_20260601T0000"] = {
            "status": "no_matching_slot",
            "reason": "no_matching_slot",
            "field": None,
            "severity": "yellow",
            "slot_path": None,
            "checked_at": "2026-04-27T11:43:44Z",
        }

        _revalidate_adapter_manifests(state)

        assert "episodic" not in state["adapter_manifest_status"], (
            "Stale RED main-tier row must be cleared once slot is healthy"
        )
        assert "episodic_interim_20260601T0000" not in state["adapter_manifest_status"], (
            "Row for a gone interim dir must be pruned"
        )

    def test_live_healthy_interim_clears_stale_row(self, tmp_path: Path) -> None:
        """A live, on-disk interim dir is now actively revalidated (not just
        left alone) — a stale problematic row for a healthy interim slot
        must be cleared by _revalidate_adapter_manifests, the same way a
        healthy main-tier slot clears its stale row."""
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260803T1200"
        interim_dir.mkdir(parents=True)
        # Fingerprints match config.adapters.episodic (rank=8) and the
        # default _make_model() (sha="abc123") — a genuinely healthy slot.
        _write_slot(interim_dir, registry_sha256="")

        state = self._state_from_config(config)
        state["adapter_manifest_status"]["episodic_interim_20260803T1200"] = {
            "status": "no_matching_slot",
            "reason": "no_matching_slot",
            "field": None,
            "severity": "yellow",
            "slot_path": None,
            "checked_at": "2026-04-27T11:43:44Z",
        }

        _revalidate_adapter_manifests(state)

        assert "episodic_interim_20260803T1200" not in state["adapter_manifest_status"], (
            "Stale row for a now-healthy live interim slot must be cleared"
        )

    def test_live_torn_interim_gets_accurate_no_matching_slot_row(self, tmp_path: Path) -> None:
        """A live, on-disk interim dir whose slot hash doesn't match the
        live registry gets (or keeps, refreshed to accurate content) a
        yellow no_matching_slot row — proving revalidation actually re-runs
        the decision tree against the interim's current on-disk state
        rather than trusting a pre-existing row."""
        from paramem.server.app import _revalidate_adapter_manifests

        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260803T1200"
        interim_dir.mkdir(parents=True)
        # A real, readable registry so the binding is REGISTRY-present.
        (interim_dir / "indexed_key_registry.json").write_text('{"active_keys": []}')
        _write_slot(interim_dir, registry_sha256="stale_hash_from_old_training_run")

        state = self._state_from_config(config)
        # Seed a stale row with the WRONG severity/reason to prove revalidation
        # overwrites it with accurate content rather than leaving it as-is.
        state["adapter_manifest_status"]["episodic_interim_20260803T1200"] = {
            "status": "mismatch",
            "reason": "fingerprint_mismatch",
            "field": "lora.rank",
            "severity": "red",
            "slot_path": None,
            "checked_at": "2026-04-27T11:43:44Z",
        }

        _revalidate_adapter_manifests(state)

        row = state["adapter_manifest_status"].get("episodic_interim_20260803T1200")
        assert row is not None
        assert row["status"] == "no_matching_slot"
        assert row["severity"] == "yellow"  # interim is never primary


# ---------------------------------------------------------------------------
# Eager consolidation-loop creation — keeps /status's adapter_loaded reading
# symmetric across a restart (local-mode boot mounts the adapters
# immediately, instead of only once the first consolidation door runs) and
# across a release→acquire cycle (_live_reload_base_model's tail).
# ---------------------------------------------------------------------------


class TestEagerConsolidationLoopCreation:
    def test_creates_loop_when_model_tokenizer_and_store_present(self) -> None:
        """Local-mode state (model + tokenizer + memory_store all present)
        creates the consolidation loop via the shared get-or-create."""
        from paramem.server import app as app_module

        config = MagicMock(name="config")
        state = {
            "model": MagicMock(name="model"),
            "tokenizer": MagicMock(name="tokenizer"),
            "memory_store": MagicMock(name="memory_store"),
            "consolidation_loop": None,
        }

        with (
            patch.object(app_module, "_state", state),
            patch.object(app_module, "_get_or_create_consolidation_loop") as mock_get_or_create,
        ):
            app_module._eager_create_consolidation_loop(config)

        mock_get_or_create.assert_called_once_with(config)

    def test_noop_in_cloud_only_mode(self) -> None:
        """No model resident (cloud-only) — the loop is never created."""
        from paramem.server import app as app_module

        config = MagicMock(name="config")
        state = {
            "model": None,
            "tokenizer": None,
            "memory_store": None,
            "consolidation_loop": None,
        }

        with (
            patch.object(app_module, "_state", state),
            patch.object(app_module, "_get_or_create_consolidation_loop") as mock_get_or_create,
        ):
            app_module._eager_create_consolidation_loop(config)

        mock_get_or_create.assert_not_called()

    def test_noop_when_loop_already_exists(self) -> None:
        """Idempotent: a second call (loop already resident) does not build
        another one — proven through the real create_consolidation_loop
        factory, not just a mocked get-or-create."""
        from paramem.server import app as app_module

        state = {
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
                app_module, "create_consolidation_loop", return_value=fake_loop
            ) as mock_create,
        ):
            app_module._eager_create_consolidation_loop(MagicMock(name="config"))
            app_module._eager_create_consolidation_loop(MagicMock(name="config"))

        mock_create.assert_called_once()


class TestLifespanEagerLoopWiring:
    def test_lifespan_invokes_eager_create_consolidation_loop(self) -> None:
        """The lifespan boot path must call _eager_create_consolidation_loop
        after config-derived state (memory_store) is built, so a refactor
        that drops the call fails here rather than silently."""
        import inspect

        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert "_eager_create_consolidation_loop(" in source, (
            "lifespan must call _eager_create_consolidation_loop so "
            "adapter_loaded is symmetric across a restart"
        )


# ---------------------------------------------------------------------------
# Boot-time keyless-tier sweep — self-heals a crash between a hard key erase
# (POST /speaker/forget) and its own on-disk reap, and reclassifies any
# payload-bearing tier whose registry legitimately tracks zero keys.
# ---------------------------------------------------------------------------


class TestKeylessTierSweep:
    def test_main_tier_zero_known_keys_reaped_interim_child_survives(self, tmp_path: Path) -> None:
        """A main tier reduced to zero known keys is reaped, but a sibling
        interim slot living under it (a separate tier with its own known
        keys) is untouched — reap_tier_artifacts never removes interim_*
        children of a main tier root."""
        from paramem.server.app import _sweep_keyless_tier_artifacts

        config = _make_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        (episodic_dir / "indexed_key_registry.json").write_text(
            '{"active_keys": [], "fidelity_history": {}, "stale": {}, "simhash": {}}'
        )
        interim_child = episodic_dir / "interim_20260803T1200"
        interim_child.mkdir()
        (interim_child / "graph.json").write_text('{"directed": true, "nodes": [], "links": []}')
        (interim_child / "indexed_key_registry.json").write_text(
            '{"active_keys": ["k1"], "fidelity_history": {}, "stale": {}, "simhash": {}}'
        )

        reaped = _sweep_keyless_tier_artifacts(config)

        assert reaped == ["episodic"]
        assert not (episodic_dir / "indexed_key_registry.json").exists()
        assert episodic_dir.exists(), "root survives — the interim child keeps it non-empty"
        assert interim_child.exists()
        assert (interim_child / "graph.json").exists()
        assert (interim_child / "indexed_key_registry.json").exists()

    def test_main_tier_stale_only_registry_preserved_and_errors_when_payloadless(
        self, tmp_path: Path, caplog
    ) -> None:
        """A registry whose only known key is stale (known ⊇ active, active
        is empty) is preserved, not reaped — list_known(), not list_active(),
        is the sweep's emptiness test. With no adapter weights and no
        graph.json at the tier root, this also fires the reworded ERROR
        naming the known-key count."""
        import logging

        from paramem.server.app import _sweep_keyless_tier_artifacts

        config = _make_config(tmp_path)
        semantic_dir = config.adapter_dir / "semantic"
        semantic_dir.mkdir(parents=True)
        (semantic_dir / "indexed_key_registry.json").write_text(
            json.dumps(
                {
                    "active_keys": [],
                    "fidelity_history": {},
                    "stale": {"k1": {"stale_since": "2026-08-01T00:00:00Z", "stale_cycles": 0}},
                    "simhash": {},
                }
            )
        )

        caplog.set_level(logging.ERROR, logger="paramem.server.app")
        reaped = _sweep_keyless_tier_artifacts(config)

        assert reaped == []
        assert semantic_dir.exists()
        assert (semantic_dir / "indexed_key_registry.json").exists()
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("semantic" in msg and "known key" in msg for msg in error_messages), (
            f"Expected a reworded 'known key(s)' ERROR, got: {error_messages}"
        )

    def test_interim_tier_active_keys_no_payload_preserved_and_errors(
        self, tmp_path: Path, caplog
    ) -> None:
        """An INTERIM slot whose registry lists ACTIVE keys but carries
        neither adapter weights nor a graph.json is preserved and logged as
        an ERROR — unfolded facts must not be deleted. Pins the INTERIM
        branch of the payload-less classification (the main-tier branch is
        pinned by test_main_tier_stale_only_registry_preserved_and_errors_when_payloadless
        above)."""
        import logging

        from paramem.server.app import _sweep_keyless_tier_artifacts

        config = _make_config(tmp_path)
        interim_dir = config.adapter_dir / "episodic" / "interim_20260803T1200"
        interim_dir.mkdir(parents=True)
        (interim_dir / "indexed_key_registry.json").write_text(
            json.dumps(
                {
                    "active_keys": ["graph1"],
                    "fidelity_history": {},
                    "stale": {},
                    "simhash": {},
                }
            )
        )

        caplog.set_level(logging.ERROR, logger="paramem.server.app")
        reaped = _sweep_keyless_tier_artifacts(config)

        assert reaped == []
        assert interim_dir.exists(), "unfolded facts must not be deleted"
        assert (interim_dir / "indexed_key_registry.json").exists()
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any(
            "episodic_interim_20260803T1200" in msg and "known key" in msg for msg in error_messages
        ), f"Expected a reworded 'known key(s)' ERROR, got: {error_messages}"

    def test_unreadable_registry_is_preserved_and_logged(self, tmp_path: Path, caplog) -> None:
        """A registry that raises on load (corrupt file, failed decrypt) is
        never inferred to hold zero keys — it is preserved and logged."""
        import logging

        from paramem.server.app import _sweep_keyless_tier_artifacts

        config = _make_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        reg_path = episodic_dir / "indexed_key_registry.json"
        reg_path.write_text('{"active_keys": []}')

        caplog.set_level(logging.ERROR, logger="paramem.server.app")
        with patch(
            "paramem.training.key_registry.KeyRegistry.load",
            side_effect=RuntimeError("decrypt failed"),
        ):
            reaped = _sweep_keyless_tier_artifacts(config)

        assert reaped == []
        assert reg_path.exists(), "an unreadable registry must never be swept"
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("episodic" in msg and "unreadable" in msg for msg in error_messages), (
            f"Expected an unreadable-registry ERROR, got: {error_messages}"
        )

    def test_foreign_shaped_registry_with_real_weights_is_preserved_and_errors(
        self, tmp_path: Path, caplog
    ) -> None:
        """A parseable-but-foreign registry payload (e.g. ``{}``) must never
        be inferred to hold zero keys just because ``KeyRegistry.load`` is
        tolerant by contract and shrugs it into an empty registry.
        ``KeyRegistry.load_simhashes`` refuses anything that is not
        affirmatively KeyRegistry-shaped, so a foreign shape is preserved
        and logged rather than reaped — even with real trained weights
        sitting right next to it, which the sweep must never delete."""
        import logging

        from paramem.server.app import _sweep_keyless_tier_artifacts

        config = _make_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        slot = episodic_dir / "20260421-000000"
        slot.mkdir()
        (slot / "adapter_config.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"weights")
        reg_path = episodic_dir / "indexed_key_registry.json"
        reg_path.write_text("{}")

        caplog.set_level(logging.ERROR, logger="paramem.server.app")
        reaped = _sweep_keyless_tier_artifacts(config)

        assert reaped == []
        assert reg_path.exists(), "a foreign-shaped registry must never be swept"
        assert (slot / "adapter_model.safetensors").exists(), "trained weights must survive"
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("episodic" in msg and "KeyRegistry-shaped" in msg for msg in error_messages), (
            f"Expected a not-affirmatively-KeyRegistry-shaped ERROR, got: {error_messages}"
        )

    def test_absent_registry_tier_is_untouched(self, tmp_path: Path) -> None:
        """A tier directory with no indexed_key_registry.json at all (fresh
        install) is not this sweep's business — left exactly as found."""
        from paramem.server.app import _sweep_keyless_tier_artifacts

        config = _make_config(tmp_path)
        procedural_dir = config.adapter_dir / "procedural"
        procedural_dir.mkdir(parents=True)
        (procedural_dir / "some_other_file.txt").write_text("noop")

        reaped = _sweep_keyless_tier_artifacts(config)

        assert reaped == []
        assert procedural_dir.exists()
        assert (procedural_dir / "some_other_file.txt").exists()

    def test_reaped_payload_bearing_tier_produces_no_manifest_row(self, tmp_path: Path) -> None:
        """Behavioural reclassification: a tier that carries real adapter
        weights but whose registry has already dropped to zero known keys
        is reaped, not surfaced — the old post-mount check never even read
        the registry for a payload-bearing dir, so this shape used to be
        left mounted-but-orphaned. After the sweep reaps it pre-mount, the
        boot validator sees a fresh install and emits no row."""
        config = _make_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        _write_slot(episodic_dir, registry_sha256="whatever-stale-hash")
        (episodic_dir / "indexed_key_registry.json").write_text(
            '{"active_keys": [], "fidelity_history": {}, "stale": {}, "simhash": {}}'
        )

        _, state = _run(config)

        assert "episodic" not in state["adapter_manifest_status"]
        assert not episodic_dir.exists(), "the emptied tier is fully reaped, not just unmounted"

    def test_sweep_runs_before_any_slot_is_resolved(self, tmp_path: Path) -> None:
        """The keyless-tier sweep runs before find_live_slot resolves any
        tier's slot — mirrors the sweep_orphan_pending-before-find_live_slot
        ordering already load-bearing inside _validate_adapter_slot."""
        from paramem.adapters.registry_binding import find_live_slot as real_find_live_slot
        from paramem.server import app as app_module

        config = _make_config(tmp_path)
        kind_dir = config.adapter_dir / "episodic"
        kind_dir.mkdir(parents=True)
        _write_slot(kind_dir, registry_sha256="")

        call_order: list[str] = []
        real_sweep = app_module._sweep_keyless_tier_artifacts

        def _tracked_sweep(cfg):
            call_order.append("sweep")
            return real_sweep(cfg)

        def _tracked_find_live_slot(*args, **kwargs):
            call_order.append("find_live_slot")
            return real_find_live_slot(*args, **kwargs)

        with (
            patch.object(app_module, "_sweep_keyless_tier_artifacts", side_effect=_tracked_sweep),
            patch(
                "paramem.adapters.registry_binding.find_live_slot",
                side_effect=_tracked_find_live_slot,
            ),
        ):
            _run(config)

        assert call_order, "expected both the sweep and find_live_slot to be called"
        assert call_order[0] == "sweep", "the sweep must run before any slot is resolved"
        assert "find_live_slot" in call_order

    def test_resume_pending_reaps_runs_before_keyless_scan(self, tmp_path: Path) -> None:
        """resume_pending_reaps (the crash-safe reap's tombstone resume) is
        the sweep's first action — it must run before the keyless-tier
        registry scan begins, not merely before find_live_slot. Moving the
        call to after the scan loop (but still inside the sweep, still
        before find_live_slot) would pass an order check pinned only against
        find_live_slot, so this test tracks the scan's own KeyRegistry.load
        call directly."""
        from paramem.memory import persistence as persistence_module
        from paramem.server.app import _sweep_keyless_tier_artifacts
        from paramem.training.key_registry import KeyRegistry

        config = _make_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        (episodic_dir / "indexed_key_registry.json").write_text(
            '{"active_keys": [], "fidelity_history": {}, "stale": {}, "simhash": {}}'
        )

        call_order: list[str] = []
        real_resume = persistence_module.resume_pending_reaps
        real_load = KeyRegistry.load

        def _tracked_resume(adapter_dir):
            call_order.append("resume")
            return real_resume(adapter_dir)

        def _tracked_load(path):
            call_order.append("scan")
            return real_load(path)

        with (
            patch.object(persistence_module, "resume_pending_reaps", side_effect=_tracked_resume),
            patch(
                "paramem.training.key_registry.KeyRegistry.load",
                side_effect=_tracked_load,
            ),
        ):
            _sweep_keyless_tier_artifacts(config)

        assert call_order == ["resume", "scan"], (
            f"resume_pending_reaps must run before the keyless registry scan, got {call_order}"
        )

    def test_pending_delete_leftover_is_resumed_at_boot(self, tmp_path: Path) -> None:
        """A ``.pending-delete/`` leftover from a prior crash (a
        reap_tier_artifacts deletion interrupted mid-rmtree) is fully
        resumed before mount — the stray content and the tombstone dir
        itself are both gone after boot."""
        from paramem.memory.persistence import _PENDING_DELETE_DIR_NAME

        config = _make_config(tmp_path)
        stray = config.adapter_dir / _PENDING_DELETE_DIR_NAME / "interim_20260417T0000"
        stray.mkdir(parents=True)
        (stray / "indexed_key_registry.json").write_text("{}")

        _run(config)

        assert not (config.adapter_dir / _PENDING_DELETE_DIR_NAME).exists()


# ---------------------------------------------------------------------------
# Registry-loss integration pins: both surfaces that verify a tier's
# registry↔slot binding — the boot mount validator (adapter_manifest_status)
# and the store-publish builder (_build_store_contents) — must agree on the
# same real, on-disk tree.
# ---------------------------------------------------------------------------


class TestRegistryLossChain:
    """Real temp-dir trees exercised through BOTH `_mount_adapters_from_slots`
    (the `adapter_manifest_status` row) and `_build_store_contents` (the
    published-registry map / incident), pinning that a single on-disk loss
    is observed consistently by every consumer of
    `verify_tier_binding`/`verify_adapter_tree`."""

    def _make_full_config(self, tmp_path: Path):
        """A config exposing both the mount-validator surface (`adapters.*`)
        and the store-builder surface (`key_metadata_path`, `paths.data`,
        `consolidation`, `inference`) against the SAME adapter_dir tree."""
        config = _make_config(tmp_path)
        config.key_metadata_path = tmp_path / "key_metadata.json"
        config.consolidation.mode = "simulate"
        config.consolidation.recall_probe_batch_size = 1
        config.inference.preload_cache = False
        config.paths.data = tmp_path
        return config

    def test_undecryptable_episodic_leaves_others_published(self, tmp_path: Path) -> None:
        """Episodic's registry is undecryptable; semantic and procedural are
        healthy. Episodic is absent from the published registry map AND
        carries a registry_unverified row; the other two tiers publish
        their real keys; the corrupt file itself is untouched afterward."""
        from paramem.backup.encryption import age_encrypt_bytes
        from paramem.backup.key_store import mint_daily_identity
        from paramem.server.app import _build_store_contents
        from paramem.training.key_registry import KeyRegistry

        config = self._make_full_config(tmp_path)

        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        encrypter = mint_daily_identity()
        corrupt_bytes = age_encrypt_bytes(b'{"active_keys":["ghost"]}', [encrypter.to_public()])
        episodic_reg_path = episodic_dir / "indexed_key_registry.json"
        episodic_reg_path.write_bytes(corrupt_bytes)

        semantic_dir = config.adapter_dir / "semantic"
        semantic_dir.mkdir(parents=True)
        sem_reg = KeyRegistry()
        sem_reg.add("sem_key1")
        sem_reg.save(semantic_dir / "indexed_key_registry.json")

        procedural_dir = config.adapter_dir / "procedural"
        procedural_dir.mkdir(parents=True)
        proc_reg = KeyRegistry()
        proc_reg.add("proc_key1")
        proc_reg.save(procedural_dir / "indexed_key_registry.json")

        # Surface 1: the store-publish builder.
        _, new_registry, _, stats = _build_store_contents(config, model=None, tokenizer=None)
        assert "episodic" not in new_registry
        assert new_registry["semantic"].list_active() == ["sem_key1"]
        assert new_registry["procedural"].list_active() == ["proc_key1"]
        assert stats["tier_bindings"]["episodic"].status == "registry_unreadable"

        # Surface 2: the boot mount validator.
        _, state = _run(config)
        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "registry_unverified"
        assert row["reason"] == "registry_unreadable"

        # The corrupt file itself is untouched by either pass.
        assert episodic_reg_path.read_bytes() == corrupt_bytes

    def test_key_count_mismatch_row_and_publish_suppressed(self, tmp_path: Path) -> None:
        """A slot stamped key_count=3 under a matching hash, while the
        registry actually holds 2 active keys, mints a key_count_mismatch
        row and is excluded from the published registry map."""
        from paramem.adapters.manifest import tier_registry_sha256
        from paramem.server.app import _build_store_contents
        from paramem.training.key_registry import KeyRegistry

        config = self._make_full_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        reg = KeyRegistry()
        reg.add("k1")
        reg.add("k2")
        reg.save(episodic_dir / "indexed_key_registry.json")
        live_hash = tier_registry_sha256(episodic_dir)
        _write_slot(episodic_dir, registry_sha256=live_hash, key_count=3)

        _, new_registry, _, stats = _build_store_contents(config, model=None, tokenizer=None)
        assert "episodic" not in new_registry
        assert stats["tier_bindings"]["episodic"].status == "key_count_mismatch"

        _, state = _run(config)
        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "key_count_mismatch"

    def test_no_matching_slot_row_and_tier_absent_from_publish(self, tmp_path: Path) -> None:
        """A healthy, readable registry with no matching slot mints a
        no_matching_slot row at the mount stage AND is excluded from the
        published registry map — NO_MATCHING_SLOT is not a publishable
        verdict, unlike NO_CANDIDATES."""
        from paramem.server.app import _build_store_contents
        from paramem.training.key_registry import KeyRegistry

        config = self._make_full_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        reg = KeyRegistry()
        reg.add("k1")
        reg.save(episodic_dir / "indexed_key_registry.json")
        # Slot stamped with a hash that will never match the real registry hash.
        _write_slot(episodic_dir, registry_sha256="stale_hash_does_not_match")

        _, new_registry, _, stats = _build_store_contents(config, model=None, tokenizer=None)
        assert "episodic" not in new_registry
        assert stats["tier_bindings"]["episodic"].status == "no_matching_slot"

        _, state = _run(config)
        row = state["adapter_manifest_status"].get("episodic")
        assert row is not None
        assert row["status"] == "no_matching_slot"

    def test_unverified_tier_error_names_the_restore_door(self, tmp_path: Path) -> None:
        """The incident detail for an unverified tier names POST
        /backup/restore + GET /integrity as the operator's exit — never
        /reconsolidate, which cannot rebuild a tier whose registry itself
        is unverified. The incident is the SOLE reporter for this
        condition (see attention.py's docstring) — the row-driven
        fingerprint populator stays silent for it, so there is exactly one
        surfaced item, not two."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place
        from paramem.server.attention import _collect_adapter_fingerprint_items
        from paramem.server.incidents import read_incidents

        config = self._make_full_config(tmp_path)
        episodic_dir = config.adapter_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        (episodic_dir / "indexed_key_registry.json").write_bytes(b"not json at all")

        live = MemoryStore()
        _hydrate_memory_store_in_place(live, config, model=None, tokenizer=None)

        incidents = read_incidents(tmp_path / "state")
        matching = [i for i in incidents if i.id == "tier_registry_unverified:episodic"]
        assert len(matching) == 1
        detail_text = str(matching[0].detail)
        assert "/backup/restore" in detail_text
        assert "/integrity" in detail_text
        assert "/reconsolidate" not in detail_text

        # The row-driven populator must NOT also emit an item for the same
        # condition — the incident above is the sole reporter.
        _, state = _run(config)
        items = _collect_adapter_fingerprint_items(state)
        assert items == []
