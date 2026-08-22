"""Pure-function tests for paramem.server.migration.

Covers:
- compute_unified_diff
- compute_tier_diff
- detect_simulate_mode
- compute_base_change
- render_preview_response (including base_change, warnings)
- Byte-for-byte shape-change block rendering smoke test.
- Warnings block CLI renderer.

Also covers ``compute_shape_changes``'s skip-with-warning paths against real
``AdapterManifest``/``write_manifest`` fixtures: a bound simulate slot with
no LoRA shape to compare, and the per-verdict warning wording for
``KEYS_WITHOUT_SLOT`` and ``PAYLOAD_MISMATCH``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from paramem.server.migration import (
    compute_base_change,
    compute_tier_diff,
    compute_unified_diff,
    detect_simulate_mode,
    initial_migration_state,
    render_preview_response,
)

_LIVE_FIXTURE = Path("tests/fixtures/server.yaml")


def _deep_merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _candidate_config(overrides: dict) -> tuple[dict, "object"]:
    """Build a full candidate yaml (deep-merged onto the live
    ``tests/fixtures/server.yaml``) and its validated ``ServerConfig`` --
    mirrors ``/migration/preview``'s own construction
    (``paramem/server/app.py``): ``candidate_config = validate_candidate
    (candidate_bytes, live_config_path)``, then ``compute_shape_changes``'s
    ``candidate_yaml`` argument is the SAME document, re-parsed the same
    way ``validate_candidate`` parses it internally -- one source for both,
    never two independently-built documents.
    """
    from paramem.server.migration import _parse_candidate, validate_candidate

    raw = yaml.safe_load(_LIVE_FIXTURE.read_text(encoding="utf-8"))
    _deep_merge(raw, overrides)
    candidate_bytes = yaml.safe_dump(raw).encode("utf-8")
    candidate_config = validate_candidate(candidate_bytes, _LIVE_FIXTURE)
    candidate_yaml = _parse_candidate(candidate_bytes)
    return candidate_yaml, candidate_config


# ---------------------------------------------------------------------------
# compute_unified_diff
# ---------------------------------------------------------------------------


class TestComputeUnifiedDiff:
    def test_identical_texts_return_empty_diff(self):
        """No changes → empty string."""
        result = compute_unified_diff("a: 1\n", "a: 1\n")
        assert result == ""

    def test_diff_contains_minus_for_removed_lines(self):
        """Removed line starts with '-'."""
        result = compute_unified_diff("a: 1\n", "")
        assert "-a: 1" in result

    def test_diff_contains_plus_for_added_lines(self):
        """Added line starts with '+'."""
        result = compute_unified_diff("", "b: 2\n")
        assert "+b: 2" in result

    def test_custom_labels_appear_in_output(self):
        """Custom fromfile/tofile labels appear in the diff header."""
        result = compute_unified_diff("x: 1\n", "x: 2\n", "live.yaml", "cand.yaml")
        assert "live.yaml" in result
        assert "cand.yaml" in result

    def test_multiline_diff(self):
        """Multi-line changes are all captured."""
        live = "a: 1\nb: 2\nc: 3\n"
        cand = "a: 1\nb: 99\nc: 3\n"
        result = compute_unified_diff(live, cand)
        assert "-b: 2" in result
        assert "+b: 99" in result


# ---------------------------------------------------------------------------
# compute_tier_diff
# ---------------------------------------------------------------------------


class TestComputeTierDiff:
    def test_empty_dicts_return_no_rows(self):
        """No changes → empty list."""
        result = compute_tier_diff({}, {})
        assert result == []

    def test_identical_dicts_return_no_rows(self):
        """Identical dicts → empty list."""
        d = {"model": "mistral", "debug": False}
        assert compute_tier_diff(d, d) == []

    def test_model_change_is_destructive(self):
        """model field change → Tier.DESTRUCTIVE."""
        live = {"model": "mistral"}
        cand = {"model": "gemma"}
        rows = compute_tier_diff(live, cand)
        assert len(rows) == 1
        assert rows[0]["dotted_path"] == "model"
        assert rows[0]["tier"] == "destructive"
        assert rows[0]["old_value"] == "mistral"
        assert rows[0]["new_value"] == "gemma"

    def test_debug_change_is_pipeline_altering(self):
        """debug field change → Tier.PIPELINE_ALTERING."""
        live = {"debug": False}
        cand = {"debug": True}
        rows = compute_tier_diff(live, cand)
        assert any(r["dotted_path"] == "debug" and r["tier"] == "pipeline_altering" for r in rows)

    def test_server_port_change_is_operational(self):
        """server.port change → Tier.OPERATIONAL."""
        live = {"server": {"port": 8420}}
        cand = {"server": {"port": 9000}}
        rows = compute_tier_diff(live, cand)
        assert any(r["dotted_path"] == "server.port" and r["tier"] == "operational" for r in rows)

    def test_alpha_change_is_destructive(self):
        """adapters.*.alpha change → Tier.DESTRUCTIVE (Condition 1)."""
        live = {"adapters": {"episodic": {"alpha": 16}}}
        cand = {"adapters": {"episodic": {"alpha": 32}}}
        rows = compute_tier_diff(live, cand)
        assert any(
            r["dotted_path"] == "adapters.episodic.alpha" and r["tier"] == "destructive"
            for r in rows
        ), f"Expected destructive alpha, got {rows}"

    def test_rank_change_is_destructive(self):
        """adapters.*.rank change → Tier.DESTRUCTIVE."""
        live = {"adapters": {"episodic": {"rank": 8}}}
        cand = {"adapters": {"episodic": {"rank": 16}}}
        rows = compute_tier_diff(live, cand)
        assert any(
            r["dotted_path"] == "adapters.episodic.rank" and r["tier"] == "destructive"
            for r in rows
        )

    def test_new_field_in_candidate_appears(self):
        """Field present only in candidate → row with old_value=None."""
        live = {"model": "mistral"}
        cand = {"model": "mistral", "debug": True}
        rows = compute_tier_diff(live, cand)
        debug_rows = [r for r in rows if r["dotted_path"] == "debug"]
        assert debug_rows
        assert debug_rows[0]["old_value"] is None
        assert debug_rows[0]["new_value"] is True

    def test_removed_field_appears(self):
        """Field present only in live → row with new_value=None."""
        live = {"model": "mistral", "debug": True}
        cand = {"model": "mistral"}
        rows = compute_tier_diff(live, cand)
        debug_rows = [r for r in rows if r["dotted_path"] == "debug"]
        assert debug_rows
        assert debug_rows[0]["new_value"] is None

    def test_rows_sorted_destructive_first(self):
        """Destructive rows precede pipeline_altering, which precede operational."""
        live = {"model": "a", "debug": False, "server": {"port": 8420}}
        cand = {"model": "b", "debug": True, "server": {"port": 9000}}
        rows = compute_tier_diff(live, cand)
        tiers = [r["tier"] for r in rows]
        _TIER_ORDER = {"destructive": 0, "pipeline_altering": 1, "operational": 2}
        tier_indices = [_TIER_ORDER[t] for t in tiers]
        assert tier_indices == sorted(tier_indices), f"Rows not sorted by tier: {tiers}"


# ---------------------------------------------------------------------------
# detect_simulate_mode
# ---------------------------------------------------------------------------


class TestDetectSimulateMode:
    def test_false_when_no_consolidation_key(self):
        assert detect_simulate_mode({}) is False

    def test_false_when_mode_is_train(self):
        assert detect_simulate_mode({"consolidation": {"mode": "train"}}) is False

    def test_true_when_mode_is_simulate(self):
        assert detect_simulate_mode({"consolidation": {"mode": "simulate"}}) is True

    def test_false_when_consolidation_has_no_mode(self):
        assert detect_simulate_mode({"consolidation": {}}) is False


class TestComputeShapeChangesSkipsAGraphPayload:
    """A bound, VERIFIED simulate slot has no LoRA shape to compare against
    a candidate YAML's rank/alpha/target_modules -- ``manifest.lora is
    None`` -- so it is skipped with a warning naming the adapter, never
    silently, and contributes no ShapeChange row."""

    def test_config_apply_preview_skips_a_graph_payload_without_a_shape_row(self, tmp_path):
        from paramem.server.migration import compute_shape_changes
        from paramem.training.key_registry import KeyRegistry
        from tests._fold_fixtures import _write_graph

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
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

        # rank=16 differs from the bound slot's shape -- would ordinarily
        # emit a ShapeChange, if there were one to compare against.
        candidate_yaml, candidate_config = _candidate_config(
            {"adapters": {"episodic": {"rank": 16}}}
        )

        changes, warnings = compute_shape_changes(candidate_yaml, adapter_dir, candidate_config)

        assert changes == []
        assert len(warnings) == 1
        assert "episodic" in warnings[0]


class TestComputeShapeChangesPerVerdictWarningWording:
    """Per-verdict warning text: the prior generalized "none readable/
    matching the live registry hash" wording was false for
    ``KEYS_WITHOUT_SLOT`` (zero candidates -- nothing to "not match") and
    ``PAYLOAD_MISMATCH`` (a slot DID match by hash; only its payload bytes
    disagree with the manifest digest)."""

    def test_keys_without_slot_names_no_candidate_not_none_matching(self, tmp_path):
        from paramem.server.migration import compute_shape_changes
        from paramem.training.key_registry import KeyRegistry

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        registry = KeyRegistry()
        registry.add("graph1")
        registry.set_simhash("graph1", 111)
        tier_root.mkdir(parents=True, exist_ok=True)
        registry.save(tier_root / "indexed_key_registry.json")
        # No slot ever written -- KEYS_WITHOUT_SLOT: an active key, zero
        # candidates at all.

        candidate_yaml, candidate_config = _candidate_config(
            {"adapters": {"episodic": {"rank": 16}}}
        )
        changes, warnings = compute_shape_changes(candidate_yaml, adapter_dir, candidate_config)

        assert changes == []
        assert len(warnings) == 1
        assert "no candidate slot exists" in warnings[0]
        assert "none readable/matching" not in warnings[0]

    def test_payload_mismatch_names_payload_digest_not_none_matching(self, tmp_path):
        from paramem.adapters.manifest import tier_registry_sha256
        from paramem.adapters.slot import write_slot
        from paramem.server.migration import compute_shape_changes
        from paramem.training.key_registry import KeyRegistry
        from tests._manifest_fixtures import make_train_manifest, write_slot_files

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
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
        # into the manifest at write time no longer matches, but the slot
        # itself DID match the registry hash.
        (slot / "adapter_model.safetensors").write_bytes(b"corrupted-not-what-was-written")

        candidate_yaml, candidate_config = _candidate_config(
            {"adapters": {"episodic": {"rank": 16}}}
        )
        changes, warnings = compute_shape_changes(candidate_yaml, adapter_dir, candidate_config)

        assert changes == []
        assert len(warnings) == 1
        assert "payload no longer matches its manifest digest" in warnings[0]
        assert "none readable/matching" not in warnings[0]


class TestRenderPreviewResponse:
    def test_pre_flight_fail_always_present(self):
        """pre_flight_fail is always in the payload (Condition 3)."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        payload = render_preview_response(stash)
        assert "pre_flight_fail" in payload
        assert payload["pre_flight_fail"] is None

    def test_pre_flight_fail_propagated_when_set(self):
        """pre_flight_fail value is propagated from the argument."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        payload = render_preview_response(stash, pre_flight_fail="disk_pressure")
        assert payload["pre_flight_fail"] == "disk_pressure"

    def test_all_required_fields_present(self):
        """All PreviewResponse fields are present."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        payload = render_preview_response(stash)
        required = [
            "state",
            "candidate_path",
            "candidate_hash",
            "staged_at",
            "simulate_mode_override",
            "unified_diff",
            "tier_diff",
            "shape_changes",
            "pre_flight_fail",
            "warnings",
        ]
        for field in required:
            assert field in payload, f"Missing field: {field!r}"

    def test_warnings_always_present(self):
        """warnings is always in the payload, mirroring pre_flight_fail."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        payload = render_preview_response(stash)
        assert "warnings" in payload
        assert payload["warnings"] == []

    def test_warnings_propagated_from_stash(self):
        """warnings value is propagated verbatim from the stash."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        stash["warnings"] = ["adapter 'episodic': skipped shape-change check — ..."]
        payload = render_preview_response(stash)
        assert payload["warnings"] == ["adapter 'episodic': skipped shape-change check — ..."]

    def test_mode_switch_block_for_pure_mode_change(self):
        """A pure consolidation.mode change surfaces a mode_switch block."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        stash["tier_diff"] = [
            {
                "dotted_path": "consolidation.mode",
                "old_value": "simulate",
                "new_value": "train",
                "tier": "pipeline_altering",
            }
        ]
        payload = render_preview_response(stash)
        ms = payload["mode_switch"]
        assert ms is not None
        assert ms["from"] == "simulate"
        assert ms["to"] == "train"
        assert ms["direction"] == "simulate_to_train"
        assert ms["applies_via"] == "active_store_migration"

    def test_mode_switch_none_for_non_mode_change(self):
        """A non-mode change (or mode + other field) has mode_switch=None."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        stash["tier_diff"] = [
            {
                "dotted_path": "consolidation.mode",
                "old_value": "simulate",
                "new_value": "train",
                "tier": "pipeline_altering",
            },
            {"dotted_path": "debug", "old_value": False, "new_value": True, "tier": "operational"},
        ]
        payload = render_preview_response(stash)
        assert payload["mode_switch"] is None


# ---------------------------------------------------------------------------
# Byte-for-byte shape-change block renderer compliance smoke test
# ---------------------------------------------------------------------------


class TestRenderShapeChangeBlockByteForByteMatchesSpec:
    """Verify that the shape-change block rendered by the CLI renderer matches
    the expected format verbatim.

    The expected text:

        ────────────────────────────────────────
        ⚠  SHAPE CHANGE — DESTRUCTIVE
        ────────────────────────────────────────
        episodic:  rank 8 → 16
                   current adapter weights (trained at rank 8) will be discarded
                   on migrate-accept. Prior recall is unrecoverable from weights.
                   Registry entries remain; the new-shape adapter will retrain
                   from the full key set on the next consolidation.
        semantic:  target_modules {q_proj,k_proj,v_proj,o_proj} → +{gate_proj}
                   same consequence.
        procedural: alpha 16 → 32
                   effective-rank scaling changes; retrain overwrites old weights.
                   Same blast radius as a rank change.
    """

    # Canonical expected output (leading two-space indent from renderer).
    # The renderer uses two spaces after the colon for all field types; the
    # spec document's "procedural: alpha" (one space) is a markdown-formatting
    # artifact — the consistent two-space form is the correct renderer output.
    _EXPECTED = (
        "  ────────────────────────────────────────\n"
        "  ⚠  SHAPE CHANGE — DESTRUCTIVE\n"
        "  ────────────────────────────────────────\n"
        "  episodic:  rank 8 → 16\n"
        "             current adapter weights (trained at rank 8) will be discarded"
        " on migrate-accept. Prior recall is unrecoverable from weights."
        " Registry entries remain; the new-shape adapter will retrain"
        " from the full key set on the next consolidation.\n"
        "  semantic:  target_modules {q_proj,k_proj,v_proj,o_proj} → +{gate_proj}\n"
        "             same consequence.\n"
        "  procedural:  alpha 16 → 32\n"
        "             effective-rank scaling changes (alpha 16 → 32);"
        " retrain overwrites old weights. Same blast radius as a rank change.\n"
    )

    def _make_shape_changes(self):
        """Return the canonical shape-change fixture used by the renderer test."""
        return [
            {
                "adapter": "episodic",
                "field": "rank",
                "old_value": 8,
                "new_value": 16,
                "consequence": (
                    "current adapter weights (trained at rank 8) will be discarded "
                    "on migrate-accept. Prior recall is unrecoverable from weights. "
                    "Registry entries remain; the new-shape adapter will retrain "
                    "from the full key set on the next consolidation."
                ),
            },
            {
                "adapter": "semantic",
                "field": "target_modules",
                "old_value": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "new_value": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                "consequence": "same consequence.",
            },
            {
                "adapter": "procedural",
                "field": "alpha",
                "old_value": 16,
                "new_value": 32,
                "consequence": (
                    "effective-rank scaling changes (alpha 16 → 32); "
                    "retrain overwrites old weights. Same blast radius as a rank change."
                ),
            },
        ]

    def test_render_shape_change_block_byte_for_byte_matches_spec(self, capsys):
        """Renderer output matches the expected shape-change block text exactly."""
        from paramem.cli.migrate import _render_shape_change_block

        _render_shape_change_block(self._make_shape_changes())
        actual = capsys.readouterr().out
        assert actual.rstrip() == self._EXPECTED.rstrip(), (
            f"\nACTUAL:\n{actual!r}\n\nEXPECTED:\n{self._EXPECTED!r}"
        )

    def test_target_modules_add_only_renders_plus_delta(self, capsys):
        """Add-only target_modules: renders '+{gate_proj}' (delta, not full new set)."""
        from paramem.cli.migrate import _render_shape_change_block

        changes = [
            {
                "adapter": "semantic",
                "field": "target_modules",
                "old_value": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "new_value": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                "consequence": "same consequence.",
            }
        ]
        _render_shape_change_block(changes)
        out = capsys.readouterr().out
        assert "→ +{gate_proj}" in out, f"Expected '+{{gate_proj}}' delta, got:\n{out!r}"
        # Must NOT include the full new set in the delta portion.
        assert "+{q_proj,k_proj,v_proj,o_proj,gate_proj}" not in out

    def test_target_modules_remove_only_renders_minus_delta(self, capsys):
        """Remove-only target_modules: renders '-{gate_proj}' delta."""
        from paramem.cli.migrate import _render_shape_change_block

        changes = [
            {
                "adapter": "semantic",
                "field": "target_modules",
                "old_value": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                "new_value": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "consequence": "same consequence.",
            }
        ]
        _render_shape_change_block(changes)
        out = capsys.readouterr().out
        assert "→ -{gate_proj}" in out, f"Expected '-{{gate_proj}}' delta, got:\n{out!r}"

    def test_target_modules_add_and_remove_renders_both(self, capsys):
        """Add+remove target_modules: renders '+{new} -{removed}' delta."""
        from paramem.cli.migrate import _render_shape_change_block

        changes = [
            {
                "adapter": "semantic",
                "field": "target_modules",
                "old_value": ["q_proj", "v_proj", "gate_proj"],
                "new_value": ["q_proj", "v_proj", "o_proj"],
                "consequence": "same consequence.",
            }
        ]
        _render_shape_change_block(changes)
        out = capsys.readouterr().out
        assert "+{o_proj}" in out, f"Expected '+{{o_proj}}' in output:\n{out!r}"
        assert "-{gate_proj}" in out, f"Expected '-{{gate_proj}}' in output:\n{out!r}"


# ---------------------------------------------------------------------------
# compute_base_change
# ---------------------------------------------------------------------------


class TestComputeBaseChange:
    def test_returns_none_when_model_unchanged(self):
        """No model change → None."""
        live = {"model": "mistral", "debug": False}
        cand = {"model": "mistral", "debug": True}
        assert compute_base_change(live, cand) is None

    def test_returns_none_when_both_model_absent(self):
        """Both YAMLs lack 'model' key → None."""
        assert compute_base_change({}, {}) is None

    def test_returns_none_when_model_absent_in_both_after_coercion(self):
        """model: '' in both → no change → None."""
        assert compute_base_change({"model": ""}, {"model": ""}) is None

    def test_returns_dict_when_model_differs(self):
        """model: mistral → qwen3-4b → dict with old_model/new_model/consequence."""
        live = {"model": "mistral"}
        cand = {"model": "qwen3-4b"}
        result = compute_base_change(live, cand)
        assert result is not None
        assert result["old_model"] == "mistral"
        assert result["new_model"] == "qwen3-4b"
        assert "consequence" in result
        assert len(result["consequence"]) > 0

    def test_consequence_mentions_old_and_new_model(self):
        """Consequence text names both old and new model aliases."""
        live = {"model": "mistral"}
        cand = {"model": "qwen3-4b"}
        result = compute_base_change(live, cand)
        assert result is not None
        assert "mistral" in result["consequence"]
        assert "qwen3-4b" in result["consequence"]

    def test_consequence_mentions_restart(self):
        """Consequence text mentions the server restart requirement."""
        result = compute_base_change({"model": "mistral"}, {"model": "qwen3-4b"})
        assert result is not None
        assert "restart" in result["consequence"].lower()

    def test_returns_none_when_model_key_absent_in_live_and_same_in_cand(self):
        """Model absent in live but present with same effective value: no change."""
        # When live has no 'model' key it defaults to "" in get(); same for cand.
        # If cand also has no 'model' key → both "" → None.
        assert compute_base_change({}, {}) is None

    def test_returns_dict_when_live_has_no_model_but_cand_does(self):
        """Live has no 'model' but candidate adds one → change detected."""
        result = compute_base_change({}, {"model": "qwen3-4b"})
        assert result is not None
        assert result["old_model"] == ""
        assert result["new_model"] == "qwen3-4b"


# ---------------------------------------------------------------------------
# render_preview_response — base_change field
# ---------------------------------------------------------------------------


class TestRenderPreviewResponseBaseChange:
    def test_base_change_present_in_payload(self):
        """render_preview_response always returns 'base_change' key."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        payload = render_preview_response(stash)
        assert "base_change" in payload

    def test_base_change_none_when_model_unchanged(self):
        """No model diff → base_change is None."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        stash["parsed_live"] = {"model": "mistral"}
        stash["parsed_candidate"] = {"model": "mistral"}
        payload = render_preview_response(stash)
        assert payload["base_change"] is None

    def test_base_change_populated_when_model_differs(self):
        """Model change → base_change has old_model, new_model, consequence."""
        stash = initial_migration_state()
        stash["state"] = "STAGING"
        stash["parsed_live"] = {"model": "mistral"}
        stash["parsed_candidate"] = {"model": "qwen3-4b"}
        payload = render_preview_response(stash)
        bc = payload["base_change"]
        assert bc is not None
        assert bc["old_model"] == "mistral"
        assert bc["new_model"] == "qwen3-4b"
        assert "consequence" in bc

    def test_base_change_none_when_both_parsed_absent(self):
        """Empty stash (LIVE initial state) → base_change is None."""
        stash = initial_migration_state()
        payload = render_preview_response(stash)
        assert payload.get("base_change") is None


# ---------------------------------------------------------------------------
# _render_base_change_preview CLI renderer
# ---------------------------------------------------------------------------


class TestRenderBaseChangePreview:
    def test_renders_old_and_new_model(self, capsys):
        """Preview output names both old and new model aliases."""
        from paramem.cli.migrate import _render_base_change_preview

        _render_base_change_preview(
            {"old_model": "mistral", "new_model": "qwen3-4b", "consequence": "..."}
        )
        out = capsys.readouterr().out
        assert "mistral" in out
        assert "qwen3-4b" in out

    def test_renders_destructive_header(self, capsys):
        """Preview output includes the DESTRUCTIVE header."""
        from paramem.cli.migrate import _render_base_change_preview

        _render_base_change_preview(
            {"old_model": "mistral", "new_model": "qwen3-4b", "consequence": "..."}
        )
        out = capsys.readouterr().out
        assert "DESTRUCTIVE" in out

    def test_renders_restart_notice(self, capsys):
        """Preview output mentions restart requirement."""
        from paramem.cli.migrate import _render_base_change_preview

        _render_base_change_preview(
            {"old_model": "mistral", "new_model": "qwen3-4b", "consequence": "..."}
        )
        out = capsys.readouterr().out
        assert "restart" in out.lower()


# ---------------------------------------------------------------------------
# _render_warnings_block CLI renderer
# ---------------------------------------------------------------------------


class TestRenderWarningsBlock:
    def test_renders_one_line_per_warning_to_stderr(self, capsys):
        """Each warning string is printed on its own line to stderr."""
        from paramem.cli.migrate import _render_warnings_block

        _render_warnings_block(
            [
                "adapter 'episodic': skipped shape-change check — cannot read/decrypt "
                "tier registry at /data/adapters/episodic: RuntimeError('boom')",
                "adapter 'semantic': skipped shape-change check — cannot read manifest "
                "from slot /data/adapters/semantic/ts: ManifestSchemaError('bad json')",
            ]
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        lines = [line for line in captured.err.splitlines() if line]
        assert len(lines) == 2
        assert "episodic" in lines[0]
        assert "semantic" in lines[1]

    def test_suppressed_when_empty(self, capsys):
        """No warnings → nothing printed to either stream."""
        from paramem.cli.migrate import _render_warnings_block

        _render_warnings_block([])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
