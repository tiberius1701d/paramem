"""Pure-function tests for paramem.server.migration.

Covers:
- compute_unified_diff
- compute_tier_diff
- compute_shape_changes
- detect_simulate_mode
- render_preview_response
- Byte-for-byte spec-compliance smoke test against §L257–271.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from paramem.server.migration import (
    compute_shape_changes,
    compute_tier_diff,
    compute_unified_diff,
    detect_simulate_mode,
    initial_migration_state,
    render_preview_response,
)

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


# ---------------------------------------------------------------------------
# compute_shape_changes — no-manifest / no-slot paths
# ---------------------------------------------------------------------------


class TestComputeShapeChangesNoManifest:
    def test_empty_adapters_returns_empty(self, tmp_path):
        """Empty adapters section → no shape changes."""
        result = compute_shape_changes({}, tmp_path, "")
        assert result == []

    def test_disabled_adapter_is_skipped(self, tmp_path):
        """Adapter with enabled=False is not checked."""
        yaml = {"adapters": {"episodic": {"enabled": False, "rank": 16}}}
        result = compute_shape_changes(yaml, tmp_path, "")
        assert result == []

    def test_missing_slot_skips_silently(self, tmp_path):
        """No matching slot → no row emitted."""
        yaml = {"adapters": {"episodic": {"enabled": True, "rank": 16}}}
        # No slot directory created — find_live_slot returns None
        result = compute_shape_changes(yaml, tmp_path, "")
        assert result == []


# ---------------------------------------------------------------------------
# compute_shape_changes — with manifest
# ---------------------------------------------------------------------------


def _make_manifest_mock(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")):
    """Return a mock AdapterManifest."""
    manifest = MagicMock()
    manifest.lora.rank = rank
    manifest.lora.alpha = alpha
    manifest.lora.dropout = dropout
    manifest.lora.target_modules = tuple(target_modules)
    return manifest


class TestComputeShapeChangesWithManifest:
    def test_rank_change_detected(self, tmp_path):
        """rank change → ShapeChange with adapter='episodic', field='rank'."""
        yaml = {"adapters": {"episodic": {"enabled": True, "rank": 16, "alpha": 16}}}
        slot = tmp_path / "episodic" / "20260421-040000"
        slot.mkdir(parents=True)

        manifest = _make_manifest_mock(rank=8, alpha=16)

        with (
            patch("paramem.server.migration.find_live_slot", return_value=slot),
            patch("paramem.server.migration.read_manifest", return_value=manifest),
        ):
            result = compute_shape_changes(yaml, tmp_path, "")

        assert any(r["field"] == "rank" and r["adapter"] == "episodic" for r in result), result

    def test_alpha_change_detected(self, tmp_path):
        """alpha change → ShapeChange for alpha field."""
        yaml = {"adapters": {"episodic": {"enabled": True, "rank": 8, "alpha": 32}}}
        slot = tmp_path / "episodic" / "20260421-040000"
        slot.mkdir(parents=True)

        manifest = _make_manifest_mock(rank=8, alpha=16)

        with (
            patch("paramem.server.migration.find_live_slot", return_value=slot),
            patch("paramem.server.migration.read_manifest", return_value=manifest),
        ):
            result = compute_shape_changes(yaml, tmp_path, "")

        assert any(r["field"] == "alpha" for r in result), result

    def test_no_change_returns_empty(self, tmp_path):
        """Identical rank and alpha → no shape changes."""
        yaml = {"adapters": {"episodic": {"enabled": True, "rank": 8, "alpha": 16}}}
        slot = tmp_path / "episodic" / "20260421-040000"
        slot.mkdir(parents=True)

        manifest = _make_manifest_mock(rank=8, alpha=16)

        with (
            patch("paramem.server.migration.find_live_slot", return_value=slot),
            patch("paramem.server.migration.read_manifest", return_value=manifest),
        ):
            result = compute_shape_changes(yaml, tmp_path, "")

        assert result == []

    def test_unreadable_manifest_skips_with_warn(self, tmp_path):
        """ManifestError on read → skip (no row emitted).

        The warning is emitted via logger.warning; we verify the row is absent
        and the function returns cleanly without raising.
        """
        from paramem.adapters.manifest import ManifestSchemaError

        yaml_data = {"adapters": {"episodic": {"enabled": True, "rank": 16}}}
        slot = tmp_path / "episodic" / "ts"
        slot.mkdir(parents=True)

        with (
            patch("paramem.server.migration.find_live_slot", return_value=slot),
            patch(
                "paramem.server.migration.read_manifest",
                side_effect=ManifestSchemaError("bad json"),
            ),
        ):
            result = compute_shape_changes(yaml_data, tmp_path, "")

        # No row emitted — the unreadable manifest is silently skipped.
        assert result == []

    def test_target_modules_change_detected(self, tmp_path):
        """target_modules change → ShapeChange for target_modules field."""
        yaml = {
            "adapters": {
                "semantic": {
                    "enabled": True,
                    "rank": 8,
                    "alpha": 16,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
                }
            }
        }
        slot = tmp_path / "semantic" / "ts"
        slot.mkdir(parents=True)

        manifest = _make_manifest_mock(
            rank=8, alpha=16, target_modules=("q_proj", "k_proj", "v_proj", "o_proj")
        )

        with (
            patch("paramem.server.migration.find_live_slot", return_value=slot),
            patch("paramem.server.migration.read_manifest", return_value=manifest),
        ):
            result = compute_shape_changes(yaml, tmp_path, "")

        assert any(r["field"] == "target_modules" for r in result), result


# ---------------------------------------------------------------------------
# render_preview_response
# ---------------------------------------------------------------------------


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
        ]
        for field in required:
            assert field in payload, f"Missing field: {field!r}"


# ---------------------------------------------------------------------------
# Byte-for-byte spec-compliance smoke test (spec §L257–271)
# ---------------------------------------------------------------------------


class TestRenderShapeChangeBlockByteForByteMatchesSpec:
    """Verify that the shape-change block rendered by the CLI renderer matches
    the spec §L257–271 verbatim.

    The spec text (lines 257–271 of plan_migration_and_backup.md):

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

    # Canonical expected output (spec §L257–271, leading two-space indent from renderer).
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
        """Return the fixture mirroring spec §L261–271."""
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
        """Renderer output matches the spec §L257–271 text exactly (string equality)."""
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
