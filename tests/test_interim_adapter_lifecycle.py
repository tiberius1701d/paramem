"""Unit tests for interim-adapter lifecycle helpers.

Covers:
  - create_interim_adapter is idempotent for the same stamp.
  - current_interim_stamp returns a correctly formatted timestamp.
  - compute_schedule_period_seconds parses all supported schedule grammars.
  - current_interim_stamp(refresh_cadence) floors to the correct cadence boundary.
    refresh_cadence IS the sub-interval directly — no division by max_interim_count.

No GPU required — all PEFT and model interactions use stub objects.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.adapters.slot import payload_filename
from paramem.memory.interim_adapter import (
    INTERIM_NAME_PREFIX,
    MAIN_TIERS,
    create_interim_adapter,
    current_interim_stamp,
    detect_legacy_adapter_layout,
    has_unbound_payload,
    interim_dir_for_name,
    interim_stamp_from_name,
    interim_tiers_newest_first,
    iter_interim_dirs,
    iter_tier_roots,
)
from paramem.server.schedule_grammar import compute_schedule_period_seconds
from paramem.training.donor import DONOR_STORE_PREFIX, iter_donor_stores
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_peft_model(*adapter_names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel.

    ``spec=PeftModel`` so ``isinstance(model, PeftModel)`` holds — the PEFT half
    of ``unload_interim_adapters`` is gated on exactly that check (the disk venue
    passes a bare base model and must skip it).

    peft_config is a real dict keyed by adapter_names so membership tests,
    iteration, and deletion all work correctly without touching torch.
    delete_adapter removes the key from the dict, mirroring PEFT behaviour.
    """
    from peft import PeftModel

    model = MagicMock(spec=PeftModel)
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


# ---------------------------------------------------------------------------
# Test 0 — current_interim_stamp format
# ---------------------------------------------------------------------------


class TestCurrentInterimStamp:
    def test_minted_stamp_round_trips_through_the_parser(self) -> None:
        """The mint and the parser share INTERIM_STAMP_FORMAT.

        Asserted as a round trip rather than against a copy of the shape:
        re-declaring ``\\d{8}T\\d{4}`` here would be a second renderer of the
        format the module is the single source of.
        """
        stamp = current_interim_stamp("every 1h")
        assert interim_stamp_from_name(f"{INTERIM_NAME_PREFIX}{stamp}") == stamp


# ---------------------------------------------------------------------------
# Test 0b — interim_stamp_from_name: THE interim adapter name parser
# ---------------------------------------------------------------------------


class TestInterimStampFromName:
    """Contract for the single interim-name parser.

    Cases carried over from the retired ``router._interim_sort_key`` tests,
    plus the two the previous ``\\d{8}T\\d{4}`` pattern accepted but that are
    not real stamps.
    """

    def test_valid_name_returns_stamp(self) -> None:
        assert interim_stamp_from_name("episodic_interim_20260417T0000") == "20260417T0000"

    @pytest.mark.parametrize("name", ["episodic", "semantic", "procedural", ""])
    def test_non_interim_names_return_none(self, name: str) -> None:
        assert interim_stamp_from_name(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "episodic_interim_",
            "episodic_interim_today",
            "episodic_interim_2026-04-17",
            "episodic_interim_20260417T0000_partial",
        ],
    )
    def test_malformed_stamps_return_none(self, name: str) -> None:
        assert interim_stamp_from_name(name) is None

    @pytest.mark.parametrize(
        "name",
        [
            "episodic_interim_99999999T9999",  # shape-valid, not a real datetime
            "episodic_interim_20260417T2400",  # hour 24 does not exist
            "episodic_interim_2026041T000",  # short fields; strptime alone accepts this
        ],
    )
    def test_shape_valid_but_impossible_stamps_return_none(self, name: str) -> None:
        assert interim_stamp_from_name(name) is None

    def test_dir_for_name_raises_on_malformed(self, tmp_path: Path) -> None:
        """Path construction must not silently produce ``interim_today/``."""
        with pytest.raises(ValueError, match="Not an interim adapter name"):
            interim_dir_for_name(tmp_path, "episodic_interim_today")


# ---------------------------------------------------------------------------
# Test 0c — interim_tiers_newest_first: THE interim-tier enumeration for
# probe ordering (router) and dedup scope (interim consolidation fold).
# Moved from tests/test_router.py — the function moved from
# paramem.server.router to this module (its natural home, alongside
# interim_stamp_from_name).  Newest-first ordering and main-tier filtering
# over a REAL store are covered more strongly by
# tests/test_router.py::TestPersonalTierOrder::test_interim_adapters_sort_newest_first,
# which drives the real router; only the two branches that harness cannot
# reach live here.
# ---------------------------------------------------------------------------


class _FakeStore:
    def __init__(self, tiers):
        self._tiers = tiers

    def tiers_with_registry(self):
        return list(self._tiers)


class TestInterimTiersNewestFirst:
    def test_none_store_yields_empty(self) -> None:
        """No ``MemoryStore`` constructed (e.g. cloud-only mode) — the branch
        that makes interim slots silently unreachable."""
        assert interim_tiers_newest_first(None) == []

    def test_malformed_interim_names_are_filtered_out(self) -> None:
        """A slot name that carries the prefix but no valid stamp is dropped
        rather than sorted as ``""``."""
        store = _FakeStore(["episodic_interim_today", "episodic_interim_99999999T9999"])
        assert interim_tiers_newest_first(store) == []


# ---------------------------------------------------------------------------
# Test 3 — create_interim_adapter is idempotent
# ---------------------------------------------------------------------------


class TestCreateInterimAdapterIdempotent:
    def test_different_stamp_creates_new_adapter(self) -> None:
        """create_interim_adapter creates a distinct adapter per stamp."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        adapter_config = MagicMock()

        # Simulate create_adapter returning a new model each time and updating
        # peft_config so the second call sees the first adapter.
        def _side_effect(m, cfg, *, adapter_name: str):  # noqa: ANN001
            m.peft_config[adapter_name] = MagicMock()
            return m

        with patch(
            "paramem.memory.interim_adapter.create_adapter",
            side_effect=_side_effect,
        ) as mock_create:
            create_interim_adapter(model, adapter_config, "20260417T0000")
            create_interim_adapter(model, adapter_config, "20260418T0000")

        assert mock_create.call_count == 2
        call_names = [c.kwargs["adapter_name"] for c in mock_create.call_args_list]
        assert "episodic_interim_20260417T0000" in call_names
        assert "episodic_interim_20260418T0000" in call_names


# ---------------------------------------------------------------------------
# Test 3b — ensure_adapter_matching (the shared warm-init config-mismatch guard)
# ---------------------------------------------------------------------------


class TestEnsureAdapterMatching:
    """``paramem.models.loader.ensure_adapter_matching`` — the single
    config-mismatch guard called from both warm-init preambles (main-tier
    fold, interim-slot mint).  Reuses ``_make_stub_peft_model`` (the same
    ``spec=PeftModel`` + real-dict ``peft_config`` double used above for
    ``tier_backup_scope``/``create_interim_adapter``).
    """

    @staticmethod
    def _adapter_config():
        from paramem.utils.config import AdapterConfig

        return AdapterConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])

    def test_alpha_mismatch_recreates_cold(self) -> None:
        """A lora_alpha change alone is also a mismatch."""
        model = _make_stub_peft_model("episodic")
        adapter_config = self._adapter_config()
        resident = model.peft_config["episodic"]
        resident.r = adapter_config.rank
        resident.lora_alpha = 32  # target is alpha=16
        resident.target_modules = list(adapter_config.target_modules)

        with patch("paramem.models.loader.create_adapter") as mock_create:
            from paramem.models.loader import ensure_adapter_matching

            ensure_adapter_matching(model, adapter_config, "episodic")

        model.delete_adapter.assert_called_once_with("episodic")
        mock_create.assert_called_once_with(model, adapter_config, "episodic")

    def test_target_modules_mismatch_recreates_cold(self) -> None:
        """A target_modules change alone is also a mismatch (order-insensitive)."""
        model = _make_stub_peft_model("episodic")
        adapter_config = self._adapter_config()
        resident = model.peft_config["episodic"]
        resident.r = adapter_config.rank
        resident.lora_alpha = adapter_config.alpha
        resident.target_modules = ["q_proj"]  # target is ["q_proj", "v_proj"]

        with patch("paramem.models.loader.create_adapter") as mock_create:
            from paramem.models.loader import ensure_adapter_matching

            ensure_adapter_matching(model, adapter_config, "episodic")

        model.delete_adapter.assert_called_once_with("episodic")
        mock_create.assert_called_once_with(model, adapter_config, "episodic")

    def test_mismatch_warning_names_the_field(self, caplog: pytest.LogCaptureFixture) -> None:
        """The recreate-on-mismatch path logs a warning naming the adapter
        and the mismatched field."""
        model = _make_stub_peft_model("episodic")
        adapter_config = self._adapter_config()
        resident = model.peft_config["episodic"]
        resident.r = 4
        resident.lora_alpha = adapter_config.alpha
        resident.target_modules = list(adapter_config.target_modules)

        caplog.set_level(logging.WARNING, logger="paramem.models.loader")
        with patch("paramem.models.loader.create_adapter"):
            from paramem.models.loader import ensure_adapter_matching

            ensure_adapter_matching(model, adapter_config, "episodic")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a mismatch warning"
        message = warnings[0].getMessage()
        assert "episodic" in message
        assert "r:" in message


# ---------------------------------------------------------------------------
# Test 3b — detach_adapters (paramem.models.loader)
# ---------------------------------------------------------------------------


class TestDetachAdapters:
    def test_survivor_preference_prefers_episodic(self) -> None:
        """Among multiple non-deleted resident adapters, episodic is picked
        as the survivor ahead of semantic/procedural."""
        from paramem.models.loader import detach_adapters

        model = _make_stub_peft_model(
            "episodic", "semantic", "procedural", "episodic_interim_20260417T0000"
        )

        deleted = detach_adapters(model, ["episodic_interim_20260417T0000"])

        assert deleted == ["episodic_interim_20260417T0000"]
        model.set_adapter.assert_called_once_with("episodic")
        assert "episodic_interim_20260417T0000" not in model.peft_config

    def test_survivor_falls_through_when_episodic_itself_is_deleted(self) -> None:
        """When "episodic" is one of the deleted names, the next main tier
        ("semantic") becomes the survivor."""
        from paramem.models.loader import detach_adapters

        model = _make_stub_peft_model("episodic", "semantic", "procedural")

        deleted = detach_adapters(model, ["episodic"])

        assert deleted == ["episodic"]
        model.set_adapter.assert_called_once_with("semantic")

    def test_no_survivor_leaves_peft_config_empty_without_raising(self) -> None:
        """Deleting every resident adapter leaves peft_config empty; no
        switch is attempted and nothing raises — the caller is emptying
        peft_config deliberately and owns the restore."""
        from paramem.models.loader import detach_adapters

        model = _make_stub_peft_model("episodic_interim_a", "episodic_interim_b")

        deleted = detach_adapters(model, ["episodic_interim_a", "episodic_interim_b"])

        assert deleted == ["episodic_interim_a", "episodic_interim_b"]
        model.set_adapter.assert_not_called()
        assert model.peft_config == {}

    def test_absent_name_is_a_no_op(self) -> None:
        """A name not resident in peft_config is silently skipped — never
        raises, and no delete/switch call is made for it."""
        from paramem.models.loader import detach_adapters

        model = _make_stub_peft_model("episodic", "semantic", "procedural")

        deleted = detach_adapters(model, ["episodic_interim_ghost"])

        assert deleted == []
        model.set_adapter.assert_not_called()
        model.delete_adapter.assert_not_called()
        assert set(model.peft_config.keys()) == {"episodic", "semantic", "procedural"}


# ---------------------------------------------------------------------------
# Test 3c — unload_interim_adapters (paramem.memory.interim_adapter)
# ---------------------------------------------------------------------------


def _seed_interim_dir(adapter_dir: Path, stamp: str) -> Path:
    """Write a minimal on-disk interim slot dir with a stub payload file —
    ``reap_tier_artifacts`` removes the whole directory by name shape, so the
    contents only need to exist, never match a real weight/graph schema."""
    path = adapter_dir / "episodic" / f"interim_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "stub.txt").write_text("x")
    return path


class TestUnloadInterimAdaptersBothVenueReap:
    """Both fold venues call ``unload_interim_adapters`` and both get the
    identical on-disk reap — the PEFT half is the only part that varies."""

    def test_weights_venue_reaps_disk_and_peft_and_returns_the_deleted_names(
        self, tmp_path: Path
    ) -> None:
        from paramem.memory.interim_adapter import unload_interim_adapters

        adapter_dir = tmp_path / "adapters"
        dir_a = _seed_interim_dir(adapter_dir, "20260101T0000")
        dir_b = _seed_interim_dir(adapter_dir, "20260102T0000")
        model = _make_stub_peft_model(
            "episodic",
            "episodic_interim_20260101T0000",
            "episodic_interim_20260102T0000",
        )

        deleted = unload_interim_adapters(model, adapter_dir)

        assert deleted == ["episodic_interim_20260101T0000", "episodic_interim_20260102T0000"]
        assert "episodic_interim_20260101T0000" not in model.peft_config
        assert "episodic_interim_20260102T0000" not in model.peft_config
        assert "episodic" in model.peft_config
        assert not dir_a.exists()
        assert not dir_b.exists()


class TestUnloadInterimAdaptersPeftHalfSkip:
    """``model=None`` is the only input that skips the PEFT half entirely —
    never touches ``peft_config``/``delete_adapter``. Under wrap-once every
    non-``None`` model is a resident ``PeftModel``; there is no bare
    base-model case left to skip for."""

    def test_none_model_never_raises_and_returns_no_names(self, tmp_path: Path) -> None:
        from paramem.memory.interim_adapter import unload_interim_adapters

        adapter_dir = tmp_path / "adapters"
        _seed_interim_dir(adapter_dir, "20260101T0000")

        deleted = unload_interim_adapters(None, adapter_dir)

        assert deleted == []


class TestUnloadInterimAdaptersSwitchBeforeDeleteDeterminism:
    """The active adapter moves onto a resident survivor before any interim
    adapter is deleted — pinned here at ``unload_interim_adapters``'s own
    call boundary (its interim-name filter feeding ``detach_adapters``),
    distinct from ``TestDetachAdapters`` above, which pins the survivor
    selection logic itself."""

    def test_set_adapter_precedes_every_delete_adapter_call(self, tmp_path: Path) -> None:
        from paramem.memory.interim_adapter import unload_interim_adapters

        adapter_dir = tmp_path / "adapters"
        _seed_interim_dir(adapter_dir, "20260101T0000")
        _seed_interim_dir(adapter_dir, "20260102T0000")
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "episodic_interim_20260101T0000",
            "episodic_interim_20260102T0000",
        )

        unload_interim_adapters(model, adapter_dir)

        call_names = [c[0] for c in model.mock_calls]
        switch_index = call_names.index("set_adapter")
        delete_indices = [i for i, name in enumerate(call_names) if name == "delete_adapter"]
        assert delete_indices, "delete_adapter must have been called at least once"
        assert switch_index < min(delete_indices)
        model.set_adapter.assert_called_once_with("episodic")

    def test_post_reap_active_adapter_is_deterministic_regardless_of_prior_active(
        self, tmp_path: Path
    ) -> None:
        """The switch runs unconditionally whenever a survivor exists, not
        only when the pre-call active adapter happens to be an interim one —
        so two calls starting from different active adapters converge on the
        identical survivor."""
        from paramem.memory.interim_adapter import unload_interim_adapters

        adapter_dir_a = tmp_path / "adapters_a"
        _seed_interim_dir(adapter_dir_a, "20260101T0000")
        model_a = _make_stub_peft_model("episodic", "semantic", "episodic_interim_20260101T0000")
        model_a.active_adapter = "episodic_interim_20260101T0000"

        adapter_dir_b = tmp_path / "adapters_b"
        _seed_interim_dir(adapter_dir_b, "20260101T0000")
        model_b = _make_stub_peft_model("episodic", "semantic", "episodic_interim_20260101T0000")
        model_b.active_adapter = "semantic"

        unload_interim_adapters(model_a, adapter_dir_a)
        unload_interim_adapters(model_b, adapter_dir_b)

        model_a.set_adapter.assert_called_once_with("episodic")
        model_b.set_adapter.assert_called_once_with("episodic")


# ---------------------------------------------------------------------------
# Test 6 — compute_schedule_period_seconds
# ---------------------------------------------------------------------------


class TestComputeSchedulePeriodSeconds:
    def test_every_2h_returns_7200(self) -> None:
        """'every 2h' → 2 × 3600 = 7200 seconds."""
        assert compute_schedule_period_seconds("every 2h") == 7200

    def test_every_1h_returns_3600(self) -> None:
        """'every 1h' → 3600 seconds."""
        assert compute_schedule_period_seconds("every 1h") == 3600

    def test_every_30m_returns_1800(self) -> None:
        """'every 30m' → 30 × 60 = 1800 seconds."""
        assert compute_schedule_period_seconds("every 30m") == 1800

    def test_every_10m_returns_600(self) -> None:
        """'every 10m' → 10 × 60 = 600 seconds."""
        assert compute_schedule_period_seconds("every 10m") == 600

    def test_daily_03_00_returns_86400(self) -> None:
        """'03:00' (daily) → 86400 seconds."""
        assert compute_schedule_period_seconds("03:00") == 86400

    def test_daily_midnight_returns_86400(self) -> None:
        """'00:00' (daily at midnight) → 86400 seconds."""
        assert compute_schedule_period_seconds("00:00") == 86400

    def test_empty_string_returns_none(self) -> None:
        """Empty schedule → None (manual only)."""
        assert compute_schedule_period_seconds("") is None

    def test_off_returns_none(self) -> None:
        """'off' → None."""
        assert compute_schedule_period_seconds("off") is None

    def test_disabled_returns_none(self) -> None:
        """'disabled' → None."""
        assert compute_schedule_period_seconds("disabled") is None

    def test_none_string_returns_none(self) -> None:
        """'none' → None."""
        assert compute_schedule_period_seconds("none") is None

    def test_invalid_raises_value_error(self) -> None:
        """An unrecognised schedule string raises ValueError for truly unknown values."""
        with pytest.raises(ValueError, match="Unrecognised"):
            compute_schedule_period_seconds("biweekly")

    def test_weekly_returns_604800(self) -> None:
        """'weekly' → 604800 seconds (7 × 86400)."""
        assert compute_schedule_period_seconds("weekly") == 604800

    def test_weekly_case_insensitive(self) -> None:
        """'Weekly' and 'WEEKLY' are accepted (case-insensitive)."""
        assert compute_schedule_period_seconds("Weekly") == 604800
        assert compute_schedule_period_seconds("WEEKLY") == 604800

    def test_daily_returns_86400(self) -> None:
        """'daily' → 86400 seconds."""
        assert compute_schedule_period_seconds("daily") == 86400

    def test_daily_case_insensitive(self) -> None:
        """'Daily' and 'DAILY' are accepted (case-insensitive)."""
        assert compute_schedule_period_seconds("Daily") == 86400
        assert compute_schedule_period_seconds("DAILY") == 86400

    def test_case_insensitive_every_H(self) -> None:
        """Grammar is case-insensitive for the unit character."""
        assert compute_schedule_period_seconds("every 2H") == 7200

    def test_case_insensitive_every_M(self) -> None:
        """Grammar is case-insensitive for the unit character."""
        assert compute_schedule_period_seconds("every 30M") == 1800


# ---------------------------------------------------------------------------
# Test 7 — current_interim_stamp with refresh_cadence
# ---------------------------------------------------------------------------


class TestCurrentInterimStampWithCadence:
    """Floor-to-cadence logic for current_interim_stamp(refresh_cadence).

    refresh_cadence IS the sub-interval directly — no division by
    max_interim_count. Full consolidation period is derived elsewhere
    (ConsolidationScheduleConfig.consolidation_period_string).
    """

    def test_every_30m_floors_to_30min_boundary(self) -> None:
        """refresh_cadence='every 30m' → boundary every 30 minutes from midnight.

        At 14:47 → seconds_since_midnight = 14*3600+47*60 = 53220.
        floored = (53220 // 1800) * 1800 = 29*1800 = 52200 = 14:30.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        stamp = current_interim_stamp("every 30m", _now=now)
        assert stamp == "20260418T1430"

    def test_every_30m_floor_at_exact_boundary(self) -> None:
        """At exactly a 30-min boundary the stamp equals that boundary."""
        now = datetime(2026, 4, 18, 14, 30, 0)
        stamp = current_interim_stamp("every 30m", _now=now)
        assert stamp == "20260418T1430"

    def test_every_4h_floors_to_4h_boundary(self) -> None:
        """refresh_cadence='every 4h' → boundary every 4h from midnight.

        At 09:15 → seconds_since_midnight = 33300.
        floored = (33300 // 14400) * 14400 = 28800 = 08:00.
        """
        now = datetime(2026, 4, 18, 9, 15, 0)
        stamp = current_interim_stamp("every 4h", _now=now)
        assert stamp == "20260418T0800"

    def test_daily_hhmm_floors_to_day_boundary(self) -> None:
        """refresh_cadence='03:00' (daily) → 86400s boundary from midnight.

        At 09:15 → seconds_since_midnight = 33300.
        floored = (33300 // 86400) * 86400 = 0 = 00:00.
        """
        now = datetime(2026, 4, 18, 9, 15, 0)
        stamp = current_interim_stamp("03:00", _now=now)
        assert stamp == "20260418T0000"

    def test_off_variant_cadence_floors_to_nearest_hour(self) -> None:
        """Explicit off-variant ("off"/"disabled"/"none") falls back to 1-h boundaries.

        Callers that want to skip stamping entirely should take the
        queue-branch earlier rather than rely on this fallback.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        for cadence in ("off", "disabled", "none"):
            stamp = current_interim_stamp(cadence, _now=now)
            # sub_interval=3600; seconds_since_midnight = 53220
            # floored = (53220 // 3600) * 3600 = 50400 = 14:00
            assert stamp == "20260418T1400", (
                f"off-variant {cadence!r} should floor to 14:00, got {stamp}"
            )

    def test_every_2h_floors_to_2h_boundary(self) -> None:
        """refresh_cadence='every 2h' → boundary every 2h from midnight.

        At 14:47 → seconds_since_midnight = 53220.
        floored = (53220 // 7200) * 7200 = 7*7200 = 50400 = 14:00.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        stamp = current_interim_stamp("every 2h", _now=now)
        assert stamp == "20260418T1400"


# ---------------------------------------------------------------------------
# detect_legacy_adapter_layout
# ---------------------------------------------------------------------------


class TestDetectLegacyAdapterLayout:
    """Flags top-level ``episodic_interim_<stamp>`` dirs (pre-2026-05-14 layout).

    The boot lifespan refuses to start on a non-empty result — the current
    code paths scan ``adapter_dir/episodic/interim_*`` (via
    :func:`iter_interim_dirs`), so a legacy top-level
    ``adapter_dir/episodic_interim_*`` dir is invisible to them and would
    silently degrade the server (see ``app.py`` boot lifespan).
    """

    def test_clean_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """A clean adapter_dir (no legacy dirs) yields an empty list."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()
        (adapter_dir / "episodic").mkdir()
        assert detect_legacy_adapter_layout(adapter_dir) == []

    def test_legacy_top_level_dirs_returned_sorted(self, tmp_path: Path) -> None:
        """Top-level episodic_interim_* dirs are returned, sorted."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()
        later = adapter_dir / f"{INTERIM_NAME_PREFIX}20260420T0000"
        earlier = adapter_dir / f"{INTERIM_NAME_PREFIX}20260101T0000"
        later.mkdir()
        earlier.mkdir()

        result = detect_legacy_adapter_layout(adapter_dir)

        assert result == sorted([earlier, later])

    def test_non_dir_adapter_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """adapter_dir that is not a directory (missing, or a file) yields []."""
        missing = tmp_path / "does_not_exist"
        assert detect_legacy_adapter_layout(missing) == []

        as_file = tmp_path / "adapters_as_file"
        as_file.write_text("not a directory")
        assert detect_legacy_adapter_layout(as_file) == []

    def test_matching_files_not_dirs_are_not_returned(self, tmp_path: Path) -> None:
        """A file (not a dir) matching the glob pattern is excluded."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()
        (adapter_dir / f"{INTERIM_NAME_PREFIX}20260420T0000").write_text("not a directory")

        assert detect_legacy_adapter_layout(adapter_dir) == []


# ---------------------------------------------------------------------------
# iter_tier_roots — THE shared tier-root enumeration
# ---------------------------------------------------------------------------


def _write_registry(path: Path, keys: list[str]) -> None:
    """Write a minimal indexed_key_registry.json with a simhash entry per key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = KeyRegistry()
    for key in keys:
        reg.add(key)
        reg.set_simhash(key, 1)
    path.write_bytes(reg.save_bytes())


def _build_sample_adapter_tree(
    adapter_dir: Path, *, main_tiers: tuple[str, ...] = MAIN_TIERS
) -> dict[str, Path]:
    """Build a temp adapter tree with main tiers, two interim slots, and
    one donor store.

    Returns a single-entry dict, ``{"donor_dir": <path>}``, giving the
    donor store's root path — the only built path any caller needs back
    (the main and interim tier roots are re-derivable from *adapter_dir*
    and :data:`MAIN_TIERS`/the known stamps).

    Args:
        adapter_dir: Root to build the tree under.
        main_tiers: Which of :data:`MAIN_TIERS` to actually write a
            registry for. Defaults to all three; a caller testing the
            absent-main-dir contract passes a subset so the omitted
            tier's directory never exists on disk.

    Shape (default *main_tiers*)::

        adapter_dir/
          episodic/indexed_key_registry.json
          episodic/interim_20260417T0000/indexed_key_registry.json
          episodic/interim_20260418T0000/indexed_key_registry.json
          semantic/indexed_key_registry.json
          procedural/indexed_key_registry.json
          donor-x-b0011223/20260101T0000/meta.json
    """
    for tier in main_tiers:
        _write_registry(adapter_dir / tier / "indexed_key_registry.json", [f"{tier}_key"])

    for stamp in ("20260417T0000", "20260418T0000"):
        interim_dir = adapter_dir / "episodic" / f"interim_{stamp}"
        _write_registry(interim_dir / "indexed_key_registry.json", [f"interim_{stamp}_key"])

    donor_dir = adapter_dir / f"{DONOR_STORE_PREFIX}x-b0011223"
    donor_slot = donor_dir / "20260101T0000"
    donor_slot.mkdir(parents=True)
    (donor_slot / "meta.json").write_text("{}", encoding="utf-8")

    return {"donor_dir": donor_dir}


class TestIterTierRoots:
    def test_main_tiers_yielded_even_when_absent(self, tmp_path: Path) -> None:
        """A fresh, empty adapter_dir still yields exactly the three main
        tiers, in MAIN_TIERS order — main tiers are named literally, not
        discovered."""
        result = list(iter_tier_roots(tmp_path))

        assert result == [(tier, tmp_path / tier) for tier in MAIN_TIERS]

    def test_donor_store_is_never_a_tier(self, tmp_path: Path) -> None:
        """A donor store is yielded by iter_donor_stores and by no memory-tier
        walk — donors are siblings of the tiers, not tiers themselves."""
        built = _build_sample_adapter_tree(tmp_path)
        donor_name = built["donor_dir"].name

        donor_names = {name for name, _ in iter_donor_stores(tmp_path)}
        tier_names = {name for name, _ in iter_tier_roots(tmp_path)}

        assert donor_name in donor_names
        assert donor_name not in tier_names


class TestIterInterimDirsPayloadOnlyVenueParity:
    """``payload_only=True`` is venue-blind: a candidate slot is "a
    subdirectory carrying its own meta.json" (``count_slot_candidates``),
    checked identically for a train payload or a simulate payload -- the
    schedule gate never asks which venue it is in."""

    def _write_slot(self, adapter_dir: Path, stamp: str, *, payload_filename: str) -> None:
        family = adapter_dir / "episodic" / f"interim_{stamp}"
        slot = family / f"{stamp}-slot"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "meta.json").write_text("{}")
        (slot / payload_filename).write_bytes(b"")

    def test_interim_payload_count_is_the_same_in_both_venues(self, tmp_path: Path) -> None:
        train_dir = tmp_path / "train"
        simulate_dir = tmp_path / "simulate"
        payload_stamps = ["20260101T0000", "20260102T0000", "20260103T0000"]
        for stamp in payload_stamps:
            self._write_slot(train_dir, stamp, payload_filename="adapter_model.safetensors")
            self._write_slot(simulate_dir, stamp, payload_filename="graph.json")

        # An extra empty-shell family (crashed between mkdir and the payload
        # write) in BOTH trees, excluded identically by either venue.
        empty_stamp = "20260104T0000"
        (train_dir / "episodic" / f"interim_{empty_stamp}").mkdir(parents=True)
        (simulate_dir / "episodic" / f"interim_{empty_stamp}").mkdir(parents=True)

        train_payload_bearing = list(iter_interim_dirs(train_dir, payload_only=True))
        simulate_payload_bearing = list(iter_interim_dirs(simulate_dir, payload_only=True))

        assert len(train_payload_bearing) == len(simulate_payload_bearing) == len(payload_stamps)
        assert {name for name, _ in train_payload_bearing} == {
            name for name, _ in simulate_payload_bearing
        }


class TestHasUnboundPayloadSkipsDotDirs:
    """An interrupted write's mid-write staging directory (``.pending/<ts>/<payload>``)
    is scratch, not a torn slot -- ``has_unbound_payload`` must never read it
    as unbound debris (a false ``torn_slot`` incident at the next backup
    capture)."""

    def test_payload_only_under_pending_reports_no_unbound_payload(self, tmp_path: Path) -> None:
        tier_root = tmp_path / "episodic"
        pending_slot = tier_root / ".pending" / "20260101-000000"
        pending_slot.mkdir(parents=True)
        (pending_slot / "graph.json").write_bytes(b"{}")
        (pending_slot / "meta.json").write_bytes(b"{}")

        assert has_unbound_payload(tier_root) is False

    def test_payload_outside_pending_still_reports_unbound_payload(self, tmp_path: Path) -> None:
        """Control: a payload file NOT under a dot-prefixed directory is
        still detected -- the fix narrows the predicate, it does not
        disable it."""
        tier_root = tmp_path / "episodic"
        stray_slot = tier_root / "20260101-000000"
        stray_slot.mkdir(parents=True)
        (stray_slot / "graph.json").write_bytes(b"{}")

        assert has_unbound_payload(tier_root) is True


class TestHasUnboundPayloadVenueAndInterimScope:
    """``has_unbound_payload``'s interim-prune and venue-blind detection
    (paramem/memory/interim_adapter.py:183-236)."""

    def test_payload_only_inside_an_interim_child_reports_no_unbound_payload(
        self, tmp_path: Path
    ) -> None:
        """A main tier root whose only payload sits inside an
        interim_<stamp>/ child reads as clean -- the scan prunes
        interim_* children before descending, so a sibling interim
        slot's own debris never satisfies the MAIN tier's signal."""
        tier_root = tmp_path / "episodic"
        interim_slot = tier_root / "interim_20260101T0000" / "20260101-000000"
        interim_slot.mkdir(parents=True)
        (interim_slot / "graph.json").write_bytes(b"{}")

        assert has_unbound_payload(tier_root) is False

    def test_train_payload_with_no_graph_json_anywhere_reports_unbound_payload(
        self, tmp_path: Path
    ) -> None:
        """A stray adapter_model.safetensors payload is detected even with
        no graph.json anywhere beneath -- has_unbound_payload is
        venue-blind (a train payload counts equally to a simulate one)."""
        tier_root = tmp_path / "episodic"
        stray_slot = tier_root / "20260101-000000"
        stray_slot.mkdir(parents=True)
        (stray_slot / payload_filename("train")).write_bytes(b"")

        assert has_unbound_payload(tier_root) is True
