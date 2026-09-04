"""Tests for /status observability: consolidation deadline prediction, tier
key counts, and the VRAM ledger.

Covers:
- _seconds_until_next_full_consolidation — the first full_window opening
  (or, without a ring, the next cadence mark) a pending deadline falls due at
- tier_key_counts — shape matches tiers_with_registry / active_keys_in_tier
- VRAM ledger — load sets entry, _release_base_model_in_process clears base
- the pending-event block (kind/since/next opportunity) and next_interim_seconds /
  next_full_consolidation_seconds, driven through the real status() route
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.server._state_builders import _write_pending_ledger

# ---------------------------------------------------------------------------
# tier_key_counts — shape from the MemoryStore
# ---------------------------------------------------------------------------


class TestTierKeyCounts:
    """tier_key_counts must mirror tiers_with_registry + active_keys_in_tier."""

    def _build_mock_store(self, tier_data: dict[str, list[str]]) -> object:
        """Return a MemoryStore-like mock for the given tier → keys mapping."""
        store = MagicMock()
        store.all_active_keys.return_value = [k for keys in tier_data.values() for k in keys]
        store.tiers_with_registry.return_value = list(tier_data.keys())
        store.active_keys_in_tier.side_effect = lambda tier: tier_data.get(tier, [])
        return store

    def test_main_tiers_only(self):
        """Main-tier-only setup: episodic/semantic/procedural counts are exact."""
        tier_data = {
            "episodic": ["e1", "e2", "e3"],
            "semantic": ["s1"],
            "procedural": [],
        }
        store = self._build_mock_store(tier_data)
        result = {
            tier: len(store.active_keys_in_tier(tier)) for tier in store.tiers_with_registry()
        }
        assert result == {"episodic": 3, "semantic": 1, "procedural": 0}

    def test_with_interim_tiers(self):
        """Interim tiers are included with their raw name as key."""
        tier_data = {
            "episodic": ["e1", "e2"],
            "episodic_interim_202607010000": ["i1", "i2", "i3"],
            "episodic_interim_202607011200": ["i4"],
        }
        store = self._build_mock_store(tier_data)
        result = {
            tier: len(store.active_keys_in_tier(tier)) for tier in store.tiers_with_registry()
        }
        assert result["episodic"] == 2
        assert result["episodic_interim_202607010000"] == 3
        assert result["episodic_interim_202607011200"] == 1

    def test_empty_store(self):
        """Empty registry returns empty dict."""
        store = self._build_mock_store({})
        result = {
            tier: len(store.active_keys_in_tier(tier)) for tier in store.tiers_with_registry()
        }
        assert result == {}


# ---------------------------------------------------------------------------
# VRAM ledger — _state["vram_components"] truthfulness
# ---------------------------------------------------------------------------


class TestVramLedger:
    """The ledger must be set at load time and cleared at release time."""

    def test_base_release_clears_ledger_entry(self):
        """_release_base_model_in_process must pop 'base' from vram_components."""
        import paramem.server.app as app_module

        # Prime the ledger with a fake base entry.
        prior_comps = dict(app_module._state.get("vram_components") or {})
        prior_model = app_module._state.get("model")
        prior_bt = app_module._state.get("background_trainer")
        prior_loop = app_module._state.get("consolidation_loop")
        prior_tokenizer = app_module._state.get("tokenizer")

        try:
            app_module._state["vram_components"] = {"base": 4_000 * 1024 * 1024, "stt": 100}
            app_module._state["model"] = None
            app_module._state["background_trainer"] = None
            app_module._state["consolidation_loop"] = None
            app_module._state["tokenizer"] = None

            # set_classifier_model is a local import inside the function
            # so must be patched at its definition site in paramem.server.intent.
            with (
                patch("paramem.server.intent.set_classifier_model"),
                patch("paramem.server.app.safe_empty_cache"),
            ):
                app_module._release_base_model_in_process()

            # "base" must be gone; "stt" must remain (STT is not released here).
            comps = app_module._state.get("vram_components") or {}
            assert "base" not in comps, "'base' must be cleared by _release_base_model_in_process"
            assert "stt" in comps, "'stt' must survive a base-model release"
        finally:
            app_module._state["vram_components"] = prior_comps
            app_module._state["model"] = prior_model
            app_module._state["background_trainer"] = prior_bt
            app_module._state["consolidation_loop"] = prior_loop
            app_module._state["tokenizer"] = prior_tokenizer

    def test_stt_cpu_profile_clears_stt_ledger(self):
        """Switching to cpu profile must pop 'stt' from the ledger."""
        import paramem.server.app as app_module

        prior_comps = dict(app_module._state.get("vram_components") or {})
        prior_profile = app_module._state.get("voice_profile")
        prior_stt = app_module._state.get("stt")
        prior_stt_gpu = app_module._state.get("stt_gpu")
        prior_tts_gpu = app_module._state.get("tts_gpu")
        prior_stt_cpu = app_module._state.get("stt_cpu")
        prior_tts_cpu = app_module._state.get("tts_cpu")
        prior_voice_box = app_module._state.get("voice_box")
        prior_tts_manager = app_module._state.get("tts_manager")
        prior_config = app_module._state.get("config")

        try:
            # Set up a loaded STT GPU entry in the ledger.
            app_module._state["vram_components"] = {"base": 100, "stt": 1_000 * 1024 * 1024}
            # Provide a mock voice_profile so the early-return guard fires correctly.
            app_module._state["voice_profile"] = "gpu"

            mock_stt_gpu = MagicMock()
            mock_stt_gpu.is_loaded = True
            mock_stt_gpu.unload = MagicMock()
            app_module._state["stt_gpu"] = mock_stt_gpu
            app_module._state["tts_gpu"] = None
            app_module._state["stt_cpu"] = MagicMock()
            app_module._state["tts_cpu"] = MagicMock()

            # Minimal config mock to keep gpu_lock import happy.
            mock_cfg = MagicMock()
            mock_cfg.stt.enabled = True
            mock_cfg.tts.enabled = False
            app_module._state["config"] = mock_cfg

            from contextlib import nullcontext

            # gpu_lock_sync is a local import inside the function — patch at source.
            with (
                patch("paramem.server.gpu_lock.gpu_lock_sync", return_value=nullcontext()),
                patch("paramem.server.app.safe_empty_cache"),
            ):
                app_module._set_voice_pipeline_profile("cpu")

            comps = app_module._state.get("vram_components") or {}
            assert "stt" not in comps, "'stt' must be cleared when voice profile flips to cpu"
            assert "base" in comps, "'base' must survive an STT unload"
        finally:
            app_module._state["vram_components"] = prior_comps
            app_module._state["voice_profile"] = prior_profile
            app_module._state["stt"] = prior_stt
            app_module._state["stt_gpu"] = prior_stt_gpu
            app_module._state["tts_gpu"] = prior_tts_gpu
            app_module._state["stt_cpu"] = prior_stt_cpu
            app_module._state["tts_cpu"] = prior_tts_cpu
            app_module._state["voice_box"] = prior_voice_box
            app_module._state["tts_manager"] = prior_tts_manager
            app_module._state["config"] = prior_config


# ---------------------------------------------------------------------------
# The pending-event block and the two ETA fields, driven through the real
# status() route against a real ServerConfig (tests/fixtures/server.yaml).
# ---------------------------------------------------------------------------


@pytest.fixture()
def berlin_timezone(monkeypatch):
    """Pin the process timezone to Europe/Berlin for a window/cadence test,
    restored (env and C library state alike) afterward."""
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _status_config(tmp_path: Path, **consolidation_overrides):
    """A real ``ServerConfig`` (``tests/fixtures/server.yaml``) rooted under
    *tmp_path*, with any ``consolidation.*`` field overridden by keyword —
    the "Test config loader" convention (per-test overrides in code, never
    fixture edits)."""
    from paramem.server.config import PathsConfig, load_server_config

    cfg = load_server_config("tests/fixtures/server.yaml")
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    cfg.paths = PathsConfig(
        data=data_root, sessions=data_root / "sessions", debug=data_root / "debug"
    )
    for key, value in consolidation_overrides.items():
        setattr(cfg.consolidation, key, value)
    return cfg


def _build_status_state(cfg) -> dict:
    """A minimal ``_state`` dict sufficient to drive the real ``status()``
    route end to end — shape mirrors ``test_attention_status_e2e.py``'s
    ``_make_state``."""
    from paramem.server.migration import initial_migration_state

    buf = MagicMock()
    buf.get_summary.return_value = {
        "total": 0,
        "orphaned": 0,
        "oldest_age_seconds": None,
        "per_speaker": {},
        "per_source_type": {},
    }
    return {
        "model": None,
        "tokenizer": None,
        "config": cfg,
        "config_path": None,
        "session_buffer": buf,
        "speaker_store": None,
        "router": None,
        "cloud_agent": None,
        "ha_client": None,
        "consolidation_loop": None,
        "consolidating": False,
        "last_consolidation": None,
        "background_trainer": None,
        "mode": "local",
        "cloud_only_reason": None,
        "tts_manager": None,
        "stt": None,
        "speaker_embedding_backend": None,
        "unknown_speakers": {},
        "pending_enrollments": set(),
        "migration": initial_migration_state(),
        "server_started_at": "2026-01-01T00:00:00+00:00",
        "config_drift": {
            "detected": False,
            "loaded_hash": "",
            "disk_hash": "",
            "last_checked_at": "",
        },
        "adapter_manifest_status": {},
        "last_consolidation_result": None,
        "last_model_use_monotonic": None,
        "pending_rehydration": False,
        "integrity_check_failed": False,
    }


class TestPendingEventBlock:
    """``GET /status``'s pending-event block: kind, since-when, and the next
    resume opportunity — the same reads the arbitrator makes for the decider,
    exercised read-only through a ``choose_consolidation_run(reason=TIMER,
    ...)`` call inside ``status()``."""

    def test_readable_pending_interim_ledger_reports_kind_since_and_zero_wait_when_idle(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        _write_pending_ledger(tmp_path / "data", event="interim")
        state = _build_status_state(cfg)
        monkeypatch.setattr(app_module, "_state", state)

        body = asyncio.run(app_module.status())

        assert body.pending_event_kind == "interim"
        assert body.pending_event_since == "2026-01-01T00:00:00+00:00"
        assert body.pending_event_next_seconds == 0

    def test_pending_event_next_seconds_equals_seconds_until_idle_mid_conversation(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        _write_pending_ledger(tmp_path / "data", event="interim")
        state = _build_status_state(cfg)
        # The idle clock is stamped at exactly "now" -- a conversation just
        # used the model, so the server is not idle yet.
        fixed_monotonic = 1_000_000.0
        state["last_model_use_monotonic"] = fixed_monotonic
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module.time, "monotonic", lambda: fixed_monotonic)

        body = asyncio.run(app_module.status())

        expected_seconds_until_idle = cfg.session.idle_timeout_minutes * 60
        assert body.pending_event_next_seconds == expected_seconds_until_idle

    def test_no_pending_ledger_reports_all_three_fields_as_none(
        self, tmp_path, monkeypatch
    ) -> None:
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        state = _build_status_state(cfg)
        monkeypatch.setattr(app_module, "_state", state)

        body = asyncio.run(app_module.status())

        assert body.pending_event_kind is None
        assert body.pending_event_since is None
        assert body.pending_event_next_seconds is None

    def test_running_consolidation_reports_no_pending_fields_even_with_a_readable_ledger(
        self, tmp_path, monkeypatch
    ) -> None:
        """A RUNNING event is already reported by ``consolidating``: the
        pending-event block is gated on the arbitrator's own busy signal,
        not on ledger presence alone, so a healthy in-progress fold does not
        also read as a pending resume."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        _write_pending_ledger(tmp_path / "data", event="interim")
        state = _build_status_state(cfg)
        state["consolidating"] = True
        monkeypatch.setattr(app_module, "_state", state)

        body = asyncio.run(app_module.status())

        assert body.pending_event_kind is None
        assert body.pending_event_since is None
        assert body.pending_event_next_seconds is None

    def test_garbage_ledger_bytes_report_unreadable_kind_and_record_no_incident(
        self, tmp_path, monkeypatch
    ) -> None:
        """A present ledger this process cannot decode reports kind
        'unreadable' with the other two fields left None, and records no
        incident of its own -- ``/status`` only observes what is on disk;
        recording is the arbitrator's own dispatch's job."""
        import paramem.server.app as app_module
        from paramem.server.incidents import INCIDENTS_FILENAME
        from paramem.training.stage_ledger import data_state_dir, ledger_path

        cfg = _status_config(tmp_path)
        state_dir = data_state_dir(cfg.paths.data)
        state_dir.mkdir(parents=True, exist_ok=True)
        ledger_path(state_dir).write_bytes(b"not json at all {{{")
        state = _build_status_state(cfg)
        monkeypatch.setattr(app_module, "_state", state)

        body = asyncio.run(app_module.status())

        assert body.pending_event_kind == "unreadable"
        assert body.pending_event_since is None
        assert body.pending_event_next_seconds is None
        assert not (state_dir / INCIDENTS_FILENAME).exists()

    def test_unsupported_schedule_state_version_narrows_next_seconds_to_none(
        self, tmp_path, monkeypatch
    ) -> None:
        """A readable ledger plus a schedule-marks stamp file carrying a
        ``schema_version`` this build does not write still reports the
        pending event's kind and since-when -- only the next resume
        opportunity narrows to None, and the route does not raise."""
        import json

        import paramem.server.app as app_module
        from paramem.server.schedule_state import SCHEDULE_STATE_FILENAME
        from paramem.training.stage_ledger import data_state_dir

        cfg = _status_config(tmp_path)
        _write_pending_ledger(tmp_path / "data", event="interim")
        state_dir = data_state_dir(cfg.paths.data)
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / SCHEDULE_STATE_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "last_cadence_mark_epoch": None,
                    "last_full_start_epoch": None,
                }
            )
        )
        state = _build_status_state(cfg)
        monkeypatch.setattr(app_module, "_state", state)

        body = asyncio.run(app_module.status())

        assert body.pending_event_kind == "interim"
        assert body.pending_event_since is not None
        assert body.pending_event_next_seconds is None


class TestNextIntervalSeconds:
    """``next_interim_seconds`` / ``next_full_consolidation_seconds`` — the
    honest ETA fields, derived from ``schedule_grammar`` rather than a raw
    timer tick."""

    def test_next_interim_seconds_is_correct_for_an_anchored_daily_cadence(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path, refresh_cadence="daily 04:00")
        state = _build_status_state(cfg)
        monkeypatch.setattr(app_module, "_state", state)
        # 01:00 local; the next "daily 04:00" mark is the same day's 04:00,
        # three hours away -- hand-computed, not re-derived from next_mark.
        fixed_now = datetime(2026, 1, 6, 1, 0).timestamp()
        monkeypatch.setattr(app_module.time, "time", lambda: fixed_now)

        body = asyncio.run(app_module.status())

        assert body.next_interim_seconds == 3 * 3600

    def test_next_full_consolidation_seconds_at_count_zero_is_the_next_cadence_mark(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path, refresh_cadence="12h", max_interim_count=0)
        state = _build_status_state(cfg)
        monkeypatch.setattr(app_module, "_state", state)
        # 13:00 local; "12h" marks fall at 00:00/12:00, so the next mark is
        # tomorrow's 00:00 -- eleven hours away, hand-computed.
        fixed_now = datetime(2026, 1, 6, 13, 0)
        monkeypatch.setattr(app_module.time, "time", lambda: fixed_now.timestamp())

        class _FixedDatetime(app_module.datetime):
            """``_seconds_until_next_full_consolidation`` defaults its own
            ``now`` argument via ``datetime.now()`` rather than ``time.time()``
            -- pin that one call site directly rather than the clock."""

            @classmethod
            def now(cls, tz=None):
                return fixed_now.replace(tzinfo=tz) if tz is not None else fixed_now

        monkeypatch.setattr(app_module, "datetime", _FixedDatetime)

        body = asyncio.run(app_module.status())

        assert body.next_full_consolidation_seconds == 39600


# ---------------------------------------------------------------------------
# _seconds_until_next_full_consolidation — called directly (not through the
# status() route), _full_fold_deadline monkeypatched to isolate the
# Window-opening-selection math from the deadline's own derivation.
# ---------------------------------------------------------------------------


class TestSecondsUntilNextFullConsolidation:
    """The first ``full_window`` opening (or, without a ring, the next
    cadence mark) a pending deadline falls due at."""

    def test_no_deadline_returns_none(self, tmp_path, monkeypatch, berlin_timezone) -> None:
        """``_full_fold_deadline`` answering None -- no payload-bearing
        interim slot, or no configured period -- propagates straight
        through."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        monkeypatch.setattr(app_module, "_full_fold_deadline", lambda config: None)

        result = app_module._seconds_until_next_full_consolidation(
            cfg, now=datetime(2026, 1, 6, 12, 0)
        )

        assert result is None

    def test_deadline_thirty_hours_out_lands_on_tomorrows_opening(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        """now=12:00 with a deadline 30h out: today's 01:00-04:00 opening has
        already passed, so the answer is tomorrow's 01:00, 13h away."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        now = datetime(2026, 1, 6, 12, 0)
        deadline = now + timedelta(hours=30)
        monkeypatch.setattr(app_module, "_full_fold_deadline", lambda config: deadline.timestamp())

        result = app_module._seconds_until_next_full_consolidation(cfg, now=now)

        assert result == 46800

    def test_deadline_six_hours_out_still_lands_on_tomorrows_opening(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        """A deadline that falls well inside today's remaining hours still
        resolves to tomorrow's opening, since today's own opening is
        already behind *now*."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        now = datetime(2026, 1, 6, 12, 0)
        deadline = now + timedelta(hours=6)
        monkeypatch.setattr(app_module, "_full_fold_deadline", lambda config: deadline.timestamp())

        result = app_module._seconds_until_next_full_consolidation(cfg, now=now)

        assert result == 46800

    def test_now_inside_the_window_with_a_same_day_deadline_is_already_due(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        """now=02:00 is inside today's own opening; a deadline later the
        same day does not push the opening forward, so the wait is zero."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        now = datetime(2026, 1, 6, 2, 0)
        deadline = datetime(2026, 1, 6, 10, 0)
        monkeypatch.setattr(app_module, "_full_fold_deadline", lambda config: deadline.timestamp())

        result = app_module._seconds_until_next_full_consolidation(cfg, now=now)

        assert result == 0

    def test_deadline_three_days_out_lands_on_the_opening_it_falls_inside(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        """A deadline several openings out resolves to the last opening that
        begins before it -- not the first opening, and not a fixed step
        count."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path)
        now = datetime(2026, 1, 6, 2, 0)
        deadline = now + timedelta(days=3)
        monkeypatch.setattr(app_module, "_full_fold_deadline", lambda config: deadline.timestamp())
        # The last daily 01:00 opening that begins at or before the
        # deadline -- computed independently of Window's own stepping.
        opening_start = deadline.replace(hour=1, minute=0, second=0, microsecond=0)
        if opening_start > deadline:
            opening_start -= timedelta(days=1)
        expected = int((opening_start - now).total_seconds())

        result = app_module._seconds_until_next_full_consolidation(cfg, now=now)

        assert result == expected == 255600

    def test_wrapping_window_with_a_same_day_deadline_is_already_due(
        self, tmp_path, monkeypatch, berlin_timezone
    ) -> None:
        """A window that wraps past midnight (23:00-02:00): now=00:30 is
        inside the opening that started the previous calendar day, and a
        same-day deadline does not push it forward."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path, full_window="23:00-02:00")
        now = datetime(2026, 1, 6, 0, 30)
        deadline = datetime(2026, 1, 6, 12, 0)
        monkeypatch.setattr(app_module, "_full_fold_deadline", lambda config: deadline.timestamp())

        result = app_module._seconds_until_next_full_consolidation(cfg, now=now)

        assert result == 0

    def test_no_ring_reads_the_next_cadence_mark(self, tmp_path, berlin_timezone) -> None:
        """``max_interim_count=0`` means there is no ring to derive a
        deadline from -- the answer is simply the next cadence mark."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path, max_interim_count=0, refresh_cadence="12h")

        result = app_module._seconds_until_next_full_consolidation(
            cfg, now=datetime(2026, 1, 6, 13, 0)
        )

        assert result == 39600

    def test_no_ring_with_a_non_exact_interval_cadence_has_no_marks(
        self, tmp_path, berlin_timezone
    ) -> None:
        """A non-exact interval cadence (7h does not divide 24h) has no
        wall-clock marks to answer with."""
        import paramem.server.app as app_module

        cfg = _status_config(tmp_path, max_interim_count=0, refresh_cadence="7h")

        result = app_module._seconds_until_next_full_consolidation(
            cfg, now=datetime(2026, 1, 6, 13, 0)
        )

        assert result is None
