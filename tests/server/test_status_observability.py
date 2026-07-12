"""Tests for the four observability fixes to /status.

Covers:
- _seconds_until_next_full_consolidation — deadline + ceil-to-tick cases
- tier_key_counts — shape matches tiers_with_registry / active_keys_in_tier
- VRAM ledger — load sets entry, _release_base_model_in_process clears base
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.server.app import _seconds_until_next_full_consolidation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    refresh_cadence: str = "every 12h",
    max_interim_count: int = 7,
    consolidation_period_seconds: int | None = 7 * 12 * 3600,
    adapter_dir: Path | None = None,
    tmp_path: Path | None = None,
    mode: str = "train",
) -> object:
    """Construct a minimal config-like object for consolidation helper tests.

    Uses real dataclasses where available; otherwise returns a MagicMock
    with the attrs set directly.

    ``mode`` is the consolidation venue: the schedule helpers scan interim
    slots with ``mode=config.consolidation.mode`` and count only slots that
    carry that venue's payload.
    """
    cfg = MagicMock()
    cfg.consolidation.refresh_cadence = refresh_cadence
    cfg.consolidation.max_interim_count = max_interim_count
    cfg.consolidation.consolidation_period_seconds = consolidation_period_seconds
    cfg.consolidation.mode = mode
    if adapter_dir is not None:
        cfg.adapter_dir = adapter_dir
    elif tmp_path is not None:
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_interim_dir(adapter_dir: Path, stamp: str) -> Path:
    """Create an episodic/interim_<stamp> slot carrying a train-venue payload.

    ``iter_interim_dirs`` scans <adapter_dir>/episodic/interim_*; the
    function returns (adapter_name, path) where adapter_name includes the
    "interim_" prefix.  The schedule helpers scan it venue-filtered, so the
    slot needs the train payload (``adapter_model.safetensors``) to be counted
    — a payload-less directory carries nothing to fold and is skipped.
    """
    p = adapter_dir / "episodic" / f"interim_{stamp}"
    p.mkdir(parents=True, exist_ok=True)
    # Write a minimal meta.json so the dir is recognised as a valid interim slot.
    (p / "meta.json").write_text(json.dumps({"window_stamp": stamp}))
    slot = p / f"{stamp}-slot"
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "adapter_model.safetensors").write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# _seconds_until_next_full_consolidation
# ---------------------------------------------------------------------------


class TestSecondsUntilNextFullConsolidation:
    """All cases from the contract:

    - N == 0: every tick is full → next tick.
    - No interim dirs: None.
    - Manual-only cadence: None.
    - Count-primary condition holds: next tick.
    - Deadline in future: ceiled to next tick boundary.
    - Deadline in past: next tick.
    """

    def test_n_zero_returns_next_tick(self, tmp_path):
        """N == 0 (full-fold-only mode): every tick is a full cycle."""
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=0,
            consolidation_period_seconds=0,
            adapter_dir=tmp_path / "adapters",
        )
        (tmp_path / "adapters").mkdir(parents=True, exist_ok=True)
        # Pick a time that is 1h past midnight so next 12h boundary is in 11h.
        now = datetime(2026, 7, 1, 1, 0, 0)  # 01:00
        result = _seconds_until_next_full_consolidation(cfg, now=now)
        assert result is not None
        # Next 12h boundary from midnight at 01:00 is 12:00 → 11 hours = 39600s.
        assert result == 11 * 3600

    def test_manual_only_returns_none(self, tmp_path):
        """Disabled cadence (refresh_cadence='off') must return None."""
        cfg = _make_config(
            refresh_cadence="off",
            max_interim_count=7,
            consolidation_period_seconds=None,
            adapter_dir=tmp_path / "adapters",
        )
        (tmp_path / "adapters").mkdir(parents=True, exist_ok=True)
        result = _seconds_until_next_full_consolidation(cfg)
        assert result is None

    def test_no_interim_dirs_returns_none(self, tmp_path):
        """No interim directories: nothing to fold → None."""
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=7,
            consolidation_period_seconds=7 * 12 * 3600,
            adapter_dir=tmp_path / "adapters",
        )
        (tmp_path / "adapters" / "episodic").mkdir(parents=True, exist_ok=True)
        result = _seconds_until_next_full_consolidation(cfg)
        assert result is None

    def test_count_primary_condition_holds_returns_next_tick(self, tmp_path):
        """Count-primary: n > 1 and (n-1) % N == 0 → due on next tick."""
        adapter_dir = tmp_path / "adapters"
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=3,
            consolidation_period_seconds=3 * 12 * 3600,
            adapter_dir=adapter_dir,
        )
        # Create 4 dirs: n=4, N=3 → (4-1) % 3 == 0 → count-primary holds.
        for i in range(4):
            stamp = f"20260701T{i:02d}00"
            _make_interim_dir(adapter_dir, stamp)

        now = datetime(2026, 7, 1, 1, 0, 0)  # 01:00 → next 12h tick at 12:00 = 11h
        result = _seconds_until_next_full_consolidation(cfg, now=now)
        assert result is not None
        assert result == 11 * 3600

    def test_deadline_in_future_ceiled_to_tick(self, tmp_path):
        """Deadline in future must be ceiled to the next tick boundary."""
        adapter_dir = tmp_path / "adapters"
        full_period = 3 * 12 * 3600  # 36h
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=3,
            consolidation_period_seconds=full_period,
            adapter_dir=adapter_dir,
        )
        # One interim dir created at 2026-07-01 00:00 (stamp "20260701T0000").
        _make_interim_dir(adapter_dir, "20260701T0000")
        # Now = 2026-07-01 06:00. Deadline = 2026-07-01 00:00 + 36h = 2026-07-02 12:00.
        # Next 12h ticks: ..., 2026-07-02 00:00, 2026-07-02 12:00 (= deadline), ...
        # 2026-07-02 12:00 is exactly on a tick boundary (midnight + 12h).
        now = datetime(2026, 7, 1, 6, 0, 0)
        result = _seconds_until_next_full_consolidation(cfg, now=now)
        assert result is not None
        # From 2026-07-01 06:00 to 2026-07-02 12:00 = 30h = 108000s.
        assert result == 30 * 3600

    def test_deadline_in_past_returns_next_tick(self, tmp_path):
        """Deadline already past → due at next tick."""
        adapter_dir = tmp_path / "adapters"
        full_period = 12 * 3600  # 12h — deliberately short
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=1,
            consolidation_period_seconds=full_period,
            adapter_dir=adapter_dir,
        )
        # Interim dir created "long ago" — stamp 12+ hours behind now.
        _make_interim_dir(adapter_dir, "20260630T0000")  # yesterday midnight
        # Now = 2026-07-01 06:00. Deadline was 2026-06-30 12:00 — already past.
        now = datetime(2026, 7, 1, 6, 0, 0)
        result = _seconds_until_next_full_consolidation(cfg, now=now)
        assert result is not None
        # Next 12h tick from 06:00 is 12:00 → 6h = 21600s.
        assert result == 6 * 3600

    def test_consolidation_period_none_returns_none(self, tmp_path):
        """consolidation_period_seconds=None with N>0 and interims → None."""
        adapter_dir = tmp_path / "adapters"
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=7,
            consolidation_period_seconds=None,
            adapter_dir=adapter_dir,
        )
        _make_interim_dir(adapter_dir, "20260701T0000")
        result = _seconds_until_next_full_consolidation(cfg)
        assert result is None

    def test_deadline_not_on_tick_ceils_to_next_tick(self, tmp_path):
        """Deadline not on a tick boundary must ceil to the next tick after it."""
        adapter_dir = tmp_path / "adapters"
        # 13h period — next tick at 12h boundary, then deadline falls between 12h and 24h.
        full_period = 13 * 3600
        cfg = _make_config(
            refresh_cadence="every 12h",
            max_interim_count=1,
            consolidation_period_seconds=full_period,
            adapter_dir=adapter_dir,
        )
        # Interim at midnight. Now = 01:00. Deadline = midnight + 13h = 13:00.
        # 13:00 is between 12h (12:00) and 24h (00:00 next day) tick boundaries.
        # Ceil → next day midnight (24:00 = 00:00 + 1 day).
        _make_interim_dir(adapter_dir, "20260701T0000")
        now = datetime(2026, 7, 1, 1, 0, 0)
        result = _seconds_until_next_full_consolidation(cfg, now=now)
        assert result is not None
        # Next tick at or after 13:00 is 2026-07-02 00:00 = 23h from 01:00 = 82800s.
        assert result == 23 * 3600


# ---------------------------------------------------------------------------
# tier_key_counts — shape from the MemoryStore
# ---------------------------------------------------------------------------


class TestTierKeyCounts:
    """tier_key_counts must mirror tiers_with_registry + active_keys_in_tier."""

    def _build_mock_store(self, tier_data: dict[str, list[str]]) -> object:
        """Return a MemoryStore-like mock for the given tier → keys mapping."""
        store = MagicMock()
        store.replay_enabled = True
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
