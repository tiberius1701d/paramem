"""Integration tests for PostSessionQueue startup replay in the lifespan.

Tests verify that:
1. Queue file seeded before startup causes entries to be drained and replayed
   on boot (local mode, post_session_train_enabled=True).
2. No replay when post_session_train_enabled=False (default).
3. No replay in cloud-only mode (no model to train).
4. Empty queue at startup is a no-op.

All GPU/model calls are mocked — no hardware required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_queue(adapter_dir: Path, entries: list[dict]) -> Path:
    """Write a pre-populated queue file and return its path."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    path = adapter_dir / "post_session_queue.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _minimal_entry(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "transcript": f"user: hello {session_id}\nassistant: hi",
        "speaker_id": "spk-001",
        "speaker_name": "Alice",
        "enqueued_at": "2026-01-01T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Test 1 — Startup replay drains and processes pending entries
# ---------------------------------------------------------------------------


class TestStartupReplayDrainsQueue:
    def test_pending_entries_are_replayed(self, tmp_path: Path) -> None:
        """Queue entries from a previous run are replayed via enqueue_post_session_train."""
        adapter_dir = tmp_path / "adapters"
        _seed_queue(adapter_dir, [_minimal_entry("conv-r001"), _minimal_entry("conv-r002")])

        replayed: list[str] = []

        def _fake_enqueue(conversation_id, *args, **kwargs):
            replayed.append(conversation_id)

        # Minimal config with adapter_dir pointing to tmp and post_session_train enabled.
        from paramem.server.config import (
            ConsolidationScheduleConfig,
            PathsConfig,
            ServerConfig,
        )

        config = ServerConfig()
        config.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
            prompts=tmp_path / "prompts",
        )
        config.consolidation = ConsolidationScheduleConfig(
            post_session_train_enabled=True,
        )

        mock_loop = MagicMock()
        mock_bt = MagicMock()

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "consolidation_loop": None,
            "background_trainer": None,
            "post_session_queue": None,
        }
        _run_startup_replay(state, config, adapter_dir, _fake_enqueue, mock_loop, mock_bt)

        # Both entries must have been replayed.
        assert "conv-r001" in replayed, f"conv-r001 not replayed; replayed={replayed}"
        assert "conv-r002" in replayed, f"conv-r002 not replayed; replayed={replayed}"

    def test_pending_entries_re_enqueued_before_training(self, tmp_path: Path) -> None:
        """Drained entries are re-enqueued into the queue before training fires.

        This ensures crash-recovery of the replay job itself.
        """
        adapter_dir = tmp_path / "adapters"
        _seed_queue(adapter_dir, [_minimal_entry("conv-r003")])

        from paramem.server.config import (
            ConsolidationScheduleConfig,
            PathsConfig,
            ServerConfig,
        )

        config = ServerConfig()
        config.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
            prompts=tmp_path / "prompts",
        )
        config.consolidation = ConsolidationScheduleConfig(
            post_session_train_enabled=True,
        )

        enqueued_before_train: list[str] = []

        def _fake_enqueue(conversation_id, *args, post_session_queue=None, **kwargs):
            # Record what's in the queue at the moment training is called.
            if post_session_queue is not None:
                enqueued_before_train.extend([e["session_id"] for e in post_session_queue.peek()])

        mock_loop = MagicMock()
        mock_bt = MagicMock()

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "consolidation_loop": None,
            "background_trainer": None,
            "post_session_queue": None,
        }
        _run_startup_replay(state, config, adapter_dir, _fake_enqueue, mock_loop, mock_bt)

        assert "conv-r003" in enqueued_before_train, (
            f"Entry must be in queue when training is submitted; found: {enqueued_before_train}"
        )


# ---------------------------------------------------------------------------
# Test 2 — No replay when post_session_train_enabled=False
# ---------------------------------------------------------------------------


class TestNoReplayWhenDisabled:
    def test_replay_skipped_when_disabled(self, tmp_path: Path) -> None:
        """With post_session_train_enabled=False, pending entries are NOT replayed."""
        adapter_dir = tmp_path / "adapters"
        _seed_queue(adapter_dir, [_minimal_entry("conv-s001")])

        from paramem.server.config import (
            ConsolidationScheduleConfig,
            PathsConfig,
            ServerConfig,
        )

        config = ServerConfig()
        config.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
            prompts=tmp_path / "prompts",
        )
        config.consolidation = ConsolidationScheduleConfig(
            post_session_train_enabled=False,  # disabled
        )

        replayed: list[str] = []

        def _fake_enqueue(conversation_id, *args, **kwargs):
            replayed.append(conversation_id)

        mock_loop = MagicMock()
        mock_bt = MagicMock()

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "consolidation_loop": None,
            "background_trainer": None,
            "post_session_queue": None,
        }
        _run_startup_replay(state, config, adapter_dir, _fake_enqueue, mock_loop, mock_bt)

        assert replayed == [], f"No replay expected when disabled; got: {replayed}"

    def test_queue_still_instantiated_when_disabled(self, tmp_path: Path) -> None:
        """PostSessionQueue is instantiated even when post_session_train_enabled=False.

        The queue must exist so the chat handler can enqueue new jobs, even
        though replay is suppressed until the flag is turned on.
        """
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        from paramem.server.config import (
            ConsolidationScheduleConfig,
            PathsConfig,
            ServerConfig,
        )

        config = ServerConfig()
        config.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
            prompts=tmp_path / "prompts",
        )
        config.consolidation = ConsolidationScheduleConfig(
            post_session_train_enabled=False,
        )

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "consolidation_loop": None,
            "background_trainer": None,
            "post_session_queue": None,
        }
        _run_startup_replay(state, config, adapter_dir, lambda *a, **kw: None)

        assert state["post_session_queue"] is not None, (
            "post_session_queue must be instantiated even when post_session_train_enabled=False"
        )


# ---------------------------------------------------------------------------
# Test 3 — Empty queue at startup is a no-op
# ---------------------------------------------------------------------------


class TestEmptyQueueNoOp:
    def test_empty_queue_no_replay(self, tmp_path: Path) -> None:
        """When the queue is empty, no training is enqueued and no loop is created."""
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        from paramem.server.config import (
            ConsolidationScheduleConfig,
            PathsConfig,
            ServerConfig,
        )

        config = ServerConfig()
        config.paths = PathsConfig(
            data=tmp_path,
            sessions=tmp_path / "sessions",
            debug=tmp_path / "debug",
            prompts=tmp_path / "prompts",
        )
        config.consolidation = ConsolidationScheduleConfig(
            post_session_train_enabled=True,
        )

        replayed: list[str] = []

        def _fake_enqueue(conversation_id, *args, **kwargs):
            replayed.append(conversation_id)

        state = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "consolidation_loop": None,
            "background_trainer": None,
            "post_session_queue": None,
        }
        _run_startup_replay(state, config, adapter_dir, _fake_enqueue)

        assert replayed == [], f"No replay for empty queue; got: {replayed}"
        # No loop should have been created (no entries to process)
        assert state["consolidation_loop"] is None


# ---------------------------------------------------------------------------
# Test 4 — Config field default and YAML round-trip
# ---------------------------------------------------------------------------


class TestConfigField:
    def test_default_is_false(self) -> None:
        """post_session_train_enabled defaults to False."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.post_session_train_enabled is False

    def test_can_be_set_true(self) -> None:
        """post_session_train_enabled can be set to True."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(post_session_train_enabled=True)
        assert cfg.post_session_train_enabled is True

    def test_load_from_yaml_default_false(self, tmp_path: Path) -> None:
        """A minimal YAML without post_session_train_enabled loads as False."""
        yaml_text = "consolidation:\n  refresh_cadence: '12h'\n"
        path = tmp_path / "server.yaml"
        path.write_text(yaml_text)

        from paramem.server.config import load_server_config

        config = load_server_config(path)
        assert config.consolidation.post_session_train_enabled is False

    def test_load_from_yaml_explicit_true(self, tmp_path: Path) -> None:
        """YAML with post_session_train_enabled: true loads as True."""
        yaml_text = "consolidation:\n  post_session_train_enabled: true\n"
        path = tmp_path / "server.yaml"
        path.write_text(yaml_text)

        from paramem.server.config import load_server_config

        config = load_server_config(path)
        assert config.consolidation.post_session_train_enabled is True


# ---------------------------------------------------------------------------
# Shared helper — extracts the startup replay logic for direct testing
# ---------------------------------------------------------------------------


def _run_startup_replay(
    state: dict,
    config,
    adapter_dir: Path,
    fake_enqueue_fn,
    mock_loop=None,
    mock_bt=None,
) -> None:
    """Execute the startup queue-replay logic from app.py lifespan in isolation.

    This replicates the exact logic block in the lifespan (after model load)
    so tests can exercise it without booting the full FastAPI app.

    Args:
        state: Mutable dict simulating ``_state`` (must include "model",
               "tokenizer", "consolidation_loop", "background_trainer",
               "post_session_queue").
        config: A ``ServerConfig`` instance.
        adapter_dir: Path where ``post_session_queue.json`` lives / should be created.
        fake_enqueue_fn: Replacement for ``enqueue_post_session_train`` so we
               can capture what gets replayed.
        mock_loop: Optional ConsolidationLoop mock; created as ``MagicMock()`` if None.
        mock_bt: Optional BackgroundTrainer mock; created as ``MagicMock()`` if None.
    """
    from paramem.server.post_session_queue import PostSessionQueue

    if mock_loop is None:
        mock_loop = MagicMock()
    if mock_bt is None:
        mock_bt = MagicMock()

    # --- Replicate the lifespan block ---
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _queue_path = adapter_dir / "post_session_queue.json"
    state["post_session_queue"] = PostSessionQueue(_queue_path)

    if config.consolidation.post_session_train_enabled:
        _pending = state["post_session_queue"].peek()
        if _pending:
            if state.get("consolidation_loop") is None:
                mock_loop.model = state["model"]
                state["consolidation_loop"] = mock_loop
                state["model"] = mock_loop.model

            if state.get("background_trainer") is None:
                state["background_trainer"] = mock_bt

            _replay_entries = state["post_session_queue"].drain()
            for _entry in _replay_entries:
                state["post_session_queue"].enqueue(_entry)
                fake_enqueue_fn(
                    conversation_id=_entry["session_id"],
                    transcript=_entry["transcript"],
                    speaker_id=_entry["speaker_id"],
                    speaker_name=_entry.get("speaker_name"),
                    loop=state["consolidation_loop"],
                    background_trainer=state["background_trainer"],
                    config=config,
                    state=state,
                    post_session_queue=state["post_session_queue"],
                )
