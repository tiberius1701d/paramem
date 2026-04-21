"""Tests for paramem.server.drift."""

from __future__ import annotations

import asyncio

import pytest

from paramem.backup.hashing import content_sha256_path
from paramem.server.drift import (
    ConfigDriftState,
    drift_poll_loop,
    initial_drift_state,
)


class TestInitialDriftState:
    def test_initial_drift_state_captures_load_time_hash(self, tmp_path):
        """initial_drift_state seeds detected=False with the correct load-time hash."""
        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_bytes(b"model: mistral\ndebug: false\n")

        state = initial_drift_state(yaml_file)

        assert state["detected"] is False, "detected must be False on first load"
        assert state["loaded_hash"] == content_sha256_path(yaml_file), (
            "loaded_hash must equal the direct file hash"
        )
        assert state["disk_hash"] == state["loaded_hash"], (
            "disk_hash must equal loaded_hash immediately after seeding"
        )
        assert state["last_checked_at"], "last_checked_at must be populated"

    def test_initial_drift_state_missing_file(self, tmp_path):
        """initial_drift_state raises FileNotFoundError when the file is absent."""
        missing = tmp_path / "no_such_server.yaml"
        with pytest.raises(FileNotFoundError):
            initial_drift_state(missing)


class TestDriftPollLoopDetectsRewrite:
    def test_poll_loop_detects_rewrite(self, tmp_path):
        """Rewriting the file with different content flips detected to True."""
        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_bytes(b"model: mistral\n")

        state: dict = {"config_drift": initial_drift_state(yaml_file)}

        async def _run():
            task = asyncio.create_task(drift_poll_loop(yaml_file, state, interval_seconds=0.01))
            # Let one poll cycle land before mutating the file
            await asyncio.sleep(0.05)
            yaml_file.write_bytes(b"model: gemma\n")
            # 20:1 ratio (sleep / interval) to absorb CI jitter
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

        drift: ConfigDriftState = state["config_drift"]
        assert drift["detected"] is True, "detected must flip to True after content change"
        assert drift["disk_hash"] != drift["loaded_hash"], (
            "disk_hash and loaded_hash must differ when drift is detected"
        )


class TestDriftPollLoopNoopOnIdenticalRewrite:
    def test_poll_loop_noop_on_identical_rewrite(self, tmp_path):
        """Writing the same bytes again leaves detected=False."""
        content = b"model: mistral\n"
        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_bytes(content)

        state: dict = {"config_drift": initial_drift_state(yaml_file)}

        async def _run():
            task = asyncio.create_task(drift_poll_loop(yaml_file, state, interval_seconds=0.01))
            await asyncio.sleep(0.05)
            # Write the same bytes — hash must not change
            yaml_file.write_bytes(content)
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

        drift: ConfigDriftState = state["config_drift"]
        assert drift["detected"] is False, "identical rewrite must leave detected=False"
        assert drift["disk_hash"] == drift["loaded_hash"]


class TestDriftPollLoopToleratesMissingFile:
    def test_poll_loop_tolerates_missing_file(self, tmp_path):
        """Deleting the file mid-poll sets disk_hash='' and detected=True without raising."""
        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_bytes(b"model: mistral\n")

        state: dict = {"config_drift": initial_drift_state(yaml_file)}
        original_loaded_hash = state["config_drift"]["loaded_hash"]

        async def _run():
            task = asyncio.create_task(drift_poll_loop(yaml_file, state, interval_seconds=0.01))
            await asyncio.sleep(0.05)
            yaml_file.unlink()
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

        drift: ConfigDriftState = state["config_drift"]
        assert drift["disk_hash"] == "", "disk_hash must be empty when file is missing"
        assert drift["detected"] is True, "missing file must be treated as drift"
        assert drift["loaded_hash"] == original_loaded_hash, (
            "loaded_hash must remain intact even when disk file is gone"
        )


class TestDriftPollLoopCancellable:
    def test_poll_loop_cancellable(self, tmp_path):
        """The poll loop can be cancelled cleanly from the lifespan shutdown path."""
        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_bytes(b"model: mistral\n")

        state: dict = {"config_drift": initial_drift_state(yaml_file)}

        async def _run():
            task = asyncio.create_task(drift_poll_loop(yaml_file, state, interval_seconds=30.0))
            # Cancel immediately — must not raise anything other than CancelledError
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_run())
