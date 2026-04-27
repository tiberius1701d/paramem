"""Tests for atomic write semantics of SessionBuffer.save_snapshot.

Verifies that save_snapshot routes through ``_atomic_write_bytes`` so the
encrypted snapshot envelope lands on disk with fsync + atomic rename
semantics, not the bare ``Path.write_bytes / Path.rename`` pattern.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from paramem.server.session_buffer import SessionBuffer


def _make_buffer_with_turn(session_dir: Path) -> SessionBuffer:
    """Return a SessionBuffer that has one in-memory turn.

    Does NOT write anything to disk — only in-memory state is set up.
    """
    buf = SessionBuffer(session_dir)
    buf.append("conv1", "user", "I live in Amsterdam")
    return buf


# ---------------------------------------------------------------------------
# Helper: monkeypatch key_store so _snapshots_enabled() returns True without
# requiring a real age key on disk.
# ---------------------------------------------------------------------------


def _enable_snapshots(monkeypatch) -> None:
    """Patch key_store so daily_identity_loadable → True."""
    monkeypatch.setattr(
        "paramem.backup.key_store.daily_identity_loadable",
        lambda *_a, **_kw: True,
    )
    # envelope_encrypt_bytes falls through to plaintext when no real identity
    # is present; that's fine for atomic-write shape tests — we only care that
    # _atomic_write_bytes is called, not what bytes it receives.


class TestSaveSnapshotUsesAtomicHelper:
    """save_snapshot must write through _atomic_write_bytes."""

    def test_atomic_helper_called_once(self, tmp_path: Path, monkeypatch) -> None:
        """_atomic_write_bytes is invoked exactly once with the snapshot path."""
        _enable_snapshots(monkeypatch)
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        buf = _make_buffer_with_turn(session_dir)
        snap_path = buf._snapshot_path

        recorded: list[tuple[Path, bytes]] = []

        def _fake_atomic(path: Path, body: bytes) -> None:
            recorded.append((path, body))

        with patch(
            "paramem.backup.encryption._atomic_write_bytes",
            side_effect=_fake_atomic,
        ):
            result = buf.save_snapshot()

        assert result is True, "save_snapshot must return True on success"
        assert len(recorded) == 1, "_atomic_write_bytes must be called exactly once"
        assert recorded[0][0] == snap_path, (
            f"_atomic_write_bytes called with wrong path: {recorded[0][0]!r} "
            f"(expected {snap_path!r})"
        )

    def test_atomic_helper_receives_bytes(self, tmp_path: Path, monkeypatch) -> None:
        """The bytes handed to _atomic_write_bytes are non-empty."""
        _enable_snapshots(monkeypatch)
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        buf = _make_buffer_with_turn(session_dir)

        captured_body: list[bytes] = []

        def _fake_atomic(path: Path, body: bytes) -> None:
            captured_body.append(body)

        with patch(
            "paramem.backup.encryption._atomic_write_bytes",
            side_effect=_fake_atomic,
        ):
            buf.save_snapshot()

        assert captured_body, "_atomic_write_bytes received no bytes"
        assert len(captured_body[0]) > 0, "body bytes must be non-empty"


class TestSaveSnapshotNoPartialFileOnFailure:
    """When _atomic_write_bytes raises, save_snapshot returns False and no
    partial file is left on disk."""

    def test_returns_false_on_write_failure(self, tmp_path: Path, monkeypatch) -> None:
        """OSError from _atomic_write_bytes is caught → save_snapshot returns False."""
        _enable_snapshots(monkeypatch)
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        buf = _make_buffer_with_turn(session_dir)

        with patch(
            "paramem.backup.encryption._atomic_write_bytes",
            side_effect=OSError("disk full"),
        ):
            result = buf.save_snapshot()

        assert result is False, "save_snapshot must return False on write failure"

    def test_snapshot_path_absent_on_write_failure(self, tmp_path: Path, monkeypatch) -> None:
        """When the write fails, the canonical snapshot path must not exist."""
        _enable_snapshots(monkeypatch)
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        buf = _make_buffer_with_turn(session_dir)
        snap_path = buf._snapshot_path

        with patch(
            "paramem.backup.encryption._atomic_write_bytes",
            side_effect=OSError("disk full"),
        ):
            buf.save_snapshot()

        assert not snap_path.exists(), (
            f"Canonical snapshot path {snap_path} must not exist after a write failure"
        )


class TestSaveSnapshotNoTmpLeftBehind:
    """Happy-path call must leave no *.tmp files in the session directory."""

    def test_no_tmp_files_after_success(self, tmp_path: Path, monkeypatch) -> None:
        """After a successful save_snapshot, no .tmp files remain in session_dir."""
        _enable_snapshots(monkeypatch)
        session_dir = tmp_path / "sessions"
        session_dir.mkdir()

        buf = _make_buffer_with_turn(session_dir)

        # Patch envelope_encrypt_bytes so the real age stack is not needed.
        with patch(
            "paramem.server.session_buffer.envelope_encrypt_bytes",
            return_value=b"fake-envelope-bytes",
        ):
            result = buf.save_snapshot()

        assert result is True

        tmp_files = list(session_dir.glob("*.tmp"))
        assert not tmp_files, (
            f"Leftover .tmp files after save_snapshot: {[str(p) for p in tmp_files]}"
        )
