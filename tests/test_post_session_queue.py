"""Unit tests for PostSessionQueue.

All tests are pure-Python (no GPU required).  They cover:

1. Round-trip enqueue / peek / drain.
2. remove() by session_id.
3. Atomic write — simulate mid-write failure, verify no partial file.
4. Concurrent enqueue from two threads (lock holds).
5. Load from pre-existing file on construction.
6. enqueue() raises ValueError when session_id is missing.
7. drain() on empty queue returns [] and leaves file empty.
8. remove() of non-existent session_id is a no-op.
9. peek() does not modify the queue.
10. Corrupt file on construction is handled gracefully (starts empty).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from paramem.server.post_session_queue import PostSessionQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(session_id: str, **extra) -> dict:
    """Build a minimal valid queue entry."""
    return {
        "session_id": session_id,
        "transcript": f"user: hello {session_id}",
        "speaker_id": "spk-001",
        "speaker_name": "Alice",
        **extra,
    }


# ---------------------------------------------------------------------------
# Test 1 — Round-trip enqueue / peek / drain
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_enqueue_peek_drain(self, tmp_path: Path) -> None:
        """enqueue adds entries; peek reads without clearing; drain returns and clears."""
        q = PostSessionQueue(tmp_path / "queue.json")

        q.enqueue(_entry("conv-001"))
        q.enqueue(_entry("conv-002"))

        # peek returns both entries without clearing
        snapshot = q.peek()
        assert len(snapshot) == 2
        assert snapshot[0]["session_id"] == "conv-001"
        assert snapshot[1]["session_id"] == "conv-002"

        # peek does not clear
        assert len(q) == 2

        # drain returns all and clears
        drained = q.drain()
        assert len(drained) == 2
        assert drained[0]["session_id"] == "conv-001"

        assert len(q) == 0
        assert q.peek() == []

    def test_drain_returns_copy_not_reference(self, tmp_path: Path) -> None:
        """Mutating the drained list does not affect subsequent queue state."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-003"))

        drained = q.drain()
        drained.append({"session_id": "injected"})

        # Queue remains empty even after list mutation
        assert len(q) == 0

    def test_enqueued_at_is_added(self, tmp_path: Path) -> None:
        """enqueue adds 'enqueued_at' to entries that lack it."""
        q = PostSessionQueue(tmp_path / "queue.json")
        entry = _entry("conv-004")
        assert "enqueued_at" not in entry

        q.enqueue(entry)
        stored = q.peek()[0]
        assert "enqueued_at" in stored

    def test_enqueued_at_not_overwritten(self, tmp_path: Path) -> None:
        """An explicit enqueued_at in the entry is preserved verbatim."""
        q = PostSessionQueue(tmp_path / "queue.json")
        entry = _entry("conv-005", enqueued_at="2026-01-01T00:00:00+00:00")
        q.enqueue(entry)
        stored = q.peek()[0]
        assert stored["enqueued_at"] == "2026-01-01T00:00:00+00:00"

    def test_caller_entry_not_mutated(self, tmp_path: Path) -> None:
        """enqueue does not mutate the caller's entry dict."""
        q = PostSessionQueue(tmp_path / "queue.json")
        entry = _entry("conv-006")
        original_keys = set(entry.keys())
        q.enqueue(entry)
        assert set(entry.keys()) == original_keys


# ---------------------------------------------------------------------------
# Test 2 — remove() by session_id
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing_entry(self, tmp_path: Path) -> None:
        """remove() deletes the matching entry and persists."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-010"))
        q.enqueue(_entry("conv-011"))
        q.enqueue(_entry("conv-012"))

        q.remove("conv-011")

        remaining = q.peek()
        ids = [e["session_id"] for e in remaining]
        assert "conv-011" not in ids
        assert "conv-010" in ids
        assert "conv-012" in ids

    def test_remove_nonexistent_is_noop(self, tmp_path: Path) -> None:
        """remove() of an absent session_id leaves the queue unchanged."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-013"))

        q.remove("does-not-exist")

        assert len(q) == 1

    def test_remove_persists_to_disk(self, tmp_path: Path) -> None:
        """After remove(), the backing file reflects the updated queue."""
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(_entry("conv-014"))
        q.enqueue(_entry("conv-015"))

        q.remove("conv-014")

        # Reload from disk
        q2 = PostSessionQueue(path)
        ids = [e["session_id"] for e in q2.peek()]
        assert "conv-014" not in ids
        assert "conv-015" in ids

    def test_remove_only_first_occurrence(self, tmp_path: Path) -> None:
        """remove() deletes ALL entries with the matching session_id.

        Duplicate session_ids are unusual but must not cause a partial removal.
        """
        q = PostSessionQueue(tmp_path / "queue.json")
        # Force duplicate by bypassing the enqueue method
        q._entries = [
            _entry("conv-016"),
            _entry("conv-016"),
            _entry("conv-017"),
        ]
        q._save_locked()

        q.remove("conv-016")
        ids = [e["session_id"] for e in q.peek()]
        assert "conv-016" not in ids
        assert "conv-017" in ids


# ---------------------------------------------------------------------------
# Test 3 — Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_no_partial_write_on_io_error(self, tmp_path: Path) -> None:
        """If the temp-file write fails, the original queue file is unchanged.

        We mock ``Path.write_text`` on the .tmp file to raise an IOError mid-write,
        then verify the backing file is still valid.
        """
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(_entry("conv-020"))

        original_content = path.read_text()

        # Make the NEXT write_text call (which targets the .tmp file) raise.
        real_write_text = Path.write_text

        def _failing_write(self, text, **kwargs):  # noqa: ANN001
            if str(self).endswith(".tmp"):
                raise IOError("simulated disk full")
            return real_write_text(self, text, **kwargs)

        with patch.object(Path, "write_text", _failing_write):
            try:
                q.enqueue(_entry("conv-021"))
            except Exception:
                pass  # write failed — this is expected

        # Backing file must be unchanged (not partially written)
        assert path.read_text() == original_content

    def test_temp_file_removed_after_failed_write(self, tmp_path: Path) -> None:
        """After a failed write, the .tmp sibling file is cleaned up."""
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(_entry("conv-022"))

        real_write_text = Path.write_text

        def _failing_write(self, text, **kwargs):  # noqa: ANN001
            if str(self).endswith(".tmp"):
                raise IOError("simulated disk full")
            return real_write_text(self, text, **kwargs)

        with patch.object(Path, "write_text", _failing_write):
            try:
                q.enqueue(_entry("conv-023"))
            except Exception:
                pass

        tmp_path_sibling = path.with_suffix(".json.tmp")
        assert not tmp_path_sibling.exists()

    def test_successful_write_is_valid_json(self, tmp_path: Path) -> None:
        """The backing file is always valid JSON after a write."""
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(_entry("conv-024"))
        q.enqueue(_entry("conv-025"))

        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2


# ---------------------------------------------------------------------------
# Test 4 — Concurrent enqueue from two threads
# ---------------------------------------------------------------------------


class TestConcurrentEnqueue:
    def test_two_threads_no_data_loss(self, tmp_path: Path) -> None:
        """Concurrent enqueue from two threads produces correct total count.

        No entries must be lost and no exception must be raised.
        """
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)

        errors: list[Exception] = []

        def _enqueue_range(start: int, count: int) -> None:
            for i in range(start, start + count):
                try:
                    q.enqueue(_entry(f"conv-t{i}"))
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=_enqueue_range, args=(0, 25))
        t2 = threading.Thread(target=_enqueue_range, args=(25, 25))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Concurrent enqueue raised errors: {errors}"
        # 50 entries total — no drops
        assert len(q) == 50

    def test_two_threads_all_session_ids_present(self, tmp_path: Path) -> None:
        """After concurrent enqueue, all session_ids appear exactly once."""
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)

        def _enqueue_range(start: int, count: int) -> None:
            for i in range(start, start + count):
                q.enqueue(_entry(f"conv-c{i}"))

        t1 = threading.Thread(target=_enqueue_range, args=(0, 20))
        t2 = threading.Thread(target=_enqueue_range, args=(20, 20))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        ids = {e["session_id"] for e in q.peek()}
        expected = {f"conv-c{i}" for i in range(40)}
        assert ids == expected


# ---------------------------------------------------------------------------
# Test 5 — Load from pre-existing file on construction
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    def test_constructor_loads_existing_file(self, tmp_path: Path) -> None:
        """PostSessionQueue reads and exposes entries written by a previous instance."""
        path = tmp_path / "queue.json"
        # Write a pre-existing queue file
        entries = [_entry("conv-030"), _entry("conv-031")]
        path.write_text(json.dumps(entries), encoding="utf-8")

        q = PostSessionQueue(path)

        assert len(q) == 2
        ids = [e["session_id"] for e in q.peek()]
        assert ids == ["conv-030", "conv-031"]

    def test_constructor_starts_empty_when_file_absent(self, tmp_path: Path) -> None:
        """PostSessionQueue starts empty when the backing file does not exist."""
        path = tmp_path / "nonexistent.json"
        q = PostSessionQueue(path)
        assert len(q) == 0

    def test_constructor_survives_corrupt_file(self, tmp_path: Path) -> None:
        """PostSessionQueue starts empty (not raises) on corrupt JSON."""
        path = tmp_path / "queue.json"
        path.write_text("not valid json {{{", encoding="utf-8")

        q = PostSessionQueue(path)
        assert len(q) == 0

    def test_constructor_survives_non_array_file(self, tmp_path: Path) -> None:
        """PostSessionQueue starts empty when the file contains a non-array value."""
        path = tmp_path / "queue.json"
        path.write_text(json.dumps({"session_id": "oops"}), encoding="utf-8")

        q = PostSessionQueue(path)
        assert len(q) == 0


# ---------------------------------------------------------------------------
# Test 6 — enqueue() raises ValueError when session_id missing
# ---------------------------------------------------------------------------


class TestEnqueueValidation:
    def test_missing_session_id_raises(self, tmp_path: Path) -> None:
        """enqueue() with no 'session_id' raises ValueError."""
        q = PostSessionQueue(tmp_path / "queue.json")
        with pytest.raises(ValueError, match="session_id"):
            q.enqueue({"transcript": "hello", "speaker_id": "spk"})

    def test_valid_entry_does_not_raise(self, tmp_path: Path) -> None:
        """enqueue() with a valid entry does not raise."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-040"))  # must not raise


# ---------------------------------------------------------------------------
# Test 7 — drain() on empty queue
# ---------------------------------------------------------------------------


class TestDrainEmpty:
    def test_drain_empty_returns_empty_list(self, tmp_path: Path) -> None:
        """drain() on an empty queue returns [] and does not raise."""
        q = PostSessionQueue(tmp_path / "queue.json")
        result = q.drain()
        assert result == []

    def test_drain_empty_writes_empty_array_to_disk(self, tmp_path: Path) -> None:
        """drain() on an empty queue persists an empty JSON array."""
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(_entry("conv-050"))
        q.drain()

        data = json.loads(path.read_text())
        assert data == []


# ---------------------------------------------------------------------------
# Test 8 — peek() is non-destructive
# ---------------------------------------------------------------------------


class TestPeekNonDestructive:
    def test_peek_does_not_clear_entries(self, tmp_path: Path) -> None:
        """peek() followed by peek() returns the same entries both times."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-060"))
        q.enqueue(_entry("conv-061"))

        first = q.peek()
        second = q.peek()

        assert len(first) == 2
        assert first == second

    def test_peek_returns_copy(self, tmp_path: Path) -> None:
        """Mutating the peek() result does not affect the queue."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-062"))

        snapshot = q.peek()
        snapshot.clear()

        assert len(q) == 1


# ---------------------------------------------------------------------------
# Test 9 — remove() does not write if nothing changed
# ---------------------------------------------------------------------------


class TestRemoveNoop:
    def test_remove_noop_does_not_rewrite_file(self, tmp_path: Path) -> None:
        """remove() of a non-existent session_id does not touch the backing file."""
        path = tmp_path / "queue.json"
        q = PostSessionQueue(path)
        q.enqueue(_entry("conv-070"))

        mtime_before = path.stat().st_mtime_ns

        # A tiny sleep to ensure mtime would change if a write occurs.
        time.sleep(0.01)
        q.remove("does-not-exist")

        mtime_after = path.stat().st_mtime_ns
        assert mtime_before == mtime_after, "File should not be rewritten for a no-op remove"


# ---------------------------------------------------------------------------
# Test 10 — __len__
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_empty(self, tmp_path: Path) -> None:
        """len() on an empty queue returns 0."""
        q = PostSessionQueue(tmp_path / "queue.json")
        assert len(q) == 0

    def test_len_after_enqueue(self, tmp_path: Path) -> None:
        """len() reflects enqueue count correctly."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-080"))
        q.enqueue(_entry("conv-081"))
        assert len(q) == 2

    def test_len_after_drain(self, tmp_path: Path) -> None:
        """len() is 0 after drain."""
        q = PostSessionQueue(tmp_path / "queue.json")
        q.enqueue(_entry("conv-082"))
        q.drain()
        assert len(q) == 0
