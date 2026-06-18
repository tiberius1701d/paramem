"""Unit tests for paramem.server.atomic_json — shared atomic-JSON I/O primitive.

Tests cover:
- atomic_write_json: round-trip + no .pending residue after write
- read_json_or_none: absent → None; bad JSON → JSONDecodeError
- flock_rmw: round-trip + concurrent serialisation (10-thread mirror of test_state.py)
"""

from __future__ import annotations

import json
import threading

import pytest

from paramem.server.atomic_json import atomic_write_json, flock_rmw, read_json_or_none


class TestAtomicWriteJson:
    def test_round_trip(self, tmp_path):
        """Write a dict and read it back — fields preserved."""
        payload = {"version": 1, "data": "hello"}
        path = atomic_write_json(tmp_path, "test.json", payload)
        assert path == tmp_path / "test.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw == payload

    def test_no_pending_residue(self, tmp_path):
        """After successful write, no .pending/<filename> remains."""
        atomic_write_json(tmp_path, "test.json", {"x": 1})
        pending = tmp_path / ".pending" / "test.json"
        assert not pending.exists(), ".pending residue must be removed after atomic rename"
        assert (tmp_path / "test.json").exists()

    def test_creates_state_dir(self, tmp_path):
        """state_dir is created when it does not exist."""
        state_dir = tmp_path / "nested" / "dir"
        assert not state_dir.exists()
        atomic_write_json(state_dir, "test.json", {"v": 1})
        assert (state_dir / "test.json").exists()

    def test_overwrite(self, tmp_path):
        """Second write replaces the first."""
        atomic_write_json(tmp_path, "f.json", {"v": 1})
        atomic_write_json(tmp_path, "f.json", {"v": 2})
        raw = json.loads((tmp_path / "f.json").read_text())
        assert raw == {"v": 2}


class TestReadJsonOrNone:
    def test_absent_returns_none(self, tmp_path):
        """File does not exist → None."""
        assert read_json_or_none(tmp_path, "missing.json") is None

    def test_round_trip(self, tmp_path):
        """Write then read returns the original dict."""
        payload = {"a": 1, "b": [1, 2, 3]}
        (tmp_path / "f.json").write_text(json.dumps(payload), encoding="utf-8")
        result = read_json_or_none(tmp_path, "f.json")
        assert result == payload

    def test_bad_json_raises_json_decode_error(self, tmp_path):
        """Malformed file → json.JSONDecodeError (domain caller wraps this)."""
        (tmp_path / "f.json").write_text("NOT JSON {{{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            read_json_or_none(tmp_path, "f.json")


class TestFlockRmw:
    def test_round_trip(self, tmp_path):
        """Mutate from None (absent) → written dict is returned and persisted."""

        def _mutate(current):
            assert current is None
            return {"version": 1, "count": 0}

        result = flock_rmw(tmp_path, "test.lock", _mutate, "test.json")
        assert result == {"version": 1, "count": 0}
        raw = read_json_or_none(tmp_path, "test.json")
        assert raw == {"version": 1, "count": 0}

    def test_existing_file_passed_to_mutate(self, tmp_path):
        """Second RMW receives the previously written dict."""
        flock_rmw(tmp_path, "test.lock", lambda _: {"v": 1}, "test.json")
        seen: list[dict | None] = []

        def _mutate(current):
            seen.append(current)
            return {"v": 2}

        flock_rmw(tmp_path, "test.lock", _mutate, "test.json")
        assert seen[0] == {"v": 1}
        assert read_json_or_none(tmp_path, "test.json") == {"v": 2}

    def test_concurrent_serialization(self, tmp_path):
        """10 threads × 10 ops each — no corruption; final state is valid JSON.

        Mirrors tests/backup/test_state.py::TestUpdateBackupStateConcurrency.
        """
        N_THREADS = 10
        N_OPS = 10
        barrier = threading.Barrier(N_THREADS)
        errors: list[Exception] = []

        def _worker(thread_idx: int) -> None:
            barrier.wait()
            for op_idx in range(N_OPS):
                try:

                    def _mutate(current, _t=thread_idx, _o=op_idx):
                        count = 0 if current is None else current.get("count", 0)
                        return {"version": 1, "count": count + 1, "last": f"{_t}:{_o}"}

                    flock_rmw(tmp_path, "test.lock", _mutate, "test.json")
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Concurrent RMW raised: {errors}"
        raw = read_json_or_none(tmp_path, "test.json")
        assert raw is not None
        assert raw["version"] == 1
        assert raw["count"] == N_THREADS * N_OPS
