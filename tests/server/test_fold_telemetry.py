"""Unit tests for paramem.server.fold_telemetry — bounded fold telemetry ring.

Mirrors tests/server/test_incidents.py's flock_rmw-based test patterns.

Covers:
- Round-trip: one write → one cycle, one record.
- Upsert: two records with the same cycle_stamp append to one cycle entry.
- Distinct cycles: different cycle_stamp values create separate entries.
- Ring bound: writing more cycles than max_cycles evicts the oldest,
  retains the newest, and settles at exactly max_cycles.
- Plaintext JSON: the artifact is not an age blob (readable via json.loads).
- Concurrency: concurrent writers serialize (no corruption, no lost writes).
- Debug-guard: the artifact is written identically regardless of any
  caller-side ``debug`` posture — this module takes no ``debug`` parameter.
- Caller convention: ``cycle_stamp`` is opaque to this module — a content
  fingerprint carried as a ``record`` field does not collapse two distinct
  runs into one cycle entry when the caller keys on run identity instead.
"""

from __future__ import annotations

import json
import threading

import pytest

from paramem.server.fold_telemetry import (
    TELEMETRY_FILENAME,
    record_fold_telemetry,
)

# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_single_write_creates_one_cycle_one_record(self, tmp_path):
        record_fold_telemetry(
            tmp_path,
            cycle_stamp="2026-07-10T00:00:00+00:00",
            kind="backup_creation",
            record={"free_before": 100, "free_after": 90, "adapter_count": 3, "interim_count": 0},
        )
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        assert data["version"] == 1
        assert len(data["cycles"]) == 1
        cycle = data["cycles"][0]
        assert cycle["cycle_stamp"] == "2026-07-10T00:00:00+00:00"
        assert len(cycle["records"]) == 1
        rec = cycle["records"][0]
        assert rec["kind"] == "backup_creation"
        assert rec["free_before"] == 100
        assert rec["free_after"] == 90
        assert rec["adapter_count"] == 3
        assert rec["interim_count"] == 0
        assert "recorded_at" in rec

    def test_absent_file_then_write_creates_it(self, tmp_path):
        assert not (tmp_path / TELEMETRY_FILENAME).exists()
        record_fold_telemetry(
            tmp_path, cycle_stamp="c1", kind="backup_creation", record={"free_before": 1}
        )
        assert (tmp_path / TELEMETRY_FILENAME).exists()


# ---------------------------------------------------------------------------
# Upsert within a cycle
# ---------------------------------------------------------------------------


class TestUpsertWithinCycle:
    def test_same_cycle_stamp_appends_to_one_entry(self, tmp_path):
        record_fold_telemetry(
            tmp_path, cycle_stamp="c1", kind="backup_creation", record={"free_before": 1}
        )
        record_fold_telemetry(
            tmp_path,
            cycle_stamp="c1",
            kind="tier_train",
            record={"tier": "episodic", "free_before": 2},
        )
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        assert len(data["cycles"]) == 1
        cycle = data["cycles"][0]
        assert cycle["cycle_stamp"] == "c1"
        assert len(cycle["records"]) == 2
        assert [r["kind"] for r in cycle["records"]] == ["backup_creation", "tier_train"]

    def test_different_cycle_stamps_create_two_entries(self, tmp_path):
        record_fold_telemetry(
            tmp_path, cycle_stamp="c1", kind="backup_creation", record={"free_before": 1}
        )
        record_fold_telemetry(
            tmp_path, cycle_stamp="c2", kind="backup_creation", record={"free_before": 2}
        )
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        assert len(data["cycles"]) == 2
        stamps = {c["cycle_stamp"] for c in data["cycles"]}
        assert stamps == {"c1", "c2"}

    def test_same_content_fingerprint_field_does_not_collapse_distinct_runs(self, tmp_path):
        """Two distinct RUNS over an unchanged dataset carry the SAME
        content-fingerprint field but MUST key on distinct run identifiers
        (``cycle_stamp``) — the fingerprint belongs inside ``record``, never
        as the ring key, or two unrelated runs collapse into one growing
        cycle entry.
        """
        record_fold_telemetry(
            tmp_path,
            cycle_stamp="run-2026-07-10T00:00:00.000001Z",
            kind="backup_creation",
            record={"fold_stamp": "sha256:unchanged-keyset", "free_before": 1},
        )
        record_fold_telemetry(
            tmp_path,
            cycle_stamp="run-2026-07-10T12:00:00.000002Z",
            kind="backup_creation",
            record={"fold_stamp": "sha256:unchanged-keyset", "free_before": 2},
        )
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        assert len(data["cycles"]) == 2
        for cycle in data["cycles"]:
            assert len(cycle["records"]) == 1
            assert cycle["records"][0]["fold_stamp"] == "sha256:unchanged-keyset"


# ---------------------------------------------------------------------------
# Ring bound
# ---------------------------------------------------------------------------


class TestRingBound:
    def test_n_plus_50_cycles_into_ring_of_n_leaves_exactly_n(self, tmp_path):
        max_cycles = 10
        n_writes = max_cycles + 50
        for i in range(n_writes):
            record_fold_telemetry(
                tmp_path,
                cycle_stamp=f"c{i:04d}",
                kind="backup_creation",
                record={"free_before": i},
                max_cycles=max_cycles,
            )
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        assert len(data["cycles"]) == max_cycles

    def test_oldest_evicted_newest_retained(self, tmp_path):
        max_cycles = 5
        n_writes = max_cycles + 3
        for i in range(n_writes):
            record_fold_telemetry(
                tmp_path,
                cycle_stamp=f"c{i:04d}",
                kind="backup_creation",
                record={"free_before": i},
                max_cycles=max_cycles,
            )
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        stamps = [c["cycle_stamp"] for c in data["cycles"]]
        # Oldest 3 (c0000-c0002) evicted; newest 5 (c0003-c0007) retained, in order.
        assert stamps == ["c0003", "c0004", "c0005", "c0006", "c0007"]


# ---------------------------------------------------------------------------
# Plaintext JSON (not an age blob)
# ---------------------------------------------------------------------------


class TestPlaintextJson:
    def test_artifact_is_valid_plaintext_json(self, tmp_path):
        record_fold_telemetry(
            tmp_path, cycle_stamp="c1", kind="backup_creation", record={"free_before": 1}
        )
        raw = (tmp_path / TELEMETRY_FILENAME).read_bytes()
        # An age envelope starts with the "age-encryption.org/v1" armor header;
        # plaintext JSON starts with "{".
        assert raw.lstrip().startswith(b"{")
        parsed = json.loads(raw.decode("utf-8"))
        assert isinstance(parsed, dict)

    def test_no_pending_residue_after_write(self, tmp_path):
        record_fold_telemetry(
            tmp_path, cycle_stamp="c1", kind="backup_creation", record={"free_before": 1}
        )
        pending = tmp_path / ".pending" / TELEMETRY_FILENAME
        assert not pending.exists()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_writers_serialize(self, tmp_path):
        n_threads = 10
        n_ops = 5
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def _worker(thread_idx: int) -> None:
            barrier.wait()
            for op_idx in range(n_ops):
                try:
                    record_fold_telemetry(
                        tmp_path,
                        cycle_stamp=f"t{thread_idx}",
                        kind="backup_creation",
                        record={"free_before": op_idx},
                    )
                except Exception as exc:  # noqa: BLE001 — collected, not swallowed
                    errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Concurrent writes raised: {errors}"
        data = json.loads((tmp_path / TELEMETRY_FILENAME).read_text())
        # 10 distinct cycle_stamps, 5 records each — no corruption, no lost writes.
        assert len(data["cycles"]) == n_threads
        for cycle in data["cycles"]:
            assert len(cycle["records"]) == n_ops


# ---------------------------------------------------------------------------
# Debug-guard: always written regardless of a caller-side debug posture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("debug", [True, False])
def test_record_fold_telemetry_writes_regardless_of_debug_flag(tmp_path, debug):
    """Telemetry is instrumentation, not the privacy/debug-artifact switch;
    ``debug`` must never gate the write. ``record_fold_telemetry`` takes no
    ``debug`` parameter at all — this guards against someone re-gating
    telemetry on it at a future call site by proving the artifact appears
    identically for both values of a caller-side ``debug`` flag.
    """
    telemetry_dir = tmp_path / f"telemetry_debug_{debug}"
    record_fold_telemetry(
        telemetry_dir,
        cycle_stamp="2026-07-10T00:00:00+00:00",
        kind="backup_creation",
        record={"free_before": 100, "free_after": 90, "adapter_count": 3, "interim_count": 0},
    )
    assert (telemetry_dir / TELEMETRY_FILENAME).exists()
