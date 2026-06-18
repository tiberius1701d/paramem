"""Unit tests for paramem.server.run_status — universal per-op-type last-run registry.

Tests cover the full T1b–T5b matrix from the plan:
- T1b round-trip + overwrite-latest: two records for same op_type → only latest survives
- T2b multi-op-type: consolidation + migration both present, keyed by type
- T3b aborted is a valid outcome (S-1): round-trips without error or incident creation
- T4b schema-version guard (C-5): version:2 → RunStatusSchemaError
- T5b restart-survival: fresh read_last_runs sees the persisted record
"""

from __future__ import annotations

import json
import threading

import pytest

from paramem.server.run_status import (
    RunStatusSchemaError,
    read_last_runs,
    record_last_run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    state_dir,
    *,
    op_type="consolidation",
    outcome="trained",
    summary="Trained OK",
    detail=None,
):
    return record_last_run(
        state_dir,
        op_type=op_type,
        outcome=outcome,
        summary=summary,
        detail=detail or {"keys": 10},
    )


# ---------------------------------------------------------------------------
# T1b — round-trip + overwrite-latest
# ---------------------------------------------------------------------------


class TestRoundTripOverwriteLatest:
    def test_round_trip(self, tmp_path):
        """record_last_run → read_last_runs returns one record."""
        _record(tmp_path)
        runs = read_last_runs(tmp_path)
        assert "consolidation" in runs
        rec = runs["consolidation"]
        assert rec.op_type == "consolidation"
        assert rec.outcome == "trained"
        assert rec.summary == "Trained OK"

    def test_absent_returns_empty_dict(self, tmp_path):
        """Empty state_dir → {}."""
        assert read_last_runs(tmp_path) == {}

    def test_overwrite_latest(self, tmp_path):
        """Second record for same op_type replaces the first; no second record created."""
        _record(tmp_path, outcome="trained")
        _record(tmp_path, outcome="noop", summary="Nothing to do")
        runs = read_last_runs(tmp_path)
        assert len(runs) == 1
        rec = runs["consolidation"]
        assert rec.outcome == "noop"
        assert rec.summary == "Nothing to do"

    def test_overwrite_no_count_field(self, tmp_path):
        """RunRecord has no count field — it is purely the latest snapshot."""
        _record(tmp_path)
        _record(tmp_path)
        rec = read_last_runs(tmp_path)["consolidation"]
        # No count attribute
        assert not hasattr(rec, "count")


# ---------------------------------------------------------------------------
# T2b — multi-op-type
# ---------------------------------------------------------------------------


class TestMultiOpType:
    def test_two_op_types_both_present(self, tmp_path):
        """Record consolidation and migration → both keyed in last_runs."""
        _record(tmp_path, op_type="consolidation", outcome="trained")
        _record(tmp_path, op_type="migration", outcome="migration_complete", summary="Done")
        runs = read_last_runs(tmp_path)
        assert "consolidation" in runs
        assert "migration" in runs
        assert runs["consolidation"].outcome == "trained"
        assert runs["migration"].outcome == "migration_complete"

    def test_updating_one_type_leaves_other_intact(self, tmp_path):
        """Overwriting consolidation does not touch the migration record."""
        _record(tmp_path, op_type="consolidation", outcome="trained")
        _record(tmp_path, op_type="migration", outcome="migration_complete", summary="Done")
        _record(tmp_path, op_type="consolidation", outcome="noop", summary="Nothing")
        runs = read_last_runs(tmp_path)
        assert runs["consolidation"].outcome == "noop"
        assert runs["migration"].outcome == "migration_complete"


# ---------------------------------------------------------------------------
# T3b — aborted is a valid outcome (S-1)
# ---------------------------------------------------------------------------


class TestAbortedIsValidOutcome:
    def test_aborted_outcome_round_trips(self, tmp_path):
        """record_last_run with outcome='aborted' succeeds; file contains only run_status."""
        _record(tmp_path, outcome="aborted", summary="Interrupted for inference")
        runs = read_last_runs(tmp_path)
        assert runs["consolidation"].outcome == "aborted"

    def test_aborted_does_not_create_incidents_file(self, tmp_path):
        """record_last_run does NOT touch incidents.json (aborted is NOT an incident)."""
        _record(tmp_path, outcome="aborted")
        incidents_file = tmp_path / "incidents.json"
        assert not incidents_file.exists(), "aborted outcome must not create incidents.json"


# ---------------------------------------------------------------------------
# T4b — schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_future_version_raises_schema_error(self, tmp_path):
        """Hand-written version:2 → RunStatusSchemaError from read_last_runs."""
        (tmp_path / "run_status.json").write_text(
            json.dumps({"version": 2, "last_runs": {}}),
            encoding="utf-8",
        )
        with pytest.raises(RunStatusSchemaError, match="version"):
            read_last_runs(tmp_path)

    def test_missing_version_raises_schema_error(self, tmp_path):
        """Missing version field → RunStatusSchemaError."""
        (tmp_path / "run_status.json").write_text(
            json.dumps({"last_runs": {}}),
            encoding="utf-8",
        )
        with pytest.raises(RunStatusSchemaError):
            read_last_runs(tmp_path)

    def test_malformed_json_raises_schema_error(self, tmp_path):
        """Corrupt file → RunStatusSchemaError."""
        (tmp_path / "run_status.json").write_text("NOT JSON {{{{", encoding="utf-8")
        with pytest.raises(RunStatusSchemaError, match="not valid JSON"):
            read_last_runs(tmp_path)


# ---------------------------------------------------------------------------
# T5b — restart-survival
# ---------------------------------------------------------------------------


class TestRestartSurvival:
    def test_record_persists_across_new_reader(self, tmp_path):
        """Write via record_last_run; fresh read_last_runs call sees the record."""
        _record(tmp_path, outcome="trained", summary="From disk")

        # Simulate process restart: call read_last_runs fresh (no in-memory state).
        runs = read_last_runs(tmp_path)
        assert "consolidation" in runs
        assert runs["consolidation"].outcome == "trained"
        assert runs["consolidation"].summary == "From disk"


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_no_pending_residue_after_record(self, tmp_path):
        """After record_last_run, no .pending/run_status.json remains."""
        _record(tmp_path)
        pending = tmp_path / ".pending" / "run_status.json"
        assert not pending.exists(), ".pending residue must be removed after atomic rename"
        assert (tmp_path / "run_status.json").exists()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_record_last_run_serializes(self, tmp_path):
        """10 threads × 10 ops each — final record is valid and readable."""
        N_THREADS = 10
        N_OPS = 10
        barrier = threading.Barrier(N_THREADS)
        errors: list[Exception] = []
        written: list[str] = []
        lock = threading.Lock()

        def _worker(thread_idx: int) -> None:
            barrier.wait()
            for op_idx in range(N_OPS):
                summary = f"t{thread_idx}:op{op_idx}"
                try:
                    record_last_run(
                        tmp_path,
                        op_type="consolidation",
                        outcome="trained",
                        summary=summary,
                        detail={"t": thread_idx, "op": op_idx},
                    )
                    with lock:
                        written.append(summary)
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Concurrent writes raised: {errors}"
        runs = read_last_runs(tmp_path)
        assert "consolidation" in runs
        rec = runs["consolidation"]
        assert rec.op_type == "consolidation"
        # The final record must be one of the 100 written summaries.
        assert rec.summary in written
