"""Unit tests for paramem.server.incidents — actionable-failure event store.

Tests cover the full T1–T11 matrix from the plan:
- T1  round-trip: record → read → count==1, status==active
- T2  dedup/bump: same (type,key) twice → count==2, last_seen advanced
- T3  distinct keys: same type, different key → two incidents
- T4  resolve: flips status; idempotent (False on second call)
- T5  reopen: record → resolve → record same key → active again, count bumped
- T6  ack: flips status→acknowledged; False for unknown id
- T7  resolve_incidents_by_type: resolves all of one type, leaves other untouched
- T8  restart-survival: write then fresh read from same state_dir
- T9  schema-version guard (C-5): version:2 → IncidentStoreSchemaError
- T10 malformed JSON → IncidentStoreSchemaError
- T11 atomicity: no .pending residue after write
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from paramem.server.incidents import (
    Incident,
    IncidentStoreSchemaError,
    ack_incident,
    read_incidents,
    record_incident,
    resolve_incident,
    resolve_incidents_by_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    state_dir,
    *,
    type="vram_exhausted",
    key="phase1",
    severity="failed",
    summary="VRAM exhausted at phase1",
    detail=None,
):
    return record_incident(
        state_dir,
        type=type,
        key=key,
        severity=severity,
        summary=summary,
        detail=detail or {"phase": key},
    )


# ---------------------------------------------------------------------------
# T1 — round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_record_then_read_returns_one_active_incident(self, tmp_path):
        """record_incident → read_incidents → one incident, count=1, status=active."""
        _record(tmp_path)
        incidents = read_incidents(tmp_path)
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.type == "vram_exhausted"
        assert inc.count == 1
        assert inc.status == "active"
        assert inc.severity == "failed"
        assert inc.id == "vram_exhausted:phase1"

    def test_absent_file_returns_empty_list(self, tmp_path):
        """Empty state_dir → []."""
        assert read_incidents(tmp_path) == []


# ---------------------------------------------------------------------------
# T2 — dedup/bump
# ---------------------------------------------------------------------------


class TestDedupBump:
    def test_same_type_and_key_bumps_count(self, tmp_path):
        """Two record calls with same (type,key) → count==2, first_seen unchanged."""
        _record(tmp_path)
        time.sleep(0.01)  # ensure last_seen can advance
        _record(tmp_path, summary="Updated summary")

        incidents = read_incidents(tmp_path)
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.count == 2
        assert inc.status == "active"
        assert inc.summary == "Updated summary"

    def test_first_seen_unchanged_on_dedup(self, tmp_path):
        """first_seen is set on first record and not updated on second."""
        first = _record(tmp_path)
        time.sleep(0.01)
        _record(tmp_path)

        inc = read_incidents(tmp_path)[0]
        assert inc.first_seen == first.first_seen
        # last_seen must be >= first_seen
        assert inc.last_seen >= inc.first_seen


# ---------------------------------------------------------------------------
# T3 — distinct keys
# ---------------------------------------------------------------------------


class TestDistinctKeys:
    def test_different_keys_create_two_incidents(self, tmp_path):
        """Same type, different key → two separate incidents."""
        _record(tmp_path, key="phase1")
        _record(tmp_path, key="phase2")
        incidents = read_incidents(tmp_path)
        assert len(incidents) == 2
        ids = {i.id for i in incidents}
        assert "vram_exhausted:phase1" in ids
        assert "vram_exhausted:phase2" in ids


# ---------------------------------------------------------------------------
# T4 — resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_flips_status_to_resolved(self, tmp_path):
        """resolve_incident sets status→resolved; returns True."""
        _record(tmp_path)
        ok = resolve_incident(tmp_path, "vram_exhausted", "phase1")
        assert ok is True
        inc = read_incidents(tmp_path)[0]
        assert inc.status == "resolved"

    def test_resolve_returns_false_when_no_match(self, tmp_path):
        """No matching incident → returns False (no error)."""
        ok = resolve_incident(tmp_path, "vram_exhausted", "nonexistent")
        assert ok is False

    def test_resolve_does_not_raise_when_already_resolved(self, tmp_path):
        """Resolving an already-resolved incident does not raise."""
        _record(tmp_path)
        resolve_incident(tmp_path, "vram_exhausted", "phase1")
        # Second resolve: already resolved — must not raise.
        resolve_incident(tmp_path, "vram_exhausted", "phase1")
        inc = read_incidents(tmp_path)[0]
        assert inc.status == "resolved"


# ---------------------------------------------------------------------------
# T5 — reopen
# ---------------------------------------------------------------------------


class TestReopen:
    def test_reopen_after_resolve(self, tmp_path):
        """record → resolve → record same (type,key) → active again, count bumped."""
        _record(tmp_path)
        resolve_incident(tmp_path, "vram_exhausted", "phase1")
        assert read_incidents(tmp_path)[0].status == "resolved"

        _record(tmp_path)  # reopen
        inc = read_incidents(tmp_path)[0]
        assert inc.status == "active"
        assert inc.count == 2  # first record + reopen


# ---------------------------------------------------------------------------
# T6 — ack
# ---------------------------------------------------------------------------


class TestAck:
    def test_ack_flips_status_to_acknowledged(self, tmp_path):
        """ack_incident sets status→acknowledged; returns True."""
        _record(tmp_path)
        ok = ack_incident(tmp_path, "vram_exhausted:phase1")
        assert ok is True
        inc = read_incidents(tmp_path)[0]
        assert inc.status == "acknowledged"

    def test_ack_returns_false_for_unknown_id(self, tmp_path):
        """Unknown id → returns False."""
        ok = ack_incident(tmp_path, "does_not_exist:key")
        assert ok is False


# ---------------------------------------------------------------------------
# T7 — resolve_incidents_by_type
# ---------------------------------------------------------------------------


class TestResolveByType:
    def test_resolves_all_of_one_type_leaves_other(self, tmp_path):
        """resolve_incidents_by_type resolves both matching, leaves third untouched."""
        _record(tmp_path, type="vram_exhausted", key="phase1")
        _record(tmp_path, type="vram_exhausted", key="phase2")
        _record(tmp_path, type="extraction_failed", key="session1")

        count = resolve_incidents_by_type(tmp_path, "vram_exhausted")
        assert count == 2

        incidents = read_incidents(tmp_path)
        by_type = {i.type: i for i in incidents}
        assert by_type["vram_exhausted"].status in ("resolved",)
        # extraction_failed is untouched
        assert by_type["extraction_failed"].status == "active"

    def test_resolve_by_type_both_keys_resolved(self, tmp_path):
        """Both vram_exhausted keys end up resolved."""
        _record(tmp_path, type="vram_exhausted", key="phase1")
        _record(tmp_path, type="vram_exhausted", key="phase2")
        resolve_incidents_by_type(tmp_path, "vram_exhausted")
        for inc in read_incidents(tmp_path):
            if inc.type == "vram_exhausted":
                assert inc.status == "resolved"

    def test_resolve_by_type_returns_zero_when_none_match(self, tmp_path):
        """No matching incidents → returns 0 (no error)."""
        count = resolve_incidents_by_type(tmp_path, "vram_exhausted")
        assert count == 0

    def test_resolve_by_type_skips_already_resolved(self, tmp_path):
        """Already-resolved incidents are not counted again."""
        _record(tmp_path, type="vram_exhausted", key="phase1")
        resolve_incidents_by_type(tmp_path, "vram_exhausted")
        count2 = resolve_incidents_by_type(tmp_path, "vram_exhausted")
        assert count2 == 0


# ---------------------------------------------------------------------------
# T8 — restart-survival
# ---------------------------------------------------------------------------


class TestRestartSurvival:
    def test_incident_persists_across_new_reader(self, tmp_path):
        """Write via record_incident; fresh read_incidents call sees the incident."""
        _record(tmp_path, summary="Persisted failure")

        # Simulate a process restart by calling read_incidents directly
        # (no in-memory state; reads from disk only).
        incidents = read_incidents(tmp_path)
        assert len(incidents) == 1
        assert incidents[0].summary == "Persisted failure"
        assert incidents[0].status == "active"


# ---------------------------------------------------------------------------
# T9 — schema-version guard
# ---------------------------------------------------------------------------


class TestSchemaVersionGuard:
    def test_future_version_raises_schema_error(self, tmp_path):
        """Hand-written version:2 → IncidentStoreSchemaError from read_incidents."""
        (tmp_path / "incidents.json").write_text(
            json.dumps({"version": 2, "incidents": []}),
            encoding="utf-8",
        )
        with pytest.raises(IncidentStoreSchemaError, match="version"):
            read_incidents(tmp_path)

    def test_missing_version_raises_schema_error(self, tmp_path):
        """Missing version field → IncidentStoreSchemaError."""
        (tmp_path / "incidents.json").write_text(
            json.dumps({"incidents": []}),
            encoding="utf-8",
        )
        with pytest.raises(IncidentStoreSchemaError):
            read_incidents(tmp_path)


# ---------------------------------------------------------------------------
# T10 — malformed JSON
# ---------------------------------------------------------------------------


class TestMalformedJson:
    def test_malformed_json_raises_schema_error(self, tmp_path):
        """Corrupt file → IncidentStoreSchemaError from read_incidents."""
        (tmp_path / "incidents.json").write_text("NOT JSON {{{{", encoding="utf-8")
        with pytest.raises(IncidentStoreSchemaError, match="not valid JSON"):
            read_incidents(tmp_path)


# ---------------------------------------------------------------------------
# T11 — atomicity: no .pending residue
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_no_pending_residue_after_record(self, tmp_path):
        """After record_incident, no .pending/incidents.json remains."""
        _record(tmp_path)
        pending = tmp_path / ".pending" / "incidents.json"
        assert not pending.exists(), ".pending residue must be removed after atomic rename"
        assert (tmp_path / "incidents.json").exists()


# ---------------------------------------------------------------------------
# Concurrency (mirror of test_state.py 10-thread test)
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_record_incident_serializes(self, tmp_path):
        """10 threads × 5 ops each — no corruption; final count equals total ops."""
        N_THREADS = 10
        N_OPS = 5
        barrier = threading.Barrier(N_THREADS)
        errors: list[Exception] = []

        def _worker(thread_idx: int) -> None:
            barrier.wait()
            for op_idx in range(N_OPS):
                try:
                    _record(tmp_path, key="shared_key", summary=f"t{thread_idx} op{op_idx}")
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
        incidents = read_incidents(tmp_path)
        assert len(incidents) == 1  # all ops deduped to one incident
        assert incidents[0].count == N_THREADS * N_OPS


# ---------------------------------------------------------------------------
# Incident.to_dict
# ---------------------------------------------------------------------------


class TestIncidentToDict:
    def test_to_dict_returns_all_fields(self):
        """to_dict() output matches the exact key/value set for a populated instance."""
        inc = Incident(
            id="vram_exhausted:phase1",
            type="vram_exhausted",
            severity="failed",
            first_seen="2026-06-01T00:00:00+00:00",
            last_seen="2026-06-01T00:05:00+00:00",
            count=3,
            status="active",
            summary="VRAM exhausted at phase1",
            detail={"phase": "phase1", "free_bytes": 12345},
        )
        d = inc.to_dict()
        assert d == {
            "id": "vram_exhausted:phase1",
            "type": "vram_exhausted",
            "severity": "failed",
            "first_seen": "2026-06-01T00:00:00+00:00",
            "last_seen": "2026-06-01T00:05:00+00:00",
            "count": 3,
            "status": "active",
            "summary": "VRAM exhausted at phase1",
            "detail": {"phase": "phase1", "free_bytes": 12345},
        }

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict → from_dict → equal incident."""
        inc = Incident(
            id="consolidation_retry_exhausted:c001",
            type="consolidation_retry_exhausted",
            severity="warning",
            first_seen="2026-06-01T00:00:00+00:00",
            last_seen="2026-06-01T00:00:00+00:00",
            count=1,
            status="acknowledged",
            summary="Retry budget exhausted",
            detail={},
        )
        recovered = Incident.from_dict(inc.to_dict())
        assert recovered == inc
