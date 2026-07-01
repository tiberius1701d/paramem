"""Tests for the GET /debug/dump endpoint.

Covers:
- 403 when ``config.debug=False`` (gating contract).
- 503 when ``memory_store`` is not constructed yet.
- Empty store → returns empty list, total=0 (correct read, not an error).
- Happy path: every (tier, key, entry) from iter_entries() flows into the response.
- Tier counts aggregate correctly.
- Per-key ``speaker_id``/``relation_type`` and other bookkeeping fields are sourced
  from ``bookkeeping_for_key`` (authoritative ``_bookkeeping``), not the entry payload.

Tests use FastAPI TestClient with monkeypatched ``_state``; no live server, no GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import paramem.server.app as app_module


def _make_config(tmp_path: Path, debug: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.debug = debug
    cfg.paths.data = tmp_path / "data"
    return cfg


class _FakeStore:
    def __init__(self, items: list[tuple[str, str, dict]], bookkeeping: dict | None = None):
        self._items = items
        self._bookkeeping = bookkeeping or {}

    def iter_entries(self):
        yield from self._items

    def bookkeeping_for_key(self, key: str) -> dict | None:
        """Return the bookkeeping record for *key*, or ``None`` when absent."""
        return self._bookkeeping.get(key)

    def bookkeeping_count(self) -> int:
        return len(self._bookkeeping)


def _make_state(tmp_path: Path, *, debug: bool = True, store_items=None) -> dict:
    state = {"config": _make_config(tmp_path, debug=debug)}
    if store_items is not None:
        state["memory_store"] = _FakeStore(store_items)
    return state


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


class TestDebugDumpGating:
    def test_debug_false_returns_403(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, debug=False, store_items=[])
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 403
        assert resp.json()["status"] == "forbidden_not_debug"

    def test_no_memory_store_returns_503(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, store_items=None)  # store_items=None → key absent
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


class TestDebugDumpHappyPath:
    def test_empty_store_returns_empty_list_200(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, store_items=[])
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"entries": [], "total": 0, "tiers": {}, "bookkeeping_total": 0}

    def test_cache_off_entries_empty_bookkeeping_nonzero(self, tmp_path, monkeypatch):
        """Under preload_cache=False: entries is empty, bookkeeping_total is N."""
        # Bookkeeping present but no content entries (cache-off scenario).
        fake_bk = {
            "k1": {"speaker_id": "alice"},
            "k2": {"speaker_id": "alice"},
        }
        state = _make_state(tmp_path, store_items=[])
        state["memory_store"] = _FakeStore([], bookkeeping=fake_bk)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["bookkeeping_total"] == 2

    def test_dump_flattens_tier_key_entry(self, tmp_path, monkeypatch):
        items = [
            (
                "episodic",
                "graph1",
                {"subject": "Mara", "predicate": "lives_in", "object": "Berlin"},
            ),
            (
                "episodic",
                "graph2",
                {"subject": "Mara", "predicate": "works_at", "object": "Anthropic"},
            ),
            (
                "procedural",
                "proc1",
                {"subject": "Mara", "predicate": "prefers", "object": "concise answers"},
            ),
        ]
        state = _make_state(tmp_path, store_items=items)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 3
        assert body["tiers"] == {"episodic": 2, "procedural": 1}
        assert len(body["entries"]) == 3
        # Each row carries tier + key + the entry payload, flattened.
        first = body["entries"][0]
        assert first["tier"] == "episodic"
        assert first["key"] == "graph1"
        assert first["subject"] == "Mara"
        assert first["object"] == "Berlin"

    def test_bookkeeping_fields_sourced_from_bookkeeping_for_key(self, tmp_path, monkeypatch):
        """speaker_id and relation_type in the dump row come from bookkeeping_for_key,
        not from the entry payload.

        This is the B3c regression guard: the entry payload may carry stale or
        absent bookkeeping fields (store.py:53-58), so the handler must overlay
        the authoritative _bookkeeping values.
        """
        items = [
            (
                "episodic",
                "graph1",
                # Entry payload carries a stale/wrong speaker_id — the overlay
                # must overwrite it with the bookkeeping value.
                {
                    "subject": "Mara",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "stale_value",
                    "relation_type": "stale_type",
                },
            ),
        ]
        bk = {
            "graph1": {
                "speaker_id": "alice",
                "relation_type": "factual",
                "last_reinforced_cycle": 5,
                "reinforcement_count": 2,
                "last_seen": "",
                "first_seen": "",
            }
        }
        state = _make_state(tmp_path, store_items=items)
        state["memory_store"] = _FakeStore(items, bookkeeping=bk)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 1
        row = body["entries"][0]
        # Bookkeeping values must win over the stale entry payload values.
        assert row["speaker_id"] == "alice", "speaker_id must come from bookkeeping_for_key"
        assert row["relation_type"] == "factual", "relation_type must come from bookkeeping_for_key"
        assert row["last_reinforced_cycle"] == 5
        assert row["reinforcement_count"] == 2

    def test_first_seen_sourced_from_bookkeeping_for_key(self, tmp_path, monkeypatch):
        """first_seen in the dump row comes from bookkeeping_for_key, alongside
        last_seen — observability parity for the assertion-window fields.
        """
        items = [
            (
                "episodic",
                "graph1",
                {"subject": "Mara", "predicate": "lives_in", "object": "Berlin"},
            ),
        ]
        bk = {
            "graph1": {
                "speaker_id": "alice",
                "relation_type": "factual",
                "last_reinforced_cycle": 5,
                "reinforcement_count": 2,
                "last_seen": "2026-06-30T12:00:00",
                "first_seen": "2026-06-01T09:00:00",
            }
        }
        state = _make_state(tmp_path, store_items=items)
        state["memory_store"] = _FakeStore(items, bookkeeping=bk)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        row = resp.json()["entries"][0]
        assert row["first_seen"] == "2026-06-01T09:00:00"
        assert row["last_seen"] == "2026-06-30T12:00:00"

    def test_bookkeeping_fields_absent_when_no_bookkeeping_record(self, tmp_path, monkeypatch):
        """When a key has no bookkeeping record, the row omits bookkeeping fields
        rather than carrying stale payload values.
        """
        items = [
            (
                "episodic",
                "graph_no_bk",
                {"subject": "X", "predicate": "p", "object": "Y"},
            ),
        ]
        # No bookkeeping for the key.
        state = _make_state(tmp_path, store_items=items)
        state["memory_store"] = _FakeStore(items, bookkeeping={})
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        row = resp.json()["entries"][0]
        assert "speaker_id" not in row
        assert "relation_type" not in row
        assert "first_seen" not in row
        assert "first_seen_cycle" not in row
