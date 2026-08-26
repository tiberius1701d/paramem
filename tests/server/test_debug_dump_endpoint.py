"""Tests for the GET /debug/dump endpoint.

Covers:
- 403 when ``config.debug=False`` (gating contract).
- 503 when ``memory_store`` is not constructed yet.
- Empty store → returns empty list, total=0 (correct read, not an error).
- Happy path: every (tier, key, entry) from iter_entries() flows into the response.
- Tier counts aggregate correctly.
- Every registry-known key carries a full bookkeeping row by invariant
  (``paramem.memory.store.MemoryStore``'s every-known-key-has-a-row
  invariant) — the dump splats that row onto its row (``debug_dump``,
  ``paramem/server/app.py``) directly, with no ``None``-tolerant default.
  ``speaker_id``/``relation_type`` and every other bookkeeping field come
  from ``bookkeeping_for_key`` (authoritative ``_bookkeeping``), overlaid
  onto — and winning over — whatever the entry payload itself carries.

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


def _default_bookkeeping_row(**overrides) -> dict:
    """The canonical full bookkeeping row every registry-known key carries
    by invariant (``paramem.memory.store.MemoryStore.bookkeeping_for_key``).
    Tests that don't care about specific field values get this row
    unmodified via ``_FakeStore``'s default; tests exercising the
    bookkeeping-overlay behavior pass their own dict (or overrides here)."""
    row = {
        "speaker_id": "speaker0",
        "relation_type": "factual",
        "reinforcement_count": 0,
        "last_reinforced_cycle": 0,
        "last_seen": "",
        "first_seen": "",
        "promoted": False,
    }
    row.update(overrides)
    return row


class _FakeStore:
    """Minimal ``MemoryStore`` stand-in for ``/debug/dump`` tests.

    Every entry this fixture serves via ``iter_entries()`` carries a
    bookkeeping row by default — matching the every-known-key-has-a-row
    invariant every registry-known key carries in production
    (``paramem.memory.store.MemoryStore``). A key with content but no row
    is a fixture bug, not a state ``/debug/dump`` needs to tolerate: the
    handler reads ``bookkeeping_for_key`` directly and splats it onto the
    row, with no ``None``-tolerant default. Pass an explicit
    ``bookkeeping`` dict to exercise the overlay behavior (bookkeeping
    values winning over stale entry-payload values) or the cache-off shape
    (bookkeeping present, no content entries).
    """

    def __init__(self, items: list[tuple[str, str, dict]], bookkeeping: dict | None = None):
        self._items = items
        if bookkeeping is None:
            self._bookkeeping = {key: _default_bookkeeping_row() for _tier, key, _entry in items}
        else:
            self._bookkeeping = bookkeeping

    def iter_entries(self):
        yield from self._items

    def bookkeeping_for_key(self, key: str) -> dict | None:
        """Return the bookkeeping record for *key* — every key this
        fixture serves via ``iter_entries()`` carries one by construction."""
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
            "k1": _default_bookkeeping_row(speaker_id="alice"),
            "k2": _default_bookkeeping_row(speaker_id="alice"),
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
        """Every (tier, key, entry) from iter_entries() flows into the
        response, flattened, with its (fixture-default) bookkeeping row
        overlaid — the every-known-key-has-a-row invariant means a cached
        entry is never dumped without one."""
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

        The entry payload may carry stale bookkeeping-shaped fields, so the
        handler must overlay the authoritative _bookkeeping values — read
        directly, with no ``None``-tolerant default, since the
        every-known-key-has-a-row invariant guarantees a row exists.
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
            "graph1": _default_bookkeeping_row(
                speaker_id="alice",
                relation_type="factual",
                last_reinforced_cycle=5,
                reinforcement_count=2,
            )
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
            "graph1": _default_bookkeeping_row(
                speaker_id="alice",
                relation_type="factual",
                last_reinforced_cycle=5,
                reinforcement_count=2,
                last_seen="2026-06-30T12:00:00",
                first_seen="2026-06-01T09:00:00",
            )
        }
        state = _make_state(tmp_path, store_items=items)
        state["memory_store"] = _FakeStore(items, bookkeeping=bk)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        row = resp.json()["entries"][0]
        assert row["first_seen"] == "2026-06-01T09:00:00"
        assert row["last_seen"] == "2026-06-30T12:00:00"

    def test_full_bookkeeping_row_present_in_the_dump(self, tmp_path, monkeypatch):
        """Every registry-known key carries a full seven-field bookkeeping
        row by invariant (``paramem.memory.store.MemoryStore``) — the
        handler splats the WHOLE record onto the row (not a hand-maintained
        field subset), so every field surfaces, including ``promoted``
        (never exercised by the other happy-path tests here).
        """
        items = [
            (
                "episodic",
                "graph1",
                {"subject": "X", "predicate": "p", "object": "Y"},
            ),
        ]
        bk = {
            "graph1": _default_bookkeeping_row(
                speaker_id="alice",
                relation_type="factual",
                reinforcement_count=2,
                last_reinforced_cycle=5,
                last_seen="2026-06-30T12:00:00",
                first_seen="2026-06-01T09:00:00",
                promoted=True,
            )
        }
        state = _make_state(tmp_path, store_items=items)
        state["memory_store"] = _FakeStore(items, bookkeeping=bk)
        client = _make_client(monkeypatch, state)
        resp = client.get("/debug/dump")
        assert resp.status_code == 200, resp.text
        row = resp.json()["entries"][0]
        for field in (
            "speaker_id",
            "relation_type",
            "reinforcement_count",
            "last_reinforced_cycle",
            "last_seen",
            "first_seen",
            "promoted",
        ):
            assert field in row, f"{field} must be present on every registry-known key's row"
        assert row["promoted"] is True
