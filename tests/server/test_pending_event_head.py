"""``_pending_event_head`` — the pending event's head, read straight off disk.

Absent, readable, and unreadable (undecodable / unsupported version /
not-a-ledger) shapes. CPU-only, no model load.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from tests.server._state_builders import _write_pending_ledger


def _config(tmp_path):
    config = MagicMock()
    config.paths.data = tmp_path
    return config


def _state_dir(tmp_path):
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ledger_path(tmp_path):
    return _state_dir(tmp_path) / "stage_ledger.json"


def test_absent_ledger_reports_nothing_pending(tmp_path):
    import paramem.server.app as app_module

    assert app_module._pending_event_head(_config(tmp_path)) is None


def test_readable_ledger_reports_kind_and_since_epoch(tmp_path):
    import paramem.server.app as app_module

    _write_pending_ledger(tmp_path, event="interim")

    head = app_module._pending_event_head(_config(tmp_path))

    assert head is not None
    assert head.kind == "interim"
    assert head.readable is True
    assert head.cause is None
    # _write_pending_ledger stamps completed_at="2026-01-01T00:00:00+00:00".
    from datetime import datetime

    expected = datetime.fromisoformat("2026-01-01T00:00:00+00:00").timestamp()
    assert head.since_epoch == expected


def test_undecodable_ledger_reports_unreadable_with_cause(tmp_path, caplog):
    import paramem.server.app as app_module

    ledger_path = _ledger_path(tmp_path)
    ledger_path.write_bytes(b"not json at all {{{")

    with caplog.at_level(logging.WARNING, logger="paramem.training.stage_ledger"):
        head = app_module._pending_event_head(_config(tmp_path))

    assert head is not None
    assert head.kind is None
    assert head.readable is False
    assert head.since_epoch is None
    assert head.cause == "undecodable"
    assert any(
        record.name == "paramem.training.stage_ledger" and str(ledger_path) in record.getMessage()
        for record in caplog.records
    ), "expected the reader's one WARNING naming the ledger path"


def test_unsupported_version_ledger_reports_unreadable_naming_the_version(tmp_path, caplog):
    import paramem.server.app as app_module

    ledger_path = _ledger_path(tmp_path)
    ledger_path.write_text(json.dumps({"version": 1}))

    with caplog.at_level(logging.WARNING, logger="paramem.training.stage_ledger"):
        head = app_module._pending_event_head(_config(tmp_path))

    assert head is not None
    assert head.readable is False
    assert head.cause == "unsupported version 1"
    assert any(
        record.name == "paramem.training.stage_ledger" and str(ledger_path) in record.getMessage()
        for record in caplog.records
    ), "expected the reader's one WARNING naming the ledger path"


def test_v2_payload_missing_a_head_field_reports_not_a_ledger(tmp_path, caplog):
    import paramem.server.app as app_module

    ledger_path = _ledger_path(tmp_path)
    # A valid, current schema_version but missing every other required field
    # (event, venue, stamp) -- _from_dict's KeyError becomes not_a_ledger.
    ledger_path.write_text(json.dumps({"version": 2}))

    with caplog.at_level(logging.WARNING, logger="paramem.training.stage_ledger"):
        head = app_module._pending_event_head(_config(tmp_path))

    assert head is not None
    assert head.readable is False
    assert head.cause == "not_a_ledger"
    assert any(
        record.name == "paramem.training.stage_ledger" and str(ledger_path) in record.getMessage()
        for record in caplog.records
    ), "expected the reader's one WARNING naming the ledger path"
