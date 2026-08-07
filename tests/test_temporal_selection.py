"""Tests for the date-group selection stage.

No GPU, no model load: the generate seam (``temporal_selection._generate``)
is monkeypatched with a plain stub returning a fixed string, matching
:func:`paramem.models.loader.generate_adapter_off`'s ``str`` return
contract.
"""

from __future__ import annotations

from datetime import date

import pytest

from paramem.server.config import ServerConfig
from paramem.server.temporal import DateWindow
from paramem.server.temporal_selection import DateSelection, select_date_groups


@pytest.fixture
def config() -> ServerConfig:
    return ServerConfig()


def _stub_generate(response: str):
    """Build a stand-in for ``generate_adapter_off`` returning *response*
    verbatim, ignoring every argument (matches the seam's call shape)."""

    def _fake(model, tokenizer, messages, *, max_new_tokens, temperature):
        return response

    return _fake


def _raising_generate(exc: Exception):
    def _fake(model, tokenizer, messages, *, max_new_tokens, temperature):
        raise exc

    return _fake


class TestSelectDateGroupsWellFormed:
    def test_all_true_response(self, config, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _stub_generate('{"all": true}'),
        )
        result = select_date_groups(
            "What do you know about me?",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result == DateSelection(all=True, ranges=(), include_undated=True)

    def test_ranges_response(self, config, monkeypatch):
        response = '{"ranges": [{"start": "2026-07-27", "end": "2026-08-02"}], "undated": false}'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "What did we talk about last week?",
            {"graph1": date(2026, 8, 1)},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is False
        assert result.ranges == (DateWindow(start=date(2026, 7, 27), end=date(2026, 8, 2)),)
        assert result.include_undated is False

    def test_ranges_with_undated_true(self, config, monkeypatch):
        response = '{"ranges": [{"start": "2026-08-04", "end": "2026-08-04"}], "undated": true}'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "What's my job, and what did we talk about two days ago?",
            {"graph_attr": None},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.include_undated is True

    def test_single_day_range_start_equals_end(self, config, monkeypatch):
        response = '{"ranges": [{"start": "2026-08-05", "end": "2026-08-05"}], "undated": false}'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "What did we discuss yesterday?",
            {"graph1": date(2026, 8, 5)},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        window = result.ranges[0]
        assert window.start == window.end == date(2026, 8, 5)

    def test_multiple_ranges(self, config, monkeypatch):
        response = (
            '{"ranges": [{"start": "2026-08-05", "end": "2026-08-05"}, '
            '{"start": "2026-07-27", "end": "2026-08-02"}], "undated": false}'
        )
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "What did we discuss yesterday, or last week?",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert len(result.ranges) == 2

    def test_prose_wrapped_json_still_parses(self, config, monkeypatch):
        """Progressive first-object extraction tolerates a trailing sentence."""
        response = '{"all": true} — that covers everything.'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "What do you know about me?",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is True


class TestSelectDateGroupsFailOpen:
    def test_malformed_json_falls_open(self, config, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _stub_generate("not json at all"),
        )
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result == DateSelection(all=True, ranges=(), include_undated=True, fail_open=True)

    def test_bad_dates_fall_open(self, config, monkeypatch):
        response = '{"ranges": [{"start": "not-a-date", "end": "2026-08-02"}], "undated": false}'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is True

    def test_start_after_end_falls_open(self, config, monkeypatch):
        response = '{"ranges": [{"start": "2026-08-10", "end": "2026-08-02"}], "undated": false}'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is True

    def test_missing_ranges_and_all_falls_open(self, config, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _stub_generate('{"undated": true}'),
        )
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is True

    def test_undated_wrong_type_falls_open(self, config, monkeypatch):
        response = '{"ranges": [{"start": "2026-08-01", "end": "2026-08-02"}], "undated": "yes"}'
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(response))
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is True

    def test_generate_raising_falls_open(self, config, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _raising_generate(RuntimeError("simulated OOM")),
        )
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result == DateSelection(all=True, ranges=(), include_undated=True, fail_open=True)

    def test_empty_response_falls_open(self, config, monkeypatch):
        monkeypatch.setattr("paramem.server.temporal_selection._generate", _stub_generate(""))
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result.all is True

    def test_never_raises(self, config, monkeypatch):
        """The function's contract: no input shape may propagate an
        exception to the caller."""
        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _raising_generate(ValueError("boom")),
        )
        # Must not raise.
        select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )

    def test_missing_recall_selection_section_falls_open(self, config, monkeypatch):
        """``config.voice.load_recall_selection_prompt()`` returning
        ``None`` (prompt file absent, or the
        ``##---RECALL-SELECTION-SECTION---`` marker missing) fails open
        exactly like a missing prompt file did before the prompt
        consolidation — the generate seam must not even be reached."""
        monkeypatch.setattr(config.voice, "load_recall_selection_prompt", lambda: None)

        def _unreachable(*args, **kwargs):
            raise AssertionError("generate must not run when the selection section is missing")

        monkeypatch.setattr("paramem.server.temporal_selection._generate", _unreachable)
        result = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )
        assert result == DateSelection(all=True, ranges=(), include_undated=True, fail_open=True)

    def test_fail_open_verdict_distinguishable_from_genuine_all_verdict(self, config, monkeypatch):
        """Both paths produce the same recall scope (``all=True``, probe
        everything) but must differ in provenance: a swallowed exception
        sets :attr:`DateSelection.fail_open`, a genuine model ``{"all":
        true}`` response does not."""
        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _stub_generate('{"all": true}'),
        )
        genuine = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )

        monkeypatch.setattr(
            "paramem.server.temporal_selection._generate",
            _raising_generate(RuntimeError("simulated failure")),
        )
        fail_open = select_date_groups(
            "anything",
            {},
            model=object(),
            tokenizer=object(),
            config=config,
            today=date(2026, 8, 6),
        )

        assert genuine.fail_open is False
        assert fail_open.fail_open is True
        # Same recall scope either way — only provenance differs.
        assert genuine.all is True
        assert fail_open.all is True
        assert genuine.selects(date(2026, 1, 1)) == fail_open.selects(date(2026, 1, 1))


class TestDateSelectionSelects:
    def test_all_true_selects_any_dated_day(self):
        selection = DateSelection(all=True, ranges=(), include_undated=False)
        assert selection.selects(date(2026, 8, 5)) is True

    def test_all_true_selects_undated_regardless_of_flag(self):
        selection = DateSelection(all=True, ranges=(), include_undated=False)
        assert selection.selects(None) is True

    def test_ranges_selects_day_within_window(self):
        selection = DateSelection(
            all=False,
            ranges=(DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7)),),
            include_undated=False,
        )
        assert selection.selects(date(2026, 8, 4)) is True

    def test_ranges_rejects_day_outside_window(self):
        selection = DateSelection(
            all=False,
            ranges=(DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7)),),
            include_undated=False,
        )
        assert selection.selects(date(2026, 8, 8)) is False

    def test_ranges_none_day_follows_include_undated_true(self):
        selection = DateSelection(
            all=False,
            ranges=(DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7)),),
            include_undated=True,
        )
        assert selection.selects(None) is True

    def test_ranges_none_day_follows_include_undated_false(self):
        selection = DateSelection(
            all=False,
            ranges=(DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7)),),
            include_undated=False,
        )
        assert selection.selects(None) is False

    def test_empty_ranges_selects_nothing_dated(self):
        selection = DateSelection(all=False, ranges=(), include_undated=True)
        assert selection.selects(date(2026, 8, 5)) is False
        assert selection.selects(None) is True

    def test_frozen(self):
        selection = DateSelection(all=True, ranges=(), include_undated=True)
        with pytest.raises(AttributeError):
            selection.all = False


class TestRenderInventory:
    """``_render_inventory`` derives per-date counts (and the undated
    count) from ``date_by_key`` in one pass — it never receives, or
    needs, the actual key lists."""

    def test_counts_tallied_from_date_by_key(self, config, monkeypatch):
        from paramem.server.temporal_selection import _render_inventory

        date_by_key = {
            "graph1": date(2026, 8, 5),
            "graph2": date(2026, 8, 5),
            "graph3": date(2026, 8, 1),
            "graph4": None,
        }
        rendered = _render_inventory(date_by_key)
        assert rendered == (
            "Saturday, 2026-08-01: 1 facts\nWednesday, 2026-08-05: 2 facts\nundated: 1 facts"
        )

    def test_no_dated_entries_only_undated_line(self, config):
        from paramem.server.temporal_selection import _render_inventory

        rendered = _render_inventory({"graph1": None, "graph2": None})
        assert rendered == "undated: 2 facts"

    def test_empty_map_zero_undated(self, config):
        from paramem.server.temporal_selection import _render_inventory

        assert _render_inventory({}) == "undated: 0 facts"
