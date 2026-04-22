"""Unit tests for build_comparison_report (Slice 5a).

Tests cover all 5 rows (3 real + 2 deferred), missing-input fallbacks,
schema-version stability, and the no-state-reach-in purity guardrail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paramem.server.migration_report import (
    COMPARISON_REPORT_OPERATOR_LINE,
    COMPARISON_REPORT_SCHEMA_VERSION,
    build_comparison_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gates(
    *,
    status: str = "pass",
    g4_status: str = "pass",
    g4_metrics: dict | None = None,
    trial_log: dict | None = None,
    n_details: int = 4,
) -> dict:
    """Build a minimal gates dict for testing."""
    g4 = {"gate": 4, "name": "live_registry_recall", "status": g4_status, "reason": None}
    if g4_metrics is not None:
        g4["metrics"] = g4_metrics
    else:
        g4["metrics"] = None

    details = [
        {"gate": i, "name": f"gate{i}", "status": "pass", "reason": None, "metrics": None}
        for i in range(1, n_details)
    ]
    if n_details >= 4:
        details.append(g4)

    result: dict[str, Any] = {"status": status, "details": details}
    if trial_log is not None:
        result["trial_log"] = trial_log
    return result


def _write_graph(path: Path, n_nodes: int = 5, predicates: list[str] | None = None) -> None:
    """Write a minimal NetworkX node_link_data JSON to path."""
    import networkx as nx

    g = nx.MultiDiGraph()
    for i in range(n_nodes):
        g.add_node(f"n{i}")
    preds = predicates or ["knows", "works_at", "lives_in"]
    for i, pred in enumerate(preds):
        g.add_edge(f"n{i % n_nodes}", f"n{(i + 1) % n_nodes}", predicate=pred)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(nx.node_link_data(g), fh)


# ---------------------------------------------------------------------------
# Row 1 — Triples extracted (DEFERRED)
# ---------------------------------------------------------------------------


def test_triples_row_pending():
    """Row 1 always has pending=True and the expected pending_reason."""
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][0]
    assert row["metric"] == "Triples extracted — last session"
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"
    assert row.get("pending") is True
    assert row.get("pending_reason") == "needs pre-trial summary persistence"


# ---------------------------------------------------------------------------
# Row 2 — Recall on prior-cycle keys (REAL)
# ---------------------------------------------------------------------------


def test_recall_row_real_when_gate4_pass():
    """Gate 4 pass with sampled=20, recalled=19 → real cells."""
    gates = _make_gates(
        g4_status="pass",
        g4_metrics={"sampled": 20, "recalled": 19, "sampled_keys": []},
    )
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][1]
    assert row["metric"] == "Recall on prior-cycle keys"
    assert row["pre_trial"] == "100% (live owns these keys)"
    assert row["trial"] == "19/20"
    assert "pending" not in row


def test_recall_row_real_when_gate4_fail():
    """Gate 4 fail still renders real numbers (the fail is informational)."""
    gates = _make_gates(
        g4_status="fail",
        g4_metrics={"sampled": 20, "recalled": 15},
    )
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][1]
    assert row["trial"] == "15/20"


def test_recall_row_skipped_under_20_keys():
    """Gate 4 status=skipped → both cells —, sub_note about < 20 keys."""
    gates = _make_gates(g4_status="skipped")
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][1]
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"
    assert "sub_note" in row
    assert "20" in row["sub_note"]


def test_recall_row_missing_gate4():
    """Only 3 gate details → both cells —, sub_note about gate 4 absent."""
    gates = _make_gates(n_details=3)
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][1]
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"
    assert "sub_note" in row


def test_recall_row_unexpected_gate4_status():
    """Unexpected gate 4 status → both cells —, sub_note with status."""
    gates = _make_gates(g4_status="no_new_sessions")
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][1]
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"
    assert "sub_note" in row


# ---------------------------------------------------------------------------
# Row 3 — Routing-probe classification (DEFERRED)
# ---------------------------------------------------------------------------


def test_routing_probe_row_pending():
    """Row 3 always has pending=True and the expected pending_reason."""
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][2]
    assert row["metric"] == "Routing-probe classification"
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"
    assert row.get("pending") is True
    assert row.get("pending_reason") == "labeled query set undefined"


# ---------------------------------------------------------------------------
# Row 4 — New ERROR lines in trial log (REAL)
# ---------------------------------------------------------------------------


def test_log_errors_row_zero():
    """trial_log with 0 errors → trial=='0', no sub_list."""
    gates = _make_gates(trial_log={"trial_log_errors": 0, "distinct_classes": []})
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][3]
    assert row["metric"] == "New ERROR lines in trial log"
    assert row["pre_trial"] == "—"
    assert row["trial"] == "0"
    assert "sub_list" not in row


def test_log_errors_row_nonzero_with_classes():
    """trial_log with 3 errors + 6 classes → trial=='3', sub_list truncated to 5."""
    classes = ["ValueError", "KeyError", "RuntimeError", "OSError", "TimeoutError", "Extra"]
    gates = _make_gates(trial_log={"trial_log_errors": 3, "distinct_classes": classes})
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][3]
    assert row["trial"] == "3"
    assert row["sub_list"] == ["ValueError", "KeyError", "RuntimeError", "OSError", "TimeoutError"]
    assert len(row["sub_list"]) == 5


def test_log_errors_row_no_capture():
    """gates without trial_log key → both cells —, sub_note set."""
    gates = _make_gates()  # no trial_log key
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][3]
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"
    assert row.get("sub_note") == "trial log capture unavailable"


def test_log_errors_row_nonzero_no_classes():
    """n=2 but distinct_classes=[] → no sub_list (no names to show)."""
    gates = _make_gates(trial_log={"trial_log_errors": 2, "distinct_classes": []})
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][3]
    assert row["trial"] == "2"
    assert "sub_list" not in row


# ---------------------------------------------------------------------------
# Row 5 — Graph shape (REAL)
# ---------------------------------------------------------------------------


def test_graph_shape_real_both_paths(tmp_path):
    """Both graph paths present → real summaries rendered."""
    pre_path = tmp_path / "pre" / "cumulative_graph.json"
    trial_path = tmp_path / "trial" / "cumulative_graph.json"
    _write_graph(pre_path, n_nodes=5, predicates=["knows", "works_at", "lives_in"])
    _write_graph(trial_path, n_nodes=8, predicates=["plays", "has"])
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=pre_path,
        trial_graph_path=trial_path,
    )
    row = report["rows"][4]
    assert row["metric"] == "Graph shape"
    assert "5 nodes" in row["pre_trial"]
    assert "8 nodes" in row["trial"]
    # Top predicates should appear
    assert "knows" in row["pre_trial"]
    assert "plays" in row["trial"]


def test_graph_shape_missing_pre_trial(tmp_path):
    """pre_trial_graph_path=None → pre_trial cell is —."""
    trial_path = tmp_path / "trial" / "cumulative_graph.json"
    _write_graph(trial_path, n_nodes=3)
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=trial_path,
    )
    row = report["rows"][4]
    assert row["pre_trial"] == "—"
    assert "3 nodes" in row["trial"]


def test_graph_shape_missing_trial(tmp_path):
    """trial_graph_path=None → trial cell is —."""
    pre_path = tmp_path / "pre" / "cumulative_graph.json"
    _write_graph(pre_path, n_nodes=4)
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=pre_path,
        trial_graph_path=None,
    )
    row = report["rows"][4]
    assert "4 nodes" in row["pre_trial"]
    assert row["trial"] == "—"


def test_graph_shape_corrupt_json(tmp_path):
    """Corrupt JSON at path → cell — without raising."""
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("this is not valid json {{{")
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=bad_path,
        trial_graph_path=None,
    )
    row = report["rows"][4]
    assert row["pre_trial"] == "—"


def test_graph_shape_both_missing():
    """Both paths None → both cells —."""
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    row = report["rows"][4]
    assert row["pre_trial"] == "—"
    assert row["trial"] == "—"


def test_graph_shape_in_memory_graph_preferred(tmp_path):
    """When pre_trial_graph is passed, it takes priority over path."""
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_node("alice")
    g.add_node("bob")
    g.add_edge("alice", "bob", predicate="knows")

    # Write a different graph to the path to confirm path is ignored.
    pre_path = tmp_path / "pre.json"
    _write_graph(pre_path, n_nodes=99)

    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=pre_path,
        trial_graph_path=None,
        pre_trial_graph=g,
    )
    row = report["rows"][4]
    assert "2 nodes" in row["pre_trial"]
    # 99-node graph from file should NOT appear
    assert "99" not in row["pre_trial"]


# ---------------------------------------------------------------------------
# Top-level report shape
# ---------------------------------------------------------------------------


def test_schema_version_unchanged():
    """schema_version must be 1 (forward-compat contract)."""
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    assert report["schema_version"] == COMPARISON_REPORT_SCHEMA_VERSION == 1


def test_operator_line_verbatim():
    """operator_line matches the module constant exactly."""
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    assert report["operator_line"] == COMPARISON_REPORT_OPERATOR_LINE


def test_five_rows_always_present():
    """Report always has exactly 5 rows regardless of input."""
    gates = _make_gates()
    report = build_comparison_report(
        gates=gates,
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
    assert len(report["rows"]) == 5


def test_gates_status_propagated():
    """gates_status in report matches the input gates dict."""
    for status in ("pass", "fail", "no_new_sessions", "unknown_status"):
        gates = _make_gates(status=status)
        report = build_comparison_report(
            gates=gates,
            pre_trial_graph_path=None,
            trial_graph_path=None,
        )
        assert report["gates_status"] == status


def test_no_state_reach_in():
    """migration_report module must not hold a reference to '_state'."""
    import paramem.server.migration_report as mr

    assert "_state" not in mr.__dict__
    # Also verify no module-level name contains _state (excluding the
    # docstring and this very test).
    for name in dir(mr):
        if name == "_state":
            pytest.fail(f"migration_report exports '_state' attribute: {name}")
