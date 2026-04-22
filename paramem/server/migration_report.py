"""Comparison report for the migration trial.

Slice 3b.3 emitted a placeholder with verbatim spec operator line and five
"—" rows.  Slice 5a replaces :func:`build_comparison_report_placeholder` with
:func:`build_comparison_report` (real evaluation, additive Optional fields
only; schema_version 1 is the stable contract).

Module isolation is intentional — Slice 5a can replace the implementation by
patching a single import (forward-compat guardrail 1).

No ``_state`` module-level references permitted.  All helpers accept plain
kwargs and are unit-testable without a running server.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

COMPARISON_REPORT_OPERATOR_LINE = (
    "These are the raw numbers before vs. after. "
    "Config/prompt changes may legitimately alter extraction behavior. "
    "See docs/config_impact.md for the expected impact of the fields you changed."
)

COMPARISON_REPORT_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Row helpers (private)
# ---------------------------------------------------------------------------


def _row_triples_extracted(gates: dict) -> dict[str, Any]:
    """Return the deferred 'Triples extracted' row.

    Slice 5a defers this row because the spec (L395) requires
    ``_state["last_consolidation_summary"]`` which is not yet persisted.
    Renders ``pending: True`` with an explicit reason so the row is
    never silently empty.

    Parameters
    ----------
    gates:
        The gates dict from ``_state["migration"]["trial"]["gates"]``.
        Not consumed here; accepted for interface symmetry with other
        row helpers.

    Returns
    -------
    dict[str, Any]
        Row dict with ``pending=True`` and ``pending_reason``.
    """
    return {
        "metric": "Triples extracted — last session",
        "pre_trial": "—",
        "trial": "—",
        "pending": True,
        "pending_reason": "needs pre-trial summary persistence",
    }


def _row_recall(gates: dict) -> dict[str, Any]:
    """Mirror the gate 4 metrics into the recall row.

    Spec L404: same 20-key sample as gate 4.  Slice 4 exposes
    ``recalled`` and ``sampled`` in ``gates["details"][3]["metrics"]``
    (Guardrail G1 — ``sampled_keys`` is also present for Slice 5b).

    Parameters
    ----------
    gates:
        Gates dict from ``_state["migration"]["trial"]["gates"]``.

    Returns
    -------
    dict[str, Any]
        Row dict.  Cells are ``"—"`` when gate 4 is absent, skipped,
        or has an unexpected status.
    """
    details = gates.get("details") or []
    # Gate 4 is always at index 3 (evaluate_gates returns exactly 4
    # GateResult dicts in gate order — see app.py:2871).
    if len(details) < 4:
        return {
            "metric": "Recall on prior-cycle keys",
            "pre_trial": "—",
            "trial": "—",
            "sub_note": "gate 4 not present in results",
        }

    g4 = details[3]
    g4_status = g4.get("status")
    g4_metrics: dict = g4.get("metrics") or {}

    if g4_status == "skipped":
        return {
            "metric": "Recall on prior-cycle keys",
            "pre_trial": "—",
            "trial": "—",
            "sub_note": "live registry has fewer than 20 keys",
        }

    if g4_status not in ("pass", "fail"):
        return {
            "metric": "Recall on prior-cycle keys",
            "pre_trial": "—",
            "trial": "—",
            "sub_note": f"gate 4 status: {g4_status}",
        }

    sampled = int(g4_metrics.get("sampled") or 0)
    recalled = int(g4_metrics.get("recalled") or 0)
    return {
        "metric": "Recall on prior-cycle keys",
        "pre_trial": "100% (live owns these keys)",
        "trial": f"{recalled}/{sampled}",
    }


def _row_routing_probe(gates: dict) -> dict[str, Any]:
    """Return the deferred 'Routing-probe classification' row.

    Deferred because the labeled query set required for the probe is not
    yet defined (spec L397).  Renders ``pending: True`` so the row is
    never silently empty.

    Parameters
    ----------
    gates:
        Gates dict.  Not consumed here; accepted for interface symmetry.

    Returns
    -------
    dict[str, Any]
        Row dict with ``pending=True`` and ``pending_reason``.
    """
    return {
        "metric": "Routing-probe classification",
        "pre_trial": "—",
        "trial": "—",
        "pending": True,
        "pending_reason": "labeled query set undefined",
    }


def _row_log_errors(gates: dict) -> dict[str, Any]:
    """Read the top-level ``trial_log`` block from :class:`TrialLogCapture`.

    ``TrialLogCapture`` (``gates.py``) writes
    ``gates["trial_log"] = {"trial_log_errors": int, "distinct_classes": list[str]}``
    at the *top level* of the gates payload — not nested in per-gate
    metrics — because the capture spans the entire consolidation + gate
    run (extraction, training, adapter reload, gate 4).

    When ``trial_log`` is absent (pre-Slice-5a runs or runs that failed
    before capture attached), both cells render ``"—"`` with an explicit
    ``sub_note`` so the row is never silently empty.

    Parameters
    ----------
    gates:
        Gates dict from ``_state["migration"]["trial"]["gates"]``.

    Returns
    -------
    dict[str, Any]
        Row dict.
    """
    trial_log: dict = gates.get("trial_log") or {}
    if not trial_log:
        return {
            "metric": "New ERROR lines in trial log",
            "pre_trial": "—",
            "trial": "—",
            "sub_note": "trial log capture unavailable",
        }

    n = int(trial_log.get("trial_log_errors") or 0)
    distinct = list(trial_log.get("distinct_classes") or [])
    row: dict[str, Any] = {
        "metric": "New ERROR lines in trial log",
        "pre_trial": "—",
        "trial": str(n),
    }
    if n > 0 and distinct:
        # Truncate to first 5 distinct exception class names for display
        # (full list lives in gates["trial_log"]["distinct_classes"] for Slice 5b).
        row["sub_list"] = distinct[:5]
    return row


def _summarise_graph(path: Path | None, graph: "Any | None" = None) -> str:
    """Compute a graph-shape summary string.

    Prefers the in-memory ``graph`` when provided; falls back to loading
    from ``path``.  Returns ``"—"`` when both are absent or loading
    fails.

    Parameters
    ----------
    path:
        Filesystem path to a NetworkX node-link JSON file.  ``None``
        when not available.
    graph:
        In-memory ``nx.MultiDiGraph`` (or any NetworkX graph with
        ``number_of_nodes()`` and ``edges(data=True)``).  ``None`` when
        not available.

    Returns
    -------
    str
        E.g. ``"42 nodes, top: knows, works_at, lives_in, has_pet, plays"``
        or ``"—"`` when no data is available.
    """
    from collections import Counter

    import networkx as nx

    g: Any | None = graph

    if g is None:
        if path is None or not path.exists():
            return "—"
        try:
            with open(path) as fh:
                data = json.load(fh)
            g = nx.node_link_graph(data, edges="links")
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            # Try without explicit edges key (older networkx format)
            try:
                with open(path) as fh:
                    data = json.load(fh)
                g = nx.node_link_graph(data)
            except Exception:  # noqa: BLE001
                return "—"

    try:
        n_nodes = g.number_of_nodes()
    except AttributeError:
        return "—"

    pred_counter: Counter[str] = Counter(
        attr.get("predicate", "related_to") for _u, _v, attr in g.edges(data=True)
    )
    top5 = [p for p, _ in pred_counter.most_common(5)]
    if not top5:
        return f"{n_nodes} nodes"
    return f"{n_nodes} nodes, top: {', '.join(top5)}"


def _row_graph_shape(
    pre_trial_graph_path: Path | None,
    trial_graph_path: Path | None,
    *,
    pre_trial_graph: "Any | None" = None,
) -> dict[str, Any]:
    """Compute the graph-shape comparison row.

    Accepts an optional in-memory graph for the pre-trial side (Slice 5a
    R1 resolution — in production, server runs with ``persist_graph=False``
    so the live cumulative graph is transient; the in-memory fallback lets
    the row render real data without materialising to disk).

    Parameters
    ----------
    pre_trial_graph_path:
        Path to the pre-trial cumulative graph JSON, or ``None``.
    trial_graph_path:
        Path to the trial cumulative graph JSON, or ``None``.
    pre_trial_graph:
        In-memory graph for the pre-trial side.  When provided, takes
        priority over ``pre_trial_graph_path``.

    Returns
    -------
    dict[str, Any]
        Row dict with ``pre_trial`` and ``trial`` cells.
    """
    pre_summary = _summarise_graph(pre_trial_graph_path, graph=pre_trial_graph)
    trial_summary = _summarise_graph(trial_graph_path)
    return {
        "metric": "Graph shape",
        "pre_trial": pre_summary,
        "trial": trial_summary,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_comparison_report(
    *,
    gates: dict,
    pre_trial_graph_path: Path | None,
    trial_graph_path: Path | None,
    pre_trial_graph: "Any | None" = None,
) -> dict[str, Any]:
    """Build the real comparison report for a TRIAL run.

    Replaces :func:`build_comparison_report_placeholder`.  Schema version
    is unchanged (1) — the additions (``pending``, ``pending_reason``,
    ``sub_note``, ``sub_list``) are Optional additive-only fields.

    Parameters
    ----------
    gates:
        ``_state["migration"]["trial"]["gates"]`` dict produced by
        ``_run_trial_consolidation``.  Must contain ``status`` and
        ``details`` (list of 4 GateResult dicts).  May contain
        ``trial_log`` (Slice 5a addition: counters from TrialLogCapture).
    pre_trial_graph_path:
        Cumulative graph from before the trial (filesystem path).
        ``None`` when no prior graph exists (fresh install).
    trial_graph_path:
        ``data/ha/state/trial_graph/cumulative_graph.json`` (or whatever
        path the trial marker recorded).  ``None`` when the trial ran
        with NO_NEW_SESSIONS.
    pre_trial_graph:
        In-memory ``nx.MultiDiGraph`` for the pre-trial side.  When
        provided, takes priority over ``pre_trial_graph_path``.
        Production server uses this when ``persist_graph=False``.

    Returns
    -------
    dict[str, Any]
        ``{schema_version, gates_status, rows, operator_line}`` where
        each row is ``{metric, pre_trial, trial, ...optional...}``.
    """
    return {
        "schema_version": COMPARISON_REPORT_SCHEMA_VERSION,
        "gates_status": gates.get("status", "unknown"),
        "rows": [
            _row_triples_extracted(gates),
            _row_recall(gates),
            _row_routing_probe(gates),
            _row_log_errors(gates),
            _row_graph_shape(
                pre_trial_graph_path,
                trial_graph_path,
                pre_trial_graph=pre_trial_graph,
            ),
        ],
        "operator_line": COMPARISON_REPORT_OPERATOR_LINE,
    }


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------


def build_comparison_report_placeholder(gates_status: str) -> dict[str, Any]:
    """Backward-compatible shim wrapping :func:`build_comparison_report`.

    .. deprecated::
        Use :func:`build_comparison_report` directly.  This shim will be
        removed in a future slice.

    Parameters
    ----------
    gates_status:
        The current gates status string.  Passed through to the real
        implementation as ``gates={"status": gates_status}``.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable comparison report matching schema_version 1.
    """
    warnings.warn(
        "build_comparison_report_placeholder is deprecated; "
        "use build_comparison_report() directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_comparison_report(
        gates={"status": gates_status},
        pre_trial_graph_path=None,
        trial_graph_path=None,
    )
