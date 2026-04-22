"""Comparison report for the migration trial.

Slice 3b.3 emits a placeholder with verbatim spec operator line and five
"—" rows. Slice 5 replaces ``build_comparison_report_placeholder`` with a
real evaluation (additive Optional fields only; do not break schema_version 1).

This module is intentionally isolated so Slice 5 can replace the
implementation by patching a single import — forward-compat guardrail 1.
"""

from __future__ import annotations

COMPARISON_REPORT_OPERATOR_LINE = (
    "These are the raw numbers before vs. after. "
    "Config/prompt changes may legitimately alter extraction behavior. "
    "See docs/config_impact.md for the expected impact of the fields you changed."
)


def build_comparison_report_placeholder(gates_status: str) -> dict:
    """Build a placeholder comparison report for Slice 3b.3.

    Returns a dict matching schema_version 1 with five metric rows containing
    "—" values.  Slice 5 replaces this function with real evaluation using
    additive Optional fields only; schema_version 1 is the stable contract.

    Parameters
    ----------
    gates_status:
        The current gates status string (e.g. ``"pass"``, ``"no_new_sessions"``).
        Stored in the report for Slice 5 to use when computing real deltas.

    Returns
    -------
    dict
        JSON-serialisable comparison report with ``schema_version``,
        ``gates_status``, ``rows``, and ``operator_line``.
    """
    return {
        "schema_version": 1,
        "gates_status": gates_status,
        "rows": [
            {"metric": "Triples extracted — last session", "pre_trial": "—", "trial": "—"},
            {"metric": "Recall on prior-cycle keys", "pre_trial": "—", "trial": "—"},
            {"metric": "Routing-probe classification", "pre_trial": "—", "trial": "—"},
            {"metric": "New ERROR lines in trial log", "pre_trial": "—", "trial": "—"},
            {"metric": "Graph shape", "pre_trial": "—", "trial": "—"},
        ],
        "operator_line": COMPARISON_REPORT_OPERATOR_LINE,
    }
