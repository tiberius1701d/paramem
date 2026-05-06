"""Backwards-compat shim — implementation moved to ``paramem.training.early_stop``.

This module is preserved as a re-export so existing experiment scripts
(Test 14, Test 15, Test 13b) and tests
(``tests/test_early_stop.py``) keep working without import changes.

The canonical home for ``EarlyStopPolicy`` / ``RecallEarlyStopCallback``
is ``paramem.training.early_stop``.  New code should import from there
directly.

Lift performed 2026-05-06 to enable production
``BackgroundTrainer`` to use the same recall-based early-stop gate.
"""

from paramem.training.early_stop import (  # noqa: F401
    ANALYSIS_POLICY,
    ANALYSIS_POLICY_KWARGS,
    EarlyStopPolicy,
    RecallEarlyStopCallback,
    _EarlyStopState,
    _safe_write_json,
)
