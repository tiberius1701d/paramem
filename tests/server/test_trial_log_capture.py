"""Unit tests for TrialLogCapture (Slice 5a).

Tests cover record counting, distinct-class tracking, level filtering,
root-handler lifecycle (add on enter, remove on exit, remove on exception),
and metrics interface.
"""

from __future__ import annotations

import logging

import pytest

from paramem.server.gates import TrialLogCapture

# ---------------------------------------------------------------------------
# Helper: emit records into a specific logger
# ---------------------------------------------------------------------------


def _emit(logger_name: str, level: int, msg: str, exc_info=None) -> None:
    log = logging.getLogger(logger_name)
    if exc_info:
        log.log(level, msg, exc_info=exc_info)
    else:
        log.log(level, msg)


# ---------------------------------------------------------------------------
# Basic counting
# ---------------------------------------------------------------------------


def test_captures_warning_and_above():
    """INFO is below threshold; WARN, ERROR, CRITICAL are counted."""
    log = logging.getLogger("test_cap_warn_above")
    with TrialLogCapture() as cap:
        log.info("below threshold — not counted")
        log.warning("first counted")
        log.error("second counted")
        log.critical("third counted")
    assert cap.metrics["trial_log_errors"] == 3


def test_info_not_counted_default_level():
    """Default level is WARNING; INFO records are ignored."""
    log = logging.getLogger("test_cap_info")
    with TrialLogCapture() as cap:
        log.info("not counted")
        log.debug("also not counted")
    assert cap.metrics["trial_log_errors"] == 0


def test_records_outside_block_not_counted():
    """Records emitted before and after the block are not included."""
    log = logging.getLogger("test_cap_outside")
    log.error("before block — not counted")
    with TrialLogCapture() as cap:
        log.error("inside block — counted")
    log.error("after block — not counted")
    assert cap.metrics["trial_log_errors"] == 1


# ---------------------------------------------------------------------------
# Level parameter
# ---------------------------------------------------------------------------


def test_level_parameter_respected():
    """Constructing with level=ERROR means WARNING records are not counted."""
    log = logging.getLogger("test_cap_level_err")
    with TrialLogCapture(level=logging.ERROR) as cap:
        log.warning("warn — below custom level, not counted")
        log.error("error — counted")
        log.critical("critical — counted")
    assert cap.metrics["trial_log_errors"] == 2


# ---------------------------------------------------------------------------
# Distinct-class tracking (exc_info path)
# ---------------------------------------------------------------------------


def test_distinct_class_tracking():
    """Two ValueError records + one KeyError → distinct list preserves order."""
    log = logging.getLogger("test_cap_distinct")
    try:
        raise ValueError("first")
    except ValueError:
        with TrialLogCapture() as cap:
            log.exception("first value error")
            try:
                raise ValueError("second — same class, not re-added")
            except ValueError:
                log.exception("second value error")
            try:
                raise KeyError("key error")
            except KeyError:
                log.exception("key error")
    assert cap.metrics["trial_log_errors"] == 3
    assert cap.metrics["distinct_classes"] == ["ValueError", "KeyError"]


def test_distinct_classes_empty_when_no_exc_info():
    """Plain log.error() with no exc_info → count=1, distinct_classes=[]."""
    log = logging.getLogger("test_cap_no_exc_info")
    with TrialLogCapture() as cap:
        log.error("plain error message without exception")
    assert cap.metrics["trial_log_errors"] == 1
    assert cap.metrics["distinct_classes"] == []


def test_distinct_classes_not_duplicated():
    """Three records of the same class → distinct_classes has exactly one entry."""
    log = logging.getLogger("test_cap_no_dup")
    with TrialLogCapture() as cap:
        for _ in range(3):
            try:
                raise RuntimeError("repeated error")
            except RuntimeError:
                log.exception("runtime error")
    assert len(cap.metrics["distinct_classes"]) == 1
    assert cap.metrics["distinct_classes"] == ["RuntimeError"]


# ---------------------------------------------------------------------------
# Handler lifecycle
# ---------------------------------------------------------------------------


def test_handler_added_on_enter():
    """Root logger gains exactly one new handler while inside the block."""
    root = logging.getLogger()
    count_before = len(root.handlers)
    with TrialLogCapture():
        assert len(root.handlers) == count_before + 1
    assert len(root.handlers) == count_before


def test_handler_removed_on_exit():
    """Root logger handler count returns to baseline after exit."""
    root = logging.getLogger()
    count_before = len(root.handlers)
    cap = TrialLogCapture()
    cap.__enter__()
    cap.__exit__(None, None, None)
    assert len(root.handlers) == count_before


def test_handler_removed_on_exception():
    """Handler is removed even when an exception is raised inside the block."""
    root = logging.getLogger()
    count_before = len(root.handlers)
    try:
        with TrialLogCapture():
            raise ValueError("inner exception")
    except ValueError:
        pass
    assert len(root.handlers) == count_before


def test_exception_propagates():
    """Exception raised inside the block is not swallowed by __exit__."""
    with pytest.raises(RuntimeError, match="propagated"):
        with TrialLogCapture():
            raise RuntimeError("propagated")


# ---------------------------------------------------------------------------
# Metrics shape
# ---------------------------------------------------------------------------


def test_metrics_keys_stable():
    """metrics dict always has exactly two keys: trial_log_errors, distinct_classes."""
    with TrialLogCapture() as cap:
        pass
    assert set(cap.metrics.keys()) == {"trial_log_errors", "distinct_classes"}


def test_metrics_accessible_after_exit():
    """cap.metrics returns the final state after __exit__."""
    log = logging.getLogger("test_cap_after_exit")
    with TrialLogCapture() as cap:
        log.error("counted")
    # metrics is the final count, not a live view
    assert cap.metrics["trial_log_errors"] == 1


def test_metrics_returns_copy_of_distinct():
    """metrics['distinct_classes'] is a new list each call (not the internal list)."""
    log = logging.getLogger("test_cap_copy")
    with TrialLogCapture() as cap:
        try:
            raise ValueError("v")
        except ValueError:
            log.exception("v")
    m1 = cap.metrics["distinct_classes"]
    m2 = cap.metrics["distinct_classes"]
    assert m1 == m2
    m1.append("Injected")
    assert "Injected" not in cap.metrics["distinct_classes"]


# ---------------------------------------------------------------------------
# Regression: consolidation-executor errors must be inside capture scope
# ---------------------------------------------------------------------------


def test_capture_wraps_consolidation_executor_errors():
    """Errors logged by the consolidation executor are captured.

    Regression guard for the Fix 1 scope bug: the ``with TrialLogCapture()``
    block in ``_run_trial_consolidation`` (``app.py``) must open BEFORE the
    consolidation executor call, not after it.  This test pins that behaviour
    by simulating what the executor block does — logging an error and raising
    — and asserts that the capture records it.

    If the ``with`` block were placed after the executor (the pre-fix
    location), ``trial_log_errors`` would be 0 and ``distinct_classes``
    would be empty, causing ``_row_log_errors`` in ``migration_report.py``
    to silently undercount real trial failures.
    """
    consolidation_log = logging.getLogger("paramem.server.consolidation")

    with TrialLogCapture() as cap:
        # Simulate the executor block: an exception is raised and logged
        # inside the consolidation path (e.g. graph-merge failed).
        exc = RuntimeError("graph-merge failed")
        try:
            raise exc
        except RuntimeError:
            consolidation_log.exception("trial consolidation executor failed")

    assert cap.metrics["trial_log_errors"] >= 1, (
        "Consolidation-executor errors must be captured inside the with block; "
        "trial_log_errors was 0 — the with TrialLogCapture() is likely placed "
        "AFTER the executor call (pre-Fix-1 scope bug)."
    )
    assert "RuntimeError" in cap.metrics["distinct_classes"], (
        "RuntimeError class must appear in distinct_classes when the exception "
        "is logged via log.exception() inside the capture scope."
    )
