"""Structural guard: every Stage-B terminal reaches a flag clear via
``_dispatch_finalize``.

Peer of ``tests/test_extraction_pipeline_guard.py`` — an AST scan rather than
a live-execution test, because the invariant is about *every* code path in
the three Stage-B closures (``_run_interim_training``, ``_run_full_cycle``,
``_run_migration_on_worker``), not just the paths a given unit test happens
to drive.

The cycle-lifecycle primitive ``_run_stage_b_cycle`` (paramem/server/app.py)
is the SOLE dispatch point for ``_dispatch_finalize`` on both the success and
the crash-envelope path.  For that single-dispatch-point invariant to hold,
the three path-specific bodies must:

1. Never call ``_dispatch_finalize`` themselves (the primitive owns it).
2. Return a ``(outcome, finalizer)`` 2-tuple terminal on every code path
   (or raise) — a bare ``return`` / ``return None`` would silently swallow
   the outcome the primitive dispatches.
"""

from __future__ import annotations

import ast
from pathlib import Path

_APP_PY = Path(__file__).resolve().parent.parent.parent / "paramem" / "server" / "app.py"

_STAGE_B_BODY_NAMES = frozenset(
    {
        "_run_interim_training",
        "_run_full_cycle",
        "_run_migration_on_worker",
    }
)


def _find_functions_by_name(tree: ast.AST, names: frozenset[str]) -> dict[str, ast.FunctionDef]:
    """Return ``{name: FunctionDef}`` for every matching def/async def in *tree*.

    Uses ``ast.walk`` so nested closures (the Stage-B bodies are defined
    inside their outer entry-point functions, not at module level) are found
    regardless of nesting depth.
    """
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found[node.name] = node
    return found


def _calls_dispatch_finalize(node: ast.FunctionDef) -> list[int]:
    """Return line numbers of any ``_dispatch_finalize(...)`` call inside *node*."""
    lines = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "_dispatch_finalize"
        ):
            lines.append(sub.lineno)
    return lines


def _direct_returns(node: ast.FunctionDef) -> list[ast.Return]:
    """Return every ``ast.Return`` belonging directly to *node*'s own body.

    Manually walks children (rather than ``ast.walk``, which does not prune
    subtrees) so a ``return`` inside a nested ``def``/``lambda`` defined
    within *node* is NOT attributed to *node* itself — e.g. a small local
    helper closure with its own unrelated ``return bool(...)``.
    """
    found: list[ast.Return] = []

    def _walk(n: ast.AST, is_root: bool) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # nested scope — its returns are not this function's terminals
            if isinstance(child, ast.Return):
                found.append(child)
            _walk(child, False)

    _walk(node, True)
    return found


def _non_tuple_returns(node: ast.FunctionDef) -> list[int]:
    """Return line numbers of ``return`` statements that are not a 2-tuple literal.

    Only inspects ``return`` statements belonging directly to *node*'s own
    body — nested function defs inside it (small local helper closures) are
    excluded via ``_direct_returns``.
    """
    bad: list[int] = []
    for ret in _direct_returns(node):
        value = ret.value
        if isinstance(value, ast.Tuple) and len(value.elts) == 2:
            continue
        bad.append(ret.lineno)
    return bad


def test_stage_b_bodies_exist_and_are_found():
    """Non-vacuity: all three Stage-B body closures are present in app.py.

    Guards against the scan silently passing because a function was renamed
    or deleted — ``_find_functions_by_name`` would just return fewer entries.
    """
    tree = ast.parse(_APP_PY.read_text())
    found = _find_functions_by_name(tree, _STAGE_B_BODY_NAMES)
    missing = _STAGE_B_BODY_NAMES - found.keys()
    assert not missing, f"Stage-B body closure(s) not found in app.py: {sorted(missing)}"


def test_stage_b_bodies_never_call_dispatch_finalize():
    """The three Stage-B bodies must never call _dispatch_finalize themselves.

    _run_stage_b_cycle is the sole dispatch point — for both the normal
    return and the crash-envelope path.  A body calling it directly would
    create a second dispatch point and could double-dispatch or race the
    primitive's own crash-envelope dispatch.
    """
    tree = ast.parse(_APP_PY.read_text())
    found = _find_functions_by_name(tree, _STAGE_B_BODY_NAMES)
    assert found, "no Stage-B body closures found — see test_stage_b_bodies_exist_and_are_found"

    offenders = {name: _calls_dispatch_finalize(node) for name, node in found.items()}
    offenders = {name: lines for name, lines in offenders.items() if lines}
    assert not offenders, (
        f"Stage-B body closure(s) call _dispatch_finalize directly: {offenders}. "
        "_run_stage_b_cycle must be the sole dispatch point."
    )


def test_stage_b_bodies_return_terminal_tuples_on_every_path():
    """Every `return` in a Stage-B body must be a (outcome, finalizer) 2-tuple.

    A bare `return` / `return None` would make `outcome, finalizer =
    body(loop, bt)` raise a TypeError inside _run_stage_b_cycle's own
    try/except — which the primitive would then misreport as a crash rather
    than the intended terminal. Catches a regression where a future edit
    reintroduces a direct early return instead of a terminal.
    """
    tree = ast.parse(_APP_PY.read_text())
    found = _find_functions_by_name(tree, _STAGE_B_BODY_NAMES)
    assert found, "no Stage-B body closures found — see test_stage_b_bodies_exist_and_are_found"

    offenders = {name: _non_tuple_returns(node) for name, node in found.items()}
    offenders = {name: lines for name, lines in offenders.items() if lines}
    assert not offenders, (
        f"Stage-B body closure(s) have a `return` that is not a 2-tuple terminal: {offenders}"
    )


def test_run_stage_b_cycle_dispatches_finalize_on_both_paths():
    """_run_stage_b_cycle calls _dispatch_finalize from both the success and
    the crash-envelope path — the single, structurally-guaranteed dispatch
    point for every Stage-B terminal (success or failure).
    """
    tree = ast.parse(_APP_PY.read_text())
    found = _find_functions_by_name(tree, frozenset({"_run_stage_b_cycle"}))
    assert "_run_stage_b_cycle" in found, "_run_stage_b_cycle not found in app.py"

    node = found["_run_stage_b_cycle"]
    call_lines = _calls_dispatch_finalize(node)
    assert len(call_lines) == 2, (
        f"_run_stage_b_cycle must call _dispatch_finalize exactly twice "
        f"(crash envelope + normal return); found at lines {call_lines}"
    )

    # One call must be inside an except handler (the crash envelope); the
    # other must be outside any except handler (the normal-return path).
    except_lines: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.ExceptHandler):
            for inner in ast.walk(sub):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "_dispatch_finalize"
                ):
                    except_lines.add(inner.lineno)
    assert len(except_lines) == 1, (
        f"Expected exactly one _dispatch_finalize call inside an except handler "
        f"(the crash envelope); found {sorted(except_lines)} of {call_lines}"
    )
    assert len(set(call_lines) - except_lines) == 1, (
        "Expected exactly one _dispatch_finalize call outside any except handler "
        "(the normal-return path)"
    )
