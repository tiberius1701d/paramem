"""Structural pins for the one-execution-envelope design: the invariants
that hold by construction across ``paramem/server/app.py``, ``paramem/server/
calibrate.py``, ``paramem/server/consolidation.py`` and ``paramem/utils/
artifacts.py`` together, so no single existing guard file owns them.

Grep/AST only — no model load, no prompt prose quoted.  Style follows
``tests/server/test_stage_b_cycle_guard.py`` and
``tests/test_extraction_pipeline_guard.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import paramem.server.app as app_module
import paramem.server.calibrate as calibrate_module
import paramem.server.consolidation as consolidation_module

_APP_SOURCE = inspect.getsource(app_module)
_APP_TREE = ast.parse(_APP_SOURCE)


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _assign_targets_setting_consolidating_false(tree: ast.AST) -> list[tuple[int, tuple[str, ...]]]:
    """Every ``_state["consolidating"] = False`` assignment in *tree*,
    as ``(lineno, enclosing_function_stack)`` — the full nesting stack
    (module-level function first) so a nested closure (e.g.
    ``_consolidation_terminal``'s own ``_run``) is attributable to its
    owning top-level function, not just its innermost name."""
    hits: list[tuple[int, tuple[str, ...]]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "_state"
                    and _subscript_key(target) == "consolidating"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is False
                ):
                    hits.append((node.lineno, tuple(self._stack)))
            self.generic_visit(node)

    def _subscript_key(node: ast.Subscript) -> "str | None":
        sl = node.slice
        if isinstance(sl, ast.Constant):
            return sl.value
        return None

    _Visitor().visit(tree)
    return hits


# ---------------------------------------------------------------------------
# Exactly four `_state["consolidating"] = False` sites in app.py:
# inside `_consolidation_terminal`, `interim_discard`, `speaker_forget`,
# `debug_erase_keys`.
# ---------------------------------------------------------------------------


class TestExactlyFourConsolidatingFalseSites:
    def test_four_sites_in_app_py(self) -> None:
        hits = _assign_targets_setting_consolidating_false(_APP_TREE)
        assert len(hits) == 4, f"expected exactly four sites; found {hits}"

    def test_sites_are_the_documented_four_functions(self) -> None:
        hits = _assign_targets_setting_consolidating_false(_APP_TREE)
        # The owning TOP-LEVEL function for each hit -- a nested closure
        # (_consolidation_terminal's own ``_run``) is attributed to its
        # enclosing top-level function, since that is the documented site.
        owning_top_level = {stack[0] for _lineno, stack in hits}
        expected_functions = {
            "_consolidation_terminal",
            "interim_discard",
            "speaker_forget",
            "debug_erase_keys",
        }
        assert owning_top_level == expected_functions, (
            f"consolidating=False sites moved to unexpected functions: {owning_top_level}"
        )


# ---------------------------------------------------------------------------
# `_set_voice_pipeline_profile(_target_profile(), ...)` appears
# only inside `_end_voice_eviction`.
# ---------------------------------------------------------------------------


def _calls_to_set_voice_pipeline_profile_with_target(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = ["<module>"]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_set_voice_pipeline_profile"
                and node.args
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "_target_profile"
            ):
                hits.append((node.lineno, self._stack[-1]))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


class TestVoiceRestoreSingleCallSite:
    def test_target_profile_restore_only_inside_end_voice_eviction(self) -> None:
        hits = _calls_to_set_voice_pipeline_profile_with_target(_APP_TREE)
        enclosing = {name for _lineno, name in hits}
        assert enclosing == {"_end_voice_eviction"}, (
            f"_set_voice_pipeline_profile(_target_profile(), ...) called from "
            f"outside _end_voice_eviction: {hits}"
        )
        assert len(hits) == 1, "exactly one restore call site inside _end_voice_eviction itself"


# ---------------------------------------------------------------------------
# `mark_consolidated(` appears in app.py only inside the three
# retirement primitives plus the allowlisted trial-arc `_run_extraction_phase`.
# ---------------------------------------------------------------------------


def _calls_to_mark_consolidated(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = ["<module>"]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "mark_consolidated":
                hits.append((node.lineno, self._stack[-1]))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


class TestMarkConsolidatedCallers:
    _GRANDFATHERED = frozenset({"_run_extraction_phase"})
    _PRIMITIVES = frozenset(
        {
            "retire_unattributable_sessions",  # paramem/server/consolidation.py
            "_retire_extracted_sessions",
            "_retire_ledger_sessions_and_dispose",
        }
    )

    def test_app_py_callers_are_the_three_primitives_or_the_allowlisted_trial_arc(
        self,
    ) -> None:
        hits = _calls_to_mark_consolidated(_APP_TREE)
        enclosing = {name for _lineno, name in hits}
        allowed = self._PRIMITIVES | self._GRANDFATHERED
        offenders = enclosing - allowed
        assert not offenders, (
            f"mark_consolidated( called from outside the three retirement "
            f"primitives / the allowlisted trial arc: {offenders} "
            f"(all hits: {hits})"
        )

    def test_retire_unattributable_sessions_calls_mark_consolidated(self) -> None:
        """Sanity: the primitive itself is actually a caller — confirms the
        scan isn't vacuously passing because the function moved out of
        app.py (it lives in paramem/server/consolidation.py)."""
        source = inspect.getsource(consolidation_module)
        tree = ast.parse(source)
        hits = _calls_to_mark_consolidated(tree)
        enclosing = {name for _lineno, name in hits}
        assert "retire_unattributable_sessions" in enclosing


# ---------------------------------------------------------------------------
# One get_or_create_consolidation_loop in the tree; calibrate and
# app both call it.
# ---------------------------------------------------------------------------


class TestOneLoopConstructor:
    def test_defined_exactly_once_in_the_tree(self) -> None:
        import subprocess

        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["grep", "-rn", "^def get_or_create_consolidation_loop", str(root / "paramem")],
            capture_output=True,
            text=True,
        )
        lines = [ln for ln in out.stdout.splitlines() if ln]
        assert len(lines) == 1, f"expected exactly one definition; found: {lines}"
        assert "paramem/server/consolidation.py" in lines[0]

    def test_calibrate_module_calls_it(self) -> None:
        source = inspect.getsource(calibrate_module)
        assert "get_or_create_consolidation_loop" in source

    def test_app_module_imports_it_from_the_single_source(self) -> None:
        assert (
            app_module.get_or_create_consolidation_loop
            is consolidation_module.get_or_create_consolidation_loop
        )

    def test_no_second_get_or_create_helper_remains_in_app_py(self) -> None:
        """The former ``_get_or_create_consolidation_loop`` app-local
        duplicate is retired in favour of the one shared constructor."""
        assert not hasattr(app_module, "_get_or_create_consolidation_loop")
        assert not hasattr(calibrate_module, "_ensure_calibration_loop")


# ---------------------------------------------------------------------------
# No arbitrator stage tests an action identity where
# `stages_event` is the real predicate: AST scan of `_dispatch_consolidation`
# allowlisting the AUTO resolution, the INTERIM interim-tier check, and the
# tail dispatch table.
# ---------------------------------------------------------------------------


def _action_identity_comparisons(func: ast.FunctionDef) -> list[ast.AST]:
    """Every ``Compare``/boolean node in *func* testing an action's identity
    against a specific :class:`ConsolidationAction` member (``action is
    ConsolidationAction.X`` / ``action == ConsolidationAction.X`` / ``action
    in (ConsolidationAction.X, ...)``), with its line number."""
    hits: list[ast.AST] = []

    def _is_action_name(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == "action"

    def _is_consolidation_action_attr(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ConsolidationAction"
        )

    for node in ast.walk(func):
        if isinstance(node, ast.Compare) and _is_action_name(node.left):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Is, ast.Eq)) and _is_consolidation_action_attr(comparator):
                    hits.append(node)
                elif isinstance(op, ast.In):
                    # `action in (ConsolidationAction.FULL, ConsolidationAction.RECONCILE)`
                    elts = getattr(comparator, "elts", [comparator])
                    if any(_is_consolidation_action_attr(e) for e in elts):
                        hits.append(node)
    return hits


def _anchor_scheduled_assign(func: ast.FunctionDef) -> ast.Assign:
    """The ``_scheduled = action is ConsolidationAction.AUTO`` binding."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_scheduled"
        ):
            return node
    raise AssertionError("`_scheduled = ...` assignment not found")


def _anchor_interim_zero_check(func: ast.FunctionDef) -> ast.If:
    """The ``INTERIM`` request at ``max_interim_count == 0`` no-op-tier check
    -- located structurally by its ``max_interim_count`` reference, not by
    line number."""
    for node in ast.walk(func):
        if isinstance(node, ast.If) and any(
            isinstance(sub, ast.Attribute) and sub.attr == "max_interim_count"
            for sub in ast.walk(node.test)
        ):
            return node
    raise AssertionError("the INTERIM-at-zero-count tier check `if` not found")


def _anchor_tuple_membership_if(func: ast.FunctionDef, *members: str) -> ast.If:
    """The ``if action in (ConsolidationAction.X, ConsolidationAction.Y):``
    statement whose tuple names exactly *members* -- located structurally,
    not by line number."""
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "action"
            and any(isinstance(op, ast.In) for op in node.test.ops)
        ):
            continue
        comparator = node.test.comparators[0]
        elts = getattr(comparator, "elts", [])
        named = {
            e.attr
            for e in elts
            if isinstance(e, ast.Attribute)
            and isinstance(e.value, ast.Name)
            and e.value.id == "ConsolidationAction"
        }
        if named == set(members):
            return node
    raise AssertionError(f"no `action in (...)` tuple-membership `if` naming {members} found")


class TestArbitratorActionIdentityAllowlist:
    """`_dispatch_consolidation`'s own body is the one place action-identity
    comparisons are allowed; the allowlisted regions are: the ``_scheduled =
    action is ConsolidationAction.AUTO`` binding, the ``INTERIM``-at-zero-
    count tier check, and the two tail dispatch-table arms
    (FULL/RECONCILE and CALIBRATE/CALIBRATE_PENDING) -- each located
    structurally (never by a hardcoded line number), so the allowlist
    tracks the code even as it moves."""

    def test_every_action_identity_check_falls_inside_one_of_the_four_anchor_statements(
        self,
    ) -> None:
        func = _find_function(_APP_TREE, "_dispatch_consolidation")
        anchors = [
            _anchor_scheduled_assign(func),
            _anchor_interim_zero_check(func),
            _anchor_tuple_membership_if(func, "FULL", "RECONCILE"),
            _anchor_tuple_membership_if(func, "CALIBRATE", "CALIBRATE_PENDING"),
        ]
        spans = [(a.lineno, a.end_lineno) for a in anchors]

        hits = _action_identity_comparisons(func)
        assert hits, "sanity: the scan must find at least one comparison"
        for hit in hits:
            assert any(start <= hit.lineno <= end for start, end in spans), (
                f"action-identity check at line {hit.lineno} falls outside every "
                f"anchor span {spans} -- a new comparison was added outside the "
                f"four allowlisted regions"
            )

    def test_stages_event_attribute_gates_exactly_the_documented_staging_stages(
        self,
    ) -> None:
        """Every ``★``-gated stage (resume, tier-binding, retiring triage)
        reads ``action.stages_event`` -- counted via AST over the function's
        real statements (``func.body[1:]``, the docstring statement
        dropped), never a substring search over the function's source text,
        which a docstring mentioning the same identifier in prose would
        satisfy without any code doing so."""
        func = _find_function(_APP_TREE, "_dispatch_consolidation")
        count = 0
        for stmt in func.body[1:]:
            for node in ast.walk(stmt):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "stages_event"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "action"
                ):
                    count += 1
        # Three real gates: resume-pending-first, the tier-binding-
        # unverified gate, and the retiring-triage retirement side effect.
        # The stamp and overdue-incident stages are gated on `_scheduled`
        # (AUTO is the sole requester of AUTO, and AUTO is itself a staging
        # action) rather than a literal `action.stages_event` read, so they
        # are not counted here.
        assert count == 3, f"expected exactly 3 real `action.stages_event` gates; found {count}"


# ---------------------------------------------------------------------------
# Sanity: the guard functions above actually parse the live module (not a
# stale cached source) -- a change to app.py's import path would otherwise
# make every test above vacuously pass against an empty tree.
# ---------------------------------------------------------------------------


def test_app_py_tree_is_nonempty() -> None:
    assert len(_APP_TREE.body) > 100, "app.py's parsed AST looks suspiciously small"


# ---------------------------------------------------------------------------
# The full cycle's consume-pending pre-stage restores voice AFTER
# ``loop.consolidate(...)``, never between the extraction pre-stage and the
# fold.  ``_run_full_cycle`` is a nested closure inside
# ``_run_full_consolidation_sync``; no voice-restore call may appear between
# its ``_extract_pending_sessions(`` call and its ``loop.consolidate(`` call.
# ---------------------------------------------------------------------------


class TestFullCycleVoiceRestoreAfterTheFold:
    def _find_nested_function(self, outer_name: str, inner_name: str) -> ast.FunctionDef:
        outer = _find_function(_APP_TREE, outer_name)
        for node in ast.walk(outer):
            if isinstance(node, ast.FunctionDef) and node.name == inner_name:
                return node
        raise AssertionError(f"{inner_name} not found nested inside {outer_name}")

    def _first_call_lineno(self, func: ast.FunctionDef, callee_substring: str) -> int:
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                target = node.func
                name = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else getattr(target, "id", None)
                )
                if name == callee_substring:
                    return node.lineno
        raise AssertionError(f"no call to {callee_substring!r} found")

    def test_no_voice_restore_between_extraction_and_the_fold_call(self) -> None:
        run_full_cycle = self._find_nested_function(
            "_run_full_consolidation_sync", "_run_full_cycle"
        )
        extraction_line = self._first_call_lineno(run_full_cycle, "_extract_pending_sessions")
        fold_line = self._first_call_lineno(run_full_cycle, "consolidate")
        assert extraction_line < fold_line, (
            "sanity: extraction must precede the fold call in source order"
        )

        for node in ast.walk(run_full_cycle):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if name in ("_end_voice_eviction", "_set_voice_pipeline_profile"):
                assert not (extraction_line < node.lineno < fold_line), (
                    f"a voice-restore call ({name}, line {node.lineno}) appears between "
                    f"extraction (line {extraction_line}) and the fold call (line {fold_line}); "
                    f"the restore must fire only AFTER the fold, at _run_stage_b_cycle's own "
                    f"worker wrapper"
                )
