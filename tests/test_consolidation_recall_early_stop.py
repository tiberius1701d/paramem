"""Tests for the recall-callback wiring recall-based early stopping depends
on, and for ``ConsolidationLoop._probe_recall`` itself.

Every production training path routes through the single shared funnel
``ConsolidationLoop._train_tier_adapter``, which is the sole call site of
both ``_maybe_make_recall_callback`` and ``train_adapter``:

  - paramem/training/consolidation.py:
      run_consolidation_cycle (unified episodic+procedural interim path)
      the full fold, ConsolidationLoop.consolidate (per-tier loop body)
  - paramem/server/active_store_migration.py:
      _migrate_tier_simulate_to_train (routed through the funnel so the
      per-fold training-budget derivation applies here too)

The structural AST gate (``TestProbeTargetIsFullReplaySet``) asserts that
every production ``train_adapter`` call site has
``_maybe_make_recall_callback`` wired in the same function body.

No GPU required.  ``TestCallSiteWiringSourcePresence`` and
``TestProbeTargetIsFullReplaySet`` parse the production modules' source with
``ast`` and never import or run the scanned code.  ``TestProbeRecall`` mocks
``paramem.training.recall_eval.evaluate_indexed_recall`` against a
``ConsolidationLoop`` built via ``__new__`` (no ``__init__`` side effects).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.training.consolidation import ConsolidationLoop

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# TestCallSiteWiringSourcePresence
#
# The recall callback is wired only through the single shared helper
# _train_tier_adapter, which every production path calls
# (run_consolidation_cycle, the full fold, and
# active_store_migration._migrate_tier_simulate_to_train). No production
# call site wires the callback directly.
#
# The invariant is two-part:
#   1. _train_tier_adapter calls _maybe_make_recall_callback (the funnel).
#   2. Every production caller (run_consolidation_cycle and the full fold,
#      both via run_build_and_publish's _train_gate_write, and
#      _migrate_tier_simulate_to_train) calls _train_tier_adapter (they use
#      the funnel, not a direct bypass).
#
# TestProbeTargetIsFullReplaySet's structural gate independently
# checks that every function containing a train_adapter call also contains
# _maybe_make_recall_callback in the same body — _train_tier_adapter is the
# sole such function, so the gate continues to enforce the "no training
# site bypasses the recall callback" contract.
# ---------------------------------------------------------------------------


class TestCallSiteWiringSourcePresence:
    """Confirm the helper invocation appears in each production site via AST.

    For sites that call train_adapter directly, check _maybe_make_recall_callback
    is present in their own body.  For sites that delegate through
    _train_tier_adapter, check they call _train_tier_adapter (the funnel),
    and separately verify the funnel itself calls _maybe_make_recall_callback.
    """

    @staticmethod
    def _function_contains_attr_call(module_path: Path, func_name: str, attr: str) -> bool:
        """Return True if the named function's body contains a call to self.<attr>(...)."""
        tree = ast.parse(module_path.read_text())

        class FuncFinder(ast.NodeVisitor):
            def __init__(self):
                self.found_node = None

            def visit_FunctionDef(self, node):
                if node.name == func_name and self.found_node is None:
                    self.found_node = node
                self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

        ff = FuncFinder()
        ff.visit(tree)
        if ff.found_node is None:
            return False

        class AttrCallFinder(ast.NodeVisitor):
            def __init__(self):
                self.found = False

            def visit_Call(self, c):
                if isinstance(c.func, ast.Attribute) and c.func.attr == attr:
                    self.found = True
                self.generic_visit(c)

        hf = AttrCallFinder()
        for child in ast.iter_child_nodes(ff.found_node):
            hf.visit(child)
        return hf.found

    def test_funnel_contains_recall_callback(self) -> None:
        """_train_tier_adapter is the single training-invocation site and must
        call _maybe_make_recall_callback.  Both run_consolidation_cycle and the
        full fold (ConsolidationLoop.consolidate) delegate to it instead of
        calling train_adapter directly.
        """
        assert self._function_contains_attr_call(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "_train_tier_adapter",
            "_maybe_make_recall_callback",
        )

    def test_train_gate_write_calls_funnel(self) -> None:
        """_train_gate_write must call _train_tier_adapter (the funnel), not
        invoke train_adapter directly.  Both run_consolidation_cycle and the
        full fold (ConsolidationLoop.consolidate) delegate their per-tier
        training to it via run_build_and_publish, so a single check on
        _train_gate_write is sufficient.
        """
        assert self._function_contains_attr_call(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "_train_gate_write",
            "_train_tier_adapter",
        )

    def test_site2_unified_cycle_uses_funnel_for_interim(self) -> None:
        """run_consolidation_cycle trains episodic+procedural via
        _train_tier_adapter (the funnel).  Procedural is trained through the
        unified interim slot, not a flat per-cycle procedural train path;
        there is no ``_run_indexed_key_procedural`` function.
        """
        # Routing through the funnel is covered by test_train_gate_write_calls_funnel.
        # This test guards that _run_indexed_key_procedural does not exist in the module.
        import ast

        src = (PROJECT_ROOT / "paramem/training/consolidation.py").read_text()
        tree = ast.parse(src)
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "_run_indexed_key_procedural" not in func_names, (
            "_run_indexed_key_procedural must not exist after unified-interim refactor"
        )
        assert "_prepare_procedural_keys_for_tier" not in func_names, (
            "_prepare_procedural_keys_for_tier must not exist after unified-interim refactor"
        )

    def test_site4_migration_routed_through_funnel(self) -> None:
        """_migrate_tier_simulate_to_train delegates to _train_tier_adapter
        (the funnel) rather than calling train_adapter or
        _maybe_make_recall_callback directly -- the per-fold training
        budget and the recall callback are inherited from the funnel, not
        duplicated at the migration call site.
        """
        assert self._function_contains_attr_call(
            PROJECT_ROOT / "paramem/server/active_store_migration.py",
            "_migrate_tier_simulate_to_train",
            "_train_tier_adapter",
        )
        assert not self._function_contains_attr_call(
            PROJECT_ROOT / "paramem/server/active_store_migration.py",
            "_migrate_tier_simulate_to_train",
            "_maybe_make_recall_callback",
        ), (
            "_migrate_tier_simulate_to_train must not wire "
            "_maybe_make_recall_callback directly any more — it is reached "
            "transitively via _train_tier_adapter"
        )


# ---------------------------------------------------------------------------
# TestProbeTargetIsFullReplaySet
#
# Structural AST test that scans every production-reachable module and
# asserts the recall helper is invoked in the same FunctionDef body as
# every train_adapter call.  This is the PR-CI gate that keeps a
# train_adapter call site from training without the recall callback wired.
# ---------------------------------------------------------------------------


# Modules to scan for production train_adapter calls.  Future contributors
# adding a new production module that imports train_adapter must add the
# module to this list AND wire the helper at every train_adapter call site
# within it.
PRODUCTION_MODULES = [
    "paramem/training/consolidation.py",
    "paramem/server/active_store_migration.py",
]

# Functions allowlisted as experiment-only (NOT subject to the wiring
# requirement).  Each entry is a (module_path, function-name) tuple.
# Verified non-production by tracing all callers transitively from
# paramem/server/app.py endpoints.
EXPERIMENT_ONLY_ALLOWLIST: set[tuple[str, str]] = set()


def _find_train_adapter_calls(tree: ast.AST) -> list[tuple[ast.AST, ast.Call]]:
    """Return [(enclosing_function_node, call_node), ...] for every Call
    to a Name == 'train_adapter' / '_train_adapter' / '_train_adapter_fn'.

    Returns the FunctionDef node identity (not its name) so subsequent
    helper-presence checks operate on the same function instance —
    avoids the "two functions with the same name" ambiguity.

    Both ``FunctionDef`` and ``AsyncFunctionDef`` are walked: a future
    ``await train_adapter(...)`` inside an async handler must not silently
    bypass this check.
    """
    target_names = {"train_adapter", "_train_adapter", "_train_adapter_fn"}
    results: list[tuple[ast.AST, ast.Call]] = []
    func_stack: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            func_stack.append(node)
            self.generic_visit(node)
            func_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name in target_names and func_stack:
                results.append((func_stack[-1], node))
            self.generic_visit(node)

    Visitor().visit(tree)
    return results


def _function_node_contains_helper_call(func_node: ast.AST) -> bool:
    """Walk only the body of ``func_node`` for a Call whose attribute is
    `_maybe_make_recall_callback`.  Operates on node identity so a sibling
    function with the same name in a different class cannot match.
    """

    class HelperVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, c):
            if isinstance(c.func, ast.Attribute) and c.func.attr == "_maybe_make_recall_callback":
                self.found = True
            self.generic_visit(c)

    hv = HelperVisitor()
    for child in ast.iter_child_nodes(func_node):
        hv.visit(child)
    return hv.found


class TestProbeTargetIsFullReplaySet:
    """The structural gate.  Asserts every production-reachable
    train_adapter call site has _maybe_make_recall_callback invoked in
    the same FunctionDef body.
    """

    def test_recall_callback_attached_at_every_production_site(self) -> None:
        failures: list[str] = []
        for module in PRODUCTION_MODULES:
            path = PROJECT_ROOT / module
            tree = ast.parse(path.read_text())
            for func_node, _call in _find_train_adapter_calls(tree):
                if (module, func_node.name) in EXPERIMENT_ONLY_ALLOWLIST:
                    continue
                if not _function_node_contains_helper_call(func_node):
                    failures.append(f"{module}::{func_node.name}")
        assert not failures, (
            "Production-reachable train_adapter call sites missing "
            "_maybe_make_recall_callback wiring:\n  "
            + "\n  ".join(failures)
            + "\n\nIf the call is genuinely experiment-only, add "
            "(module, function) to EXPERIMENT_ONLY_ALLOWLIST in this test "
            "with a one-line rationale comment."
        )

    def test_only_train_tier_adapter_calls_train_adapter(self) -> None:
        """Budget-derivation guard: _train_tier_adapter is the ONLY function
        across the scanned production modules that calls train_adapter.

        _train_tier_adapter is where the per-fold training budget is derived
        (paramem.utils.config.budget_for) and applied via dataclasses.replace.
        A second function calling train_adapter directly would train with an
        un-derived (and possibly stale) budget, bypassing budget_for
        entirely -- this test fails the moment that happens, independent of
        whether the new call site also happens to wire the recall callback.
        """
        enclosing_funcs: set[str] = set()
        for module in PRODUCTION_MODULES:
            path = PROJECT_ROOT / module
            tree = ast.parse(path.read_text())
            for func_node, _call in _find_train_adapter_calls(tree):
                enclosing_funcs.add(func_node.name)
        assert enclosing_funcs == {"_train_tier_adapter"}, (
            "train_adapter must be called ONLY from _train_tier_adapter (the "
            f"single training funnel); found calls in: {sorted(enclosing_funcs)}"
        )

    def test_allowlist_entries_actually_exist(self) -> None:
        """Every allowlist entry must reference a real FunctionDef in the
        listed module.  Catches stale allowlist entries when a function
        is renamed / deleted.
        """
        for module, func_name in EXPERIMENT_ONLY_ALLOWLIST:
            path = PROJECT_ROOT / module
            tree = ast.parse(path.read_text())
            found = False

            class Finder(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    nonlocal found
                    if node.name == func_name:
                        found = True
                    self.generic_visit(node)

                visit_AsyncFunctionDef = visit_FunctionDef

            Finder().visit(tree)
            assert found, (
                f"EXPERIMENT_ONLY_ALLOWLIST entry ({module}, {func_name}) "
                f"references a non-existent function — was it renamed or removed?"
            )


# ---------------------------------------------------------------------------
# TestProbeRecall
# Tests for the ConsolidationLoop._probe_recall primitive.
# ---------------------------------------------------------------------------


class TestProbeRecall:
    """_probe_recall runs the one staged-weights recall probe and returns a
    RecallProbe carrying the per-key verdict."""

    def _make_loop_with_helpers(self, tmp_path: Path) -> "ConsolidationLoop":
        from paramem.utils.config import TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.training_config = TrainingConfig(
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        return loop

    def test_probe_recall_calls_evaluate_indexed_recall(self, tmp_path: Path) -> None:
        """_probe_recall calls evaluate_indexed_recall and wraps the result
        in a RecallProbe carrying per_key verbatim."""
        from unittest.mock import patch

        from paramem.training.recall_eval import RecallProbe

        loop = self._make_loop_with_helpers(tmp_path)
        entries = [
            {"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"},
            {"key": "graph2", "subject": "S2", "predicate": "p", "object": "O2"},
        ]
        fake_result = {
            "exact_count": 1,
            "total": 2,
            "rate": 0.5,
            "mean_confidence": 0.8,
            "per_key": [
                {"key": "graph1", "exact_match": True},
                {"key": "graph2", "exact_match": False},
            ],
        }

        with patch(
            "paramem.training.recall_eval.evaluate_indexed_recall", return_value=fake_result
        ) as mock_eval:
            result = loop._probe_recall("episodic", entries)

        mock_eval.assert_called_once()
        assert isinstance(result, RecallProbe)
        assert result.passing_keys == {"graph1"}
        assert result.per_key == tuple(fake_result["per_key"])

    def test_probe_recall_propagates_exceptions(self, tmp_path: Path) -> None:
        """A probe that cannot run is not a verdict — _probe_recall must not
        swallow the exception into an empty/failing result."""
        from unittest.mock import patch

        loop = self._make_loop_with_helpers(tmp_path)
        entries = [{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}]

        with (
            patch(
                "paramem.training.recall_eval.evaluate_indexed_recall",
                side_effect=RuntimeError("probe harness broke"),
            ),
            pytest.raises(RuntimeError, match="probe harness broke"),
        ):
            loop._probe_recall("episodic", entries)

    def test_probe_recall_re_enables_gradient_checkpointing_when_configured(
        self, tmp_path: Path
    ) -> None:
        """When training_config.gradient_checkpointing is True, the probe
        re-enables it afterward (in a finally) — the probe runs mid-fold,
        before the promote and before the next tier trains, so the pre-probe
        state must be restored rather than left disabled.
        """
        from unittest.mock import patch

        from paramem.utils.config import TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.training_config = TrainingConfig(
            recall_early_stopping=False,
            recall_probe_batch_size=1,
            gradient_checkpointing=True,
        )
        entries = [{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}]
        fake_result = {"per_key": [{"key": "graph1", "exact_match": True}]}

        with patch(
            "paramem.training.recall_eval.evaluate_indexed_recall", return_value=fake_result
        ):
            loop._probe_recall("episodic", entries)

        loop.model.gradient_checkpointing_enable.assert_called_once()

    def test_probe_recall_leaves_checkpointing_off_when_not_configured(
        self, tmp_path: Path
    ) -> None:
        """When training_config.gradient_checkpointing is False, the probe
        does not force it back on."""
        from unittest.mock import patch

        from paramem.utils.config import TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.training_config = TrainingConfig(
            recall_early_stopping=False,
            recall_probe_batch_size=1,
            gradient_checkpointing=False,
        )
        entries = [{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}]
        fake_result = {"per_key": [{"key": "graph1", "exact_match": True}]}

        with patch(
            "paramem.training.recall_eval.evaluate_indexed_recall", return_value=fake_result
        ):
            loop._probe_recall("episodic", entries)

        loop.model.gradient_checkpointing_enable.assert_not_called()
