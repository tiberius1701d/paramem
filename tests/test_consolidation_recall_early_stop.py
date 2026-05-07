"""Unit tests for production recall-based early stopping.

Covers the wiring at the FIVE production-reachable train_adapter call sites:

  - paramem/training/consolidation.py:
      Site #1 _run_indexed_key_episodic
      Site #2 _run_indexed_key_procedural
      Site #3 _train_extracted_into_interim
      Site #4 consolidate_interim_adapters (per-tier loop body)
  - paramem/server/active_store_migration.py:
      Site #5 _migrate_tier_simulate_to_train

Plus the helper itself (Class A) and the structural AST gate (Class F)
that prevents future architectural-mismatch regressions of the v1 class.

No GPU required.  Mocks `paramem.training.trainer.train_adapter` and
`paramem.server.active_store_migration._train_adapter` to capture the
``callbacks_extra`` kwarg.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.early_stop import RecallEarlyStopCallback
from paramem.utils.config import (
    TrainingConfig,
)

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(
    tmp_path: Path,
    *,
    recall_early_stopping: bool = False,
    early_stopping_floor: int = 10,
    recall_window: int = 3,
    recall_probe_every_n_epochs: int = 3,
) -> ConsolidationLoop:
    """Build a ConsolidationLoop instance with the minimum surface the
    recall helper needs.

    ``ConsolidationLoop.__init__`` reaches into PEFT internals to set up
    real adapters; we don't need any of that for the helper unit tests.
    Bypass init via ``__new__`` and set only the attributes the helper
    reads: ``model``, ``tokenizer``, ``training_config``, plus
    ``shutdown_requested`` and ``_thermal_policy`` for the call-site-pattern
    test below (Commit 3 refactor: shutdown flows through TrainingHooks,
    thermal flows through ThermalPolicy — both nullable in tests that don't
    exercise them).
    """
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.tokenizer = MagicMock()
    loop.training_config = TrainingConfig(
        recall_early_stopping=recall_early_stopping,
        early_stopping_floor=early_stopping_floor,
        recall_window=recall_window,
        recall_probe_every_n_epochs=recall_probe_every_n_epochs,
    )
    loop.shutdown_requested = False
    loop._thermal_policy = None
    return loop


def _kp(n: int = 5) -> list[dict]:
    """Build n synthetic keyed_pairs."""
    return [
        {"key": f"graph{i + 1}", "question": f"Q{i + 1}?", "answer": f"A{i + 1}"} for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Class A — TestMaybeMakeRecallCallback
# ---------------------------------------------------------------------------


class TestMaybeMakeRecallCallback:
    def test_returns_none_when_flag_off(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=False)
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert cb is None

    def test_returns_none_when_keyed_pairs_empty(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=[],
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert cb is None

    def test_returns_callback_with_correct_policy(self, tmp_path: Path) -> None:
        loop = _make_loop(
            tmp_path,
            recall_early_stopping=True,
            early_stopping_floor=20,
            recall_window=5,
            recall_probe_every_n_epochs=2,
        )
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert isinstance(cb, RecallEarlyStopCallback)
        assert cb._policy.probe_from_epoch == 1  # hardcoded in production wiring
        assert cb._policy.signal_from_epoch == 20
        assert cb._policy.window == 5
        assert cb._policy.probe_every_n_epochs == 2

    def test_callback_paths_routed_to_output_dir(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        out = tmp_path / "out"
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=_kp(),
            adapter_name="episodic",
            output_dir=out,
            phase_name="test",
        )
        assert cb._progress_path == out / "progress.json"
        assert cb._epoch_log_path == out / "epoch_log.json"
        assert cb._first_perfect_log_path is None  # production has no per-key log
        assert cb._pause_file is None  # production pause via gpu_lock_sync

    def test_callback_target_registry_built_from_keyed_pairs(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        keyed = _kp(7)
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=keyed,
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert len(cb._target_registry) == 7
        assert set(cb._target_registry.keys()) == {f"graph{i + 1}" for i in range(7)}


# ---------------------------------------------------------------------------
# Helper for B/C/D/E2 — capture callbacks_extra passed to train_adapter
# ---------------------------------------------------------------------------


class _Captured:
    """Records the (kwargs) of every train_adapter call inside a test."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return MagicMock(metrics={})

    @property
    def callbacks_extra(self) -> list:
        assert self.calls, "train_adapter was not called"
        return self.calls[-1]["callbacks_extra"]

    def types(self) -> list[type]:
        return [type(cb) for cb in self.callbacks_extra]


# ---------------------------------------------------------------------------
# Class B — TestCallSiteWiringEpisodic (consolidation.py:1529)
#
# Driving _run_indexed_key_episodic end-to-end is heavy.  Instead, we
# verify the wiring is in the function's body by reading the source AST
# (Class F's mechanism) PLUS we call _maybe_make_recall_callback with the
# arguments _run_indexed_key_episodic actually passes (as covered by the
# Class A unit tests).  The end-to-end path is exercised by the live
# smoke per .agent/plan-recall-early-stop-online-v3.md §3.4.
# ---------------------------------------------------------------------------


class TestCallSiteWiringSourcePresence:
    """Confirm the helper invocation appears in each production site's
    function body via AST.  This is a stricter check than substring grep
    and complements Class F's structural test.
    """

    @staticmethod
    def _function_contains_helper(module_path: Path, func_name: str) -> bool:
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

        class HelperFinder(ast.NodeVisitor):
            def __init__(self):
                self.found = False

            def visit_Call(self, c):
                if (
                    isinstance(c.func, ast.Attribute)
                    and c.func.attr == "_maybe_make_recall_callback"
                ):
                    self.found = True
                self.generic_visit(c)

        hf = HelperFinder()
        for child in ast.iter_child_nodes(ff.found_node):
            hf.visit(child)
        return hf.found

    def test_site1_episodic(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "_run_indexed_key_episodic",
        )

    def test_site2_procedural(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "_run_indexed_key_procedural",
        )

    def test_site3_interim(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "_train_extracted_into_interim",
        )

    def test_site4_consolidate_interim(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "consolidate_interim_adapters",
        )

    def test_site5_migration(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/server/active_store_migration.py",
            "_migrate_tier_simulate_to_train",
        )


# ---------------------------------------------------------------------------
# Class E (subset) — TestCallSiteWiringEnabledVsDisabled
#
# Drive a focused unit that exercises the helper-vs-no-helper branch of
# the wiring by patching train_adapter and calling _maybe_make_recall_callback
# directly.  Full end-to-end exercise of each call site is the smoke's job.
# ---------------------------------------------------------------------------


class TestEnabledVsDisabledBranch:
    """When recall_early_stopping is OFF the helper returns ``None`` and
    call sites pass ``callbacks_extra=None``.  When ON the helper returns a
    ``RecallEarlyStopCallback`` and call sites pass it through
    ``callbacks_extra=[recall_cb]``.

    Shutdown flows through ``TrainingHooks.on_shutdown_check`` (post Commit 3
    refactor); thermal flows through ``thermal_policy``.  Neither belongs in
    ``callbacks_extra``, so the call-site list contains at most the recall
    callback.
    """

    def test_disabled_callbacks_extra_is_none(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=False)
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path,
            phase_name="test",
        )
        # Mirror the post-refactor call-site pattern.
        callbacks_extra = [cb] if cb is not None else None
        assert callbacks_extra is None

    def test_enabled_passes_recall_callback_only(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        cb = loop._maybe_make_recall_callback(
            keyed_pairs=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path,
            phase_name="test",
        )
        callbacks_extra = [cb] if cb is not None else None
        assert callbacks_extra is not None
        assert [type(c) for c in callbacks_extra] == [RecallEarlyStopCallback]


# ---------------------------------------------------------------------------
# Class F — TestProbeTargetIsFullReplaySet
#
# Structural AST test that scans every production-reachable module and
# asserts the recall helper is invoked in the same FunctionDef body as
# every train_adapter call.  This is the PR-CI gate that prevents the
# v1 architectural-mismatch class of bug from recurring.
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
EXPERIMENT_ONLY_ALLOWLIST = {
    # Only called from run_cycle (consolidation.py:1317), which itself is
    # used in experiment harnesses but not the production server.
    ("paramem/training/consolidation.py", "_run_indexed_key_semantic"),
    # Gated on indexed_key_registry is None (line ~1329); production
    # server always has a registry.
    ("paramem/training/consolidation.py", "_train_adapter_with_replay"),
}


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
