"""Unit tests for production recall-based early stopping.

Covers the wiring at the THREE production-reachable train_adapter call sites:

  - paramem/training/consolidation.py:
      Site #1 run_consolidation_cycle (unified episodic interim path; formerly
              _run_indexed_key_episodic + _train_extracted_into_interim)
      Site #2 _run_indexed_key_procedural
      Site #3 consolidate_interim_adapters (per-tier loop body)
  - paramem/server/active_store_migration.py:
      Site #4 _migrate_tier_simulate_to_train

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
    """Build n synthetic entry-format entries."""
    return [
        {
            "key": f"graph{i + 1}",
            "subject": f"S{i + 1}",
            "predicate": "p",
            "object": f"O{i + 1}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Class A — TestMaybeMakeRecallCallback
# ---------------------------------------------------------------------------


class TestMaybeMakeRecallCallback:
    def test_returns_none_when_flag_off(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=False)
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert cb is None

    def test_returns_none_when_keyed_pairs_empty(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        cb, _state = loop._maybe_make_recall_callback(
            entries=[],
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
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert isinstance(cb, RecallEarlyStopCallback)
        # probe_from_epoch is pinned to the signal floor so we don't pay for
        # pre-floor probes that can never trigger a stop (a single probe is
        # 12-40× more expensive than a training epoch — see consolidation.py
        # ::_maybe_make_recall_callback).  Both fields therefore equal
        # early_stopping_floor for production loops.
        assert cb._policy.probe_from_epoch == 20
        assert cb._policy.signal_from_epoch == 20
        assert cb._policy.window == 5
        assert cb._policy.probe_every_n_epochs == 2

    def test_callback_paths_routed_to_output_dir(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        out = tmp_path / "out"
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
            adapter_name="episodic",
            output_dir=out,
            phase_name="test",
        )
        assert cb._progress_path == out / "progress.json"
        assert cb._epoch_log_path == out / "epoch_log.json"
        assert cb._first_perfect_log_path is None  # production has no per-key log
        assert cb._pause_file is None  # production pause via gpu_lock_sync

    def test_num_epochs_override_propagates_to_callback(self, tmp_path: Path) -> None:
        """Regression: consolidate_interim_adapters trains with refresh_epochs,
        not num_epochs.  When num_epochs != refresh_epochs the forced
        final-epoch probe must fire at refresh_epochs, not at num_epochs.

        Concretely: if an operator sets consolidation.max_epochs (refresh_epochs)
        to a value other than training_config.num_epochs (30), the callback's
        _num_epochs must track the ACTUAL trainer epoch count (refresh_epochs),
        not the stale training_config default.  A stale _num_epochs silently
        skips the forced probe and leaves state.last_per_key with a mid-training
        cadence verdict — wrong registration admission/rejection.
        """
        # num_epochs in TrainingConfig is 30 (default).  Use 20 as refresh_epochs
        # so the mismatch condition from the bug is clearly visible.
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        assert loop.training_config.num_epochs != 20, (
            "test precondition: training_config.num_epochs must differ from the "
            "refresh_epochs value used below (20)"
        )
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="consolidate-episodic",
            num_epochs=20,
        )
        assert isinstance(cb, RecallEarlyStopCallback)
        assert cb._num_epochs == 20, (
            f"callback._num_epochs should be the passed refresh_epochs (20), "
            f"got {cb._num_epochs} (training_config.num_epochs={loop.training_config.num_epochs})"
        )

    def test_num_epochs_default_falls_back_to_training_config(self, tmp_path: Path) -> None:
        """When num_epochs is not passed (callers using training_config.num_epochs),
        the callback's _num_epochs must equal training_config.num_epochs —
        preserving existing behaviour for all non-stage-9 callers.
        """
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert isinstance(cb, RecallEarlyStopCallback)
        assert cb._num_epochs == loop.training_config.num_epochs

    def test_callback_target_registry_built_from_keyed_pairs(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        keyed = _kp(7)
        cb, _state = loop._maybe_make_recall_callback(
            entries=keyed,
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
# smoke (end-to-end path exercised by the live GPU smoke test).
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

    def test_site1_unified_cycle(self) -> None:
        """run_consolidation_cycle is the unified entry for episodic interim training.

        _run_indexed_key_episodic and _train_extracted_into_interim were merged
        into run_consolidation_cycle (Phase 1–3 refactor).  The recall callback
        must be wired here.
        """
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "run_consolidation_cycle",
        )

    def test_site2_procedural(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "_run_indexed_key_procedural",
        )

    def test_site3_consolidate_interim(self) -> None:
        assert self._function_contains_helper(
            PROJECT_ROOT / "paramem/training/consolidation.py",
            "consolidate_interim_adapters",
        )

    def test_site4_migration(self) -> None:
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
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
            adapter_name="episodic",
            output_dir=tmp_path,
            phase_name="test",
        )
        # Mirror the post-refactor call-site pattern.
        callbacks_extra = [cb] if cb is not None else None
        assert callbacks_extra is None

    def test_enabled_passes_recall_callback_only(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        cb, _state = loop._maybe_make_recall_callback(
            entries=_kp(),
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


# ---------------------------------------------------------------------------
# Class G — TestCallbackStateTuple
# Tests for the (callback, state) return seam.
# ---------------------------------------------------------------------------


class TestCallbackStateTuple:
    """_maybe_make_recall_callback returns (callback, state) so callers can
    read state.last_per_key after training to gate registration."""

    def test_returns_tuple_when_enabled(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        result = loop._maybe_make_recall_callback(
            entries=_kp(3),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert isinstance(result, tuple) and len(result) == 2
        cb, state = result
        assert isinstance(cb, RecallEarlyStopCallback)
        assert state is not None
        # state should initially have no verdict
        assert state.last_per_key is None

    def test_returns_none_none_when_disabled(self, tmp_path: Path) -> None:
        loop = _make_loop(tmp_path, recall_early_stopping=False)
        cb, state = loop._maybe_make_recall_callback(
            entries=_kp(3),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        assert cb is None
        assert state is None

    def test_state_shared_with_callback(self, tmp_path: Path) -> None:
        """The state returned by the helper is the same object the callback uses."""
        loop = _make_loop(tmp_path, recall_early_stopping=True)
        cb, state = loop._maybe_make_recall_callback(
            entries=_kp(2),
            adapter_name="episodic",
            output_dir=tmp_path / "out",
            phase_name="test",
        )
        # Mutate through the callback's internal state; the returned state
        # should reflect the change (same object).
        cb._state.last_per_key = [{"key": "graph1", "exact_match": True}]
        assert state.last_per_key == [{"key": "graph1", "exact_match": True}]


# ---------------------------------------------------------------------------
# Class H — TestRecallPassingKeys
# Tests for the _recall_passing_keys / _probe_passing_keys helpers.
# ---------------------------------------------------------------------------


class TestRecallPassingKeys:
    """_recall_passing_keys converts state.last_per_key to a passing set.
    _probe_passing_keys is the fail-safe path when no verdict is available.
    """

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

    def test_recall_passing_keys_returns_passing_set(self, tmp_path: Path) -> None:
        from paramem.training.early_stop import _EarlyStopState

        loop = self._make_loop_with_helpers(tmp_path)
        state = _EarlyStopState()
        state.last_per_key = [
            {"key": "graph1", "exact_match": True},
            {"key": "graph2", "exact_match": False},
            {"key": "graph3", "exact_match": True},
        ]
        entries = [
            {"key": f"graph{i}", "subject": "S", "predicate": "p", "object": "O"}
            for i in range(1, 4)
        ]
        result = loop._recall_passing_keys(state, entries)
        assert result == {"graph1", "graph3"}

    def test_recall_passing_keys_returns_none_when_state_none(self, tmp_path: Path) -> None:
        loop = self._make_loop_with_helpers(tmp_path)
        result = loop._recall_passing_keys(None, _kp(3))
        assert result is None

    def test_recall_passing_keys_returns_none_when_last_per_key_none(self, tmp_path: Path) -> None:
        from paramem.training.early_stop import _EarlyStopState

        loop = self._make_loop_with_helpers(tmp_path)
        state = _EarlyStopState()
        assert state.last_per_key is None
        result = loop._recall_passing_keys(state, _kp(3))
        assert result is None

    def test_probe_passing_keys_calls_evaluate_indexed_recall(self, tmp_path: Path) -> None:
        """_probe_passing_keys calls evaluate_indexed_recall and returns passing set."""
        from unittest.mock import patch

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
            result = loop._probe_passing_keys("episodic", entries)

        mock_eval.assert_called_once()
        assert result == {"graph1"}


# ---------------------------------------------------------------------------
# Class I — TestRegistrationFilter
# Tests for the recall gate filtering registration in the consolidation paths.
# ---------------------------------------------------------------------------


class TestRegistrationFilter:
    """Verify that a failed key is excluded from store.put and simhash registration."""

    def test_reset_main_tier_filters_failed_keys(self, tmp_path: Path) -> None:
        """_reset_main_tier_registries_and_simhashes filters by passing_sets_by_tier.

        A key with exact_match=False must NOT appear in the rebuilt registry.
        """
        from unittest.mock import MagicMock, patch

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.training_config = TrainingConfig(
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop._thermal_policy = None

        # Build a real MemoryStore so registry/simhash calls work.
        from paramem.memory.store import MemoryStore

        store = MemoryStore()
        loop.store = store

        tier_keyed = {
            "episodic": [
                {"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"},
                {"key": "graph2", "subject": "S2", "predicate": "p", "object": "O2"},
            ],
            "semantic": [],
            "procedural": [],
        }
        # graph2 failed recall; only graph1 should be registered
        passing_sets = {
            "episodic": {"graph1"},
            "semantic": None,  # no keys, probe wouldn't fire
            "procedural": None,  # no keys
        }

        # For the None-verdict tiers (semantic/procedural with no entries),
        # the helper would normally call _probe_passing_keys but keyed is [],
        # so the loop continues early.  Patch _probe_passing_keys to ensure
        # it is NOT called (tier with empty keyed skips the probe).
        with patch.object(ConsolidationLoop, "_probe_passing_keys") as mock_probe:
            loop._reset_main_tier_registries_and_simhashes(tier_keyed, passing_sets)
            # No probe needed for empty tiers
            mock_probe.assert_not_called()

        epi_reg = store.registry("episodic")
        assert "graph1" in epi_reg
        assert "graph2" not in epi_reg, "graph2 failed recall gate and must not be in the registry"

    def test_reset_main_tier_probes_when_verdict_none(self, tmp_path: Path) -> None:
        """When passing_sets_by_tier has None for a tier, _probe_passing_keys is called."""
        from unittest.mock import patch

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.training_config = TrainingConfig(
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop.store = MemoryStore()

        tier_keyed = {
            "episodic": [
                {"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"},
            ],
            "semantic": [],
            "procedural": [],
        }
        # None verdict for episodic → must invoke _probe_passing_keys
        passing_sets = {"episodic": None, "semantic": None, "procedural": None}

        # Fake probe returns the key as passing
        with patch.object(
            ConsolidationLoop,
            "_probe_passing_keys",
            return_value={"graph1"},
        ) as mock_probe:
            loop._reset_main_tier_registries_and_simhashes(tier_keyed, passing_sets)
            # Probe must be called for episodic (non-empty tier with None verdict)
            called_tiers = [call.args[0] for call in mock_probe.call_args_list]
            assert "episodic" in called_tiers

        assert "graph1" in loop.store.registry("episodic")

    def test_none_verdict_never_wipes_tier(self, tmp_path: Path) -> None:
        """Fail-safe: a None verdict must NOT result in an empty registry.

        This is the drop-all guard: if _probe_passing_keys is not invoked for
        a non-empty tier when the verdict is None, all keys would be dropped.
        """
        from unittest.mock import patch

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.training_config = TrainingConfig(
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop.store = MemoryStore()

        tier_keyed = {
            "episodic": [
                {"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"},
                {"key": "graph2", "subject": "S2", "predicate": "p", "object": "O2"},
            ],
            "semantic": [],
            "procedural": [],
        }
        passing_sets = {"episodic": None, "semantic": None, "procedural": None}

        # Probe returns both keys as passing — registry must be non-empty
        with patch.object(
            ConsolidationLoop,
            "_probe_passing_keys",
            return_value={"graph1", "graph2"},
        ):
            loop._reset_main_tier_registries_and_simhashes(tier_keyed, passing_sets)

        reg = loop.store.registry("episodic")
        assert "graph1" in reg
        assert "graph2" in reg, "Fail-safe: None verdict must invoke the probe, not drop all keys"
