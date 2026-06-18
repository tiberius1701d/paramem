"""Structural guard: experiments must not import private (underscored) symbols
from production modules.

The boundary between ``experiments/`` and ``paramem/`` is the curated façade
(:mod:`experiments.utils.production`) plus direct imports of *public*
``paramem.*`` symbols.  Underscored names are private — they may move, change
signature, or vanish in a refactor without notice, and a research script
silently breaking weeks later is exactly the failure mode this guard prevents.

The guard pairs with the retired-symbol stubs in
:mod:`experiments.utils.test_harness` (which catch the inverse mistake:
importing a name the harness used to export but no longer does).

This is an AST-level scan, not a runtime check.  Imports that pass the test
file's syntax check pass the guard regardless of whether the script's larger
runtime path works — that is by design.  Repair-vs-retire of any specific
experiment is a separate decision, tracked outside this file.

Allowlists
----------
Existing private imports are *grandfathered* in :data:`_GRANDFATHERED_IMPORTS`.
Existing public imports are *grandfathered* in :data:`_GRANDFATHERED_PUBLIC_IMPORTS`.
Both allowlists have the shape ``(experiment_path, module, symbol)``.  Adding a
new entry requires *either* removing it by routing through
:mod:`experiments.utils.production` *or* explicitly extending the allowlist
with a comment justifying it.  New entries should be rare — the goal is for
the allowlists to shrink over time, not grow.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# Path prefix for the boundary-layer modules that legitimately import from
# paramem.* — they ARE the façade, so the public-import guard skips them.
_UTILS_PREFIX = EXPERIMENTS_ROOT / "utils"


def _is_boundary_layer(py_file: Path) -> bool:
    """Return True when *py_file* lives under ``experiments/utils/``."""
    try:
        py_file.relative_to(_UTILS_PREFIX)
        return True
    except ValueError:
        return False


# Existing private imports as of 2026-05-28 (9 imports across 4 files).
# Format: (relative_path_from_repo_root, module, symbol_name).
# Each entry is a B2-tracked candidate for either (a) promotion to public,
# (b) rewriting the call site through a public path, or (c) explicit retire.
_GRANDFATHERED_IMPORTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # smoke_graph_enrichment.py — reaches into the SOTA enrichment helpers
        # for direct probing.  Migrate to ExtractionPipeline once it exposes a
        # debug surface (B2).
        (
            "experiments/smoke_graph_enrichment.py",
            "paramem.graph.extractor",
            "_graph_enrich_with_sota",
        ),
        (
            "experiments/smoke_graph_enrichment.py",
            "paramem.training.consolidation",
            "_safe_to_merge_surface",
        ),
        # test11_adapter_extraction.py — calls the raw extractor internals.
        # CLAUDE.md says all extraction must go through ExtractionPipeline.run;
        # this experiment violates that.  Rewrite (B2).
        (
            "experiments/test11_adapter_extraction.py",
            "paramem.graph.extractor",
            "_generate_extraction",
        ),
        (
            "experiments/test11_adapter_extraction.py",
            "paramem.graph.extractor",
            "_parse_extraction",
        ),
        # test16_repair_sweep.py / test18_probe_batching.py — adapter-slot
        # plumbing internals.  Promote to public or wrap via a load_adapter
        # variant that returns the slot path (B2).
        ("experiments/test16_repair_sweep.py", "paramem.models.loader", "_adapter_slot_for_load"),
        ("experiments/test18_probe_batching.py", "paramem.models.loader", "_adapter_slot_for_load"),
        # experiments/utils/early_stop.py — already a thin re-export of the
        # private early_stop state.  Either inline into the harness or promote
        # the underlying symbols (B2).
        ("experiments/utils/early_stop.py", "paramem.training.early_stop", "_EarlyStopState"),
    }
)

# Existing public paramem.* imports in consumer experiments as of 2026-06-15
# (96 imports across 20 files, all outside experiments/utils/).
# Format: (relative_path_from_repo_root, module, symbol_name).
# This allowlist is expected to SHRINK, not grow: new experiment code should
# import through the experiments.utils.production façade instead.  Adding a
# new entry requires either (a) first exposing the symbol via
# experiments.utils.production or (b) an explicit justifying comment here.
_GRANDFATHERED_PUBLIC_IMPORTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # fmt: off
        # dataset_probe.py
        ("experiments/dataset_probe.py", "paramem.memory.store", "MemoryStore"),
        ("experiments/dataset_probe.py", "paramem.models.loader", "load_base_model"),
        ("experiments/dataset_probe.py", "paramem.server.config", "load_server_config"),
        (
            "experiments/dataset_probe.py",
            "paramem.server.consolidation",
            "create_consolidation_loop",
        ),
        # lme_graph_builder.py
        ("experiments/lme_graph_builder.py", "paramem.models.loader", "load_base_model"),
        ("experiments/lme_graph_builder.py", "paramem.server.config", "load_server_config"),
        (
            "experiments/lme_graph_builder.py",
            "paramem.server.consolidation",
            "create_consolidation_loop",
        ),
        # lme_qa_from_triples_probe.py
        (
            "experiments/lme_qa_from_triples_probe.py",
            "paramem.evaluation.recall",
            "generate_answer",
        ),
        (
            "experiments/lme_qa_from_triples_probe.py",
            "paramem.models.loader",
            "adapt_messages",
        ),
        (
            "experiments/lme_qa_from_triples_probe.py",
            "paramem.models.loader",
            "load_base_model",
        ),
        # probe_adapter.py
        ("experiments/probe_adapter.py", "paramem.models.loader", "unload_model"),
        # quadruple_adapter.py
        (
            "experiments/quadruple_adapter.py",
            "paramem.backup.age_envelope",
            "is_age_envelope",
        ),
        (
            "experiments/quadruple_adapter.py",
            "paramem.backup.checkpoint_shard",
            "materialize_checkpoint_to_shm",
        ),
        ("experiments/quadruple_adapter.py", "paramem.memory.entry", "assign_keys"),
        ("experiments/quadruple_adapter.py", "paramem.memory.entry", "build_registry"),
        (
            "experiments/quadruple_adapter.py",
            "paramem.memory.entry",
            "format_entry_training",
        ),
        ("experiments/quadruple_adapter.py", "paramem.models.loader", "create_adapter"),
        ("experiments/quadruple_adapter.py", "paramem.models.loader", "switch_adapter"),
        (
            "experiments/quadruple_adapter.py",
            "paramem.training.early_stop",
            "EarlyStopPolicy",
        ),
        (
            "experiments/quadruple_adapter.py",
            "paramem.training.early_stop",
            "safe_write_json",
        ),
        (
            "experiments/quadruple_adapter.py",
            "paramem.training.recall_eval",
            "probe_entries",
        ),
        ("experiments/quadruple_adapter.py", "paramem.training.trainer", "TrainingHooks"),
        ("experiments/quadruple_adapter.py", "paramem.training.trainer", "train_adapter"),
        ("experiments/quadruple_adapter.py", "paramem.utils.config", "AdapterConfig"),
        ("experiments/quadruple_adapter.py", "paramem.utils.config", "TrainingConfig"),
        # smoke_graph_enrichment.py
        (
            "experiments/smoke_graph_enrichment.py",
            "paramem.graph.extractor",
            "PROVIDER_KEY_ENV",
        ),
        (
            "experiments/smoke_graph_enrichment.py",
            "paramem.training.consolidation",
            "serialize_subgraph_triples",
        ),
        # smoke_interim_rollover_enrichment.py
        (
            "experiments/smoke_interim_rollover_enrichment.py",
            "paramem.graph.extraction_pipeline",
            "ExtractionConfig",
        ),
        (
            "experiments/smoke_interim_rollover_enrichment.py",
            "paramem.graph.extraction_pipeline",
            "ExtractionPipeline",
        ),
        (
            "experiments/smoke_interim_rollover_enrichment.py",
            "paramem.training.consolidation",
            "ConsolidationLoop",
        ),
        # smoke_interim_rollover_live_gpu.py
        (
            "experiments/smoke_interim_rollover_live_gpu.py",
            "paramem.models.loader",
            "load_base_model",
        ),
        (
            "experiments/smoke_interim_rollover_live_gpu.py",
            "paramem.server.config",
            "load_server_config",
        ),
        (
            "experiments/smoke_interim_rollover_live_gpu.py",
            "paramem.server.consolidation",
            "create_consolidation_loop",
        ),
        # smoke_procedural_mlp.py
        ("experiments/smoke_procedural_mlp.py", "paramem.models.loader", "load_base_model"),
        ("experiments/smoke_procedural_mlp.py", "paramem.server.config", "ServerConfig"),
        # smoke_quiet_hours_live_gpu.py
        (
            "experiments/smoke_quiet_hours_live_gpu.py",
            "paramem.server.gpu_lock",
            "gpu_lock_sync",
        ),
        (
            "experiments/smoke_quiet_hours_live_gpu.py",
            "paramem.training.thermal_throttle",
            "ThermalPolicy",
        ),
        (
            "experiments/smoke_quiet_hours_live_gpu.py",
            "paramem.training.thermal_throttle",
            "ThermalThrottleCallback",
        ),
        # test10b_diverse_rephrase.py
        (
            "experiments/test10b_diverse_rephrase.py",
            "paramem.evaluation.recall",
            "generate_answer",
        ),
        ("experiments/test10b_diverse_rephrase.py", "paramem.models.loader", "load_adapter"),
        (
            "experiments/test10b_diverse_rephrase.py",
            "paramem.models.loader",
            "load_base_model",
        ),
        # test11_adapter_extraction.py
        (
            "experiments/test11_adapter_extraction.py",
            "paramem.graph.extractor",
            "load_extraction_prompts",
        ),
        ("experiments/test11_adapter_extraction.py", "paramem.models.loader", "load_adapter"),
        (
            "experiments/test11_adapter_extraction.py",
            "paramem.models.loader",
            "load_base_model",
        ),
        # test16_repair_sweep.py
        ("experiments/test16_repair_sweep.py", "paramem.adapters", "resolve_adapter_slot"),
        (
            "experiments/test16_repair_sweep.py",
            "paramem.adapters.manifest",
            "build_manifest_for",
        ),
        ("experiments/test16_repair_sweep.py", "paramem.memory.entry", "assign_keys"),
        ("experiments/test16_repair_sweep.py", "paramem.memory.entry", "build_registry"),
        (
            "experiments/test16_repair_sweep.py",
            "paramem.memory.entry",
            "format_entry_training",
        ),
        ("experiments/test16_repair_sweep.py", "paramem.memory.persistence", "load_registry"),
        ("experiments/test16_repair_sweep.py", "paramem.memory.persistence", "save_registry"),
        ("experiments/test16_repair_sweep.py", "paramem.models.loader", "create_adapter"),
        ("experiments/test16_repair_sweep.py", "paramem.models.loader", "save_adapter"),
        ("experiments/test16_repair_sweep.py", "paramem.models.loader", "switch_adapter"),
        ("experiments/test16_repair_sweep.py", "paramem.models.loader", "unload_model"),
        (
            "experiments/test16_repair_sweep.py",
            "paramem.training.recall_eval",
            "evaluate_indexed_recall",
        ),
        ("experiments/test16_repair_sweep.py", "paramem.training.trainer", "train_adapter"),
        ("experiments/test16_repair_sweep.py", "paramem.utils.config", "AdapterConfig"),
        ("experiments/test16_repair_sweep.py", "paramem.utils.config", "TrainingConfig"),
        # test18_probe_batching.py
        (
            "experiments/test18_probe_batching.py",
            "paramem.memory.entry",
            "DEFAULT_CONFIDENCE_THRESHOLD",
        ),
        ("experiments/test18_probe_batching.py", "paramem.memory.entry", "RECALL_TEMPLATE"),
        ("experiments/test18_probe_batching.py", "paramem.memory.entry", "build_registry"),
        ("experiments/test18_probe_batching.py", "paramem.memory.entry", "finalize_recalled"),
        (
            "experiments/test18_probe_batching.py",
            "paramem.memory.entry",
            "format_entry_training",
        ),
        ("experiments/test18_probe_batching.py", "paramem.models.loader", "create_adapter"),
        ("experiments/test18_probe_batching.py", "paramem.models.loader", "load_base_model"),
        ("experiments/test18_probe_batching.py", "paramem.models.loader", "save_adapter"),
        ("experiments/test18_probe_batching.py", "paramem.models.loader", "switch_adapter"),
        (
            "experiments/test18_probe_batching.py",
            "paramem.training.dataset",
            "format_inference_prompt",
        ),
        (
            "experiments/test18_probe_batching.py",
            "paramem.training.recall_eval",
            "derive_stop_ids",
        ),
        (
            "experiments/test18_probe_batching.py",
            "paramem.training.recall_eval",
            "evaluate_indexed_recall",
        ),
        ("experiments/test18_probe_batching.py", "paramem.training.trainer", "train_adapter"),
        ("experiments/test18_probe_batching.py", "paramem.utils.config", "AdapterConfig"),
        ("experiments/test18_probe_batching.py", "paramem.utils.config", "TrainingConfig"),
        # test_distillation_models.py / test_prompt_engineering.py
        (
            "experiments/test_distillation_models.py",
            "paramem.evaluation.embedding_scorer",
            "compute_similarity",
        ),
        (
            "experiments/test_prompt_engineering.py",
            "paramem.evaluation.embedding_scorer",
            "compute_similarity",
        ),
        # test_step6_step7_live_gpu.py
        (
            "experiments/test_step6_step7_live_gpu.py",
            "paramem.models.loader",
            "load_base_model",
        ),
        (
            "experiments/test_step6_step7_live_gpu.py",
            "paramem.server.config",
            "load_server_config",
        ),
        (
            "experiments/test_step6_step7_live_gpu.py",
            "paramem.server.consolidation",
            "create_consolidation_loop",
        ),
        (
            "experiments/test_step6_step7_live_gpu.py",
            "paramem.server.gpu_lock",
            "gpu_lock_sync",
        ),
        # fmt: on
    }
)


def _collect_paramem_imports(py_file: Path) -> list[tuple[int, str, str]]:
    """Return ``(lineno, module, symbol)`` for every ``paramem.*`` ImportFrom node.

    Both private (underscored) and public symbols are returned; callers filter
    by ``symbol.startswith("_")``.  Star imports (``alias.name == '*'``) are
    skipped — they cannot be analyzed statically.
    """
    try:
        text = py_file.read_text()
    except UnicodeDecodeError:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not node.module or not node.module.startswith("paramem"):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            out.append((node.lineno, node.module, alias.name))
    return out


def _collect_private_paramem_imports(py_file: Path) -> list[tuple[int, str, str]]:
    """Return ``(lineno, module, symbol)`` for every private ``paramem.*`` import.

    A private symbol is any name that begins with an underscore.
    Star imports (``from x import *``) are skipped — they have ``alias.name == '*'``
    which is not actually underscored, and they cannot be analyzed statically.
    """
    return [
        (ln, mod, sym) for ln, mod, sym in _collect_paramem_imports(py_file) if sym.startswith("_")
    ]


def test_no_new_private_paramem_imports_in_experiments():
    """Fail if a new ``from paramem.*._symbol`` import appears outside the
    grandfathered allowlist.

    To remove an entry from :data:`_GRANDFATHERED_IMPORTS`, either:

    1. Promote the symbol to public in ``paramem/`` (rename without the leading
       underscore, update all callers), or
    2. Rewrite the experiment to use a public alternative — typically a
       function on :mod:`experiments.utils.production`.

    Then drop the corresponding tuple from the allowlist and re-run this test.
    """
    offenders: list[tuple[str, int, str, str]] = []
    for py_file in sorted(EXPERIMENTS_ROOT.rglob("*.py")):
        parts = py_file.parts
        if "archive" in parts or "__pycache__" in parts:
            continue
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        for lineno, module, symbol in _collect_private_paramem_imports(py_file):
            if (rel, module, symbol) in _GRANDFATHERED_IMPORTS:
                continue
            offenders.append((rel, lineno, module, symbol))

    assert not offenders, (
        "New private-symbol imports from paramem.* found in experiments/.\n"
        "Private (underscored) names are off-limits across the boundary — they "
        "may move or change without notice.\n"
        "Options: (a) promote the symbol to public in paramem/, "
        "(b) rewrite the experiment to use experiments.utils.production, or "
        "(c) extend _GRANDFATHERED_IMPORTS with a justifying comment.\n"
        "Offenders:\n" + "\n".join(f"  {p}:{ln} — from {m} import {s}" for p, ln, m, s in offenders)
    )


def test_allowlist_entries_still_exist():
    """An allowlist entry that no longer matches anything in the tree is dead
    weight — either the experiment was deleted, or the import was already
    rewritten.  Forces the allowlist to track reality.
    """
    actual: set[tuple[str, str, str]] = set()
    for py_file in sorted(EXPERIMENTS_ROOT.rglob("*.py")):
        parts = py_file.parts
        if "archive" in parts or "__pycache__" in parts:
            continue
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        for _, module, symbol in _collect_private_paramem_imports(py_file):
            actual.add((rel, module, symbol))

    stale = sorted(_GRANDFATHERED_IMPORTS - actual)
    assert not stale, (
        "Stale entries in _GRANDFATHERED_IMPORTS (no longer found in the tree). "
        "Remove them:\n" + "\n".join(f"  {p}: from {m} import {s}" for p, m, s in stale)
    )


def test_retired_symbol_stubs_steer_callers():
    """The retired-symbol stubs in :mod:`experiments.utils.test_harness` must
    raise ``NotImplementedError`` with a message that names the replacement
    path.  Catches accidental re-deletion or signature drift in the stubs.
    """
    import pytest

    from experiments.utils import test_harness

    retired_names = (
        "distill_session",
        "distill_qa_pairs",
        "train_indexed_keys",
        "evaluate_indexed_recall",
        "evaluate_individual_qa",
        "smoke_test_adapter",
    )

    for name in retired_names:
        fn = getattr(test_harness, name, None)
        assert fn is not None, (
            f"{name!r} stub missing from experiments.utils.test_harness — "
            "the retired-symbol contract is broken."
        )
        with pytest.raises(NotImplementedError, match="retired 2026-05-20"):
            fn()


def test_production_facade_is_pure_reexport():
    """Every name in :data:`experiments.utils.production.__all__` must resolve
    to a non-underscored symbol from ``paramem.*``.

    Guards against accidentally adding a wrapper, defaulting helper, or local
    function under the façade — which would defeat the boundary's purpose.
    """
    from experiments.utils import production

    bad: list[str] = []
    for name in production.__all__:
        if not hasattr(production, name):
            bad.append(f"{name} (missing)")
            continue
        if name.startswith("_"):
            bad.append(f"{name} (underscored — private)")
            continue
        obj = getattr(production, name)
        # Module-origin check only applies to callables and classes —
        # plain constants (strings, floats, ints) have no ``__module__``.
        # The boundary guard above already proves the symbol was imported
        # from a ``paramem.*`` module, which is the actual contract.
        if not (callable(obj) or isinstance(obj, type)):
            continue
        module = getattr(obj, "__module__", "") or ""
        if not module.startswith("paramem"):
            bad.append(f"{name} (resolves to {module!r}, not paramem.*)")

    assert not bad, (
        "experiments.utils.production must be a pure re-export façade. "
        "Issues:\n  " + "\n  ".join(bad)
    )


def test_no_new_public_paramem_imports_in_experiments():
    """Fail if a new public ``from paramem.*`` import appears in a consumer
    experiment outside the grandfathered allowlist.

    Consumer experiments are all ``experiments/*.py`` files NOT under
    ``experiments/utils/`` (the boundary layer legitimately imports from
    paramem.* — that is its job and it is whitelisted).

    To add a new public paramem.* import, either:

    1. Expose the symbol via :mod:`experiments.utils.production` and import it
       from there, or
    2. Extend :data:`_GRANDFATHERED_PUBLIC_IMPORTS` with a comment justifying
       the direct import.

    The goal is for the allowlist to shrink over time, not grow.
    """
    offenders: list[tuple[str, int, str, str]] = []
    for py_file in sorted(EXPERIMENTS_ROOT.rglob("*.py")):
        parts = py_file.parts
        if "archive" in parts or "__pycache__" in parts:
            continue
        if _is_boundary_layer(py_file):
            continue
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        for lineno, module, symbol in _collect_paramem_imports(py_file):
            if symbol.startswith("_"):
                continue
            if (rel, module, symbol) in _GRANDFATHERED_PUBLIC_IMPORTS:
                continue
            offenders.append((rel, lineno, module, symbol))

    assert not offenders, (
        "New public paramem.* imports found in consumer experiments/.\n"
        "New experiment code should import through experiments.utils.production "
        "instead of reaching into paramem.* directly.\n"
        "Options: (a) add the symbol to experiments.utils.production and import "
        "from there, or (b) extend _GRANDFATHERED_PUBLIC_IMPORTS with a "
        "justifying comment.\n"
        "Offenders:\n" + "\n".join(f"  {p}:{ln} — from {m} import {s}" for p, ln, m, s in offenders)
    )


def test_public_allowlist_entries_still_exist():
    """An allowlist entry that no longer matches anything in the tree is dead
    weight — either the experiment was deleted, or the import was already
    rewritten.  Forces :data:`_GRANDFATHERED_PUBLIC_IMPORTS` to track reality.
    """
    actual: set[tuple[str, str, str]] = set()
    for py_file in sorted(EXPERIMENTS_ROOT.rglob("*.py")):
        parts = py_file.parts
        if "archive" in parts or "__pycache__" in parts:
            continue
        if _is_boundary_layer(py_file):
            continue
        rel = py_file.relative_to(REPO_ROOT).as_posix()
        for _, module, symbol in _collect_paramem_imports(py_file):
            if symbol.startswith("_"):
                continue
            actual.add((rel, module, symbol))

    stale = sorted(_GRANDFATHERED_PUBLIC_IMPORTS - actual)
    assert not stale, (
        "Stale entries in _GRANDFATHERED_PUBLIC_IMPORTS (no longer found in "
        "the tree).  Remove them:\n" + "\n".join(f"  {p}: from {m} import {s}" for p, m, s in stale)
    )
