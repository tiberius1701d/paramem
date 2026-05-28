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

Allowlist
---------
Existing private imports are *grandfathered* in :data:`_GRANDFATHERED_IMPORTS`.
The allowlist has the shape ``(experiment_path, module, symbol)``.  Adding a
new private import requires *either* removing the entry by promoting the
symbol to public in production (a separate, gated change) *or* explicitly
extending the allowlist with a comment justifying it.  New entries should be
rare — the goal is for the allowlist to shrink over time, not grow.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# Existing private imports as of 2026-05-28 (15 imports across 6 files).
# Format: (relative_path_from_repo_root, module, symbol_name).
# Each entry is a B2-tracked candidate for either (a) promotion to public,
# (b) rewriting the call site through a public path, or (c) explicit retire.
_GRANDFATHERED_IMPORTS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # quadruple_adapter.py — uses _safe_write_json for atomic dumps.
        # Promote to paramem.training.early_stop.safe_write_json (B2).
        ("experiments/quadruple_adapter.py", "paramem.training.early_stop", "_safe_write_json"),
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
        (
            "experiments/smoke_graph_enrichment.py",
            "paramem.training.consolidation",
            "_serialize_subgraph_triples",
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
        ("experiments/test18_probe_batching.py", "paramem.memory.entry", "_finalize_recalled"),
        (
            "experiments/test18_probe_batching.py",
            "paramem.training.dataset",
            "_format_inference_prompt",
        ),
        (
            "experiments/test18_probe_batching.py",
            "paramem.training.recall_eval",
            "_derive_stop_ids",
        ),
        # experiments/utils/early_stop.py — already a thin re-export of the
        # private early_stop state.  Either inline into the harness or promote
        # the underlying symbols (B2).
        ("experiments/utils/early_stop.py", "paramem.training.early_stop", "_EarlyStopState"),
        ("experiments/utils/early_stop.py", "paramem.training.early_stop", "_safe_write_json"),
    }
)


def _collect_private_paramem_imports(py_file: Path) -> list[tuple[int, str, str]]:
    """Return ``(lineno, module, symbol)`` for every private ``paramem.*`` import.

    A private symbol is any name that begins with an underscore.
    Star imports (``from x import *``) are skipped — they have ``alias.name == '*'``
    which is not actually underscored, and they cannot be analyzed statically.
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
            if alias.name.startswith("_"):
                out.append((node.lineno, node.module, alias.name))
    return out


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
