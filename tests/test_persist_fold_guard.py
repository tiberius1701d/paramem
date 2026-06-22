"""Structural guard: ``_run_fold`` must not call persist functions directly.

After B8b, all persist tails in :meth:`ConsolidationLoop._run_fold` route
through :meth:`ConsolidationLoop._persist_fold`.  Direct calls to
``save_memory_to_disk``, ``commit_tier_slot``, or ``self._save_adapters``
inside the body of ``_run_fold`` are forbidden — they would re-introduce the
fragmentation that this unification eliminates.

The guard uses Python's AST to locate the ``_run_fold`` function body and
scan for ``ast.Call`` nodes whose target matches the forbidden names.  This
prevents drift: a future maintainer who adds a fourth persist venue must
route through ``_persist_fold``, not add a fourth inline tail.

Pattern mirrors :mod:`tests.test_extraction_pipeline_guard`.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_CONSOLIDATION_PATH = _REPO_ROOT / "paramem" / "training" / "consolidation.py"

# Names that must NOT appear as direct calls inside _run_fold.
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "save_memory_to_disk",
        "commit_tier_slot",
        "_save_adapters",
    }
)


def _find_run_fold_body(tree: ast.Module) -> ast.FunctionDef | None:
    """Return the AST node for ``ConsolidationLoop._run_fold``, or ``None``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ConsolidationLoop":
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "_run_fold"
                ):
                    return item  # type: ignore[return-value]
    return None


def _find_forbidden_direct_calls(
    func_node: ast.FunctionDef,
) -> list[tuple[int, str]]:
    """Return ``(lineno, call_name)`` for every forbidden direct call in *func_node*.

    Inspects all :class:`ast.Call` nodes in the function body.  A call is
    "direct" when its ``func`` attribute is an :class:`ast.Name` (bare name
    call like ``save_memory_to_disk(...)``) or an :class:`ast.Attribute` whose
    attribute name matches (like ``self._save_adapters(...)``).

    Calls that occur inside ``_persist_fold`` itself are excluded: we look
    only inside ``_run_fold``, and ``_persist_fold`` is a sibling method — not
    nested inside ``_run_fold`` — so they will not appear in the walk.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        call_name: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALL_NAMES:
            call_name = node.func.attr
        if call_name is not None:
            hits.append((node.lineno, call_name))
    return hits


def test_run_fold_has_no_direct_persist_calls() -> None:
    """``_run_fold`` must contain no direct calls to persist functions.

    All three venue persist actions (graph_json / interim_slot / main_tiers)
    must flow through ``_persist_fold``.  A direct call here means a fourth
    inline persist tail has been re-introduced, fragmenting the unified dispatch.

    Failure message names the line and call so the developer knows exactly
    where to look.
    """
    src = _CONSOLIDATION_PATH.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(_CONSOLIDATION_PATH))
    except SyntaxError as exc:
        raise AssertionError(f"Could not parse {_CONSOLIDATION_PATH}: {exc}") from exc

    run_fold_node = _find_run_fold_body(tree)
    assert run_fold_node is not None, (
        "_run_fold not found in ConsolidationLoop — was it renamed or removed?"
    )

    hits = _find_forbidden_direct_calls(run_fold_node)
    assert not hits, (
        "_run_fold contains direct persist call(s) outside _persist_fold.\n"
        "Route all persist tails through _persist_fold instead:\n\n"
        + "\n".join(
            f"  {_CONSOLIDATION_PATH.relative_to(_REPO_ROOT)}:{lineno}  (call: {name!r})"
            for lineno, name in hits
        )
    )
