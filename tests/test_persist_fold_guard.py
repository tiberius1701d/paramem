"""Structural guard: ``_run_fold`` must not call persist functions directly.

Every persist tail in :meth:`ConsolidationLoop._run_fold` routes through
:meth:`ConsolidationLoop._persist_fold`.  Direct calls to
``save_memory_to_disk``, ``commit_tier_slot``, or ``self._save_adapters``
inside the body of ``_run_fold`` are forbidden — they would re-introduce the
fragmentation that the unified dispatch eliminates.

A second, narrower guard forbids a direct registry write inside
``_run_fold`` too: ``self.store.registry(t).save(...)`` /
``.save_from_bytes(...)``.  The registry write is part of the same durable
act as the tier's payload (weights or ``graph.json``) and lives inside
``_persist_fold`` (via ``_save_adapters`` or
:func:`~paramem.memory.persistence.restamp_tier_manifest`) — a registry
write back in ``_run_fold`` would re-introduce the unstamped rewrite that
used to run ahead of (and independently of) the persist act.

The guard uses Python's AST to locate the ``_run_fold`` function body and
scan for ``ast.Call`` nodes whose target matches the forbidden names.  This
prevents drift: the two fold scopes (``interim_slot`` / ``main_tiers``) are
each written in two venues (weights / disk), and every one of those four
combinations must reach disk through ``_persist_fold`` — never through an
inline tail added back into the spine.

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


_REGISTRY_WRITE_METHODS = frozenset({"save", "save_from_bytes"})


def _find_forbidden_registry_writes(
    func_node: ast.FunctionDef,
) -> list[tuple[int, str]]:
    """Return ``(lineno, attr)`` for every direct registry-write call in *func_node*.

    Matches narrowly: ``Call(func=Attribute(attr in {"save",
    "save_from_bytes"}, value=Call(func=Attribute(attr="registry"))))`` —
    i.e. ``<anything>.registry(<tier>).save(...)`` or
    ``.save_from_bytes(...)``.  This flags ``self.store.registry(t).save(...)``
    without flagging every unrelated ``.save()`` call in the function (e.g.
    ``atomic_save_adapter`` internals are a different callee entirely, and
    are not reached by this narrow pattern in the first place).
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _REGISTRY_WRITE_METHODS):
            continue
        receiver = func.value
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr == "registry"
        ):
            hits.append((node.lineno, func.attr))
    return hits


def test_run_fold_has_no_direct_registry_write() -> None:
    """``_run_fold`` must contain no direct ``registry(...).save(...)`` call.

    The registry write for a main tier or an interim slot happens inside
    ``_persist_fold`` (via ``_save_adapters``'s registry-last atomic commit
    or ``restamp_tier_manifest``'s no-retrain commit) — never as a separate,
    unstamped rewrite ahead of or outside that call.
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

    hits = _find_forbidden_registry_writes(run_fold_node)
    assert not hits, (
        "_run_fold contains a direct registry(...).save(...) call outside"
        " _persist_fold.\nRoute the registry write through _persist_fold instead:\n\n"
        + "\n".join(
            f"  {_CONSOLIDATION_PATH.relative_to(_REPO_ROOT)}:{lineno}  (call: .{name}(...))"
            for lineno, name in hits
        )
    )


def test_run_fold_has_no_direct_persist_calls() -> None:
    """``_run_fold`` must contain no direct calls to persist functions.

    Every persist action — both fold scopes (interim_slot / main_tiers) in both
    venues (weights / disk) — must flow through ``_persist_fold``.  A direct call
    here means an inline persist tail has been re-introduced, fragmenting the
    unified dispatch.

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
