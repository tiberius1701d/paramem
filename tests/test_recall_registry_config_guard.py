"""Structural drift-guard: recall probes must never omit ``registry=``.

:func:`paramem.memory.entry.verify_confidence` returns ``1.0`` unconditionally
when its ``registry`` argument is ``None`` (no verification, not a strict
score) — see ``paramem/memory/entry.py:352-353``.  A production call to
:func:`paramem.training.recall_eval.probe_entries` or
:func:`paramem.memory.probe.probe_keys_grouped_by_adapter` that omits
``registry`` therefore accepts ANY well-formed JSON that echoes the queried
key, regardless of content — a hallucination surface, not a liveness check.

This was live for Gate 3 (``paramem/server/gates.py::_gate_3_reload_smoke``)
until fixed alongside this guard: the call omitted ``registry=`` entirely, so
``finalize_recalled`` -> ``verify_confidence`` always scored 1.0 and the gate
passed on any well-formed JSON echoing the right key, despite its own comment
claiming a "low confidence" check.

This test scans production source files and fails if any call to a watched
function does not supply ``registry`` — neither as a keyword nor positionally
in the parameter slot the function signature reserves for it.

Design notes
------------
- Mirrors ``tests/test_recall_batch_size_config_guard.py`` (AST scan +
  import-alias resolution + explicit allowlist keyed on
  ``(relative_path, function_name)``), including its fixed ``_SCAN_FILES``
  tuple rather than a whole-tree walk — a fixed list is immune to unrelated
  working-tree drift (e.g. a file staged-then-deleted outside this change)
  breaking an unrelated guard test.
- ``registry`` is supplied either as a keyword (``registry=...``) or
  positionally at the slot the function signature reserves for it:
  ``probe_entries(model, tokenizer, entries, registry=None, *, ...)`` —
  slot index 3 (0-based) — and
  ``probe_keys_grouped_by_adapter(model, tokenizer, keys_by_adapter,
  max_new_tokens=200, registry=None, ...)`` — slot index 4.
- ``registry=None`` explicitly passed still counts as "supplied" — the
  caller made a deliberate choice, visible in a diff/review, rather than a
  silent omission.  This guard only catches omission.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Watched functions and their `registry` parameter's positional slot.
# ---------------------------------------------------------------------------

_REGISTRY_SLOT_INDEX: dict[str, int] = {
    "probe_entries": 3,
    "probe_keys_grouped_by_adapter": 4,
}

# ---------------------------------------------------------------------------
# Production source files to scan — every known call site of a watched
# function (grepped against paramem/ at guard-authoring time).
# ---------------------------------------------------------------------------

_SCAN_FILES = (
    "paramem/server/inference.py",
    "paramem/server/app.py",
    "paramem/server/gates.py",
    "paramem/memory/source.py",
    "paramem/memory/probe.py",
    "paramem/training/consolidation.py",
    "paramem/training/recall_eval.py",
    "paramem/graph/reconstruct.py",
)

# ---------------------------------------------------------------------------
# Allowlist — one entry per legitimately-omitted call site
# ---------------------------------------------------------------------------

# Format: frozenset of (relative_path_posix, enclosing_function_name) tuples.
# Empty by design: every current production call site supplies `registry`
# explicitly. Add an entry ONLY for a call site that deliberately wants
# unconditional-pass behaviour (e.g. a pure liveness ping with no
# hallucination-detection requirement) with a comment justifying why.
_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


# ---------------------------------------------------------------------------
# AST helpers (mirrors test_recall_batch_size_config_guard.py)
# ---------------------------------------------------------------------------


def _enclosing_function_name(node: ast.AST, tree: ast.AST) -> str | None:
    """Return the name of the innermost function definition enclosing *node*."""
    parent: dict[int, ast.AST] = {}
    for n in ast.walk(tree):
        for child in ast.iter_child_nodes(n):
            parent[id(child)] = n

    current: ast.AST | None = parent.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parent.get(id(current))
    return None


def _build_alias_map(tree: ast.AST) -> dict[str, str]:
    """Return local name -> canonical watched-function name.

    Walks all ``ast.ImportFrom`` / ``ast.Import`` nodes (including ones
    nested inside function bodies, where production code keeps lazy
    imports) and records aliases for names in ``_REGISTRY_SLOT_INDEX``.
    """
    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                canonical = alias.name
                if canonical in _REGISTRY_SLOT_INDEX:
                    local = alias.asname if alias.asname else alias.name
                    alias_map[local] = canonical
        elif isinstance(node, ast.Import):
            for alias in node.names:
                canonical = alias.name
                if canonical in _REGISTRY_SLOT_INDEX:
                    local = alias.asname if alias.asname else alias.name
                    alias_map[local] = canonical
    return alias_map


def _find_missing_registry_calls(py_file: Path) -> list[tuple[int, str, str | None, str]]:
    """Return offending call sites in *py_file*.

    Returns ``(lineno, canonical_func_name, enclosing_function, source_line)``
    tuples for every call to a watched function that supplies ``registry``
    neither as a keyword nor positionally in its reserved slot.
    """
    try:
        text = py_file.read_text()
    except UnicodeDecodeError:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    alias_map = _build_alias_map(tree)
    out: list[tuple[int, str, str | None, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Name):
            local_name = func.id
        elif isinstance(func, ast.Attribute):
            local_name = func.attr
        else:
            continue

        if local_name in _REGISTRY_SLOT_INDEX:
            canonical_name = local_name
        elif local_name in alias_map:
            canonical_name = alias_map[local_name]
        else:
            continue

        slot = _REGISTRY_SLOT_INDEX[canonical_name]
        has_keyword = any(kw.arg == "registry" for kw in node.keywords)
        # `**kwargs` unpacking (kw.arg is None) could theoretically carry
        # `registry` — treat it as supplied since a static scan cannot
        # resolve it, and flagging it would be a false positive.
        has_star_kwargs = any(kw.arg is None for kw in node.keywords)
        has_enough_positional = len(node.args) > slot or any(
            isinstance(a, ast.Starred) for a in node.args
        )

        if has_keyword or has_star_kwargs or has_enough_positional:
            continue

        enclosing = _enclosing_function_name(node, tree)
        src_line = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
        out.append((node.lineno, canonical_name, enclosing, src_line))

    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_recall_probes_never_omit_registry():
    """No production recall-probe call site may omit ``registry``.

    ``verify_confidence`` returns 1.0 unconditionally when ``registry`` is
    ``None`` — an omitted argument silently defaults to it, so any
    well-formed JSON echoing the right key passes verification regardless
    of content. Fix by threading the applicable SimHash registry (or pass
    ``registry=None`` explicitly if unconditional-pass really is intended),
    or add an allowlist entry with a justifying comment.
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str, str | None, str]] = []

    for rel in _SCAN_FILES:
        py_file = repo_root / rel
        if not py_file.exists():
            continue
        for lineno, func_name, enclosing, src in _find_missing_registry_calls(py_file):
            key = (rel, enclosing)
            if key in _ALLOWLIST:
                continue
            offenders.append((rel, lineno, func_name, enclosing, src))

    assert not offenders, (
        "Recall-probe calls must supply `registry=` — omitting it makes "
        "verify_confidence() return 1.0 unconditionally (no verification), "
        "which accepts ANY well-formed JSON regardless of content.\n"
        "Either pass the applicable SimHash registry, or for a genuinely "
        "unconditional-pass call, pass `registry=None` explicitly (a visible "
        "choice, not a silent omission) or add an entry to _ALLOWLIST in "
        "tests/test_recall_registry_config_guard.py.\n"
        "Offending call sites:\n"
        + "\n".join(
            f"  {path}:{lineno} in {enclosing or '<module>'} — {func_name}(...) — {src}"
            for path, lineno, func_name, enclosing, src in offenders
        )
    )
