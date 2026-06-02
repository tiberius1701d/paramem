"""Structural drift-guard: recall batch size must come from config, not literals.

The production recall pipeline threads ``batch_size`` (or
``recall_probe_batch_size``) from ``config.consolidation.recall_probe_batch_size``
at every call site.  A caller that hard-codes an integer literal bypasses this
and silently diverges from the config value.

This test scans production source files and fails if any call to a
recall-probe function passes ``batch_size=<int-literal>`` or
``recall_probe_batch_size=<int-literal>``.

ALLOWLISTED call:
  ``paramem/server/gates.py :: _gate_3_reload_smoke`` — ``batch_size=1`` on the
  single-key liveness probe is intentional (``probe_entries`` is called with
  exactly one key, so batching is genuinely N/A).  The allowlist is keyed on
  ``(relative_path, function_name)`` and kept tight so it cannot accidentally
  permit real drift.

Design notes
------------
- The scan targets ``ast.Call`` **keywords** only, not function-signature
  ``arguments`` defaults.  Low-level primitive definitions like
  ``probe_entries(... batch_size: int = 1 ...)`` carry a default for
  direct / test use and are NOT caught by this scan.
- Recall-probe function names are enumerated explicitly so unrelated
  ``batch_size=1`` uses (e.g. ``DataLoader``) are not flagged.
- Import aliases are resolved per-file: ``from x import probe_entries as _p``
  makes ``_p`` a watched name.  Module-level and function-local imports are
  both covered by walking all ``ast.ImportFrom`` / ``ast.Import`` nodes in the
  full AST.  The reported function name is the canonical one so the error
  message stays unambiguous regardless of the local alias used.
- Mirror of the extraction-pipeline guard in
  ``tests/test_extraction_pipeline_guard.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Recall-probe function names to watch
# ---------------------------------------------------------------------------

# Every function whose ``batch_size`` / ``recall_probe_batch_size`` kwarg
# must be config-sourced at production call sites.  Extend this set when a
# new recall-probe entry point is added.
_RECALL_PROBE_FUNC_NAMES = frozenset(
    {
        "probe_entries",
        "probe_keys_grouped_by_adapter",
        "WeightMemorySource",
        "evaluate_indexed_recall",
        "evaluate_gates",
        "_gate_4_recall_check",
    }
)

# Keywords whose integer-literal value indicates a drift risk.
_WATCHED_KWARG_NAMES = frozenset({"batch_size", "recall_probe_batch_size"})

# ---------------------------------------------------------------------------
# Allowlist — one entry per legitimately-literal call site
# ---------------------------------------------------------------------------

# Format: frozenset of (relative_path_posix, enclosing_function_name) tuples.
#
# ONLY the gate-3 liveness probe is allowlisted:
#   paramem/server/gates.py :: _gate_3_reload_smoke
#   ``probe_entries(model, tokenizer, [{"key": first_key}], batch_size=1)``
#   — exactly one key, so batching is N/A.  The line carries an explicit
#   comment to that effect.
_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        # single-key liveness probe — batching N/A (1 key)
        ("paramem/server/gates.py", "_gate_3_reload_smoke"),
    }
)

# ---------------------------------------------------------------------------
# Production source files to scan
# ---------------------------------------------------------------------------

_SCAN_FILES = (
    "paramem/server/inference.py",
    "paramem/server/app.py",
    "paramem/server/gates.py",
    "paramem/memory/source.py",
    "paramem/training/consolidation.py",
    "paramem/graph/reconstruct.py",
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _enclosing_function_name(node: ast.AST, tree: ast.AST) -> str | None:
    """Return the name of the innermost function definition enclosing *node*.

    Walks the parent map built from *tree*.  Returns ``None`` when the call
    is at module level (unusual but possible in scripts).
    """
    # Build a parent map once per tree.
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
    """Return a mapping of local name → canonical function name for watched functions.

    Walks all ``ast.ImportFrom`` and ``ast.Import`` nodes in *tree* (including
    those nested inside function bodies, which is where production code keeps
    its lazy imports).  For each alias where the imported canonical name is in
    ``_RECALL_PROBE_FUNC_NAMES``, records ``local_name → canonical_name``.

    Examples::

        from paramem.training.recall_eval import probe_entries as _probe_entries
        # → {"_probe_entries": "probe_entries"}

        from paramem.memory.source import WeightMemorySource as _WeightMemorySource
        # → {"_WeightMemorySource": "WeightMemorySource"}

    Direct (non-aliased) imports where local name == canonical name are also
    included so the lookup is uniform.
    """
    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                canonical = alias.name
                if canonical in _RECALL_PROBE_FUNC_NAMES:
                    local = alias.asname if alias.asname else alias.name
                    alias_map[local] = canonical
        elif isinstance(node, ast.Import):
            # Plain `import foo as bar` — unlikely for these functions, but
            # handle for completeness.
            for alias in node.names:
                canonical = alias.name
                if canonical in _RECALL_PROBE_FUNC_NAMES:
                    local = alias.asname if alias.asname else alias.name
                    alias_map[local] = canonical
    return alias_map


def _find_literal_batch_size_calls(
    py_file: Path,
) -> list[tuple[int, str, str | None, str]]:
    """Return offending call sites in *py_file*.

    Returns a list of ``(lineno, canonical_func_name, enclosing_function, source_line)``
    tuples for every call to a watched function that passes a watched kwarg
    with an integer-literal value.

    The returned ``canonical_func_name`` is always the name from
    ``_RECALL_PROBE_FUNC_NAMES``, even when the call site uses an import alias
    (e.g. ``_probe_entries`` → ``"probe_entries"``).  This keeps error messages
    unambiguous.

    Only ``ast.Call`` nodes are inspected — function-signature defaults in
    ``ast.arguments`` are not considered call sites and are not flagged.

    Import aliases are resolved by ``_build_alias_map`` so that::

        from paramem.training.recall_eval import probe_entries as _probe_entries
        _probe_entries(..., batch_size=16)   # ← caught

    is correctly identified as a call to the watched ``probe_entries`` function.
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
    # Map local (possibly aliased) name → canonical watched name.
    alias_map = _build_alias_map(tree)
    out: list[tuple[int, str, str | None, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Determine the called local name (simple Name or attr access).
        func = node.func
        if isinstance(func, ast.Name):
            local_name = func.id
        elif isinstance(func, ast.Attribute):
            local_name = func.attr
        else:
            continue

        # Resolve to canonical name: direct match OR via import-alias map.
        if local_name in _RECALL_PROBE_FUNC_NAMES:
            canonical_name = local_name
        elif local_name in alias_map:
            canonical_name = alias_map[local_name]
        else:
            continue

        for kw in node.keywords:
            if kw.arg not in _WATCHED_KWARG_NAMES:
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                enclosing = _enclosing_function_name(node, tree)
                src_line = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
                out.append((node.lineno, canonical_name, enclosing, src_line))

    return out


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_recall_batch_size_comes_from_config_not_literals():
    """No production recall call site may pass batch_size / recall_probe_batch_size
    as an integer literal.

    If this test fails, a caller has hard-coded a batch size instead of
    threading ``config.consolidation.recall_probe_batch_size``.  Fix by
    passing the config value, or — only for genuinely N/A single-key probes
    — add an entry to ``_ALLOWLIST`` in this file with a justifying comment.

    Scan scope: files in ``_SCAN_FILES`` (production recall path).
    The guard does NOT scan tests/ — test helpers may use literal defaults
    on the low-level primitives without risk.
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str, str | None, str]] = []

    for rel in _SCAN_FILES:
        py_file = repo_root / rel
        if not py_file.exists():
            continue
        for lineno, func_name, enclosing, src in _find_literal_batch_size_calls(py_file):
            key = (rel, enclosing)
            if key in _ALLOWLIST:
                continue
            offenders.append((rel, lineno, func_name, enclosing, src))

    assert not offenders, (
        "Recall batch size must come from config.consolidation.recall_probe_batch_size, "
        "not an integer literal.  Hard-coded literals silently diverge from the config "
        "value and un-batch production recall when callers forget to thread config.\n"
        "Either pass the config value, or for a single-key liveness probe add an entry "
        "to _ALLOWLIST in tests/test_recall_batch_size_config_guard.py.\n"
        "Offending call sites:\n"
        + "\n".join(
            f"  {path}:{lineno} in {enclosing or '<module>'} — {func_name}(...) — {src}"
            for path, lineno, func_name, enclosing, src in offenders
        )
    )
