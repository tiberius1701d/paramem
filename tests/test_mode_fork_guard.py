"""Structural guard: mode comparison literals must stay in the allowlist.

Scans ``paramem/`` for the three mode-comparison patterns:

* ``mode == "simulate"``
* ``mode == "train"``
* ``config.consolidation.mode ==``

Every match must resolve to a (file, function) pair that is in
``_ALLOWLIST``.  Any new site that does not appear there fails the test,
forcing an explicit review and an allowlist update rather than silently
multiplying mode forks.

The guard uses Python's AST to determine the enclosing function for each
matching line, which correctly skips matches inside docstrings and
multi-line string literals (those are not present in the AST as executable
nodes).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist: (relative_path, function_name)
# ---------------------------------------------------------------------------
# ``function_name`` is the innermost ``def`` / ``async def`` that contains
# the matching line, as returned by ast.FunctionDef.name.  Use ``None`` to
# allow all functions in a file.
#
# Update procedure: if a new mode fork is intentionally introduced, add its
# (file, function) pair here and update the module docstring.

_ALLOWLIST: frozenset[tuple[str, str | None]] = frozenset(
    [
        # persistence layer — the one sanctioned fork point for venue dispatch
        ("paramem/memory/persistence.py", "commit_tier_slot"),
        # consolidation — unified entry and its helpers
        ("paramem/training/consolidation.py", "run_consolidation_cycle"),
        ("paramem/training/consolidation.py", "_resolve_target_slot"),
        ("paramem/training/consolidation.py", "_prepare_episodic_keys_for_tier"),
        ("paramem/training/consolidation.py", "_prepare_procedural_keys_for_tier"),
        ("paramem/training/consolidation.py", "_run_indexed_key_procedural"),
        # server — hydration, extraction dispatch, training dispatch
        ("paramem/server/app.py", "lifespan"),
        ("paramem/server/app.py", "_run_extraction_phase"),
        ("paramem/server/app.py", "_extract_and_start_training"),
        ("paramem/server/app.py", "_run_full_cycle"),
        # migration tooling — mode-switch detection and graph migration
        ("paramem/server/active_store_migration.py", "detect_mode_switch"),
        ("paramem/server/active_store_migration.py", "migrate"),
        # inference — simulate-mode probe path
        ("paramem/server/inference.py", "_probe_and_reason"),
        # migration tooling — detect_simulate_mode reads candidate YAML mode key
        ("paramem/server/migration.py", "detect_simulate_mode"),
    ]
)

# ---------------------------------------------------------------------------
# Patterns to search for
# ---------------------------------------------------------------------------
# Match any variable whose name ends in ``mode`` (bare ``mode``,
# ``target_mode``, ``_mode``, ``_active_mode``, etc.) compared to the
# literal strings ``"simulate"`` or ``"train"``, plus the common
# attribute-access forms (``config.consolidation.mode ==`` and
# ``.get("mode") ==``).

_PATTERNS = [
    # bare and prefixed mode variables: mode, _mode, target_mode, _active_mode, …
    re.compile(r'\b\w*mode\s*==\s*"(?:simulate|train)"'),
    # attribute access: config.consolidation.mode ==
    re.compile(r'\.mode\s*==\s*"(?:simulate|train)"'),
    # dict .get("mode") == "simulate"|"train"
    re.compile(r'\.get\("mode"\)\s*==\s*"(?:simulate|train)"'),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent


def _build_func_map(tree: ast.Module) -> list[tuple[int, int, str]]:
    """Return sorted list of (start_line, end_line, func_name) for all defs."""
    func_ranges: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_ranges.append((node.lineno, node.end_lineno, node.name))
    func_ranges.sort()
    return func_ranges


def _enclosing_function(line: int, func_ranges: list[tuple[int, int, str]]) -> str:
    """Return the name of the innermost function that contains *line*.

    Returns ``"<module>"`` when *line* is at module scope (no function
    contains it).
    """
    best: tuple[int, int, str] | None = None
    for start, end, name in func_ranges:
        if start <= line <= end:
            if best is None or start > best[0]:
                best = (start, end, name)
    return best[2] if best is not None else "<module>"


def _collect_executable_lines(src: str) -> frozenset[int]:
    """Return line numbers that contain executable AST nodes.

    This is used to skip matches that appear only inside string literals
    (docstrings) or comments — those are not executable statements.
    """
    tree = ast.parse(src)
    executable_lines: set[int] = set()
    for node in ast.walk(tree):
        if hasattr(node, "lineno"):
            # Include only nodes that represent real statements or expressions,
            # not constants (which includes docstrings when they are the sole
            # child of a function body).
            if not isinstance(node, ast.Constant):
                executable_lines.add(node.lineno)
            # Also accept Constant nodes that are part of comparisons, calls,
            # etc. — we cannot distinguish those here, so we rely on the caller
            # to check the raw line text against our patterns instead.
    return frozenset(executable_lines)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, enclosing_function) for every pattern match.

    Skips lines that are pure docstring/comment text (i.e. the line contains
    the pattern but the pattern appears only inside a string literal with no
    surrounding comparison operator context at the AST level).
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    func_ranges = _build_func_map(tree)

    # Collect all lines with string-only nodes (module docstrings, class
    # docstrings, function docstrings).  An Expr node whose value is a
    # Constant string is a docstring.
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                # Mark the entire span of the string literal as docstring.
                for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    docstring_lines.add(ln)

    hits: list[tuple[int, str]] = []
    for lineno, raw_line in enumerate(lines, start=1):
        if lineno in docstring_lines:
            continue
        # Skip pure comments
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            continue
        for pat in _PATTERNS:
            if pat.search(raw_line):
                func_name = _enclosing_function(lineno, func_ranges)
                hits.append((lineno, func_name))
                break  # one hit per line is enough

    return hits


# ---------------------------------------------------------------------------
# The guard test
# ---------------------------------------------------------------------------


def test_no_unallowlisted_mode_forks() -> None:
    """Every mode comparison in paramem/ must appear in _ALLOWLIST.

    Failure message names the file, line, and enclosing function so the
    developer knows exactly where to look.
    """
    paramem_root = _REPO_ROOT / "paramem"
    violations: list[str] = []

    for py_file in sorted(paramem_root.rglob("*.py")):
        rel = py_file.relative_to(_REPO_ROOT).as_posix()
        hits = _scan_file(py_file)
        for lineno, func_name in hits:
            key = (rel, func_name)
            # Check exact match first, then wildcard (None means "all functions")
            if key not in _ALLOWLIST and (rel, None) not in _ALLOWLIST:
                violations.append(
                    f"  {rel}:{lineno}  (function: {func_name!r})  — not in _ALLOWLIST"
                )

    assert not violations, (
        "Unallowlisted mode fork(s) detected.\n"
        "Add the entry to _ALLOWLIST in tests/test_mode_fork_guard.py "
        "after reviewing that the fork is intentional:\n\n" + "\n".join(violations)
    )


def test_allowlist_has_no_dead_entries() -> None:
    """Every entry in _ALLOWLIST must match at least one scanned line.

    This prevents the allowlist from accumulating stale entries after
    refactors remove a fork site.

    Entries with ``function_name=None`` (whole-file allowlist) are checked
    for the presence of any match anywhere in the file; the test fails only
    when the file itself no longer exists or contains no matches at all.
    """
    paramem_root = _REPO_ROOT / "paramem"

    # Build a map: rel_path -> list of (lineno, func_name) hits
    hits_by_file: dict[str, list[tuple[int, str]]] = {}
    for py_file in sorted(paramem_root.rglob("*.py")):
        rel = py_file.relative_to(_REPO_ROOT).as_posix()
        hits = _scan_file(py_file)
        if hits:
            hits_by_file[rel] = hits

    dead: list[str] = []
    for rel, func_name in sorted(_ALLOWLIST):
        file_hits = hits_by_file.get(rel, [])
        if func_name is None:
            # Whole-file entry: the file must have at least one hit
            if not file_hits:
                dead.append(f"  {rel}  (any function)  — file has no matches")
        else:
            matched = any(fn == func_name for _ln, fn in file_hits)
            if not matched:
                dead.append(f"  {rel}  (function: {func_name!r})  — no match found")

    assert not dead, (
        "Dead allowlist entry(ies) detected — remove them from _ALLOWLIST "
        "in tests/test_mode_fork_guard.py:\n\n" + "\n".join(dead)
    )
