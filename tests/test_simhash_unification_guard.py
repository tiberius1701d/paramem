"""Structural guard: SimHash unification — no stale accessor or sidecar references.

After the SimHash unification refactor, fingerprints live in
``indexed_key_registry.json`` under the ``"simhash"`` key of each tier's
:class:`~paramem.training.key_registry.KeyRegistry`.  The legacy accessor
``simhashes_in_tier`` (the read path that returned the live dict directly,
enabling callers to mutate it) and the sidecar ``simhash_registry.json`` have
been deleted.  The private methods ``_active_simhashes`` / ``_known_simhashes``
on ``KeyRegistry`` are the only internal implementation, callable only from
``MemoryStore.tier_simhashes`` (the public facade) and ``integrity.py`` (the
cross-check tool).

This test scans the codebase and fails if:
1. ``simhashes_in_tier`` or ``active_simhashes_in_tier`` appear as call sites.
2. ``simhash_registry.json`` is referenced as a *write target or read path*
   (comments and docstrings explaining its elimination are allowed).
3. ``_active_simhashes`` or ``_known_simhashes`` are called directly outside
   the allowed-access files (``store.py``, ``key_registry.py``,
   ``integrity.py``).

Mirrors the structure of ``tests/test_extraction_pipeline_guard.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._guard_utils import tracked_python_files

# ---------------------------------------------------------------------------
# Files allowed to reference ``_known_simhashes`` / ``_active_simhashes``
# directly — these are the implementation sites and authorised callers.
# ---------------------------------------------------------------------------
_PRIVATE_ACCESSOR_ALLOWLIST = frozenset(
    {
        "paramem/memory/store.py",  # MemoryStore.tier_simhashes dispatcher
        "paramem/training/key_registry.py",  # method definitions
        "paramem/backup/integrity.py",  # cross-check tool (explicitly allowed)
        "tests/test_key_registry.py",  # unit tests for the methods themselves
        "tests/test_memory_store.py",  # unit tests for MemoryStore.tier_simhashes
        "tests/server/test_integrity_endpoint.py",  # integrity endpoint tests
        "tests/backup/test_integrity.py",  # backup integrity tests
        "tests/server/test_active_store_migration.py",  # migration tests
    }
)

# Regex patterns that identify CALL SITES (not definitions or docstrings).
# We look for method-call syntax ``something.simhashes_in_tier(`` rather than
# a bare name so that the old docstring in store.py (which still mentions the
# removed API in a comment) doesn't trigger a false positive.
_FORBIDDEN_CALL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "simhashes_in_tier call (read accessor deleted)",
        re.compile(r"\.\s*simhashes_in_tier\s*\("),
    ),
    (
        "active_simhashes_in_tier call (deleted)",
        re.compile(r"\.\s*active_simhashes_in_tier\s*\("),
    ),
]

# Files allowed to write / create ``simhash_registry.json`` — NONE.
# Any remaining write is a regression.
_SIMHASH_SIDECAR_WRITE_RE = re.compile(
    r"""(?x)
    (?:open|write_text|write_bytes|\.write\b|json\.dump\b)   # write verbs
    .*?                                                        # lazy gap
    simhash_registry\.json                                     # sidecar name
    |
    simhash_registry\.json                                     # name first …
    .*?
    (?:open|write_text|write_bytes|\.write\b|json\.dump\b)    # … then write
    """,
)

# Simpler heuristic: any live code line (not a comment, not a docstring) that
# contains ``simhash_registry.json`` as a string literal is suspicious.
# We use a line-level scan and exclude comment lines and known explanation sites.
_SIMHASH_SIDECAR_STRING_RE = re.compile(r'["\']simhash_registry\.json["\']')


def _is_comment_or_docstring_line(line: str) -> bool:
    """Return True if the line is a pure comment or clearly inside a docstring."""
    stripped = line.strip()
    return stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''")


def _find_forbidden_call_sites(
    py_file: Path,
) -> list[tuple[str, int, str]]:
    """Return ``(violation_label, lineno, source)`` tuples for forbidden calls.

    Only scans actual source lines; skips comment-only lines so that surviving
    docstring references (explaining the deletion) don't generate false alarms.
    """
    try:
        text = py_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    violations: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _is_comment_or_docstring_line(line):
            continue
        for label, pattern in _FORBIDDEN_CALL_PATTERNS:
            if pattern.search(line):
                violations.append((label, lineno, line.strip()))
    return violations


def _find_simhash_sidecar_string_sites(
    py_file: Path,
) -> list[tuple[str, int, str]]:
    """Return ``(label, lineno, source)`` for live string literals ``'simhash_registry.json'``.

    Skips comment-only lines because a number of docstrings legitimately
    explain that the sidecar has been removed.
    """
    try:
        text = py_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    violations: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _is_comment_or_docstring_line(line):
            continue
        if _SIMHASH_SIDECAR_STRING_RE.search(line):
            violations.append(("simhash_registry.json string in live code", lineno, line.strip()))
    return violations


def _find_private_accessor_calls_outside_allowlist(
    py_file: Path, rel: str
) -> list[tuple[str, int, str]]:
    """Return violations for direct ``._active_simhashes(`` / ``._known_simhashes(``
    calls in files that are not in the private-accessor allowlist.
    """
    if rel in _PRIVATE_ACCESSOR_ALLOWLIST:
        return []
    try:
        text = py_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    pattern = re.compile(r"\._(?:active|known)_simhashes\s*\(")
    violations: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _is_comment_or_docstring_line(line):
            continue
        if pattern.search(line):
            violations.append(
                (
                    "direct _active_simhashes/_known_simhashes call outside allowlist",
                    lineno,
                    line.strip(),
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_simhashes_in_tier_calls():
    """``simhashes_in_tier`` and ``active_simhashes_in_tier`` must not appear as call sites."""
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, str, int, str]] = []

    for py_file in sorted(tracked_python_files(repo_root)):
        rel = py_file.relative_to(repo_root).as_posix()
        # Allow archive/ — historical scripts that are not executed.
        # Allow the guard test itself — its error strings mention the old API names.
        if rel.startswith("archive/") or rel == "tests/test_simhash_unification_guard.py":
            continue
        for label, lineno, src in _find_forbidden_call_sites(py_file):
            offenders.append((rel, label, lineno, src))

    assert not offenders, (
        "Stale simhash accessor call sites found. The read accessor "
        "``simhashes_in_tier`` was deleted; route callers through "
        "``MemoryStore.tier_simhashes(tier, *, include_stale=bool)``:\n"
        + "\n".join(f"  {p}:{n} [{lbl}] — {s}" for p, lbl, n, s in offenders)
    )


def test_no_simhash_sidecar_string_in_live_code():
    """``'simhash_registry.json'`` must not appear as a string literal in live code.

    The sidecar file has been eliminated; any surviving string literal (outside
    a comment or docstring) is a regression — either a write path that was
    missed or a test that still expects the file to exist.
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str]] = []

    # Files with a legitimate surviving string reference (commenting out old
    # paths or guarding that the file does NOT exist).
    _SIDECAR_ALLOWLIST: frozenset[str] = frozenset(
        {
            # test_restore.py asserts the file does NOT exist; the string is
            # in an ``assert not (...).exists()`` expression.
            "tests/backup/test_restore.py",
            # test_bundle.py asserts the file is NOT in the bundle.
            "tests/backup/test_bundle.py",
            # test_integrity.py uses the string to assert the file is skipped.
            "tests/backup/test_integrity.py",
            # trial_inference_isolation.py writes a stale sidecar to assert
            # the reader does NOT pick it up.
            "tests/server/test_trial_inference_isolation.py",
        }
    )

    guard_self = "tests/test_simhash_unification_guard.py"
    for py_file in sorted(tracked_python_files(repo_root)):
        rel = py_file.relative_to(repo_root).as_posix()
        # Scope: production code and tests only. Experiments and scripts are
        # standalone historical artifacts that may reference the old sidecar name
        # without affecting production behaviour.
        if not (rel.startswith("paramem/") or rel.startswith("tests/")):
            continue
        if rel.startswith("archive/") or rel in _SIDECAR_ALLOWLIST or rel == guard_self:
            continue
        for label, lineno, src in _find_simhash_sidecar_string_sites(py_file):
            offenders.append((rel, lineno, src))

    assert not offenders, (
        "``simhash_registry.json`` string literal found in live code. "
        "The sidecar has been eliminated; simhashes now live in "
        "``indexed_key_registry.json``.  Remove the reference or add the "
        "file to the allowlist with a comment explaining why it legitimately "
        "references the old name:\n" + "\n".join(f"  {p}:{n} — {s}" for p, n, s in offenders)
    )


def test_no_private_simhash_accessor_calls_outside_allowlist():
    """``._active_simhashes(`` / ``._known_simhashes(`` must only be called from
    the designated implementation files.

    The two private accessors exist to support ``MemoryStore.tier_simhashes``
    (the mandatory-keyword public facade) and the integrity cross-check tool.
    Any other caller bypasses the ``include_stale`` guard and reopens the
    enumeration bug.
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str]] = []

    guard_self = "tests/test_simhash_unification_guard.py"
    for py_file in sorted(tracked_python_files(repo_root)):
        rel = py_file.relative_to(repo_root).as_posix()
        if rel.startswith("archive/") or rel == guard_self:
            continue
        for label, lineno, src in _find_private_accessor_calls_outside_allowlist(py_file, rel):
            offenders.append((rel, lineno, src))

    assert not offenders, (
        "Direct ``._active_simhashes()`` / ``._known_simhashes()`` calls found "
        "outside the allowlist.  Route callers through "
        "``MemoryStore.tier_simhashes(tier, *, include_stale=bool)``:\n"
        + "\n".join(f"  {p}:{n} — {s}" for p, n, s in offenders)
    )
