"""Structural guard: SimHash unification — one fingerprint map, no sidecar.

Fingerprints live in ``indexed_key_registry.json`` under the ``"simhash"``
key of each tier's :class:`~paramem.training.key_registry.KeyRegistry`. The
tier's ONE fingerprint map is :attr:`KeyRegistry._simhash`, private, with a
single private accessor :meth:`KeyRegistry._simhashes` — the only public path
to a fingerprint set is :meth:`~paramem.memory.store.MemoryStore.tier_simhashes`
(no ``include_stale`` keyword — that distinction was deleted along with the
stale-record fingerprint). The separate ``simhash_registry.json`` sidecar
file has been eliminated.

This test scans the codebase and fails if:
1. ``KeyRegistry._simhash`` / ``KeyRegistry._simhashes`` are named (as an
   attribute access) anywhere outside ``paramem/training/key_registry.py``
   except inside ``MemoryStore.tier_simhashes`` (``paramem/memory/store.py``)
   — the one accessor outside ``key_registry.py`` that still reads the
   private map directly.  ``MemoryStore.replace_simhashes_in_tier`` writes
   through the public :meth:`KeyRegistry.replace_simhashes` primitive
   instead, so it is no longer on this allow-list — a regression back to a
   direct ``reg._simhash`` write there is exactly what this guard now
   catches. ``paramem/backup/integrity.py`` reads fingerprints through the
   public ``KeyRegistry.load_simhashes`` leaf, not the private map, and is
   therefore NOT on this allow-list.
2. ``simhash_registry.json`` is referenced as a *write target or read path*
   (comments and docstrings explaining its elimination are allowed).

Scans via ``ast`` (real ``Attribute``/string-literal nodes), not text
matching, so a docstring or comment mentioning ``_simhash``/``_simhashes`` in
prose never produces a false positive — the old text-scan guard this file
replaces missed exactly that case (``paramem/memory/store.py``'s own module
docstring names ``registry._simhash`` in prose).

Mirrors the structure of ``tests/test_extraction_pipeline_guard.py``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests._guard_utils import find_function, tracked_python_files

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KEY_REGISTRY_PATH = _REPO_ROOT / "paramem" / "training" / "key_registry.py"
_STORE_PATH = _REPO_ROOT / "paramem" / "memory" / "store.py"

_PRIVATE_FINGERPRINT_ATTRS = frozenset({"_simhash", "_simhashes"})


def _attribute_accesses(py_file: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, attr)`` for every real ``ast.Attribute`` node in
    *py_file* whose attribute name is one of :data:`_PRIVATE_FINGERPRINT_ATTRS`.

    AST-based, not text matching: a docstring or comment merely mentioning
    ``_simhash``/``_simhashes`` in prose produces no ``ast.Attribute`` node
    and is never flagged.
    """
    try:
        text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_file))
    except (UnicodeDecodeError, SyntaxError):
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _PRIVATE_FINGERPRINT_ATTRS:
            hits.append((node.lineno, node.attr))
    return hits


def test_the_fingerprint_map_is_named_only_by_its_one_store_accessor():
    """``KeyRegistry._simhash`` / ``._simhashes`` are named outside
    ``key_registry.py`` only inside ``MemoryStore.tier_simhashes``.

    ``MemoryStore.replace_simhashes_in_tier`` delegates to the public
    :meth:`KeyRegistry.replace_simhashes` and no longer names either private
    attribute directly, so it carries no allowance here.  Any other module
    naming either — including a stray direct access from a test, a
    regression in ``replace_simhashes_in_tier``, or a resurrected read site
    in ``integrity.py`` bypassing the public ``load_simhashes`` leaf — is a
    guard failure.
    """
    store_tree = ast.parse(_STORE_PATH.read_text(encoding="utf-8"), filename=str(_STORE_PATH))
    tier_simhashes_fn = find_function(store_tree, "tier_simhashes")
    assert tier_simhashes_fn is not None, "MemoryStore.tier_simhashes not found — guard is stale"
    allowed_ranges = [
        (tier_simhashes_fn.lineno, tier_simhashes_fn.end_lineno),
    ]

    def _within_allowed_range(lineno: int) -> bool:
        return any(start <= lineno <= end for start, end in allowed_ranges)

    offenders: list[tuple[str, int, str]] = []
    for py_file in sorted(tracked_python_files(_REPO_ROOT)):
        rel = py_file.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith("archive/"):
            continue
        if py_file == _KEY_REGISTRY_PATH:
            continue
        for lineno, attr in _attribute_accesses(py_file):
            if py_file == _STORE_PATH and _within_allowed_range(lineno):
                continue
            offenders.append((rel, lineno, attr))

    assert not offenders, (
        "KeyRegistry._simhash / ._simhashes named outside key_registry.py and "
        "outside MemoryStore.tier_simhashes / replace_simhashes_in_tier:\n"
        + "\n".join(f"  {p}:{n} .{a}" for p, n, a in offenders)
    )


# ---------------------------------------------------------------------------
# The eliminated sidecar file
# ---------------------------------------------------------------------------

_SIMHASH_SIDECAR_STRING_RE = re.compile(r'["\']simhash_registry\.json["\']')


def _simhash_sidecar_string_sites(py_file: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, source)`` for real string-literal AST nodes whose
    value is ``simhash_registry.json`` — a docstring/comment mention is not
    an ``ast.Constant`` string-literal node in live code and is never
    flagged."""
    try:
        text = py_file.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_file))
    except (UnicodeDecodeError, SyntaxError):
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "simhash_registry.json" in node.value
        ):
            hits.append((node.lineno, node.value))
    return hits


def test_no_simhash_sidecar_string_in_live_code():
    """``'simhash_registry.json'`` must not appear as a string literal in live code.

    The sidecar file has been eliminated; any surviving string-literal
    reference is either a missed write path or a test that still expects the
    file to exist.
    """
    offenders: list[tuple[str, int, str]] = []

    # Files with a legitimate surviving string reference (asserting the file
    # does NOT exist / is NOT in a bundle / is skipped / is not picked up).
    _SIDECAR_ALLOWLIST: frozenset[str] = frozenset(
        {
            "tests/backup/test_restore.py",
            "tests/backup/test_bundle.py",
            "tests/backup/test_integrity.py",
            "tests/server/test_trial_inference_isolation.py",
        }
    )

    guard_self = "tests/test_simhash_unification_guard.py"
    for py_file in sorted(tracked_python_files(_REPO_ROOT)):
        rel = py_file.relative_to(_REPO_ROOT).as_posix()
        if not (rel.startswith("paramem/") or rel.startswith("tests/")):
            continue
        if rel.startswith("archive/") or rel in _SIDECAR_ALLOWLIST or rel == guard_self:
            continue
        for lineno, value in _simhash_sidecar_string_sites(py_file):
            if _SIMHASH_SIDECAR_STRING_RE.search(f'"{value}"'):
                offenders.append((rel, lineno, value))

    assert not offenders, (
        "'simhash_registry.json' string literal found in live code. The "
        "sidecar has been eliminated; simhashes now live in "
        "indexed_key_registry.json. Remove the reference or add the file to "
        "the allowlist with a comment explaining why it legitimately "
        "references the old name:\n" + "\n".join(f"  {p}:{n} — {s!r}" for p, n, s in offenders)
    )
