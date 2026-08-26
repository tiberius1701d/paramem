"""Structural guard: regex is confined to declared-syntax modules.

Pattern matching over text decides nothing semantic in this project — see
``ARCHITECTURE.md``, "AD-22: Regex Confined to Declared Syntax". A regex is
admissible only where it recognizes syntax the project itself declares and
fully specifies (the schedule grammar, placeholder and speaker token shapes,
identity folding, turn-marker framing). This test pins the set of modules
under ``paramem/`` that may author or hold a compiled pattern by importing
``re`` at all, so a new call site elsewhere is caught before it ships.

The scan is static: it parses every tracked module with :mod:`ast` and
collects an ``Import``/``ImportFrom`` of ``re`` that is not nested inside an
``if TYPE_CHECKING:`` block (a type-only import that never runs authors no
pattern). It covers only import statements the interpreter actually
executes — a module that reaches ``re`` through :func:`importlib.import_module`
or an indirect alias is invisible to this guard, and is not a case the
project's modules use.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._guard_utils import tracked_python_files

# The declared-syntax modules admitted by AD-22. Each composes its patterns
# from a single fragment declaration and recognizes a shape the project
# itself mints — never text of unbounded origin.
_ALLOWED_RE_IMPORTERS = frozenset(
    {
        "paramem/cloud/placeholders.py",
        "paramem/server/schedule_grammar.py",
        "paramem/utils/identity.py",
        "paramem/utils/turn_markers.py",
    }
)


def _is_type_checking_test(test: ast.expr) -> bool:
    """Return True if *test* is the ``TYPE_CHECKING`` guard expression.

    Matches both ``if TYPE_CHECKING:`` and ``if typing.TYPE_CHECKING:``.
    """
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _imports_re(stmts: list[ast.stmt], in_type_checking: bool = False) -> bool:
    """Return True if any statement in *stmts* imports ``re`` outside a
    ``TYPE_CHECKING`` block.

    Recurses into every compound statement's sub-bodies (function/class
    bodies, if/else, try/except/finally, for/while/with) so an import
    nested anywhere in the module is found, not only at module level.
    """
    for stmt in stmts:
        if isinstance(stmt, ast.If) and _is_type_checking_test(stmt.test):
            if _imports_re(stmt.body, in_type_checking=True):
                return True
            if _imports_re(stmt.orelse, in_type_checking=in_type_checking):
                return True
            continue
        if isinstance(stmt, ast.Import) and not in_type_checking:
            if any(alias.name == "re" for alias in stmt.names):
                return True
        elif isinstance(stmt, ast.ImportFrom) and not in_type_checking:
            if stmt.module == "re":
                return True
        for field in ("body", "orelse", "finalbody"):
            sub = getattr(stmt, field, None)
            if sub and _imports_re(sub, in_type_checking=in_type_checking):
                return True
        for handler in getattr(stmt, "handlers", ()):
            if _imports_re(handler.body, in_type_checking=in_type_checking):
                return True
    return False


def test_regex_confined_to_declared_syntax_modules():
    """The set of ``paramem/`` modules importing ``re`` must equal the AD-22
    allowlist exactly.

    Adding ``import re`` to any other module under ``paramem/`` fails this
    test; removing one of the allowlisted modules' need for ``re`` should
    shrink the allowlist in the same change, not leave it stale.
    """
    repo_root = Path(__file__).resolve().parent.parent

    importers: set[str] = set()
    for py_file in tracked_python_files(repo_root):
        rel = py_file.relative_to(repo_root).as_posix()
        if not rel.startswith("paramem/"):
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue
        if _imports_re(tree.body):
            importers.add(rel)

    missing = _ALLOWED_RE_IMPORTERS - importers
    unexpected = importers - _ALLOWED_RE_IMPORTERS
    assert not missing and not unexpected, (
        "paramem/ modules importing `re` diverge from the AD-22 allowlist.\n"
        f"  missing from the scan (allowlisted but no longer importing re): {sorted(missing)}\n"
        f"  unexpected (importing re but not allowlisted): {sorted(unexpected)}\n"
        "A new regex call site outside the declared-syntax modules is a "
        "defect, not a style preference — see ARCHITECTURE.md, AD-22."
    )
