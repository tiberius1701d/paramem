"""Shared helpers for whole-repo and single-function structural guard tests.

``tracked_python_files`` — structural guards (``test_extraction_pipeline_guard.py``,
``test_simhash_unification_guard.py``) scan the codebase for forbidden call
sites or stale references. They must only ever look at files git considers
part of this repository — never a nested, gitignored checkout such as an
agent worktree under ``.claude/worktrees/``, a stray venv, or a build
directory. ``Path.rglob("*.py")`` walks the literal working tree and picks
up all of those, misattributing a nested checkout's own source as if it
were this repo's.

``git ls-files`` returns exactly the tracked set, so it is the single
source of truth for "in this repo" for a guard that walks the whole tree.

``find_function`` / ``enters_context_manager`` / ``call_inside_context_manager``
— AST helpers shared by every test that pins "function X must run inside
context manager Y" (the ``base_model_inference`` family of guards). One
implementation, so a fix to the walk logic (e.g. a new context-manager
call shape) lands once instead of drifting across per-test-file copies.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def tracked_python_files(repo_root: Path) -> list[Path]:
    """Return absolute paths of every git-tracked ``*.py`` file that exists on disk.

    Uses ``git ls-files -z`` (NUL-separated output) so paths survive intact
    regardless of content, and ``check=True`` so a git failure (e.g. running
    outside a repository) raises loudly instead of silently returning an
    empty — falsely passing — file list.

    Tracked-but-deleted files are filtered out.  ``git ls-files`` reports the
    index, so a file deleted in the working tree but not yet staged is still
    listed while having no content to scan; reading it raises
    ``FileNotFoundError`` and takes the guard down with an error that has
    nothing to do with the invariant it protects.  The scannable set is
    "tracked AND present", and a deleted file is genuinely empty of call
    sites.  Once the deletion is staged the path leaves ``ls-files`` too, so
    this filter changes nothing about which real source gets scanned.

    A non-empty result is asserted: a guard that scans zero files passes
    vacuously, which is worse than failing.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [repo_root / rel for rel in result.stdout.split("\0") if rel]
    present = [path for path in tracked if path.is_file()]
    if not present:
        raise AssertionError(
            f"tracked_python_files({repo_root}) resolved to zero readable files "
            f"({len(tracked)} tracked). Structural guards would pass vacuously."
        )
    return present


def find_function(tree: ast.AST, fn_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the first ``FunctionDef``/``AsyncFunctionDef`` named *fn_name*
    found by walking *tree* (module-level or nested), or ``None``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return node
    return None


def enters_context_manager(fn: ast.FunctionDef | ast.AsyncFunctionDef, manager_name: str) -> bool:
    """Return True if *fn* contains a ``with <manager_name>(...):`` anywhere
    in its body (a plain-``Name`` callee, not an attribute access).

    Only checks that the with-statement exists somewhere in the function —
    not that any particular call happens inside its body. Use
    :func:`call_inside_context_manager` when the guard must pin that a
    specific call happens INSIDE the context-manager's scope.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == manager_name
                ):
                    return True
    return False


def call_inside_context_manager(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, manager_name: str, called_name: str
) -> bool:
    """Return True if, within *fn*, a ``with <manager_name>(...):`` block's
    BODY contains a call to *called_name* (a plain-``Name`` callee).

    Stronger than :func:`enters_context_manager`: pins that the call
    actually happens INSIDE the with-block's scope, not merely that both
    the with-statement and the call exist somewhere in the function —
    moving the call outside the block fails this check even though
    :func:`enters_context_manager` would still pass.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.With):
            continue
        entered = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == manager_name
            for item in node.items
        )
        if not entered:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == called_name
                ):
                    return True
    return False
