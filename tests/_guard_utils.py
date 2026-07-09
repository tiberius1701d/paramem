"""Shared helper for whole-repo structural guard tests.

Structural guards (``test_extraction_pipeline_guard.py``,
``test_simhash_unification_guard.py``) scan the codebase for forbidden call
sites or stale references. They must only ever look at files git considers
part of this repository — never a nested, gitignored checkout such as an
agent worktree under ``.claude/worktrees/``, a stray venv, or a build
directory. ``Path.rglob("*.py")`` walks the literal working tree and picks
up all of those, misattributing a nested checkout's own source as if it
were this repo's.

``git ls-files`` returns exactly the tracked set, so it is the single
source of truth for "in this repo" for a guard that walks the whole tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def tracked_python_files(repo_root: Path) -> list[Path]:
    """Return absolute paths of every git-tracked ``*.py`` file under ``repo_root``.

    Uses ``git ls-files -z`` (NUL-separated output) so paths survive intact
    regardless of content, and ``check=True`` so a git failure (e.g. running
    outside a repository) raises loudly instead of silently returning an
    empty — falsely passing — file list.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [repo_root / rel for rel in result.stdout.split("\0") if rel]
