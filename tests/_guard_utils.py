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
