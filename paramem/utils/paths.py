"""Shared filesystem-path helpers."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path) -> Path | None:
    """Nearest ancestor of *start* containing ``pyproject.toml``.

    Walks upward from *start* (checking *start* itself when it is a directory)
    to the filesystem root. Returns ``None`` when no ``pyproject.toml`` anchor
    exists — repo checkouts always have one; an installed package under
    site-packages does not, so callers supply their own fixed-layout fallback.
    """
    p = start.resolve()
    for candidate in (p, *p.parents):
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return None
