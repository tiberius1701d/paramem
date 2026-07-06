"""Tests for paramem.utils.paths.find_project_root.

Covers:
- Found from a nested subdirectory below the anchor.
- Found when *start* is a file path inside the tree.
- None when no ``pyproject.toml`` exists above *start*.
"""

from __future__ import annotations

from paramem.utils.paths import find_project_root


class TestFindProjectRoot:
    def test_found_from_nested_subdir(self, tmp_path):
        """A nested subdirectory below the pyproject.toml anchor resolves to the anchor."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        nested = tmp_path / "paramem" / "cli"
        nested.mkdir(parents=True)

        assert find_project_root(nested) == tmp_path

    def test_found_from_file_path(self, tmp_path):
        """A file path (not just a directory) below the anchor resolves to the anchor."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
        nested = tmp_path / "paramem" / "cli"
        nested.mkdir(parents=True)
        start = nested / "http_client.py"

        assert find_project_root(start) == tmp_path

    def test_none_without_pyproject(self, tmp_path):
        """No pyproject.toml anywhere above *start* -> None."""
        nested = tmp_path / "some" / "deep" / "path"
        nested.mkdir(parents=True)

        assert find_project_root(nested) is None
