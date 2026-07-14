"""Packaging contract — what an operator must install to switch an option on.

A knob whose dependency is undeclared has environment-dependent behaviour:
the same code takes a different path on a dev box (spaCy happens to be
importable) than on a clean install (``except ImportError`` -> silent no-op).
Declaring the extra is what makes "flip the switch" a single, reproducible
command.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from paramem.utils.paths import find_project_root


def _project_table() -> dict:
    """Parse the ``[project]`` table out of the repo's ``pyproject.toml``."""
    root = find_project_root(Path(__file__))
    assert root is not None, "tests must run from inside the repo checkout"
    with open(root / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["project"]


def _optional_dependencies() -> dict[str, list[str]]:
    """Parse ``[project.optional-dependencies]`` out of the repo's pyproject."""
    return _project_table()["optional-dependencies"]


class TestNerExtra:
    """The EXPERIMENTAL spaCy PII cross-check ships as the ``ner`` extra."""

    def test_ner_extra_is_declared_in_pyproject(self):
        """``pip install paramem[ner]`` must pull BOTH the library and the
        model, so switching ``consolidation.extraction_ner_check`` on for a
        comparison run needs nothing else.

        Mutation: remove the ``ner`` extra (leave spaCy undeclared, as it was)
        -> the knob's behaviour becomes environment-dependent -> this test
        fails.
        """
        extras = _optional_dependencies()
        assert "ner" in extras, "the `ner` extra must be declared"
        spec = " ".join(extras["ner"])
        assert "spacy" in spec, "the `ner` extra must carry the spaCy library"
        assert "en_core_web_sm" in spec, "the `ner` extra must carry the spaCy model"

    def test_ner_is_not_a_default_dependency(self):
        """spaCy stays OUT of the base install: it is an experimental,
        off-by-default cross-check, not a shipped control.

        Mutation: move spaCy into ``[project.dependencies]`` -> every install
        pays for it and the estimator looks like a supported control -> this
        test fails.
        """
        deps = _project_table()["dependencies"]
        assert not any("spacy" in dep.lower() for dep in deps)
