"""Regression guard: speaker_id must be a required argument everywhere.

Phase 2a removed every ``speaker_id: str = ""`` default from:
  - ``paramem/graph/extractor.py``
  - ``paramem/training/consolidation.py``

This test scans both files and fails if any default is re-introduced, closing the
silent-omission hole that Phase 2a was designed to close.  Re-introduce a default
and the test fails immediately — callers that forget to supply a speaker_id get a
TypeError at import time rather than propagating ``""`` into the graph silently.

Verification protocol:
  To confirm the test actually catches a regression, temporarily add
  ``speaker_id: str = ""`` back to any function in the two files and run this
  test — it must fail.  Remove the default to restore green.
"""

from __future__ import annotations

from pathlib import Path

_TARGET_FILES = (
    "paramem/graph/extractor.py",
    "paramem/training/consolidation.py",
)

_FORBIDDEN_PATTERN = 'speaker_id: str = ""'


def test_no_speaker_id_empty_string_default_in_extractor_or_consolidation():
    """Every ``speaker_id`` parameter in the extraction and consolidation modules
    must be a required keyword-only argument (no ``= ""`` default).

    A caller that omits ``speaker_id`` must receive a ``TypeError`` at the call
    site — not silently propagate an empty string into the knowledge graph.
    """
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[tuple[str, int, str]] = []

    for rel in _TARGET_FILES:
        py_file = repo_root / rel
        assert py_file.exists(), f"Expected file not found: {rel}"
        for lineno, line in enumerate(py_file.read_text().splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if _FORBIDDEN_PATTERN in line:
                offenders.append((rel, lineno, line.strip()))

    assert not offenders, (
        'speaker_id defaults (= "") found — all callers must supply a real speaker ID:\n'
        + "\n".join(f"  {path}:{lineno} — {src}" for path, lineno, src in offenders)
    )
