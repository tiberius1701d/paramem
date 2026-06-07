"""Shared prompt-loading utilities for the graph package.

Dependency-light module: only imports from the standard library so that
``paramem.graph.merger`` can import the loader without pulling in the
heavyweight ``paramem.graph.extractor`` transitive dependency chain
(models.loader / vram_guard / evaluation.recall).
"""

from pathlib import Path

_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"


def _load_prompt(filename: str, default: str, prompts_dir: Path | None = None) -> str:
    """Load a prompt file, falling back to hardcoded default.

    Single chokepoint for ALL prompt loading in the extraction pipeline
    (extraction.txt, extraction_system.txt, extraction_procedural.txt,
    anonymization.txt, sota_enrichment.txt, sota_plausibility.txt, …).

    Before editing any file under ``configs/prompts/`` — or adding a new
    template slot here — note the empirical rules that govern these files:
    few-shot examples carry the schema; verbatim taxonomy slots like
    ``{entity_types}`` are anti-patterns; long prose rules dilute the
    example signal.  Edit the prompt files directly to tune; no code
    changes are needed.
    """
    search_dirs = []
    if prompts_dir:
        search_dirs.append(Path(prompts_dir))
    search_dirs.append(_DEFAULT_PROMPT_DIR)

    for d in search_dirs:
        path = d / filename
        if path.exists():
            return path.read_text().strip()
    return default
