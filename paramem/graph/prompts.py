"""Shared prompt-loading utilities for the graph package.

Dependency-light module: only imports from the standard library so that
``paramem.graph.merger`` can import the loader without pulling in the
heavyweight ``paramem.graph.extractor`` transitive dependency chain
(models.loader / vram_guard / evaluation.recall).
"""

from pathlib import Path

_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"


def _load_prompt(
    filename: str,
    default: str,
    prompts_dir: Path | None = None,
    model: str | None = None,
) -> str:
    """Load a prompt file, falling back to hardcoded default.

    Single chokepoint for ALL prompt loading in the extraction pipeline
    (extraction.txt, extraction_system.txt, extraction_procedural.txt,
    anonymization.txt, sota_enrichment.txt, sota_plausibility.txt, …).

    Resolution is per-file, per-model.  When *model* is provided, the
    search order is::

        [prompts_dir/model, prompts_dir, _DEFAULT_PROMPT_DIR]

    A model overrides only the files it provides — any file absent from
    the per-model sub-directory falls through to the shared directory.
    This means a model override is never all-or-nothing: adding a single
    file to ``configs/prompts/<model>/`` is sufficient to override just
    that file while all others inherit the shared default.

    The *model* parameter is intentionally only threaded into the
    local-model extraction prompts (``extraction.txt``,
    ``extraction_system.txt``, ``extraction_procedural.txt``).  SOTA
    cloud prompts (``sota_enrichment.txt``, ``sota_plausibility.txt``,
    ``sota_graph_enrichment.txt``) and ``anonymization.txt`` are
    model-independent by design and always call this function with
    ``model=None``.

    Before editing any file under ``configs/prompts/`` — or adding a new
    template slot here — note the empirical rules that govern these files:
    few-shot examples carry the schema; verbatim taxonomy slots like
    ``{entity_types}`` are anti-patterns; long prose rules dilute the
    example signal.  Edit the prompt files directly to tune; no code
    changes are needed.
    """
    search_dirs: list[Path] = []
    if prompts_dir:
        base = Path(prompts_dir)
        if model:
            search_dirs.append(base / model)  # per-model override (per file)
        search_dirs.append(base)
    search_dirs.append(_DEFAULT_PROMPT_DIR)

    for d in search_dirs:
        path = d / filename
        if path.exists():
            return path.read_text().strip()
    return default
