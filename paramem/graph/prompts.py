"""Shared prompt-loading utilities for the graph package.

Dependency-light module: only imports from the standard library so that
``paramem.graph.merger`` can import the loader without pulling in the
heavyweight ``paramem.graph.extractor`` transitive dependency chain
(models.loader / vram_guard / evaluation.recall).
"""

from pathlib import Path

_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"

_SPEAKER_DIRECTIVE_FILE = "speaker_directive.txt"
_SPEAKER_DIRECTIVE_SENTINEL = "==="


def _load_speaker_directive_section(section_name: str) -> str:
    """Load a named section from ``configs/prompts/speaker_directive.txt``.

    The file contains two sentinel-delimited sections::

        === EXTRACTION-DIRECTIVE ===
        ...directive text...

        === INFERENCE-IDENTITY ===
        ...mapping text...

    Args:
        section_name: One of ``"EXTRACTION-DIRECTIVE"`` or
            ``"INFERENCE-IDENTITY"``.  The sentinel format is
            ``=== <NAME> ===`` (leading/trailing ``===`` with spaces).

    Returns:
        The section body as a stripped string.

    Raises:
        FileNotFoundError: When ``speaker_directive.txt`` is absent from
            ``configs/prompts/`` and no fallback is registered.
        KeyError: When the requested section is not found in the file.
    """
    path = _DEFAULT_PROMPT_DIR / _SPEAKER_DIRECTIVE_FILE
    raw = path.read_text()
    sections: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(_SPEAKER_DIRECTIVE_SENTINEL) and stripped.endswith(
            _SPEAKER_DIRECTIVE_SENTINEL
        ):
            # Sentinel line: flush previous section, start a new one.
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = (
                stripped[len(_SPEAKER_DIRECTIVE_SENTINEL) :]
                .rstrip(_SPEAKER_DIRECTIVE_SENTINEL)
                .strip()
            )
            current_lines = []
        else:
            if current_name is not None:
                current_lines.append(line)
    if current_name is not None:
        sections[current_name] = "\n".join(current_lines).strip()
    if section_name not in sections:
        raise KeyError(
            f"Section {section_name!r} not found in {_SPEAKER_DIRECTIVE_FILE}. "
            f"Available sections: {list(sections)}"
        )
    return sections[section_name]


def _load_prompt(
    filename: str,
    default: str = "",
    prompts_dir: Path | None = None,
    model: str | None = None,
    *,
    required: bool = False,
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

    When *required* is ``True`` and the file is not found in any search
    directory, raises :exc:`FileNotFoundError` with the searched paths
    listed in the message.  When *required* is ``False`` (default),
    returns *default* unchanged.  Use ``required=True`` for production
    enrollment paths where a missing prompt file must surface immediately
    rather than silently yielding an empty prompt.

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
    if required:
        searched = ", ".join(str(d / filename) for d in search_dirs)
        raise FileNotFoundError(
            f"Required prompt file {filename!r} not found. Searched: {searched}"
        )
    return default
