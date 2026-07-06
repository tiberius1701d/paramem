"""Shared prompt-loading utilities for the graph package.

Dependency-light module: only imports from the standard library plus
``paramem.utils.paths`` (itself stdlib-only) so that ``paramem.graph.merger``
can import the loader without pulling in the heavyweight
``paramem.graph.extractor`` transitive dependency chain (models.loader /
vram_guard / evaluation.recall).
"""

from pathlib import Path

from paramem.utils.paths import find_project_root

_DEFAULT_PROMPT_DIR = (
    (find_project_root(Path(__file__)) or Path(__file__).resolve().parents[2])
    / "configs"
    / "prompts"
)

_SPEAKER_DIRECTIVE_FILE = "speaker_directive.txt"
_SPEAKER_DIRECTIVE_SENTINEL = "==="

# Load-bearing prompt files. Absent, ``_load_prompt`` silently returns the empty
# default (``required=False``), so the extraction pipeline degrades quietly
# rather than failing. ``ensure_prompt_assets`` validates them at startup.
_REQUIRED_PROMPT_FILES = (
    "extraction.txt",
    "extraction_system.txt",
    "extraction_procedural.txt",
)


def ensure_prompt_assets() -> None:
    """Fail loudly when the shared prompt assets are missing at startup.

    ParaMem deploys from a repo checkout (editable install under systemd), so
    ``configs/prompts/`` — the guaranteed final fallback in ``_load_prompt``'s
    search order — is always present in a correct deployment. A missing
    directory or file means a broken checkout or a non-editable ``pip install``
    (prompts are not shipped as package data). Surface that at boot instead of
    letting the extraction pipeline silently load empty prompts.

    Raises:
        RuntimeError: When ``_DEFAULT_PROMPT_DIR`` is not a directory, or a
            required prompt file is absent from it.
    """
    if not _DEFAULT_PROMPT_DIR.is_dir():
        raise RuntimeError(
            f"Prompt asset directory not found: {_DEFAULT_PROMPT_DIR}. ParaMem "
            "deploys from a repo checkout — run the server from the cloned "
            "repository (editable install). Prompts are not shipped as package data."
        )
    missing = [f for f in _REQUIRED_PROMPT_FILES if not (_DEFAULT_PROMPT_DIR / f).is_file()]
    if missing:
        raise RuntimeError(
            f"Required prompt file(s) missing from {_DEFAULT_PROMPT_DIR}: "
            f"{', '.join(missing)}. Restore configs/prompts/ from the repository."
        )


def _load_speaker_directive_section(section_name: str) -> str:
    """Load a named section from ``configs/prompts/speaker_directive.txt``.

    The file contains sentinel-delimited sections::

        === EXTRACTION-DIRECTIVE ===
        ...directive text...

        === THIRD-PARTY-DESCRIPTOR ===
        ...descriptor text...

    Sections are loaded individually by name.  Currently defined:

    * ``EXTRACTION-DIRECTIVE`` — loaded by ``build_speaker_context`` for
      the extraction user prompt.
    * ``THIRD-PARTY-DESCRIPTOR`` — loaded at module import by
      ``inference.py`` as the fallback label when a ``speaker{N}`` token
      has no display name (e.g. anonymous or unknown profile).

    Args:
        section_name: Name of the section to load (e.g.
            ``"EXTRACTION-DIRECTIVE"``).  The sentinel format is
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
