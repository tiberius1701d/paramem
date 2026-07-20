"""Shared prompt-loading utilities for the graph package.

Dependency-light module: only imports from the standard library plus
``paramem.utils.paths`` (itself stdlib-only) and
``paramem.graph.phase_trace`` (also dependency-light — stdlib-only at
runtime; its only non-stdlib reference, ``SessionGraph``, is
``TYPE_CHECKING``-guarded and never imported at runtime) so that
``paramem.graph.merger`` can import the loader without pulling in the
heavyweight ``paramem.graph.extractor`` transitive dependency chain
(models.loader / vram_guard / evaluation.recall).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from paramem.graph.phase_trace import record_prompt
from paramem.utils.paths import find_project_root

_DEFAULT_PROMPT_DIR = (
    (find_project_root(Path(__file__)) or Path(__file__).resolve().parents[2])
    / "configs"
    / "prompts"
)

_SPEAKER_DIRECTIVE_FILE = "speaker_directive.txt"
_SPEAKER_DIRECTIVE_SENTINEL = "==="

# ContextVar holding an active calibration prompt-override mapping
# (``{basename: content}``), consulted as the first act of ``_load_prompt``.
# ``None`` (the default) means no override is active — every caller's
# normal ``prompts_dir``/``model`` resolution is unaffected.  Scoped via
# :func:`prompt_overrides` so an override never outlives the ``with`` block
# that requested it, regardless of how deeply nested the prompt-loading
# call site is (no threading through every layer's parameter list).
_PROMPT_OVERRIDES: ContextVar[dict[str, str] | None] = ContextVar(
    "paramem_prompt_overrides", default=None
)


@contextmanager
def prompt_overrides(mapping: dict[str, str]) -> Iterator[None]:
    """Substitute prompt content for the duration of the block.

    Every :func:`_load_prompt` call made anywhere inside the ``with`` body
    — no matter how deeply nested the caller — checks ``mapping`` for its
    ``filename`` FIRST, before any ``prompts_dir``/``model`` search.  A
    match short-circuits straight to the override's content; no match falls
    through to the normal resolution unchanged.

    Args:
        mapping: ``{basename: content}`` — e.g.
            ``{"sota_enrichment_system.txt": "<calibration variant>"}``.
            ``content`` is used verbatim, exactly as if it had been read
            from a file (slots such as ``{transcript}`` remain literal).

    Nesting: an inner ``prompt_overrides`` call fully replaces the outer
    mapping for its duration (no merge) and the outer mapping is restored
    on exit, via the same token-based ``ContextVar.reset`` discipline
    :func:`~paramem.graph.phase_trace.phase_trace` uses for
    ``_ACTIVE_SCOPE``.
    """
    token = _PROMPT_OVERRIDES.set(mapping)
    try:
        yield
    finally:
        _PROMPT_OVERRIDES.reset(token)


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

    The file is read via :func:`_load_prompt` (``required=True``) rather
    than a bare ``Path.read_text()`` — this is the SAME chokepoint every
    other prompt in the extraction pipeline resolves through, so the file
    is overridable via :func:`prompt_overrides` (a calibration probe can
    substitute the whole file's content) and its resolution is recorded
    via :func:`~paramem.graph.phase_trace.record_prompt` like any other
    prompt load.  Section parsing below is unchanged — only the read.

    Args:
        section_name: Name of the section to load (e.g.
            ``"EXTRACTION-DIRECTIVE"``).  The sentinel format is
            ``=== <NAME> ===`` (leading/trailing ``===`` with spaces).

    Returns:
        The section body as a stripped string.

    Raises:
        FileNotFoundError: When ``speaker_directive.txt`` is absent from
            ``configs/prompts/`` and no override or fallback is registered.
        KeyError: When the requested section is not found in the file.
    """
    raw = _load_prompt(_SPEAKER_DIRECTIVE_FILE, required=True)
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

    Before any of the above, an active :func:`prompt_overrides` mapping is
    consulted for ``filename``.  A match short-circuits straight to the
    override's content — ``prompts_dir``/``model`` resolution never runs —
    and is reported via :func:`~paramem.graph.phase_trace.record_prompt`
    with a synthetic ``<override:{filename}>`` path so provenance clearly
    marks it as substituted rather than resolved from disk.  ``required``
    is satisfied by an override the same as by a found file — the
    "missing" case only fires when no override matched AND the normal
    search exhausted every directory.

    Every resolution — the override case, the found-file case, AND the
    not-found-anywhere fallback to *default* — is reported via
    :func:`paramem.graph.phase_trace.record_prompt`, so a
    :class:`~paramem.graph.phase_trace.PhaseRecord` for the calling phase
    always reflects the path/content this function actually returned,
    never a re-derivation of it.  ``record_prompt`` no-ops when no
    :func:`~paramem.graph.phase_trace.phase_trace` scope is active (e.g.
    a caller invoked at module import time, before any phase scope can
    exist), so this call is always safe.
    """
    override = _PROMPT_OVERRIDES.get()
    if override is not None and filename in override:
        content = override[filename]
        record_prompt(path=f"<override:{filename}>", content=content)
        return content

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
            content = path.read_text(encoding="utf-8").strip()
            record_prompt(path=str(path), content=content)
            return content
    if required:
        searched = ", ".join(str(d / filename) for d in search_dirs)
        raise FileNotFoundError(
            f"Required prompt file {filename!r} not found. Searched: {searched}"
        )
    record_prompt(path=None, content=default)
    return default
