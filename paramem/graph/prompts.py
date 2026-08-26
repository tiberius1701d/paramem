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

_SECTION_SENTINEL = "==="

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
            ``{"cloud_enrichment_system.txt": "<calibration variant>"}``.
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


# Load-bearing prompt files, checked eagerly at startup by
# ``ensure_prompt_assets`` so a missing file surfaces at boot rather than at
# the first request that needs it — every one of these is loaded with no
# fallback: absence raises ``FileNotFoundError`` from ``_load_prompt``.
_REQUIRED_PROMPT_FILES = (
    "extraction.txt",
    "extraction_system.txt",
    "extraction_procedural.txt",
    "document_directive.txt",
    "speaker_directive.txt",
    "trained_recall.txt",
    "serving_system.txt",
    "serving_directives.txt",
    "cloud_serving_system.txt",
    "intent_classifier.txt",
    "recall_selection.txt",
    "anonymization.txt",
)

# The sectioned local-anonymization prompt home
# (`paramem.graph.anonymizer_prompts.load_anonymizer_prompts` is the one
# composer that reads it). Every section here is required — an anchor call
# with a missing section would otherwise fail at the first request that
# needs it rather than at boot. The home is anchor-only and
# category-independent: every render is the same two sections regardless of
# the configured scrub categories.
_ANONYMIZATION_PROMPT_FILE = "anonymization.txt"
_ANONYMIZATION_REQUIRED_SECTIONS = (
    "ANCHOR-SYSTEM",
    "ANCHOR",
)
# Per-section required slots — mirrors _ALWAYS_SUPPLIED_EXTRACTION_SLOTS'
# role for the extraction templates: a section present but missing one of
# its own slots renders a payload-less call that silently under-scrubs
# rather than raising, because `str.format` ignores a slot the section text
# never references.
_ANONYMIZATION_SECTION_SLOTS: dict[str, tuple[str, ...]] = {
    "ANCHOR": ("{speaker_id}", "{values}", "{text}"),
}

# The extraction user templates — every one of these local-extraction call
# sites always supplies every slot in _ALWAYS_SUPPLIED_EXTRACTION_SLOTS (see
# paramem.graph.extractor._generate_extraction).  A per-model or operator
# copy that drops one is a config-load failure (str.format ignores surplus
# kwargs, so the drop would silently revert that copy to cue-less/context-
# less extraction), not a runtime surprise — checked by ensure_prompt_assets.
_EXTRACTION_USER_TEMPLATES = (
    "extraction.txt",
    "extraction_second_order.txt",
    "extraction_procedural.txt",
)
# {named_people} is deliberately NOT here: it is an extra_slots member
# supplied by one caller (second_order_extract) and has its own existing
# guard (TestSecondOrderExtractionPromptRenderAllVariants).
_ALWAYS_SUPPLIED_EXTRACTION_SLOTS = ("{transcript}", "{speaker_context}", "{document_context}")


def ensure_prompt_assets(*, prompts_dir: Path | None = None) -> None:
    """Fail loudly when the shared prompt assets are missing at startup, or
    when a resolvable extraction user template drops a slot every
    local-extraction call always supplies.

    ParaMem deploys from a repo checkout (editable install under systemd), so
    ``configs/prompts/`` — the guaranteed final fallback in ``_load_prompt``'s
    search order — is always present in a correct deployment. A missing
    directory or file means a broken checkout or a non-editable ``pip install``
    (prompts are not shipped as package data). Surface that at boot instead of
    letting the extraction pipeline silently load empty prompts.

    The slot gate walks ``([prompts_dir] if prompts_dir is not None else [])
    + [_DEFAULT_PROMPT_DIR]`` and each walked directory's immediate
    subdirectories (per-model override dirs). Roots are deduped by
    ``resolve()`` before walking — the default deployment has
    ``paths.prompts`` equal to the shipped dir, so without the dedupe the
    identical tree would be walked twice and every missing-slot report
    would appear duplicated. For every :data:`_EXTRACTION_USER_TEMPLATES`
    file that exists in a walked directory, every
    :data:`_ALWAYS_SUPPLIED_EXTRACTION_SLOTS` literal must appear in its
    raw text — a template silently missing the cue slot is a config-load
    failure, not a runtime surprise (``str.format`` ignores surplus
    kwargs, so a per-model copy without the slot would silently revert
    that model to cue-less document extraction).

    Args:
        prompts_dir: Operator-configured ``paths.prompts`` override
            (``ServerConfig.prompts_dir``), passed by the server's lifespan
            startup at its one call site. ``None`` (default — tests, any
            caller with no config) checks only the shipped
            ``configs/prompts/`` tree; the walk never constructs ``Path(None)``.

    Raises:
        RuntimeError: When ``_DEFAULT_PROMPT_DIR`` is not a directory, a
            required prompt file is absent from it, the sectioned
            anonymization home is missing one of its required sections, or
            an extraction user template resolvable under the walked
            directories is missing one of the slots every local-extraction
            call always supplies.
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

    # `prompts_dir` threaded through — an operator override tree can
    # replace the anonymization home, so the boot check must validate the
    # copy that will actually be loaded at runtime, not only the shipped
    # default.
    anon_sections = _load_prompt_sections(_ANONYMIZATION_PROMPT_FILE, prompts_dir=prompts_dir)
    missing_sections = [s for s in _ANONYMIZATION_REQUIRED_SECTIONS if s not in anon_sections]
    if missing_sections:
        raise RuntimeError(
            f"Required section(s) missing from "
            f"{_DEFAULT_PROMPT_DIR / _ANONYMIZATION_PROMPT_FILE}: "
            f"{', '.join(missing_sections)}. Restore configs/prompts/anonymization.txt "
            "from the repository."
        )

    # Per-section slot gate: a present section missing one of its own
    # required slots (see `_ANONYMIZATION_SECTION_SLOTS`) renders a
    # payload-less call that `str.format` accepts silently — no `KeyError`,
    # no error of any kind, just a call the model answers having never been
    # shown what it was supposed to decide.
    slot_failures: list[str] = []
    for section_name, required_slots in _ANONYMIZATION_SECTION_SLOTS.items():
        body = anon_sections.get(section_name, "")
        missing_slots = [slot for slot in required_slots if slot not in body]
        if missing_slots:
            slot_failures.append(f"{section_name}: missing {', '.join(missing_slots)}")
    if slot_failures:
        raise RuntimeError(
            f"Section(s) missing a required slot in "
            f"{_DEFAULT_PROMPT_DIR / _ANONYMIZATION_PROMPT_FILE}: " + "; ".join(slot_failures)
        )

    search_roots = ([prompts_dir] if prompts_dir is not None else []) + [_DEFAULT_PROMPT_DIR]
    walked: list[Path] = []
    seen_resolved: set[Path] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        resolved = root.resolve()
        if resolved in seen_resolved:
            # The default deployment has paths.prompts == the shipped dir
            # (both resolve to the same tree) — walking it twice would
            # duplicate every missing-slot report below.
            continue
        seen_resolved.add(resolved)
        walked.append(root)
        walked.extend(d for d in root.iterdir() if d.is_dir())

    slot_failures: list[str] = []
    for directory in walked:
        for filename in _EXTRACTION_USER_TEMPLATES:
            path = directory / filename
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            missing_slots = [s for s in _ALWAYS_SUPPLIED_EXTRACTION_SLOTS if s not in content]
            if missing_slots:
                slot_failures.append(f"{path}: missing {', '.join(missing_slots)}")
    if slot_failures:
        raise RuntimeError(
            "Extraction user template(s) missing a required slot: " + "; ".join(slot_failures)
        )


def _load_prompt_sections(filename: str, *, prompts_dir: Path | None = None) -> dict[str, str]:
    """Parse every sentinel-delimited section out of a prompt file.

    The file contains sentinel-delimited sections::

        === SECTION-ONE ===
        ...section one text...

        === SECTION-TWO ===
        ...section two text...

    Any text above the first sentinel (e.g. a behaviour-level header
    comment) is never picked up by any section — it is discarded before
    the first sentinel line is seen.

    Sectioned files in this codebase:

    * ``speaker_directive.txt`` — ``EXTRACTION-DIRECTIVE`` (loaded by
      ``build_speaker_context`` for the extraction user prompt) and
      ``THIRD-PARTY-DESCRIPTOR`` (loaded at module import by
      ``paramem.server.speaker`` as the fallback label when a
      ``speaker{N}`` token has no display name).
    * ``trained_recall.txt`` — ``SYSTEM`` and ``RECALL``, the weight-coupled
      training/probe interface (:mod:`paramem.training.dataset`).
    * ``serving_directives.txt`` — ``IDENTITY-LINE``, ``LANGUAGE-LINE``,
      ``REASONING-TURN``, ``RECORDED-DATES-SUFFIX``, ``EMPTY-PERIOD-NOTE``,
      the serving turn's slot-bearing fragments (:mod:`paramem.server.prompts`).
    * ``anonymization.txt`` — the local-anonymizer home: ``ANCHOR-SYSTEM``
      and ``ANCHOR``, the two sections every render composes, regardless of
      the configured scrub categories — composed by
      :func:`paramem.graph.anonymizer_prompts.load_anonymizer_prompts`.

    The file is read via :func:`_load_prompt` rather than a bare
    ``Path.read_text()`` — this is the SAME chokepoint every other prompt
    in the codebase resolves through, so the file is overridable via
    :func:`prompt_overrides` (a calibration probe can substitute the whole
    file's content) and its resolution is recorded via
    :func:`~paramem.graph.phase_trace.record_prompt` like any other prompt
    load.

    Args:
        filename: Basename of the sectioned prompt file (e.g.
            ``"speaker_directive.txt"``).
        prompts_dir: Forwarded to :func:`_load_prompt` unchanged — the
            operator ``paths.prompts`` override. ``None`` (default)
            resolves only the shipped ``configs/prompts/`` copy.

    Returns:
        ``{section_name: stripped_body}`` for every sentinel-delimited
        section found. A file with no sentinel line at all returns an
        empty dict.

    Raises:
        FileNotFoundError: When *filename* is absent from
            ``configs/prompts/`` and no override is registered.
    """
    raw = _load_prompt(filename, prompts_dir=prompts_dir)
    sections: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(_SECTION_SENTINEL) and stripped.endswith(_SECTION_SENTINEL):
            # Sentinel line: flush previous section, start a new one.
            if current_name is not None:
                sections[current_name] = "\n".join(current_lines).strip()
            current_name = stripped[len(_SECTION_SENTINEL) :].rstrip(_SECTION_SENTINEL).strip()
            current_lines = []
        else:
            if current_name is not None:
                current_lines.append(line)
    if current_name is not None:
        sections[current_name] = "\n".join(current_lines).strip()
    return sections


def _load_prompt_section(filename: str, section: str) -> str:
    """Load one named section from a sentinel-delimited prompt file.

    Delegates to :func:`_load_prompt_sections` for the parse (one
    implementation of the sentinel format) and raises ``KeyError`` when
    *section* is absent.

    Args:
        filename: Basename of the sectioned prompt file (e.g.
            ``"speaker_directive.txt"``).
        section: Name of the section to load (e.g.
            ``"EXTRACTION-DIRECTIVE"``).  The sentinel format is
            ``=== <NAME> ===`` (leading/trailing ``===`` with spaces).

    Returns:
        The section body as a stripped string.

    Raises:
        FileNotFoundError: When *filename* is absent from
            ``configs/prompts/`` and no override is registered.
        KeyError: When *section* is not found in the file.
    """
    sections = _load_prompt_sections(filename)
    if section not in sections:
        raise KeyError(
            f"Section {section!r} not found in {filename}. Available sections: {list(sections)}"
        )
    return sections[section]


def _load_prompt(
    filename: str,
    *,
    prompts_dir: Path | None = None,
    model: str | None = None,
) -> str:
    """Load a prompt file. Every parameter after *filename* is keyword-only.

    Single chokepoint for ALL model-facing prompt text in the codebase
    (extraction.txt, extraction_system.txt, extraction_procedural.txt,
    document_directive.txt, anonymization.txt, cloud_enrichment.txt,
    cloud_plausibility.txt, trained_recall.txt, serving_system.txt, …). No
    inline prompt literals live in Python; every string a model sees
    resolves through here.

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
    ``extraction_system.txt``, ``extraction_procedural.txt``).  Cloud
    prompts (``cloud_enrichment.txt``, ``cloud_plausibility.txt``,
    ``cloud_graph_enrichment.txt``) and the sectioned local-anonymizer
    home (``anonymization.txt`` — every scan/anchor/apply call, across
    every tier, selects one section from this one file via
    ``paramem.graph.anonymizer_prompts.load_anonymizer_prompts``) are
    model-independent by design and always call this function with
    ``model=None``.

    Keyword-only after *filename* is deliberate: a positional call site
    would otherwise silently rebind into the wrong parameter if this
    parameter list is ever reordered, instead of raising ``TypeError``.

    Two outcomes only: found → the file's stripped content; not found in
    any search directory → :exc:`FileNotFoundError` with the searched
    paths listed in the message. There is no fallback and no empty-string
    default — a missing prompt file always surfaces immediately rather
    than silently degrading behaviour.

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
    marks it as substituted rather than resolved from disk.  An override
    satisfies the load the same as a found file — the "missing" case only
    fires when no override matched AND the normal search exhausted every
    directory.

    Every resolution — the override case and the found-file case — is
    reported via :func:`paramem.graph.phase_trace.record_prompt`, so a
    :class:`~paramem.graph.phase_trace.PhaseRecord` for the calling phase
    always reflects the path/content this function actually returned,
    never a re-derivation of it.  ``record_prompt`` no-ops when no
    :func:`~paramem.graph.phase_trace.phase_trace` scope is active (e.g.
    a caller invoked at module import time, before any phase scope can
    exist), so this call is always safe.

    Raises:
        FileNotFoundError: When *filename* is not found in any search
            directory and no :func:`prompt_overrides` mapping matched it.
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
    searched = ", ".join(str(d / filename) for d in search_dirs)
    raise FileNotFoundError(f"Required prompt file {filename!r} not found. Searched: {searched}")
