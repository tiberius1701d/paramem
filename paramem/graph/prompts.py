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

import string
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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


# The sectioned local-anonymization prompt home's basename — the one place
# this filename is written. `paramem.graph.anonymizer_prompts` and the
# anonymizer gate tool (`scripts/dev/anonymizer_gate.py`) both import this
# constant rather than repeating the literal.
ANONYMIZATION_PROMPT_FILE = "anonymization.txt"

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
    ANONYMIZATION_PROMPT_FILE,
)

# The sectioned local-anonymization prompt home
# (`paramem.graph.anonymizer_prompts.load_anonymizer_prompts` is the one
# composer that reads it). Every section here is required — a scan or
# anchor call with a missing section would otherwise fail at the first
# request that needs it rather than at boot. The SCAN section carries the
# configured categories only through its own `{keywords}` render slot,
# formatted per call — the home itself takes no category argument.
_ANONYMIZATION_REQUIRED_SECTIONS = (
    "SCAN-SYSTEM",
    "SCAN",
    "ANCHOR-SYSTEM",
    "ANCHOR",
)
# Per-section required slots — mirrors _EXTRACTION_TEMPLATE_SLOTS' role for
# the extraction templates: a section present but missing one of its own
# slots renders a payload-less call that silently under-scrubs rather than
# raising, because `str.format` ignores a slot the section text never
# references.
_ANONYMIZATION_SECTION_SLOTS: dict[str, tuple[str, ...]] = {
    "SCAN": ("{keywords}", "{text}"),
    "ANCHOR": ("{speaker_id}", "{values}", "{text}"),
}


def _template_slots(body: str) -> set[str]:
    """Every slot a template body references — the one reader every slot
    check in this module uses.

    A slot is a format field the standard library's own format-string
    parser (``string.Formatter().parse``) yields whose field name is not
    ``None`` — this includes the empty field name an auto-numbered ``{}``
    carries. A doubled brace (``{{``/``}}``) is not a slot: the parser
    already renders it as literal text, so a JSON example embedded in a
    prompt (e.g. ``{{"mapping": {{}}}}``) is never mistaken for a
    placeholder.

    Args:
        body: The template text to inspect.

    Returns:
        Each referenced slot in its braced form (e.g. ``"{text}"``;
        ``"{}"`` for an auto-numbered slot).

    Raises:
        ValueError: *body* has a malformed placeholder (e.g. a lone
            ``}``) — the format parser's own error, restated in plain
            words.
    """
    try:
        fields = list(string.Formatter().parse(body))
    except ValueError as exc:
        raise ValueError(f"malformed placeholder ({exc})") from exc
    return {"{" + field_name + "}" for _, field_name, _, _ in fields if field_name is not None}


@dataclass(frozen=True)
class SlotProblem:
    """One slot-contract problem :func:`_slot_problems` found.

    ``kind`` is the one field every caller dispatches on to decide how to
    word the problem (a malformed placeholder is reported bare; a missing
    or unknown slot is reported against its section) — never a substring
    test on ``text`` (e.g. ``text.startswith("malformed")``), so the
    wording of ``text`` is free to change without breaking a caller.

    Attributes:
        kind: ``"malformed"``, ``"missing"`` or ``"unknown"``.
        text: The plain-language problem description, with no section or
            file name prefixed — the caller adds that.
    """

    kind: str
    text: str


def _slot_problems(body: str, *, required: tuple[str, ...]) -> list[SlotProblem]:
    """Every problem *body* has against its own required slot set.

    The required set doubles as the allowed set — a template may carry no
    slot outside what it is required to carry — so this is the one
    comparison :func:`check_anonymization_prompt_sections` and
    :func:`ensure_prompt_assets`'s extraction-template walk both read
    through, instead of each writing its own missing/unknown comparison.

    Slots are read through :func:`_template_slots`, the one slot reader
    every check in this module uses, so a doubled brace (``{{``/``}}``) is
    never mistaken for a placeholder.

    Args:
        body: The template text to inspect.
        required: Slots *body* must carry, and the only slots it may
            carry — each absence is its own problem, in *required* order;
            each slot found outside this set is its own problem,
            alphabetical.

    Returns:
        Every problem found, never raised: a malformed placeholder stops
        the read and is the sole item (:func:`_template_slots`'s own
        message, already plain words, ``kind="malformed"``); otherwise
        zero or more ``kind="missing"``/``kind="unknown"`` entries. Empty
        when *body* satisfies the contract.
    """
    try:
        found = _template_slots(body)
    except ValueError as exc:
        return [SlotProblem(kind="malformed", text=str(exc))]
    problems = [
        SlotProblem(kind="missing", text=f"is missing slot {slot}")
        for slot in required
        if slot not in found
    ]
    problems.extend(
        SlotProblem(kind="unknown", text=f"carries an unknown slot {slot}")
        for slot in sorted(found - set(required))
    )
    return problems


class AnonymizationPromptInvalid(ValueError):
    """The anonymization prompt home fails its own section/slot contract.

    Raised by :func:`check_anonymization_prompt_sections`, carrying every
    problem found in one pass — a required section absent, a present
    section carrying a malformed placeholder, a present section missing
    one of its own required slots, or a present section carrying a slot
    its table does not list for it — rather than only the first.

    Attributes:
        problems: Every problem found, in plain words, in the order
            checked (every missing section first, then, per required
            section in table order, its own missing/unknown-slot or
            malformed-placeholder problems).
    """

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def check_anonymization_prompt_sections(*, prompts_dir: Path | None = None) -> None:
    """Check the anonymization prompt home's sections and slots.

    Loads ``anonymization.txt`` (an active :func:`prompt_overrides`
    substitution is honored, exactly as any other prompt load, so this
    reads whichever copy is actually loaded: an operator's
    ``--prompt-file`` override when one is active, otherwise whichever
    copy ``paths.prompts`` resolves — an operator's configured prompt
    directory when set, the shipped ``configs/prompts/anonymization.txt``
    otherwise) and checks, over every problem rather than stopping at the
    first: every required section (:data:`_ANONYMIZATION_REQUIRED_SECTIONS`)
    is present; every section listed in :data:`_ANONYMIZATION_SECTION_SLOTS`
    that IS present carries every slot its own row lists and no slot its
    row does not list (:func:`_slot_problems`, the one required-slot
    comparison every check in this module shares).

    This is the one place either check runs: :func:`ensure_prompt_assets`
    calls it at boot for whichever copy the server loads (an operator's
    configured prompts directory, or the shipped
    ``configs/prompts/anonymization.txt`` when none is configured), and the
    anonymizer gate tool (``scripts/dev/anonymizer_gate.py``) calls it for a
    ``--prompt-file`` override, inside its own :func:`prompt_overrides`
    scope.

    Args:
        prompts_dir: Forwarded to :func:`_load_prompt_sections` unchanged
            — the operator ``paths.prompts`` override. ``None`` (default)
            resolves only the shipped ``configs/prompts/`` copy (or an
            active override).

    Raises:
        AnonymizationPromptInvalid: One or more problems found — carrying
            every one of them (see :attr:`AnonymizationPromptInvalid.problems`).
    """
    sections = _load_prompt_sections(ANONYMIZATION_PROMPT_FILE, prompts_dir=prompts_dir)

    problems: list[str] = [
        f"missing section {name}"
        for name in _ANONYMIZATION_REQUIRED_SECTIONS
        if name not in sections
    ]
    for section_name, required_slots in _ANONYMIZATION_SECTION_SLOTS.items():
        if section_name not in sections:
            continue  # already reported as a missing section above
        for problem in _slot_problems(sections[section_name], required=required_slots):
            if problem.kind == "malformed":
                problems.append(f"section {section_name}: {problem.text}")
            else:
                problems.append(f"section {section_name} {problem.text}")

    if problems:
        raise AnonymizationPromptInvalid(problems)


# The extraction user templates and each one's own closed slot set — the
# slots its one render site always supplies (paramem.graph.extractor.
# _generate_extraction, via `format_kwargs`), derived from the call graph:
# `extraction.txt` and `extraction_procedural.txt` receive exactly
# `transcript`/`speaker_context`/`document_context` (extract_graph,
# extract_procedural_graph — neither passes `extra_slots`);
# `extraction_second_order.txt` receives those three plus `named_people`,
# the one `extra_slots` member its one caller (`_stage_second_order_extract`,
# paramem.graph.flows) supplies. The set is closed for all three, so a
# template must carry every slot listed AND no other — a per-model or
# operator copy that drops one silently reverts that copy to cue-less/
# context-less extraction, and one that adds an unlisted slot renders a
# `KeyError` at call time; `str.format` ignores surplus kwargs either way,
# so neither failure is otherwise visible before it fires against a real
# request. Checked by ensure_prompt_assets.
_EXTRACTION_TEMPLATE_SLOTS: dict[str, tuple[str, ...]] = {
    "extraction.txt": ("{transcript}", "{speaker_context}", "{document_context}"),
    "extraction_procedural.txt": ("{transcript}", "{speaker_context}", "{document_context}"),
    "extraction_second_order.txt": (
        "{transcript}",
        "{speaker_context}",
        "{document_context}",
        "{named_people}",
    ),
}


def _active_override(filename: str) -> str | None:
    """The content an active :func:`prompt_overrides` substitution provides
    for *filename*, or ``None`` when no override matches.

    The one place ``_PROMPT_OVERRIDES.get()`` is read and matched against a
    filename — :func:`_load_prompt` (which needs the content) and
    :func:`_prompt_source_label` (which needs only to know an override is
    active) both call this rather than each repeating the same two-line
    check.
    """
    override = _PROMPT_OVERRIDES.get()
    if override is not None and filename in override:
        return override[filename]
    return None


def _prompt_source_label(*, prompts_dir: Path | None = None) -> str:
    """The location :func:`_load_prompt` would actually load
    ``anonymization.txt`` from — the one label
    :func:`ensure_prompt_assets`'s anonymization report names, so a report
    never re-derives a second search order that could disagree with the
    real load.

    Hardcoded to :data:`ANONYMIZATION_PROMPT_FILE` rather than taking a
    *filename* parameter: this is the one model-independent home every
    caller resolves, so there is exactly one file for this function to
    name.

    Checks the same active :func:`prompt_overrides` substitution
    :func:`_load_prompt` checks first (:func:`_active_override`, the one
    shared override lookup), then :func:`_resolve_prompt_path`. Callers
    reach this function only after :func:`ensure_prompt_assets`'s own
    required-file check has already confirmed the file's presence in the
    shipped tree, so :func:`_resolve_prompt_path` never raises here — no
    fallback branch is needed.

    Args:
        prompts_dir: Operator ``paths.prompts`` override, forwarded to
            :func:`_resolve_prompt_path`.

    Returns:
        ``<override:anonymization.txt>`` under an active
        :func:`prompt_overrides` substitution; otherwise the on-disk path
        :func:`_resolve_prompt_path` finds.
    """
    if _active_override(ANONYMIZATION_PROMPT_FILE) is not None:
        return f"<override:{ANONYMIZATION_PROMPT_FILE}>"
    return str(_resolve_prompt_path(ANONYMIZATION_PROMPT_FILE, prompts_dir=prompts_dir))


def _prompt_fix_advice(path: str) -> str:
    """Plain-language remediation for one prompt-asset problem, keyed on
    where the named file lives.

    A ``<override:...>`` marker (an active :func:`prompt_overrides`
    substitution — no production caller of :func:`ensure_prompt_assets` runs
    under an override; the gate tool (``scripts/dev/anonymizer_gate.py``)
    validates its ``--prompt-file`` through
    :func:`check_anonymization_prompt_sections` directly, and a calibration
    run's prompt variants (:func:`~paramem.server.calibrate.resolve_prompt_variants`)
    are checked only for on-disk existence, neither path reaching this
    function) or a path resolving outside the shipped
    ``configs/prompts/`` tree is the operator's own copy, with no shipped
    fallback to fall back on — the advice is to fix that file directly.

    A path resolving inside the shipped tree (:data:`_DEFAULT_PROMPT_DIR`)
    could equally be a broken checkout or an operator's own in-place edit
    of the shipped copy — editing these files in place to tune behaviour
    is a documented operator practice (``DEPLOYMENT.md``'s Prompt
    Engineering section), so the path alone never tells them apart. Both
    remedies are named.

    Args:
        path: One problem's location string, as named in the raised
            report (an override marker or an on-disk path).

    Returns:
        ``"fix that file"`` or
        ``"fix that file, or restore the shipped copy from the repository"``.
    """
    if path.startswith("<override:"):
        return "fix that file"
    try:
        Path(path).resolve().relative_to(_DEFAULT_PROMPT_DIR.resolve())
    except ValueError:
        return "fix that file"
    return "fix that file, or restore the shipped copy from the repository"


def ensure_prompt_assets(*, prompts_dir: Path | None = None) -> None:
    """Fail loudly when the shared prompt assets are missing at startup, or
    when the anonymization home or a resolvable extraction user template
    carries the wrong section/slot set.

    ParaMem deploys from a repo checkout (editable install under systemd), so
    ``configs/prompts/`` — the guaranteed final fallback in ``_load_prompt``'s
    search order — is always present in a correct deployment. A missing
    directory or file means a broken checkout or a non-editable ``pip install``
    (prompts are not shipped as package data). Surface that at boot instead of
    letting the extraction pipeline silently load empty prompts.

    Two content checks run in full, over BOTH files' worth of problems,
    before either raises — an operator fixing one problem per boot cycle
    never has to wait for a second run to learn about the next one: the
    anonymization home (:func:`check_anonymization_prompt_sections`,
    reading whichever copy is actually loaded — an operator override or
    the shipped tree) and every resolvable extraction user template's slot
    set. The slot walk covers ``([prompts_dir] if prompts_dir is not None
    else []) + [_DEFAULT_PROMPT_DIR]`` and each walked directory's
    immediate subdirectories (per-model override dirs). Roots are deduped
    by ``resolve()`` before walking — the default deployment has
    ``paths.prompts`` equal to the shipped dir, so without the dedupe the
    identical tree would be walked twice and every problem would appear
    duplicated. For every :data:`_EXTRACTION_TEMPLATE_SLOTS` file that
    exists in a walked directory, its own row is the required (and, since
    the two coincide, allowed) slot list, compared through
    :func:`_slot_problems` (the one required-slot comparison this module
    shares): a template silently missing a listed slot is a config-load
    failure, not a runtime surprise (``str.format`` ignores surplus
    kwargs, so a per-model copy without the slot would silently revert
    that model to cue-less/context-less extraction), and a template
    carrying a slot its row does not list would ``KeyError`` at call time
    instead. A malformed placeholder in either file is reported as a
    problem of that file, never a crash.

    Args:
        prompts_dir: Operator-configured ``paths.prompts`` override
            (``ServerConfig.prompts_dir``), passed by the server's lifespan
            startup at its one call site. ``None`` (default — tests, any
            caller with no config) checks only the shipped
            ``configs/prompts/`` tree; the walk never constructs ``Path(None)``.

    Raises:
        RuntimeError: When ``_DEFAULT_PROMPT_DIR`` is not a directory or a
            required prompt file is absent from it (checked first, since
            neither content check below has anything to read without
            them); otherwise, when the anonymization home or an extraction
            user template carries one or more problems — one error naming
            every problem found across both files, each prefixed with the
            path it was found in and its own remediation
            (:func:`_prompt_fix_advice`: fix an operator's own copy
            directly; fix or restore a shipped-tree copy, since an
            in-place edit there is as legitimate as a broken checkout).
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

    problems: list[tuple[str, str]] = []

    # `prompts_dir` threaded through — an operator override tree can
    # replace the anonymization home, so this validates the copy that will
    # actually be loaded at runtime, not only the shipped default. One
    # check, shared with the anonymizer gate tool's `--prompt-file`
    # validation: see `check_anonymization_prompt_sections`.
    try:
        check_anonymization_prompt_sections(prompts_dir=prompts_dir)
    except AnonymizationPromptInvalid as exc:
        path = _prompt_source_label(prompts_dir=prompts_dir)
        problems.extend((path, problem) for problem in exc.problems)

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
            # duplicate every problem below.
            continue
        seen_resolved.add(resolved)
        walked.append(root)
        walked.extend(d for d in root.iterdir() if d.is_dir())

    for directory in walked:
        for filename, slots in _EXTRACTION_TEMPLATE_SLOTS.items():
            path = directory / filename
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8")
            problems.extend(
                (str(path), problem.text) for problem in _slot_problems(content, required=slots)
            )

    if problems:
        lines = [f"{path}: {message} ({_prompt_fix_advice(path)})" for path, message in problems]
        raise RuntimeError("Prompt asset problem(s): " + "; ".join(lines))


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
    * ``anonymization.txt`` — the local-anonymizer home: ``SCAN-SYSTEM``,
      ``SCAN``, ``ANCHOR-SYSTEM`` and ``ANCHOR``, the four sections every
      render composes — composed by
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
    home (``anonymization.txt`` — every scan/anchor call, across
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
    override_content = _active_override(filename)
    if override_content is not None:
        record_prompt(path=f"<override:{filename}>", content=override_content)
        return override_content

    path = _resolve_prompt_path(filename, prompts_dir=prompts_dir, model=model)
    content = path.read_text(encoding="utf-8").strip()
    record_prompt(path=str(path), content=content)
    return content


def _resolve_prompt_path(
    filename: str, *, prompts_dir: Path | None = None, model: str | None = None
) -> Path:
    """The on-disk file :func:`_load_prompt` reads for *filename*, ignoring
    any active :func:`prompt_overrides` substitution (:func:`_load_prompt`
    checks that separately, before calling this).

    The one search-order implementation :func:`_load_prompt` and every
    path-naming caller (:func:`_prompt_source_label`, read by
    :func:`ensure_prompt_assets`'s anonymization report) share — a second
    copy of this order would let a boot report name a different file than
    the one actually loaded.

    Args:
        filename: Basename to resolve.
        prompts_dir: Operator ``paths.prompts`` override, as
            :func:`_load_prompt` receives it.
        model: Per-model override sub-directory, as :func:`_load_prompt`
            receives it.

    Returns:
        The first existing ``directory / filename`` in search order
        (``[prompts_dir/model, prompts_dir, _DEFAULT_PROMPT_DIR]``, the
        model directory only when both *prompts_dir* and *model* are
        given).

    Raises:
        FileNotFoundError: No candidate directory contains *filename*; the
            message lists every path searched.
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
            return path
    searched = ", ".join(str(d / filename) for d in search_dirs)
    raise FileNotFoundError(f"Required prompt file {filename!r} not found. Searched: {searched}")
