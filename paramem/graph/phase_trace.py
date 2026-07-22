"""Per-phase tracing for the extraction pipeline.

Every prompt-touching phase of :func:`paramem.graph.flows.extract_graph`
records a :class:`PhaseRecord` to the **active extraction trace** as it
runs.  The trace is the canonical observation surface for:

* **Calibration** — diff per-phase prompt outputs across prompt variants.
* **Debugging** — reconstruct a failing run phase-by-phase from the
  dumped records alone.
* **Optimization** — per-phase wall-clock timings for budget allocation.

Architecture
------------

The trace is **decoupled from any particular** :class:`SessionGraph`
**instance**.  Pipeline helpers freely rebind ``graph = ...`` (e.g.
:func:`_parse_extraction` returns a new ``SessionGraph``
object) — the trace survives because it lives in a
:class:`contextvars.ContextVar` set up by an outer
:func:`extraction_trace` scope, not on the graph.

Usage
-----

::

    def extract_graph(...) -> SessionGraph:
        with extraction_trace() as trace:
            graph = SessionGraph(...)
            try:
                with phase_trace("local_extract") as t:
                    raw = _generate_extraction(...)
                    t.set_raw(raw)
                    graph = _parse_extraction(raw, ...)   # rebinding is FINE
                    t.set_parsed({...})
                # ... more phases, more rebinds, all OK ...
            finally:
                trace.attach_to(graph)        # materialise on final graph
        return graph

:func:`phase_trace` is callable from any function reachable from the
``extraction_trace`` scope — no parameter threading.  Outside such a
scope it raises :class:`RuntimeError` immediately rather than silently
dropping records.

Chain-stop signal
------------------

:func:`stop_at` — opened by the CALLER, exactly mirroring
:func:`paramem.graph.prompts.prompt_overrides` — names a phase from
:data:`PHASE_NAMES` after which the pipeline should stop.
:func:`phase_trace` flips the internal "stopped" contextvar to ``True``
the moment that phase finishes with a non-``"failed"`` outcome; the
pipeline checks :func:`chain_stopped` — never a phase-name string —
after each phase block and returns early when it is set.  Three sites
check it: the stage-flow runner
(:func:`paramem.graph.flow.run_flow`, after every stage that ran — this
is what stops the walk at the ``anonymize`` stage boundary now; that
stage no longer checks the flag itself), the ``deanonymize`` stage body
between its two phases, and the ``enrich`` stage body
(:func:`~paramem.graph.stage_enrich._stage_enrich`) between its own
sub-phases.  Because the flag lives in a
:class:`contextvars.ContextVar` rather than being threaded as a
parameter, it is reachable from arbitrarily nested helpers with no
signature changes down the call chain.  Production opens no
:func:`stop_at` scope, so nothing ever stops the chain — this is
calibration-only.

Prompt provenance
-----------------

Every stage's prompt LOADER (:func:`paramem.graph.prompts._load_prompt`)
calls :func:`record_prompt` right after it resolves a file, so
``PhaseRecord.prompts`` holds exactly what the loader handed the stage —
the same resolution the production run made, not a re-implementation of
it.  Each entry is ``{"path", "sha", "template"}``:

* ``template`` is the loader's raw output — the prompt TEMPLATE with
  slots such as ``{transcript}`` still literal and unsubstituted, never
  the rendered prompt.  A rendered prompt would embed the session
  transcript (PII) in ``graph.diagnostics`` and therefore in
  ``data/ha/debug/graph_snapshot.json``; the template never does.
* ``path`` is the file actually resolved (which may be the shipped
  default under ``_DEFAULT_PROMPT_DIR`` even when the caller requested
  an operator-supplied ``prompts_dir`` — that fallback is exactly the
  divergence this mechanism exists to make visible), or ``None`` when no
  file was found anywhere and the caller's hardcoded default was used.
* ``sha`` is the first 12 hex characters of the SHA-256 of ``template``.

:func:`record_prompt` is a no-op when called with no active
:func:`phase_trace` scope — unlike :func:`phase_trace` itself, which
raises.  This is deliberate, not an inconsistency: ``_load_prompt`` is a
shared loader that cannot know whether it is being called from inside a
phase; two real production call paths run it with no phase scope open
(``anonymize_transcript`` from ``anonymize_turn``,
reached from the live chat egress, and from
``paramem.training.consolidation``), so a raise there would break
legitimate production behaviour. ``phase_trace``'s caller, by contrast,
has explicitly declared a phase boundary — a lost record there is a
contract breach the raise is meant to catch.

Pipeline contract
-----------------

:func:`extract_graph` emits exactly these phases when the corresponding
step runs (in firing order for the default configuration):

==========================  ===============================================
Phase                       Notes
==========================  ===============================================
``local_extract``           Mistral runs the extraction prompt.
``second_order_extract``    ``local_extract`` is speaker-centric, so a
                            clause naming a non-speaker person by
                            relationship tends to keep only the
                            speaker->person edge and drop that person's
                            OWN fact. This pass re-extracts facts ABOUT
                            each named non-speaker person ``local_extract``
                            surfaced. Gated on the pass-1 graph containing
                            such an entity; skipped entirely (no record)
                            otherwise.
``anonymize``               Mistral runs the anonymizer; emits ONLY the
                            real->placeholder mapping.  The script builds
                            the anonymized facts + transcript from
                            ``graph.relations`` and this mapping.
``entity_correction``       Mistral corrects misspelled real place /
                            organization / concept surfaces in the
                            reverse anonymization map (values only).
``cloud_enrich``             cloud (Anthropic by default) runs the
                            enrichment prompt; emits enriched facts +
                            new_entity_bindings + updated_anon_transcript.
``anon_plausibility``       Optional cloud judge at anonymized stage.
``deanon``                  Pure-Python dict substitution restoring real
                            names from placeholders.  ``raw_output=None``.
``deanon_plausibility``     Local Mistral plausibility filter at
                            de-anonymized stage.
``merge_into_cumulative``   Pure-Python merge of the session graph into
                            the cumulative consolidation graph.  Fires
                            from ``ConsolidationLoop.extract_session``.
                            ``raw_output=None``.
``procedural_extract``      Mistral runs the procedural extraction
                            prompt via the shared ``_run_local_extraction``
                            primitive inside ``extract_procedural_graph``,
                            which self-traces this phase — its own
                            ``extraction_trace`` scope nest-no-ops onto
                            whatever outer trace (session or calibration)
                            is already active.
``dedup_episodic``          Pure-Python triple-identity dedup of the
                            episodic relation set.  ``raw_output=None``.
``dedup_procedural``        Pure-Python triple-identity dedup of the
                            procedural relation set.  ``raw_output=None``.
``normalize``               Synonym-predicate clustering over same-
                            (subject, object) groups via
                            ``normalize_predicates``; loads
                            ``predicate_normalization.txt`` +
                            ``predicate_normalization_system.txt``.
``name_extract``            Local Mistral runs the name-extraction prompt
                            on an explicit turn list (calibration/
                            enrollment). Loads ``name_extraction.txt`` +
                            ``name_extraction_system.txt``.
==========================  ===============================================

Adding a new phase: extend :data:`PHASE_NAMES`, document it in the
table above, and wrap the new phase boundary with
:func:`phase_trace`.  The whitelist guards against typos.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from paramem.graph.schema import SessionGraph


# Canonical phase-name whitelist.  Order matches the firing order in
# extract_graph for the default configuration; phases not configured for
# a given run (e.g. anon_plausibility when plausibility_stage='deanon')
# simply do not fire — order in the trace reflects what actually ran.
PHASE_NAMES: tuple[str, ...] = (
    "local_extract",
    "second_order_extract",
    "anonymize",
    "entity_correction",
    "cloud_enrich",
    "anon_plausibility",
    "deanon",
    "deanon_plausibility",
    "merge_into_cumulative",
    "procedural_extract",
    "dedup_episodic",
    "dedup_procedural",
    "normalize",
    "name_extract",
)


@dataclass(frozen=True)
class PhaseRecord:
    """One phase observation captured by :func:`phase_trace`.

    Uniform shape across all phases so consumers (calibration tool,
    debug dumps, reviewers) iterate generically.

    Attributes
    ----------
    name:
        Canonical phase name from :data:`PHASE_NAMES`.
    outcome:
        ``"ok"`` (phase produced its output normally), ``"skipped"`` (a
        precondition didn't hold — e.g. Cloud missing API key),
        ``"no_input"`` (phase had nothing to do), ``"failed"`` (an
        exception propagated through the scope; ``reason`` is set), or
        ``"rejected"`` (the phase produced output but the caller rejected
        it as a unit and fell back to a prior-stage input — e.g.
        ``cloud_enrich``'s binding-totality gate rejecting an enrichment
        delta that references an orphan or CORE-conflicting placeholder;
        ``reason`` is set). No consumer branches on the value.
    wall_clock_seconds:
        Time inside the ``with phase_trace(...)`` block, including all
        nested work.  Captured automatically by the context manager.
    raw_output:
        Raw LLM/transformer text the phase generated, when applicable.
        ``None`` for pure-Python phases (HA validation, deanon,
        dedup, etc.).
    parsed:
        Phase-specific structured result.  Shape is per-phase but each
        phase's writer documents its keys (entity counts, mapping
        size, dropped-fact list, …).
    prompts:
        Prompt templates :func:`paramem.graph.prompts._load_prompt`
        resolved for this phase, recorded via :func:`record_prompt` at
        the loader's own chokepoint.  Each entry is
        ``{"path", "sha", "template"}``.  ``template`` is the
        loader's OUTPUT — the template with slots such as
        ``{transcript}`` still UNSUBSTITUTED — never the rendered
        prompt: rendering would put the session transcript (PII) into
        ``graph.diagnostics`` and thus ``data/ha/debug/graph_snapshot.json``.
        ``None`` when the phase loaded no prompt (pure-Python phases) or
        the phase's loader calls happened outside any active phase scope.
    reason:
        Optional one-line explanation, set when ``outcome != "ok"``.
    extra:
        Phase-specific additional fields the writer wants to record but
        that don't fit ``parsed``.
    """

    name: str
    outcome: str = "ok"
    wall_clock_seconds: float = 0.0
    raw_output: str | None = None
    parsed: dict[str, Any] | None = None
    prompts: list[dict[str, Any]] | None = None
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for ``graph.diagnostics`` serialisation."""
        d = asdict(self)
        if not d["extra"]:
            d.pop("extra")
        if not d["prompts"]:
            d.pop("prompts")
        return d


class _PhaseScope:
    """Mutable scope handed to the ``with phase_trace(...)`` body.

    Caller fills via setter methods; the context manager finalises into
    an immutable :class:`PhaseRecord` on exit.
    """

    __slots__ = (
        "name",
        "_raw_output",
        "_parsed",
        "_prompts",
        "_outcome",
        "_reason",
        "_extra",
    )

    def __init__(self, name: str):
        self.name: str = name
        self._raw_output: str | None = None
        self._parsed: dict[str, Any] | None = None
        self._prompts: list[dict[str, Any]] = []
        self._outcome: str = "ok"
        self._reason: str | None = None
        self._extra: dict[str, Any] = {}

    def set_raw(self, raw_output: str | None) -> None:
        """Record the raw LLM/transformer output for this phase."""
        self._raw_output = raw_output

    def set_parsed(self, parsed: dict[str, Any] | None) -> None:
        """Record the structured per-phase result."""
        self._parsed = parsed

    def set_outcome(self, outcome: str, *, reason: str | None = None) -> None:
        """Set ``outcome`` (``ok``/``skipped``/``no_input``/``failed``/
        ``rejected``) with an optional one-line reason."""
        self._outcome = outcome
        self._reason = reason

    def add(self, key: str, value: Any) -> None:
        """Attach a phase-specific extra field."""
        self._extra[key] = value


class ExtractionTrace:
    """Owns the per-phase records for a single ``extract_graph`` call.

    Returned by :func:`extraction_trace`.  Decoupled from any particular
    :class:`SessionGraph` instance — the pipeline freely rebinds
    ``graph = ...`` and the trace survives.  Call :meth:`attach_to` once
    when a final graph is in hand.

    Records what happened, never what was asked — the chain-stop request
    (see :func:`stop_at`) is a separate, independent contextvar and has
    no representation here.
    """

    __slots__ = ("_records",)

    def __init__(self):
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[PhaseRecord]:
        """Typed view of the phase records (for in-process consumers)."""
        out: list[PhaseRecord] = []
        for item in self._records:
            out.append(
                PhaseRecord(
                    name=item.get("name", ""),
                    outcome=item.get("outcome", "ok"),
                    wall_clock_seconds=float(item.get("wall_clock_seconds") or 0.0),
                    raw_output=item.get("raw_output"),
                    parsed=item.get("parsed"),
                    prompts=item.get("prompts"),
                    reason=item.get("reason"),
                    extra=item.get("extra") or {},
                )
            )
        return out

    def to_dict_list(self) -> list[dict[str, Any]]:
        """JSON-friendly list of records for transport / on-disk dumps."""
        return [dict(r) for r in self._records]

    def attach_to(self, graph: "SessionGraph") -> None:
        """Materialise the records on ``graph.diagnostics["phases"]``.

        Call once, on the final graph the pipeline produces.  Idempotent
        within a single trace (later calls overwrite earlier attachments).
        """
        graph.diagnostics["phases"] = self.to_dict_list()


# ContextVar holding the active :class:`ExtractionTrace` instance.  Set by
# :func:`extraction_trace`; read by :func:`phase_trace`.  Default ``None``
# makes :func:`phase_trace` raise immediately when called outside a trace
# scope — silent record loss is unacceptable.
_ACTIVE_TRACE: ContextVar[ExtractionTrace | None] = ContextVar(
    "paramem_extraction_trace", default=None
)

# ContextVar holding the active :class:`_PhaseScope` instance.  Set and reset
# by :func:`phase_trace` around its ``yield``; read by :func:`record_prompt`.
# Default ``None`` means :func:`record_prompt` called with no phase open
# NO-OPS rather than raising — see "Prompt provenance" in the module
# docstring for why this differs from :func:`phase_trace`'s own contract.
_ACTIVE_SCOPE: ContextVar[_PhaseScope | None] = ContextVar("paramem_phase_scope", default=None)

# ContextVar holding the phase name (from PHASE_NAMES) after which the chain
# should stop, or ``None`` for the full pipeline (production default).  Set
# by :func:`stop_at`; read by :func:`phase_trace`.  Mirrors
# ``paramem.graph.prompts._PROMPT_OVERRIDES`` — opened by the CALLER, never
# threaded as a parameter down the pipeline.
_STOP_AT: ContextVar[str | None] = ContextVar("paramem_stop_at", default=None)

# ContextVar tracking whether the requested stop phase has fired.  ``False``
# until :func:`phase_trace` records a phase whose name matches ``_STOP_AT``
# with a non-``"failed"`` outcome, then ``True`` for the rest of the
# :func:`stop_at` scope's lifetime.  Read via :func:`chain_stopped`.
_STOPPED: ContextVar[bool] = ContextVar("paramem_stopped", default=False)


@contextmanager
def stop_at(phase: str | None) -> Iterator[None]:
    """Request the extraction chain stop after ``phase`` completes.

    Mirrors :func:`paramem.graph.prompts.prompt_overrides`: opened by the
    CALLER (calibration), never threaded as a parameter through
    :class:`~paramem.graph.extraction_pipeline.ExtractionPipeline`,
    :func:`paramem.graph.flows.extract_graph`, or
    :func:`~paramem.graph.stage_enrich._stage_enrich`.  Every
    :func:`phase_trace` call anywhere inside the ``with`` body sees the
    same requested phase via the contextvar, however deeply nested.

    Args:
        phase: A name from :data:`PHASE_NAMES` after which the pipeline
            should stop, or ``None`` to run the full pipeline (the
            production default — production opens no :func:`stop_at`
            scope at all, so this is effectively always ``None`` outside
            calibration).

    Raises:
        ValueError: When ``phase`` is not ``None`` and not in
            :data:`PHASE_NAMES`.  Validated once here, before any work
            happens, rather than running the full pipeline and silently
            never matching.

    Nesting: like :func:`~paramem.graph.prompts.prompt_overrides`, an
    inner call fully replaces the outer value for its duration (no
    merge); both the requested phase and the stopped flag are restored
    to the outer state on exit via ``ContextVar.reset``.
    """
    if phase is not None and phase not in PHASE_NAMES:
        raise ValueError(
            f"stop_phase {phase!r} is not a valid phase name. Valid: {list(PHASE_NAMES)}"
        )
    token_phase = _STOP_AT.set(phase)
    token_stopped = _STOPPED.set(False)
    try:
        yield
    finally:
        _STOP_AT.reset(token_phase)
        _STOPPED.reset(token_stopped)


def chain_stopped() -> bool:
    """Whether the active :func:`stop_at` request has been satisfied.

    ``False`` when no :func:`stop_at` scope is open, or when the
    requested phase has not yet completed.  Set permanently ``True`` for
    the rest of the scope's lifetime by :func:`phase_trace`'s success
    branch the moment the requested phase records with a
    non-``"failed"`` outcome.  Pipeline functions check this (never a
    phase-name string comparison) after each phase block to decide
    whether to return early — see :func:`paramem.graph.flow.run_flow`,
    ``paramem.graph.flows._stage_deanonymize`` and
    :func:`paramem.graph.stage_enrich._stage_enrich`.
    """
    return _STOPPED.get()


@contextmanager
def extraction_trace() -> Iterator[ExtractionTrace]:
    """Open a top-level extraction trace scope.

    Inside this scope, every :func:`phase_trace` call appends to the
    yielded :class:`ExtractionTrace`.  Helpers nested arbitrarily deep
    (parsing, anonymization, cloud enrichment, etc.) participate without
    any explicit threading.

    **Nesting is a no-op**: re-entering ``extraction_trace`` while a
    trace is already active yields the existing trace, so the inner
    scope's phase records still land on the outer trace.  This makes
    the API safe to use from test fixtures that wrap every test (the
    test's own ``with extraction_trace():`` becomes the same as the
    fixture's, no conflict).
    """
    existing = _ACTIVE_TRACE.get()
    if existing is not None:
        yield existing
        return
    trace = ExtractionTrace()
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE_TRACE.reset(token)


@contextmanager
def phase_trace(name: str) -> Iterator[_PhaseScope]:
    """Append a :class:`PhaseRecord` to the active extraction trace.

    Wall-clock is captured automatically.  Exceptions raised inside the
    body propagate AFTER a partial record is appended with
    ``outcome="failed"`` and ``reason=<exception class>: <message>``.

    Raises
    ------
    RuntimeError
        When called outside an active :func:`extraction_trace` scope.
        Silent record loss is unacceptable, so this fails loudly.
    ValueError
        When ``name`` is not in :data:`PHASE_NAMES`.  The whitelist is
        the contract between the pipeline and the trace.
    """
    trace = _ACTIVE_TRACE.get()
    if trace is None:
        raise RuntimeError(
            "phase_trace called outside an active extraction_trace() scope. "
            "Wrap the pipeline call (or the test setup) in `with "
            "extraction_trace() as trace:` first."
        )
    if name not in PHASE_NAMES:
        raise ValueError(
            f"Unknown phase name {name!r}. Add it to PHASE_NAMES in "
            f"paramem/graph/phase_trace.py before using it."
        )
    scope = _PhaseScope(name)
    token = _ACTIVE_SCOPE.set(scope)
    t0 = time.perf_counter()
    try:
        try:
            yield scope
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            trace._records.append(
                PhaseRecord(
                    name=name,
                    outcome="failed",
                    wall_clock_seconds=round(elapsed, 4),
                    raw_output=scope._raw_output,
                    parsed=scope._parsed,
                    prompts=list(scope._prompts) or None,
                    reason=f"{type(exc).__name__}: {exc}",
                    extra=dict(scope._extra),
                ).to_dict()
            )
            raise
        else:
            elapsed = time.perf_counter() - t0
            trace._records.append(
                PhaseRecord(
                    name=name,
                    outcome=scope._outcome,
                    wall_clock_seconds=round(elapsed, 4),
                    raw_output=scope._raw_output,
                    parsed=scope._parsed,
                    prompts=list(scope._prompts) or None,
                    reason=scope._reason,
                    extra=dict(scope._extra),
                ).to_dict()
            )
            # Chain-stop signal — set AFTER the record above is appended,
            # so the phase's own output (raw/parsed/prompts) is never
            # lost.  Only a phase that finished normally can stop the
            # chain: a phase this block marked ``outcome="failed"`` (a
            # soft failure the caller handles without raising, e.g. a
            # cloud call returning ``None``) must NOT be mistaken for the
            # requested stop point, since the caller's own downstream
            # fallback/rejection logic still needs to run.  A phase that
            # raised never reaches this branch at all (see the ``except``
            # branch above), so a hard failure can never set this either.
            if name == _STOP_AT.get() and scope._outcome != "failed":
                _STOPPED.set(True)
    finally:
        _ACTIVE_SCOPE.reset(token)


def record_prompt(*, path: str | None, content: str) -> None:
    """Record a loaded prompt template on the active :func:`phase_trace` scope.

    Called from :func:`paramem.graph.prompts._load_prompt` right after it
    resolves a file, so the recorded ``path``/``template`` are exactly
    what the loader resolved and returned — never a re-implementation of
    that resolution.

    Args:
        path: The resolved file path as a string, or ``None`` when no file
            was found anywhere and the caller's hardcoded ``default`` was
            used instead.
        content: The exact text the loader is about to return — the
            template with slots such as ``{transcript}`` still
            UNSUBSTITUTED.  Never the rendered prompt (see the module
            docstring's "Prompt provenance" section for why).

    No-op when there is no active phase scope — unlike :func:`phase_trace`,
    which raises in the analogous situation.  This is intentional, not an
    oversight to "fix" into a raise:

    * ``_load_prompt`` (transitively, via :func:`paramem.graph.prompts.
      _load_speaker_directive_section`) is called at MODULE IMPORT TIME to
      build ``paramem.server.inference.THIRD_PARTY_DESCRIPTOR``
      (``inference.py:52``) — long before any :func:`extraction_trace`/
      :func:`phase_trace` scope can exist.  A raise here would break
      ``import paramem.server.inference`` outright; this is the decisive
      reason a raise is not an option, independent of any call-time
      scoping question.
    * Two real production call paths also run it with no phase scope
      open at CALL time — ``anonymize_transcript`` from
      ``anonymize_turn`` (reached from the live chat
      egress, ``paramem/server/inference.py:639``,
      ``extractor.py:1108``) and from
      ``paramem.training.consolidation`` (``consolidation.py:2739``).

    ``phase_trace``'s raise exists because ITS caller has explicitly
    declared a phase boundary, so a lost record there is a contract
    breach; ``record_prompt``'s caller has made no such declaration.
    """
    scope = _ACTIVE_SCOPE.get()
    if scope is None:
        return
    scope._prompts.append(
        {
            "path": path,
            "sha": hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
            "template": content,
        }
    )


def get_phases(graph: "SessionGraph") -> list[PhaseRecord]:
    """Read-side typed accessor over ``graph.diagnostics["phases"]``.

    Returns the records in the order they fired.  Empty list when no
    phases ran (e.g. extract_graph never opened a trace, or
    ``trace.attach_to(graph)`` was never called).
    """
    diag = getattr(graph, "diagnostics", None) or {}
    raw = diag.get("phases") or []
    out: list[PhaseRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            PhaseRecord(
                name=item.get("name", ""),
                outcome=item.get("outcome", "ok"),
                wall_clock_seconds=float(item.get("wall_clock_seconds") or 0.0),
                raw_output=item.get("raw_output"),
                parsed=item.get("parsed"),
                prompts=item.get("prompts"),
                reason=item.get("reason"),
                extra=item.get("extra") or {},
            )
        )
    return out
