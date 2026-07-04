"""Per-phase tracing for the extraction pipeline.

Every prompt-touching phase of :func:`paramem.graph.extractor.extract_graph`
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
:func:`_parse_extraction`, :func:`_validate_with_ha_context` both
return new ``SessionGraph`` objects) — the trace survives because it lives in a
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

Pipeline contract
-----------------

:func:`extract_graph` emits exactly these phases when the corresponding
step runs (in firing order for the default configuration):

==========================  ===============================================
Phase                       Notes
==========================  ===============================================
``local_extract``           Mistral runs the extraction prompt.
``ha_validation``           Pure-Python location validation against HA
                            home context. ``raw_output=None``.
``anonymize``               Mistral runs the anonymizer; emits mapping
                            + anonymized facts + anonymized transcript.
``anonymize_verify``        Pure-Python residual-leak verifier.
                            ``raw_output=None``.
``anonymize_repair``        Pure-Python leak repair (extend mapping or
                            drop facts).  Fires only when verify finds
                            leaks.  ``raw_output=None``.
``sota_enrich``             SOTA cloud (Anthropic by default) runs the
                            enrichment prompt; emits enriched facts +
                            new_entity_bindings + updated_anon_transcript.
``anon_plausibility``       Optional SOTA judge at anonymized stage.
``deanon``                  Pure-Python dict substitution restoring real
                            names from placeholders.  ``raw_output=None``.
``grounding_gate``          Pure-Python drop of facts whose entities
                            don't appear in the original transcript.
                            ``raw_output=None``.
``deanon_plausibility``     Local Mistral plausibility filter at
                            de-anonymized stage.
``merge_into_cumulative``   Pure-Python merge of the session graph into
                            the cumulative consolidation graph.  Fires
                            from ``ConsolidationLoop.extract_session``.
                            ``raw_output=None``.
``procedural_extract``      Mistral runs the procedural extraction
                            prompt.  Fires from
                            ``ConsolidationLoop.extract_session`` (its
                            own nested ``extract_graph`` trace records
                            into the outer session trace via the
                            nesting-no-op).
``dedup_episodic``          Pure-Python triple-identity dedup of the
                            episodic relation set.  ``raw_output=None``.
``dedup_procedural``        Pure-Python triple-identity dedup of the
                            procedural relation set.  ``raw_output=None``.
==========================  ===============================================

Adding a new phase: extend :data:`PHASE_NAMES`, document it in the
table above, and wrap the new phase boundary with
:func:`phase_trace`.  The whitelist guards against typos.
"""

from __future__ import annotations

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
    "ha_validation",
    "anonymize",
    "anonymize_verify",
    "anonymize_repair",
    "sota_enrich",
    "anon_plausibility",
    "deanon",
    "grounding_gate",
    "deanon_plausibility",
    "merge_into_cumulative",
    "procedural_extract",
    "dedup_episodic",
    "dedup_procedural",
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
        precondition didn't hold — e.g. SOTA missing API key),
        ``"no_input"`` (phase had nothing to do), or ``"failed"`` (an
        exception propagated through the scope; ``reason`` is set).
    wall_clock_seconds:
        Time inside the ``with phase_trace(...)`` block, including all
        nested work.  Captured automatically by the context manager.
    raw_output:
        Raw LLM/transformer text the phase generated, when applicable.
        ``None`` for pure-Python phases (HA validation, deanon,
        grounding gate, etc.).
    parsed:
        Phase-specific structured result.  Shape is per-phase but each
        phase's writer documents its keys (entity counts, mapping
        size, dropped-fact list, …).
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
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for ``graph.diagnostics`` serialisation."""
        d = asdict(self)
        if not d["extra"]:
            d.pop("extra")
        return d


class _PhaseScope:
    """Mutable scope handed to the ``with phase_trace(...)`` body.

    Caller fills via setter methods; the context manager finalises into
    an immutable :class:`PhaseRecord` on exit.
    """

    __slots__ = ("name", "_raw_output", "_parsed", "_outcome", "_reason", "_extra")

    def __init__(self, name: str):
        self.name: str = name
        self._raw_output: str | None = None
        self._parsed: dict[str, Any] | None = None
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
        """Set ``outcome`` (``ok``/``skipped``/``no_input``/``failed``)
        with an optional one-line reason."""
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


@contextmanager
def extraction_trace() -> Iterator[ExtractionTrace]:
    """Open a top-level extraction trace scope.

    Inside this scope, every :func:`phase_trace` call appends to the
    yielded :class:`ExtractionTrace`.  Helpers nested arbitrarily deep
    (parsing, anonymization, SOTA enrichment, etc.) participate without
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
    t0 = time.perf_counter()
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
                reason=scope._reason,
                extra=dict(scope._extra),
            ).to_dict()
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
                reason=item.get("reason"),
                extra=item.get("extra") or {},
            )
        )
    return out
