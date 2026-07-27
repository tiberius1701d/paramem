"""The one primitive that lays down inspection artifacts, and the two scopes
that decide where they land.

An *artifact* is a file written for a human to read: a graph snapshot, a
relation dump, a recall probe, a calibration run's full record.  It is not
stored knowledge and not infrastructure state — those go through
:func:`paramem.backup.encryption.write_infra_bytes`, which honours the
operator's ``security.require_encryption`` posture.  Artifacts are
inspection-first and therefore always plaintext: an operator must be able to
``cat``/``grep`` them with no decrypt step.

Two producers, one primitive
----------------------------

Artifacts come from two triggers, gated independently:

* **debug** — ``config.debug`` (``save_cycle_snapshots``), the operator's
  standing "record everything" switch.  Scope opened by :func:`debug_run`.
* **calibration** — a ``/calibrate/*`` run.  Scope opened by
  :func:`calibration_run`.

Both are caller-opened context scopes, symmetric to ``start_at`` / ``stop_at``
in :mod:`paramem.graph.phase_trace`: a caller opens one once and every write
reachable from inside it, however deeply nested, lands in that root with no
parameter threading and no object handed down.  A calibration run captures its
artifacts whether or not the production debug switch is on — the two are
unrelated, and requiring ``debug: true`` to calibrate would make every probe
depend on a setting the operator changes for other reasons.  With both open the
same artifact is written to both roots: the debug tree stays a complete record
and the calibration run's directory stays self-contained.

:func:`_active_bases` returns an empty list when neither is open, which is what
makes every hook below a no-op without checking a flag itself.

Because the destination comes from a scope rather than from an object, this
module holds no reference to the pipeline: any layer can emit its own artifact
by calling its hook directly.  ``paramem.graph``, ``paramem.training``,
``paramem.server`` and ``scripts/dev/calibrate_prompts.py`` all import from
here.

The one exception
-----------------

:func:`on_main_adapters_saved` writes PEFT adapter *weights* — a directory of
safetensors produced by ``save_pretrained``, with its own atomicity (pending
slot + ``os.rename`` + manifest).  It is not a byte payload, and routing it
through :func:`write_artifact` would mean re-implementing PEFT serialisation
here — a second renderer of it.  This is the single stated exception to
"every artifact goes through :func:`write_artifact`"; nothing else is exempt.
Its import of ``save_adapter`` is local to the function so this module stays a
leaf over :mod:`paramem.backup.encryption`.

Purpose-keyed graph snapshot vocabulary (BINDING), each written by
:func:`on_fold_graph` under ``<base>/fold/`` as ``graph_<label>_snapshot.json``:

- ``reconstructed`` — the graph rebuilt from stored weights/disk at fold entry.
- ``merged`` — after registry-true re-merge (resolve_contradictions per config).
- ``enriched`` — after the graph-tier refine step.
- ``keyed`` — after keyed-entry assembly.

The root-level artifacts (relation lists, cycle summary) nest under
``interim_<stamp>/`` on an interim cycle because the caller's :func:`debug_run`
scope resolves its root from ``snapshot_dir_for(interim_stamp=…)``; the
``<base>/fold/`` graph snapshots keep the same basenames across cycle types.

Layout::

    <root>/
        sessions/<session_id>/<kind>_snapshot.json  # on_session_extracted
        episodic_rels_snapshot.json              # on_extraction_end
        procedural_rels_snapshot.json            # on_extraction_end
        recall_probes/<phase>_<adapter>.json     # on_recall_probe
        fold/
            graph_reconstructed_snapshot.json   # on_fold_graph reconstructed
            graph_merged_snapshot.json           # on_fold_graph merged (fold)
            graph_enriched_snapshot.json         # on_fold_graph enriched (fold)
            graph_keyed_snapshot.json            # on_fold_graph keyed
            removal_ledger.json                  # on_removal_ledger
            fold_assignments.json                # on_fold_assignments
            tier_delta.json                      # on_tier_delta
            normalization_snapshot.json          # on_normalization
        training/tiers/<tier>/adapter_weights/  # on_main_adapters_saved
        cycle_summary_snapshot.json              # on_cycle_end
        calibration_<stage>_<ts>.json            # on_calibration_result

For the debug root the caller resolves ``<root>`` from
``ConsolidationLoop.snapshot_dir_for(...)``
(``paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/``); for the
calibration root it is ``paths.calibration/artifacts/<stage>_<ts>/``.

The procedural relations file is omitted when the procedural list is empty.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import networkx as nx

from paramem.backup.encryption import write_plaintext_atomic

if TYPE_CHECKING:
    from paramem.graph.schema import SessionGraph


logger = logging.getLogger(__name__)


def _safe_path_component(value: Any) -> str:
    """Map any character that isn't alnum/``-``/``_`` to ``_`` (regex-free).

    Neutralises path separators and ``.`` so a request-controlled value
    (e.g. ``session_id``) cannot traverse out of the artifact root when
    used as a filename or directory component.
    """
    sanitized = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(value))
    return sanitized or "_"


# The two artifact roots.  ``None`` means that producer is not active:
# for debug, ``config.debug`` is off (the caller's ``snapshot_dir_for``
# returns None); for calibration, no run is in flight.
_DEBUG_ROOT: ContextVar[Path | None] = ContextVar("paramem_debug_root", default=None)
_CALIBRATION_ROOT: ContextVar[Path | None] = ContextVar("paramem_calibration_root", default=None)


@contextmanager
def debug_run(root: Path | None) -> Iterator[None]:
    """Direct every artifact produced inside the block to the debug tree.

    Args:
        root: Directory for this cycle's artifacts, or ``None`` when debug
            snapshots are disabled — passing ``None`` is how the gate is
            expressed, so callers never test the flag themselves. Created on
            first write.
    """
    token = _DEBUG_ROOT.set(root)
    try:
        yield
    finally:
        _DEBUG_ROOT.reset(token)


@contextmanager
def calibration_run(root: Path) -> Iterator[None]:
    """Direct every artifact produced inside the block to *root*.

    This is what makes a calibration run capture the artifacts of the
    production code it exercises — the normalization pass's raw outputs, a
    session graph, anything else a hook emits — independently of the
    production ``debug`` switch.  The debug root keeps its own gate: with
    ``debug: true`` both receive the artifact.

    Args:
        root: Directory for this run's artifacts. Created on first write.
    """
    token = _CALIBRATION_ROOT.set(root)
    try:
        yield
    finally:
        _CALIBRATION_ROOT.reset(token)


def _active_bases() -> list[Path]:
    """Every root the current write should land under, in order.

    Returns an empty list when neither producer is active.
    """
    return [root for root in (_DEBUG_ROOT.get(), _CALIBRATION_ROOT.get()) if root is not None]


def write_artifact(path: Path, payload: Any) -> None:
    """THE write routine for every artifact, wherever it is produced.

    One entry point so no artifact can be written by a route with different
    atomicity or different directory semantics than its siblings — which is
    exactly what a bare ``path.write_text`` beside a set of atomic writers
    gives you, silently.

    Accepts the three payload shapes the pipeline actually produces:

    * ``bytes`` — already serialized;
    * ``str`` — already serialized to text (e.g. a Pydantic model's
      ``model_dump_json``), encoded here rather than re-serialized;
    * anything else — a JSON-able object, serialized here with ``default=str``
      so a stray non-JSON value degrades to its repr instead of aborting the
      cycle that was only trying to record itself.

    Always plaintext (see the module docstring) and always atomic: a
    half-written artifact is worse than none.  The parent directory is created
    here because :func:`~paramem.backup.encryption._atomic_write_bytes` opens
    ``<path>.tmp`` directly and does not create it.
    """
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_plaintext_atomic(path, data)


# ---------------------------------------------------------------------------
# Phase hooks — one per pipeline boundary, so call sites read as
# ``artifacts.on_X(...)`` with no conditional.
# ---------------------------------------------------------------------------


def on_session_extracted(graph: "SessionGraph", session_id: str, kind: str) -> None:
    """Persist a per-session :class:`SessionGraph` (with diagnostics).

    Called immediately after every ``extraction.run`` /
    ``extraction.run_procedural`` invocation so every extractor output is
    captured before downstream merging strips per-session diagnostics
    (``cloud_raw_response``, ``residual_dropped_facts``,
    ``cloud_updated_transcript``, ``fallback_path``).

    ``kind`` names the extractor that produced the graph (``"graph"`` /
    ``"procedural_graph"``), not the adapter the relations eventually flow
    into — adapter allocation is downstream of extraction.
    """
    for base in _active_bases():
        write_artifact(
            base / "sessions" / _safe_path_component(session_id) / f"{kind}_snapshot.json",
            graph.model_dump_json(indent=2),
        )


def on_extraction_end(episodic_rels: list[dict], procedural_rels: list[dict]) -> None:
    """Persist per-tier relation lists.

    Called once per cycle at the end of extraction (after any interim graph
    enrichment mutates the cumulative graph, before training).  The relation
    lists are the per-cycle inputs to training; they are dumped verbatim so a
    calibration tool can compare extracted-vs-trained sets.

    The cumulative graph is NOT written here — it is emitted as purpose-keyed
    snapshots by :func:`on_fold_graph`: ``graph_merged_snapshot.json``
    (pre-enrichment) and ``graph_enriched_snapshot.json`` (post-enrichment).

    Skipped whenever ``run_consolidation_cycle`` short-circuits before
    reaching the interim fold body: the ``noop`` guards (no registry / no
    relations) and the ``cap_pending`` guard (interim ring full) all return
    without calling this function.  Of those three, only ``cap_pending``
    additionally calls :func:`on_cycle_end` with its own summary — the two
    ``noop`` guards call neither.
    """
    for base in _active_bases():
        write_artifact(base / "episodic_rels_snapshot.json", episodic_rels)
        if procedural_rels:
            write_artifact(base / "procedural_rels_snapshot.json", procedural_rels)
        logger.info(
            "Debug artifacts written to %s: %d episodic, %d procedural relations",
            base,
            len(episodic_rels),
            len(procedural_rels),
        )


def on_fold_graph(graph: "nx.MultiDiGraph", *, label: str) -> None:
    """Persist a purpose-keyed graph snapshot at a fold boundary.

    Serialises through :func:`write_artifact` like every other artifact — the
    node-link rendering is ``nx.node_link_data``, called once, here.

    Args:
        graph: The cumulative graph to snapshot.
        label: Purpose token — one of ``"reconstructed"``, ``"merged"``,
            ``"enriched"``, ``"keyed"``.  The output is
            ``<base>/fold/graph_<label>_snapshot.json``.
    """
    payload = nx.node_link_data(graph)
    for base in _active_bases():
        out_path = base / "fold" / f"graph_{label}_snapshot.json"
        write_artifact(out_path, payload)
        logger.info("Debug fold-graph snapshot written: %s (%s)", label, out_path)


def on_removal_ledger(ledger: dict) -> None:
    """Persist the merger's removal ledger.

    Writes ``<base>/fold/removal_ledger.json``.  The ledger maps each removed
    edge's ``ik_key`` to a dict carrying ``"reason"`` (one of ``"dedup"``,
    ``"contradiction_same_pred"``, ``"enrichment_same_as"``) and per-reason
    detail fields.

    Called once per fold AFTER the drift classifier has consumed the ledger
    (the ledger is final — ``reset_graph`` cleared it before the fold's
    re-merge populated it).
    """
    for base in _active_bases():
        write_artifact(base / "fold" / "removal_ledger.json", ledger)
        logger.info("Debug removal_ledger written: %d entries", len(ledger))


def on_fold_assignments(tier_keyed: dict) -> None:
    """Persist the fold's per-tier key assignment.

    Writes ``<base>/fold/fold_assignments.json`` with per-tier key lists (not
    full entry dicts — keys are the stable identifiers; SPO is recoverable
    from the registry and the keyed graph snapshot).

    Called once the fold's per-tier assignment is final, before
    ``_all_keyed`` is computed.

    Args:
        tier_keyed: Mapping of tier → list of entry dicts.
    """
    payload = {
        "tier_keyed": {tier: [e["key"] for e in entries] for tier, entries in tier_keyed.items()},
    }
    for base in _active_bases():
        write_artifact(base / "fold" / "fold_assignments.json", payload)
        logger.info(
            "Debug fold_assignments written: tier_keyed=%s",
            {t: len(v) for t, v in tier_keyed.items()},
        )


def on_main_adapters_saved(model, tier_names: list[str]) -> None:
    """Dump per-cycle adapter-weight shadows for inspection/diff.

    THE ONE EXCEPTION to "every artifact goes through :func:`write_artifact`"
    — see the module docstring.  Each tier lands at
    ``<base>/training/tiers/<tier>/adapter_weights/``.

    Called from ``_save_adapters`` after the canonical slot writes in
    ``paths.adapters/`` succeed.
    """
    from paramem.models.loader import save_adapter

    for base in _active_bases():
        tiers_root = base / "training" / "tiers"
        for tier in tier_names:
            save_adapter(model, tiers_root / tier / "adapter_weights", tier)


def on_recall_probe(per_key: list[dict] | None, *, phase: str, adapter_name: str) -> None:
    """Persist a per-key recall verdict (with raw_output) for post-mortem.

    Writes ``<base>/recall_probes/<phase>_<adapter_name>.json``.  Each element
    of *per_key* carries at minimum ``key``, ``exact_match``, ``confidence``,
    SPO ground-truth and recalled fields, ``failure_reason``, and
    ``raw_output`` — exactly the shape produced by
    :func:`~paramem.training.recall_eval.evaluate_indexed_recall`.

    No-op when *per_key* is ``None``.
    """
    if per_key is None:
        return
    for base in _active_bases():
        write_artifact(base / "recall_probes" / f"{phase}_{adapter_name}.json", per_key)


def on_tier_delta(tier_delta: dict) -> None:
    """Persist the per-tier before/after/staled/minted delta record.

    Writes ``<base>/fold/tier_delta.json``.  The payload is the
    ``"tier_delta"`` entry emitted in the fold result dict — a mapping of tier
    name to ``{active_before, active_after, staled_by_reason, minted}``.

    Called once per fold, after ``tier_keyed`` and ``minted_by_tier`` are
    finalised.  Emitted for BOTH the train and simulate fold paths.
    ``staled_by_reason`` is derived from ``merger.removal_ledger`` and includes
    all removal reasons (dedup, enrichment_same_as, etc.).  For simulate mode,
    store entries are absent so ``tier_of`` returns None for all removed keys,
    making ``staled_by_reason`` effectively ``{}`` in practice — a
    persistence-tail divergence, not a skipped grooming step.
    """
    for base in _active_bases():
        write_artifact(base / "fold" / "tier_delta.json", tier_delta)
        logger.info("Debug tier_delta written: %d tier(s)", len(tier_delta))


def on_normalization(
    raw_outputs: list[str], decisions: list[dict], applied: dict[str, int]
) -> None:
    """Persist the whole-graph normalization pass outputs.

    Writes ``<base>/fold/normalization_snapshot.json`` carrying the raw model
    output(s), the parsed decisions, and the count of applied operations, so
    personal facts stay out of the system journal.

    Args:
        raw_outputs: Raw model output strings (one per chunk call).
        decisions: Parsed JSON decision dicts (one per chunk call, ``None``
            entries excluded so length may be less than ``raw_outputs``).
        applied: Counts of applied operations, e.g.
            ``{"groups_collapsed": N, "edges_retired": M}``.
    """
    payload = {"raw_outputs": raw_outputs, "decisions": decisions, "applied": applied}
    for base in _active_bases():
        write_artifact(base / "fold" / "normalization_snapshot.json", payload)
        logger.info(
            "Debug normalization_snapshot written: %d chunk(s) %s",
            len(raw_outputs),
            applied,
        )


def on_calibration_result(payload: dict[str, Any]) -> None:
    """Persist a calibration run's full result.

    Writes ``calibration_<stage>_<timestamp>.json`` — the parsed output
    (including ``diagnostics``, e.g. entity-correction proposals the apply gate
    rejected, which never reach ``graph.diagnostics["entity_corrections"]``),
    every phase record, and the raw model output.  The timestamp keeps
    repeated runs of the same stage from clobbering each other, and the
    artifact is complete enough to be posted back as the next run's seed.

    Lands in the calibration run's directory whenever one is open
    (:func:`calibration_run`) and in the debug tree whenever debug is on.
    """
    stage = _safe_path_component(payload.get("stage", "run"))
    for base in _active_bases():
        out_path = base / f"calibration_{stage}_{int(time.time())}.json"
        write_artifact(out_path, payload)
        logger.info("Calibration artifact written: %s", out_path)


def on_cycle_end(cycle_summary: dict[str, Any]) -> None:
    """Persist the per-cycle summary record.

    Called from exactly two sites in ``run_consolidation_cycle``: the
    ``cap_pending`` early return (interim ring full — no fold attempted) and
    the interim fold's terminal return (``trained`` / ``simulated`` /
    ``recall_failed``).  NOT called for the two ``noop`` early returns (no
    registry / no relations) or the ``aborted`` return (training yielded to
    an inference request before the commit window) — those short-circuit
    before either call site is reached.  Schema is the
    ``run_consolidation_cycle`` return dict, kept open-ended so callers can
    extend without coordinating a writer change.
    """
    for base in _active_bases():
        write_artifact(base / "cycle_summary_snapshot.json", cycle_summary)
