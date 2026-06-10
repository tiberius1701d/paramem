"""Single owner for every plaintext debug-snapshot write produced by the
consolidation pipeline.

Before this module landed, four separate writer sites in
``paramem/training/consolidation.py`` and ``paramem/server/consolidation.py``
each gated on the same ``save_cycle_snapshots`` / ``config.debug`` flag and
each laid bytes under overlapping subtrees of ``paths.debug``.  Operators
had to reason about which write happened on which code path.

This class is the *only* surface that lays bytes under ``paths.debug``.
All gating lives in :meth:`_active_base`; callers do not check the flag.
Methods are phase-shaped — each one corresponds to a single pipeline
boundary — so the call sites read as ``self._debug_writer.on_X(...)``
with no further conditionals.

Purpose-keyed graph snapshot vocabulary (BINDING):

- ``reconstructed`` (fold only): ``graph_reconstructed_snapshot.json``
- ``merged`` (fold + interim): ``graph_merged_snapshot.json``
  Fold: after additive re-merge; interim: after per-session merge,
  before enrichment.
- ``enriched`` (fold + interim): ``graph_enriched_snapshot.json``
- ``keyed`` (fold only): ``graph_keyed_snapshot.json``

Fold snapshots write under ``<base>/fold/``; interim snapshots write
under the interim base root (``_active_base`` nests them under
``interim_<stamp>/``).  The filename is identical across both paths so
``grep graph_merged_snapshot.json debug/`` finds both.

The legacy cumulative ``graph_snapshot.json`` (from ``on_extraction_end``) is
RETIRED — replaced by ``graph_merged_snapshot.json`` (pre-enrich) +
``graph_enriched_snapshot.json`` (post-enrich).  The per-session
``graph_snapshot.json`` under ``sessions/<sid>/`` is UNCHANGED.

Layout::

    paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/
        sessions/<session_id>/<kind>_snapshot.json  # on_session_extracted
        graph_merged_snapshot.json               # on_fold_graph merged (interim)
        graph_enriched_snapshot.json             # on_fold_graph enriched (interim)
        episodic_rels_snapshot.json              # on_extraction_end
        procedural_rels_snapshot.json            # on_extraction_end
        fold/
            graph_reconstructed_snapshot.json   # on_fold_graph reconstructed
            graph_merged_snapshot.json           # on_fold_graph merged (fold)
            graph_enriched_snapshot.json         # on_fold_graph enriched (fold)
            graph_keyed_snapshot.json            # on_fold_graph keyed
            removal_ledger.json                  # on_removal_ledger
            fold_assignments.json                # on_fold_assignments
        training/tiers/<tier>/adapter_weights/  # on_main_adapters_saved
        cycle_summary_snapshot.json              # on_cycle_end

The procedural relations file is omitted when the procedural list is
empty (mirrors the pre-refactor behaviour of ``_save_debug_artifacts``).
All output is plaintext regardless of the server's Security posture —
debug output is inspection-first; operators must be able to read it
with ``cat``/``grep`` without any decrypt step.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paramem.backup.encryption import write_plaintext_atomic
from paramem.models.loader import save_adapter

if TYPE_CHECKING:
    from paramem.graph.schema import SessionGraph
    from paramem.training.consolidation import ConsolidationLoop


logger = logging.getLogger(__name__)


class DebugSnapshotWriter:
    """Owns every plaintext debug write under ``paths.debug``.

    Instantiated once per :class:`~paramem.training.consolidation.ConsolidationLoop`
    as ``loop._debug_writer``.  Self-gates on
    ``loop.save_cycle_snapshots and loop._debug_base is not None`` so
    callers never check the flag.
    """

    def __init__(self, loop: "ConsolidationLoop"):
        self._loop = loop

    # ------------------------------------------------------------------
    # Gate
    # ------------------------------------------------------------------

    def _active_base(self, *, interim_stamp: str | None = None) -> Path | None:
        """Return the per-cycle debug directory or ``None`` when disabled.

        Resolves the active interim stamp from the loop when the caller
        does not pass one explicitly — mirrors the pre-refactor
        ``snapshot_dir_for`` semantics.
        """
        loop = self._loop
        if not loop.save_cycle_snapshots or loop._debug_base is None:
            return None
        resolved = interim_stamp or loop._current_interim_stamp_or_none()
        return loop.snapshot_dir_for(interim_stamp=resolved)

    # ------------------------------------------------------------------
    # Phase hooks
    # ------------------------------------------------------------------

    def on_session_extracted(
        self,
        graph: "SessionGraph",
        session_id: str,
        kind: str,
        *,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist a per-session :class:`SessionGraph` (with diagnostics).

        Called immediately after every ``self.extraction.run`` /
        ``self.extraction.run_procedural`` invocation so every extractor
        output is captured before downstream merging strips per-session
        diagnostics (``sota_raw_response``, ``residual_dropped_facts``,
        ``sota_updated_transcript``, ``fallback_path``).

        ``kind`` names the extractor that produced the graph
        (``"graph"`` / ``"procedural_graph"``), not the adapter the
        relations eventually flow into — adapter allocation is downstream
        of extraction.
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        out_path = base / "sessions" / session_id / f"{kind}_snapshot.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(graph.model_dump_json(indent=2))

    def on_extraction_end(
        self,
        episodic_rels: list[dict],
        procedural_rels: list[dict],
        *,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist per-tier relation lists.

        Called once per cycle at the end of extraction (after any
        interim graph enrichment mutates the cumulative graph, before
        training).  The relation lists are the per-cycle inputs to
        training; they are dumped verbatim so a calibration tool can
        compare extracted-vs-trained sets.

        The cumulative graph is NO LONGER written here — it is emitted
        as purpose-keyed snapshots by :meth:`on_fold_graph`:
        ``graph_merged_snapshot.json`` (pre-enrichment) and
        ``graph_enriched_snapshot.json`` (post-enrichment).

        Skipped for short-circuit cycles (queue-only, degenerated-skip);
        those emit only :meth:`on_cycle_end`.
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        base.mkdir(parents=True, exist_ok=True)
        _write_json(base / "episodic_rels_snapshot.json", episodic_rels)
        if procedural_rels:
            _write_json(base / "procedural_rels_snapshot.json", procedural_rels)
        logger.info(
            "Debug artifacts written to %s: %d episodic, %d procedural relations",
            base,
            len(episodic_rels),
            len(procedural_rels),
        )

    def on_fold_graph(
        self,
        graph_or_merger,
        *,
        label: str,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist a purpose-keyed graph snapshot at a fold or interim boundary.

        Args:
            graph_or_merger: Either a ``GraphMerger`` (duck-typed by
                ``hasattr(..., "save_graph")``) or a bare
                ``nx.MultiDiGraph``.  Dispatch is done once; no
                isinstance check so the caller never imports both types.
            label: Purpose token — one of ``"reconstructed"``,
                ``"merged"``, ``"enriched"``, ``"keyed"``.  The output
                filename is ``graph_<label>_snapshot.json``.
            interim_stamp: When set, the base directory is resolved
                under the matching ``interim_<stamp>/`` subdirectory.
                Leave ``None`` for the fold path (no interim nesting).

        Output paths:
            - Fold: ``<base>/fold/graph_<label>_snapshot.json``
            - Interim: ``<base>/graph_<label>_snapshot.json``
              (``_active_base`` already nests under ``interim_<stamp>/``)

        The basename is identical across both paths so a grep on the
        filename finds both fold and interim snapshots.
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        if interim_stamp is not None:
            # Interim path: write at the base root (already nested under interim_<stamp>/).
            out_path = base / f"graph_{label}_snapshot.json"
        else:
            # Fold path: use a dedicated fold/ subdir to avoid naming collision with
            # the interim root-level snapshots.
            out_path = base / "fold" / f"graph_{label}_snapshot.json"
        if hasattr(graph_or_merger, "save_graph"):
            # GraphMerger: saves serialized cumulative graph.  save_graph mkdirs parent.
            graph_or_merger.save_graph(out_path, encrypted=False)
        else:
            # Bare nx.MultiDiGraph: save via persistence helper.
            # save_memory_to_disk docstring guarantees it creates the parent directory.
            from paramem.memory.persistence import save_memory_to_disk

            save_memory_to_disk(graph_or_merger, out_path, encrypted=False)
        logger.info("Debug fold-graph snapshot written: %s (%s)", label, out_path)

    def on_removal_ledger(
        self,
        ledger: dict,
        *,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist the merger's removal ledger as a debug artifact.

        Writes ``<base>/fold/removal_ledger.json`` via the plaintext
        atomic writer.  The ledger maps each removed edge's ``ik_key``
        to a dict carrying ``"reason"`` (one of ``"dedup"``,
        ``"contradiction_same_pred"``, ``"contradiction_cross_pred"``,
        ``"enrichment_same_as"``) and per-reason detail fields.

        Called once per fold AFTER the drift classifier has consumed the
        ledger (the ledger is final — ``reset_graph`` cleared it before
        the fold's re-merge populated it).
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        _write_json(base / "fold" / "removal_ledger.json", ledger)
        logger.info("Debug removal_ledger written: %d entries", len(ledger))

    def on_fold_assignments(
        self,
        serve_assignment: dict,
        train_assignment: dict,
        *,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist serve and train tier assignment maps.

        Writes ``<base>/fold/fold_assignments.json`` with per-tier key
        lists (not full entry dicts — keys are the stable identifiers;
        SPO is recoverable from the registry and the keyed graph
        snapshot).

        Called after Pass-2 finalises ``serve_assignment`` (the two-map
        decoupling point) and before ``_all_keyed`` is computed.

        Args:
            serve_assignment: Mapping of tier → list of entry dicts.
                Keys extracted per-tier; full dicts are NOT stored here
                to keep the artifact compact.
            train_assignment: Same shape; differs from ``serve_assignment``
                only for fast-start graduating tiers.
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        payload = {
            "serve_assignment": {
                tier: [e["key"] for e in entries] for tier, entries in serve_assignment.items()
            },
            "train_assignment": {
                tier: [e["key"] for e in entries] for tier, entries in train_assignment.items()
            },
        }
        _write_json(base / "fold" / "fold_assignments.json", payload)
        logger.info(
            "Debug fold_assignments written: serve=%s train=%s",
            {t: len(v) for t, v in serve_assignment.items()},
            {t: len(v) for t, v in train_assignment.items()},
        )

    def on_main_adapters_saved(
        self,
        tier_names: list[str],
        *,
        interim_stamp: str | None = None,
    ) -> None:
        """Dump per-cycle adapter-weight shadows for inspection/diff.

        Each tier in *tier_names* lands at
        ``<debug_base>/training/tiers/<tier>/adapter_weights/`` — the
        2026-05-17 relocation from the legacy
        ``<output_dir|snapshot_dir>/cycle_<N>/<tier>/`` path.  No
        deprecation alias; the relocation is one-time.

        Called from ``_save_adapters`` after the canonical slot writes
        in ``paths.adapters/`` succeed.
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        tiers_root = base / "training" / "tiers"
        for tier in tier_names:
            save_adapter(self._loop.model, tiers_root / tier / "adapter_weights", tier)

    def on_recall_probe(
        self,
        per_key: list[dict] | None,
        *,
        phase: str,
        adapter_name: str,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist a per-key recall verdict (with raw_output) for post-mortem.

        Writes ``<base>/recall_probes/<phase>_<adapter_name>.json``.  Each
        element of *per_key* carries at minimum ``key``, ``exact_match``,
        ``confidence``, SPO ground-truth and recalled fields, ``failure_reason``,
        and ``raw_output`` — exactly the shape produced by
        :func:`~paramem.training.recall_eval.evaluate_indexed_recall`.

        No-op when debug is disabled (``save_cycle_snapshots=False`` or
        ``_debug_base is None``) or when *per_key* is ``None``.
        """
        if per_key is None:
            return
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        _write_json(base / "recall_probes" / f"{phase}_{adapter_name}.json", per_key)

    def on_cycle_end(
        self,
        cycle_summary: dict[str, Any],
        *,
        interim_stamp: str | None = None,
    ) -> None:
        """Persist the per-cycle summary record.

        Fires from every return branch of ``run_consolidation_cycle``
        (normal, queue-only, degenerated-skip) so an operator can
        confirm the cycle was reached even when no triples were
        produced.  Schema is the ``run_consolidation_cycle`` return
        dict, kept open-ended so callers can extend without coordinating
        a writer change.
        """
        base = self._active_base(interim_stamp=interim_stamp)
        if base is None:
            return
        base.mkdir(parents=True, exist_ok=True)
        _write_json(base / "cycle_summary_snapshot.json", cycle_summary)


def _write_json(path: Path, data: Any) -> None:
    """Plaintext atomic JSON write — debug output is inspection-first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_plaintext_atomic(path, json.dumps(data, indent=2).encode("utf-8"))
