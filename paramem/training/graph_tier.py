"""Post-materialize graph-tier refinement: predicate normalization + cloud enrichment.

Owns the two passes that mutate the merged graph after the Materialize stage —
whole-graph local-model predicate-synonym normalization
(:meth:`GraphTierRefiner.run_normalization`) and cloud-cloud ego-graph
enrichment (:meth:`GraphTierRefiner.run_enrichment`, which dispatches into
:func:`paramem.training.graph_enrich.enrich_graph`) — behind the single
entry point :meth:`GraphTierRefiner.refine`.

Boundary: a :class:`GraphTierRefiner` mutates exactly the ``GraphMerger`` it is
constructed with and reaches nothing else — no memory store, no cycle counter,
no :class:`~paramem.training.consolidation.ConsolidationLoop`. The caller owns
scheduling, gating, the store's recurrence bumps, and any post-refine debug
snapshot of the whole graph; this module only ever sees the merger, a model
handle, and the small set of scalars/callables listed on
:class:`GraphTierRefiner`'s constructor.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

from paramem.cloud.admission import evaluate_cloud_egress
from paramem.graph.extractor import normalize_predicates
from paramem.graph.merger import GraphMerger, min_nonempty
from paramem.training import graph_enrich
from paramem.utils.identity import canonical

if TYPE_CHECKING:
    from paramem.graph.extraction_pipeline import ExtractionConfig

logger = logging.getLogger(__name__)


class NormalizationSink(Protocol):
    """Structural protocol for the one debug-write the normalization pass needs.

    :class:`~paramem.training.debug_snapshot.DebugSnapshotWriter` satisfies
    this protocol structurally — it is never imported here. Handing the
    refiner the full writer would give it a back-reference into
    :class:`~paramem.training.consolidation.ConsolidationLoop` (the writer
    stores ``self._loop``), which defeats the module boundary described in
    the module docstring. The refiner only ever needs this one method.
    """

    def on_normalization(
        self,
        raw_outputs: list[str],
        decisions: list[dict],
        applied: dict[str, int],
    ) -> None: ...


@dataclass(frozen=True)
class RefineResult:
    """Outcome of a single :meth:`GraphTierRefiner.refine` call.

    ``normalization`` and ``enrichment`` are the diagnostics dicts produced by
    each pass, or ``None`` when that pass was not requested. ``reinforcements``
    and ``adopt_reinforcements`` are copied off the merger (``dict(...)``) at
    the moment ``refine()`` returns, so a later mutation to the merger's live
    bookkeeping cannot retroactively change what the caller acts on.
    """

    normalization: "dict | None"
    enrichment: "dict | None"
    reinforcements: "dict[str, tuple[str, str]]"
    adopt_reinforcements: "dict[str, tuple[str, str]]"


class GraphTierRefiner:
    """Runs the post-materialize graph-tier passes over one ``GraphMerger``.

    Takes, deliberately, no memory store, no cycle counter, and no
    :class:`~paramem.training.consolidation.ConsolidationLoop` reference — see
    the module docstring. A probe can construct its own ``GraphMerger``, hand
    it to its own ``GraphTierRefiner``, and nothing either pass does can reach
    anything outside that merger.

    Intended lifetime is per call: construct fresh immediately before
    :meth:`refine`, drop on return. ``model``/``tokenizer`` are read fresh at
    construction rather than cached across folds because callers may re-wrap
    the model (``create_adapter`` / ``get_peft_model``) between folds; a
    longer-lived refiner would risk pinning a stale wrapper. See
    :meth:`release` for the (currently unused) cleanup path that keeps the
    class safe to hold if a future caller ever does.
    """

    def __init__(
        self,
        merger: "GraphMerger",
        *,
        model,
        tokenizer,
        extraction_config_provider: "Callable[[], ExtractionConfig]",
        cloud_enabled: bool,
        neighborhood_hops: int,
        max_entities_per_pass: int,
        gc_disable: "Callable[[], None] | None" = None,
        gc_enable: "Callable[[], None] | None" = None,
        normalization_sink: "NormalizationSink | None" = None,
    ) -> None:
        """Construct a refiner bound to one merger and one model handle.

        Args:
            merger: The single mutation target for both passes. Both read and
                write ``merger.graph`` and ``merger.removal_ledger``.
            model: Local model handle. ``None`` short-circuits both passes to
                their "no local model" skip path.
            tokenizer: Tokenizer paired with ``model``.
            extraction_config_provider: Zero-arg callable returning the
                :class:`~paramem.graph.extraction_pipeline.ExtractionConfig`
                (not the owning ``ExtractionPipeline`` — the pipeline is a
                second base-model holder this refiner has no business
                reaching). Only ``enrichment_provider``, ``enrichment_provider_model``,
                ``enrichment_provider_endpoint``, ``scrub``, and ``max_tokens`` are
                read, across both passes.

                Deferred, not a resolved value. Every one of this refiner's
                skip paths — ``no_model`` above all, which is what a released
                base model produces — must be reachable without touching the
                caller's extraction state, and a constructor argument is
                evaluated by the caller before any guard in this class can
                run. So the caller hands over the *read*, not its result, and
                each pass invokes it only past its own guards
                (:meth:`run_normalization` past its ``no_model``/``floor``
                skips, immediately before the cloud-admission verdict that
                selects the cloud or local engine; :meth:`run_enrichment`
                threads it unresolved into
                :func:`~paramem.training.graph_enrich.enrich_graph`,
                which resolves it past its own ``no_model``/``floor`` skips).
                This holds for a refiner constructed once and reused across
                both passes, not just a per-call one.
            cloud_enabled: Whether the caller's consolidation config permits
                cloud-cloud engine selection for normalization. Passed as a
                bare bool rather than the owning config so this module never
                needs to import ``ConsolidationConfig``.
            neighborhood_hops: Ego-graph radius for enrichment chunking.
            max_entities_per_pass: Max high-recurrence entities considered per
                enrichment pass.
            gc_disable: Zero-arg callable that disables gradient checkpointing,
                or ``None`` for a no-op. Toggled per chunk by
                :meth:`run_enrichment` around each cloud call — normalization
                does not use it (its own model calls disable/enable GC
                internally, inside :func:`~paramem.graph.extractor.normalize_predicates`).
            gc_enable: Counterpart to ``gc_disable``.
            normalization_sink: Optional :class:`NormalizationSink` that
                receives the raw model outputs, parsed decisions, and applied
                counts from :meth:`run_normalization`. ``None`` means no
                debug write.
        """
        # BASE-MODEL HOLDER (GraphTierRefiner): per-call lifetime, dropped when
        # the caller returns; no persistent reference exists.
        self.model = model
        self.tokenizer = tokenizer
        self._merger = merger
        self._extraction_config_provider = extraction_config_provider
        self._cloud_enabled = cloud_enabled
        self._neighborhood_hops = neighborhood_hops
        self._max_entities_per_pass = max_entities_per_pass
        self._gc_disable = gc_disable
        self._gc_enable = gc_enable
        self._normalization_sink = normalization_sink

    def release(self) -> None:
        """Null all references this refiner holds. Idempotent.

        Not on any release path today — a ``GraphTierRefiner`` is constructed
        per call and dropped by its caller on return (see the class
        docstring) — but must exist so the class is safe to hold if a future
        caller ever does.
        """
        self.model = None
        self.tokenizer = None
        self._merger = None
        self._gc_disable = None
        self._gc_enable = None

    def run_enrichment(self) -> dict:
        """Dispatch into :func:`paramem.training.graph_enrich.enrich_graph`.

        Not a shim — this is where the tier's shared state (the merger, the
        model handle, the GC toggle pair, the pass-sizing scalars) lives; the
        module-level function in ``graph_enrich`` is the implementation, this
        method is the seam that supplies it with this refiner's state.

        Returns:
            The enrichment diagnostics dict produced by
            :func:`~paramem.training.graph_enrich.enrich_graph`.
        """
        return graph_enrich.enrich_graph(
            self._merger,
            model=self.model,
            tokenizer=self.tokenizer,
            extraction_config_provider=self._extraction_config_provider,
            neighborhood_hops=self._neighborhood_hops,
            max_entities_per_pass=self._max_entities_per_pass,
            gc_disable=self._gc_disable,
            gc_enable=self._gc_enable,
        )

    def run_normalization(self) -> dict:
        """Whole-graph normalization pass using :func:`normalize_predicates`.

        Runs after the Materialize stage at refinement level ``light`` or higher,
        over the cumulative ``merger.graph``. Collapses synonym predicates on the
        same ``(subject, object)`` pair. One model call per candidate group.

        Engine selection (resolved once before the primitive call):

        * **Local** (default): ``model`` + ``tokenizer`` from this refiner.
        * **cloud**: when
          :func:`~paramem.cloud.admission.evaluate_cloud_egress` —
          the one shared cloud-admission component — permits egress for
          ``cloud_enabled`` plus the extraction config's ``enrichment_provider``
          provider/model/endpoint. Falls back to local silently when it
          refuses; normalization still runs.

        The model receives the predicate list for each candidate group via the
        ``{predicates_json}`` slot in ``predicate_normalization.txt`` and returns the
        cluster schema ``{"clusters": [["predA", "predB"], ...]}``. Both the
        filter prompt and ``predicate_normalization_system.txt`` load through
        ``_load_prompt`` inside :func:`normalize_predicates` itself
        (production fold passes no ``prompts_dir`` override, so the default
        search path is used). GC disable/enable is handled inside
        :func:`normalize_predicates`.

        Survivor selection: within each cluster, the predicate with the highest
        ``reinforcement_count`` (summed across all edges for that predicate)
        survives. Tie broken by max ``last_seen``; remaining ties by cluster
        iteration order.

        Overlapping clusters (same predicate in two clusters — a model
        hallucination) are handled by ``_retired_in_so_group`` tracking per
        ``(s, o)`` group: a predicate retired by cluster N is excluded from
        cluster N+1's candidate list, and ``graph.remove_edge`` is called
        fail-loud (no try/except) since the tracking guarantees each edge is
        retired at most once.

        Applied changes — same-(subject, object) collapse only (subtractive,
        grounded in input):

        1. Build ``_flat_relations`` and ``_so_groups`` from all predicate-bearing
           edges in ``merger.graph``.
        2. Call :func:`normalize_predicates` with the resolved engine kwargs.
        3. For each returned ``(can_s, can_o)`` cluster group: pick the MAX
           ``reinforcement_count`` edge as survivor; retire the rest.

           a. Union ``sessions``, sum ``reinforcement_count``, max ``confidence``,
              max ``last_seen`` of all retired edges onto the survivor BEFORE removal.
           b. Write ``removal_ledger[ik_key] = {"reason": "predicate_synonym_collapse",
              "merged_into": <survivor_pred>}`` for each retired keyed edge.
           c. Remove retired edges from the graph.

        4. Single-predicate ``(s, o)`` facts are NEVER retired.
        5. Hallucinated predicates in model output are grounded inside
           :func:`normalize_predicates` — no new edges are minted.

        Early-return conditions (all return ``skipped=True``):

        - No local model available (``model is None``).
        - Graph has fewer than 10 nodes (too little signal).
        - Prompt file missing: raises ``FileNotFoundError`` (fail-loud).

        Returns:
            Diagnostics dict with keys:
                - ``groups_collapsed`` (int): ``(s, o)`` groups with >=1 retirement.
                - ``edges_retired`` (int): total edges removed across all groups.
                - ``chunks`` (int): total model calls made.
                - ``skipped`` (bool): ``True`` when the pass was bypassed.
                - ``skip_reason`` (str | None): reason token when skipped.
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        _empty = {"groups_collapsed": 0, "edges_retired": 0, "chunks": 0}

        if self.model is None:
            logger.info("graph_normalization: no local model — skipping")
            return {**_empty, "skipped": True, "skip_reason": "no_model"}

        graph = self._merger.graph
        node_count = graph.number_of_nodes()

        if node_count < 10:
            logger.info(
                "graph_normalization: graph too small (%d nodes < 10 floor) — skipping",
                node_count,
            )
            return {**_empty, "skipped": True, "skip_reason": "floor"}

        # Build flat relation list and per-(s,o) group index.
        # Each _info entry: {u, v, eid, edata, ik_key, can_s, can_o, can_pred}
        _flat_relations: list[dict] = []
        _so_groups: dict[tuple, dict[str, list[dict]]] = {}

        for _u, _v, _eid, _edata in graph.edges(keys=True, data=True):
            _pred = _edata.get("predicate", "")
            if not _pred:
                continue
            _ik = _edata.get(_IK_KEY_ATTR) or None
            _can_s = canonical(_u)
            _can_o = canonical(_v)
            _can_pred = canonical(_pred)
            _info = {
                "u": _u,
                "v": _v,
                "eid": _eid,
                "edata": _edata,
                "ik_key": _ik,
                "can_s": _can_s,
                "can_o": _can_o,
                "can_pred": _can_pred,
            }
            _flat_relations.append({"subject": _u, "predicate": _pred, "object": _v})
            _so_groups.setdefault((_can_s, _can_o), {}).setdefault(_can_pred, []).append(_info)

        # Resolve which backend to use via the shared cloud-admission verdict;
        # fall back to local SILENTLY (info-level) when it refuses.  The
        # fallback is deliberate: the local model is always present, so
        # normalization still runs on the operator's knob rather than being
        # skipped because a cloud credential is absent.
        engine_kwargs: dict = {"model": self.model, "tokenizer": self.tokenizer}
        ext_cfg = self._extraction_config_provider()
        verdict = evaluate_cloud_egress(
            cloud_enabled=self._cloud_enabled,
            provider=ext_cfg.enrichment_provider,
            model=ext_cfg.enrichment_provider_model,
            endpoint=ext_cfg.enrichment_provider_endpoint or None,
        )
        if verdict.permitted:
            engine_kwargs = {
                "cloud": {
                    "api_key": verdict.api_key,
                    "provider": verdict.provider,
                    "filter_model": verdict.model,
                    "endpoint": verdict.endpoint,
                }
            }
            logger.info("graph_normalization: engine=cloud provider=%s", verdict.provider)
        else:
            logger.info(
                "graph_normalization: engine=local — cloud egress refused (%s)",
                "; ".join(verdict.gaps),
            )

        # Delegate model calls to normalize_predicates.
        # GC disable/enable is handled inside normalize_predicates.
        clusters_by_so, _normalization_diag = normalize_predicates(_flat_relations, **engine_kwargs)

        total_edges_retired = 0
        total_groups_collapsed = 0

        for so_key, group_clusters in clusters_by_so.items():
            pred_map = _so_groups.get(so_key)
            if not pred_map:
                continue

            # Track predicates already retired in this (s,o) group so that
            # overlapping clusters (model hallucination: same predicate in two
            # clusters) do not attempt a second remove_edge on an already-absent
            # edge, which would be an undetectable bug if silenced by try/except.
            _retired_in_so_group: set[str] = set()

            for cluster in group_clusters:
                # cluster is a list of canonical predicates confirmed as synonyms.
                # At least 2 members guaranteed by normalize_predicates.
                # Exclude predicates already retired by a prior cluster in this group
                # (can arise when model returns overlapping clusters).
                cluster_preds = [
                    cp for cp in cluster if cp in pred_map and cp not in _retired_in_so_group
                ]
                if len(cluster_preds) < 2:
                    continue

                # Survivor selection: MAX reinforcement_count (summed across all edges
                # for that predicate); tie-broken by MAX last_seen (ISO string max).
                def _pred_sort_key(cp: str) -> tuple:
                    edges = pred_map[cp]
                    rec = sum(e["edata"].get("reinforcement_count", 1) for e in edges)
                    ls = max((e["edata"].get("last_seen", "") for e in edges), default="")
                    return (rec, ls)

                _survivor_pred = max(cluster_preds, key=_pred_sort_key)
                _survivor_edges = pred_map[_survivor_pred]
                _survivor_edata = _survivor_edges[0]["edata"]

                # Union provenance from retired edges onto survivor BEFORE removal.
                _surv_sessions: list[str] = list(_survivor_edata.get("sessions", []))
                _surv_recurrence: int = _survivor_edata.get("reinforcement_count", 1)
                _surv_confidence: float = _survivor_edata.get("confidence", 0.0)
                _surv_last_seen: str = _survivor_edata.get("last_seen", "")
                _surv_first_seen: str = _survivor_edata.get("first_seen", "")

                _retired_preds = [cp for cp in cluster_preds if cp != _survivor_pred]
                for _ret_pred in _retired_preds:
                    for _ret_info in pred_map[_ret_pred]:
                        for _sid in _ret_info["edata"].get("sessions", []):
                            if _sid not in _surv_sessions:
                                _surv_sessions.append(_sid)
                        _surv_recurrence += _ret_info["edata"].get("reinforcement_count", 1)
                        _surv_confidence = max(
                            _surv_confidence, _ret_info["edata"].get("confidence", 0.0)
                        )
                        _surv_last_seen = max(
                            _surv_last_seen, _ret_info["edata"].get("last_seen", "")
                        )
                        _surv_first_seen = min_nonempty(
                            _surv_first_seen, _ret_info["edata"].get("first_seen", "")
                        )

                _survivor_edata["sessions"] = _surv_sessions
                _survivor_edata["reinforcement_count"] = _surv_recurrence
                _survivor_edata["confidence"] = _surv_confidence
                _survivor_edata["last_seen"] = _surv_last_seen
                _survivor_edata["first_seen"] = _surv_first_seen

                # Retire each non-survivor edge.
                group_retired = 0
                for _ret_pred in _retired_preds:
                    for _ret_info in pred_map[_ret_pred]:
                        _ret_ik = _ret_info["ik_key"]
                        if _ret_ik:
                            self._merger.removal_ledger[_ret_ik] = {
                                "reason": "predicate_synonym_collapse",
                                "merged_into": _survivor_pred,
                            }
                        # Fail-loud: the (u, v, eid) triple was built from
                        # graph.edges(keys=True) and each predicate is retired at most
                        # once (guaranteed by _retired_in_so_group tracking above),
                        # so remove_edge must always succeed.
                        graph.remove_edge(_ret_info["u"], _ret_info["v"], _ret_info["eid"])
                        logger.info(
                            "graph_normalization: retired (%r, %r, %r) -> survivor=%r",
                            _ret_info["u"],
                            _ret_info["v"],
                            _ret_pred,
                            _survivor_pred,
                        )
                        group_retired += 1

                _retired_in_so_group.update(_retired_preds)
                if group_retired:
                    total_groups_collapsed += 1
                    total_edges_retired += group_retired

        logger.info(
            "graph_normalization: groups_collapsed=%d edges_retired=%d model_calls=%d",
            total_groups_collapsed,
            total_edges_retired,
            _normalization_diag.get("model_calls", 0),
        )

        _applied = {
            "groups_collapsed": total_groups_collapsed,
            "edges_retired": total_edges_retired,
        }
        _decisions = [
            {"subject": s, "object": o, "clusters": cl} for (s, o), cl in clusters_by_so.items()
        ]
        if self._normalization_sink is not None:
            self._normalization_sink.on_normalization(
                _normalization_diag.get("raw_outputs", []), _decisions, _applied
            )

        return {
            **_applied,
            "chunks": _normalization_diag.get("model_calls", 0),
            "skipped": False,
            "skip_reason": None,
        }

    def refine(self, *, normalize: bool = False, enrich: bool = False) -> RefineResult:
        """Run normalization, then enrichment, over this refiner's merger.

        Normalization runs first (when ``normalize`` is ``True``) so that, when
        both passes run, enrichment operates on the already-normalized graph.

        Args:
            normalize: When ``True``, run :meth:`run_normalization`.
            enrich: When ``True``, run :meth:`run_enrichment`.

        Returns:
            A :class:`RefineResult` carrying each pass's diagnostics dict (or
            ``None`` when not requested) plus a copy of the merger's
            reinforcement maps.
        """
        normalization: "dict | None" = None
        enrichment: "dict | None" = None

        if normalize:
            normalization = self.run_normalization()
            if not normalization.get("skipped"):
                logger.info(
                    "graph_normalization complete: chunks=%d groups_collapsed=%d edges_retired=%d",
                    normalization.get("chunks", 0),
                    normalization.get("groups_collapsed", 0),
                    normalization.get("edges_retired", 0),
                )

        if enrich:
            enrichment = self.run_enrichment()
            if not enrichment.get("skipped"):
                logger.info(
                    "graph_enrichment complete: chunks=%d new_edges=%d same_as_merges=%d",
                    enrichment.get("chunks", 0),
                    enrichment.get("new_edges", 0),
                    enrichment.get("same_as_merges", 0),
                )

        reinforcements: "dict[str, tuple[str, str]]" = dict(
            getattr(self._merger, "reinforcements", {})
        )
        adopt_reinforcements: "dict[str, tuple[str, str]]" = dict(
            getattr(self._merger, "adopt_reinforcements", {})
        )

        return RefineResult(
            normalization=normalization,
            enrichment=enrichment,
            reinforcements=reinforcements,
            adopt_reinforcements=adopt_reinforcements,
        )
