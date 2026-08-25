"""Server consolidation — thin wrapper around ConsolidationLoop.

Uses the same ConsolidationLoop that powers Tests 1-8. The graph is
transient (RAM-only). Promotion is key-level: per-key session counts persisted in
key_metadata.json (no personal data on disk).

The loop saves adapters directly to output_dir (= adapter_dir), so
the router can reload from the standard paths without any bridging.
"""

import enum
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted
from paramem.server.config import ServerConfig
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.stage_ledger import data_state_dir
from paramem.training.thermal_throttle import ThermalPolicy

logger = logging.getLogger(__name__)


class SessionClass(enum.Enum):
    """3-way classification of a pending session's attribution state.

    NAMED         — speaker_id is present and not anonymous_voice.
                    Extract and train immediately.
    HOLDABLE      — speaker is anonymous (anonymous_voice enroll_method), OR
                    the session has no speaker_id but at least one user turn
                    carries a voice embedding (may be attributed later via
                    retro-claim).  Hold pending; retire only past the
                    orphan_retirement TTL.
    UNIDENTIFIABLE — no speaker_id AND no voice embedding anywhere — an
                    unauthenticated text session that can never be attributed.
                    Drop immediately; no TTL.
    """

    NAMED = "named"
    HOLDABLE = "holdable"
    UNIDENTIFIABLE = "unidentifiable"


def classify_session(
    *,
    speaker_id: str | None,
    is_anonymous: bool,
    has_voice_embedding: bool,
) -> SessionClass:
    """Classify a pending session by attribution state.

    Pure function — callers resolve ``is_anonymous`` via
    ``store.is_anonymous(speaker_id)`` before calling; pass ``False`` when
    the speaker store is unavailable.

    Parameters
    ----------
    speaker_id:
        Dominant speaker id from the session (may be ``None``).
    is_anonymous:
        ``True`` iff the speaker profile's ``enroll_method == "anonymous_voice"``.
        Callers obtain this via ``SpeakerStore.is_anonymous(speaker_id)``.
        When the store is ``None``, pass ``False``.
    has_voice_embedding:
        ``True`` iff any user-role turn in the session carries a non-``None``
        ``"embedding"`` field.  Obtained from :meth:`SessionBuffer.pending_facts`.

    Returns
    -------
    SessionClass
        NAMED if the speaker is enrolled and non-anonymous.
        HOLDABLE if anonymous or holds a voice embedding (retro-claimable).
        UNIDENTIFIABLE if no speaker_id and no embedding.
    """
    if speaker_id and not is_anonymous:
        return SessionClass.NAMED
    if is_anonymous or has_voice_embedding:
        return SessionClass.HOLDABLE
    return SessionClass.UNIDENTIFIABLE


def discard_session_sink(config: ServerConfig) -> Path:
    """Return the discard-sink directory for unattributable / retired-holdable sessions.

    Under ``config.debug_dir/discarded_sessions/``.  Distinct from
    :func:`session_retention_dir` (trained-session archive) so the trained
    archive is never polluted with dropped sessions.

    Under ``debug=False`` (privacy mode), :meth:`SessionBuffer.mark_consolidated`
    unlinks the JSONL unconditionally regardless of ``retention_dir`` — this
    sink is still passed so the code path is identical in both modes.
    """
    return config.debug_dir / "discarded_sessions"


def create_consolidation_loop(
    model,
    tokenizer,
    config: ServerConfig,
    memory_store,
    state_provider=None,
    *,
    output_dir=None,
    save_cycle_snapshots: bool | None = None,
    seed_state_from_disk: bool = True,
    keep_prior_slots: int | None = None,
) -> ConsolidationLoop:
    """Create a ConsolidationLoop configured for the server.

    Graph is transient (RAM-only). Key metadata is seeded
    from key_metadata.json to restore cycle count, promoted keys, and
    per-key bookkeeping (reinforcement_count, last_reinforced_cycle, last_seen,
    first_seen) across restarts.

    Parameters
    ----------
    state_provider:
        Optional zero-argument callable returning the server ``_state`` dict.
        Used only to wire the base-model weight-hash cache from server
        ``_state`` into the loop (below); experiment scripts that do not pass
        ``state_provider`` are unaffected (default ``None`` → no cache wired).
    output_dir:
        Override ``config.adapter_dir`` as the loop's output directory.
        ``None`` (default) falls through to ``config.adapter_dir``, which
        preserves production behaviour.
    save_cycle_snapshots:
        Override ``config.debug`` as the cycle-snapshot toggle.  ``None``
        (default) falls through to ``config.debug``.
    seed_state_from_disk:
        When ``False``, skip seeding key metadata and keyed-pairs QA from
        disk.  Use for probe/experiment runs that must start from a clean
        state rather than inheriting live-system data.  Default ``True``
        preserves production behaviour.
    keep_prior_slots:
        Override ``config.consolidation.training_keep_prior_slots``.
        ``None`` (default) falls through to the config value.
    """
    _output_dir = output_dir if output_dir is not None else config.adapter_dir
    _save_cycle_snapshots = (
        save_cycle_snapshots if save_cycle_snapshots is not None else config.debug
    )

    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=config.consolidation_config,
        training_config=config.training_config,
        memory_store=memory_store,
        tier_adapters=config.tier_config_map(),
        output_dir=_output_dir,
        extraction_temperature=0.0,
        extraction_max_tokens=config.consolidation.extraction_max_tokens,
        extraction_plausibility_max_tokens=config.consolidation.extraction_plausibility_max_tokens,
        extraction_anonymize_token_envelope=(
            config.consolidation.extraction_anonymize_token_envelope
        ),
        save_cycle_snapshots=_save_cycle_snapshots,
        snapshot_dir=config.debug_dir if _save_cycle_snapshots else None,
        prompts_dir=config.prompts_dir,
        model_name=config.model_name,
        graph_config=config.graph_config,
        # The ONE cloud master switch (ServerConfig.cloud.enabled) — same
        # switch the conversation agent and /calibrate/enrich read.
        cloud_enabled=config.cloud.enabled,
        graph_enrichment_neighborhood_hops=config.consolidation.graph_enrichment_neighborhood_hops,
        graph_enrichment_max_entities_per_pass=config.consolidation.graph_enrichment_max_entities_per_pass,
        # Same scrub categories as inference-time cloud egress: the cloud
        # enrichment cycle sends placeholders to the cloud just like the
        # cloud_anonymizer egress path, so the privacy policy must match.
        # ``config.sanitization.scrub_categories`` is resolved once at
        # config construction from the operator's ``scrub`` hints — the
        # span tagger's configured labels are the sole scope authority.
        # An empty tuple disables anonymization entirely (the operator's
        # opt-out): no tagger call, no anonymizer call, content egresses
        # verbatim.
        extraction_scrub_categories=config.sanitization.scrub_categories,
        extraction_correction_entity_types=set(
            config.consolidation.extraction_correction_entity_types
        ),
        extraction_enrichment_provider=config.consolidation.extraction_enrichment_provider,
        extraction_enrichment_provider_model=config.consolidation.extraction_enrichment_provider_model,
        extraction_enrichment_provider_endpoint=config.consolidation.extraction_enrichment_provider_endpoint,
        extraction_plausibility_judge=config.consolidation.extraction_plausibility_judge,
        extraction_plausibility_stage=config.consolidation.extraction_plausibility_stage,
        extraction_plausibility_model=config.consolidation.extraction_plausibility_model,
        extraction_plausibility_endpoint=(
            config.consolidation.extraction_plausibility_endpoint or None
        ),
        # Thermal fields live on ConsolidationScheduleConfig
        # (config.consolidation), NOT on ConsolidationConfig (which the loop
        # accepts as consolidation_config).  Build the policy here where both
        # configs are reachable, pass it to the loop precomputed.
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
        keep_prior_slots=(
            keep_prior_slots
            if keep_prior_slots is not None
            else config.consolidation.training_keep_prior_slots
        ),
        telemetry_dir=config.telemetry_dir,
        incidents_state_dir=data_state_dir(config.paths.data),
    )

    # Wire the base-model weight-hash cache from server _state into the loop so
    # build_manifest_for memoizes the SHA-256 across consolidations within one
    # process lifetime. Without this, every cycle re-hashes the full base model
    # (~2 min for Mistral 7B). Cache is keyed by id(model); resets on restart.
    if state_provider is not None:
        state = state_provider()
        if state is not None:
            loop.fingerprint_cache = state.setdefault("base_model_hash_cache", {})

    # Wire the full-consolidation period string so commit_tier_slot can stamp
    # main slots with the current full-cycle window.  The stamp is manifest
    # PROVENANCE only — nothing reads it back and no gate compares stamps
    # (_is_full_cycle_due counts payload-bearing interim slots instead).  An
    # empty string in experiment paths (state_provider=None) therefore only
    # means "window unknown" on those slots; it changes no scheduling decision.
    loop.full_consolidation_period_string = config.consolidation.consolidation_period_string

    if seed_state_from_disk:
        # Key metadata (cycle counts, promotion bookkeeping) is loop-state
        # and still seeded here.  Entry payloads (subject/predicate/object/
        # speaker_id) live in the lifespan-owned MemoryStore — preload runs
        # at lifespan boot, not here, so the loop factory no longer touches
        # the model or reads graph.json for that purpose.  ``memory_store``
        # must already have its registries AND bookkeeping loaded (the
        # ordinary boot sequence loads both before this loop is
        # constructed) so ``seed_key_metadata`` can rebuild ``promoted_keys``
        # from the per-key flags.
        cycle_count = load_max_tier_cycle(config.adapter_dir)
        if cycle_count is not None:
            loop.seed_key_metadata(cycle_count)

    return loop


@dataclass(frozen=True)
class PendingTriage:
    """Classification of every pending session, before any retirement.

    The pure output of :func:`classify_pending_sessions` — the arbitrator's
    own retirement side effect (:func:`retire_unattributable_sessions`)
    consumes ``drop_ids`` separately, so classifying is same-arguments-
    same-answer with no side effects.

    Attributes
    ----------
    pending_count:
        Pending sessions seen, before any retirement.
    named_count:
        How many of those classified NAMED (attributable).
    drop_ids:
        Session ids classified UNIDENTIFIABLE, plus HOLDABLE sessions past
        ``orphan_retirement_seconds`` — everything :func:`retire_unattributable_sessions`
        should retire.
    """

    pending_count: int
    named_count: int
    drop_ids: "list[str]"


def classify_pending_sessions(config: ServerConfig, buffer, store) -> PendingTriage:
    """Classify every pending session.  Reads; mutates nothing.

    The arbitrator's unconditional pre-stage runs this on every dispatch
    (whichever action asked), then feeds ``drop_ids`` to
    :func:`retire_unattributable_sessions` and ``pending_count``/``named_count``
    to the content gate — retiring what can never be attributed no longer
    depends on whether the content gate itself runs.

    Parameters
    ----------
    config:
        Live server config.
    buffer:
        The ``SessionBuffer``.
    store:
        The ``SpeakerStore`` (or ``None`` when no speakers are enrolled).

    Returns
    -------
    PendingTriage
    """
    facts = buffer.pending_facts()
    if not facts:
        return PendingTriage(pending_count=0, named_count=0, drop_ids=[])

    ttl_seconds = config.consolidation.orphan_retirement_seconds
    drop_ids: list[str] = []
    named_count = 0

    for fact in facts:
        sid = fact["speaker_id"]
        is_anon = store.is_anonymous(sid) if store is not None and sid else False
        cls = classify_session(
            speaker_id=sid,
            is_anonymous=is_anon,
            has_voice_embedding=fact["has_voice_embedding"],
        )
        if cls == SessionClass.NAMED:
            named_count += 1
        elif cls == SessionClass.UNIDENTIFIABLE:
            drop_ids.append(fact["session_id"])
        else:
            # HOLDABLE — retire only when TTL set and exceeded.
            if ttl_seconds is not None:
                age = fact.get("age_seconds")
                if age is not None and age > ttl_seconds:
                    drop_ids.append(fact["session_id"])

    return PendingTriage(pending_count=len(facts), named_count=named_count, drop_ids=drop_ids)


def retire_unattributable_sessions(config: ServerConfig, buffer, drop_ids: "list[str]") -> None:
    """Retire sessions that can never be attributed, or expired holdables.

    A staging-action-only primitive: called by the arbitrator's retiring
    triage pre-stage after :func:`classify_pending_sessions`, never by a
    non-staging (calibration) run.  A no-op on an empty *drop_ids*.

    Parameters
    ----------
    config:
        Live server config.
    buffer:
        The ``SessionBuffer``.
    drop_ids:
        Session ids to retire — :attr:`PendingTriage.drop_ids`.
    """
    if not drop_ids:
        return
    logger.info(
        "Consolidation dispatch: retiring %d unattributable/expired-holdable session(s)",
        len(drop_ids),
    )
    _retain = config.consolidation.retain_sessions or config.debug
    buffer.mark_consolidated(
        drop_ids,
        retention_dir=discard_session_sink(config) if _retain else None,
    )


def get_or_create_consolidation_loop(state: dict, *, store=None) -> ConsolidationLoop:
    """Return the process-lifetime ``ConsolidationLoop``, creating it on first use.

    THE single get-or-create for the whole tree — every module that needs
    the loop (``paramem.server.app``, ``paramem.server.calibrate``) calls
    this one function.  Idempotent: a second call finds the loop already
    on ``state`` and returns it unchanged — *store* is then a no-op, since
    only a first-time construction reads it.

    Parameters
    ----------
    state:
        The live server state dict.  ``config`` is always
        ``state["config"]`` — not a separate parameter, since every caller
        has exactly one live config to build against.
    store:
        Optional store override used ONLY when a fresh loop is being
        constructed (``state["consolidation_loop"] is None``).  ``None``
        (default, every caller except the pending-event resume) falls
        through to ``state["memory_store"]``.  The pending-event resume
        passes a locally-constructed empty :class:`~paramem.memory.store.MemoryStore`
        here when ``state["memory_store"] is None`` (the store is
        quarantined), so the resumed event's loop has something to fold
        into without waiting on a lift.

    Returns
    -------
    ConsolidationLoop
        The singleton stored on ``state["consolidation_loop"]``.
    """
    loop = state.get("consolidation_loop")
    if loop is not None:
        return loop
    config = state["config"]
    loop = create_consolidation_loop(
        state["model"],
        state["tokenizer"],
        config,
        store if store is not None else state["memory_store"],
        state_provider=lambda: state,
    )
    state["consolidation_loop"] = loop
    return loop


def session_retention_dir(loop, config) -> Path | None:
    """Return the directory to retain consolidated session JSONL into.

    Returns ``None`` when neither retention nor debug mode is enabled —
    the SessionBuffer will unlink the JSONL after consume.  Otherwise
    returns ``loop.snapshot_dir_for()/sessions/`` — always the un-stamped
    per-cycle root (``paths.debug/episodic/cycle_<N>/run_<run_id>/sessions/``).
    An interim event's own debug snapshots live under that cycle's stamped
    ``interim_<stamp>/`` root, but retention deliberately does not follow
    it, so the retention location stays stable across cycle kinds.

    Falls back to ``config.debug_dir/cycle_<N>/sessions/`` when ``debug`` is
    off and only ``retain_sessions`` asked for this: ``snapshot_dir_for``
    returns ``None`` there, but the transcripts were still requested.
    """
    if not (config.consolidation.retain_sessions or config.debug):
        return None
    # Deliberately no interim stamp here: retention always lands under the
    # stable per-cycle root regardless of which event kind (full or interim)
    # consumed the session, so retained transcripts have one predictable
    # location rather than following the consuming event's interim scope.
    snap = loop.snapshot_dir_for()
    if snap is None:
        return config.debug_dir / f"cycle_{loop.cycle_count}" / "sessions"
    return snap / "sessions"


# run_consolidation was deleted when it was merged into paramem.server.app._run_extraction_phase,
# which closes over _state instead of taking model/tokenizer/config/session_buffer as args.
# Trial-migration callers: use _run_extraction_phase(loop, mark_sessions=False).
# Dev scripts that imported run_consolidation directly will need updating separately.


# --- Key-level promotion ---


# Dedup moved onto ConsolidationLoop (see paramem/training/consolidation.py).
# Re-exported as module-level aliases for existing call sites.
_dedup_episodic = ConsolidationLoop.dedup_episodic
_dedup_procedural = ConsolidationLoop.dedup_procedural


# --- Persistence ---


def load_max_tier_cycle(adapter_dir: Path) -> int | None:
    """Return the maximum ``tier_cycle`` recorded across every tier's
    on-disk ``key_metadata.json`` under *adapter_dir*.

    THE per-tier boot-seed reader for the loop's cycle counter: walks
    :func:`~paramem.memory.interim_adapter.iter_tier_roots` (main tiers,
    then interim slots) and reads ``<tier_root>/key_metadata.json`` where
    present, taking each file's ``tier_cycle`` field. Per-key bookkeeping
    rows are not read here, and no cross-tier ownership conflict rule is
    applied — that is
    :meth:`~paramem.memory.store.MemoryStore.load_bookkeeping_from_disk`'s
    job, the one canonical implementation of that conflict rule.

    Returns ``None`` when no tier has a ``key_metadata.json`` file at all
    (fresh install); callers only invoke
    :meth:`~paramem.training.consolidation.ConsolidationLoop.seed_key_metadata`
    when this is not ``None``.

    Callers: the loop-construction boot seed
    (:func:`create_consolidation_loop`) and the base-swap migration
    carry-over (``paramem.server.app``, Phase B).
    """
    from paramem.memory.interim_adapter import iter_tier_roots

    adapter_dir = Path(adapter_dir)
    max_cycle = 0
    found_any = False
    for _tier_name, tier_root in iter_tier_roots(adapter_dir):
        path = tier_root / "key_metadata.json"
        if not path.exists():
            continue
        found_any = True
        raw = json.loads(read_maybe_encrypted(path).decode("utf-8"))
        max_cycle = max(max_cycle, raw.get("tier_cycle", 0))

    if not found_any:
        logger.info("No key metadata found under %s, starting fresh", adapter_dir)
        return None
    return max_cycle
