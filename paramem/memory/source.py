"""Source-of-truth abstraction for the indexed-key memory layer.

A :class:`MemorySource` retrieves indexed-key entries from the authoritative
medium for the current consolidation mode.  Two implementations exist:

* :class:`WeightMemorySource` — train mode.  Probes adapter weights via the
  recall template and reconstructs the entry from generated output.
* :class:`DiskMemorySource` — simulate mode.  Reads encrypted per-tier
  ``graph.json`` files and decodes the entry directly.

Both implementations return the same canonical entry shape so callers are
mode-agnostic.  :func:`build_memory_source` is the ONE construction site —
callers name the mode, never the class.  Three real callers keep it: the
boot cache fill (``app._build_store_contents``), the fold's own recall
(``ConsolidationLoop._hydrate_store_for_fold``, which probes the built
source directly into fold-local state), and the per-turn live door
(``paramem.server.inference._probe_and_reason``, which builds the source
and passes it to
:meth:`~paramem.memory.store.MemoryStore.probe_source`, reached only under
``inference.preload_cache=False``).  The active-store migration constructs
no source of its own.

The source is **not** the cache — :class:`paramem.memory.store.MemoryStore`
is.  Serving reads one of two intent-named doors, selected once at the
serving boundary by ``inference.preload_cache``: the cache door
(:meth:`~paramem.memory.store.MemoryStore.probe_cache`) is a plain lookup
against the RAM mirror; the live door
(:meth:`~paramem.memory.store.MemoryStore.probe_source`) probes a source
built here, in one grouped call, with no cache contact in either
direction.  The mirror is a non-authoritative shortcut with exactly two
writers (the boot fill, go-live adoption) — nothing validates its
completeness and it never gates a turn.

Naming
------
``entry`` is the shape-agnostic term for "one keyed record" and is
content-only: ``{key, subject, predicate, object}`` plus a source's own
derived fields (``fact_text``, ``raw_output``, and a real SimHash-verified
``confidence`` — both implementations gate their own results against the
registry :func:`build_memory_source` supplies, so no venue is exempt).  A
hit that fails the gate returns the failure shape
:func:`~paramem.memory.entry.finalize_recalled` uses
(``{"raw_output", "failure_reason": "low_confidence:..."}``) instead of the
content dict.  No source emits ``speaker_id`` or a fabricated confidence.
Speaker attribution lives exclusively in
:attr:`~paramem.memory.store.MemoryStore._bookkeeping`, written by
consolidation at fold time (never derived from a probe result), and is read
back by :func:`~paramem.memory.persistence.build_tier_graph_from_store` when
re-persisting a tier graph.  The store-boundary SimHash gate in
:meth:`~paramem.memory.store.MemoryStore.probe_source` still runs on every
live-door result on top of the source's own gate (harmless — both compute
the identical confidence from the same registry entry); the cache door
performs no fingerprint work of its own — its content was gated once at
admission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from paramem.memory.entry import DEFAULT_CONFIDENCE_THRESHOLD


@runtime_checkable
class MemorySource(Protocol):
    """Read-side contract for the indexed-key source of truth.

    Implementations resolve a batch of (adapter, key) pairs into a flat
    ``{key → entry-or-None}`` mapping.  Adapter ordering in
    *keys_by_adapter* is preserved through the result so callers can rely
    on the router's preferred probe order (procedural → episodic →
    semantic → newest-interim) reaching the model in that order.

    Each hit carries the canonical entry fields documented in
    :func:`paramem.memory.probe.probe_keys_grouped_by_adapter`.
    Misses (unknown key, decoding failure, missing adapter) map to ``None``.
    """

    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
    ) -> dict[str, dict | None]:  # pragma: no cover — Protocol
        ...


class WeightMemorySource:
    """Train-mode source.  Materialises entries by probing adapter weights.

    Wraps :func:`probe_keys_grouped_by_adapter`.  The wrapped function does
    one ``switch_adapter`` per group and ``batch_size`` keys per
    ``model.generate`` call, so the per-call cost scales linearly with the
    total key count divided by ``batch_size``.

    The model, tokenizer, and per-adapter format mapping are captured at
    construction so callers don't thread them through every call.  When the
    set of mounted adapters or their formats changes (e.g. after a
    consolidation cycle finalize) the lifespan rebuilds the source.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        registry: dict[str, int] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_new_tokens: int = 200,
        batch_size: int,
    ) -> None:
        """Initialise the weight-based memory source.

        Args:
            model: PeftModel already loaded in memory.
            tokenizer: Tokenizer matching the model.
            registry: Optional SimHash registry for confidence verification.
            confidence_threshold: Minimum confidence to accept a recalled entry.
            max_new_tokens: Maximum tokens to generate per probe.
            batch_size: Number of keys per ``model.generate`` call.  MUST be
                supplied from ``config.consolidation.recall_probe_batch_size``
                — no default is provided so callers cannot silently fall back
                to single-key generation.
        """
        # BASE-MODEL HOLDER (WeightMemorySource): every construction goes
        # through build_memory_source, and every caller of that keeps the result
        # as a frame-local it drops before returning —
        # _release_base_model_in_process cannot reach a caller-frame local.
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.confidence_threshold = confidence_threshold
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
    ) -> dict[str, dict | None]:
        """Probe adapter weights for the given keys.

        Args:
            keys_by_adapter: Ordered mapping of adapter name → list of keys.

        Returns:
            Flat ``{key → result | None}`` mapping.
        """
        # Lazy import so test monkeypatches against
        # ``paramem.memory.probe.probe_keys_grouped_by_adapter``
        # take effect without re-binding through this module.
        from paramem.memory.probe import probe_keys_grouped_by_adapter

        return probe_keys_grouped_by_adapter(
            self.model,
            self.tokenizer,
            keys_by_adapter,
            max_new_tokens=self.max_new_tokens,
            registry=self.registry,
            confidence_threshold=self.confidence_threshold,
            batch_size=self.batch_size,
        )


class DiskMemorySource:
    """Simulate-mode source.  Materialises entries by reading per-tier graph.json.

    No model interaction, no GPU, no switch_adapter — pure disk read +
    JSON decode.  Per-call cost scales with the per-tier graph size (a
    few hundred edges typically).

    Path resolution uses :func:`paramem.memory.interim_adapter.adapter_slot_root_for_name`
    to resolve each key's TIER ROOT — both main tiers (``"episodic"``,
    ``"semantic"``, ``"procedural"`` — flat under ``<store_dir>/<tier>/``)
    and interim adapters (``"episodic_interim_<stamp>"`` — nested under
    ``<store_dir>/episodic/interim_<stamp>/``) resolve correctly — then binds
    the tier's LIVE slot the same way the train venue does:
    :func:`~paramem.adapters.manifest.find_live_slot` against
    :func:`~paramem.adapters.manifest.tier_registry_sha256`. ``graph.json``
    lives inside that bound slot (``<tier_root>/<ts>/graph.json``, written by
    :func:`~paramem.memory.persistence.commit_tier_slot` /
    :func:`~paramem.memory.persistence.write_tier_slot` through the shared
    slot envelope), never directly at the tier root — there is no tier-root
    fallback. A tier with no bound slot yields the ordinary per-key miss
    shape (``None``) for every key requested of it, never a silently empty
    graph.

    *store_dir* is the adapter root (``config.adapter_dir``) — the same
    directory the bound slot is resolved under.

    Gated exactly like :class:`WeightMemorySource`: a disk read is not
    inherently trustworthy — a corrupted or hand-edited ``graph.json`` would
    otherwise flow ungated into every caller (the fold's reconstruction, the
    boot fill), get retrained, and have its corrupted content re-fingerprinted
    as "verified" on the next mint.  ``registry`` is optional (``None`` passes
    every result through unverified, matching :class:`WeightMemorySource`'s
    own no-registry contract) so existing callers that never supplied one
    keep working; :func:`build_memory_source` always supplies one.
    """

    def __init__(
        self,
        store_dir: Path,
        *,
        registry: dict[str, int] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.registry = registry
        self.confidence_threshold = confidence_threshold

    def probe(
        self,
        keys_by_adapter: dict[str, list[str]],
    ) -> dict[str, dict | None]:
        """Read entries from each tier's bound ``graph.json`` slot, SimHash-gated.

        Args:
            keys_by_adapter: Ordered mapping of adapter name → list of keys.

        Returns:
            Flat ``{key → result | None}`` mapping.  A tier with no bound
            slot maps every one of its keys to ``None`` — the same miss
            shape an unknown key produces, never a silently empty graph.  A
            hit below :attr:`confidence_threshold` against :attr:`registry`
            returns the same failure shape
            :func:`~paramem.memory.entry.finalize_recalled` uses
            (``{"raw_output", "failure_reason": "low_confidence:..."}``)
            rather than the content dict — every caller's existing miss
            predicate (``"failure_reason" in result``) already treats this as
            a miss, unchanged.
        """
        import json

        from paramem.adapters.manifest import find_live_slot, tier_registry_sha256
        from paramem.adapters.slot import payload_filename
        from paramem.memory.entry import entry_fact_text, verify_confidence
        from paramem.memory.interim_adapter import adapter_slot_root_for_name
        from paramem.memory.persistence import (
            entry_by_key,
            load_memory_from_disk,
        )

        results: dict[str, dict | None] = {}
        for adapter_name, keys in keys_by_adapter.items():
            if not keys:
                continue
            tier_root = adapter_slot_root_for_name(self.store_dir, adapter_name)
            bound_slot = find_live_slot(tier_root, tier_registry_sha256(tier_root))
            if bound_slot is None:
                for key in keys:
                    results[key] = None
                continue
            graph = load_memory_from_disk(bound_slot / payload_filename("simulate"))
            for key in keys:
                entry = entry_by_key(graph, key)
                if entry is None:
                    results[key] = None
                    continue
                base = {
                    "key": key,
                    "subject": entry.get("subject", ""),
                    "predicate": entry.get("predicate", ""),
                    "object": entry.get("object", ""),
                }
                raw_output = json.dumps(base)
                confidence = verify_confidence(base, self.registry)
                if confidence < self.confidence_threshold:
                    results[key] = {
                        "raw_output": raw_output,
                        "failure_reason": f"low_confidence:{confidence:.3f}",
                    }
                    continue
                results[key] = {
                    **base,
                    "confidence": confidence,
                    "fact_text": entry_fact_text(entry),
                    "raw_output": raw_output,
                }
        return results


def train_venue_deferred(mode: str, model) -> bool:
    """True when *mode* is the train venue and no model is resident.

    THE exported deferral predicate: :func:`build_memory_source` raises
    rather than returning ``None`` for this exact case, so every caller that
    might hit it checks this predicate BEFORE constructing, not after a
    raise.  A simulate-venue caller never defers — :class:`DiskMemorySource`
    needs no model — so this predicate is ``False`` whenever ``mode !=
    "train"`` regardless of *model*.  Every valid ParaMem configuration
    names a base model; what varies at runtime is residency (a temporal,
    cloud-only deferral), never configuration — this predicate names exactly
    that non-resident window for the train venue.

    Args:
        mode: ``"train"`` or ``"simulate"`` — the same venue vocabulary
            :func:`build_memory_source` validates.
        model: The candidate base model handle, or ``None`` when not
            resident.

    Returns:
        ``True`` when the fill act this predicate guards must be deferred
        to the next act with a model; ``False`` otherwise.
    """
    return mode == "train" and model is None


def build_memory_source(
    *,
    mode: "Literal['train', 'simulate']",
    adapter_dir: "Path | str",
    batch_size: int,
    model=None,
    tokenizer=None,
    cached_registry: bool = False,
) -> "MemorySource":
    """Construct the :class:`MemorySource` for *mode* — the ONE construction site.

    Every path that needs a source goes through here: boot / post-fold store
    hydration (``app._build_store_contents``), the per-query live door
    (``inference._probe_and_reason``), and the per-fold recall
    (``ConsolidationLoop._hydrate_store_for_fold``).  The mode → class mapping
    exists exactly once, which is why this is the only function in
    ``paramem/memory/`` on the mode-fork allowlist.

    Validates *mode* against the closed venue vocabulary (``train`` |
    ``simulate``) and requires a live *model* + *tokenizer* in train mode —
    RAISES on either violation and never returns ``None``.  A caller that
    might call this with ``mode="train"`` and no model resident checks
    :func:`train_venue_deferred` BEFORE calling, and skips the whole fill
    act instead of reaching this raise: a simulate boot fills from disk
    regardless of model residency, and a train-venue boot with no model
    resident defers the fill to the next act with a model.

    **BASE-MODEL HOLDER** — a returned :class:`WeightMemorySource` captures
    *model*.  The caller owns the lifetime: keep it as a frame-local and drop it
    before returning, never on ``self`` (see the invariant header on
    ``app._release_base_model_in_process``).

    Args:
        mode: Consolidation persistence mode.  ``"simulate"`` → graph.json on
            disk; ``"train"`` → adapter weights.  Production sources:
            ``config.consolidation.mode`` (server sites) and
            ``ConsolidationLoop._venue_from_scope(scope)`` (fold site).
        adapter_dir: Adapter root.  ``config.adapter_dir`` on the server sites,
            ``ConsolidationLoop.output_dir`` in the fold — the same directory
            the per-tier ``graph.json`` and ``indexed_key_registry.json`` files
            are written into.
        batch_size: Keys per ``model.generate`` call for the weight probe.
            Production source: ``config.consolidation.recall_probe_batch_size``
            (server sites) / ``TrainingConfig.recall_probe_batch_size`` (fold),
            which ``ServerConfig`` derives from the same field.  Required even
            in simulate mode so the signature does not fork.
        model: Loaded ``PeftModel``.  Train mode only; must be non-``None`` —
            callers check :func:`train_venue_deferred` first rather than
            relying on a raise here.
        tokenizer: Tokenizer matching *model*.  Train mode only; must be
            non-``None`` in train mode — same raise treatment as *model*.
        cached_registry: Forwarded to
            :meth:`~paramem.memory.store.MemoryStore.read_simhash_registry_from_disk`
            as its ``cached`` keyword — read for EITHER venue now, since
            both :class:`WeightMemorySource` and :class:`DiskMemorySource`
            gate their own results against it.  Default ``False`` re-reads
            every tier registry from disk on every call — the correct
            choice for hydration callers, which run before
            :meth:`~paramem.server.router.QueryRouter.reload` and must see
            disk truth.  Only the per-turn inference probe
            (``inference._probe_and_reason``) opts in.

    Returns:
        A :class:`DiskMemorySource` in simulate mode; a
        :class:`WeightMemorySource` in train mode.  Never ``None``.

    Raises:
        ValueError: *mode* is outside the closed venue vocabulary, or *mode*
            is ``"train"`` and *model* or *tokenizer* is ``None``.
    """
    if mode not in ("train", "simulate"):
        raise ValueError(f"build_memory_source: unknown venue {mode!r} — must be train|simulate")

    # SimHash registry is DERIVED from adapter_dir, never passed in, for
    # EITHER venue: it gates recalled entries at the one place output leaves
    # a source, before they cross into any caller — the fold's
    # reconstruction, the boot fill, or the live door's turn.
    from paramem.memory.store import MemoryStore

    registry = MemoryStore.read_simhash_registry_from_disk(adapter_dir, cached=cached_registry)

    if mode == "simulate":
        return DiskMemorySource(adapter_dir, registry=registry)
    if model is None:
        raise ValueError(
            "build_memory_source: mode='train' requires a live model — caller must check "
            "train_venue_deferred(mode, model) before constructing"
        )
    if tokenizer is None:
        raise ValueError("build_memory_source: mode='train' requires a live tokenizer")

    return WeightMemorySource(
        model,
        tokenizer,
        registry=registry,
        batch_size=batch_size,
    )
