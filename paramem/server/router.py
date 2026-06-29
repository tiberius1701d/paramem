"""Query routing — speaker-only key resolution + intent-driven tier selection.

The router answers two questions per query:

* **Which keys can this speaker probe?**  Answer: the speaker's entry in
  :attr:`QueryRouter._speaker_key_index` — the privacy boundary.  Speaker
  enrollment populates ``allowed_keys``; nothing more.
* **Which tiers should we deliver context from?**  Answer: derived from
  the intent classifier on the query content.  See
  :meth:`QueryRouter._steps_for_intent`.

Intent is decided by :func:`paramem.server.intent.classify_intent` — HA
lexical fast-path + encoder-cosine residual + fail-closed default.
Speaker enrollment is intentionally NOT a routing signal: a query from
an enrolled speaker is classified by content, not by who said it.  This
removes the "speaker-in-graph → PERSONAL" short-circuit that previously
caused imperatives from enrolled speakers to misroute (HR3 bug).

The router is stateless per query — all state lives in the registry-
backed ``_speaker_key_index`` (preload-independent) and the optional
HA entity graph.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.server.config import IntentConfig
    from paramem.server.ha_graph import HAEntityGraph, HAMatchResult

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Routing intent — single axis the chat handler dispatches on.

    * ``PERSONAL`` — speaker's life / personal facts; never escalates to cloud.
    * ``COMMAND``  — imperative directed at the home / device fleet; routed
      to the HA conversation agent.
    * ``GENERAL``  — general-knowledge query; full escalation chain available.
    * ``UNKNOWN``  — classifier unavailable or below confidence margin;
      treated per :class:`IntentConfig.fail_closed_intent`
      (privacy-preserving default).
    """

    PERSONAL = "personal"
    COMMAND = "command"
    GENERAL = "general"
    UNKNOWN = "unknown"


@dataclass
class RoutingStep:
    """One step in a routing plan: activate an adapter and probe keys."""

    adapter_name: str
    keys_to_probe: list[str] = field(default_factory=list)


@dataclass
class RoutingPlan:
    """What to activate and probe for a given query.

    * ``intent`` — single routing axis ``inference.py`` dispatches on.
    * ``steps`` — per-tier :class:`RoutingStep` with speaker-scoped keys
      for the tiers selected by intent (see
      :meth:`QueryRouter._steps_for_intent`).
    * ``ha_domains`` — HA-graph observability metadata.
    * ``strategy`` — ``"targeted_probe"`` when ``steps`` is non-empty,
      ``"direct"`` otherwise.
    """

    steps: list[RoutingStep] = field(default_factory=list)
    strategy: str = "direct"
    ha_domains: list[str] = field(default_factory=list)
    intent: Intent = Intent.UNKNOWN


# Tier ordering for PERSONAL probe:
# 1. procedural first — preferences shape style (load-bearing rule: preferences
#    must be loaded before facts so style context is already active).
# 2. interim adapters next, newest-first — they hold the freshest factual
#    state.  A user correction (move, rename, change of mind) lands in the
#    newest ``episodic_interim_<stamp>`` slot ahead of the next full-cycle
#    merge into main; probing interim before main ensures the latest answer
#    wins instead of returning the stale pre-merge baseline.
# 3. main episodic — baseline factual snapshot from the last full cycle.
# 4. semantic — durable but most lossy/abstract; corroboration fallback.
_PERSONAL_TIERS_PRE_INTERIM = ("procedural",)
_PERSONAL_TIERS_POST_INTERIM = ("episodic", "semantic")

_INTERIM_DATE_RE = re.compile(r"^episodic_interim_(\d{8}T\d{4})$")


def _interim_sort_key(adapter_name: str) -> str | None:
    """Extract the YYYYMMDDTHHMM stamp from an ``episodic_interim_<stamp>``
    name, or ``None`` for non-interim names.  Used to sort interim tiers
    newest-first for the PERSONAL probe order."""
    m = _INTERIM_DATE_RE.match(adapter_name)
    return m.group(1) if m else None


# ``_is_interrogative`` + ``_INTERROGATIVE_*`` are NOT consumed by routing
# any more — intent classification is the routing signal.  They remain in
# this module solely to support the abstention gate in
# ``paramem/server/inference.py`` (currently imports from here).  Future
# cleanup: move to ``paramem/server/sentence_type.py`` where the encoder-
# tier classifier already lives.

_INTERROGATIVE_PREFIXES = frozenset(
    {
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "is",
        "are",
        "does",
        "do",
        "can",
        "could",
        "would",
        "will",
        "which",
    }
)

# Terminal punctuation that marks a query as interrogative regardless of
# leading word.  Covers the question-mark glyphs ParaMem-supported scripts
# actually use:
#   ``?``   — ASCII / Latin / most European scripts (U+003F).
#   ``？``  — fullwidth, used in CJK rendering (U+FF1F).
#   ``؟``  — Arabic / Persian / Urdu (U+061F).
# Greek question mark (U+003B, ``;``) is intentionally excluded because it
# is glyphically identical to the ASCII semicolon and would mis-classify
# declarative sentences ending in ``;``.
_INTERROGATIVE_PUNCT = frozenset({"?", "？", "؟"})


def _is_interrogative(text: str, config=None) -> bool:
    """Check if the query is a question (not an imperative command).

    Three-tier detection: encoder-based classifier (when ``config`` is a
    :class:`paramem.server.config.SentenceTypeConfig` and the encoder +
    exemplar bank are loaded), then terminal-punctuation, then English
    first-word lexicon.  Each tier produces a definitive answer when it
    can, otherwise falls through.

    Consumed only by the abstention gate in
    :mod:`paramem.server.inference`; routing itself does not call this.
    """
    stripped = text.strip()
    if not stripped:
        return False

    if config is not None:
        from paramem.server.sentence_type import (
            SentenceType,
            classify_sentence_type,
        )

        verdict = classify_sentence_type(stripped, config=config)
        if verdict is SentenceType.INTERROGATIVE:
            return True
        if verdict is SentenceType.NON_INTERROGATIVE:
            return False

    if stripped[-1] in _INTERROGATIVE_PUNCT:
        return True
    first_word = stripped.split()[0].lower()
    if first_word.endswith("'s"):
        first_word = first_word[:-2]
    return first_word in _INTERROGATIVE_PREFIXES


class QueryRouter:
    """Routes queries: identifies speaker scope and selects tiers by intent.

    State: a single per-speaker ``speaker_id → set[key]`` index built from
    the injected :class:`paramem.memory.store.MemoryStore`.  The index is
    preload-independent — it iterates
    :meth:`MemoryStore.iter_bookkeeping` which is populated by
    :meth:`MemoryStore.load_bookkeeping_from_disk` at boot regardless of
    ``inference.preload_cache``.  ``_entries`` (content cache) is never
    consulted for index building.

    Rebuilt after each consolidation cycle via :meth:`reload`.
    """

    def __init__(
        self,
        adapter_dir: Path,
        memory_store,
        ha_graph: "HAEntityGraph | None" = None,
        intent_config: "IntentConfig | None" = None,
    ):
        self.adapter_dir = Path(adapter_dir)
        self._ha_graph = ha_graph
        self._intent_config = intent_config
        self._memory_store = memory_store
        # speaker_id -> set[key]; flat across tiers.  Built by reload().
        self._speaker_key_index: dict[str, set[str]] = {}

        self.reload()

    def reload(self) -> None:
        """Rebuild ``_speaker_key_index`` from the injected :class:`MemoryStore`.

        Iterates :meth:`MemoryStore.iter_bookkeeping` — the per-key provenance
        map populated by :meth:`MemoryStore.load_bookkeeping_from_disk` at boot
        regardless of ``inference.preload_cache``.  The content cache
        (``_entries``) is NOT consulted: under ``preload_cache=False`` entries
        are intentionally empty, but bookkeeping is always present.  The old
        ``len(store) > 0`` guard (which counted ``_entries`` and
        short-circuited on empty cache) has been removed — the correct gate is
        ``store is not None`` only.

        Call after every consolidation cycle so the index reflects the
        current in-memory state.
        """
        self._speaker_key_index.clear()

        store = self._memory_store
        if store is not None:
            for key, bk in store.iter_bookkeeping():
                sid = bk.get("speaker_id", "")
                if sid:
                    self._speaker_key_index.setdefault(sid, set()).add(key)

        logger.info(
            "Router loaded: %d speakers indexed (%d keys total)",
            len(self._speaker_key_index),
            sum(len(v) for v in self._speaker_key_index.values()),
        )

    def route(
        self,
        text: str,
        speaker_id: str | None = None,
    ) -> RoutingPlan:
        """Route a query: classify intent, scope keys to the speaker.

        ``speaker_id`` is the privacy boundary — only the speaker's own
        keys can reach ``plan.steps``.  Intent comes from the query
        content via the encoder residual + HA-match fast path.

        Returns a :class:`RoutingPlan` with:

        * ``intent`` — encoder + HA-match verdict.
        * ``steps`` — per-tier :class:`RoutingStep` with speaker-scoped
          keys for the tiers selected by intent.
        * ``ha_domains`` — HA-graph observability metadata.
        """
        allowed_keys = self._speaker_key_index.get(speaker_id, set()) if speaker_id else set()

        ha_match: HAMatchResult | None = None
        if self._ha_graph is not None:
            ha_match = self._ha_graph.match(text)
        has_ha = ha_match is not None and ha_match.has_entity_match
        ha_domains = ha_match.domains if ha_match else []

        # Lazy import: avoids the router↔intent module cycle at load time.
        from paramem.server.intent import classify_intent

        intent = classify_intent(
            text,
            has_ha_match=has_ha,
            config=self._intent_config,
        )

        steps = self._steps_for_intent(intent, allowed_keys)

        if steps or has_ha:
            logger.info(
                "Routed query: intent=%s tiers=%s ha=%s",
                intent.value,
                [s.adapter_name for s in steps],
                ha_domains,
            )

        return RoutingPlan(
            steps=steps,
            strategy="targeted_probe" if steps else "direct",
            ha_domains=ha_domains,
            intent=intent,
        )

    def _steps_for_intent(self, intent: Intent, allowed_keys: set[str]) -> list[RoutingStep]:
        """Map intent → ordered :class:`RoutingStep` list, speaker-scoped.

        * ``PERSONAL`` — procedural → newest interim first → episodic → semantic.
        * ``COMMAND``  — ``procedural`` MAIN (unfiltered, pure preferences after
          a fold) then interim slots filtered to preference keys only (proc-prefix
          OR ``relation_type=="preference"`` in bookkeeping).  Identity facts in
          interim slots are excluded by construction (DEFAULT-DENY).
        * ``GENERAL``  — empty (route to SOTA with no personal injection).
        * ``UNKNOWN``  — resolved via ``IntentConfig.fail_closed_intent``
          (default ``PERSONAL`` — privacy-preserving fallback).

        Returns an empty list when ``allowed_keys`` is empty (anonymous
        speaker or enrolled speaker with no keys).
        """
        if not allowed_keys:
            return []

        effective = intent
        if intent is Intent.UNKNOWN:
            effective = self._resolve_unknown_intent()

        if effective is Intent.PERSONAL:
            tiers = self._personal_tier_order()
        elif effective is Intent.COMMAND:
            # COMMAND: probe procedural MAIN unfiltered, then interim slots with
            # the preference-only filter so factual/temporal/social interim keys
            # never reach HA payloads.
            steps: list[RoutingStep] = []
            proc_keys = self._tier_keys("procedural")
            scoped_proc = sorted(allowed_keys & proc_keys)
            if scoped_proc:
                steps.append(RoutingStep(adapter_name="procedural", keys_to_probe=scoped_proc))
            for interim_tier in self._command_interim_tiers():
                interim_keys = self._tier_keys(interim_tier, preference_only=True)
                scoped = sorted(allowed_keys & interim_keys)
                if scoped:
                    steps.append(RoutingStep(adapter_name=interim_tier, keys_to_probe=scoped))
            return steps
        else:
            return []

        steps = []
        for tier in tiers:
            tier_keys = self._tier_keys(tier)
            scoped = sorted(allowed_keys & tier_keys)
            if scoped:
                steps.append(RoutingStep(adapter_name=tier, keys_to_probe=scoped))
        return steps

    def _resolve_unknown_intent(self) -> Intent:
        """Resolve ``Intent.UNKNOWN`` via ``IntentConfig.fail_closed_intent``.

        Defaults to ``Intent.PERSONAL`` so unrecognised queries from an
        enrolled speaker keep their context on-device.
        """
        config = self._intent_config
        if config is None:
            return Intent.PERSONAL
        try:
            return Intent(config.fail_closed_intent)
        except ValueError:
            return Intent.PERSONAL

    def _personal_tier_order(self) -> list[str]:
        """PERSONAL probe order: procedural → newest interim first → episodic → semantic.

        Interim tier names are NOT normalised — ``episodic_interim_<stamp>``
        is the canonical adapter name during an interim window and must
        match ``model.peft_config`` at probe time so
        ``switch_adapter(model, step.adapter_name)`` lands on the trained
        slot.

        When :class:`MemoryStore` has ``replay_enabled=False``,
        ``tiers_with_registry()`` returns an empty list and this method
        silently degrades to the bare ``["procedural", "episodic",
        "semantic"]`` order — interim slots are unreachable without the
        registry to enumerate them.  That's the expected behaviour for
        replay-disabled stores (no lifecycle tracking → no interim
        rotation), but worth noting because the degradation is silent.
        """
        store = self._memory_store
        interim_names: list[str] = []
        if store is not None:
            for tier in store.tiers_with_registry():
                if _interim_sort_key(tier) is not None:
                    interim_names.append(tier)
            interim_names.sort(key=lambda n: _interim_sort_key(n) or "", reverse=True)
        return [
            *_PERSONAL_TIERS_PRE_INTERIM,
            *interim_names,
            *_PERSONAL_TIERS_POST_INTERIM,
        ]

    def _command_interim_tiers(self) -> list[str]:
        """Interim slot names in newest-first order for the COMMAND probe.

        Reuses the same :meth:`MemoryStore.tiers_with_registry` enumeration
        as :meth:`_personal_tier_order` — no duplicate logic.  Only interim
        slots are returned; ``procedural`` MAIN is handled separately in
        :meth:`_steps_for_intent`.
        """
        store = self._memory_store
        interim_names: list[str] = []
        if store is not None:
            for tier in store.tiers_with_registry():
                if _interim_sort_key(tier) is not None:
                    interim_names.append(tier)
            interim_names.sort(key=lambda n: _interim_sort_key(n) or "", reverse=True)
        return interim_names

    def _tier_keys(self, tier: str, *, preference_only: bool = False) -> set[str]:
        """Active key set for *tier*.

        Reads :meth:`MemoryStore.active_keys_in_tier`, which walks the
        registry, not ``_entries``.  Preload-independent; empty when replay
        is disabled or the tier has no registry.

        When ``preference_only=True``, restricts to keys whose key name
        starts with ``"proc"`` OR whose bookkeeping ``relation_type`` is
        ``"preference"``.  This is the COMMAND-path filter for interim slots:
        factual, temporal, social, and unknown interim keys are excluded
        (DEFAULT-DENY) so identity facts never reach HA payloads.

        ``preference_only`` applies ONLY to interim slots on the COMMAND path.
        ``procedural`` MAIN is always probed unfiltered.
        """
        if self._memory_store is None:
            return set()
        keys = set(self._memory_store.active_keys_in_tier(tier))
        if not preference_only:
            return keys
        filtered: set[str] = set()
        for key in keys:
            if key.startswith("proc"):
                filtered.add(key)
                continue
            bk = self._memory_store.bookkeeping_for_key(key) or {}
            if bk.get("relation_type") == "preference":
                filtered.add(key)
        return filtered
