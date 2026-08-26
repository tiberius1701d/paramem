"""Off-store assembly of one tier's shadow artifact set into a writable increment.

:func:`build_tier_increment` is THE ONE PRODUCER of :class:`TierIncrement` —
every increment in the system, both training events, the trial path, and
every resume, comes out of this one function reading one shadow directory.
Nothing else constructs one.

Imports :mod:`paramem.memory.entry` inside the function body rather than at
module scope: ``entry.py`` imports ``persistence`` at its own module bottom,
so a module-scope import here would close a
``persistence -> increment -> entry -> persistence`` cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase

    from paramem.memory.store import MemoryStore
    from paramem.training.key_registry import KeyRegistry
    from paramem.utils.config import AdapterConfig

_REGISTRY_FILENAME = "indexed_key_registry.json"
_KEY_METADATA_FILENAME = "key_metadata.json"
_KEYED_FILENAME = "keyed.json"


@dataclass(frozen=True)
class TierIncrement:
    """One tier's shadow artifact set, assembled off-store, ready to write.

    :func:`build_tier_increment` is the SOLE producer: every increment in the
    system — both events, the trial path, and every resume — comes out of
    that one function reading one shadow directory.  Nothing else constructs
    one.
    """

    tier: str  # the driver's tier loop variable
    adapter_name: str  # ledger["tiers"][tier]["adapter"] — recorded in
    # phase 1, never re-derived from the schedule
    registry: "KeyRegistry"  # active keys + withheld ids + simhash; the PARSE of
    # registry_bytes, for adoption and for len()
    registry_bytes: bytes  # the shadow indexed_key_registry.json's
    # plaintext payload, read verbatim.  The write stamps
    # sha256(registry_bytes) into the manifest and the publish flushes
    # these exact bytes — byte-stability across a process boundary is
    # what binds the slot
    rows_bytes: bytes  # the shadow key_metadata.json's plaintext payload,
    # published verbatim for the same one-rule reason
    entries: "dict[str, dict]"  # key -> content_only_entry: a MATERIALIZED
    # PROJECTION of `keyed`, computed once here and never re-derived
    # downstream.  EMPTY for a rows-only member
    bookkeeping: "dict[str, dict]"  # key -> the per-key row, incl. the
    # promoted flag; ALWAYS the tier's complete row set; the parse of
    # rows_bytes
    keyed: "list[dict]"  # the PERSISTED keyed list, loaded verbatim:
    # key/subject/predicate/object/speaker_id/relation_type
    rebuilt: bool  # True: this event re-derived the tier's CONTENT
    # (keyed.json exists in the shadow dir), so the member replaces the
    # live entry cache — and, when `keyed` is non-empty, trains and writes
    # a payload.  False: a ROWS-ONLY member — registry and rows changed,
    # content did not — no training, no gate, no payload, and convergence
    # leaves _entries alone.  Derived from file presence, so an empty
    # keyed.json (a tier rebuilt to zero keys) stays distinguishable from
    # an absent one.
    pre_sha: str  # ledger["tiers"][tier]["pre_sha"] — this tier's live
    # registry digest at phase 1
    # A retired key is gone by ABSENCE from registry/entries/bookkeeping;
    # adoption is convergence, so nothing tracks deletion.

    @property
    def has_payload(self) -> bool:
        """True when this increment carries a payload to train and write.

        The payload gate of the go-live's act 1-3: ``rebuilt`` AND a
        non-empty ``keyed`` list.  A tier this event rebuilt to zero keys
        has a keyed list (empty) and nothing to train, so it publishes its
        rows and registry like every other member and writes no artifact.
        """
        return self.rebuilt and bool(self.keyed)


@dataclass(frozen=True)
class TierWriteContext:
    """The collaborators the per-tier write acts need.  Derives nothing.

    Rather than passing the whole live consolidation loop into
    ``persistence.py`` and ``go_live.py``, this names only the
    collaborators each act needs: the write reads five of these fields,
    the publish one, the go-live five, and naming them is what makes the
    module boundary real rather than decorative.  Built once per event by
    the fold driver, from its own state.
    """

    model: "PeftModel"  # loop.model
    tokenizer: "PreTrainedTokenizerBase"  # loop.tokenizer
    fingerprint_cache: dict  # loop.fingerprint_cache
    output_dir: Path  # loop.output_dir (the trial tree for a trial loop)
    tier_configs: "Mapping[str, AdapterConfig]"  # a TOTAL map over every
    # tier the caller's bundle names -- the {episodic,semantic,procedural}
    # configs plus, when the bundle includes one, an interim tier's own
    # entry (episodic-shaped, resolved via
    # ConsolidationLoop._tier_adapter_config, the one rule home).
    # publish_bundle's mount step does a plain ``tier_configs[tier]``
    # lookup -- a KeyError there means the driver failed to resolve one of
    # its own bundle's members, never a tier this map is allowed to omit.
    store: "MemoryStore"  # loop.store — the go-live's adoption target ONLY;
    # neither write nor publish reads it
    keep_prior_slots: int  # consolidation.training_keep_prior_slots


def build_tier_increment(
    *,
    tier: str,
    adapter_name: str,
    pre_sha: str,
    shadow_dir: Path,
) -> TierIncrement:
    """Assemble one tier's increment from its persisted shadow artifacts.

    THE ONE PRODUCER of :class:`TierIncrement`.  Reads
    ``indexed_key_registry.json`` and ``key_metadata.json`` always — keeping
    each file's plaintext payload verbatim as ``registry_bytes`` /
    ``rows_bytes`` AND parsing it once into ``registry`` / ``bookkeeping``,
    so the bytes that go live and the objects that are adopted cannot
    diverge — and ``keyed.json`` when *shadow_dir* holds one.  That file's
    presence is what sets ``rebuilt``: this function itself does no ledger
    verification, it only reads what is on disk right now.  Presence is
    trustworthy upstream because the caller,
    :meth:`~paramem.training.consolidation.ConsolidationLoop.run_build_and_publish`,
    verifies the ledger's extraction entry against on-disk bytes before
    calling this function for any tier — a ``keyed.json`` the ledger names
    but disk lacks fails that verification and the event re-extracts,
    rather than this function silently reading the gap as a rows-only
    member.

    Entries are the materialized projection of the keyed rows, computed here
    and only here: never from graph topology and never from node ids (see
    ``paramem/memory/entry.py``, ``verify_confidence``'s fingerprint
    rebuild). Holds no reference to any live structure — the increment is
    assembled entirely off the live store, from the shadow artifacts alone.

    Args:
        tier: The driver's tier loop variable (e.g. ``"episodic"``).
        adapter_name: ``ledger["tiers"][tier]["adapter"]`` — recorded in
            phase 1, never re-derived from the schedule.
        pre_sha: ``ledger["tiers"][tier]["pre_sha"]`` — this tier's live
            registry digest at phase 1.
        shadow_dir: ``<data>/state/extraction/<event>/shadow/<tier>/`` — the
            directory phase 1 wrote this tier's shadow artifacts into.

    Returns:
        The assembled :class:`TierIncrement`.

    Raises:
        FileNotFoundError: *shadow_dir* holds neither a shadow registry nor
            shadow rows — there is no shadow artifact set to build an
            increment from for this tier.
    """
    from paramem.backup.encryption import read_maybe_encrypted
    from paramem.memory.entry import content_only_entry
    from paramem.training.key_registry import KeyRegistry

    shadow_dir = Path(shadow_dir)
    registry_path = shadow_dir / _REGISTRY_FILENAME
    rows_path = shadow_dir / _KEY_METADATA_FILENAME
    keyed_path = shadow_dir / _KEYED_FILENAME

    if not registry_path.exists() or not rows_path.exists():
        raise FileNotFoundError(
            f"build_tier_increment: no shadow artifact set for tier {tier!r} under "
            f"{shadow_dir} — expected {_REGISTRY_FILENAME} and {_KEY_METADATA_FILENAME}"
        )

    registry_bytes = read_maybe_encrypted(registry_path)
    registry = KeyRegistry.load_from_bytes(registry_bytes, path=registry_path)

    rows_bytes = read_maybe_encrypted(rows_path)
    rows_payload = json.loads(rows_bytes.decode("utf-8"))
    bookkeeping = dict(rows_payload.get("keys", {}))

    rebuilt = keyed_path.exists()
    keyed: list[dict] = []
    entries: dict[str, dict] = {}
    if rebuilt:
        keyed = json.loads(read_maybe_encrypted(keyed_path).decode("utf-8"))
        entries = {kp["key"]: content_only_entry(kp) for kp in keyed}

    return TierIncrement(
        tier=tier,
        adapter_name=adapter_name,
        registry=registry,
        registry_bytes=registry_bytes,
        rows_bytes=rows_bytes,
        entries=entries,
        bookkeeping=bookkeeping,
        keyed=keyed,
        rebuilt=rebuilt,
        pre_sha=pre_sha,
    )
