"""Single-sourced row-status vocabulary for the adapter-manifest health block.

The operator-facing tier-health row has one minting site
(:func:`~paramem.server.app._record_manifest_row`, driven by
:func:`~paramem.server.app._validate_adapter_slot`), one gate (the
main-tier dispatch deferral in :mod:`paramem.server.app`), and one renderer
(:mod:`paramem.server.attention`) — this module is the boundary that makes
those three mirrors into one.

Row status is a DISTINCT vocabulary from
:mod:`paramem.adapters.registry_binding`'s verdicts — not an import-graph
isolation boundary: :mod:`paramem.backup` carries no re-export surface and
importing ``paramem.backup.types`` (as :mod:`paramem.server.config` does)
pulls in nothing beyond stdlib, so :mod:`paramem.server.attention` stays
torch/peft-free at module scope too, same as
:mod:`paramem.adapters.registry_binding` (verified: importing it in
isolation stays pure-Python — no torch, no peft, no transformers — its
exact transitive module count is not pinned here since that count drifts
with unrelated stdlib/internal imports; the import-graph guard test is the
source of truth for the live count). The real
reason the two vocabularies stay separate is that they answer different
questions: a verdict is "does
this tier's registry bind to a slot manifest"; a row status is "what does
the operator see on this tier's health row" — several verdicts collapse
onto one row status (``registry_unreadable``/``registry_absent_with_slots``
-> ``registry_unverified``), and :data:`~paramem.adapters.registry_binding.VERIFIED`
/ :data:`~paramem.adapters.registry_binding.NO_CANDIDATES` mint no row at
all. One operator-facing row vocabulary, one minting site, one gate, one
renderer — this module is where that row vocabulary is defined, once.

:data:`ROW_STATUS_FOR_VERDICT`'s KEYS are imported from
:mod:`paramem.adapters.registry_binding`, never restated as literals — a
verdict string changing there is a type error here, not a silent drift. The
row-status STRINGS (the dict's values, and the two frozensets built from
them) are this module's own single definition — operator-facing vocabulary,
not a mirror of anything.
"""

from __future__ import annotations

from typing import Final

from paramem.adapters.registry_binding import (
    KEY_COUNT_MISMATCH,
    KEYS_WITHOUT_SLOT,
    NO_MATCHING_SLOT,
    PAYLOAD_MISMATCH,
    REGISTRY_ABSENT_WITH_SLOTS,
    REGISTRY_UNREADABLE,
)

FINGERPRINT_ROW_STATUSES: Final[frozenset[str]] = frozenset(
    {"mismatch", "manifest_missing", "migrated_unverified"}
)
"""Row statuses for a slot that exists but whose manifest disagrees with the
loaded model's own fingerprint (base model / tokenizer / LoRA shape), or
carries ``UNKNOWN`` fields left by a migrated manifest."""

UNBOUND_ROW_STATUSES: Final[frozenset[str]] = frozenset(
    {"no_matching_slot", "keys_without_slot", "payload_mismatch"}
)
"""Row statuses for a tier whose registry↔slot-manifest binding resolved no
usable live slot at all: no candidate matches the registry hash
(``no_matching_slot``), active keys exist with no slot candidate at all
(``keys_without_slot``), or a bound slot's payload bytes no longer hash to
the manifest's stamped digest (``payload_mismatch``)."""

BINDING_ROW_STATUSES: Final[frozenset[str]] = UNBOUND_ROW_STATUSES | frozenset(
    {"registry_unverified", "key_count_mismatch"}
)
"""Every row status that traces back to
:func:`~paramem.adapters.registry_binding.verify_tier_binding` returning a
non-publishable verdict — :data:`UNBOUND_ROW_STATUSES` plus the two verdicts
that DO resolve a live slot but still fail verification (an
unreadable/absent registry, or a bound slot's ``key_count`` disagreeing
with the registry's active count)."""

PROBLEM_ROW_STATUSES: Final[frozenset[str]] = FINGERPRINT_ROW_STATUSES | BINDING_ROW_STATUSES
"""Every row status a tier-health row can carry that signals a problem — the
fingerprint-mismatch family plus the binding family. Used to suppress a
less-specific attention item when a more-specific one already covers the
same tier."""

ROW_STATUS_FOR_VERDICT: Final[dict[str, str]] = {
    NO_MATCHING_SLOT: "no_matching_slot",
    KEYS_WITHOUT_SLOT: "keys_without_slot",
    KEY_COUNT_MISMATCH: "key_count_mismatch",
    PAYLOAD_MISMATCH: "payload_mismatch",
    REGISTRY_UNREADABLE: "registry_unverified",
    REGISTRY_ABSENT_WITH_SLOTS: "registry_unverified",
}
"""Maps a :mod:`paramem.adapters.registry_binding` verdict string to the row
status :func:`~paramem.server.app._record_manifest_row` mints for it. Every
verdict maps onto itself except the two registry-unreadable verdicts, which
collapse onto the single ``registry_unverified`` row status — one problem
family, with the distinct cause recorded in the row's own ``reason`` field
rather than a second status string.
:data:`~paramem.adapters.registry_binding.VERIFIED` and
:data:`~paramem.adapters.registry_binding.NO_CANDIDATES` are deliberately
absent — both mint no row at all (see ``_validate_adapter_slot``)."""
