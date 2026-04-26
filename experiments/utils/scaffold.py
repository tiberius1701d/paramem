"""Scaffold builders for Test 14 — content-free placeholder scaffolding.

Five variants differ only in what placeholder text is used during the
scaffold-build phase (Phase B).  Phase C (fill) always uses real Q+A for
all variants.  V3_extended is an orchestration-level concept (reuse V3 Phase
B adapter + continue training) and shares the V3 builder.

Variant definitions
-------------------
V1 — per-slot unique Q and A placeholders:
    question = "TBD-Q-{N}"  answer = "TBD-A-{N}"
V2 — structural Q + placeholder A:
    question = "Question for slot {N}"  answer = "TBD-A-{N}"
V3 — uniform sentinel for both fields:
    question = "pending"  answer = "pending"
V3_extended — same builder as V3; orchestration loads V3's existing Phase B
    adapter and trains for additional epochs.
V4 — no JSON content scaffold:
    target JSON contains only the ``key`` field; no ``question`` or ``answer``.
    Tests whether the Q/A framing in V1/V2/V3 was load-bearing.
V5 — random-byte placeholder (deterministic per slot):
    question = sha256("V5-Q-{N}")[:16]  answer = sha256("V5-A-{N}")[:16]
    Tests whether maximal per-slot uniqueness routes better than V3's uniform
    sentinel.

All builders use ``f"graph{i}"`` keys (1-based, matching ``assign_keys``
convention) so that registry / SimHash / probe / confusion-matrix tooling
treats scaffold and real keys identically.
"""

from __future__ import annotations

import hashlib

V1 = "V1"
V2 = "V2"
V3 = "V3"
V3_EXTENDED = "V3_extended"
V4 = "V4"
V5 = "V5"

VARIANTS = (V1, V2, V3, V3_EXTENDED, V4, V5)

# Placeholder strings used for leakage detection in phase results
PLACEHOLDER_STRINGS = ("TBD-", "pending", "Question for slot")


def build_v1_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V1 scaffold: per-slot unique Q and A placeholders.

    Each slot gets ``question="TBD-Q-{N}"`` and ``answer="TBD-A-{N}"``
    where N is the 1-based slot index.  Keys follow the ``graph{i}``
    convention used by ``assign_keys``.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1; override for extension).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {
            "key": f"graph{i}",
            "question": f"TBD-Q-{i}",
            "answer": f"TBD-A-{i}",
        }
        for i in range(start_index, start_index + n_keys)
    ]


def build_v2_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V2 scaffold: structural Q (slot-indexed) + placeholder A.

    Each slot gets ``question="Question for slot {N}"`` and
    ``answer="TBD-A-{N}"``.  The structural question gives more
    discriminative signal than V1's "TBD-Q-{N}" because it reads as a
    natural phrase, but the answer remains content-free.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {
            "key": f"graph{i}",
            "question": f"Question for slot {i}",
            "answer": f"TBD-A-{i}",
        }
        for i in range(start_index, start_index + n_keys)
    ]


def build_v3_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V3 scaffold: uniform sentinel for both fields.

    All slots share the same ``question="pending"`` and ``answer="pending"``.
    This is the lowest-information variant; discrimination between slots
    must emerge purely from key identity (``graph{i}``).

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {
            "key": f"graph{i}",
            "question": "pending",
            "answer": "pending",
        }
        for i in range(start_index, start_index + n_keys)
    ]


def build_v4_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V4 scaffold: empty-string Q/A — no per-slot content.

    Each slot has empty ``question`` and ``answer`` strings.  This is the
    minimum-content scaffold: the JSON Q/A framing is preserved (so
    downstream consumers like ``build_registry``, ``format_indexed_training``,
    and ``validate_recall`` receive the fields they expect) but no per-slot
    content is encoded.  Tests whether the JSON Q/A *fields* were
    load-bearing for binding pre-formation or whether key identity alone
    suffices.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1; override for extension).

    Returns:
        List of dicts with ``key``, empty ``question``, empty ``answer``.
    """
    return [
        {"key": f"graph{i}", "question": "", "answer": ""}
        for i in range(start_index, start_index + n_keys)
    ]


def build_v5_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V5 scaffold: deterministic random-hex placeholders per slot.

    Each slot gets a unique 16-character hex string derived by hashing the
    slot index:

    - ``question = hashlib.sha256(b"V5-Q-{i}").hexdigest()[:16]``
    - ``answer   = hashlib.sha256(b"V5-A-{i}").hexdigest()[:16]``

    The strings are fully deterministic across runs: the same ``i`` always
    produces the same string.  This tests whether maximal per-slot uniqueness
    provides better discriminative signal than V3's uniform ``"pending"``
    sentinel.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {
            "key": f"graph{i}",
            "question": hashlib.sha256(f"V5-Q-{i}".encode()).hexdigest()[:16],
            "answer": hashlib.sha256(f"V5-A-{i}".encode()).hexdigest()[:16],
        }
        for i in range(start_index, start_index + n_keys)
    ]


VARIANT_BUILDERS = {
    V1: build_v1_scaffold,
    V2: build_v2_scaffold,
    V3: build_v3_scaffold,
    # V3_extended reuses V3's scaffold pattern; the "extended" semantics
    # (load existing Phase B adapter, continue training) live in test14.py.
    V3_EXTENDED: build_v3_scaffold,
    V4: build_v4_scaffold,
    V5: build_v5_scaffold,
}


def build_fill_keyed(
    scaffold_keyed: list[dict],
    real_qa_pool: list[dict],
    fill_indices: list[int],
) -> list[dict]:
    """Replace selected scaffold slots with real (Q, A) pairs.

    Walks ``scaffold_keyed`` and replaces slots whose 0-based position is in
    ``fill_indices`` with the corresponding entry from ``real_qa_pool``.
    The ``key`` field is preserved from the scaffold entry; ``question`` and
    ``answer`` come from ``real_qa_pool[k]`` where k is the position in
    ``fill_indices``.

    Only the replaced (fill) slots are returned — the unchanged slots form
    the *retention probe set* in the caller and are NOT included here.
    This matches Test 13's Phase C2 contract.

    Args:
        scaffold_keyed: Full scaffold list (all N entries).
        real_qa_pool: Real QA pairs; must have at least ``len(fill_indices)``
            entries.  Entry k maps to fill slot ``fill_indices[k]``.
        fill_indices: 0-based indices into ``scaffold_keyed`` to replace.

    Returns:
        List of dicts with ``key`` (from scaffold), ``question`` and
        ``answer`` (from ``real_qa_pool``).  Length equals
        ``len(fill_indices)``.

    Raises:
        IndexError: If any fill index is out of range for ``scaffold_keyed``
            or ``real_qa_pool`` does not have enough entries.
        ValueError: If ``fill_indices`` contains duplicates.
    """
    if len(fill_indices) != len(set(fill_indices)):
        raise ValueError("fill_indices must not contain duplicates")
    if len(real_qa_pool) < len(fill_indices):
        raise IndexError(
            f"real_qa_pool has {len(real_qa_pool)} entries but fill_indices "
            f"requires at least {len(fill_indices)}"
        )
    result = []
    for k, slot_idx in enumerate(fill_indices):
        scaffold_entry = scaffold_keyed[slot_idx]
        real_entry = real_qa_pool[k]
        result.append(
            {
                "key": scaffold_entry["key"],
                "question": real_entry["question"],
                "answer": real_entry["answer"],
            }
        )
    return result
