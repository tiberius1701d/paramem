"""Scaffold builders for Test 14 — content-free placeholder scaffolding.

Variants differ only in what placeholder text is used during the
scaffold-build phase (Phase B).  Phase C (fill) always uses real Q+A for
all variants.  V3_extended is an orchestration-level concept (reuse V3 Phase
B adapter + continue training) and shares the V3 builder.

Variant numbering reflects multi-seed batch run order.  V1-V4 are the
original 14a-pre cells (V1/V2/V3 plus V4 empty-Q/A); V5-V8 are the
expanded uniform-axis probe added 2026-04-29 after the n=1 V1-V4
results suggested a "shared transfer channel" mechanism worth
testing.

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
V4 — empty Q/A:
    question = ""  answer = ""  (JSON shape preserved; content empty)
V5 — uniform long natural-language template (most informative new variant):
    question = "What is the answer to this query?"
    answer   = "The answer is currently unknown."
    Tests whether longer uniform natural beats V3's short uniform under
    the shared-transfer-channel hypothesis.
V6 — uniform short non-natural sentinel:
    question = "<PLACEHOLDER>"  answer = "<PLACEHOLDER>"
    Tests whether shorter than V3's "pending" is faster (isolates
    length from naturalness within the uniform regime).
V7 — random-byte placeholder (deterministic per slot):
    question = sha256("V7-Q-{N}")[:16]  answer = sha256("V7-A-{N}")[:16]
    Per-slot unique OOD tokens — upper bound of unique-OOD scaffold cost.
V8 — uniform long OOD hex (same hex string for every slot, fallback):
    question = answer = "a1b2c3d4e5f60718"  (single shared 16-char hex)
    Long uniform OOD — disentangles length from naturalness within the
    uniform regime.  Run only if V5/V6 results are ambiguous.

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
V6 = "V6"
V7 = "V7"
V8 = "V8"

VARIANTS = (V1, V2, V3, V3_EXTENDED, V4, V5, V6, V7, V8)

# Placeholder strings used for leakage detection in phase results.
# Any of these substrings appearing in a recalled answer post-fill counts
# as scaffold-content leakage into the trained Q+A binding.
PLACEHOLDER_STRINGS = (
    "TBD-",
    "pending",
    "Question for slot",
    "What is the answer to this query",
    "The answer is currently unknown",
    "<PLACEHOLDER>",
    "a1b2c3d4e5f60718",
)


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
    """Build a V5 scaffold: uniform long natural-language template.

    All slots share a fluent English question/answer template.  Tests
    whether longer uniform natural placeholder beats V3's short uniform
    sentinel under the shared-transfer-channel hypothesis.  Confounds
    length and naturalness against V3 — V6 + V8 are needed to fully
    disentangle if the result is ambiguous.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {
            "key": f"graph{i}",
            "question": "What is the answer to this query?",
            "answer": "The answer is currently unknown.",
        }
        for i in range(start_index, start_index + n_keys)
    ]


def build_v6_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V6 scaffold: uniform single short non-natural sentinel.

    All slots share ``question="<PLACEHOLDER>"`` and ``answer="<PLACEHOLDER>"``.
    Same uniformity as V3 with a shorter, less natural-language token —
    isolates length from naturalness within the uniform regime.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {"key": f"graph{i}", "question": "<PLACEHOLDER>", "answer": "<PLACEHOLDER>"}
        for i in range(start_index, start_index + n_keys)
    ]


def build_v7_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V7 scaffold: deterministic random-hex placeholders per slot.

    Each slot gets a unique 16-character hex string derived by hashing the
    slot index:

    - ``question = hashlib.sha256(b"V7-Q-{i}").hexdigest()[:16]``
    - ``answer   = hashlib.sha256(b"V7-A-{i}").hexdigest()[:16]``

    The strings are fully deterministic across runs: the same ``i`` always
    produces the same string.  Falsifiable upper-bound check on the
    per-slot-unique-OOD scaffold cost — if V7 converges within budget,
    the shared-transfer-channel hypothesis is wrong.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {
            "key": f"graph{i}",
            "question": hashlib.sha256(f"V7-Q-{i}".encode()).hexdigest()[:16],
            "answer": hashlib.sha256(f"V7-A-{i}".encode()).hexdigest()[:16],
        }
        for i in range(start_index, start_index + n_keys)
    ]


def build_v8_scaffold(n_keys: int, start_index: int = 1) -> list[dict]:
    """Build a V8 scaffold: uniform long OOD hex (same hex for every slot).

    All slots share ``question=answer="a1b2c3d4e5f60718"``.  Same length as
    V7 but uniform across slots — disentangles length from naturalness
    within the uniform regime.  Run only if V5/V6 results are ambiguous.

    Args:
        n_keys: Number of scaffold entries to generate.
        start_index: First slot index (default 1).

    Returns:
        List of dicts with ``key``, ``question``, ``answer``.
    """
    return [
        {"key": f"graph{i}", "question": "a1b2c3d4e5f60718", "answer": "a1b2c3d4e5f60718"}
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
    V6: build_v6_scaffold,
    V7: build_v7_scaffold,
    V8: build_v8_scaffold,
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
