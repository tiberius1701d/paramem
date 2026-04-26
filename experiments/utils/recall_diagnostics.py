"""Recall diagnostic helpers for Test 14.

Stateless pure functions — no GPU, no model, no imports from paramem.

Functions
---------
per_field_split_counts
    Aggregate ``question_match`` / ``answer_match`` flags from
    ``evaluate_indexed_recall``'s ``per_key`` list.

update_first_perfect_log
    Record the first epoch a key achieves ``exact_match=True``.
    Never overwrites an existing entry.

serialize_confusion_matrix
    At phase end: scan never-converged keys to detect cross-slot
    answer confusions.
"""

from __future__ import annotations


def per_field_split_counts(per_key_results: list[dict]) -> dict:
    """Aggregate Q/A field match flags from per-key recall results.

    Consumes the ``per_key`` list returned by ``evaluate_indexed_recall``
    (each entry has ``question_match`` and ``answer_match`` boolean fields).

    The four exhaustive cases:

    - ``both``    — question AND answer both correct (ideal; exact_match).
    - ``q_only``  — question correct, answer wrong (placeholder binding to Q).
    - ``a_only``  — answer correct, question wrong (unusual; possible if
                    scaffold binds A before Q converges).
    - ``neither`` — neither field correct (early training or total failure).

    Args:
        per_key_results: List of per-key dicts, each containing at least
            ``question_match`` (bool) and ``answer_match`` (bool).

    Returns:
        Dict with keys:
            ``both``       — count where q_match AND a_match.
            ``q_only``     — count where q_match AND NOT a_match.
            ``a_only``     — count where NOT q_match AND a_match.
            ``neither``    — count where neither.
            ``total``      — total entries.
            ``q_correct``  — total with q_match (= both + q_only).
            ``a_correct``  — total with a_match (= both + a_only).
    """
    both = 0
    q_only = 0
    a_only = 0
    neither = 0

    for entry in per_key_results:
        qm = bool(entry.get("question_match", False))
        am = bool(entry.get("answer_match", False))
        if qm and am:
            both += 1
        elif qm:
            q_only += 1
        elif am:
            a_only += 1
        else:
            neither += 1

    total = len(per_key_results)
    return {
        "both": both,
        "q_only": q_only,
        "a_only": a_only,
        "neither": neither,
        "total": total,
        "q_correct": both + q_only,
        "a_correct": both + a_only,
    }


def update_first_perfect_log(
    per_key_results: list[dict],
    first_perfect_log: dict,
    epoch: int,
) -> None:
    """Update the per-key first-perfect log in place.

    For each key in ``per_key_results``:

    - If the key has ``exact_match=True`` and is NOT yet in
      ``first_perfect_log``, record it with ``epoch_first_perfect``,
      ``recalled_q``, ``recalled_a``, and ``confidence``.
    - If the key does NOT have ``exact_match=True``:
      - If it IS already in the log with a non-null
        ``epoch_first_perfect``, leave it as-is (already converged).
      - If it is NOT in the log yet, create a stub with
        ``epoch_first_perfect=null`` and update ``last_recalled_*`` fields.
      - If it IS in the log with ``epoch_first_perfect=null``, update the
        ``last_recalled_*`` fields.

    This means ``epoch_first_perfect`` is set ONCE and never overwritten.

    Args:
        per_key_results: Per-key recall result list from
            ``evaluate_indexed_recall``.
        first_perfect_log: Mutable dict keyed by ``key`` string.  Modified
            in place.
        epoch: Current training epoch number.
    """
    for entry in per_key_results:
        key = entry.get("key", "")
        recalled = entry.get("recalled") or {}
        exact = bool(entry.get("exact_match", False))
        confidence = entry.get("confidence", 0.0)
        recalled_q = recalled.get("question", "") if isinstance(recalled, dict) else ""
        recalled_a = recalled.get("answer", "") if isinstance(recalled, dict) else ""

        if key not in first_perfect_log:
            if exact:
                first_perfect_log[key] = {
                    "epoch_first_perfect": epoch,
                    "recalled_q": recalled_q,
                    "recalled_a": recalled_a,
                    "confidence": confidence,
                }
            else:
                first_perfect_log[key] = {
                    "epoch_first_perfect": None,
                    "last_recalled_q": recalled_q,
                    "last_recalled_a": recalled_a,
                    "last_confidence": confidence,
                }
        else:
            existing = first_perfect_log[key]
            if existing.get("epoch_first_perfect") is None:
                if exact:
                    # Promote from stub to converged.
                    first_perfect_log[key] = {
                        "epoch_first_perfect": epoch,
                        "recalled_q": recalled_q,
                        "recalled_a": recalled_a,
                        "confidence": confidence,
                    }
                else:
                    # Update last-seen fields.
                    existing["last_recalled_q"] = recalled_q
                    existing["last_recalled_a"] = recalled_a
                    existing["last_confidence"] = confidence
            # If epoch_first_perfect is already set, never overwrite.


def serialize_confusion_matrix(
    first_perfect_log: dict,
    expected_by_key: dict[str, str],
) -> dict:
    """Detect cross-slot answer confusions for never-converged keys.

    At phase end, for each key that never achieved ``exact_match``
    (``epoch_first_perfect is None``), checks whether the key's
    ``last_recalled_a`` string-equals any OTHER slot's expected answer.
    If so, records ``emitted_under_other_slot`` in ``first_perfect_log``
    for that key.

    This is O(N^2) in the number of never-converged keys but N ≤ 500 so
    the worst case is 250 000 string comparisons — consistently < 1 s.

    Args:
        first_perfect_log: Per-key first-perfect dict (modified in place to
            add ``emitted_under_other_slot`` where applicable).
        expected_by_key: Mapping of ``key → expected_answer`` string, used
            as the cross-slot lookup table.

    Returns:
        Summary dict with:
            ``converged``                      — count with epoch_first_perfect set.
            ``never_converged``                — count without.
            ``never_converged_emitted_other_slot``  — subset that emitted another
                slot's answer.
            ``never_converged_emitted_placeholder`` — subset whose last_recalled_a
                contains a placeholder substring ("TBD-", "pending",
                "Question for slot").
            ``mean_first_perfect_epoch``       — mean over converged keys (None if 0).
            ``std_first_perfect_epoch``        — std over converged keys (None if < 2).
    """
    from experiments.utils.scaffold import PLACEHOLDER_STRINGS  # local import avoids circular

    converged_epochs: list[int] = []
    never_converged_keys: list[str] = []

    for key, entry in first_perfect_log.items():
        if entry.get("epoch_first_perfect") is not None:
            converged_epochs.append(entry["epoch_first_perfect"])
        else:
            never_converged_keys.append(key)

    # Build answer lookup for cross-slot detection.
    # answer → set of keys that have that answer as their expected value.
    answer_to_keys: dict[str, list[str]] = {}
    for k, expected_a in expected_by_key.items():
        answer_to_keys.setdefault(expected_a, []).append(k)

    emitted_other_slot = 0
    emitted_placeholder = 0

    for key in never_converged_keys:
        entry = first_perfect_log[key]
        last_a = entry.get("last_recalled_a", "") or ""

        # Check placeholder content.
        if any(ph in last_a for ph in PLACEHOLDER_STRINGS):
            entry["emitted_placeholder"] = True
            emitted_placeholder += 1

        # Check cross-slot confusion.
        if last_a and last_a in answer_to_keys:
            other_slots = [k for k in answer_to_keys[last_a] if k != key]
            if other_slots:
                entry["emitted_under_other_slot"] = other_slots[0]
                emitted_other_slot += 1

    converged = len(converged_epochs)
    never_converged = len(never_converged_keys)

    mean_fpe: float | None = None
    std_fpe: float | None = None
    if converged_epochs:
        mean_fpe = sum(converged_epochs) / len(converged_epochs)
        if len(converged_epochs) >= 2:
            var = sum((e - mean_fpe) ** 2 for e in converged_epochs) / len(converged_epochs)
            std_fpe = var**0.5

    return {
        "converged": converged,
        "never_converged": never_converged,
        "never_converged_emitted_other_slot": emitted_other_slot,
        "never_converged_emitted_placeholder": emitted_placeholder,
        "mean_first_perfect_epoch": mean_fpe,
        "std_first_perfect_epoch": std_fpe,
    }
