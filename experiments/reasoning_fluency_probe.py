"""Reasoning Fluency Probe: graph triples vs. QA-pair context.

Apples-to-apples comparison: when the base model reasons over recalled
context, does it answer natural-language questions as well from **graph
triples** as from the **QA-pair answer text** that the production pipeline
uses?

Same fact set, two context representations, same questions.

Methodology
-----------
1. Load 95 source triples from the quadruple-adapter run's graph snapshot.
2. Load 193 keyed pairs from the simulate kp store (semantic + episodic +
   procedural).
3. Map kps to triples via (norm_predicate, norm_object) + subject compat.
4. Build two context blocks over the matched fact set:
   - **QA context**: bulleted answer text (production format).
   - **Triple context**: bulleted ``<subject> <predicate_humanized> <object>``.
5. Sample N questions from the matched kps (deterministic, seed 42).
6. Run the base Mistral model (no adapter) over all questions x conditions.
7. Score each response against ground-truth answer (token-overlap heuristic).
8. Write comparison.md + metrics.json + results.json to output dir.

Output dir: ``outputs/reasoning_fluency/<timestamp>/``

CLI
---
    python experiments/reasoning_fluency_probe.py
    python experiments/reasoning_fluency_probe.py --n-questions 20 --seed 0
    python experiments/reasoning_fluency_probe.py --output outputs/reasoning_fluency/custom
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fluency")

DEFAULT_QUAD_RUN = project_root / "outputs/quad_scale/mistral/20260511_082251"
DEFAULT_GRAPH_SNAPSHOT = (
    project_root / "data/ha/debug/run_20260510T170022Z_8c1cca/cycle_26/graph_snapshot.json"
)
DEFAULT_KP_STORE = project_root / "data/ha/simulate"

# Shared system prompt — MUST be identical for both conditions.
REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's question using only the facts provided. "
    "If the facts do not contain the answer, say you don't know."
)

CONTEXT_PREAMBLE = "Known facts:"

# Stop-words for the token-overlap scorer (mirrors test9_natural_recall.py).
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "that",
    "this",
    "it",
    "its",
    "and",
    "or",
    "but",
    "not",
    "no",
    "so",
    "if",
    "then",
    "than",
    "who",
    "what",
    "where",
    "when",
    "how",
    "which",
    "their",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "you",
    "your",
    "i",
    "my",
    "me",
    "we",
    "our",
    "us",
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — environment
# ─────────────────────────────────────────────────────────────────────────────


def _load_env() -> None:
    """Source .env into the current process (call from main only)."""
    from experiments.utils.test_harness import load_test_env

    load_test_env()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — triple ↔ kp mapping
# ─────────────────────────────────────────────────────────────────────────────


def _norm_pred(p: str) -> str:
    """Normalise predicate: lowercase, strip, spaces → underscores."""
    return p.strip().lower().replace(" ", "_")


def _norm_ent(e: str) -> str:
    """Normalise entity name: lowercase, strip."""
    return e.strip().lower()


def _humanize_pred(p: str) -> str:
    """Make a predicate human-readable: underscores → spaces, strip."""
    return p.strip().replace("_", " ")


def _derive_display_name(all_kps: list[dict]) -> str:
    """Return the most common non-Speaker0 source_subject among all kps.

    This is the display name for the CV subject, used to bridge the
    Speaker0 alias in the source triples with the resolved name in the kp store.

    Returns an empty string if no non-Speaker0 subject is found.
    """
    non_speaker0 = [
        kp["source_subject"]
        for kp in all_kps
        if kp["source_subject"].strip().lower() != "speaker0" and kp["source_subject"].strip()
    ]
    if not non_speaker0:
        return ""
    counts = Counter(non_speaker0)
    return counts.most_common(1)[0][0]


def _subject_compat(
    triple_subj_norm: str,
    kp_subj_norm: str,
    display_name_norm: str,
) -> bool:
    """Return True if the two normalised subjects refer to the same entity.

    Subjects match if:
    - They are string-equal after normalisation, OR
    - Both are "the speaker" — one is ``speaker0`` and the other is the
      resolved display name.
    """
    if triple_subj_norm == kp_subj_norm:
        return True
    # Speaker0 ↔ display name aliasing.
    speaker_tokens = {"speaker0"}
    triple_is_speaker = triple_subj_norm in speaker_tokens
    kp_is_speaker = kp_subj_norm in speaker_tokens
    if display_name_norm:
        triple_is_speaker = triple_is_speaker or triple_subj_norm == display_name_norm
        kp_is_speaker = kp_is_speaker or kp_subj_norm == display_name_norm
    return triple_is_speaker and kp_is_speaker


def load_all_kps(kp_store: Path) -> list[dict]:
    """Load keyed pairs from semantic, episodic, and procedural tiers.

    Uses ``read_keyed_pairs`` from the canonical I/O facade.  Requires
    ``PARAMEM_DAILY_PASSPHRASE`` in the environment for age-encrypted files.

    Returns a flat list of all kps with a ``_tier`` key injected for
    downstream tier-aware sampling.
    """
    from paramem.training.keyed_pairs_io import read_keyed_pairs

    tiers = ["semantic", "episodic", "procedural"]
    all_kps: list[dict] = []
    for tier in tiers:
        path = kp_store / tier / "keyed_pairs.json"
        pairs = read_keyed_pairs(path)
        for kp in pairs:
            kp_copy = dict(kp)
            kp_copy["_tier"] = tier
            all_kps.append(kp_copy)
        logger.info("Loaded %d kps from %s", len(pairs), tier)
    logger.info("Total kps loaded: %d", len(all_kps))
    return all_kps


def build_triple_kp_mapping(
    source_triples: list[tuple[str, str, str]],
    all_kps: list[dict],
    display_name: str,
) -> dict:
    """Map each source triple (by graph key) to its matching kp keys.

    Matching logic:
    - Primary key: ``(norm_predicate, norm_object)``
    - Subject compatibility check (handles Speaker0 ↔ display_name aliasing)
    - If the (norm_pred, norm_obj) pair is ambiguous after subject-compat filter,
      the kp is skipped with a logged warning.

    Returns a dict with:
    - ``triple_key_to_kp_keys``: dict ``graph<N>`` → list[kp_key]
    - ``mapping_stats``: breakdown dict (reported + written to disk)
    - ``matched_triples``: list of (triple_idx, triple, kp_list) for the fact set
    """
    display_name_norm = _norm_ent(display_name)

    # Build a lookup: (norm_pred, norm_obj) → list of triple indices
    pred_obj_to_triple_idxs: dict[tuple[str, str], list[int]] = {}
    for i, (s, p, o) in enumerate(source_triples):
        key = (_norm_pred(p), _norm_ent(o))
        pred_obj_to_triple_idxs.setdefault(key, []).append(i)

    # For each kp, find the matching triple(s).
    triple_to_kps: dict[int, list[str]] = {}
    skipped_ambiguous = 0
    skipped_no_match = 0

    for kp in all_kps:
        kp_pred_norm = _norm_pred(kp["source_predicate"])
        kp_obj_norm = _norm_ent(kp["source_object"])
        kp_subj_norm = _norm_ent(kp["source_subject"])
        kp_key = kp["key"]

        po_key = (kp_pred_norm, kp_obj_norm)
        candidate_idxs = pred_obj_to_triple_idxs.get(po_key, [])

        if not candidate_idxs:
            skipped_no_match += 1
            logger.debug(
                "kp %s: no source triple matches (pred=%s, obj=%s)",
                kp_key,
                kp_pred_norm,
                kp_obj_norm,
            )
            continue

        # Filter by subject compatibility.
        compat_idxs = [
            idx
            for idx in candidate_idxs
            if _subject_compat(
                _norm_ent(source_triples[idx][0]),
                kp_subj_norm,
                display_name_norm,
            )
        ]

        if not compat_idxs:
            # Fall through — none passed subject compat; try all candidates.
            compat_idxs = candidate_idxs

        if len(compat_idxs) > 1:
            logger.warning(
                "kp %s ambiguous: %d source triples match (pred=%s, obj=%s) — skipping",
                kp_key,
                len(compat_idxs),
                kp_pred_norm,
                kp_obj_norm,
            )
            skipped_ambiguous += 1
            continue

        triple_idx = compat_idxs[0]
        triple_to_kps.setdefault(triple_idx, []).append(kp_key)

    # Build graph-key → [kp_keys].
    triple_key_to_kp_keys: dict[str, list[str]] = {}
    for triple_idx, kp_keys in triple_to_kps.items():
        graph_key = f"graph{triple_idx + 1}"
        triple_key_to_kp_keys[graph_key] = kp_keys

    # Compute counts.
    matched_zero = sum(1 for i in range(len(source_triples)) if i not in triple_to_kps)
    matched_exactly_one = sum(
        1 for i in range(len(source_triples)) if len(triple_to_kps.get(i, [])) == 1
    )
    matched_more_than_one = sum(
        1 for i in range(len(source_triples)) if len(triple_to_kps.get(i, [])) > 1
    )
    m = len(triple_to_kps)
    k = sum(len(v) for v in triple_to_kps.values())

    mapping_stats = {
        "total_source_triples": len(source_triples),
        "matched_zero_kps": matched_zero,
        "matched_exactly_one_kp": matched_exactly_one,
        "matched_more_than_one_kp": matched_more_than_one,
        "matched_fact_set_size_M": m,
        "matched_kp_count_K": k,
        "skipped_no_match": skipped_no_match,
        "skipped_ambiguous": skipped_ambiguous,
        "display_name_used": display_name,
    }

    logger.info(
        "Mapping: %d source triples → M=%d matched facts, K=%d matched kps "
        "(0-kp=%d, 1-kp=%d, >1-kp=%d, no-match=%d, ambiguous=%d)",
        len(source_triples),
        m,
        k,
        matched_zero,
        matched_exactly_one,
        matched_more_than_one,
        skipped_no_match,
        skipped_ambiguous,
    )

    # Build matched_triples: list of (triple_idx, triple, [kp_dicts])
    kp_by_key: dict[str, dict] = {kp["key"]: kp for kp in all_kps}
    matched_triples = []
    for triple_idx in sorted(triple_to_kps.keys()):
        triple = source_triples[triple_idx]
        kp_keys = triple_to_kps[triple_idx]
        kp_dicts = [kp_by_key[kk] for kk in kp_keys if kk in kp_by_key]
        matched_triples.append((triple_idx, triple, kp_dicts))

    return {
        "triple_key_to_kp_keys": triple_key_to_kp_keys,
        "mapping_stats": mapping_stats,
        "matched_triples": matched_triples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — build context blocks
# ─────────────────────────────────────────────────────────────────────────────


def build_qa_context(matched_triples: list, all_kps: list[dict]) -> str:
    """Build the QA-context block: bulleted answer text, one bullet per kp.

    Mirrors production ``inference.py:722``:
    ``layer_facts.append(f"- {result.get('answer', '')}")``.

    Bullets are ordered by: tier (semantic → episodic → procedural) then kp key.
    """
    tier_order = {"semantic": 0, "episodic": 1, "procedural": 2}
    bullets: list[tuple[int, str, str]] = []  # (tier_rank, kp_key, answer)

    for _triple_idx, _triple, kp_dicts in matched_triples:
        for kp in kp_dicts:
            tier_rank = tier_order.get(kp.get("_tier", "episodic"), 1)
            bullets.append((tier_rank, kp["key"], kp["answer"]))

    bullets.sort(key=lambda x: (x[0], x[1]))
    lines = [CONTEXT_PREAMBLE] + [f"- {ans}" for _, _, ans in bullets]
    return "\n".join(lines)


def build_triple_context(
    matched_triples: list,
    display_name: str,
    source_triples: list[tuple[str, str, str]],
) -> str:
    """Build the triple-context block: bulleted ``subject predicate object``.

    Bullets are ordered by triple index (stable, mirrors key assignment order).
    Speaker0 is substituted with the resolved display name.
    """
    bullets: list[tuple[int, str]] = []  # (triple_idx, bullet_text)

    for triple_idx, triple, _kp_dicts in matched_triples:
        s, p, o = triple
        s_norm = display_name if s.strip().lower() == "speaker0" and display_name else s
        p_human = _humanize_pred(p)
        bullet = f"- {s_norm} {p_human} {o}"
        bullets.append((triple_idx, bullet))

    bullets.sort(key=lambda x: x[0])
    lines = [CONTEXT_PREAMBLE] + [b for _, b in bullets]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — question set
# ─────────────────────────────────────────────────────────────────────────────


def sample_questions(
    matched_triples: list,
    n: int,
    seed: int,
) -> list[dict]:
    """Sample N questions from matched kps, deterministically.

    Strategy: round-robin by tier (semantic → episodic → procedural) to
    spread coverage, then truncate to N.  Fixed-seed shuffle within each
    tier bucket to break ties deterministically.

    Each returned dict has: ``question``, ``answer``, ``kp_key``, ``tier``.
    """
    tier_order = ["semantic", "episodic", "procedural"]

    # Collect all (kp) candidates per tier.
    buckets: dict[str, list[dict]] = {t: [] for t in tier_order}
    for _triple_idx, _triple, kp_dicts in matched_triples:
        for kp in kp_dicts:
            tier = kp.get("_tier", "episodic")
            if kp.get("question") and kp.get("answer"):
                buckets.setdefault(tier, []).append(
                    {
                        "question": kp["question"],
                        "answer": kp["answer"],
                        "kp_key": kp["key"],
                        "tier": tier,
                    }
                )

    # Sort + shuffle within each tier bucket for reproducibility.
    rng = random.Random(seed)
    for tier in tier_order:
        bucket = buckets[tier]
        bucket.sort(key=lambda x: x["kp_key"])
        rng.shuffle(bucket)

    # Round-robin pick up to N.
    selected: list[dict] = []
    seen_keys: set[str] = set()
    tier_iters = {t: iter(buckets[t]) for t in tier_order}
    while len(selected) < n:
        added_this_round = False
        for tier in tier_order:
            if len(selected) >= n:
                break
            it = tier_iters[tier]
            try:
                candidate = next(it)
                if candidate["kp_key"] not in seen_keys:
                    selected.append(candidate)
                    seen_keys.add(candidate["kp_key"])
                    added_this_round = True
            except StopIteration:
                pass
        if not added_this_round:
            break

    if len(selected) < n:
        logger.warning("Only %d questions available (requested %d)", len(selected), n)

    return selected[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — scoring
# ─────────────────────────────────────────────────────────────────────────────


def _content_tokens(text: str) -> set[str]:
    """Extract content words (lowercase, stop-words removed)."""
    return set(text.lower().split()) - _STOP_WORDS


def _salient_tokens(answer: str) -> set[str]:
    """Extract salient ground-truth tokens for overlap scoring.

    Returns distinct content tokens from the answer that are either:
    - length ≥ 4 characters, or
    - digit-containing.
    """
    tokens = _content_tokens(answer)
    return {t for t in tokens if len(t) >= 4 or any(c.isdigit() for c in t)}


def contains_truth(response: str, ground_truth_answer: str, threshold: float = 0.70) -> bool:
    """Return True if ≥70% of salient ground-truth tokens appear in the response.

    This is a rough heuristic signal, not a verdict.  The threshold of 0.70
    means that minor paraphrasing still scores True while a completely off-topic
    or empty response scores False.

    Args:
        response: Model response string.
        ground_truth_answer: Ground-truth answer string from the kp store.
        threshold: Fraction of salient tokens that must appear (default 0.70).

    Returns:
        True when the response contains at least ``threshold`` fraction of the
        salient tokens from the ground-truth answer.
    """
    salient = _salient_tokens(ground_truth_answer)
    if not salient:
        # No salient tokens — fall back to any content-token overlap.
        ct = _content_tokens(ground_truth_answer)
        if not ct:
            return True  # Empty answer — vacuously true.
        response_lower = response.lower()
        matched = sum(1 for t in ct if t in response_lower)
        return matched / len(ct) >= threshold

    response_lower = response.lower()
    matched = sum(1 for t in salient if t in response_lower)
    return matched / len(salient) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — run the reasoner
# ─────────────────────────────────────────────────────────────────────────────


def _build_user_message(context_block: str, question: str) -> str:
    """Build the user message for a single inference call.

    Format mirrors production ``inference.py`` reasoning context assembly.
    """
    return f"{context_block}\n\nQuestion: {question}"


def run_reasoning_probe(
    model,
    tokenizer,
    questions: list[dict],
    qa_context: str,
    triple_context: str,
    output_dir: Path,
) -> list[dict]:
    """Run both context conditions for each question; write results incrementally.

    For each question and each condition (qa, triple), builds the user message,
    formats with ``_format_inference_prompt``'s template approach, and calls
    ``generate_answer`` at temperature 0.0.

    Results are appended to ``output_dir/results.json`` after each question so
    a crash does not lose progress.

    Args:
        model: Base model (no adapter active; gradient_checkpointing disabled).
        tokenizer: Tokenizer paired with model.
        questions: List of question dicts from ``sample_questions``.
        qa_context: Formatted QA-context block string.
        triple_context: Formatted triple-context block string.
        output_dir: Directory to write incremental results.json.

    Returns:
        List of per-question result dicts.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    results_path = output_dir / "results.json"
    results: list[dict] = []

    for i, q in enumerate(questions):
        logger.info("Question %d/%d [%s]: %s", i + 1, len(questions), q["tier"], q["question"])
        record: dict = {
            "question": q["question"],
            "ground_truth_answer": q["answer"],
            "source_kp_key": q["kp_key"],
            "source_tier": q["tier"],
        }

        for condition, ctx_block in [("qa", qa_context), ("triple", triple_context)]:
            user_msg = _build_user_message(ctx_block, q["question"])
            messages = adapt_messages(
                [
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                tokenizer,
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate_answer(
                model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
            )
            record[f"{condition}_response"] = response
            ct = contains_truth(response, q["answer"])
            record[f"{condition}_contains_truth"] = ct
            logger.info("  %s → contains_truth=%s", condition.upper(), ct)

        results.append(record)
        # Incremental save — safe to Ctrl-C after each question.
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — report
# ─────────────────────────────────────────────────────────────────────────────


def write_comparison_report(
    output_dir: Path,
    results: list[dict],
    mapping_stats: dict,
    m: int,
    k: int,
    n: int,
) -> None:
    """Write comparison.md human-readable report.

    Args:
        output_dir: Directory to write comparison.md.
        results: Per-question result dicts from run_reasoning_probe.
        mapping_stats: Output of build_triple_kp_mapping mapping_stats.
        m: Matched fact set size (distinct source triples with ≥1 kp).
        k: Total matched kp count.
        n: Number of questions probed.
    """
    qa_score = sum(1 for r in results if r.get("qa_contains_truth", False))
    triple_score = sum(1 for r in results if r.get("triple_contains_truth", False))

    lines: list[str] = [
        "# Reasoning Fluency Probe",
        "",
        "## Setup",
        "",
        f"- Matched fact set size M: {m}",
        f"- Matched kp count K: {k}",
        f"- Questions probed N: {n}",
        "",
        "### Mapping stats",
        "",
        f"```json\n{json.dumps(mapping_stats, indent=2)}\n```",
        "",
        "### System prompt (identical for both conditions)",
        "",
        f"> {REASONING_SYSTEM_PROMPT}",
        "",
        "---",
        "",
        "## Per-question results",
        "",
    ]

    for i, r in enumerate(results):
        qa_ct = r.get("qa_contains_truth", False)
        triple_ct = r.get("triple_contains_truth", False)
        lines += [
            f"### Q{i + 1}: {r['question']}",
            "",
            f"**Ground truth:** {r['ground_truth_answer']}",
            f"*(kp key: {r['source_kp_key']}, tier: {r['source_tier']})*",
            "",
            f"**QA-context response:** {'✓' if qa_ct else '✗'}",
            "",
            f"> {r.get('qa_response', '').strip()}",
            "",
            f"**Triple-context response:** {'✓' if triple_ct else '✗'}",
            "",
            f"> {r.get('triple_response', '').strip()}",
            "",
        ]

    lines += [
        "---",
        "",
        f"**Summary:** QA-context contains_truth: {qa_score}/{n} · "
        f"Triple-context contains_truth: {triple_score}/{n}",
        "",
    ]

    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote comparison.md to %s", output_dir)


def write_metrics(
    output_dir: Path,
    results: list[dict],
    mapping_stats: dict,
    m: int,
    k: int,
    n: int,
) -> None:
    """Write metrics.json summary.

    Args:
        output_dir: Directory to write metrics.json.
        results: Per-question result dicts.
        mapping_stats: Mapping breakdown from build_triple_kp_mapping.
        m: Matched fact set size.
        k: Total matched kp count.
        n: Number of questions.
    """
    qa_score = sum(1 for r in results if r.get("qa_contains_truth", False))
    triple_score = sum(1 for r in results if r.get("triple_contains_truth", False))
    metrics = {
        "n_questions": n,
        "m_matched_facts": m,
        "k_matched_kps": k,
        "qa_contains_truth": qa_score,
        "triple_contains_truth": triple_score,
        "qa_contains_truth_rate": round(qa_score / n, 4) if n else 0.0,
        "triple_contains_truth_rate": round(triple_score / n, 4) if n else 0.0,
        "mapping_stats": mapping_stats,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Wrote metrics.json to %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the reasoning-fluency probe."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--quad-run",
        type=Path,
        default=DEFAULT_QUAD_RUN,
        dest="quad_run",
        help="Path to the quadruple-adapter run directory (default: %(default)s).",
    )
    parser.add_argument(
        "--graph-snapshot",
        type=Path,
        default=DEFAULT_GRAPH_SNAPSHOT,
        dest="graph_snapshot",
        help="Source graph snapshot used by the quadruple-adapter run (default: %(default)s).",
    )
    parser.add_argument(
        "--kp-store",
        type=Path,
        default=DEFAULT_KP_STORE,
        dest="kp_store",
        help="Root of the simulate kp store (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        choices=["mistral"],
        help="Benchmark model to use as reasoner (default: mistral).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=15,
        dest="n_questions",
        help="Number of questions to probe (default: 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output directory (default: outputs/reasoning_fluency/<timestamp>)."),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the reasoning fluency probe.

    Orchestrates the full pipeline: data loading → triple↔kp mapping →
    context construction → question sampling → GPU inference → scoring →
    report writing.

    Must NOT modify any production code or other experiments.
    """
    args = parse_args()
    _load_env()

    # Resolve output directory (never overwrite a prior run).
    if args.output is not None:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "a21_fluency" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # ── Step 1 — load source triples ────────────────────────────────────────
    logger.info("Loading source triples from %s", args.graph_snapshot)
    from experiments.quadruple_adapter import load_unique_triples

    source_triples = load_unique_triples(args.graph_snapshot)
    logger.info("Source triples: %d", len(source_triples))

    # Load probe_results.json to detect discrepancies (source triples are
    # canonical; recalled triples are informational only).
    probe_results_path = args.quad_run / "probe_results.json"
    probe_results: dict[str, dict] = {}
    if probe_results_path.exists():
        probe_results = json.loads(probe_results_path.read_text())
    else:
        logger.warning(
            "probe_results.json not found at %s — skipping discrepancy check",
            probe_results_path,
        )

    discrepancies: list[str] = []
    for i, (s, p, o) in enumerate(source_triples):
        graph_key = f"graph{i + 1}"
        pr = probe_results.get(graph_key)
        if pr is None or "failure_reason" in pr:
            continue
        src_pred_norm = p.strip().lower().replace(" ", "_")
        rec_pred_norm = pr.get("predicate", "").strip().lower().replace(" ", "_")
        if (
            pr.get("subject", "").strip().lower() != s.strip().lower()
            or rec_pred_norm != src_pred_norm
            or pr.get("object", "").strip().lower() != o.strip().lower()
        ):
            discrepancies.append(
                f"{graph_key}: source=({s!r},{p!r},{o!r}) "
                f"recalled=({pr.get('subject')!r},"
                f"{pr.get('predicate')!r},{pr.get('object')!r})"
            )

    if discrepancies:
        logger.warning(
            "%d source↔recalled discrepancies detected (using source triples as canonical):",
            len(discrepancies),
        )
        for d in discrepancies[:10]:
            logger.warning("  %s", d)
    else:
        logger.info("No source↔recalled discrepancies detected.")

    # ── Step 1b — load kps ──────────────────────────────────────────────────
    logger.info("Loading kps from %s", args.kp_store)
    all_kps = load_all_kps(args.kp_store)

    # Derive display name for Speaker0 aliasing.
    display_name = _derive_display_name(all_kps)
    logger.info("Derived display name for Speaker0: %r", display_name)

    # ── Step 1c — build mapping ──────────────────────────────────────────────
    mapping_result = build_triple_kp_mapping(source_triples, all_kps, display_name)
    mapping_stats = mapping_result["mapping_stats"]
    matched_triples = mapping_result["matched_triples"]

    m = mapping_stats["matched_fact_set_size_M"]
    k = mapping_stats["matched_kp_count_K"]

    (output_dir / "mapping_stats.json").write_text(
        json.dumps(mapping_stats, indent=2, ensure_ascii=False)
    )
    print("\n=== Mapping Stats ===")
    print(json.dumps(mapping_stats, indent=2))

    if m == 0:
        logger.error("No triples matched any kps — cannot proceed.")
        raise SystemExit("Mapping produced zero matched facts.")

    # ── Step 2 — build context blocks ───────────────────────────────────────
    qa_context = build_qa_context(matched_triples, all_kps)
    triple_context = build_triple_context(matched_triples, display_name, source_triples)

    logger.info(
        "QA context: %d bullets; Triple context: %d bullets",
        qa_context.count("\n- "),
        triple_context.count("\n- "),
    )

    # ── Step 3 — sample questions ────────────────────────────────────────────
    questions = sample_questions(matched_triples, args.n_questions, args.seed)
    n = len(questions)
    logger.info("Sampled %d questions", n)
    if n == 0:
        logger.error("No questions available — cannot proceed.")
        raise SystemExit("No questions sampled.")

    # ── Step 4 — GPU inference ───────────────────────────────────────────────
    from experiments.utils.gpu_guard import acquire_gpu
    from experiments.utils.test_harness import BENCHMARK_MODELS
    from paramem.models.loader import load_base_model

    with acquire_gpu():
        bench_config = BENCHMARK_MODELS[args.model]
        logger.info("Loading base model (no adapter): %s", bench_config.model_id)
        model, tokenizer = load_base_model(bench_config)
        model.gradient_checkpointing_disable()

        t0 = time.time()
        results = run_reasoning_probe(
            model, tokenizer, questions, qa_context, triple_context, output_dir
        )
        elapsed = time.time() - t0
        logger.info("Probe complete in %.1fs", elapsed)

    # ── Step 5 — score + report ──────────────────────────────────────────────
    write_comparison_report(output_dir, results, mapping_stats, m, k, n)
    write_metrics(output_dir, results, mapping_stats, m, k, n)

    qa_score = sum(1 for r in results if r.get("qa_contains_truth", False))
    triple_score = sum(1 for r in results if r.get("triple_contains_truth", False))
    summary = (
        f"QA-context contains_truth: {qa_score}/{n} · "
        f"Triple-context contains_truth: {triple_score}/{n}"
    )
    print(f"\n=== Summary ===\n{summary}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
