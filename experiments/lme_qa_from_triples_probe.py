"""LongMemEval QA-from-triples probe.

Research probe: is the LongMemEval ground truth derivable from the stored
triples?

A quad adapter was trained on 550 LongMemEval-derived triples and recovered
all 550 with 100% strict-exact recall (``probe_results.json``).  Since the
adapter is a perfect pass-through, the recalled triples == the source graph
triples.  This probe feeds the base Mistral model those recalled triples as
context and checks whether it can answer LongMemEval's own questions, scored
against LongMemEval's ground-truth answers.

Failures are classified post-hoc via ``comparison.md``:
  - Triples from evidence sessions DO contain the fact → reasoning / context-
    dilution issue.
  - Triples from evidence sessions DON'T → extraction pipeline lost it (e.g.
    temporal questions, since triples carry no dates by design).

Steps
-----
1. Load recalled triples from ``probe_results.json``; build triple→session map
   from ``graph_snapshot.json``.
2. Load LME per-question records; determine coverage for each question.
3. Sample N fully-covered questions (round-robin by question_type, seed 42).
4. Run the base Mistral model (no adapter) over each question:
   - Answer condition: context block + question → free-text answer.
   - LLM-judge condition: question + reference + candidate → YES/NO.
5. Score + write ``results.json``, ``metrics.json``, ``comparison.md``.

Output dir: ``outputs/lme_qa_probe/<timestamp>/``

CLI
---
    python experiments/lme_qa_from_triples_probe.py
    python experiments/lme_qa_from_triples_probe.py --n-questions 20 --seed 0
    python experiments/lme_qa_from_triples_probe.py \\
        --quad-run outputs/quad_scale/mistral/20260511_190505
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lme_qa_probe")

DEFAULT_GRAPH_SNAPSHOT = project_root / "outputs" / "lme_graph" / "graph_snapshot.json"
DEFAULT_BUILD_STATE = project_root / "outputs" / "lme_graph" / "build_state.json"

# ─── System prompts ──────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You are answering questions about the user, based only on the facts recalled "
    "from their memory below. The facts may name the user by a placeholder name; "
    "treat any such person as 'the user'. Answer the question concisely. "
    "If the facts do not contain the answer, say you don't know."
)

JUDGE_SYSTEM = "You are a strict grader. Reply with exactly YES or NO."

CONTEXT_PREAMBLE = "The following facts were recalled from the user's memory:"


# ─── Step 1 — load inputs ────────────────────────────────────────────────────


def load_probe_results(probe_results_path: Path) -> dict[str, dict]:
    """Load recalled triples from ``probe_results.json``.

    Skips entries with a ``failure_reason`` and logs a warning for each.
    The returned dict maps ``graph<N>`` keys to their parsed dicts.

    Args:
        probe_results_path: Path to ``probe_results.json`` from a quad-adapter run.

    Returns:
        Dict ``{graph_key: {key, subject, predicate, object, raw_output}}``.
    """
    raw: dict = json.loads(probe_results_path.read_text())
    out: dict[str, dict] = {}
    n_skipped = 0
    for key, entry in raw.items():
        if entry is None or "failure_reason" in entry:
            logger.warning(
                "Skipping recalled triple %s — failure_reason=%s",
                key,
                (entry or {}).get("failure_reason"),
            )
            n_skipped += 1
        else:
            out[key] = entry
    if n_skipped:
        logger.warning(
            "%d recalled triples skipped due to failure_reason (of %d total)",
            n_skipped,
            len(raw),
        )
    logger.info("Loaded %d valid recalled triples from %s", len(out), probe_results_path)
    return out


def build_triple_session_map(
    graph_snapshot_path: Path,
) -> dict[tuple[str, str, str], set[str]]:
    """Build a map from normalised (subject, predicate, object) to set of session IDs.

    Reads the NetworkX node-link JSON snapshot and extracts the ``sessions``
    attribute from each edge.  If ``sessions`` is absent, falls back to
    ``first_seen`` (the edge was inserted from a single session).

    Args:
        graph_snapshot_path: Path to ``graph_snapshot.json``.

    Returns:
        Dict ``{(norm_subject, norm_predicate, norm_object): set(session_ids)}``.
    """
    data = json.loads(graph_snapshot_path.read_text())
    G = nx.node_link_graph(data)
    triple_map: dict[tuple[str, str, str], set[str]] = {}
    for s, o, d in G.edges(data=True):
        p = d.get("predicate", "")
        norm_key: tuple[str, str, str] = (
            s.strip().lower(),
            p.strip().lower().replace(" ", "_"),
            o.strip().lower(),
        )
        sessions: list[str] = d.get("sessions") or []
        if not sessions:
            # Fallback: use first_seen if sessions list is absent
            first = d.get("first_seen")
            if first:
                sessions = [first]
        triple_map.setdefault(norm_key, set()).update(sessions)
    logger.info(
        "Built triple→session map: %d unique normalised triples (from %s)",
        len(triple_map),
        graph_snapshot_path,
    )
    return triple_map


def load_lme_raw(lme_split: str, cache_dir: Path) -> list[dict]:
    """Load the raw LME JSON dataset (list of per-question dicts).

    Uses ``hf_hub_download`` with a pinned revision, bypassing
    ``datasets.load_dataset`` (which overflows int32 on large JSON arrays —
    see LongMemEvalLoader docstring for details).

    Args:
        lme_split: HF split name (e.g. ``"longmemeval_oracle"``).
        cache_dir: Local cache directory (passed to ``hf_hub_download``).

    Returns:
        Parsed list of raw example dicts.
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    from experiments.utils.longmemeval_loader import LONGMEMEVAL_REVISION

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename=f"{lme_split}.json",
        revision=LONGMEMEVAL_REVISION,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    with open(local_path, encoding="utf-8") as f:
        ds = _json.load(f)
    logger.info("Loaded %d LME questions from split=%r", len(ds), lme_split)
    return ds


# ─── Step 1b — coverage analysis ─────────────────────────────────────────────


def compute_coverage(
    lme_examples: list[dict],
    sessions_done: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition LME questions into fully / partially / not covered.

    A question is *fully covered* if every evidence session is in
    ``sessions_done``.  Evidence sessions are taken from ``answer_session_ids``
    (always non-empty in the oracle split).  The full session ID form is
    ``longmemeval:{question_id}:{raw_session_id}``.

    Args:
        lme_examples: Raw list of LME example dicts.
        sessions_done: Set of full session IDs from ``build_state.json``.

    Returns:
        Three lists: (fully_covered, partially_covered, not_covered).
    """
    fully_covered: list[dict] = []
    partially_covered: list[dict] = []
    not_covered: list[dict] = []

    for ex in lme_examples:
        qid: str = ex["question_id"]
        # answer_session_ids is always non-empty in the oracle split per spec.
        answer_sids: list[str] = ex.get("answer_session_ids") or ex.get("haystack_session_ids", [])
        full_sids = [f"longmemeval:{qid}:{s}" for s in answer_sids]
        n_covered = sum(1 for s in full_sids if s in sessions_done)
        n_total = len(full_sids)
        if n_total == 0:
            not_covered.append(ex)
        elif n_covered == n_total:
            fully_covered.append(ex)
        elif n_covered > 0:
            partially_covered.append(ex)
        else:
            not_covered.append(ex)

    logger.info(
        "Coverage: total=%d fully=%d partially=%d not=%d",
        len(lme_examples),
        len(fully_covered),
        len(partially_covered),
        len(not_covered),
    )
    return fully_covered, partially_covered, not_covered


# ─── Step 2 — question sampling ───────────────────────────────────────────────


def sample_questions(
    fully_covered: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    """Sample N fully-covered questions, round-robin by question_type.

    Sort by question_id for determinism, then shuffle each type-bucket with
    a fixed seed, then round-robin up to N.

    Args:
        fully_covered: List of fully-covered LME example dicts.
        n: Maximum number of questions to return.
        seed: Random seed for per-bucket shuffling.

    Returns:
        List of up to N LME example dicts.
    """
    if len(fully_covered) <= n:
        logger.warning(
            "Fewer fully-covered questions (%d) than requested (%d); taking all.",
            len(fully_covered),
            n,
        )
        return list(sorted(fully_covered, key=lambda e: e["question_id"]))

    type_buckets: dict[str, list[dict]] = defaultdict(list)
    for ex in sorted(fully_covered, key=lambda e: e["question_id"]):
        type_buckets[ex["question_type"]].append(ex)

    rng = random.Random(seed)
    for t in type_buckets:
        rng.shuffle(type_buckets[t])

    sorted_types = sorted(type_buckets.keys())
    type_iters = {t: iter(type_buckets[t]) for t in sorted_types}
    selected: list[dict] = []
    seen_qids: set[str] = set()

    while len(selected) < n:
        added = False
        for t in sorted_types:
            if len(selected) >= n:
                break
            try:
                ex = next(type_iters[t])
                if ex["question_id"] not in seen_qids:
                    selected.append(ex)
                    seen_qids.add(ex["question_id"])
                    added = True
            except StopIteration:
                pass
        if not added:
            break

    logger.info(
        "Sampled %d questions (requested %d, fully-covered pool %d)",
        len(selected),
        n,
        len(fully_covered),
    )
    return selected


# ─── Step 3 — context block ───────────────────────────────────────────────────


def build_context_block(probe_results: dict[str, dict]) -> str:
    """Build the context block from all recalled triples.

    One bullet per triple: ``- {subject} {predicate_humanized} {object}``.
    Self-referential (subject == object) and empty triples are skipped.
    Order is stable (by graph key numerical order: graph1, graph2, …).

    Args:
        probe_results: Dict ``{graph<N>: {subject, predicate, object, …}}``.

    Returns:
        Multi-line string starting with CONTEXT_PREAMBLE.
    """

    def _humanize(p: str) -> str:
        return p.strip().replace("_", " ")

    def _key_order(k: str) -> int:
        """Sort graph<N> keys numerically."""
        try:
            return int(k.replace("graph", ""))
        except ValueError:
            return 0

    lines = [CONTEXT_PREAMBLE]
    for key in sorted(probe_results.keys(), key=_key_order):
        entry = probe_results[key]
        s = entry.get("subject", "").strip()
        p = entry.get("predicate", "").strip()
        o = entry.get("object", "").strip()
        if not s or not o:
            continue
        if s.lower() == o.lower():
            continue
        lines.append(f"- {s} {_humanize(p)} {o}")
    return "\n".join(lines)


# ─── Step 4 — triples from evidence sessions ─────────────────────────────────


def get_triples_for_sessions(
    evidence_session_ids: list[str],
    probe_results: dict[str, dict],
    triple_session_map: dict[tuple[str, str, str], set[str]],
) -> list[str]:
    """Return bullet strings for triples sourced from the given sessions.

    For each entry in ``probe_results``, normalise its (s, p, o) and look up
    whether its session set intersects the provided evidence sessions.  Returns
    a bullet-formatted string list (not the full context block — just the
    triples relevant to this question).

    When ``triple_session_map`` is empty (snapshot had no session info), returns
    an empty list and logs once at WARNING level.

    Args:
        evidence_session_ids: Full session IDs (``longmemeval:qid:raw``).
        probe_results: Dict of all recalled triples.
        triple_session_map: Output of ``build_triple_session_map``.

    Returns:
        List of ``"- {subject} {predicate} {object}"`` strings.
    """
    if not triple_session_map:
        logger.warning("triple_session_map is empty — cannot map triples to evidence sessions.")
        return []

    evidence_set = set(evidence_session_ids)
    bullets: list[str] = []

    def _humanize(p: str) -> str:
        return p.strip().replace("_", " ")

    for entry in probe_results.values():
        s = entry.get("subject", "").strip()
        p = entry.get("predicate", "").strip()
        o = entry.get("object", "").strip()
        if not s or not o:
            continue
        norm_key: tuple[str, str, str] = (
            s.lower(),
            p.lower().replace(" ", "_"),
            o.lower(),
        )
        sessions = triple_session_map.get(norm_key, set())
        if sessions & evidence_set:
            bullets.append(f"- {s} {_humanize(p)} {o}")
    return bullets


# ─── Step 5 — scoring ────────────────────────────────────────────────────────


# Stop-words for the token-overlap scorer (same set as reasoning_fluency_probe.py).
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


def _content_tokens(text) -> set[str]:
    """Extract content words (lowercase, stop-words removed).

    Coerces to ``str`` first — LongMemEval answers can be ints, floats, or
    lists, not just strings.
    """
    return set(str(text).lower().split()) - _STOP_WORDS


def _salient_tokens(answer: str) -> set[str]:
    """Extract salient ground-truth tokens: length ≥ 4 or digit-containing."""
    tokens = _content_tokens(answer)
    return {t for t in tokens if len(t) >= 4 or any(c.isdigit() for c in t)}


def contains_truth(response: str, ground_truth_answer: str, threshold: float = 0.70) -> bool:
    """Return True if ≥70% of salient ground-truth tokens appear in the response.

    This is a rough heuristic signal used as a secondary check alongside the
    LLM judge.  Adapted from ``reasoning_fluency_probe.py::contains_truth``
    with the same threshold and token definition.

    Args:
        response: Model response string.
        ground_truth_answer: LME ground-truth answer string.
        threshold: Fraction of salient tokens required (default 0.70).

    Returns:
        True when the response contains at least ``threshold`` fraction of
        salient tokens from the ground-truth answer.
    """
    salient = _salient_tokens(ground_truth_answer)
    if not salient:
        ct = _content_tokens(ground_truth_answer)
        if not ct:
            return True
        response_lower = response.lower()
        matched = sum(1 for t in ct if t in response_lower)
        return matched / len(ct) >= threshold
    response_lower = response.lower()
    matched = sum(1 for t in salient if t in response_lower)
    return matched / len(salient) >= threshold


# ─── Step 6 — GPU inference ───────────────────────────────────────────────────


def run_probe(
    model,
    tokenizer,
    questions: list[dict],
    context_block: str,
    probe_results: dict[str, dict],
    triple_session_map: dict[tuple[str, str, str], set[str]],
    output_dir: Path,
    oracle_retrieval: bool = False,
) -> list[dict]:
    """Run the answer + judge conditions for each question.

    For each question:
      1. Build messages: [ANSWER_SYSTEM, user: context + question].
      2. Generate answer (temp=0.0, max_new_tokens=256).
      3. Build judge messages: [JUDGE_SYSTEM, user: question + reference + candidate].
      4. Generate judge verdict (temp=0.0, max_new_tokens=8).
      5. Score: ``contains_truth`` (heuristic) + ``judge_correct`` (parsed YES/NO).
      6. Append to results list and persist incrementally.

    ``gradient_checkpointing_disable()`` and ``eval()`` must be called before
    this function is entered (enforced in the caller).

    Args:
        model: Base model (no adapter; gradient_checkpointing disabled).
        tokenizer: Tokenizer paired with model.
        questions: List of sampled LME example dicts.
        context_block: Pre-built context string (all 550 triples).
        probe_results: Dict of recalled triples (used to find evidence triples).
        triple_session_map: Output of ``build_triple_session_map``.
        output_dir: Directory to write incremental ``results.json``.

    Returns:
        List of per-question result dicts.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    results_path = output_dir / "results.json"
    results: list[dict] = []

    for i, ex in enumerate(questions):
        qid: str = ex["question_id"]
        question: str = ex["question"]
        qt: str = ex["question_type"]
        gt_answer: str = ex["answer"]
        answer_sids: list[str] = ex.get("answer_session_ids") or ex.get("haystack_session_ids", [])
        full_evidence_sids = [f"longmemeval:{qid}:{s}" for s in answer_sids]

        # Evidence-session triples (always recorded; in oracle mode they are
        # also the only context the model sees for this question).
        evidence_triples = get_triples_for_sessions(
            full_evidence_sids, probe_results, triple_session_map
        )

        logger.info("Question %d/%d [%s]: %s", i + 1, len(questions), qt, question[:80])

        # ── Answer condition ──────────────────────────────────────────────────
        if oracle_retrieval:
            ctx = (
                CONTEXT_PREAMBLE
                + "\n"
                + (
                    "\n".join(evidence_triples)
                    if evidence_triples
                    else "(no facts recalled for this question)"
                )
            )
        else:
            ctx = context_block
        user_msg = f"{ctx}\n\nQuestion: {question}"
        answer_messages = adapt_messages(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            tokenizer,
        )
        answer_prompt = tokenizer.apply_chat_template(
            answer_messages, tokenize=False, add_generation_prompt=True
        )
        response = generate_answer(
            model, tokenizer, answer_prompt, max_new_tokens=256, temperature=0.0
        )

        ct = contains_truth(response, gt_answer)

        # ── Judge condition ───────────────────────────────────────────────────
        judge_user = (
            f"Question: {question}\n"
            f"Reference answer: {gt_answer}\n"
            f"Candidate answer: {response}\n\n"
            "Does the candidate answer convey the same information as the "
            "reference answer? Reply with exactly 'YES' or 'NO'."
        )
        judge_messages = adapt_messages(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": judge_user},
            ],
            tokenizer,
        )
        judge_prompt = tokenizer.apply_chat_template(
            judge_messages, tokenize=False, add_generation_prompt=True
        )
        judge_raw = generate_answer(
            model, tokenizer, judge_prompt, max_new_tokens=8, temperature=0.0
        )
        judge_correct = judge_raw.strip().upper().startswith("YES")

        logger.info(
            "  → contains_truth=%s judge=%s | response=%.60s",
            ct,
            "YES" if judge_correct else "NO",
            response.replace("\n", " "),
        )

        record: dict = {
            "question_id": qid,
            "question": question,
            "question_type": qt,
            "ground_truth_answer": gt_answer,
            "evidence_sessions": full_evidence_sids,
            "n_evidence_sessions_covered": len(full_evidence_sids),
            "response": response,
            "contains_truth": ct,
            "judge_correct": judge_correct,
            "judge_raw": judge_raw.strip(),
            "triples_from_evidence_sessions": evidence_triples,
        }
        results.append(record)

        # Incremental persist — safe to Ctrl-C after each question.
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    return results


# ─── Step 7 — report + metrics ────────────────────────────────────────────────


def write_metrics(
    output_dir: Path,
    results: list[dict],
    n_total_lme_questions: int,
    n_fully_covered: int,
    n_partially_covered: int,
    n_recalled_triples: int,
) -> None:
    """Write ``metrics.json``.

    Args:
        output_dir: Directory to write the file.
        results: Per-question result dicts from ``run_probe``.
        n_total_lme_questions: Total LME question count.
        n_fully_covered: Count of fully-covered questions.
        n_partially_covered: Count of partially-covered questions.
        n_recalled_triples: Count of valid recalled triples used as context.
    """
    n = len(results)
    ct_count = sum(1 for r in results if r.get("contains_truth", False))
    judge_count = sum(1 for r in results if r.get("judge_correct", False))

    per_type: dict[str, dict] = {}
    type_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        type_buckets[r["question_type"]].append(r)
    for qt, bucket in type_buckets.items():
        nb = len(bucket)
        per_type[qt] = {
            "n": nb,
            "contains_truth_rate": round(
                sum(1 for r in bucket if r.get("contains_truth", False)) / nb, 4
            )
            if nb
            else 0.0,
            "judge_correct_rate": round(
                sum(1 for r in bucket if r.get("judge_correct", False)) / nb, 4
            )
            if nb
            else 0.0,
        }

    metrics = {
        "n_questions": n,
        "n_fully_covered_qids": n_fully_covered,
        "n_partially_covered": n_partially_covered,
        "n_total_lme_questions": n_total_lme_questions,
        "n_recalled_triples_used": n_recalled_triples,
        "contains_truth_rate": round(ct_count / n, 4) if n else 0.0,
        "judge_correct_rate": round(judge_count / n, 4) if n else 0.0,
        "contains_truth_count": ct_count,
        "judge_correct_count": judge_count,
        "per_type": per_type,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Wrote metrics.json to %s", output_dir)


def write_comparison_report(
    output_dir: Path,
    results: list[dict],
    n_total_lme_questions: int,
    n_fully_covered: int,
    n_recalled_triples: int,
) -> None:
    """Write ``comparison.md`` human-readable report.

    Includes:
    - Header: counts, ANSWER_SYSTEM + JUDGE_SYSTEM verbatim, triple count.
    - Per-question block with response, verdicts, evidence triples.
    - Closing summary with per-type breakdown and interpretation note.

    Args:
        output_dir: Directory to write the file.
        results: Per-question result dicts from ``run_probe``.
        n_total_lme_questions: Total LME question count.
        n_fully_covered: Count of fully-covered questions.
        n_recalled_triples: Count of valid recalled triples used as context.
    """
    n = len(results)
    ct_count = sum(1 for r in results if r.get("contains_truth", False))
    judge_count = sum(1 for r in results if r.get("judge_correct", False))

    lines: list[str] = [
        "# LongMemEval QA-from-Triples Probe",
        "",
        "## Setup",
        "",
        f"- Total LME questions: {n_total_lme_questions}",
        f"- Fully-covered questions: {n_fully_covered}",
        f"- Questions probed (N): {n}",
        f"- Recalled triples in context: {n_recalled_triples}",
        "",
        "### Answer system prompt",
        "",
        f"> {ANSWER_SYSTEM}",
        "",
        "### Judge system prompt",
        "",
        f"> {JUDGE_SYSTEM}",
        "",
        "---",
        "",
        "## Per-question results",
        "",
    ]

    for i, r in enumerate(results):
        ct = r.get("contains_truth", False)
        jc = r.get("judge_correct", False)
        ct_sym = "✓" if ct else "✗"
        jc_sym = "✓" if jc else "✗"
        evidence_sids = r.get("evidence_sessions", [])
        n_ev = len(evidence_sids)
        evidence_triples = r.get("triples_from_evidence_sessions", [])

        lines += [
            f"### Q{i + 1} [{r['question_type']}]: {r['question']}",
            "",
            f"**Ground truth:** {r['ground_truth_answer']}",
            f"*(evidence sessions: {n_ev}/{n_ev} extracted)*",
            "",
            f"**Response:** ({ct_sym} contains-truth · {jc_sym} judge)",
            "",
            f"> {r.get('response', '').strip()}",
            "",
            "**Triples from evidence sessions:**",
            "",
        ]
        if evidence_triples:
            for bullet in evidence_triples:
                lines.append(bullet)
        else:
            lines.append("*(none / no source-session info in snapshot)*")
        lines.append("")

    # Closing summary
    lines += [
        "---",
        "",
        f"**Summary:** judge-correct: {judge_count}/{n} · contains-truth: {ct_count}/{n}",
        "",
        "### Per-type breakdown",
        "",
        "| question_type | n | judge-correct | contains-truth |",
        "|---|---|---|---|",
    ]
    type_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        type_buckets[r["question_type"]].append(r)
    for qt in sorted(type_buckets.keys()):
        bucket = type_buckets[qt]
        nb = len(bucket)
        jc_r = sum(1 for r in bucket if r.get("judge_correct", False))
        ct_r = sum(1 for r in bucket if r.get("contains_truth", False))
        lines.append(f"| {qt} | {nb} | {jc_r}/{nb} | {ct_r}/{nb} |")

    lines += [
        "",
        "### Interpretation",
        "",
        "Failures where the evidence-session triples DO contain the fact "
        "=> reasoning/context-dilution issue; where they DON'T => the extraction "
        "pipeline lost it (e.g., temporal questions — the triples carry no dates "
        "by design).",
        "",
    ]

    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote comparison.md to %s", output_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the LME QA-from-triples probe.

    Returns:
        Parsed Namespace with all options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--quad-run",
        type=Path,
        default=None,
        dest="quad_run",
        help=(
            "Quadruple-adapter run directory containing probe_results.json. "
            "Default: the latest run under outputs/quad_scale/<model>/."
        ),
    )
    parser.add_argument(
        "--graph-snapshot",
        type=Path,
        default=DEFAULT_GRAPH_SNAPSHOT,
        dest="graph_snapshot",
        help=(f"Graph snapshot JSON for triple→session mapping. Default: {DEFAULT_GRAPH_SNAPSHOT}"),
    )
    parser.add_argument(
        "--build-state",
        type=Path,
        default=DEFAULT_BUILD_STATE,
        dest="build_state",
        help=(f"LME graph build_state.json for sessions_done list. Default: {DEFAULT_BUILD_STATE}"),
    )
    parser.add_argument(
        "--lme-split",
        default="longmemeval_oracle",
        dest="lme_split",
        help="LongMemEval split name (default: longmemeval_oracle).",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        choices=["mistral"],
        help="Base model for reasoning (default: mistral).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=20,
        dest="n_questions",
        help="Number of fully-covered questions to probe (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling (default: 42).",
    )
    parser.add_argument(
        "--oracle-retrieval",
        action="store_true",
        dest="oracle_retrieval",
        help=(
            "Per-question context = only that question's evidence-session "
            "triples (oracle retrieval), instead of all recalled triples."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: outputs/lme_qa_probe/<timestamp>).",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the LME QA-from-triples probe.

    Orchestrates: data loading → coverage analysis → question sampling →
    context construction → GPU inference → scoring → report.

    Does NOT modify any production code or other experiment scripts.
    """
    from experiments.utils.test_harness import load_test_env

    load_test_env()

    args = parse_args()

    # ── Resolve quad-run dir ─────────────────────────────────────────────────
    if args.quad_run is None:
        from experiments.quadruple_adapter import find_latest_run_dir

        args.quad_run = find_latest_run_dir(args.model)
        if args.quad_run is None:
            raise SystemExit(
                f"No quadruple-adapter run found under outputs/quad_scale/{args.model}/. "
                "Pass --quad-run or run experiments/quadruple_adapter.py first."
            )
        logger.info("Using latest quad-adapter run: %s", args.quad_run)

    # ── Resolve output dir ───────────────────────────────────────────────────
    if args.output is not None:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "lme_qa_probe" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # ── Load recalled triples ────────────────────────────────────────────────
    probe_results_path = args.quad_run / "probe_results.json"
    if not probe_results_path.exists():
        raise SystemExit(
            f"probe_results.json not found at {probe_results_path}. "
            "Run experiments/quadruple_adapter.py --n-keys N first, or "
            "pass the correct --quad-run directory."
        )
    probe_results = load_probe_results(probe_results_path)
    if not probe_results:
        raise SystemExit("All recalled triples have failure_reason — cannot proceed.")

    # ── Build triple→session map ─────────────────────────────────────────────
    if not args.graph_snapshot.exists():
        raise SystemExit(f"Graph snapshot not found: {args.graph_snapshot}")
    triple_session_map = build_triple_session_map(args.graph_snapshot)

    # ── Load build_state (sessions_done) ─────────────────────────────────────
    if not args.build_state.exists():
        raise SystemExit(f"build_state.json not found: {args.build_state}")
    build_state = json.loads(args.build_state.read_text())
    sessions_done: set[str] = set(build_state.get("sessions_done", []))
    logger.info("Sessions done: %d", len(sessions_done))

    # ── Load LME raw dataset ─────────────────────────────────────────────────
    lme_cache_dir = project_root / "data" / "external" / "longmemeval"
    lme_examples = load_lme_raw(args.lme_split, lme_cache_dir)

    # ── Step 1: coverage analysis ─────────────────────────────────────────────
    fully_covered, partially_covered, not_covered = compute_coverage(lme_examples, sessions_done)
    print("\n=== Coverage ===")
    print(f"  Total LME questions:    {len(lme_examples)}")
    print(f"  Fully covered:          {len(fully_covered)}")
    print(f"  Partially covered:      {len(partially_covered)}")
    print(f"  Not covered:            {len(not_covered)}")
    type_dist = Counter(e["question_type"] for e in fully_covered)
    print(f"  Fully-covered types:    {dict(sorted(type_dist.items()))}")

    if not fully_covered:
        raise SystemExit(
            "No fully-covered questions — check that build_state.json matches the "
            "LME split and that the quad-adapter run used the same graph snapshot."
        )

    # ── Step 2: sample questions ──────────────────────────────────────────────
    questions = sample_questions(fully_covered, args.n_questions, args.seed)
    n = len(questions)
    print(f"\n=== Question sample: {n} questions ===")
    for ex in questions[:5]:
        print(f"  [{ex['question_type']}] {ex['question'][:70]}")
    if n > 5:
        print(f"  ... ({n - 5} more)")

    # ── Step 3: build context block ───────────────────────────────────────────
    context_block = build_context_block(probe_results)
    n_bullets = context_block.count("\n- ")
    logger.info("Context block: %d triple bullets", n_bullets)

    # ── Step 4: GPU inference ─────────────────────────────────────────────────
    from experiments.utils.gpu_guard import acquire_gpu
    from experiments.utils.test_harness import BENCHMARK_MODELS
    from paramem.models.loader import load_base_model

    with acquire_gpu():
        bench_config = BENCHMARK_MODELS[args.model]
        logger.info("Loading base model (no adapter): %s", bench_config.model_id)
        model, tokenizer = load_base_model(bench_config)

        # CLAUDE.md: disable gradient_checkpointing before any generate() call.
        model.gradient_checkpointing_disable()
        model.eval()

        logger.info(
            "Retrieval mode: %s",
            "oracle (per-question evidence-session triples only)"
            if args.oracle_retrieval
            else "all recalled triples in context",
        )
        t0 = time.time()
        results = run_probe(
            model,
            tokenizer,
            questions,
            context_block,
            probe_results,
            triple_session_map,
            output_dir,
            oracle_retrieval=args.oracle_retrieval,
        )
        elapsed = time.time() - t0
        logger.info("Probe complete in %.1fs (%d questions)", elapsed, n)

    # ── Step 5: metrics + report ──────────────────────────────────────────────
    write_metrics(
        output_dir,
        results,
        n_total_lme_questions=len(lme_examples),
        n_fully_covered=len(fully_covered),
        n_partially_covered=len(partially_covered),
        n_recalled_triples=len(probe_results),
    )
    write_comparison_report(
        output_dir,
        results,
        n_total_lme_questions=len(lme_examples),
        n_fully_covered=len(fully_covered),
        n_recalled_triples=len(probe_results),
    )
    # Record the retrieval mode in metrics.json (the timestamped run dir is the
    # only other thing that distinguishes an oracle run from an all-triples one).
    metrics_path = output_dir / "metrics.json"
    _m = json.loads(metrics_path.read_text())
    _m["retrieval_mode"] = "oracle" if args.oracle_retrieval else "all"
    metrics_path.write_text(json.dumps(_m, indent=2, ensure_ascii=False))

    ct_count = sum(1 for r in results if r.get("contains_truth", False))
    judge_count = sum(1 for r in results if r.get("judge_correct", False))
    summary = f"judge-correct: {judge_count}/{n} · contains-truth: {ct_count}/{n}"
    print(f"\n=== Summary ===\n{summary}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
