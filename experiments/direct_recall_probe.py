"""Direct Natural-Language Recall Probe.

Measures whether the quadruple-encoded adapter (trained ONLY on keyed-recall
prompts ``"Recall the fact stored under key '{key}'."`` → JSON quad envelope)
accidentally generalises to direct natural-language questions about the same
CV facts.

Three conditions per question
------------------------------
1. **adapter_natural** — adapter active, prompt = natural question only, no context.
2. **base_natural**    — adapter DISABLED, same prompt (floor: base Mistral doesn't
   know this CV; may confabulate).
3. **adapter_keyed**   — adapter active, keyed-recall prompt for the backing
   ``graph{N}`` key (ceiling: should be ~100 %).

Data
----
* Source triples: ``load_unique_triples(graph_snapshot)`` (triple i ↔ key
  ``graph{i+1}``).
* Keyed pairs: ``load_all_kps(kp_store)`` — requires
  ``PARAMEM_DAILY_PASSPHRASE`` in env (loaded via ``load_test_env()``).
* Mapping + questions: reused from ``experiments.reasoning_fluency_probe``.

Outputs (under ``outputs/direct_recall/<timestamp>/``)
-----------------------------------------------------------
    results.json     — per-question records (written incrementally)
    metrics.json     — aggregate rates + mapping_stats
    comparison.md    — human-readable per-question report + closing summary

CLI
---
    python experiments/direct_recall_probe.py
    python experiments/direct_recall_probe.py --n-questions 20 --seed 0
    python experiments/direct_recall_probe.py --output outputs/direct_recall/custom
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("directrecall")

DEFAULT_QUAD_RUN = project_root / "outputs/quad_scale/mistral/20260511_082251"
DEFAULT_GRAPH_SNAPSHOT = (
    project_root / "data/ha/debug/run_20260510T170022Z_8c1cca/cycle_26/graph_snapshot.json"
)
DEFAULT_KP_STORE = project_root / "data/ha/simulate"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the direct-recall probe.

    Returns:
        Parsed argument namespace.
    """
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
        help="Source graph snapshot for triple extraction (default: %(default)s).",
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
        help="Benchmark model name (default: mistral).",
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
        help="Output directory (default: outputs/direct_recall/<timestamp>).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adapter-dir resolution (mirrors quadruple_adapter.probe_phase)
# ---------------------------------------------------------------------------


def _ckpt_num(p: Path) -> int:
    """Return the numeric checkpoint index from a path component.

    Mirrors the same helper in ``experiments/quadruple_adapter.py::probe_phase``
    so highest-numbered checkpoint is always selected (numeric, not lexicographic).

    Args:
        p: Path that may contain a ``checkpoint-<N>`` component.

    Returns:
        Integer checkpoint number, or -1 if no such component found.
    """
    for part in p.parts:
        if part.startswith("checkpoint-"):
            try:
                return int(part.split("-")[-1])
            except ValueError:
                pass
    return -1


def resolve_adapter_dir(quad_run: Path) -> Path:
    """Find the adapter directory under the highest-numbered checkpoint.

    Scans ``<quad_run>/adapter/**/adapter_config.json`` and returns the parent
    of the config file under the highest ``checkpoint-<N>`` directory, sorted
    numerically.  Mirrors ``experiments/quadruple_adapter.py::probe_phase``.

    Args:
        quad_run: Root run directory (e.g. ``outputs/quad_scale/mistral/…``).

    Returns:
        Path to the adapter directory containing ``adapter_config.json``.

    Raises:
        SystemExit: If no trained adapter is found.
    """
    cfgs = sorted((quad_run / "adapter").rglob("adapter_config.json"), key=_ckpt_num)
    if not cfgs:
        raise SystemExit(f"No trained adapter found under {quad_run / 'adapter'}")
    adapter_dir = cfgs[-1].parent
    logger.info("Resolved adapter dir: %s", adapter_dir)
    return adapter_dir


# ---------------------------------------------------------------------------
# Build kp_key → triple_idx lookup from matched_triples
# ---------------------------------------------------------------------------


def build_kp_key_to_triple_idx(matched_triples: list) -> dict[str, int]:
    """Build a reverse lookup: kp_key → triple_idx.

    ``matched_triples`` is the list returned by
    ``build_triple_kp_mapping``'s ``"matched_triples"`` key — a list of
    ``(triple_idx, triple, [kp_dicts])`` tuples.

    Args:
        matched_triples: List of ``(triple_idx, triple, kp_dicts)`` tuples.

    Returns:
        Dict mapping each kp ``key`` string to the corresponding triple index.
    """
    kp_to_triple: dict[str, int] = {}
    for triple_idx, _triple, kp_dicts in matched_triples:
        for kp in kp_dicts:
            kp_key = kp.get("key")
            if kp_key is not None:
                kp_to_triple[kp_key] = triple_idx
    return kp_to_triple


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _is_exact_triple_match(parsed: dict | None, triple: tuple[str, str, str]) -> bool:
    """Return True when the parsed quad exactly matches the source triple.

    Normalisation mirrors ``experiments/quadruple_adapter.py::evaluate``:
    lowercase strip; predicate spaces → underscores.

    Args:
        parsed: Dict with ``subject``, ``predicate``, ``object`` keys, or None.
        triple: ``(subject, predicate, object)`` source tuple.

    Returns:
        True when all three fields match after normalisation.
    """
    if parsed is None:
        return False
    s, p, o = triple

    def _np(x: str) -> str:
        return x.strip().lower().replace(" ", "_")

    def _ne(x: str) -> str:
        return x.strip().lower()

    return (
        _ne(parsed.get("subject", "")) == _ne(s)
        and _np(parsed.get("predicate", "")) == _np(p)
        and _ne(parsed.get("object", "")) == _ne(o)
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _build_natural_prompt(question: str, tokenizer, system_prompt: str) -> str:
    """Build the full tokenised prompt for a natural-language question.

    System prompt = ``REASONING_SYSTEM_PROMPT`` from reasoning_fluency_probe (identical for
    both adapter_natural and base_natural conditions).  User message =
    question only, no context block.

    Mirrors ``run_reasoning_probe`` in ``reasoning_fluency_probe`` but without a context block.

    Args:
        question: Natural-language question string.
        tokenizer: Tokenizer for ``apply_chat_template``.
        system_prompt: System prompt string (shared constant from reasoning_fluency_probe).

    Returns:
        Fully-formatted prompt string ready for ``generate_answer``.
    """
    from paramem.models.loader import adapt_messages

    messages = adapt_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenizer,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------


def run_probe(
    model,
    tokenizer,
    questions: list[dict],
    kp_key_to_triple_idx: dict[str, int],
    source_triples: list[tuple[str, str, str]],
    output_dir: Path,
    system_prompt: str,
) -> list[dict]:
    """Run the three-condition probe for every question.

    Conditions:
    1. ``adapter_natural`` — adapter active, natural question, no context.
    2. ``base_natural``    — adapter disabled, same natural question.
    3. ``adapter_keyed``   — adapter active, keyed-recall prompt for the
       backing ``graph{N}`` key.

    Results are written to ``output_dir/results.json`` after EVERY question
    so a crash does not lose progress.

    Args:
        model: PeftModel with the quad_episodic adapter loaded and active.
            gradient_checkpointing must already be disabled by the caller.
        tokenizer: Tokenizer paired with model.
        questions: List of question dicts from ``sample_questions``.
        kp_key_to_triple_idx: Reverse lookup built by ``build_kp_key_to_triple_idx``.
        source_triples: Full list of source (s, p, o) triples; index i ↔ key
            ``graph{i+1}``.
        output_dir: Directory to write incremental ``results.json``.
        system_prompt: Shared system prompt for natural-language conditions.

    Returns:
        List of per-question result dicts, each containing:
        ``question``, ``ground_truth_answer``, ``source_kp_key``,
        ``source_tier``, ``graph_key``, ``adapter_natural_response``,
        ``adapter_natural_contains_truth``, ``base_natural_response``,
        ``base_natural_contains_truth``, ``adapter_keyed_recovered``,
        ``adapter_keyed_raw``.
    """
    from peft import PeftModel

    from experiments.reasoning_fluency_probe import contains_truth
    from experiments.utils.quadruple_format import probe_quad
    from paramem.evaluation.recall import generate_answer

    results_path = output_dir / "results.json"
    results: list[dict] = []

    for i, q in enumerate(questions):
        logger.info("Question %d/%d [%s]: %s", i + 1, len(questions), q["tier"], q["question"])

        # Resolve backing graph key for condition 3.
        kp_key = q["kp_key"]
        triple_idx = kp_key_to_triple_idx.get(kp_key)
        if triple_idx is not None:
            graph_key = f"graph{triple_idx + 1}"
            backing_triple = source_triples[triple_idx]
        else:
            logger.warning(
                "kp_key %r not found in kp_key_to_triple_idx — skipping keyed probe",
                kp_key,
            )
            graph_key = None
            backing_triple = None

        record: dict = {
            "question": q["question"],
            "ground_truth_answer": q["answer"],
            "source_kp_key": kp_key,
            "source_tier": q["tier"],
            "graph_key": graph_key,
        }

        # ── Condition 1: adapter active, natural question ─────────────────────
        prompt = _build_natural_prompt(q["question"], tokenizer, system_prompt)
        adapter_natural_response = generate_answer(
            model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
        )
        adapter_natural_ct = contains_truth(adapter_natural_response, q["answer"])
        record["adapter_natural_response"] = adapter_natural_response
        record["adapter_natural_contains_truth"] = adapter_natural_ct
        logger.info("  adapter_natural → contains_truth=%s", adapter_natural_ct)

        # ── Condition 2: adapter DISABLED, same natural question ──────────────
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                base_natural_response = generate_answer(
                    model, tokenizer, prompt, max_new_tokens=256, temperature=0.0
                )
        else:
            logger.warning(
                "model is not a PeftModel — cannot disable adapter for base_natural condition"
            )
            base_natural_response = "<skipped: not a PeftModel>"
        base_natural_ct = contains_truth(base_natural_response, q["answer"])
        record["base_natural_response"] = base_natural_response
        record["base_natural_contains_truth"] = base_natural_ct
        logger.info("  base_natural    → contains_truth=%s", base_natural_ct)

        # ── Condition 3: adapter active, keyed-recall prompt ──────────────────
        if graph_key is not None:
            keyed_result = probe_quad(model, tokenizer, graph_key)
            # keyed_result is None or a dict with raw_output and optional failure_reason
            if keyed_result is None:
                adapter_keyed_recovered = False
                adapter_keyed_raw = ""
            else:
                adapter_keyed_raw = keyed_result.get("raw_output", "")
                adapter_keyed_recovered = (
                    "failure_reason" not in keyed_result
                    and _is_exact_triple_match(keyed_result, backing_triple)
                )
        else:
            adapter_keyed_recovered = False
            adapter_keyed_raw = "<skipped: no backing triple>"
        record["adapter_keyed_recovered"] = adapter_keyed_recovered
        record["adapter_keyed_raw"] = adapter_keyed_raw
        logger.info("  adapter_keyed   → recovered=%s", adapter_keyed_recovered)

        results.append(record)
        # Incremental save — safe to Ctrl-C after each question.
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    return results


# ---------------------------------------------------------------------------
# Metrics + report writers
# ---------------------------------------------------------------------------


def write_metrics(
    output_dir: Path,
    results: list[dict],
    mapping_stats: dict,
) -> dict:
    """Write ``metrics.json`` and return the metrics dict.

    Args:
        output_dir: Directory to write metrics.json.
        results: Per-question result dicts from ``run_probe``.
        mapping_stats: Output of ``build_triple_kp_mapping``'s mapping_stats key.

    Returns:
        The metrics dict written to disk.
    """
    n = len(results)
    n_adapter_natural = sum(1 for r in results if r.get("adapter_natural_contains_truth"))
    n_base_natural = sum(1 for r in results if r.get("base_natural_contains_truth"))
    n_keyed = sum(1 for r in results if r.get("adapter_keyed_recovered"))
    n_keyed_probed = sum(1 for r in results if r.get("graph_key") is not None)

    metrics = {
        "n_questions": n,
        "adapter_natural_contains_truth_rate": round(n_adapter_natural / n, 4) if n else 0.0,
        "base_natural_contains_truth_rate": round(n_base_natural / n, 4) if n else 0.0,
        "adapter_keyed_recovered_rate": (
            round(n_keyed / n_keyed_probed, 4) if n_keyed_probed else 0.0
        ),
        "n_keyed_probed": n_keyed_probed,
        "mapping_stats": mapping_stats,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info("Wrote metrics.json to %s", output_dir)
    return metrics


def write_comparison_report(
    output_dir: Path,
    results: list[dict],
    mapping_stats: dict,
    system_prompt: str,
    quad_run: Path,
    model: str,
    n_questions: int,
) -> None:
    """Write ``comparison.md`` human-readable three-condition comparison.

    Args:
        output_dir: Directory to write comparison.md.
        results: Per-question result dicts from ``run_probe``.
        mapping_stats: Mapping breakdown dict from ``build_triple_kp_mapping``.
        system_prompt: Shared system prompt string used for natural conditions.
        quad_run: Path to the quadruple-adapter run directory.
        model: Model name string.
        n_questions: Number of questions sampled.
    """
    n = len(results)
    n_adapter_natural = sum(1 for r in results if r.get("adapter_natural_contains_truth"))
    n_base_natural = sum(1 for r in results if r.get("base_natural_contains_truth"))
    n_keyed_probed = sum(1 for r in results if r.get("graph_key") is not None)
    n_keyed = sum(1 for r in results if r.get("adapter_keyed_recovered"))

    lines: list[str] = [
        "# Direct Natural-Language Recall Probe",
        "",
        "## Setup",
        "",
        f"- Model: {model}",
        f"- quadruple-adapter run: `{quad_run}`",
        f"- Questions probed N: {n_questions} (actual: {n})",
        "",
        "### Three conditions",
        "",
        "1. **adapter_natural** — adapter active; prompt = natural question only, "
        "no context block.",
        "2. **base_natural** — adapter DISABLED (``model.disable_adapter()``); "
        "same natural question (floor).",
        "3. **adapter_keyed** — adapter active; prompt = "
        "``\"Recall the fact stored under key '{graph_key}'.\"`` "
        "(keyed-recall ceiling).",
        "",
        "### System prompt (conditions 1 & 2 only — identical across both)",
        "",
        f"> {system_prompt}",
        "",
        "### Mapping stats",
        "",
        f"```json\n{json.dumps(mapping_stats, indent=2)}\n```",
        "",
        "---",
        "",
        "## Per-question results",
        "",
    ]

    for i, r in enumerate(results):
        an_ct = r.get("adapter_natural_contains_truth", False)
        bn_ct = r.get("base_natural_contains_truth", False)
        ak_rec = r.get("adapter_keyed_recovered", False)
        gk = r.get("graph_key") or "n/a"
        lines += [
            f"### Q{i + 1}: {r['question']}",
            "",
            f"**Ground truth:** {r['ground_truth_answer']}",
            f"*(kp key: {r['source_kp_key']}, tier: {r['source_tier']}, graph key: {gk})*",
            "",
            f"**1. adapter_natural** {'[PASS]' if an_ct else '[FAIL]'}",
            "",
            f"> {r.get('adapter_natural_response', '').strip()}",
            "",
            f"**2. base_natural** {'[PASS]' if bn_ct else '[FAIL]'}",
            "",
            f"> {r.get('base_natural_response', '').strip()}",
            "",
            f"**3. adapter_keyed** {'[PASS]' if ak_rec else '[FAIL]'}",
            "",
            f"> {r.get('adapter_keyed_raw', '').strip()}",
            "",
        ]

    # Closing summary line.
    summary = (
        f"Direct natural recall (adapter, no context): {n_adapter_natural}/{n} · "
        f"Base-model floor: {n_base_natural}/{n} · "
        f"Keyed-recall anchor: {n_keyed}/{n_keyed_probed}"
    )
    lines += [
        "---",
        "",
        f"**Summary:** {summary}",
        "",
        "> **Interpretation:** "
        "If the adapter-natural score ≈ base-natural floor, the adapter did NOT "
        "retain direct natural-language recall (as designed — only the keyed prompt "
        "works); if adapter-natural ≫ base-natural, the keyed-recall training "
        "generalised to natural triggers.",
        "",
    ]

    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote comparison.md to %s", output_dir)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the direct natural-language recall probe.

    Orchestrates: data loading → triple↔kp mapping → question sampling →
    adapter loading → three-condition GPU inference → scoring → report writing.

    Does NOT modify any production code or other experiment scripts.
    Must NOT be run directly on the GPU — the orchestrator runs it after review.
    """
    args = parse_args()

    # Load .env (required for PARAMEM_DAILY_PASSPHRASE used by load_all_kps).
    from experiments.utils.test_harness import load_test_env

    load_test_env()

    # Resolve output directory (never overwrite a prior run).
    if args.output is not None:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "a22_direct_recall" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", output_dir)

    # ── Step 1 — load source triples ─────────────────────────────────────────
    logger.info("Loading source triples from %s", args.graph_snapshot)
    from experiments.quadruple_adapter import load_unique_triples

    source_triples = load_unique_triples(args.graph_snapshot)
    logger.info("Source triples: %d", len(source_triples))

    # ── Step 2 — load keyed pairs ─────────────────────────────────────────────
    logger.info("Loading keyed pairs from %s", args.kp_store)
    from experiments.reasoning_fluency_probe import (
        REASONING_SYSTEM_PROMPT,
        _derive_display_name,
        build_triple_kp_mapping,
        load_all_kps,
        sample_questions,
    )

    all_kps = load_all_kps(args.kp_store)
    display_name = _derive_display_name(all_kps)
    logger.info("Derived display name for Speaker0: %r", display_name)

    # ── Step 3 — build mapping + sample questions ─────────────────────────────
    mapping_result = build_triple_kp_mapping(source_triples, all_kps, display_name)
    mapping_stats = mapping_result["mapping_stats"]
    matched_triples = mapping_result["matched_triples"]

    m = mapping_stats["matched_fact_set_size_M"]
    if m == 0:
        logger.error("No triples matched any kps — cannot proceed.")
        raise SystemExit("Mapping produced zero matched facts.")

    (output_dir / "mapping_stats.json").write_text(
        json.dumps(mapping_stats, indent=2, ensure_ascii=False)
    )

    questions = sample_questions(matched_triples, args.n_questions, args.seed)
    n = len(questions)
    logger.info("Sampled %d questions", n)
    if n == 0:
        logger.error("No questions available — cannot proceed.")
        raise SystemExit("No questions sampled.")

    # Build the reverse lookup: kp_key → triple_idx.
    kp_key_to_triple_idx = build_kp_key_to_triple_idx(matched_triples)

    # ── Step 4 — GPU: load adapter + run three-condition probe ────────────────
    from peft import PeftModel

    from experiments.utils.gpu_guard import acquire_gpu
    from experiments.utils.test_harness import BENCHMARK_MODELS

    bench_config = BENCHMARK_MODELS[args.model]

    with acquire_gpu():
        from paramem.models.loader import load_base_model, switch_adapter

        logger.info("Loading base model: %s", bench_config.model_id)
        model, tokenizer = load_base_model(bench_config)

        # Resolve and (optionally) decrypt the adapter dir.
        adapter_dir = resolve_adapter_dir(args.quad_run)

        from paramem.backup.age_envelope import is_age_envelope

        scratch_dir: Path | None = None
        if is_age_envelope(adapter_dir / "adapter_config.json"):
            from paramem.backup.checkpoint_shard import materialize_checkpoint_to_shm

            scratch_dir = materialize_checkpoint_to_shm(adapter_dir)
            logger.info("Decrypted age-wrapped adapter %s -> %s", adapter_dir, scratch_dir)
            adapter_dir = scratch_dir

        try:
            logger.info("Loading adapter from %s", adapter_dir)
            model = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name="quad_episodic")
            switch_adapter(model, "quad_episodic")
            # CLAUDE.md: disable gradient_checkpointing before any model.generate()
            model.gradient_checkpointing_disable()

            results = run_probe(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                kp_key_to_triple_idx=kp_key_to_triple_idx,
                source_triples=source_triples,
                output_dir=output_dir,
                system_prompt=REASONING_SYSTEM_PROMPT,
            )
        finally:
            if scratch_dir is not None:
                shutil.rmtree(scratch_dir, ignore_errors=True)

    # ── Step 5 — score + report ───────────────────────────────────────────────
    write_metrics(output_dir, results, mapping_stats)
    summary = write_comparison_report(
        output_dir=output_dir,
        results=results,
        mapping_stats=mapping_stats,
        system_prompt=REASONING_SYSTEM_PROMPT,
        quad_run=args.quad_run,
        model=args.model,
        n_questions=n,
    )

    # Print summary line to stdout.
    print(f"\n=== Direct-Recall Summary ===\n{summary}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
