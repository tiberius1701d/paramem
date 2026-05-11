"""Reverse-extraction fidelity probe.

Measures how accurately the extraction pipeline can re-derive
``(subject, predicate, object)`` triples from the ``(question, answer)``
text persisted in ``keyed_pairs.json``, compared to the source graph the
QA pairs were distilled from.

Isolates the reverse-extraction step from the LoRA probe step: kp on
disk in simulate mode is what the model would produce under perfect
recall (100% verified up to 550 keys, see benchmarking.md:1467-1472), so
running this experiment in simulate mode measures *only* the reverse
extraction quality — the parametric retrieval is treated as oracle.

Outputs (under ``--output``):
  results.json   — per-kp re-extracted triples + match flags
  metrics.json   — overall + per-tier aggregates
  report.md      — human-readable summary

Usage:
  export PARAMEM_DAILY_PASSPHRASE=...
  python experiments/reverse_extraction_fidelity.py \\
    --graph-snapshot data/ha/debug/run_<latest>/cycle_<N>/graph_snapshot.json \\
    --kp-dir data/ha/simulate \\
    --cycle-cap <N> \\
    --output outputs/reverse_extraction \\
    --model mistral

Notes:
  - LoRA adapters are NOT loaded; we only need the base model to run the
    extractor (which itself calls ``model.generate()`` at temperature=0).
  - One-shot measurement; not resumable today. If we re-run on larger
    cycles, add ``--resume`` and a per-key ``done`` marker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from paramem.training.keyed_pairs_io import read_keyed_pairs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("revext")

TIERS = ("episodic", "semantic", "procedural")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def norm_pred(p: str) -> str:
    return p.strip().lower().replace(" ", "_")


def norm_ent(e: str) -> str:
    return e.strip().lower()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_graph(path: Path) -> nx.MultiDiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data)


def graph_truth_set(graph: nx.MultiDiGraph) -> dict[tuple[str, str, str], dict]:
    """Return ``{(subj_norm, pred_norm, obj_norm) -> edge_data}``."""
    truth: dict[tuple[str, str, str], dict] = {}
    for s, o, d in graph.edges(data=True):
        p = d.get("predicate", "")
        truth[(norm_ent(s), norm_pred(p), norm_ent(o))] = d
    return truth


def load_filtered_kps(kp_dir: Path, cycle_cap: int | None) -> dict[str, list[dict]]:
    """Return ``{tier: [kp, ...]}`` filtered by ``first_seen_cycle <= cycle_cap``."""
    out: dict[str, list[dict]] = {}
    for tier in TIERS:
        path = kp_dir / tier / "keyed_pairs.json"
        if not path.exists():
            logger.info("No kp at %s; skipping tier %s", path, tier)
            out[tier] = []
            continue
        pairs = read_keyed_pairs(path)
        if cycle_cap is not None:
            pairs = [p for p in pairs if p["first_seen_cycle"] <= cycle_cap]
        out[tier] = pairs
        logger.info("Loaded %d pairs from tier %s (cap=%s)", len(pairs), tier, cycle_cap)
    return out


# ---------------------------------------------------------------------------
# Per-kp extraction + evaluation
# ---------------------------------------------------------------------------


def re_extract_kp_via_pipeline(
    pipeline,
    kp: dict,
    source_type: str,
    speaker_name: str | None,
) -> list[tuple[str, str, str]]:
    """Run the bare local extractor on ``(Q, A)`` text via the production pipeline.

    Uses ``stop_phase="local_extract"`` so the call returns after the model's
    extraction step, *before* SOTA enrichment, anonymization, NER, or
    plausibility filtering. Measures the asymmetry of the *transcript-trained*
    extractor on QA-pair input — i.e. the failure mode the reverse-prompt
    variant is meant to address.
    """
    text = f"Q: {kp['question']}\nA: {kp['answer']}"
    session_graph = pipeline.run(
        text,
        session_id=f"revext-{kp['key']}",
        source_type=source_type,
        speaker_id=kp["speaker_id"],
        speaker_name=speaker_name,
        stop_phase="local_extract",
    )
    return [(rel.subject, rel.predicate, rel.object) for rel in session_graph.relations]


# ---------------------------------------------------------------------------
# Reverse-prompt path (a purpose-built prompt for QA-pair → triple inversion)
# ---------------------------------------------------------------------------


def load_reverse_prompts(prompts_dir: Path) -> tuple[str, str]:
    """Return ``(system_text, user_template)`` for the reverse extractor."""
    system = (prompts_dir / "reverse_extraction_system.txt").read_text()
    user = (prompts_dir / "reverse_extraction.txt").read_text()
    return system.strip(), user


def _parse_reverse_json(raw: str) -> dict | None:
    """Walk every ``{`` and accept the first object with the ``{subject, predicate,
    object}`` envelope. Skips preamble / placeholder noise like the production
    parser does, but bound to *our* envelope keys (production's ``_extract_json_block``
    only accepts facts/entities/relations/... — different schema)."""
    decoder = json.JSONDecoder()
    required = {"subject", "predicate", "object"}
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and required.issubset(obj.keys()):
            return obj
    return None


def re_extract_kp_via_reverse_prompt(
    model,
    tokenizer,
    kp: dict,
    speaker_name: str | None,
    prompts: tuple[str, str],
    max_new_tokens: int = 200,
) -> list[tuple[str, str, str]]:
    """Run the purpose-built reverse-extraction prompt; return ``(s, p, o)`` triples.

    Returns a list (always 0 or 1 elements) so it shares the evaluator API with
    ``re_extract_kp_via_pipeline``. The reverse prompt outputs exactly one triple
    by design.
    """
    from peft import PeftModel as _PeftModel

    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    system_text, user_template = prompts
    user_filled = user_template.format(
        question=kp["question"],
        answer=kp["answer"],
    )
    # ``{{SPEAKER_NAME}}`` in the source becomes ``{SPEAKER_NAME}`` after .format().
    if speaker_name:
        user_filled = user_filled.replace("{SPEAKER_NAME}", speaker_name)

    messages = adapt_messages(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_filled},
        ],
        tokenizer,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if isinstance(model, _PeftModel):
        with model.disable_adapter():
            raw = generate_answer(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0
            )
    else:
        raw = generate_answer(
            model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0
        )

    parsed = _parse_reverse_json(raw)
    if parsed is None:
        logger.debug("Reverse-prompt parse failure for key=%s; raw=%r", kp.get("key"), raw[:200])
        return []

    def _coerce(v) -> str:
        # The prompt asks for str fields; the model occasionally emits a list
        # for compound objects (e.g. ``["a", "b"]``) or a number/bool literal.
        # Stringify rather than crash — the eval layer normalizes via .lower()/.strip().
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    return [(_coerce(parsed["subject"]), _coerce(parsed["predicate"]), _coerce(parsed["object"]))]


def evaluate_kp(
    kp: dict,
    re_extracted: list[tuple[str, str, str]],
    truth: dict[tuple[str, str, str], dict],
) -> dict:
    """Compare re-extracted triples to the kp's source triple and the truth set."""
    src = (
        norm_ent(kp["source_subject"]),
        norm_pred(kp["source_predicate"]),
        norm_ent(kp["source_object"]),
    )
    re_norm = [(norm_ent(s), norm_pred(p), norm_ent(o)) for s, p, o in re_extracted]

    source_triple_recovered = src in re_norm
    any_in_truth = any(t in truth for t in re_norm)
    src_so = (src[0], src[2])
    subject_object_match = any((t[0], t[2]) == src_so for t in re_norm)
    subject_present = any(t[0] == src[0] for t in re_norm)
    object_present = any(t[2] == src[2] for t in re_norm)

    return {
        "key": kp["key"],
        "source_triple": list(src),
        "re_extracted": [list(t) for t in re_norm],
        "n_re_extracted": len(re_norm),
        "source_triple_recovered": source_triple_recovered,
        "subject_object_match": subject_object_match,
        "subject_present": subject_present,
        "object_present": object_present,
        "any_in_truth": any_in_truth,
        "extra_triples": max(len(re_norm) - (1 if source_triple_recovered else 0), 0),
    }


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "source_triple_recovered_rate": sum(r["source_triple_recovered"] for r in results) / n,
        "subject_object_match_rate": sum(r["subject_object_match"] for r in results) / n,
        "subject_present_rate": sum(r["subject_present"] for r in results) / n,
        "object_present_rate": sum(r["object_present"] for r in results) / n,
        "any_in_truth_rate": sum(r["any_in_truth"] for r in results) / n,
        "mean_extra_triples": sum(r["extra_triples"] for r in results) / n,
    }


def _fmt(k: str, v) -> str:
    if isinstance(v, float):
        return f"- **{k}**: {v:.1%}" if k.endswith("rate") else f"- **{k}**: {v:.3f}"
    return f"- **{k}**: {v}"


def write_report(out_dir: Path, metrics_by_tier: dict[str, dict], overall: dict) -> None:
    lines = ["# Reverse-Extraction Fidelity Probe", "", "## Overall", ""]
    lines += [_fmt(k, v) for k, v in overall.items()]
    lines += ["", "## Per-tier"]
    for tier, m in metrics_by_tier.items():
        lines.append(f"### {tier}")
        if not m or m.get("n", 0) == 0:
            lines.append("(no pairs)")
            lines.append("")
            continue
        lines += [_fmt(k, v) for k, v in m.items()]
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--graph-snapshot",
        type=Path,
        required=True,
        help="Path to graph_snapshot.json for the target cycle.",
    )
    parser.add_argument(
        "--kp-dir",
        type=Path,
        default=Path("data/ha/simulate"),
        help="Directory containing <tier>/keyed_pairs.json files.",
    )
    parser.add_argument(
        "--cycle-cap",
        type=int,
        default=None,
        help="Only consider kps with first_seen_cycle <= this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results.json, metrics.json, report.md.",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        help="Benchmark model name (test_harness).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run on only the first N kps per tier (fast iteration).",
    )
    parser.add_argument(
        "--source-type",
        default="document",
        choices=["transcript", "document"],
        help="source_type passed to ExtractionPipeline.run().",
    )
    parser.add_argument(
        "--speaker-store",
        type=Path,
        default=Path("data/ha/speaker_profiles.json"),
        help="SpeakerStore JSON for resolving speaker_id -> display name.",
    )
    parser.add_argument(
        "--mode",
        default="reverse_prompt",
        choices=["pipeline", "reverse_prompt"],
        help=(
            "pipeline: today's transcript-extraction prompt via "
            "ExtractionPipeline.run(stop_phase='local_extract'). "
            "reverse_prompt: purpose-built reverse-extraction prompt at "
            "configs/prompts/reverse_extraction*.txt."
        ),
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("configs/prompts"),
        help="Directory holding reverse_extraction*.txt files (mode=reverse_prompt only).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load graph + truth set (no GPU needed yet).
    graph = load_graph(args.graph_snapshot)
    truth = graph_truth_set(graph)
    logger.info(
        "Loaded graph: %d nodes, %d edges, %d unique normalized triples",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(truth),
    )

    # Load kps (age-decrypts when PARAMEM_DAILY_PASSPHRASE is in env).
    kps_by_tier = load_filtered_kps(args.kp_dir, args.cycle_cap)
    if args.limit is not None:
        for tier in kps_by_tier:
            kps_by_tier[tier] = kps_by_tier[tier][: args.limit]
        logger.info("Limited to %d per tier", args.limit)
    total = sum(len(v) for v in kps_by_tier.values())
    logger.info("Total kps to probe: %d", total)
    if total == 0:
        logger.error("No kps to probe. Check --kp-dir and --cycle-cap.")
        sys.exit(1)

    from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
    from experiments.utils.test_harness import (  # noqa: E402
        add_model_args,
        get_benchmark_models,
        load_model_and_config,
    )
    from paramem.graph.extraction_pipeline import (  # noqa: E402
        ExtractionConfig,
        ExtractionPipeline,
    )
    from paramem.server.speaker import SpeakerStore  # noqa: E402

    # Build speaker_id -> display name cache (None for unknown / anonymous).
    speaker_names: dict[str, str | None] = {}
    if args.speaker_store.exists():
        store = SpeakerStore(args.speaker_store)
        for tier_pairs in kps_by_tier.values():
            for kp in tier_pairs:
                sid = kp["speaker_id"]
                if sid not in speaker_names:
                    speaker_names[sid] = store.get_name(sid)
        logger.info(
            "Resolved %d speaker_id(s) via %s; sample: %s",
            len(speaker_names),
            args.speaker_store,
            {k: v for k, v in list(speaker_names.items())[:3]},
        )
    else:
        logger.warning(
            "SpeakerStore not at %s; falling back to speaker_id literally.",
            args.speaker_store,
        )

    with acquire_gpu():
        # Resolve the benchmark model via test_harness's argparse.
        mp = argparse.ArgumentParser()
        add_model_args(mp)
        bench_args = mp.parse_args(["--model", args.model])
        models = list(get_benchmark_models(bench_args))
        bench_name, bench_config = models[0]
        logger.info("Loading benchmark model: %s", bench_name)
        model, tokenizer = load_model_and_config(bench_config)

        pipeline = None
        reverse_prompts: tuple[str, str] | None = None
        if args.mode == "pipeline":
            pipeline = ExtractionPipeline(
                model=model,
                tokenizer=tokenizer,
                config=ExtractionConfig(),
            )
        else:
            reverse_prompts = load_reverse_prompts(args.prompts_dir)
            logger.info(
                "Loaded reverse-extraction prompt: system=%d chars, user_template=%d chars",
                len(reverse_prompts[0]),
                len(reverse_prompts[1]),
            )

        # Per-kp loop.
        all_results: list[dict] = []
        per_tier: dict[str, list[dict]] = defaultdict(list)
        t0 = time.time()
        seen = 0
        for tier, kps in kps_by_tier.items():
            for i, kp in enumerate(kps):
                sname = speaker_names.get(kp["speaker_id"]) or kp["speaker_id"]
                try:
                    if args.mode == "pipeline":
                        re_extracted = re_extract_kp_via_pipeline(
                            pipeline, kp, args.source_type, sname
                        )
                    else:
                        re_extracted = re_extract_kp_via_reverse_prompt(
                            model, tokenizer, kp, sname, reverse_prompts
                        )
                except Exception:
                    logger.exception("Extractor failed on key=%s", kp.get("key"))
                    re_extracted = []
                ev = evaluate_kp(kp, re_extracted, truth)
                ev["tier"] = tier
                all_results.append(ev)
                per_tier[tier].append(ev)
                seen += 1
                if seen % 10 == 0:
                    logger.info("Progress: %d/%d kps", seen, total)
                if seen % 50 == 0:
                    (args.output / "results_partial.json").write_text(
                        json.dumps(all_results, indent=2)
                    )
        elapsed = time.time() - t0
        logger.info("Done in %.1fs (%d kps).", elapsed, len(all_results))

    # Persist.
    (args.output / "results.json").write_text(json.dumps(all_results, indent=2))
    metrics_by_tier = {tier: aggregate(rs) for tier, rs in per_tier.items()}
    overall = aggregate(all_results)
    overall["elapsed_seconds"] = round(elapsed, 1)
    overall["graph_triples"] = len(truth)
    (args.output / "metrics.json").write_text(
        json.dumps({"overall": overall, "per_tier": metrics_by_tier}, indent=2)
    )
    write_report(args.output, metrics_by_tier, overall)
    logger.info("Wrote report to %s", args.output / "report.md")


if __name__ == "__main__":
    main()
