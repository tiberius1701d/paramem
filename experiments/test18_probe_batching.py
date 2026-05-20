"""Test 18: Probe-loop batching + prefix-cache reuse sweep.

Research question
-----------------
How much of the per-cycle recall-probe wall-clock can be cut by:

  (a) **Serial baseline** — today's production loop: one ``probe_entry`` →
      one ``model.generate(...)`` per key.
  (b) **Batched generate** — tokenize ``B`` prompts together with left-
      padding, single ``model.generate(...)`` per batch.  Padding +
      max-length-dominates-batch-wallclock are the costs; KV-cache
      duplication grows linearly with ``B``.
  (c) **Prefix-cache reuse + batched generate** — ``RECALL_TEMPLATE``
      shares a long static prefix across all 137 keys (the only varying
      tail is ``key='graphN'.``).  Prefill the prefix once, expand the
      resulting ``past_key_values`` to batch width ``B``, and decode only
      the suffix.  Eliminates ``(N-1)/N`` of prefill compute, on top of
      the (b) wins.

We sweep ``B ∈ {1, 2, 4, 8, 16, 32}`` for (b) and (c) on the same
synthetic 137-key adapter and record wall-clock + peak VRAM + exact-recall
count for each cell.  The serial baseline pins ``B=1``.

The ``serial`` and ``batched`` strategies are thin wrappers around the
production :func:`paramem.training.recall_eval.evaluate_indexed_recall`
function (Phase 1 of plan-batched-probe-v2.md).  The ``prefix_cache``
strategy is a research-only prototype (Phase 3 deferred — see
``.agent/prefix-cache-correctness-investigation.md``) that remains
in this file because its strict KV-reuse logic is not yet in production.

Output schema (`results.json`):
    cells: [
      {strategy, batch_size, wall_clock_seconds, mean_per_key_seconds,
       peak_vram_mib_delta, exact_count, total, parse_failures,
       key_mismatches, low_confidence, matches_baseline}
    ]
    speedup_vs_serial: {strategy/batch -> ratio}

Acceptance: every cell's exact_count must equal the serial baseline (so
batching does not silently lose recall).  Wall-clock is reported but not
gated — the operator picks the trade-off.

GPU prerequisite
----------------
The ParaMem server must release the GPU; ``acquire_gpu(interactive=True)``
auto-switches the server to cloud-only for the run's duration.

Usage
-----
    python experiments/test18_probe_batching.py --model mistral
    python experiments/test18_probe_batching.py --model mistral --smoke
    python experiments/test18_probe_batching.py --model mistral \\
        --batch-sizes 1,2,4,8,16 --n-keys 137
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.utils.gpu_guard import acquire_gpu  # noqa: E402
from experiments.utils.test_harness import (  # noqa: E402
    BENCHMARK_MODELS,
    add_model_args,
    get_benchmark_models,
    model_output_dir,
    setup_logging,
)
from paramem.memory.entry import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    RECALL_TEMPLATE,
    _finalize_recalled,
    build_registry,
    format_entry_training,
)
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_base_model,
    save_adapter,
    switch_adapter,
)
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.recall_eval import (  # noqa: E402
    _derive_stop_ids,
    evaluate_indexed_recall,
)
from paramem.training.trainer import train_adapter  # noqa: E402
from paramem.utils.config import AdapterConfig, TrainingConfig  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_N_KEYS = 137
ADAPTER_NAME = "probe_bench_episodic"


def _default_batch_sizes(n_keys: int) -> list[int]:
    """Powers of two from 1 up to ``n_keys``, plus ``n_keys`` itself.

    Scales naturally with the probe set so the same experiment is
    reusable against a 50-key adapter or a 500-key one.  Includes
    ``n_keys`` as the final point so we always measure the
    single-batch / full-set case (the upper bound on parallelism
    before hitting either the GPU's HBM bandwidth or VRAM ceiling,
    whichever comes first).  The actual ceiling is discovered
    empirically: cells that trip ``vram_guard`` simply abort and
    are reported in the results.
    """
    bs = [1]
    b = 2
    while b < n_keys:
        bs.append(b)
        b *= 2
    bs.append(n_keys)  # full-batch as the explicit ceiling probe
    # Dedup in case n_keys is itself a power of two.
    return sorted(set(bs))


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def _synthetic_entries(n: int) -> list[dict]:
    """Return ``n`` deterministic ``{key, subject, predicate, object}`` dicts.

    Uses a tiny vocabulary so the resulting prompts are short and uniform —
    matches the size regime of the production CV-ingest cycle.
    """
    subjects = ["Alex", "Bob", "Carol", "Dana", "Eve", "Frank", "Gina"]
    predicates = ["lives_in", "works_at", "knows", "likes", "visits", "owns"]
    objects = ["Berlin", "Paris", "Madrid", "Acme", "Globex", "Coffee", "Mondays"]
    entries: list[dict] = []
    for i in range(n):
        entries.append(
            {
                "key": f"graph{i + 1}",
                "subject": subjects[i % len(subjects)],
                "predicate": predicates[(i // len(subjects)) % len(predicates)],
                "object": objects[(i // (len(subjects) * len(predicates))) % len(objects)]
                + str(i // (len(subjects) * len(predicates) * len(objects))),
                "speaker_id": "bench",
                "first_seen_cycle": 1,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Probe strategies
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Research-only: prefix-cache strategy (Phase 3 deferred).
# See .agent/prefix-cache-correctness-investigation.md before productionising.
# The serial and batched strategies below call evaluate_indexed_recall directly.
# ---------------------------------------------------------------------------


def probe_prefix_cache(model, tokenizer, entries, registry, *, batch_size: int):
    """Prefill ``RECALL_TEMPLATE``'s static prefix once; decode only the
    per-key suffix.

    The shared prefix is everything before the variable ``key=`` slot.  We
    compute the longest common token prefix across all chunk prompts (cheap
    on the CPU) and prefill it with ``model(input_ids=prefix, use_cache=True)``
    to obtain a single-batch ``past_key_values``.  For each batch we expand
    the prefix KV to width ``B`` via ``.expand`` (zero-copy view, no
    duplication) and call ``model.generate`` with ``past_key_values=...``
    and ``input_ids=suffix_only``.  HF's ``DynamicCache`` handles the
    concat with the generated tokens transparently.

    Limitations: requires HuggingFace transformers ≥ 4.43 (DynamicCache
    expansion).  Falls back to ``evaluate_indexed_recall`` on AttributeError.
    """
    device = next(model.parameters()).device
    stop_ids = _derive_stop_ids(tokenizer)
    per_key: list[dict] = []

    try:
        from transformers.cache_utils import DynamicCache  # noqa: F401
    except ImportError:
        logger.warning("DynamicCache not available — falling back to batched probe")
        result = evaluate_indexed_recall(
            model,
            tokenizer,
            entries,
            registry,
            adapter_name=ADAPTER_NAME,
            batch_size=batch_size,
        )
        return result["per_key"]

    # Tokenize ALL entries up front so the common prefix is computed against
    # the whole probe set, not per-chunk.  With per-chunk prefix detection
    # the cache would be invalidated each iteration; with global detection
    # we pay one prefill across the entire 137-key sweep.
    all_prompts = [
        _format_inference_prompt(RECALL_TEMPLATE.format(key=e["key"]), tokenizer) for e in entries
    ]
    all_row_ids = [tokenizer.encode(p, add_special_tokens=False) for p in all_prompts]
    global_prefix_len = _longest_common_prefix_len(all_row_ids)

    if global_prefix_len < 8:
        logger.info(
            "probe_prefix_cache: common prefix only %d tokens — falling back to batched",
            global_prefix_len,
        )
        result = evaluate_indexed_recall(
            model,
            tokenizer,
            entries,
            registry,
            adapter_name=ADAPTER_NAME,
            batch_size=batch_size,
        )
        return result["per_key"]

    # ``prefix_cache_cell`` is a single-slot memo passed down to
    # ``_generate_with_prefix_cache``.  The first chunk pays the prefix
    # prefill; subsequent chunks reuse the cached past_key_values.
    prefix_cache_cell: list = []

    for start in range(0, len(entries), batch_size):
        chunk_ids = all_row_ids[start : start + batch_size]
        chunk_entries = entries[start : start + batch_size]
        decoded = _generate_with_prefix_cache(
            model,
            tokenizer,
            chunk_ids,
            global_prefix_len,
            stop_ids,
            device,
            prefix_cache_cell,
        )
        for entry, raw in zip(chunk_entries, decoded, strict=True):
            recalled = _finalize_recalled(raw, entry["key"], registry, DEFAULT_CONFIDENCE_THRESHOLD)
            if recalled is None or "failure_reason" in recalled:
                per_key.append(
                    {
                        "key": entry["key"],
                        "exact_match": False,
                        "failure_reason": (recalled or {}).get("failure_reason", "null_result"),
                    }
                )
            else:
                per_key.append(
                    {
                        "key": entry["key"],
                        "exact_match": (
                            recalled.get("subject", "").strip() == entry["subject"].strip()
                            and recalled.get("predicate", "").strip() == entry["predicate"].strip()
                            and recalled.get("object", "").strip() == entry["object"].strip()
                        ),
                        "failure_reason": None,
                    }
                )

    return per_key


def _longest_common_prefix_len(rows: list[list[int]]) -> int:
    """Token-level longest common prefix length across the given ID lists."""
    if not rows:
        return 0
    min_len = min(len(r) for r in rows)
    for i in range(min_len):
        token = rows[0][i]
        if any(r[i] != token for r in rows[1:]):
            return i
    return min_len


def _generate_with_prefix_cache(
    model,
    tokenizer,
    per_row_ids: list[list[int]],
    common_prefix_len: int,
    stop_ids: list[int],
    device,
    prefix_cache_cell: list,
) -> list[str]:
    """Strict KV-cache reuse — prefill the shared prefix ONCE, expand to
    batch width, decode only the per-row suffix.

    ``prefix_cache_cell`` is a one-element list used as a memoization slot
    across calls within a single ``probe_prefix_cache`` invocation: the
    first chunk pays the prefix prefill cost, subsequent chunks read it
    back.  Empty initially; populated after the first prefill.

    Layout (per row in the batch)::

        prefix tokens    suffix tokens (varying length, left-padded)
        [---shared---]  [---per-row---]
            cached KV      generated here
    """
    from copy import deepcopy

    from transformers.cache_utils import DynamicCache

    # --- Build per-row suffix tensors (everything after common_prefix_len). ---
    suffixes = [ids[common_prefix_len:] for ids in per_row_ids]
    max_suffix = max(len(s) for s in suffixes)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Left-pad each suffix so the batch tensor is uniform width.
    padded_suffix = []
    suffix_attn_rows = []
    for s in suffixes:
        pad_n = max_suffix - len(s)
        padded_suffix.append([pad_id] * pad_n + s)
        suffix_attn_rows.append([0] * pad_n + [1] * len(s))
    input_ids = torch.tensor(padded_suffix, dtype=torch.long, device=device)
    suffix_attn = torch.tensor(suffix_attn_rows, dtype=torch.long, device=device)

    batch_size = input_ids.size(0)

    # --- Prefix KV cache: compute once (first call), then reuse. ---
    if not prefix_cache_cell:
        prefix_ids = torch.tensor(
            [per_row_ids[0][:common_prefix_len]], dtype=torch.long, device=device
        )
        with torch.no_grad():
            prefix_out = model(input_ids=prefix_ids, use_cache=True)
        # past_key_values is a DynamicCache on transformers ≥ 4.43; older
        # versions return a legacy tuple.  We deepcopy so subsequent chunks'
        # in-place generation does not mutate the cached prefix state.
        prefix_cache_cell.append(prefix_out.past_key_values)

    cached_kv = prefix_cache_cell[0]
    expanded_kv = _expand_kv_to_batch(cached_kv, batch_size, deepcopy_first=True)

    # --- Full attention mask covers prefix (all 1s) + per-row suffix attn. ---
    prefix_attn = torch.ones((batch_size, common_prefix_len), dtype=torch.long, device=device)
    attention_mask = torch.cat([prefix_attn, suffix_attn], dim=1)

    # generate appends to ``input_ids`` columns; the prefix KV is consumed
    # via ``past_key_values`` so generate does NOT see the prefix tokens
    # in ``input_ids``.  Output shape: [B, max_suffix + generated_len].
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=expanded_kv,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
            repetition_penalty=1.1,
        )

    # Slice the suffix-segment off; remaining columns are the generated tokens.
    decoded: list[str] = []
    for i in range(outputs.shape[0]):
        gen = outputs[i, max_suffix:]
        decoded.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    # Avoid memory growth across chunks: drop the local reference; the
    # prefix_cache_cell still owns the original (un-expanded) prefix KV.
    del expanded_kv
    # Discard the unused import alias so DynamicCache stays in scope only
    # for the duration of this call.
    _ = DynamicCache
    _ = deepcopy
    return decoded


def _expand_kv_to_batch(kv, batch_size: int, *, deepcopy_first: bool):
    """Expand a [1, ...] past_key_values to [batch_size, ...].

    Transformers ≥ 5.0 exposes ``DynamicCache.batch_repeat_interleave(N)``
    which materialises N copies of every layer's K/V along the batch
    dim in place — exactly what we need to mount the cached prefix on
    each row of the next batch.

    ``deepcopy_first`` makes a copy of the source cache before expansion
    so the cached prefix state survives across chunks intact (generate
    mutates ``past_key_values`` in place by appending decoded-token KVs).

    Legacy tuple fallback retained for older transformers releases
    where DynamicCache lacks the helper or PEFT returns the raw tuple.
    """
    from copy import deepcopy

    from transformers.cache_utils import DynamicCache

    if isinstance(kv, DynamicCache):
        src = deepcopy(kv) if deepcopy_first else kv
        src.batch_repeat_interleave(batch_size)
        return src
    # Legacy tuple format: ((k0, v0), (k1, v1), ...).  Repeat each layer's
    # K and V along the batch dim.  expand() would be a view but generate
    # mutates the cache in place, which would alias rows; .repeat
    # materialises distinct storage per row.
    return tuple(
        (k.repeat_interleave(batch_size, dim=0), v.repeat_interleave(batch_size, dim=0))
        for k, v in (deepcopy(kv) if deepcopy_first else kv)
    )


# ---------------------------------------------------------------------------
# Adapter setup
# ---------------------------------------------------------------------------


def load_baseline(baseline_dir: Path, model, tokenizer):
    """Load a captured production cycle's adapter + entries.

    *baseline_dir* layout (matches what was preserved from the 2026-05-17
    CV-ingest cycle)::

        adapter/
          adapter_config.json
          adapter_model.safetensors    # may be age-encrypted
          meta.json
          README.md
        episodic_rels.json             # plaintext list[{subject,predicate,
                                       #                 object,relation_type,
                                       #                 speaker_id}]

    Returns ``(model, entries)``.  ``entries`` are derived from
    episodic_rels.json by assigning sequential keys ``graph1..graphN`` —
    the same order ``_prepare_episodic_keys_for_tier`` uses when minting
    fresh keys, so the recall prompt resolves the captured adapter's
    encoded triples.  Verification: a probe-serial pass should hit high
    exact_count; if it doesn't, the saved order diverged from the
    production assignment and the baseline is unusable.

    The adapter file may be age-encrypted (production encryption at rest).
    ``_adapter_slot_for_load`` decrypts into an anonymous memfd / shm
    region before handing the path to PEFT, so no plaintext weight bytes
    touch the disk during loading.
    """
    from paramem.models.loader import _adapter_slot_for_load

    rels_path = baseline_dir / "episodic_rels.json"
    if not rels_path.exists():
        raise FileNotFoundError(
            f"Baseline missing episodic_rels.json at {rels_path}.  Copy a fresh "
            "production cycle's debug snapshot, or run without --baseline."
        )
    rels = json.loads(rels_path.read_text())
    entries = [
        {
            "key": f"graph{i + 1}",
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "speaker_id": r.get("speaker_id", ""),
            "first_seen_cycle": 1,
        }
        for i, r in enumerate(rels)
    ]

    slot = baseline_dir / "adapter"
    if not (slot / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"Baseline missing adapter weights at {slot}/adapter_model.safetensors."
        )

    from peft import PeftModel

    with _adapter_slot_for_load(slot) as load_path:
        # The context manager yields a directory containing the decrypted
        # adapter_model.safetensors + adapter_config.json.  PEFT's
        # load_adapter / from_pretrained accept that path directly.
        if isinstance(model, PeftModel):
            model.load_adapter(str(load_path), adapter_name=ADAPTER_NAME)
        else:
            model = PeftModel.from_pretrained(model, str(load_path), adapter_name=ADAPTER_NAME)
    switch_adapter(model, ADAPTER_NAME)
    return model, entries


def train_bench_adapter(model, tokenizer, entries: list[dict], num_epochs: int):
    """Train ``ADAPTER_NAME`` on ``entries`` so probes have a target to
    recall.  Returns the (possibly re-wrapped) model.
    """
    adapter_cfg = AdapterConfig(
        rank=8,
        alpha=16,
        dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        learning_rate=1e-4,
    )
    model = create_adapter(model, adapter_cfg, ADAPTER_NAME)
    switch_adapter(model, ADAPTER_NAME)
    examples = format_entry_training(entries, tokenizer, max_length=512)

    class _Dataset:
        def __len__(self):
            return len(examples)

        def __getitem__(self, idx):
            return examples[idx]

    train_cfg = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        warmup_ratio=0.0,
        lr_decay_steps=num_epochs * len(examples) // 4,
        weight_decay=0.1,
        gradient_checkpointing=True,
        recall_early_stopping=False,
    )
    train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_Dataset(),
        adapter_name=ADAPTER_NAME,
        training_config=train_cfg,
        adapter_config=adapter_cfg,
    )
    return model


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------


def _peak_vram_mib_delta(callable_):
    """Run *callable_* and return (result, peak_vram_mib_delta).

    The delta is reset_peak → callable → peak_allocated, so we isolate the
    callable's own peak from whatever the model/adapter is holding.
    """
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    result = callable_()
    peak = torch.cuda.max_memory_allocated()
    return result, (peak - baseline) / (1024 * 1024)


def run_cell(model, tokenizer, entries, registry, strategy: str, batch_size: int):
    """Time one strategy/batch_size cell and return its metrics dict."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    if strategy == "serial":
        runner = lambda: evaluate_indexed_recall(  # noqa: E731
            model,
            tokenizer,
            entries,
            registry,
            adapter_name=ADAPTER_NAME,
            batch_size=1,
        )["per_key"]
    elif strategy == "batched":
        runner = lambda: evaluate_indexed_recall(  # noqa: E731
            model,
            tokenizer,
            entries,
            registry,
            adapter_name=ADAPTER_NAME,
            batch_size=batch_size,
        )["per_key"]
    elif strategy == "prefix_cache":
        runner = lambda: probe_prefix_cache(  # noqa: E731
            model, tokenizer, entries, registry, batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    per_key, peak_mib = _peak_vram_mib_delta(runner)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # Normalise to the 3-field shape across all strategies — production
    # evaluate_indexed_recall returns a 10-field per-row dict; prefix_cache
    # already returns 3 fields; the reference results.json is 3-field
    # throughout.  Stripping keeps results.json comparable cell-for-cell.
    per_key = [
        {
            "key": r["key"],
            "exact_match": r["exact_match"],
            "failure_reason": r.get("failure_reason"),
        }
        for r in per_key
    ]

    exact = sum(1 for r in per_key if r["exact_match"])
    parse_failures = sum(1 for r in per_key if (r.get("failure_reason") or "") == "parse_failure")
    key_mismatches = sum(
        1 for r in per_key if str(r.get("failure_reason") or "").startswith("key_mismatch")
    )
    low_conf = sum(
        1 for r in per_key if str(r.get("failure_reason") or "").startswith("low_confidence")
    )

    return {
        "strategy": strategy,
        "batch_size": batch_size,
        "wall_clock_seconds": round(wall, 3),
        "mean_per_key_seconds": round(wall / max(len(entries), 1), 4),
        "peak_vram_mib_delta": round(peak_mib, 1),
        "exact_count": exact,
        "total": len(entries),
        "parse_failures": parse_failures,
        "key_mismatches": key_mismatches,
        "low_confidence": low_conf,
        "per_key": per_key,
    }


def run_sweep(model, tokenizer, entries, registry, batch_sizes: list[int]):
    """Run the full strategy × batch_size grid.  Serial fires once (b=1)."""
    cells: list[dict] = []

    logger.info("== serial baseline ==")
    cells.append(run_cell(model, tokenizer, entries, registry, "serial", 1))

    for b in batch_sizes:
        logger.info("== batched b=%d ==", b)
        cell = run_cell(model, tokenizer, entries, registry, "batched", b)
        cells.append(cell)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for b in batch_sizes:
        logger.info("== prefix_cache b=%d ==", b)
        cell = run_cell(model, tokenizer, entries, registry, "prefix_cache", b)
        cells.append(cell)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cells


def _summarise(cells: list[dict]) -> dict:
    baseline = next(c for c in cells if c["strategy"] == "serial")
    baseline_wall = baseline["wall_clock_seconds"] or 1e-9
    summary = {
        "baseline_wall_clock_seconds": baseline["wall_clock_seconds"],
        "baseline_exact_count": baseline["exact_count"],
        "speedup": [],
    }
    for c in cells:
        c["matches_baseline"] = c["exact_count"] == baseline["exact_count"]
        if c["strategy"] == "serial":
            continue
        summary["speedup"].append(
            {
                "strategy": c["strategy"],
                "batch_size": c["batch_size"],
                "ratio": round(baseline_wall / max(c["wall_clock_seconds"], 1e-9), 2),
                "matches_baseline": c["matches_baseline"],
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    parser.add_argument(
        "--n-keys",
        type=int,
        default=DEFAULT_N_KEYS,
        help=(
            "Number of synthetic indexed-key entries to probe "
            "(default: 137 — matches the CV-ingest baseline)."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated batch sizes to sweep for batched + prefix_cache "
            "strategies.  Default: powers of two from 1 up to n_keys, plus "
            "n_keys itself (full-batch ceiling probe).  Override for narrower "
            "sweeps, e.g. --batch-sizes 1,8,32."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=25,
        help=(
            "Training epochs for the bench adapter "
            "(default: 25 — enough for ≥0.95 recall on synthetic data)."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: n_keys=20, batch_sizes={1,4,8}, num_epochs=10. ~2-3 min on RTX 5070.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Skip training and reuse a captured production cycle's adapter + entries. "
            "PATH must contain adapter/ (with adapter_model.safetensors, possibly "
            "age-encrypted) and episodic_rels.json.  Default fixture: "
            "experiments/fixtures/test18_baseline/."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="outputs/test18_probe_batching",
        help="Output base directory.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.smoke:
        args.n_keys = 20
        args.batch_sizes = "1,4,8"
        args.num_epochs = 10

    # Resolve the effective n_keys: the --baseline path's entries take
    # precedence (their count is canonical) but n_keys is needed earlier
    # to compute the default sweep, so use --n-keys here as the planning
    # value and re-check after baseline load.
    effective_n_keys = args.n_keys
    if args.batch_sizes is None:
        batch_sizes = _default_batch_sizes(effective_n_keys)
    else:
        batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    models = get_benchmark_models(args)
    output_base = Path(args.output_base)

    for model_name, model_cfg in models:
        if model_name not in BENCHMARK_MODELS:
            logger.warning("Skipping unknown model: %s", model_name)
            continue

        run_dir = model_output_dir(output_base, model_name)
        if args.smoke:
            run_dir = run_dir.parent / "_smoke" / run_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)

        with acquire_gpu(interactive=True):
            logger.info("Loading model %s into %s", model_name, run_dir)
            model, tokenizer = load_base_model(model_cfg)

            if args.baseline:
                baseline_dir = Path(args.baseline)
                logger.info("Loading captured baseline from %s", baseline_dir)
                model, entries = load_baseline(baseline_dir, model, tokenizer)
                registry = build_registry(entries)
                logger.info(
                    "Baseline loaded: %d entries, adapter mounted as %r",
                    len(entries),
                    ADAPTER_NAME,
                )
            else:
                entries = _synthetic_entries(args.n_keys)
                registry = build_registry(entries)
                logger.info(
                    "Training bench adapter on %d entries × %d epochs",
                    len(entries),
                    args.num_epochs,
                )
                model = train_bench_adapter(model, tokenizer, entries, args.num_epochs)
                save_adapter(model, run_dir / "adapter", ADAPTER_NAME)

            # Re-derive the sweep grid from the actual entry count when the
            # user did not pin --batch-sizes.  ``effective_n_keys`` above used
            # ``args.n_keys`` as a planning value; the baseline path may load
            # a different count (e.g. 137 from production CV cycle).
            if args.batch_sizes is None and len(entries) != effective_n_keys:
                batch_sizes = _default_batch_sizes(len(entries))
                logger.info(
                    "Re-derived batch sizes for %d actual entries: %s",
                    len(entries),
                    batch_sizes,
                )

            logger.info("Sweeping probe strategies × batch sizes %s", batch_sizes)
            cells = run_sweep(model, tokenizer, entries, registry, batch_sizes)
            summary = _summarise(cells)

            payload = {
                "model": model_name,
                "n_keys": args.n_keys,
                "num_epochs": args.num_epochs,
                "batch_sizes": batch_sizes,
                "started_at": datetime.utcnow().isoformat() + "Z",
                "summary": summary,
                "cells": cells,
            }
            (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
            logger.info("Wrote %s", run_dir / "results.json")

            # Free GPU memory before the next model in the loop.
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
