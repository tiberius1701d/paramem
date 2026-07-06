#!/usr/bin/env python3
"""Live prompt-calibration tool.

Probes each LLM-touching pipeline stage independently against the running
paramem-server (1:1 with production code paths).  Stages are stop points
along the existing pipeline — selecting fewer stages just spares out the
later steps.  Iterate on prompt files and inference params; measure
compliance variance across seeds; compare candidate vs production
baseline side-by-side.

Stages:

* ``extract``       — local Mistral.  POST /calibrate/extract
* ``anonymize``     — local Mistral.  POST /calibrate/anonymize
* ``enrich``        — Anthropic SOTA.  Client-side (no server endpoint).
* ``plausibility``  — local Mistral.  POST /calibrate/plausibility

Usage::

    # Full pipeline on the resume's chunk 0, baseline + candidate compared:
    python scripts/dev/calibrate_prompts.py \\
        --input ingest/resume.pdf --chunk 0 \\
        --speaker Alex --speaker-id speaker0 \\
        --stages extract,anonymize,enrich,plausibility \\
        --baseline auto --prompt-prefix calib_

    # Variance probe — 5 seeds, just the extract stage:
    python scripts/dev/calibrate_prompts.py \\
        --input ingest/resume.pdf --chunk 0 \\
        --speaker-id speaker0 --stages extract \\
        --seeds 42,7,1337,2024,99

    # Re-iterate on the enrichment prompt only, seeded from a prior dump:
    python scripts/dev/calibrate_prompts.py \\
        --input ingest/resume.pdf --chunk 0 --stages enrich \\
        --seed-from data/ha/debug/calibration/<prior-ts>/

Out of scope: merger, QA generator, adapter training, recall/chat.  The
tool stops at "what did the LLM emit for this stage given this prompt
and these params."
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Ensure project root is importable when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paramem.cli.http_client import resolve_token  # noqa: E402
from paramem.graph.document_chunker import (  # noqa: E402
    chunk_markdown_file,
    chunk_pdf_file,
    chunk_text_file,
)

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_chunks(path: Path, source_type_override: str | None) -> tuple[list[dict], str]:
    """Return ``(chunks, source_type)`` for ``path``.

    Each chunk dict carries ``text`` and ``chunk_index``.  ``source_type`` is
    auto-inferred from extension unless overridden.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        cs = chunk_pdf_file(path)
        return (
            [{"text": c.text, "chunk_index": c.chunk_index} for c in cs],
            source_type_override or "document",
        )
    if suffix in (".md", ".markdown"):
        cs = chunk_markdown_file(path)
        return (
            [{"text": c.text, "chunk_index": c.chunk_index} for c in cs],
            source_type_override or "document",
        )
    if suffix == ".txt":
        cs = chunk_text_file(path)
        return (
            [{"text": c.text, "chunk_index": c.chunk_index} for c in cs],
            source_type_override or "document",
        )
    if suffix == ".jsonl":
        # Multi-turn transcript: load and concatenate as a single chunk.
        with path.open() as f:
            turns = [json.loads(line) for line in f if line.strip()]
        text = "\n".join(t.get("text", "") for t in turns if t.get("text"))
        return ([{"text": text, "chunk_index": 0}], source_type_override or "transcript")
    raise SystemExit(f"Unsupported input extension: {suffix}")


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------


def _calibration_variant_exists(
    prompts_dir: Path,
    base_filenames: list[str],
    prefix: str,
) -> bool:
    """True iff at least one prefixed prompt file exists for this stage.

    Used by ``--baseline auto`` to decide whether to do the side-by-side
    comparison (only worth running baseline when there's something to
    compare against).
    """
    return any((prompts_dir / f"{prefix}{base}").exists() for base in base_filenames)


# ---------------------------------------------------------------------------
# Stage runners — local stages POST to the server, enrich runs locally
# ---------------------------------------------------------------------------


def _post_stage(
    server: str,
    stage: str,
    payload: dict,
    timeout: float = 600.0,
) -> dict:
    """POST *payload* to ``/calibrate/<stage>`` and return the parsed JSON response.

    Resolves the bearer token from the environment, secret file, or repo ``.env``
    (via :func:`paramem.cli.http_client.resolve_token`) and attaches it as an
    ``Authorization: Bearer <token>`` header.  When no token is present the
    header is omitted so auth-OFF servers keep working.
    """
    url = f"{server.rstrip('/')}/calibrate/{stage}"
    token = resolve_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.post(url, json=payload, timeout=timeout, headers=headers)
    if r.status_code == 503:
        raise SystemExit(
            f"Server returned 503 for /calibrate/{stage}: {r.text}\n"
            f"Likely a real consolidation cycle is in progress; retry later."
        )
    if r.status_code == 404:
        raise SystemExit(
            f"Server returned 404 for /calibrate/{stage}: {r.text}\n"
            f"Set consolidation.calibrate_endpoint_enabled: true in server.yaml."
        )
    if r.status_code == 401:
        raise SystemExit(
            f"Server returned 401 for /calibrate/{stage}: no/invalid bearer token.\n"
            f"Set PARAMEM_API_TOKEN (env, ~/.config/paramem/secrets/PARAMEM_API_TOKEN,"
            f" or repo .env)."
        )
    r.raise_for_status()
    return r.json()


def _run_enrich(
    *,
    facts: list[dict],
    transcript: str,
    prompts_dir: Path,
    prompt_filename: str,
    sota_route: str,
    sota_provider: str,
    sota_model: str,
    params: dict,
) -> dict:
    """Run the SOTA enrichment stage entirely client-side.

    Reuses the production prompt template from ``prompts_dir`` so the
    enrichment text the model sees is byte-identical to what production
    would build.  Routes through ``cli`` (subprocess to ``claude
    --print``) when available, else through the Anthropic SDK.
    """
    prompt_path = prompts_dir / prompt_filename
    if not prompt_path.exists():
        raise SystemExit(f"Enrichment prompt not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    sha = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]

    rendered = template.format(
        facts_json=json.dumps(facts, indent=2),
        transcript=transcript or "(not available)",
    )

    chosen_route = sota_route
    if chosen_route == "auto":
        chosen_route = "cli" if shutil.which("claude") else "api"

    t0 = time.perf_counter()
    raw_output: str | None = None
    parse_error: str | None = None
    seed_dropped = params.get("seed") is not None

    if chosen_route == "cli":
        instruction = (
            "\n\n--- IMPORTANT: return ONLY the JSON envelope as specified. "
            "No markdown fences, no preamble, no closing remarks. ---"
        )
        try:
            res = subprocess.run(
                ["claude", "--print", rendered + instruction],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if res.returncode != 0:
                parse_error = f"claude CLI exited {res.returncode}: {res.stderr[:400]}"
            else:
                raw_output = res.stdout
        except FileNotFoundError:
            parse_error = "claude CLI not on PATH; pass --sota-route api"
        except subprocess.TimeoutExpired:
            parse_error = "claude CLI timed out after 600s"
    elif chosen_route == "api":
        from paramem.graph.extractor import (
            _SOTA_ENRICHMENT_SYSTEM_PROMPT,
            _filter_anthropic,
        )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not set; pass --sota-route cli to use "
                "the local Claude Code CLI instead."
            )
        raw_output = _filter_anthropic(
            rendered,
            api_key=api_key,
            filter_model=sota_model,
            system_prompt=_SOTA_ENRICHMENT_SYSTEM_PROMPT,
            max_tokens=params.get("max_tokens") or 8192,
            temperature=(
                params.get("temperature") if params.get("temperature") is not None else 0.0
            ),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
        )
        if raw_output is None:
            parse_error = "Anthropic call returned None (SDK or transport failure)"
    else:
        raise SystemExit(f"Unknown --sota-route value: {chosen_route}")

    elapsed = time.perf_counter() - t0

    parsed: dict | None = None
    if raw_output and not parse_error:
        try:
            from paramem.graph.extractor import _extract_json_block

            parsed = json.loads(_extract_json_block(raw_output))
        except Exception as exc:  # noqa: BLE001
            parse_error = f"JSON parse failed: {exc}"

    return {
        "stage": "enrich",
        "prompts": [
            {"role": "user", "path": str(prompt_path), "sha": sha, "content": template},
        ],
        "raw_output": raw_output or "",
        "parsed": parsed or {},
        "parse_error": parse_error,
        "wall_clock_seconds": elapsed,
        "model": sota_model,
        "sota_route_effective": chosen_route,
        "params_effective": {
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p") if chosen_route == "api" else None,
            "top_k": params.get("top_k") if chosen_route == "api" else None,
            "seed": None,  # Anthropic accepts no seed
            "max_tokens": params.get("max_tokens"),
            "seed_dropped": seed_dropped,
        },
    }


# ---------------------------------------------------------------------------
# Diff + variance reporting
# ---------------------------------------------------------------------------


def _triples(parsed: dict) -> set[tuple[str, str, str]]:
    """Extract ``(subject, predicate, object)`` triples from a parsed
    payload (extract, enrich, or plausibility output)."""
    relations = parsed.get("relations") or parsed.get("kept_facts") or parsed.get("facts") or []
    out: set[tuple[str, str, str]] = set()
    for r in relations:
        if isinstance(r, dict):
            s = str(r.get("subject", ""))
            p = str(r.get("predicate", ""))
            o = str(r.get("object", ""))
            if s and p and o:
                out.add((s, p, o))
    return out


def _entity_names(parsed: dict) -> set[str]:
    ents = parsed.get("entities") or []
    return {str(e.get("name", "")) for e in ents if isinstance(e, dict) and e.get("name")}


def _speaker_attrs(parsed: dict) -> dict[str, str]:
    ents = parsed.get("entities") or []
    for e in ents:
        if isinstance(e, dict) and e.get("speaker_id"):
            return {str(k): str(v) for k, v in (e.get("attributes") or {}).items()}
    return {}


def _diff_blocks(baseline: dict, candidate: dict) -> dict:
    """Compute the diff block for a side-by-side dump."""
    b_parsed = baseline.get("parsed") or {}
    c_parsed = candidate.get("parsed") or {}
    b_trip = _triples(b_parsed)
    c_trip = _triples(c_parsed)
    b_ents = _entity_names(b_parsed)
    c_ents = _entity_names(c_parsed)
    b_attrs = _speaker_attrs(b_parsed)
    c_attrs = _speaker_attrs(c_parsed)

    attr_changes: dict = {}
    for k in set(b_attrs) | set(c_attrs):
        if b_attrs.get(k) != c_attrs.get(k):
            attr_changes[k] = [b_attrs.get(k), c_attrs.get(k)]

    return {
        "entities_added": sorted(c_ents - b_ents),
        "entities_removed": sorted(b_ents - c_ents),
        "triples_added": sorted([list(t) for t in (c_trip - b_trip)]),
        "triples_removed": sorted([list(t) for t in (b_trip - c_trip)]),
        "speaker_attrs_changed": attr_changes,
        "counts": {
            "baseline_entities": len(b_ents),
            "candidate_entities": len(c_ents),
            "baseline_triples": len(b_trip),
            "candidate_triples": len(c_trip),
        },
    }


def _phase_diff(baseline: dict, candidate: dict) -> dict:
    """Per-phase diff: align ``baseline.phases`` with ``candidate.phases``
    by name and compute, for each phase, whether the prompt change moved
    anything.

    The result is a list of per-phase diff blocks the operator can read
    sequentially to localise which phase a prompt change actually
    affects.  Phases that fired in only one run are flagged as
    asymmetric so the operator notices configuration drift.
    """
    b_phases = {p["name"]: p for p in (baseline.get("phases") or []) if isinstance(p, dict)}
    c_phases = {p["name"]: p for p in (candidate.get("phases") or []) if isinstance(p, dict)}
    all_names = []
    seen = set()
    for p in (baseline.get("phases") or []) + (candidate.get("phases") or []):
        if isinstance(p, dict) and p.get("name") not in seen:
            seen.add(p["name"])
            all_names.append(p["name"])

    blocks: list[dict] = []
    for name in all_names:
        b = b_phases.get(name)
        c = c_phases.get(name)
        block: dict = {"phase": name}
        if b is None:
            block["asymmetric"] = "candidate-only"
        elif c is None:
            block["asymmetric"] = "baseline-only"
        b_parsed = (b or {}).get("parsed") or {}
        c_parsed = (c or {}).get("parsed") or {}
        # Common parsed-key diff: any key whose value differs.
        parsed_changed: dict = {}
        for k in set(b_parsed) | set(c_parsed):
            if b_parsed.get(k) != c_parsed.get(k):
                parsed_changed[k] = {"baseline": b_parsed.get(k), "candidate": c_parsed.get(k)}
        if parsed_changed:
            block["parsed_changed"] = parsed_changed
        # Raw output: report length delta and a short head snippet so the
        # operator can spot prompt-compliance differences at a glance.
        b_raw = (b or {}).get("raw_output") or ""
        c_raw = (c or {}).get("raw_output") or ""
        if b_raw or c_raw:
            block["raw_output"] = {
                "baseline_bytes": len(b_raw),
                "candidate_bytes": len(c_raw),
                "baseline_head": b_raw[:160],
                "candidate_head": c_raw[:160],
            }
        # Wall-clock delta.
        b_wall = (b or {}).get("wall_clock_seconds")
        c_wall = (c or {}).get("wall_clock_seconds")
        if b_wall is not None or c_wall is not None:
            block["wall_seconds"] = {"baseline": b_wall, "candidate": c_wall}
        # Outcome divergence (skipped/failed/ok).
        b_out = (b or {}).get("outcome")
        c_out = (c or {}).get("outcome")
        if b_out != c_out:
            block["outcome_changed"] = {"baseline": b_out, "candidate": c_out}
        blocks.append(block)
    return {"per_phase": blocks}


def _print_phase_diff(diff: dict) -> None:
    """Render the per-phase diff to stdout in operator-readable shape."""
    blocks = (diff or {}).get("per_phase") or []
    if not blocks:
        return
    print("\n=== per-phase diff (baseline → candidate) ===")
    for b in blocks:
        name = b.get("phase", "?")
        suffix = ""
        if b.get("asymmetric"):
            suffix = f"  [{b['asymmetric']}]"
        if b.get("outcome_changed"):
            oc = b["outcome_changed"]
            suffix += f"  outcome: {oc['baseline']} → {oc['candidate']}"
        wall = b.get("wall_seconds") or {}
        wall_str = ""
        if wall:
            wall_str = f"  wall {wall.get('baseline')}s → {wall.get('candidate')}s"
        print(f"\n--- phase: {name}{suffix}{wall_str}")
        if b.get("raw_output"):
            r = b["raw_output"]
            print(f"  raw {r['baseline_bytes']}b → {r['candidate_bytes']}b")
            if r["baseline_head"] and r["baseline_head"] != r["candidate_head"]:
                print(f"    baseline_head: {r['baseline_head']!r}")
                print(f"    candidate_head: {r['candidate_head']!r}")
        if b.get("parsed_changed"):
            for key, diff_pair in sorted(b["parsed_changed"].items()):
                bp = diff_pair["baseline"]
                cp = diff_pair["candidate"]
                # Truncate large values for terminal readability; full
                # data stays in the dump JSON.
                bp_s = str(bp)[:120]
                cp_s = str(cp)[:120]
                print(f"    {key}: {bp_s} → {cp_s}")


def _jaccard_pairwise(sets: list[set]) -> dict:
    if len(sets) < 2:
        return {"mean": None, "min": None, "max": None}
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                pairs.append(1.0)
                continue
            if not a or not b:
                pairs.append(0.0)
                continue
            pairs.append(len(a & b) / len(a | b))
    return {
        "mean": round(sum(pairs) / len(pairs), 4),
        "min": round(min(pairs), 4),
        "max": round(max(pairs), 4),
    }


def _variance_report(stage: str, runs: list[dict]) -> dict:
    """Per-stage variance summary across N seed runs.

    Reports Jaccard similarity (pairwise mean / min / max) over triples
    and entities, plus the per-entity-instance compliance ratios for
    the prompt-tuning signals we care about.
    """
    if len(runs) < 2:
        return {}
    triple_sets = [_triples(r.get("parsed") or {}) for r in runs]
    entity_sets = [_entity_names(r.get("parsed") or {}) for r in runs]

    speaker_entity_count = 0
    person_typed = 0
    has_email_total = 0
    has_email_present = 0
    has_phone_total = 0
    has_phone_present = 0
    has_linkedin_total = 0
    has_linkedin_present = 0
    for r in runs:
        parsed = r.get("parsed") or {}
        ents = parsed.get("entities") or []
        for e in ents:
            if not isinstance(e, dict):
                continue
            if e.get("speaker_id"):
                speaker_entity_count += 1
                if e.get("entity_type") == "person":
                    person_typed += 1
                attrs = e.get("attributes") or {}
                has_email_total += 1
                has_phone_total += 1
                has_linkedin_total += 1
                if attrs.get("has_email"):
                    has_email_present += 1
                if attrs.get("has_phone"):
                    has_phone_present += 1
                if attrs.get("has_linkedin"):
                    has_linkedin_present += 1

    return {
        "stage": stage,
        "n_runs": len(runs),
        "triples_jaccard": _jaccard_pairwise(triple_sets),
        "entities_jaccard": _jaccard_pairwise(entity_sets),
        "speaker_entity_count": speaker_entity_count,
        "person_typed": (
            f"{person_typed}/{speaker_entity_count}" if speaker_entity_count else "n/a"
        ),
        "has_email_present": (
            f"{has_email_present}/{has_email_total}" if has_email_total else "n/a"
        ),
        "has_phone_present": (
            f"{has_phone_present}/{has_phone_total}" if has_phone_total else "n/a"
        ),
        "has_linkedin_present": (
            f"{has_linkedin_present}/{has_linkedin_total}" if has_linkedin_total else "n/a"
        ),
        "wall_clock_seconds": [r.get("wall_clock_seconds") for r in runs],
        "n_output_tokens": [r.get("n_output_tokens") for r in runs],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_STAGE_FILENAME = {
    # One prompt-pair for every source type — see
    # paramem/graph/extraction_pipeline.py.  ``source_type`` survives as
    # a runtime gate-default flag but no longer selects prompt files.
    "extract_user": "extraction.txt",
    "extract_system": "extraction_system.txt",
    "anonymize": "anonymization.txt",
    "enrich": "sota_enrichment.txt",
    "plausibility": "sota_plausibility.txt",
    "normalize_filter": "graph_dedup_filter.txt",
    "name_user": "name_extraction.txt",
    "name_system": "name_extraction_system.txt",
}


def _parse_seeds(spec: str | None) -> list[int | None]:
    if not spec or spec == "none":
        return [None]
    if spec.startswith("random:"):
        n = int(spec.split(":", 1)[1])
        # Deterministic seed derivation so re-running produces the same set.
        return [hash(f"calib_seed_{i}") & 0xFFFFFFFF for i in range(n)]
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--server", default="http://localhost:8420")
    parser.add_argument(
        "--input",
        required=False,
        default=None,
        help=(
            "input file (PDF, markdown, txt, jsonl). Required for chunk stages "
            "(extract, anonymize, enrich, plausibility). Not used for 'normalize'."
        ),
    )
    parser.add_argument("--source-type", choices=["transcript", "document"], default=None)
    parser.add_argument("--speaker", default="Alex")
    parser.add_argument("--speaker-id", default="speaker0")
    parser.add_argument(
        "--stages",
        default="extract,anonymize,enrich,plausibility",
        help=(
            "comma-separated stage names; stages are stop points along the "
            "production pipeline — selecting fewer stages just spares out "
            "the later steps. 'normalize' is a standalone graph-level stage "
            "that requires --snapshot and cannot be combined with chunk stages."
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help=(
            "path to a graph_merged_snapshot.json (NetworkX node-link format). "
            "Required when --stages includes 'normalize'. "
            "Cannot be combined with chunk stages (extract, anonymize, enrich, plausibility)."
        ),
    )
    parser.add_argument(
        "--prompts-dir",
        default=str(_REPO_ROOT / "configs" / "prompts"),
        help="default: project's configs/prompts/",
    )
    parser.add_argument("--prompt-prefix", default="calib_")
    parser.add_argument(
        "--baseline",
        choices=["auto", "production", "none"],
        default="auto",
        help=(
            "auto: run baseline only when a calib_-prefixed file exists in "
            "--prompts-dir; production: always; none: skip baseline"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="none",
        help=(
            "comma-separated seed list, 'random:N', or 'none' (default). "
            "Seeds only vary output at --temperature>0; at the default "
            "greedy temperature 0.0 they are a no-op."
        ),
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--plausibility-max-tokens", type=int, default=None)
    parser.add_argument(
        "--sota-route",
        choices=["auto", "cli", "api"],
        default="auto",
        help="auto: cli if `claude` on PATH else api",
    )
    parser.add_argument("--sota-provider", default="anthropic")
    parser.add_argument("--sota-model", default="claude-sonnet-4-6")
    parser.add_argument("--chunk", type=int, default=None, help="run on this chunk index only")
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="default: data/ha/debug/calibration/<utc-timestamp>/",
    )
    parser.add_argument(
        "--seed-from",
        default=None,
        help="prior dump dir; load earlier-stage outputs from here",
    )
    parser.add_argument(
        "--stop-phase",
        default=None,
        help=(
            "Forwarded to /calibrate/extract — pipeline returns immediately "
            "after the named phase completes (saves compute when only early "
            "phases need inspection). Valid names: local_extract, "
            "ha_validation, anonymize, anonymize_verify, "
            "anonymize_repair, sota_enrich, anon_plausibility, deanon, "
            "deanon_plausibility. Default: run full pipeline."
        ),
    )
    parser.add_argument(
        "--turns-jsonl",
        default=None,
        help=(
            'JSONL file of conversation turns ({"role": str, "text": str} per line). '
            "Required when --stages includes 'name'. Cannot be combined with "
            "chunk stages (extract, anonymize, enrich, plausibility)."
        ),
    )
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    seeds = _parse_seeds(args.seeds)
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.exists():
        raise SystemExit(f"--prompts-dir does not exist: {prompts_dir}")
    params_base: dict = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }

    # Default dump dir under data/ha/debug/calibration/<ts>/.
    if args.dump_dir is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump_dir = _REPO_ROOT / "data" / "ha" / "debug" / "calibration" / ts
    else:
        dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        chunks, source_type = _load_chunks(Path(args.input), args.source_type)
        if args.chunk is not None:
            if args.chunk < 0 or args.chunk >= len(chunks):
                raise SystemExit(
                    f"--chunk {args.chunk} out of range (input has {len(chunks)} chunks)"
                )
            chunks = [chunks[args.chunk]]
    elif "normalize" not in stages:
        raise SystemExit(
            "Error: --input is required for chunk stages "
            "(extract, anonymize, enrich, plausibility)."
        )
    else:
        chunks = []
        source_type = "document"

    # Decide whether baseline runs alongside candidate.
    base_filenames_per_stage = {
        "extract": [_STAGE_FILENAME["extract_user"], _STAGE_FILENAME["extract_system"]],
        "anonymize": [_STAGE_FILENAME["anonymize"]],
        "enrich": [_STAGE_FILENAME["enrich"]],
        "plausibility": [_STAGE_FILENAME["plausibility"]],
        "normalize": [_STAGE_FILENAME["normalize_filter"]],
        "name": [_STAGE_FILENAME["name_user"], _STAGE_FILENAME["name_system"]],
    }

    def _run_baseline_for(stage: str) -> bool:
        if args.baseline == "none":
            return False
        if args.baseline == "production":
            return True
        return _calibration_variant_exists(
            prompts_dir, base_filenames_per_stage[stage], args.prompt_prefix
        )

    # Guard: normalize cannot be combined with chunk stages.
    _CHUNK_STAGES = {"extract", "anonymize", "enrich", "plausibility"}
    if "normalize" in stages and stages != ["normalize"]:
        raise SystemExit(
            "Error: 'normalize' cannot be combined with chunk stages "
            "(extract, anonymize, enrich, plausibility). "
            "Run normalize in a separate invocation: --stages normalize --snapshot <path>"
        )
    if "normalize" in stages and not args.snapshot:
        raise SystemExit(
            "Error: --stages normalize requires --snapshot <graph_merged_snapshot.json>"
        )
    # Guard: name cannot be combined with chunk stages (it uses --turns-jsonl,
    # not --input).  Mirror the normalize mutual-exclusion pattern.
    if "name" in stages and stages != ["name"]:
        raise SystemExit(
            "Error: 'name' cannot be combined with other stages "
            "(extract, anonymize, enrich, plausibility, normalize). "
            "Run name in a separate invocation: --stages name --turns-jsonl <file.jsonl>"
        )
    # Guard: name requires --turns-jsonl (a JSONL file of {role, text} dicts).
    if "name" in stages and not getattr(args, "turns_jsonl", None):
        raise SystemExit(
            "Error: --stages name requires --turns-jsonl <file.jsonl> "
            '(a JSONL file of {"role": str, "text": str} dicts).'
        )

    invocation = {
        "args": vars(args),
        "source_type": source_type,
        "n_chunks": len(chunks),
        "seeds": seeds,
        "dump_dir": str(dump_dir),
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (dump_dir / "00_invocation.json").write_text(json.dumps(invocation, indent=2, default=str))

    # ----- run stages per chunk per seed ---------------------------------
    for chunk in chunks:
        chunk_idx = chunk["chunk_index"]
        prior_extract: dict | None = None
        prior_anonymize: dict | None = None
        prior_enrich: dict | None = None

        if args.seed_from:
            seed_from = Path(args.seed_from)
            for fname, slot in [
                (f"01_extract_chunk_{chunk_idx}.json", "extract"),
                (f"02_anonymize_chunk_{chunk_idx}.json", "anonymize"),
                (f"03_enrich_chunk_{chunk_idx}.json", "enrich"),
            ]:
                f = seed_from / fname
                if f.exists():
                    blob = json.loads(f.read_text())
                    if slot == "extract":
                        prior_extract = blob
                    elif slot == "anonymize":
                        prior_anonymize = blob
                    elif slot == "enrich":
                        prior_enrich = blob

        if "extract" in stages:
            extract_runs: list[dict] = []
            # Single prompt-pair for every source type — same baseline file
            # regardless of whether the input is a transcript or a document.
            user_basename = _STAGE_FILENAME["extract_user"]
            system_basename = _STAGE_FILENAME["extract_system"]
            user_calib = f"{args.prompt_prefix}{user_basename}"
            system_calib = f"{args.prompt_prefix}{system_basename}"
            user_override = user_calib if (prompts_dir / user_calib).exists() else None
            system_override = system_calib if (prompts_dir / system_calib).exists() else None
            for seed in seeds:
                params = dict(params_base)
                params["seed"] = seed
                candidate = _post_stage(
                    args.server,
                    "extract",
                    {
                        "transcript": chunk["text"],
                        "speaker_id": args.speaker_id,
                        "speaker_name": args.speaker,
                        "source_type": source_type,
                        "session_id": f"calib-chunk-{chunk_idx}",
                        "prompts_dir": (args.prompts_dir if args.prompts_dir else None),
                        "extraction_prompt_filename": user_override,
                        "extraction_system_prompt_filename": system_override,
                        "stop_phase": args.stop_phase,
                        "params": {k: v for k, v in params.items() if v is not None},
                    },
                )
                run_record = {"seed": seed, **candidate}
                extract_runs.append(run_record)
            baseline_blob = None
            if _run_baseline_for("extract"):
                baseline_blob = _post_stage(
                    args.server,
                    "extract",
                    {
                        "transcript": chunk["text"],
                        "speaker_id": args.speaker_id,
                        "speaker_name": args.speaker,
                        "source_type": source_type,
                        "session_id": f"calib-chunk-{chunk_idx}-baseline",
                        "prompts_dir": args.prompts_dir,
                        "extraction_prompt_filename": None,
                        "extraction_system_prompt_filename": None,
                        "stop_phase": args.stop_phase,
                        "params": {k: v for k, v in params_base.items() if v is not None},
                    },
                )
            out_blob: dict[str, Any] = {
                "stage": "extract",
                "chunk_index": chunk_idx,
                "candidate_runs": extract_runs,
            }
            if baseline_blob is not None:
                out_blob["baseline"] = baseline_blob
                out_blob["diff"] = _diff_blocks(baseline_blob, extract_runs[0])
                # Per-phase diff over the phase trace — localises which
                # phase a prompt change actually moved.  Renders to
                # stdout for immediate operator feedback; the full data
                # stays in the dump JSON.
                phase_diff = _phase_diff(baseline_blob, extract_runs[0])
                out_blob["phase_diff"] = phase_diff
                _print_phase_diff(phase_diff)
            if len(seeds) > 1:
                out_blob["variance"] = _variance_report("extract", extract_runs)
            (dump_dir / f"01_extract_chunk_{chunk_idx}.json").write_text(
                json.dumps(out_blob, indent=2, default=str)
            )
            prior_extract = extract_runs[0]

        if "anonymize" in stages and prior_extract is not None:
            anon_calib = f"{args.prompt_prefix}{_STAGE_FILENAME['anonymize']}"
            anon_override = anon_calib if (prompts_dir / anon_calib).exists() else None
            anon = _post_stage(
                args.server,
                "anonymize",
                {
                    "graph": prior_extract.get("parsed", {}),
                    "transcript": chunk["text"],
                    "session_id": f"calib-chunk-{chunk_idx}",
                    "prompts_dir": args.prompts_dir,
                    "anonymization_prompt_filename": anon_override,
                    "params": {k: v for k, v in params_base.items() if v is not None},
                },
            )
            (dump_dir / f"02_anonymize_chunk_{chunk_idx}.json").write_text(
                json.dumps(anon, indent=2, default=str)
            )
            prior_anonymize = anon

        if "enrich" in stages and prior_anonymize is not None:
            anon_parsed = prior_anonymize.get("parsed") or {}
            enrich_calib = f"{args.prompt_prefix}{_STAGE_FILENAME['enrich']}"
            enrich_filename = (
                enrich_calib if (prompts_dir / enrich_calib).exists() else _STAGE_FILENAME["enrich"]
            )
            enriched = _run_enrich(
                facts=anon_parsed.get("anonymized_facts") or [],
                transcript=anon_parsed.get("anonymized_transcript") or "",
                prompts_dir=prompts_dir,
                prompt_filename=enrich_filename,
                sota_route=args.sota_route,
                sota_provider=args.sota_provider,
                sota_model=args.sota_model,
                params=params_base,
            )
            (dump_dir / f"03_enrich_chunk_{chunk_idx}.json").write_text(
                json.dumps(enriched, indent=2, default=str)
            )
            prior_enrich = enriched

        if "plausibility" in stages and prior_enrich is not None:
            facts = (prior_enrich.get("parsed") or {}).get("facts") or []
            params_plaus = dict(params_base)
            if args.plausibility_max_tokens is not None:
                params_plaus["max_tokens"] = args.plausibility_max_tokens
            plaus_calib = f"{args.prompt_prefix}{_STAGE_FILENAME['plausibility']}"
            plaus_override = plaus_calib if (prompts_dir / plaus_calib).exists() else None
            plaus = _post_stage(
                args.server,
                "plausibility",
                {
                    "facts": facts,
                    "transcript": chunk["text"],
                    "prompts_dir": args.prompts_dir,
                    "plausibility_prompt_filename": plaus_override,
                    "params": {k: v for k, v in params_plaus.items() if v is not None},
                },
            )
            (dump_dir / f"04_plausibility_chunk_{chunk_idx}.json").write_text(
                json.dumps(plaus, indent=2, default=str)
            )

    # ----- normalize stage (graph-level, runs outside the chunk loop) -------
    if "normalize" in stages:
        filter_calib = f"{args.prompt_prefix}{_STAGE_FILENAME['normalize_filter']}"
        filter_override = filter_calib if (prompts_dir / filter_calib).exists() else None
        normalize_runs: list[dict] = []
        for seed in seeds:
            params = dict(params_base)
            params["seed"] = seed
            norm = _post_stage(
                args.server,
                "normalize",
                {
                    "snapshot_path": args.snapshot,
                    "filter_prompt_filename": filter_override,
                    "prompts_dir": args.prompts_dir,
                    "params": {k: v for k, v in params.items() if v is not None},
                },
            )
            normalize_runs.append({"seed": seed, **norm})
        out_blob: dict[str, Any] = {
            "stage": "normalize",
            "snapshot_path": args.snapshot,
            "candidate_runs": normalize_runs,
        }
        if len(seeds) > 1:
            out_blob["variance"] = _variance_report("normalize", normalize_runs)
        (dump_dir / "05_normalize.json").write_text(json.dumps(out_blob, indent=2, default=str))

    # ----- name stage (standalone — requires --turns-jsonl) ----------------
    if "name" in stages:
        turns_path = Path(args.turns_jsonl)
        with turns_path.open(encoding="utf-8") as _f:
            turns_data = [json.loads(line) for line in _f if line.strip()]
        name_user_calib = f"{args.prompt_prefix}{_STAGE_FILENAME['name_user']}"
        name_sys_calib = f"{args.prompt_prefix}{_STAGE_FILENAME['name_system']}"
        name_user_override = name_user_calib if (prompts_dir / name_user_calib).exists() else None
        name_sys_override = name_sys_calib if (prompts_dir / name_sys_calib).exists() else None
        name_runs: list[dict] = []
        for seed in seeds:
            params = dict(params_base)
            params["seed"] = seed
            name_result = _post_stage(
                args.server,
                "name",
                {
                    "turns": turns_data,
                    "prompts_dir": args.prompts_dir,
                    "name_prompt_filename": name_user_override or _STAGE_FILENAME["name_user"],
                    "name_system_prompt_filename": (
                        name_sys_override or _STAGE_FILENAME["name_system"]
                    ),
                    "user_turns_only": True,
                    "params": {k: v for k, v in params.items() if v is not None},
                },
            )
            name_runs.append({"seed": seed, **name_result})
        out_blob_name: dict[str, Any] = {
            "stage": "name",
            "turns_jsonl": str(turns_path),
            "candidate_runs": name_runs,
        }
        if len(seeds) > 1:
            out_blob_name["variance"] = _variance_report("name", name_runs)
        (dump_dir / "06_name.json").write_text(json.dumps(out_blob_name, indent=2, default=str))

    print(f"Calibration complete.  Dump: {dump_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
