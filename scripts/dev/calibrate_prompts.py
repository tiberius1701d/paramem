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
* ``enrich``        — cloud provider.  POST /calibrate/enrich
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
        --seed-from data/ha/calibration/artifacts/<prior-ts>/

Out of scope: merger, keyed-entry assembly, adapter training, recall/chat.  The
tool stops at "what did the LLM emit for this stage given this prompt
and these params."
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
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
from paramem.server.session_buffer import SessionBuffer  # noqa: E402
from paramem.utils.artifacts import write_artifact  # noqa: E402

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_chunks(path: Path, source_type_override: str | None) -> tuple[list[dict], str]:
    """Return ``(chunks, source_type)`` for ``path``.

    Each chunk dict carries ``text`` and ``chunk_index``.  ``text`` is
    rendered through :meth:`SessionBuffer._format_turns` — the SAME
    turn-marking renderer every production producer uses (``/chat``,
    document ingest, cloud egress) — so the transcript this tool POSTs to
    ``/calibrate/*`` is the ``[user] <text>`` / ``[assistant] <text>``
    surface the prompts' few-shots are actually calibrated on, not a
    hand-rolled second renderer.  ``source_type`` is auto-inferred from
    extension unless overridden.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        cs = chunk_pdf_file(path)
        chunks = [{"text": c.text, "chunk_index": c.chunk_index} for c in cs]
        source_type = source_type_override or "document"
    elif suffix in (".md", ".markdown"):
        cs = chunk_markdown_file(path)
        chunks = [{"text": c.text, "chunk_index": c.chunk_index} for c in cs]
        source_type = source_type_override or "document"
    elif suffix == ".txt":
        cs = chunk_text_file(path)
        chunks = [{"text": c.text, "chunk_index": c.chunk_index} for c in cs]
        source_type = source_type_override or "document"
    elif suffix == ".jsonl":
        # Multi-turn transcript: render every turn through the production
        # renderer — preserves the real per-turn role alternation, unlike
        # the previous bare-text concatenation (which lost role info and
        # never emitted a marker at all).
        with path.open() as f:
            turns = [json.loads(line) for line in f if line.strip()]
        lines, _ = SessionBuffer._format_turns(turns)
        return (
            [{"text": "\n".join(lines), "chunk_index": 0}],
            source_type_override or "transcript",
        )
    else:
        raise SystemExit(f"Unsupported input extension: {suffix}")

    # Document-shaped inputs: production always ingests document chunks via
    # ``SessionBuffer.append_document_chunk(..., "user", ...)``
    # (paramem/server/app.py), so each chunk renders as a single
    # ``[user] <chunk text>`` line — call the SAME renderer, not a shape
    # mimic, so a future marker-format change here changes with it.
    for c in chunks:
        marked_lines, _ = SessionBuffer._format_turns([{"role": "user", "text": c["text"]}])
        c["text"] = "\n".join(marked_lines)
    return chunks, source_type


# ---------------------------------------------------------------------------
# Prompt resolution
# ---------------------------------------------------------------------------


_STAGE_PROMPTS: dict[str, tuple[str, ...]] = {
    "extract": ("extract_user", "extract_system"),
    "anonymize": ("anonymize",),
    "enrich": ("enrich",),
    "plausibility": ("plausibility",),
    "normalize": ("normalize_filter",),
    "anonymize_facts": ("anonymize_facts",),
    "name": ("name_user", "name_system"),
}


_INDEX_FILENAME = "runs.json"


def _load_run(index_dir: Path, chunk_idx: int) -> dict | None:
    """Read a previously recorded extract run back from the SERVER's record.

    The client keeps an index of which run covered which chunk
    (``runs.json``); the run itself is the artifact the server wrote under
    ``paths.calibration/artifacts/<run>/``. Nothing is re-recorded on this
    side, so there is exactly one copy of any run's bytes.

    Returns ``None`` when the index has no entry for *chunk_idx* or the
    recorded artifact is gone.
    """
    index_path = index_dir / _INDEX_FILENAME
    if not index_path.exists():
        return None
    artifact_dir = json.loads(index_path.read_text()).get("extract", {}).get(str(chunk_idx))
    if artifact_dir is None:
        return None
    results = sorted(Path(artifact_dir).glob("calibration_extract_*.json"))
    if not results:
        return None
    return json.loads(results[-1].read_text())


def _variants(prompts_dir: Path, prefix: str, stage: str) -> dict[str, str]:
    """Build the ``prompt_variants`` mapping for *stage*.

    ``{production basename: variant basename}`` for every prompt this
    stage loads that the operator has a prefixed variant of. The server
    resolves the variant names against its own calibration prompt
    directory — the names travel, not the paths — so *prompts_dir* is the
    local view of that same directory.
    """
    mapping: dict[str, str] = {}
    for key in _STAGE_PROMPTS[stage]:
        basename = _STAGE_FILENAME[key]
        variant = f"{prefix}{basename}"
        if (prompts_dir / variant).exists():
            mapping[basename] = variant
    return mapping


# ---------------------------------------------------------------------------
# Stage runners — every stage POSTs to the server
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
    "enrich": "cloud_enrichment.txt",
    "plausibility": "cloud_plausibility.txt",
    "normalize_filter": "predicate_normalization.txt",
    # Graph-tier facts-only anonymize variant — see
    # configs/prompts/anonymization_facts.txt. Distinct from "anonymize"
    # (the session-tier, transcript-bearing template) above; POST
    # /calibrate/anonymize_facts, not /calibrate/anonymize.
    "anonymize_facts": "anonymization_facts.txt",
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
            "the later steps. 'normalize' and 'anonymize_facts' are standalone "
            "graph-level stages that require --snapshot and cannot be combined "
            "with chunk stages (or each other)."
        ),
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help=(
            "path to a graph_merged_snapshot.json (NetworkX node-link format). "
            "Required when --stages includes 'normalize' or 'anonymize_facts'. "
            "Cannot be combined with chunk stages (extract, anonymize, enrich, plausibility)."
        ),
    )
    parser.add_argument(
        "--prompts-dir",
        default=str(_REPO_ROOT / "data" / "ha" / "calibration" / "prompts"),
        help=(
            "Local view of the server's calibration prompt directory "
            "(paths.calibration/prompts). Variant NAMES travel to the server, "
            "which resolves them there; this path is only read to discover "
            "which variants exist. default: data/ha/calibration/prompts/"
        ),
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
    parser.add_argument("--chunk", type=int, default=None, help="run on this chunk index only")
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="default: data/ha/calibration/artifacts/<utc-timestamp>/",
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
            "second_order_extract, anonymize, entity_correction, cloud_enrich, "
            "anon_plausibility, deanon, deanon_plausibility. Default: run "
            "full pipeline."
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

    # Default dump dir under the calibration workspace (paths.calibration),
    # never under paths.debug: probing artifacts are not debug output and
    # must not depend on, or pollute, the production debug switch.
    if args.dump_dir is None:
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump_dir = _REPO_ROOT / "data" / "ha" / "calibration" / "artifacts" / ts
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
    elif "normalize" not in stages and "anonymize_facts" not in stages and "name" not in stages:
        raise SystemExit(
            "Error: --input is required for chunk stages "
            "(extract, anonymize, enrich, plausibility)."
        )
    else:
        chunks = []
        source_type = "document"

    # Decide whether baseline runs alongside candidate.
    def _run_baseline_for(stage: str) -> bool:
        if args.baseline == "none":
            return False
        if args.baseline == "production":
            return True
        # "auto": only worth a side-by-side when a variant exists to compare.
        return bool(_variants(prompts_dir, args.prompt_prefix, stage))

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
    # Guard: anonymize_facts cannot be combined with other stages — it uses
    # --snapshot (the same graph-level artifact "normalize" reads), not
    # --input.  Mirror the normalize mutual-exclusion pattern.
    if "anonymize_facts" in stages and stages != ["anonymize_facts"]:
        raise SystemExit(
            "Error: 'anonymize_facts' cannot be combined with other stages "
            "(extract, anonymize, enrich, plausibility, normalize, name). "
            "Run it in a separate invocation: --stages anonymize_facts --snapshot <path>"
        )
    if "anonymize_facts" in stages and not args.snapshot:
        raise SystemExit(
            "Error: --stages anonymize_facts requires --snapshot <graph_merged_snapshot.json>"
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
    write_artifact(dump_dir / "00_invocation.json", invocation)

    # Which server-side run covered which chunk. The runs themselves live
    # where the server put them (each response names its own
    # ``artifact_dir``); this side records only the mapping, so a run's
    # bytes exist in exactly one place.
    run_index: dict[str, dict[str, str]] = {}

    def _record_run(stage: str, key: str, response: dict) -> None:
        run_index.setdefault(stage, {})[key] = response.get("artifact_dir", "")

    # ----- run stages per chunk per seed ---------------------------------
    # Every chunk stage runs the SAME production chain server-side; what
    # differs is where it is entered and which step's output comes back.
    # So the client hands over exactly ONE artifact — the graph the local
    # extractor produced — and the chain re-derives everything downstream
    # of it by running. The old anonymize->enrich->plausibility relay
    # (anon facts, anonymized transcript, enriched facts, and a
    # client-side fail-closed check on the anonymizer's status) is gone:
    # each of those was a client-side copy of a decision the chain makes.
    _SEEDED_STAGES = ("anonymize", "enrich", "plausibility")
    for chunk in chunks:
        chunk_idx = chunk["chunk_index"]
        prior_extract: dict | None = None

        if args.seed_from:
            prior_extract = _load_run(Path(args.seed_from), chunk_idx)

        if "extract" in stages:
            extract_runs: list[dict] = []
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
                        "prompt_variants": _variants(prompts_dir, args.prompt_prefix, "extract"),
                        "stop_phase": args.stop_phase,
                        "params": {k: v for k, v in params.items() if v is not None},
                    },
                )
                extract_runs.append({"seed": seed, **candidate})
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
                        "prompt_variants": {},
                        "stop_phase": args.stop_phase,
                        "params": {k: v for k, v in params_base.items() if v is not None},
                    },
                )
            _record_run("extract", str(chunk_idx), extract_runs[0])
            # The runs are the server's record; what this file adds is the
            # comparison across them, which nothing else computes.
            out_blob: dict[str, Any] = {
                "stage": "extract",
                "chunk_index": chunk_idx,
                "runs": [
                    {"seed": r["seed"], "artifact_dir": r.get("artifact_dir", "")}
                    for r in extract_runs
                ],
            }
            if baseline_blob is not None:
                out_blob["baseline_artifact_dir"] = baseline_blob.get("artifact_dir", "")
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
            if "diff" in out_blob or "variance" in out_blob:
                write_artifact(dump_dir / f"01_extract_chunk_{chunk_idx}.json", out_blob)
            prior_extract = extract_runs[0]

        for ordinal, stage in enumerate(_SEEDED_STAGES, start=2):
            if stage not in stages:
                continue
            if prior_extract is None:
                print(
                    f"Chunk {chunk_idx}: no extract result to seed {stage!r} — "
                    f"run --stages extract first, or point --seed-from at a dump "
                    f"that contains 01_extract_chunk_{chunk_idx}.json."
                )
                continue
            params_stage = dict(params_base)
            if stage == "plausibility" and args.plausibility_max_tokens is not None:
                params_stage["max_tokens"] = args.plausibility_max_tokens
            result = _post_stage(
                args.server,
                stage,
                {
                    "transcript": chunk["text"],
                    "graph": prior_extract.get("parsed", {}),
                    "speaker_id": args.speaker_id,
                    # Same runtime-known speaker name every production
                    # caller threads through — a calibration run that never
                    # speaker-seeds diverges from production fidelity.
                    "speaker_name": args.speaker,
                    "source_type": source_type,
                    "session_id": f"calib-chunk-{chunk_idx}",
                    "prompt_variants": _variants(prompts_dir, args.prompt_prefix, stage),
                    "params": {k: v for k, v in params_stage.items() if v is not None},
                },
            )
            _record_run(stage, str(chunk_idx), result)
            print(f"  {stage} chunk {chunk_idx}: {result.get('artifact_dir', '<not recorded>')}")

    # ----- normalize stage (graph-level, runs outside the chunk loop) -------
    if "normalize" in stages:
        normalize_runs: list[dict] = []
        for seed in seeds:
            params = dict(params_base)
            params["seed"] = seed
            norm = _post_stage(
                args.server,
                "normalize",
                {
                    "snapshot_path": args.snapshot,
                    "prompt_variants": _variants(prompts_dir, args.prompt_prefix, "normalize"),
                    "params": {k: v for k, v in params.items() if v is not None},
                },
            )
            normalize_runs.append({"seed": seed, **norm})
        for run in normalize_runs:
            _record_run("normalize", str(run["seed"]), run)
        if len(seeds) > 1:
            write_artifact(
                dump_dir / "05_normalize.json",
                {
                    "stage": "normalize",
                    "snapshot_path": args.snapshot,
                    "variance": _variance_report("normalize", normalize_runs),
                },
            )

    # ----- anonymize_facts stage (graph-level, runs outside the chunk loop) -
    # The facts-only anonymize variant graph-tier enrichment uses
    # (configs/prompts/anonymization_facts.txt) — distinct from the
    # session-tier "anonymize" chunk stage above. Reads the same --snapshot
    # artifact "normalize" reads; the server derives facts + identity_domain
    # from it (paramem.server.calibrate.calibrate_anonymize_facts).
    if "anonymize_facts" in stages:
        anonymize_facts_runs: list[dict] = []
        for seed in seeds:
            params = dict(params_base)
            params["seed"] = seed
            af = _post_stage(
                args.server,
                "anonymize_facts",
                {
                    "snapshot_path": args.snapshot,
                    "prompt_variants": _variants(
                        prompts_dir, args.prompt_prefix, "anonymize_facts"
                    ),
                    "params": {k: v for k, v in params.items() if v is not None},
                },
            )
            anonymize_facts_runs.append({"seed": seed, **af})
        for run in anonymize_facts_runs:
            _record_run("anonymize_facts", str(run["seed"]), run)
        if len(seeds) > 1:
            write_artifact(
                dump_dir / "05b_anonymize_facts.json",
                {
                    "stage": "anonymize_facts",
                    "snapshot_path": args.snapshot,
                    "variance": _variance_report("anonymize_facts", anonymize_facts_runs),
                },
            )

    # ----- name stage (standalone — requires --turns-jsonl) ----------------
    if "name" in stages:
        turns_path = Path(args.turns_jsonl)
        with turns_path.open(encoding="utf-8") as _f:
            turns_data = [json.loads(line) for line in _f if line.strip()]
        name_runs: list[dict] = []
        for seed in seeds:
            params = dict(params_base)
            params["seed"] = seed
            name_result = _post_stage(
                args.server,
                "name",
                {
                    "turns": turns_data,
                    "prompt_variants": _variants(prompts_dir, args.prompt_prefix, "name"),
                    "user_turns_only": True,
                    "params": {k: v for k, v in params.items() if v is not None},
                },
            )
            name_runs.append({"seed": seed, **name_result})
        for run in name_runs:
            _record_run("name", str(run["seed"]), run)
        if len(seeds) > 1:
            write_artifact(
                dump_dir / "06_name.json",
                {
                    "stage": "name",
                    "turns_jsonl": str(turns_path),
                    "variance": _variance_report("name", name_runs),
                },
            )

    write_artifact(dump_dir / _INDEX_FILENAME, run_index)
    print(f"Calibration complete.  Index: {dump_dir / _INDEX_FILENAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
