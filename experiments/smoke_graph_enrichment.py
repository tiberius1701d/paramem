"""Graph-level SOTA enrichment smoke test (Task #10 live verification).

Loads Test 8's final cumulative graph (623 nodes, 550 edges, 55 cycles of
real PerLTQA extraction), chunks by N-hop ego neighborhoods, calls the
real Anthropic SOTA endpoint, and reports what the enrichment prompt
actually emits end-to-end.

Read-only against the Test 8 artifact. CPU-only — no GPU, no adapter
changes. Outputs go to outputs/smoke_graph_enrichment/<timestamp>/.

Usage:
    # .env must export ANTHROPIC_API_KEY
    export $(grep -v '^#' .env | xargs)
    python experiments/smoke_graph_enrichment.py \\
        [--max-chunks N] [--hops H] [--max-entities-per-pass C] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx

from paramem.graph.extractor import PROVIDER_KEY_ENV, _graph_enrich_with_sota
from paramem.training.consolidation import (
    _safe_to_merge_surface,
    _serialize_subgraph_triples,
)

DEFAULT_GRAPH = Path(
    "outputs/test8_large_scale/mistral/20260323_161747/cycle_056/cumulative_graph.json"
)
DEFAULT_OUT_DIR = Path("outputs/smoke_graph_enrichment")
PROVIDER = "anthropic"
MODEL = "claude-sonnet-4-6"


def load_graph(path: Path) -> nx.MultiDiGraph:
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def build_chunks(
    graph: nx.MultiDiGraph, hops: int, max_entities: int, chunk_cap: int
) -> list[list[str]]:
    """Mirror ConsolidationLoop._run_graph_enrichment chunking exactly."""
    nodes_by_recurrence = sorted(
        graph.nodes(data=True),
        key=lambda nd: nd[1].get("recurrence_count", 0),
        reverse=True,
    )
    undirected = graph.to_undirected(as_view=True)
    seen: set[frozenset] = set()
    chunks: list[list[str]] = []
    for focal, _ in nodes_by_recurrence:
        if len(chunks) >= chunk_cap:
            break
        if focal not in undirected:
            continue
        ego = nx.ego_graph(undirected, focal, radius=hops)
        nodes = list(ego.nodes)
        if len(nodes) > max_entities:
            neighbours = sorted(
                (n for n in nodes if n != focal),
                key=lambda n: undirected.degree(n),
                reverse=True,
            )
            nodes = [focal] + neighbours[: max_entities - 1]
        key = frozenset(nodes)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(nodes)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-entities-per-pass", type=int, default=50, dest="max_entities")
    ap.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Cap total SOTA calls (0 = full cover per the default chunk_cap formula)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunk shapes + triple counts without calling SOTA",
    )
    args = ap.parse_args()

    if not args.graph.exists():
        print(f"FATAL: graph not found at {args.graph}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading graph: {args.graph}")
    graph = load_graph(args.graph)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"  nodes={n_nodes} edges={n_edges}")

    chunk_cap = max(1, math.ceil(n_nodes / max(1, args.max_entities)))
    if args.max_chunks > 0:
        chunk_cap = min(chunk_cap, args.max_chunks)
    print(
        f"Chunking: hops={args.hops} max_entities_per_pass={args.max_entities} "
        f"chunk_cap={chunk_cap}"
    )

    chunks = build_chunks(graph, args.hops, args.max_entities, chunk_cap)
    print(f"Built {len(chunks)} distinct chunks (post-dedup)")
    for i, chunk in enumerate(chunks):
        sub = graph.subgraph(chunk)
        triples = _serialize_subgraph_triples(sub)
        print(f"  chunk {i:02d}: nodes={len(chunk)} triples={len(triples)}")

    if args.dry_run:
        print("\n--dry-run: not calling SOTA. Exiting.")
        return

    provider = PROVIDER
    api_key = os.environ.get(PROVIDER_KEY_ENV.get(provider, ""), "")
    if not api_key:
        print(
            f"FATAL: {PROVIDER_KEY_ENV[provider]} not set. "
            "`export $(grep -v '^#' .env | xargs)` first.",
            file=sys.stderr,
        )
        sys.exit(3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_sub = args.out_dir / ts
    out_sub.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {out_sub}")

    chunks_log: list[dict] = []
    all_relations: list[dict] = []
    all_same_as: list[list[str]] = []
    total_new_edges_if_applied = 0
    total_merges_if_applied = 0
    total_same_as_rejected_dup = 0
    total_same_as_rejected_gate = 0
    knows_count = 0
    seen_merge_keys: set[frozenset] = set()
    t0 = time.perf_counter()

    for i, chunk in enumerate(chunks):
        sub = graph.subgraph(chunk)
        triples = _serialize_subgraph_triples(sub)
        print(
            f"[chunk {i:02d}/{len(chunks)}] nodes={len(chunk)} triples={len(triples)} → SOTA ...",
            end="",
            flush=True,
        )
        t_chunk = time.perf_counter()
        try:
            result = _graph_enrich_with_sota(
                triples,
                api_key,
                provider,
                MODEL,
                None,
            )
        except Exception as exc:
            print(f" EXC {type(exc).__name__}: {exc}")
            chunks_log.append(
                {
                    "index": i,
                    "nodes": chunk,
                    "n_triples": len(triples),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        dt = time.perf_counter() - t_chunk

        if result is None:
            print(f" None ({dt:.1f}s)")
            chunks_log.append(
                {
                    "index": i,
                    "nodes": chunk,
                    "n_triples": len(triples),
                    "error": "result_none",
                    "elapsed_s": round(dt, 2),
                }
            )
            continue

        new_rels, same_as_pairs, raw = result

        # Count what would actually land (confidence >= 0.7, valid rtype, subj != obj)
        valid_rtypes = {"factual", "temporal", "preference", "social"}
        would_land = 0
        chunk_knows = 0
        for rel in new_rels:
            if not isinstance(rel, dict):
                continue
            try:
                conf = float(rel.get("confidence", 0.8))
            except (TypeError, ValueError):
                conf = 0.8
            if conf < 0.7:
                continue
            rt = rel.get("relation_type", "factual")
            if rt not in valid_rtypes:
                rt = "factual"
            if (
                rel.get("subject")
                and rel.get("object")
                and rel.get("subject") != rel.get("object")
                and rel.get("predicate")
            ):
                would_land += 1
                if str(rel.get("predicate", "")).lower() == "knows":
                    chunk_knows += 1

        # Apply the production same_as gate: endpoints-in-graph +
        # unordered-pair dedup across the whole pass + surface-form safety.
        valid_same_as = 0
        chunk_dup_rejects = 0
        chunk_gate_rejects = 0
        for p in same_as_pairs:
            if not (len(p) == 2 and p[0] and p[1] and p[0] != p[1]):
                continue
            if p[0] not in graph or p[1] not in graph:
                continue
            merge_key = frozenset({p[0].lower(), p[1].lower()})
            if merge_key in seen_merge_keys:
                chunk_dup_rejects += 1
                continue
            seen_merge_keys.add(merge_key)
            if not _safe_to_merge_surface(p[0], p[1]):
                chunk_gate_rejects += 1
                continue
            valid_same_as += 1

        total_new_edges_if_applied += would_land
        total_merges_if_applied += valid_same_as
        total_same_as_rejected_dup += chunk_dup_rejects
        total_same_as_rejected_gate += chunk_gate_rejects
        knows_count += chunk_knows
        all_relations.extend(new_rels)
        all_same_as.extend(same_as_pairs)

        print(
            f" OK ({dt:.1f}s) rels_raw={len(new_rels)} rels_kept={would_land} "
            f"knows={chunk_knows} same_as_raw={len(same_as_pairs)} "
            f"same_as_kept={valid_same_as} dup={chunk_dup_rejects} "
            f"gate_rej={chunk_gate_rejects}"
        )
        chunks_log.append(
            {
                "index": i,
                "nodes": chunk,
                "n_triples": len(triples),
                "elapsed_s": round(dt, 2),
                "relations_raw": len(new_rels),
                "relations_would_land": would_land,
                "knows_count": chunk_knows,
                "same_as_raw": len(same_as_pairs),
                "same_as_would_apply": valid_same_as,
                "same_as_rejected_dup": chunk_dup_rejects,
                "same_as_rejected_gate": chunk_gate_rejects,
                "raw_response": raw,
                "input_triples": triples,
                "returned_relations": new_rels,
                "returned_same_as": same_as_pairs,
            }
        )

    total_dt = time.perf_counter() - t0

    # Persist
    (out_sub / "chunks.json").write_text(json.dumps(chunks_log, indent=2))
    (out_sub / "relations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_relations) + "\n" if all_relations else ""
    )
    (out_sub / "same_as.jsonl").write_text(
        "\n".join(json.dumps(p) for p in all_same_as) + "\n" if all_same_as else ""
    )

    knows_share = (
        round(knows_count / total_new_edges_if_applied, 3) if total_new_edges_if_applied else 0.0
    )
    summary = {
        "graph": str(args.graph),
        "nodes": n_nodes,
        "edges": n_edges,
        "chunks_attempted": len(chunks),
        "chunks_ok": sum(1 for c in chunks_log if "error" not in c),
        "chunks_failed": sum(1 for c in chunks_log if "error" in c),
        "relations_raw_total": len(all_relations),
        "relations_would_land": total_new_edges_if_applied,
        "knows_count": knows_count,
        "knows_share_of_kept": knows_share,
        "same_as_raw_total": len(all_same_as),
        "same_as_would_apply": total_merges_if_applied,
        "same_as_rejected_dup": total_same_as_rejected_dup,
        "same_as_rejected_gate": total_same_as_rejected_gate,
        "wall_clock_s": round(total_dt, 2),
        "provider": provider,
        "model": MODEL,
        "hops": args.hops,
        "max_entities_per_pass": args.max_entities,
    }
    (out_sub / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nArtifacts: {out_sub}")


if __name__ == "__main__":
    main()
