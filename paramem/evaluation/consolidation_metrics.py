"""Metrics for evaluating the consolidation loop (Phase 3).

Tracks promoted memory retention, episodic decay rates,
and semantic adapter stability across consolidation cycles.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationMetrics:
    """Aggregate metrics across consolidation cycles."""

    total_cycles: int = 0
    total_entities_extracted: int = 0
    total_promotions: int = 0
    total_decays: int = 0
    mean_wall_clock_seconds: float = 0.0

    # Per-cycle tracking
    episodic_losses: list[float] = field(default_factory=list)
    semantic_losses: list[float] = field(default_factory=list)
    promotions_per_cycle: list[int] = field(default_factory=list)
    decays_per_cycle: list[int] = field(default_factory=list)
    wall_clock_per_cycle: list[float] = field(default_factory=list)


def compute_consolidation_metrics(cycle_results: list) -> ConsolidationMetrics:
    """Compute aggregate metrics from a list of CycleResults.

    Args:
        cycle_results: List of CycleResult from ConsolidationLoop.

    Returns:
        ConsolidationMetrics with aggregated stats.
    """
    metrics = ConsolidationMetrics()
    metrics.total_cycles = len(cycle_results)

    for result in cycle_results:
        metrics.total_entities_extracted += result.entities_extracted
        metrics.total_promotions += result.nodes_promoted
        metrics.total_decays += result.nodes_decayed
        metrics.promotions_per_cycle.append(result.nodes_promoted)
        metrics.decays_per_cycle.append(result.nodes_decayed)
        metrics.wall_clock_per_cycle.append(result.wall_clock_seconds)

        if result.episodic_train_loss is not None:
            metrics.episodic_losses.append(result.episodic_train_loss)
        if result.semantic_train_loss is not None:
            metrics.semantic_losses.append(result.semantic_train_loss)

    if metrics.wall_clock_per_cycle:
        metrics.mean_wall_clock_seconds = sum(metrics.wall_clock_per_cycle) / len(
            metrics.wall_clock_per_cycle
        )

    return metrics


def compute_promoted_retention(
    recall_scores: list[dict[str, float]],
    promoted_nodes: set[str],
) -> dict:
    """Compute retention rate for promoted memories over cycles.

    Args:
        recall_scores: List of per-cycle dicts mapping fact/node to recall score.
        promoted_nodes: Set of nodes that have been promoted.

    Returns:
        Dict with retention stats.
    """
    if not recall_scores or not promoted_nodes:
        return {"mean_retention": 0.0, "min_retention": 0.0, "per_node": {}}

    # Track each promoted node's recall across cycles
    node_recalls = {node: [] for node in promoted_nodes}
    for cycle_scores in recall_scores:
        for node in promoted_nodes:
            score = cycle_scores.get(node, 0.0)
            node_recalls[node].append(score)

    per_node = {
        node: {
            "mean": sum(scores) / len(scores),
            "final": scores[-1],
            "peak": max(scores),
        }
        for node, scores in node_recalls.items()
        if scores
    }

    retention_values = [v["final"] for v in per_node.values() if v["final"] > 0]
    mean_retention = sum(retention_values) / len(retention_values) if retention_values else 0.0
    min_retention = min(retention_values) if retention_values else 0.0

    return {
        "mean_retention": mean_retention,
        "min_retention": min_retention,
        "per_node": per_node,
    }


def compute_episodic_decay_rate(
    recall_scores: list[dict[str, float]],
    decayed_nodes: set[str],
    decay_start_cycle: dict[str, int],
) -> dict:
    """Measure how quickly unreinforced episodic memories decay.

    Args:
        recall_scores: Per-cycle recall dicts.
        decayed_nodes: Set of nodes that were decayed.
        decay_start_cycle: Dict mapping node -> cycle when decay started.

    Returns:
        Dict with decay rate stats.
    """
    if not recall_scores or not decayed_nodes:
        return {"mean_decay_rate": 0.0, "per_node": {}}

    per_node = {}
    for node in decayed_nodes:
        start = decay_start_cycle.get(node, 0)
        scores_after_decay = [
            recall_scores[i].get(node, 0.0) for i in range(start, len(recall_scores))
        ]

        if len(scores_after_decay) >= 2:
            initial = scores_after_decay[0] if scores_after_decay[0] > 0 else 1.0
            final = scores_after_decay[-1]
            decay_rate = (initial - final) / initial if initial > 0 else 0.0
            per_node[node] = {
                "initial": scores_after_decay[0],
                "final": final,
                "decay_rate": decay_rate,
                "cycles_observed": len(scores_after_decay),
            }

    rates = [v["decay_rate"] for v in per_node.values()]
    mean_rate = sum(rates) / len(rates) if rates else 0.0

    return {
        "mean_decay_rate": mean_rate,
        "per_node": per_node,
    }


def compute_semantic_drift(
    recall_scores: list[dict[str, float]],
    semantic_nodes: set[str],
    baseline_cycle: int = 0,
) -> dict:
    """Measure semantic adapter stability (drift from baseline).

    Args:
        recall_scores: Per-cycle recall dicts for the semantic adapter.
        semantic_nodes: Set of nodes in the semantic adapter.
        baseline_cycle: Cycle index to use as the baseline.

    Returns:
        Dict with drift stats. Target: <5% drift.
    """
    if not recall_scores or not semantic_nodes or baseline_cycle >= len(recall_scores):
        return {"mean_drift": 0.0, "max_drift": 0.0, "per_node": {}}

    baseline = recall_scores[baseline_cycle]
    latest = recall_scores[-1]

    per_node = {
        node: {
            "baseline": baseline.get(node, 0.0),
            "latest": latest.get(node, 0.0),
            "drift": abs(latest.get(node, 0.0) - baseline.get(node, 0.0)),
        }
        for node in semantic_nodes
    }

    drifts = [v["drift"] for v in per_node.values()]
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    max_drift = max(drifts) if drifts else 0.0

    return {
        "mean_drift": mean_drift,
        "max_drift": max_drift,
        "per_node": per_node,
    }


def format_phase3_summary(
    consolidation_metrics: ConsolidationMetrics,
    retention: dict,
    decay: dict,
    drift: dict,
) -> str:
    """Format a human-readable summary of Phase 3 results."""
    lines = [
        "=" * 60,
        "Phase 3: Consolidation Loop Results",
        "=" * 60,
        f"Total cycles: {consolidation_metrics.total_cycles}",
        f"Total entities extracted: {consolidation_metrics.total_entities_extracted}",
        f"Total promotions: {consolidation_metrics.total_promotions}",
        f"Total decays: {consolidation_metrics.total_decays}",
        f"Mean wall-clock per cycle: {consolidation_metrics.mean_wall_clock_seconds:.1f}s",
        "",
        "--- Targets ---",
        f"Promoted retention: {retention['mean_retention']:.1%} (target: >80%)",
        f"  {'PASS' if retention['mean_retention'] >= 0.80 else 'FAIL'}",
        f"Episodic decay rate: {decay['mean_decay_rate']:.1%} (target: measurable >0%)",
        f"  {'PASS' if decay['mean_decay_rate'] > 0 else 'FAIL'}",
        f"Semantic drift: {drift['mean_drift']:.1%} (target: <5%)",
        f"  {'PASS' if drift['mean_drift'] < 0.05 else 'FAIL'}",
        f"Wall-clock per cycle: "
        f"{consolidation_metrics.mean_wall_clock_seconds:.0f}s (target: <1800s)",
        f"  {'PASS' if consolidation_metrics.mean_wall_clock_seconds < 1800 else 'FAIL'}",
        "=" * 60,
    ]
    return "\n".join(lines)
