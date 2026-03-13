"""Forgetting metrics and retention analysis for sequential learning."""

from dataclasses import dataclass, field


@dataclass
class ForgettingMetrics:
    """Forgetting metrics for a single topic across training steps."""

    topic_name: str
    peak_recall: float
    final_recall: float
    forgetting_rate: float
    recall_curve: list[float] = field(default_factory=list)


def compute_forgetting_metrics(
    results: list,
) -> dict[str, ForgettingMetrics]:
    """Compute per-topic forgetting from sequential training results.

    Args:
        results: List of SequentialResult from train_sequential().

    Returns:
        Dict mapping topic name to ForgettingMetrics.
    """
    # Collect all unique topic names
    topic_names = list(dict.fromkeys(r.topic_name for r in results))

    metrics = {}
    for topic_name in topic_names:
        # Build recall curve: value at each step (None if not yet trained)
        curve = [
            r.recall_per_topic[topic_name] for r in results if topic_name in r.recall_per_topic
        ]

        if not curve:
            continue

        peak = max(curve)
        final = curve[-1]
        forgetting_rate = (peak - final) / peak if peak > 0 else 0.0

        metrics[topic_name] = ForgettingMetrics(
            topic_name=topic_name,
            peak_recall=peak,
            final_recall=final,
            forgetting_rate=forgetting_rate,
            recall_curve=curve,
        )

    return metrics


def compute_forgetting_reduction(
    baseline_metrics: dict[str, ForgettingMetrics],
    strategy_metrics: dict[str, ForgettingMetrics],
) -> dict:
    """Compare forgetting rates between baseline and a mitigation strategy.

    Returns:
        Dict with overall_reduction (fraction), per_topic breakdown,
        and mean forgetting rates for both conditions.
    """
    # Only compare topics present in both
    common_topics = set(baseline_metrics.keys()) & set(strategy_metrics.keys())
    if not common_topics:
        return {
            "overall_reduction": 0.0,
            "baseline_mean_forgetting": 0.0,
            "strategy_mean_forgetting": 0.0,
            "per_topic": {},
        }

    per_topic = {}
    baseline_rates = []
    strategy_rates = []

    for topic in sorted(common_topics):
        b_rate = baseline_metrics[topic].forgetting_rate
        s_rate = strategy_metrics[topic].forgetting_rate
        baseline_rates.append(b_rate)
        strategy_rates.append(s_rate)

        reduction = (b_rate - s_rate) / b_rate if b_rate > 0 else 0.0
        per_topic[topic] = {
            "baseline_forgetting": b_rate,
            "strategy_forgetting": s_rate,
            "reduction": reduction,
        }

    baseline_mean = sum(baseline_rates) / len(baseline_rates)
    strategy_mean = sum(strategy_rates) / len(strategy_rates)
    overall_reduction = (
        (baseline_mean - strategy_mean) / baseline_mean if baseline_mean > 0 else 0.0
    )

    return {
        "overall_reduction": overall_reduction,
        "baseline_mean_forgetting": baseline_mean,
        "strategy_mean_forgetting": strategy_mean,
        "per_topic": per_topic,
    }


def format_results_table(
    all_conditions: dict[str, dict[str, ForgettingMetrics]],
) -> str:
    """Format a comparison table of forgetting metrics across conditions."""
    if not all_conditions:
        return "No results to display."

    # Collect all unique topic names
    topic_names = []
    for metrics in all_conditions.values():
        for name in metrics:
            if name not in topic_names:
                topic_names.append(name)

    conditions = list(all_conditions.keys())

    # Header
    lines = []
    header = f"{'Topic':<15}"
    for cond in conditions:
        header += f" | {cond:>20}"
    lines.append(header)
    lines.append("-" * len(header))

    # Per-topic rows: show "peak → final (forgetting%)"
    for topic in topic_names:
        row = f"{topic:<15}"
        for cond in conditions:
            m = all_conditions[cond].get(topic)
            if m is None:
                row += f" | {'N/A':>20}"
            else:
                cell = f"{m.peak_recall:.2f}→{m.final_recall:.2f} ({m.forgetting_rate:.0%})"
                row += f" | {cell:>20}"
        lines.append(row)

    # Mean forgetting row
    lines.append("-" * len(header))
    row = f"{'Mean forgetting':<15}"
    for cond in conditions:
        metrics = all_conditions[cond]
        rates = [m.forgetting_rate for m in metrics.values()]
        mean = sum(rates) / len(rates) if rates else 0.0
        row += f" | {mean:>19.1%} "
    lines.append(row)

    return "\n".join(lines)
