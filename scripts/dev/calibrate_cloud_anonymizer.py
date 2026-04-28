"""Empirical calibration tool for the cloud_anonymizer contract test.

``tests/test_cloud_anonymizer_contract_gpu.py`` ships with
``_MATCH_THRESHOLD = 0.6`` as an initial guess.  Run this script against
a real GPU + model to measure the actual Mistral 7B baseline on the
shipped fixture, eyeball failure-mode distribution, and recommend a
calibrated threshold.

Mirrors the calibration pattern of
``tests/test_plausibility_contract_gpu.py`` (75% measured baseline).
The script itself is GPU-free in code; the run needs GPU.

Usage::

    set -a && source .env && set +a && \\
      /home/tiberius/miniforge3/envs/paramem/bin/python \\
      scripts/dev/calibrate_cloud_anonymizer.py

Optional flags:
  ``--query "your query"`` — calibrate against an ad-hoc query
                             (skips the shipped fixture).
  ``--repeat N``           — re-run each query N times to estimate
                             variance (default 1; temperature is 0,
                             so >1 mostly catches non-determinism in
                             tokenization or sampling fallbacks).
  ``--out path.json``      — write the full per-query record to JSON
                             for offline analysis.

Outcome classification per query:
  success            — non-empty mapping, no leak, every expected name
                       in mapping, round-trip preserves the original
                       text (whitespace-normalised).
  leak_blocked       — anonymizer returned ('', {}) — the wrapper's
                       forward leak guard tripped or the model
                       returned an empty result.  This is privacy-
                       safe (the cloud call would have been blocked),
                       but counts as "anonymizer failed to deliver"
                       for calibration.
  missing_coverage   — mapping non-empty but at least one expected
                       name from the fixture is absent.  The
                       forward gate did NOT block, but the cloud
                       would receive a partially anonymized query.
  round_trip_failed  — mapping non-empty, no leak, but the response
                       fails to whitespace-equal the original after
                       deanon.  Indicates lossy whitespace handling
                       in the anonymizer prompt.

The script does NOT update ``_MATCH_THRESHOLD`` automatically.
Eyeball the report and update the test docstring + threshold by
hand to keep the calibration decision auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Re-use the contract test's fixture as the single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
try:
    from test_cloud_anonymizer_contract_gpu import _FIXTURE  # noqa: E402
except ImportError as e:
    raise SystemExit(
        f"could not import _FIXTURE from tests/test_cloud_anonymizer_contract_gpu.py: {e}"
    )


def _normalise(text: str) -> str:
    """Whitespace-normalise for round-trip comparison."""
    return " ".join(text.split())


def _classify(
    *,
    expected_names: list[str],
    anon_text: str,
    mapping: dict[str, str],
    round_trip: str,
    original: str,
) -> str:
    if not mapping:
        return "leak_blocked"
    # Forward leak-guard contract: anon_text shouldn't contain any real name
    leaked = [n for n in mapping if re.search(r"\b" + re.escape(n) + r"\b", anon_text)]
    if leaked:
        return "leak_blocked"
    missing = [n for n in expected_names if n not in mapping]
    if missing:
        return "missing_coverage"
    if _normalise(round_trip) != _normalise(original):
        return "round_trip_failed"
    return "success"


@dataclass
class QueryResult:
    id: str
    query: str
    expected_names: list[str]
    anon_text: str
    mapping: dict[str, str]
    round_trip: str
    outcome: str
    iteration: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _run_one(query: str, expected_names: list[str], model, tokenizer) -> tuple[str, dict, str]:
    from paramem.server.cloud_anonymizer import (
        anonymize_outbound,
        deanonymize_inbound,
    )

    anon_text, mapping = anonymize_outbound(query, model, tokenizer)
    if not mapping or not anon_text:
        return anon_text or "", mapping or {}, ""
    # Synthetic cloud-response = the anon_text itself; the round-trip checks
    # that deanon restores the names without altering surrounding tokens.
    round_trip = deanonymize_inbound(anon_text, mapping)
    return anon_text, mapping, round_trip


def _print_summary(results: list[QueryResult], total_personal: int) -> tuple[int, float]:
    by_outcome: dict[str, int] = {}
    for r in results:
        by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1

    print("\n" + "=" * 72)
    print("Calibration summary")
    print("=" * 72)
    print(f"  total queries scanned : {len(results)}")
    print(f"  with personal markers : {total_personal}")
    for outcome in ("success", "leak_blocked", "missing_coverage", "round_trip_failed"):
        n = by_outcome.get(outcome, 0)
        if n:
            pct = 100.0 * n / max(1, len(results))
            print(f"  {outcome:<20s}: {n:>3d}  ({pct:5.1f}%)")

    # Personal-fixture pass rate (the metric the contract test gates on)
    personal_results = [r for r in results if r.expected_names]
    personal_success = sum(1 for r in personal_results if r.outcome == "success")
    rate = personal_success / max(1, len(personal_results))
    print()
    print(
        f"  Personal-fixture success rate: {personal_success}/{len(personal_results)} ({rate:.1%})"
    )

    # Recommended threshold: 0.9× measured rate, floored to the nearest 0.05,
    # so a single transient failure doesn't trip the test.  Mirrors the
    # plausibility-contract 75% calibration logic.
    if personal_results:
        recommended = math.floor(rate * 0.9 * 20) / 20.0
        print(
            f"  Recommended _MATCH_THRESHOLD : {recommended:.2f}  (0.9 × measured, floored to 0.05)"
        )
        print("  Currently shipped value      : 0.60")
    return personal_success, rate


def _print_per_query(results: list[QueryResult]) -> None:
    print("\n" + "-" * 72)
    print("Per-query detail")
    print("-" * 72)
    for r in results:
        outcome_marker = {
            "success": "OK",
            "leak_blocked": "BLK",
            "missing_coverage": "MISS",
            "round_trip_failed": "RT!",
        }[r.outcome]
        print(f"\n[{outcome_marker}] {r.id} (iter {r.iteration})")
        print(f"  query    : {r.query!r}")
        print(f"  expected : {r.expected_names}")
        print(f"  mapping  : {r.mapping}")
        print(f"  anon     : {r.anon_text!r}")
        if r.outcome == "missing_coverage":
            missing = [n for n in r.expected_names if n not in r.mapping]
            print(f"  missing  : {missing}")
        if r.outcome == "round_trip_failed":
            print(f"  round_trip: {r.round_trip!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ad-hoc query to calibrate against (skips the shipped fixture)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Re-run each query N times for variance estimation (default 1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: write the full per-query record to this JSON path",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("PARAMEM_DAILY_PASSPHRASE"):
        # The anonymizer reads the live local model; it doesn't need the
        # daily passphrase.  But upstream config loading sometimes does.
        # Warn rather than fail -- if it actually breaks something, the
        # underlying call will surface the missing key.
        print(
            "WARN: PARAMEM_DAILY_PASSPHRASE not set; encrypted-config paths may fail",
            file=sys.stderr,
        )

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    print("=" * 72)
    print("Cloud anonymizer calibration")
    print("=" * 72)

    print("\nLoading local model + tokenizer...")
    from paramem.models.loader import load_base_model
    from paramem.utils.config import load_config

    cfg = load_config()
    print(f"  model: {cfg.model.model_id}")
    model, tokenizer = load_base_model(cfg.model)
    print("  ready")

    if args.query is not None:
        queries = [{"id": "ad-hoc", "query": args.query, "expected_names": []}]
    else:
        queries = list(_FIXTURE)

    results: list[QueryResult] = []
    for entry in queries:
        for iteration in range(args.repeat):
            print(f"\n[{entry['id']} iter {iteration}] query: {entry['query']!r}")
            anon_text, mapping, round_trip = _run_one(
                entry["query"], entry["expected_names"], model, tokenizer
            )
            outcome = _classify(
                expected_names=entry["expected_names"],
                anon_text=anon_text,
                mapping=mapping,
                round_trip=round_trip,
                original=entry["query"],
            )
            print(f"  outcome: {outcome}")
            results.append(
                QueryResult(
                    id=entry["id"],
                    query=entry["query"],
                    expected_names=entry["expected_names"],
                    anon_text=anon_text,
                    mapping=mapping,
                    round_trip=round_trip,
                    outcome=outcome,
                    iteration=iteration,
                )
            )

    _print_per_query(results)
    total_personal = sum(1 for q in queries if q["expected_names"]) * args.repeat
    _print_summary(results, total_personal)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(
                {
                    "schema": "cloud_anonymizer_calibration.v1",
                    "model": cfg.model.model_id,
                    "queries_per_iteration": len(queries),
                    "repeat": args.repeat,
                    "results": [r.to_dict() for r in results],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nFull record written to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
