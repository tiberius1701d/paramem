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
  success            — non-empty mapping, every expected name absent
                       from anon_text (privacy contract), round-trip
                       preserves the original text (whitespace-
                       normalised).  A name being absent from the
                       mapping is fine if the name also doesn't
                       appear in anon_text — the cloud sees nothing
                       to deanonymize either way.
  leak_blocked       — anonymizer returned ('', {}) — the wrapper's
                       repair-and-verify pipeline blocked the call.
                       Privacy-safe (the cloud call doesn't happen),
                       but counts as "anonymizer failed to deliver".
  privacy_leak       — mapping non-empty but at least one expected
                       name still appears in anon_text.  This is a
                       hard failure: extraction + NER both missed
                       the name and the cloud would receive it
                       verbatim.
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

# Re-use the contract test's fixture and threshold as the single source
# of truth — calibration and CI must agree on the bar they're measuring
# against, otherwise the calibrator's "currently shipped" line drifts
# from reality after every threshold bump.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
try:
    from test_cloud_anonymizer_contract_gpu import _FIXTURE, _MATCH_THRESHOLD  # noqa: E402
except ImportError as e:
    raise SystemExit(
        f"could not import _FIXTURE / _MATCH_THRESHOLD from "
        f"tests/test_cloud_anonymizer_contract_gpu.py: {e}"
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
    # Privacy contract: every expected name (and every mapping key) must
    # be absent from anon_text.  A name leaking means the cloud sees it.
    leaked_keys = [n for n in mapping if re.search(r"\b" + re.escape(n) + r"\b", anon_text)]
    leaked_expected = [
        n for n in expected_names if re.search(r"\b" + re.escape(n) + r"\b", anon_text)
    ]
    if leaked_keys or leaked_expected:
        return "privacy_leak"
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


def _run_one(
    transcript: str,
    expected_names: list[str],
    model,
    tokenizer,
    *,
    speaker_name: str,
    pii_scope: set[str],
) -> tuple[str, dict, str]:
    from paramem.graph.extractor import (
        deanonymize_text,
        extract_and_anonymize_for_cloud,
    )

    anon_text, mapping = extract_and_anonymize_for_cloud(
        transcript,
        model,
        tokenizer,
        speaker_name=speaker_name,
        pii_scope=pii_scope,
    )
    if not mapping or not anon_text:
        return anon_text or "", mapping or {}, ""
    round_trip = deanonymize_text(anon_text, mapping)
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
    for outcome in ("success", "leak_blocked", "privacy_leak", "round_trip_failed"):
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
        print(f"  Currently shipped value      : {_MATCH_THRESHOLD:.2f}")
    return personal_success, rate


def _print_per_query(results: list[QueryResult]) -> None:
    print("\n" + "-" * 72)
    print("Per-query detail")
    print("-" * 72)
    for r in results:
        outcome_marker = {
            "success": "OK",
            "leak_blocked": "BLK",
            "privacy_leak": "LEAK",
            "round_trip_failed": "RT!",
        }[r.outcome]
        print(f"\n[{outcome_marker}] {r.id} (iter {r.iteration})")
        print(f"  query    : {r.query!r}")
        print(f"  expected : {r.expected_names}")
        print(f"  mapping  : {r.mapping}")
        print(f"  anon     : {r.anon_text!r}")
        if r.outcome == "privacy_leak":
            leaked = [
                n
                for n in (list(r.mapping) + r.expected_names)
                if re.search(r"\b" + re.escape(n) + r"\b", r.anon_text)
            ]
            print(f"  leaked   : {sorted(set(leaked))}")
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
    parser.add_argument(
        "--scope",
        nargs="*",
        default=None,
        help=(
            "NER categories to anonymize (e.g. --scope person place). "
            "Defaults to the production default in server.yaml.example "
            "(read from disk to stay in sync with the shipped config). "
            "Pass --scope with no values to disable anonymization."
        ),
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
    from paramem.server.config import load_server_config

    # Load the CI test fixture so the calibration target matches what
    # ``tests/test_cloud_anonymizer_contract_gpu.py`` runs against —
    # same model (Mistral 7B), same default cloud_scope.  Loading
    # ``configs/server.yaml.example`` instead would re-anchor the
    # calibration whenever the shipped template's ship-default drifts
    # (e.g. cloud_mode block ↔ anonymize) without changing what the
    # contract test actually exercises.  Per CLAUDE.md, calibration
    # and tests share the fixture as the single calibration anchor.
    server_cfg = load_server_config("tests/fixtures/server.yaml")
    model_cfg = server_cfg.model_config
    print(f"  model: {model_cfg.model_id}")
    model, tokenizer = load_base_model(model_cfg)
    print("  ready")

    # Resolve the scope.  CLI override wins; otherwise inherit from the
    # fixture so the calibration result reflects the contract test's
    # default scope unless explicitly varied.
    if args.scope is None:
        scope = set(server_cfg.sanitization.cloud_scope)
    else:
        scope = set(args.scope)
    print(f"  scope: {sorted(scope) or '[]  (anonymization disabled)'}")

    if args.query is not None:
        # Ad-hoc input: wrap as a single-turn transcript so the helper
        # gets the production input shape.  Caller intent is "treat
        # this as one [user] turn".  Synthesize a speaker_name so the
        # extraction prompt's {SPEAKER_NAME} slot resolves cleanly --
        # production always has a real speaker by the time cloud
        # egress runs (greeting flow).
        entries = [
            {
                "id": "ad-hoc",
                "speaker_name": "Anna",
                "transcript": f"[user] {args.query}",
                "expected_names": [],
            }
        ]
    else:
        entries = list(_FIXTURE)

    results: list[QueryResult] = []
    for entry in entries:
        for iteration in range(args.repeat):
            print(f"\n[{entry['id']} iter {iteration}]")
            anon_text, mapping, round_trip = _run_one(
                entry["transcript"],
                entry["expected_names"],
                model,
                tokenizer,
                speaker_name=entry["speaker_name"],
                pii_scope=scope,
            )
            outcome = _classify(
                expected_names=entry["expected_names"],
                anon_text=anon_text,
                mapping=mapping,
                round_trip=round_trip,
                original=entry["transcript"],
            )
            print(f"  outcome: {outcome}")
            results.append(
                QueryResult(
                    id=entry["id"],
                    query=entry["transcript"],
                    expected_names=entry["expected_names"],
                    anon_text=anon_text,
                    mapping=mapping,
                    round_trip=round_trip,
                    outcome=outcome,
                    iteration=iteration,
                )
            )

    _print_per_query(results)
    total_personal = sum(1 for e in entries if e["expected_names"]) * args.repeat
    _print_summary(results, total_personal)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(
                {
                    "schema": "cloud_anonymizer_calibration.v1",
                    "model": model_cfg.model_id,
                    "queries_per_iteration": len(entries),
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
