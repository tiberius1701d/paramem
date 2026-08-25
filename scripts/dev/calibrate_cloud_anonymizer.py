"""Empirical calibration tool for the local cloud-egress anonymizer
(``paramem.graph.flows.anonymize_turn``, the span-tagger SCAN + local
ANCHOR chain — placeholder substitution is deterministic string
replacement, not a local model call).

Owns its own fixture and threshold (2026-08-24): the prior CI contract
test this script shared them with, ``tests/test_cloud_anonymizer_contract_gpu.py``,
was retired by the anonymizer-split design — the fixed-threshold CI gate
it enforced is superseded by a post-implementation GPU validation ladder
run separately against real traffic. This script keeps its calibration
role standalone: run it against a real GPU + model to measure the
current baseline on the shipped fixture and eyeball the failure-mode
distribution.

Mirrors the calibration pattern of
``tests/test_plausibility_contract_gpu.py`` (75% measured baseline).
The script itself is GPU-free in code; the run needs GPU.

Usage::

    set -a && source .env && set +a && \\
      $HOME/miniforge3/envs/paramem/bin/python \\
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
  leak_blocked       — anonymizer returned ('', {}): the local model's
                       mapping call came back empty (parse failure, or
                       nothing in the configured scope), so the caller
                       never sends anything to the cloud.  Privacy-safe
                       (the cloud call doesn't happen), but counts as
                       "anonymizer failed to deliver".
  privacy_leak       — mapping non-empty but at least one expected
                       name still appears in anon_text.  This is a
                       hard failure: extraction missed the name
                       entirely, or the local model classified it as
                       an out-of-scope entity type, and the cloud
                       would receive it verbatim.
  round_trip_failed  — mapping non-empty, no leak, but the response
                       fails to whitespace-equal the original after
                       deanon.  Indicates lossy whitespace handling
                       in the anonymizer prompt.

The script does NOT update ``_MATCH_THRESHOLD`` automatically.
Eyeball the report — there is no longer a CI contract test to update in
lockstep; a re-measured baseline is applied by hand once the live
calibration gate (against ``tests/fixtures/anonymizer_gate.json``, the
labelled fixture this script and the span-tagger SCAN step's live gate
share) reports a new one.
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

from paramem.config.taxonomy import ScrubCategory, resolve_scrub_categories

# Ensure repo-relative paths resolve the same way regardless of the
# caller's cwd — mirrors the ``_REPO_ROOT`` pattern used by every other
# ``scripts/dev/*.py`` calibration tool (e.g. ``calibrate_prompts.py``).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Reference floor only (2026-08-24) — threshold re-derived from the live
# labelled calibration gate against the fixture below once the
# span-tagger SCAN step is running; kept at the prior conservative value
# until that gate measures a new one.
_MATCH_THRESHOLD = 0.80

# Tracked, fictional calibration corpus shared with the span-tagger's own
# live threshold-measurement gate — single-turn and multi-turn
# transcripts, the production input shape ``answer_via_cloud``
# (``paramem.server.inference``) passes to ``anonymize_turn`` (the
# cloud-egress entry point anonymizes only the current-turn text;
# conversation history flows separately through ``_sanitize_history``).
_FIXTURE_PATH = _REPO_ROOT / "tests" / "fixtures" / "anonymizer_gate.json"


def _load_fixture(path: Path = _FIXTURE_PATH) -> list[dict]:
    """Load the shared calibration corpus and project it to the shape
    ``_run_one``/``main`` consume.

    Reads ``tests/fixtures/anonymizer_gate.json`` — a tracked, fictional
    payload set (single- and multi-turn transcripts, dense contact lists,
    a long planted-value document, a case-variant and a German-inflection
    probe) that also backs the span-tagger live calibration gate. Only
    payloads carrying a ``transcript`` field apply here — this script's
    ``anonymize_turn`` round-trip only takes bare transcript text, so the
    JSON's one ``facts``-kind payload (which exercises the separate
    ``/calibrate/anonymize_facts`` door) is skipped. Each returned entry
    carries exactly the fields ``_run_one`` reads: ``id``, ``speaker_id``,
    ``speaker_name``, ``transcript``, and ``expected_names`` — the
    payload's ``expected.person`` list, the one category this script's
    privacy-contract check scores (the query text's own scrub scope may
    include phone/email/address/profile values too, but the round-trip
    check here has always been name-scoped).
    """
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    entries = []
    for entry in doc["payloads"]:
        if "transcript" not in entry:
            continue
        entries.append(
            {
                "id": entry["id"],
                "speaker_id": entry["speaker_id"],
                "speaker_name": entry["speaker_name"],
                "transcript": entry["transcript"],
                "expected_names": list(entry["expected"].get("person", [])),
            }
        )
    return entries


# ``token_envelope`` — the total (prompt + output) token budget one
# ``anonymize_turn`` call may occupy; matches the shipped operator
# default (``consolidation.extraction_anonymize_token_envelope``).
_TOKEN_ENVELOPE = 8192


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
    speaker_id: str,
    speaker_name: str,
    scrub_categories: tuple[ScrubCategory, ...],
    token_envelope: int = _TOKEN_ENVELOPE,
) -> tuple[str, dict, str]:
    from paramem.cloud.deanonymize import CloudScope, deanonymize_text
    from paramem.cloud.placeholders import _substitute_whole_words
    from paramem.graph.flows import anonymize_turn

    payload = anonymize_turn(
        transcript,
        model,
        tokenizer,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        categories=scrub_categories,
        token_envelope=token_envelope,
    )
    if payload.status != "ok":
        # Covers both "failed" (fail-closed) and "opted_out"
        # (categories=()) — neither has anything to round-trip.  Note this
        # also fixes a latent truthiness bug the old
        # ``if not mapping or not anon_text`` check had: a legitimate "ran,
        # found nothing in scope" verdict (status == "ok", forward == {})
        # is no longer misclassified as a failure — CLAUDE.md forbids
        # truthiness checks on registries.
        return "", dict(payload.forward), ""
    # Derived the same way production's chat-egress path derives it
    # (``paramem.server.inference.answer_via_cloud``): whole-word
    # substitution of the bare turn text against ``payload.forward``, not
    # ``payload.anon_transcript`` — that field stays marker-bearing on
    # this call, since the marker strip lives only in the session-tier
    # transcript path.
    anon_text = _substitute_whole_words(transcript, payload.forward)
    scope = CloudScope.response(payload, cloud_bindings=None, sent=(anon_text,))
    round_trip = deanonymize_text(scope, anon_text)
    return anon_text, dict(payload.forward), round_trip or ""


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
            "PII-vocabulary hints to scrub (e.g. --scope 'person name' "
            "'phone number'); the model is the sole scope authority "
            "against this list, so any free-form category is valid. "
            "Defaults to the production default in "
            "tests/fixtures/server.yaml's sanitization.scrub (read from "
            "disk to stay in sync with the shipped config). Pass --scope "
            "with no values to disable anonymization (operator opt-out)."
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
    # same model (Mistral 7B), same default scrub.  Loading
    # ``configs/server.yaml.example`` instead would re-anchor the
    # calibration whenever the shipped template's ship-default drifts
    # (e.g. cloud_mode block ↔ anonymize) without changing what the
    # contract test actually exercises.  Per CLAUDE.md, calibration
    # and tests share the fixture as the single calibration anchor.
    server_cfg = load_server_config("tests/fixtures/server.yaml")
    model_cfg = server_cfg.model_config
    print(f"  model: {model_cfg.model_id}")
    model, tokenizer = load_base_model(model_cfg, server_cfg.tier_config_map())
    print("  ready")

    # Resolve the scrub scope.  CLI override wins; otherwise inherit from
    # the fixture so the calibration result reflects the contract test's
    # default scope unless explicitly varied.  Hints resolve to the
    # ``ScrubCategory`` tuple ``anonymize_turn`` actually takes
    # (``categories=``) via the same
    # ``paramem.config.taxonomy.resolve_scrub_categories`` production
    # reads off ``SanitizationConfig``.
    if args.scope is None:
        # Already resolved by SanitizationConfig.__post_init__ at
        # ``server_cfg`` load time — reuse it rather than re-running
        # resolve_scrub_categories over a re-sorted set, which would
        # alphabetise the operator's configured order.
        scrub_categories = server_cfg.sanitization.scrub_categories
        print(f"  scope: {[c.name for c in scrub_categories] or '[]  (anonymization disabled)'}")
    else:
        scrub_categories = resolve_scrub_categories(args.scope)
        print(f"  scope: {args.scope or '[]  (anonymization disabled)'}")

    if args.query is not None:
        # Ad-hoc input: wrap as a single-turn transcript so the helper
        # gets the production input shape.  Bare text, same as every
        # fixture entry — ``anonymize_turn`` is the one marker
        # producer (it renders the turn through
        # ``turn_markers.format_turn`` itself); a hand-built ``[user]``
        # prefix here would double-mark it.  Synthesize a speaker_name so
        # the extraction prompt's {SPEAKER_NAME} slot resolves cleanly --
        # production always has a real speaker by the time cloud
        # egress runs (greeting flow).
        entries = [
            {
                "id": "ad-hoc",
                "speaker_id": "speaker0",
                "speaker_name": "Anna",
                "transcript": args.query,
                "expected_names": [],
            }
        ]
    else:
        entries = _load_fixture()

    results: list[QueryResult] = []
    for entry in entries:
        for iteration in range(args.repeat):
            print(f"\n[{entry['id']} iter {iteration}]")
            anon_text, mapping, round_trip = _run_one(
                entry["transcript"],
                entry["expected_names"],
                model,
                tokenizer,
                speaker_id=entry["speaker_id"],
                speaker_name=entry["speaker_name"],
                scrub_categories=scrub_categories,
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
