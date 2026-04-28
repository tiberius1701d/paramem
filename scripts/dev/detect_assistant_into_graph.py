"""Detect assistant-into-graph hallucinations in extraction probe runs.

Given a `dataset_probe` run that wrote `raw_qa.json` diagnostics and the
matching source dataset (currently PerLTQA), cross-reference each
extracted triple's `(subject, predicate, object)` against the original
dialogue.  Flag triples where:

* the subject is the canonical speaker, AND
* every significant token in the object appears ONLY in non-speaker
  turns (i.e. the substantive content was authored by an assistant or
  a third-party participant, not by the speaker themselves).

This is the structural failure mode that the planned role-aware
provenance gate (extension to ``_drop_ungrounded_facts``) is designed
to catch.  The detector serves three purposes:

1. **Diagnostic** — surface real hallucinations from a probe run for
   eyeballing.
2. **Fixture seed** — emit a JSON file shaped like a fixture so the
   gate's contract test can assert these triples get dropped.
3. **Calibration** — measure the failure rate on a real corpus before
   and after the gate ships, to confirm the structural defense is
   working as intended.

Runs without the GPU.  Pure file IO + string/regex matching.

Usage::

    python scripts/dev/detect_assistant_into_graph.py \\
        --run outputs/dataset_probe/perltqa/mistral/20260417_165915 \\
        --dataset data/external/PerLTQA/Dataset/en_v2/perltmem_en_v2.json \\
        --out tests/fixtures/provenance_gate_failures.json

Without ``--out`` the report is printed to stdout only.

Filename convention assumed for the run's diagnostics directory:
``perltqa_<Speaker Name>_<dialogue_key>.raw_qa.json``.  The detector
auto-discovers all such files under the run dir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Tokens commonly used as direct address (nicknames / role labels) that
# appear in non-speaker turns but are not assistant-into-graph evidence
# when matched.  Filter to reduce false positives.
_NICKNAME_TOKENS = {
    "xiaoyu",
    "xia",
    "yu",
    "you",
    "your",
    "user",
    "speaker",
    "assistant",
}

_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "and",
    "or",
    "with",
    "is",
    "are",
    "for",
    "my",
    "our",
    "this",
    "that",
}

_FILENAME_PATTERN = re.compile(r"^perltqa_(?P<speaker>.+?)_(?P<dialogue>[\w#]+)\.raw_qa\.json$")


def _split_turns(speaker: str, dialogue: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (user_turns, [(other_speaker, text), ...]) from a PerLTQA dialogue.

    PerLTQA dialogues are timestamp-keyed dicts whose values are lists of
    ``"<Speaker>: <text>"`` strings.  The speaker named in the filename is
    the canonical "user"; everyone else (named third parties, the chatbot
    persona) goes into ``others``.
    """
    user: list[str] = []
    others: list[tuple[str, str]] = []
    for _ts, turns in dialogue.get("contents", {}).items():
        for raw in turns:
            if ":" not in raw:
                continue
            name, _, content = raw.partition(":")
            name = name.strip()
            content = content.strip()
            if name == speaker:
                user.append(content)
            else:
                others.append((name, content))
    return user, others


def _appears(token: str, text: str) -> bool:
    """Whole-word, case-insensitive substring check."""
    return bool(re.search(r"\b" + re.escape(token.lower()) + r"\b", text.lower()))


def _significant_tokens(obj: str) -> list[str]:
    """Tokenize an object field into significant content tokens.

    Drops stopwords, very short tokens, and the speaker-name nicknames
    that are commonly used as direct address ("Xiaoyu", "you").
    """
    return [
        t.lower()
        for t in re.split(r"\W+", obj)
        if t and len(t) > 2 and t.lower() not in _STOPWORDS and t.lower() not in _NICKNAME_TOKENS
    ]


def detect_run(
    run_dir: Path, dataset: dict, *, max_per_speaker: int | None = None
) -> tuple[list[dict], int]:
    """Walk a probe run directory and return all assistant-into-graph candidates.

    Returns ``(candidates, total_qa_scanned)``.  Each candidate is a dict
    with ``speaker``, ``dialogue_key``, ``triple`` (subject/predicate/object),
    ``said_by`` (the non-speaker name from the source turn), and ``turn``
    (the full source utterance, for human review).

    ``max_per_speaker`` truncates per-speaker candidate lists for terser
    reports; ``None`` (default) means no truncation.
    """
    diag_dir = run_dir / "diagnostics"
    if not diag_dir.exists():
        raise FileNotFoundError(f"No diagnostics directory at {diag_dir}")

    candidates: list[dict] = []
    per_speaker_seen: dict[str, int] = defaultdict(int)
    total = 0

    for fp in sorted(diag_dir.glob("perltqa_*.raw_qa.json")):
        m = _FILENAME_PATTERN.match(fp.name)
        if not m:
            continue
        speaker = m.group("speaker")
        dialogue_key = m.group("dialogue")
        dialogue = dataset.get(speaker, {}).get("dialogues", {}).get(dialogue_key)
        if not dialogue:
            continue

        user_turns, other_turns = _split_turns(speaker, dialogue)
        user_text = " ".join(user_turns)
        others_text = " ".join(c for _, c in other_turns)

        with fp.open() as f:
            rawqa = json.load(f)

        for q in rawqa.get("episodic_qa", []):
            total += 1
            if q.get("source_subject") != speaker:
                continue  # third-party triples are not assistant-into-graph
            obj = q.get("source_object", "")
            if not obj or len(obj) < 3:
                continue

            tokens = _significant_tokens(obj)
            if not tokens:
                continue

            in_user = any(_appears(t, user_text) for t in tokens)
            in_others = any(_appears(t, others_text) for t in tokens)
            if in_user or not in_others:
                continue

            if max_per_speaker is not None and per_speaker_seen[speaker] >= max_per_speaker:
                continue

            matching = next(
                (
                    (name, content)
                    for name, content in other_turns
                    if any(_appears(t, content) for t in tokens)
                ),
                None,
            )
            candidates.append(
                {
                    "speaker": speaker,
                    "dialogue_key": dialogue_key,
                    "subject": q.get("source_subject"),
                    "predicate": q.get("source_predicate"),
                    "object": obj,
                    "tokens_checked": tokens,
                    "said_by": matching[0] if matching else "?",
                    "source_turn": matching[1] if matching else "",
                    "source_file": fp.name,
                }
            )
            per_speaker_seen[speaker] += 1

    return candidates, total


def _print_report(candidates: list[dict], total: int) -> None:
    rate = (len(candidates) / total) if total else 0.0
    print("=" * 78)
    print(f"Assistant-into-graph candidates: {len(candidates)} / {total} QA scanned ({rate:.1%})")
    print("=" * 78)
    for c in candidates:
        print(
            f"\n[{c['speaker']:<12} :: {c['dialogue_key']}]"
            f"  {c['subject']} | {c['predicate']} | {c['object']!r}"
        )
        print(f"   said_by  : {c['said_by']}")
        print(f"   utterance: {c['source_turn']!r}")


def _write_fixture(candidates: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "schema": "provenance_gate_failures.v1",
        "description": (
            "Hallucinated triples surfaced by detect_assistant_into_graph.py. "
            "The role-aware extension of _drop_ungrounded_facts must drop "
            "every entry below."
        ),
        "candidates": candidates,
    }
    with out_path.open("w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    print(f"\nWrote fixture: {out_path} ({len(candidates)} candidates)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help=(
            "Path to a dataset_probe run directory "
            "(e.g. outputs/dataset_probe/perltqa/mistral/<timestamp>)"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/external/PerLTQA/Dataset/en_v2/perltmem_en_v2.json"),
        help="Path to the PerLTQA source dataset JSON (default: en_v2)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: write candidates as a fixture-shaped JSON to this path",
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=None,
        help="Optional: cap candidates per speaker for terser reports",
    )
    args = parser.parse_args(argv)

    if not args.run.exists():
        print(f"ERROR: run directory not found: {args.run}", file=sys.stderr)
        return 2
    if not args.dataset.exists():
        print(f"ERROR: dataset file not found: {args.dataset}", file=sys.stderr)
        return 2

    with args.dataset.open() as f:
        dataset = json.load(f)

    candidates, total = detect_run(args.run, dataset, max_per_speaker=args.max_per_speaker)
    _print_report(candidates, total)

    if args.out is not None:
        _write_fixture(candidates, args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
