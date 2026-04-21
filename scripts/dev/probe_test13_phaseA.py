"""Probe Phase A adapter of test-13 to identify the 199/200 outlier.

One-shot diagnostic. Writes per-key results to outlier_report.json next to the adapter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.utils.test_harness import smoke_test_adapter

CYCLE_DIR = Path(
    "outputs/test13_journal_scaffold/mistral/20260420_231031/A"
)


def main() -> int:
    result = smoke_test_adapter(CYCLE_DIR, "mistral")

    failures = [
        {
            "key": r["key"],
            "expected": r.get("expected_answer") or r.get("expected"),
            "recalled": r.get("recalled_answer") or r.get("recalled") or r.get("raw_output"),
            "confidence": r.get("confidence"),
            "expected_word_count": r.get("expected_word_count"),
            "recalled_word_count": r.get("recalled_word_count"),
            "exact_match": r.get("exact_match"),
        }
        for r in result["per_key"]
        if not r.get("exact_match")
    ]

    report = {
        "exact_count": result["exact_count"],
        "total": result["total"],
        "rate": result["rate"],
        "mean_confidence": result["mean_confidence"],
        "failure_count": len(failures),
        "failures": failures,
    }

    out_path = CYCLE_DIR / "outlier_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    print(f"Failures: {len(failures)}")
    for f in failures:
        print(f"\n  {f['key']} (conf={f['confidence']:.3f}, "
              f"expected={f['expected_word_count']}w, got={f['recalled_word_count']}w)")
        print(f"    expected: {f['expected']!r}")
        print(f"    recalled: {f['recalled']!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
