"""The anonymizer gate's CLI arms (``scripts/dev/anonymizer_gate.py``): the
``--accept``/``--limit`` precondition, the ``--cooldown-every`` parser
validation, and the cooldown chunking's resume exemption — no model, no
GPU.

``test_anonymizer_gate_scorer.py`` covers the scorer itself
(``score_entry``/``score_corpus``/``Result``); this module covers the
argument parser and the corpus-run loop's own control flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the tool importable without installing it as a package — the same
# shim tests/test_anonymizer_gate_scorer.py and
# tests/test_calibrate_prompts_harness.py use for scripts/dev.
_SCRIPTS_DEV = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(_SCRIPTS_DEV) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DEV))

import anonymizer_gate  # noqa: E402 (scripts/dev is not a package)

from paramem.cloud.anonymize import AnonymizedContract  # noqa: E402


def _fake_contract(*, raw: str = "{}") -> AnonymizedContract:
    """A minimal ``"ok"`` contract — its content is irrelevant to the CLI
    tests here, only that ``_write_entry_artifact``/``_load_entry_artifact``
    round-trip it.
    """
    return AnonymizedContract(
        status="ok",
        forward={},
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        rekey_dropped=0,
        raw=raw,
        failure=None,
        facts=[],
        model_calls=1,
        call_tokens=(),
        scan_dropped=0,
        scan_dropped_entries=[],
        inert_dropped=0,
    )


class TestAcceptRefusedUnderLimit:
    """``--accept`` with ``--limit`` is refused before the corpus is even
    loaded — a pilot's scorecard is not a corpus-wide baseline.
    """

    def test_accept_with_limit_is_refused_before_any_corpus_or_model_work(
        self, monkeypatch, capsys
    ) -> None:
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("load_corpus must not run when --accept + --limit is refused")

        monkeypatch.setattr(anonymizer_gate, "load_corpus", _must_not_be_called)

        code = anonymizer_gate.main(["--accept", "--limit", "5"])

        assert code == 1
        assert "refused" in capsys.readouterr().err

    def test_accept_alone_does_not_trip_the_refusal(self, monkeypatch) -> None:
        # Only the accept+limit combination is refused at this gate — a
        # bare --accept must still reach load_corpus.
        called = []
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: called.append(True) or [])
        monkeypatch.setattr(
            anonymizer_gate,
            "validate_corpus",
            lambda entries: (_ for _ in ()).throw(ValueError("stop here, past the refusal")),
        )

        anonymizer_gate.main(["--accept"])

        assert called == [True]


class TestCooldownEveryParserValidation:
    """``--cooldown-every`` accepts only a positive integer — the chunk
    size for the GPU-cooldown pauses.
    """

    def test_cooldown_every_zero_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            anonymizer_gate.build_arg_parser().parse_args(["--cooldown-every", "0"])

    def test_cooldown_every_negative_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            anonymizer_gate.build_arg_parser().parse_args(["--cooldown-every", "-1"])

    def test_cooldown_every_positive_is_accepted(self) -> None:
        args = anonymizer_gate.build_arg_parser().parse_args(["--cooldown-every", "3"])
        assert args.cooldown_every == 3

    def test_cooldown_every_defaults_to_twenty(self) -> None:
        args = anonymizer_gate.build_arg_parser().parse_args([])
        assert args.cooldown_every == 20


class TestResumedEntryDoesNotAdvanceCooldownCount:
    """A resumed entry (already on disk under ``--resume``) never issues a
    model call, so it must never count toward ``--cooldown-every``'s chunk
    — only entries that actually run through :func:`anonymizer_gate._run_entry`
    advance the count :func:`anonymizer_gate._cooldown_if_due` reads.
    """

    def test_resumed_entry_is_skipped_and_only_the_fresh_entry_pauses(
        self, monkeypatch, tmp_path
    ) -> None:
        run_dir = tmp_path / "run"
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)

        fresh_contract = _fake_contract(raw="fresh")
        monkeypatch.setattr(
            anonymizer_gate,
            "_run_entry",
            lambda entry, model, tokenizer, *, categories, token_envelope: fresh_contract,
        )
        pauses = []
        monkeypatch.setattr(anonymizer_gate, "_wait_for_cooldown", lambda: pauses.append(True))

        contracts = anonymizer_gate._run_corpus(
            [{"id": "e1"}, {"id": "e2"}],
            model=None,
            tokenizer=None,
            categories=(),
            token_envelope=100,
            run_dir=run_dir,
            resume=True,
            cooldown_every=1,
        )

        # e1 came from disk (resumed), e2 actually ran — the pause fires
        # exactly once, for e2 alone.
        assert contracts["e2"] is fresh_contract
        assert len(pauses) == 1

    def test_two_resumed_entries_never_pause_at_cooldown_every_one(
        self, monkeypatch, tmp_path
    ) -> None:
        run_dir = tmp_path / "run"
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e2", _fake_contract(), 0.1)

        def _must_not_run(*args, **kwargs):
            raise AssertionError("a resumed entry must never reach _run_entry")

        monkeypatch.setattr(anonymizer_gate, "_run_entry", _must_not_run)
        pauses = []
        monkeypatch.setattr(anonymizer_gate, "_wait_for_cooldown", lambda: pauses.append(True))

        anonymizer_gate._run_corpus(
            [{"id": "e1"}, {"id": "e2"}],
            model=None,
            tokenizer=None,
            categories=(),
            token_envelope=100,
            run_dir=run_dir,
            resume=True,
            cooldown_every=1,
        )

        assert pauses == []
