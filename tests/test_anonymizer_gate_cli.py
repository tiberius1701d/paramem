"""The anonymizer gate's CLI arms (``scripts/dev/anonymizer_gate.py``): the
``--accept``/``--limit`` precondition, the ``--cooldown-every`` parser
validation, the cooldown chunking's resume exemption, the skeleton
re-measurement print, ``main``'s ``--dry-run``/``--resume`` dispatch, and
the gate's own input identity — ``RunRecord``, its prompt/table/test-set
digests, the baseline copy, the plain-language difference phrases, an
unusable ``--prompt-file``'s six refusal cases, the resume identity
check's refuse/proceed/notice split, and the ``--accept`` refusal on a
differing or unrecorded input — no model, no GPU.

``test_anonymizer_gate_scorer.py`` covers the scorer itself
(``score_entry``/``score_corpus``/``Result``); this module covers the
argument parser, the corpus-run loop's own control flow, and the gate's
input-identity machinery.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
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
from tests.anonymizer_doubles import (  # noqa: E402
    INVALID_ANONYMIZATION_SECTIONS_MALFORMED_PLACEHOLDER,
    INVALID_ANONYMIZATION_SECTIONS_MISSING_SECTION,
    INVALID_ANONYMIZATION_SECTIONS_MISSING_SLOT,
    INVALID_ANONYMIZATION_SECTIONS_UNKNOWN_SLOT,
    VALID_ANONYMIZATION_SECTIONS,
)


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


def _stub_gpu_guard(monkeypatch, acquire_gpu) -> None:
    """Put *acquire_gpu* behind the tool's own GPU-guard imports.

    ``main`` reaches the guard through ``gpu_guard`` and
    ``experiments.utils.gpu_guard``; the package behind both is a separate
    lab-tools repo that this one does not depend on, so patching either by
    name would import it and fail wherever it is not installed. Both are
    stubbed in ``sys.modules`` instead, which leaves every assertion these
    tests make intact and never imports the real guard.
    """
    guard = types.ModuleType("gpu_guard")
    guard.GPUConfigMissing = type("GPUConfigMissing", (Exception,), {})
    wrapper = types.ModuleType("experiments.utils.gpu_guard")
    wrapper.acquire_gpu = acquire_gpu
    monkeypatch.setitem(sys.modules, "gpu_guard", guard)
    monkeypatch.setitem(sys.modules, "experiments.utils.gpu_guard", wrapper)


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


class TestLimitParserValidation:
    """``--limit`` accepts only a positive integer, via the same
    ``_positive_int`` validator ``--cooldown-every`` uses — zero is
    refused at the parser rather than reaching the corpus slice.
    """

    def test_limit_zero_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            anonymizer_gate.build_arg_parser().parse_args(["--limit", "0"])

    def test_limit_negative_is_rejected_by_the_parser(self) -> None:
        with pytest.raises(SystemExit):
            anonymizer_gate.build_arg_parser().parse_args(["--limit", "-1"])

    def test_limit_positive_is_accepted(self) -> None:
        args = anonymizer_gate.build_arg_parser().parse_args(["--limit", "5"])
        assert args.limit == 5


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


# ---------------------------------------------------------------------------
# _run_dir_complete — the score-only path's completeness test.
# ---------------------------------------------------------------------------


class TestRunDirComplete:
    def test_true_only_when_every_entry_has_an_artifact(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)

        assert anonymizer_gate._run_dir_complete([{"id": "e1"}], run_dir) is True
        assert anonymizer_gate._run_dir_complete([{"id": "e1"}, {"id": "e2"}], run_dir) is False

    def test_a_freshly_created_run_directory_is_never_complete(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        assert anonymizer_gate._run_dir_complete([{"id": "e1"}], run_dir) is False

    def test_an_empty_entry_list_is_never_complete(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        assert anonymizer_gate._run_dir_complete([], run_dir) is False


# ---------------------------------------------------------------------------
# _print_skeleton_measurements — the SCAN/ANCHOR skeleton re-measurement
# print every run makes, ``--dry-run`` included.
# ---------------------------------------------------------------------------


class _CharCountTokenizer:
    """A deterministic tokenizer stub: one token per character. Exposes
    ``apply_chat_template`` (real enough that
    ``paramem.models.loader.supports_system_role``'s own marker-survival
    probe reports the system role as supported) and ``__call__`` returning
    ``{"input_ids": [...]}`` — the shape
    ``paramem.utils.tokens.encode_rendered`` requires.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": list(text)}


class TestPrintSkeletonMeasurements:
    """A print only — no threshold, no refusal, no exit code change on
    drift; a prompt edit that lengthens one section moves only that
    section's measurement.
    """

    def test_prints_one_line_per_skeleton_with_measured_reference_and_diff(self, capsys) -> None:
        tokenizer = _CharCountTokenizer()

        anonymizer_gate._print_skeleton_measurements(tokenizer, prompt_override_text=None)

        out = capsys.readouterr().out
        assert "SCAN skeleton: measured=" in out
        assert "reference=" in out
        assert "diff=" in out
        assert "ANCHOR skeleton: measured=" in out

    def test_never_raises_or_exits_on_a_drifted_measurement(self, capsys) -> None:
        # The shipped prompt home is real production text, guaranteed to
        # differ from a one-token-per-character count — reaching this
        # point without an exception or SystemExit IS the assertion.
        anonymizer_gate._print_skeleton_measurements(
            _CharCountTokenizer(), prompt_override_text=None
        )

    def test_a_prompt_override_that_lengthens_scan_grows_only_the_scan_measurement(
        self, capsys
    ) -> None:
        prompts_path = (
            Path(__file__).resolve().parents[1] / "configs" / "prompts" / "anonymization.txt"
        )
        shipped_text = prompts_path.read_text(encoding="utf-8")
        filler = " filler word" * 200
        lengthened_scan_text = shipped_text.replace(
            "=== SCAN ===\n", "=== SCAN ===\n" + filler + "\n", 1
        )
        assert lengthened_scan_text != shipped_text  # guard the fixture edit itself

        tokenizer = _CharCountTokenizer()

        anonymizer_gate._print_skeleton_measurements(tokenizer, prompt_override_text=None)
        baseline_out = capsys.readouterr().out
        anonymizer_gate._print_skeleton_measurements(
            tokenizer, prompt_override_text=lengthened_scan_text
        )
        lengthened_out = capsys.readouterr().out

        def _measured(out: str, label: str) -> int:
            line = next(line for line in out.splitlines() if line.startswith(f"{label} skeleton"))
            return int(line.split("measured=")[1].split(" ")[0])

        assert _measured(lengthened_out, "SCAN") > _measured(baseline_out, "SCAN")
        assert _measured(lengthened_out, "ANCHOR") == _measured(baseline_out, "ANCHOR")


# ---------------------------------------------------------------------------
# main() — ``--dry-run`` and ``--resume`` dispatch.
# ---------------------------------------------------------------------------


def _valid_entry(entry_id: str, text: str = "hello world") -> dict:
    """A minimal corpus entry that ``validate_corpus`` accepts unmodified —
    no gold, so no offset to keep in sync."""
    return {
        "id": entry_id,
        "lang": "en",
        "casing": "cased",
        "kind": "name_mention",
        "description": "a hand-built entry for the CLI's own tests",
        "speaker_id": None,
        "speaker_name": None,
        "history": [],
        "text": text,
        "gold": [],
        "decoys": [],
    }


class TestMainDryRun:
    """``--dry-run`` assembles and validates the corpus, prints both
    skeleton measurements and the closing line, and never reaches the GPU
    guard or a model load."""

    def test_dry_run_prints_the_closing_line_and_never_touches_gpu_or_model(
        self, monkeypatch, capsys
    ) -> None:
        load_tokenizer_calls: list[object] = []
        monkeypatch.setattr(
            "paramem.models.loader.load_tokenizer",
            lambda cfg: load_tokenizer_calls.append(cfg) or _CharCountTokenizer(),
        )

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("--dry-run must never reach the GPU guard or a model load")

        _stub_gpu_guard(monkeypatch, _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_base_model", _must_not_be_called)

        code = anonymizer_gate.main(["--dry-run"])

        out = capsys.readouterr().out
        assert code == 0
        assert "SCAN skeleton: measured=" in out
        assert "ANCHOR skeleton: measured=" in out
        assert "dry run complete; tokenizer loaded, no model, no GPU touched" in out
        assert len(load_tokenizer_calls) == 1


class TestMainResume:
    """``--resume`` onto a run directory that is already complete (every
    corpus entry has a written artifact) scores straight from disk — no
    GPU, no model. A run directory still missing an entry proceeds under
    the GPU guard only when its own recorded inputs match this
    invocation's; see ``TestFreshRunRecordAndResumeInputIdentity`` and
    ``TestResumeRefusesWhenAnInputIsNotRecorded`` for the refusal that
    inputs mismatch triggers instead."""

    def _stub_load_tokenizer(self, monkeypatch) -> list[object]:
        calls: list[object] = []
        monkeypatch.setattr(
            "paramem.models.loader.load_tokenizer",
            lambda cfg: calls.append(cfg) or _CharCountTokenizer(),
        )
        return calls

    def _forbid_gpu_and_model(self, monkeypatch) -> None:
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("the score-only path must never reach the GPU guard or a model")

        _stub_gpu_guard(monkeypatch, _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_base_model", _must_not_be_called)

    def test_a_complete_run_directory_is_scored_from_disk_without_gpu_or_model(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        corpus = [_valid_entry("e1"), _valid_entry("e2"), _valid_entry("e3")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e2", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e3", _fake_contract(), 0.1)

        # A run directory's own recorded provenance, built through the
        # module's own record builder/writer with exactly the inputs this
        # invocation resolves below (no --scrub, no --model, no
        # --prompt-file) — so it matches this run and `scorecard.json` is
        # written (a run directory's own `scorecard.json` is written only
        # when the inputs it was scored under match the run's recorded
        # ones).
        from paramem.server.config import load_server_config

        server_cfg = load_server_config("tests/fixtures/server.yaml")
        configured = {c.prefix for c in anonymizer_gate.resolve_categories(None)}
        record = anonymizer_gate._current_run_record(
            configured=configured,
            model_id=server_cfg.model_config.model_id,
            prompt_file=None,
            token_envelope=server_cfg.consolidation.extraction_anonymize_token_envelope,
        )
        anonymizer_gate._write_run_record(run_dir, record)

        load_tokenizer_calls = self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 0
        assert "scored from disk" in out
        # The run directory's own recorded provenance matches this
        # invocation's own, so no mismatch is reported either way.
        assert "has no run.json" not in out
        assert "differs from this invocation" not in out
        assert len(load_tokenizer_calls) == 1
        scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
        assert "failed" in scorecard

    def test_a_complete_run_directory_with_no_run_json_scores_anyway_and_never_writes_the_scorecard(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """A run directory carrying no ``run.json`` at all records nothing
        to match this scoring's own inputs against. It is still scored
        (scoring from disk issues no model call, so nothing could blend),
        but ``run_dir/scorecard.json`` is never written — a run directory
        with no recorded provenance cannot match, the same rule a
        differing recorded input triggers, with no special case for the
        unrecorded run."""
        corpus = [_valid_entry("e1")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")
        anonymizer_gate._write_entry_artifact(run_dir / "entries", "e1", _fake_contract(), 0.1)
        # No run.json written at all.

        self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 0
        assert "scored from disk" in out
        assert "has no run.json; none of" in out
        assert "SCORECARD" in out
        assert "scorecard.json not written" in out
        assert not (run_dir / "scorecard.json").exists()

    def test_limit_on_a_complete_run_directory_never_writes_the_scorecard(
        self, monkeypatch, tmp_path
    ) -> None:
        """``--limit`` never overwrites a complete run's own full-corpus
        ``scorecard.json`` from a slice — even when this scoring's own run
        identity otherwise matches the run's recorded one (a matching
        ``run.json`` is written below, over the same whole corpus this
        ``--limit 2`` invocation's own run identity records; removing the
        ``--limit`` condition from the write guard would make this test
        fail)."""
        corpus = [_valid_entry("e1"), _valid_entry("e2"), _valid_entry("e3")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e2", _fake_contract(), 0.1)

        from paramem.server.config import load_server_config

        server_cfg = load_server_config("tests/fixtures/server.yaml")
        configured = {c.prefix for c in anonymizer_gate.resolve_categories(None)}
        record = anonymizer_gate._current_run_record(
            configured=configured,
            model_id=server_cfg.model_config.model_id,
            prompt_file=None,
            token_envelope=server_cfg.consolidation.extraction_anonymize_token_envelope,
        )
        anonymizer_gate._write_run_record(run_dir, record)

        self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        code = anonymizer_gate.main(["--resume", "--limit", "2"])

        assert code == 0
        assert not (run_dir / "scorecard.json").exists()

    def test_a_complete_run_directory_with_differing_recorded_inputs_scores_anyway_and_notes_it(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """A complete run directory issues no model call, so a differing
        prompt, table, model, scrubbed-kinds set or test set cannot blend
        into the entries already on disk — it is scored either way, with
        the difference printed as a notice rather than a refusal (without
        ``--accept``; see ``TestMainAcceptRefusesOnDifferingInputs`` for
        the refusal ``--accept`` triggers on the same mismatch). The
        printed scorecard is never written to ``run_dir/scorecard.json``
        in this case — that file always reflects the run's OWN recorded
        inputs, never a re-score under different ones."""
        corpus = [_valid_entry("e1")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")
        anonymizer_gate._write_entry_artifact(run_dir / "entries", "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_run_record(
            run_dir,
            anonymizer_gate.RunRecord(
                configured_prefixes=[],
                model_id="a-different-model-id",
                prompt_file=None,
                prompt_sha256="deadbeef",
                table_sha256="deadbeef",
                corpus_sha256="deadbeef",
                token_envelope=1,
            ),
        )

        self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 0
        assert "scored from disk" in out
        # Every one of the six recorded inputs above differs from this
        # invocation's own (a different model id, "deadbeef" digests, a
        # token budget of 1, and the shipped default scrub set against an
        # empty recorded one) — the exact difference line, not a substring
        # any print could satisfy by accident.
        assert (
            "differs from this invocation: scrubbed kinds, model, prompt, keyword table, "
            "test set, token budget" in out
        )
        assert "scorecard.json not written" in out
        assert not (run_dir / "scorecard.json").exists()


# ---------------------------------------------------------------------------
# The gate's input identity: RunRecord, its hashes, the baseline copy, and
# the verdict's plain-language differences.
# ---------------------------------------------------------------------------


def _shipped_anonymization_prompt_text() -> str:
    path = Path(__file__).resolve().parents[1] / "configs" / "prompts" / "anonymization.txt"
    return path.read_text(encoding="utf-8")


class TestCurrentRunRecord:
    """``_current_run_record`` builds this invocation's own recorded
    inputs: sorted scrubbed kinds, the model id, the prompt/table/
    test-set digests, and the per-call token budget every later comparison
    reads."""

    def test_configured_prefixes_are_recorded_sorted_regardless_of_input_order(self) -> None:
        record = anonymizer_gate._current_run_record(
            configured={"Org", "City", "Address"},
            model_id="m",
            prompt_file=None,
            token_envelope=100,
        )

        assert record.configured_prefixes == ["Address", "City", "Org"]

    def test_prompt_digest_is_identical_for_the_shipped_home_and_a_byte_identical_prompt_file(
        self, tmp_path
    ) -> None:
        shipped_text = _shipped_anonymization_prompt_text()
        shipped_record = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        override_record = anonymizer_gate._current_run_record(
            configured=set(),
            model_id="m",
            prompt_file=tmp_path / "anonymization.txt",
            token_envelope=100,
            prompt_override_text=shipped_text,
        )

        assert override_record.prompt_sha256 == shipped_record.prompt_sha256

    def test_prompt_digest_differs_when_a_composed_section_text_differs(self) -> None:
        shipped_text = _shipped_anonymization_prompt_text()
        filler = " filler word" * 5
        lengthened = shipped_text.replace("=== SCAN ===\n", "=== SCAN ===\n" + filler + "\n", 1)
        assert lengthened != shipped_text  # guard the fixture edit itself

        shipped_record = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )
        changed_record = anonymizer_gate._current_run_record(
            configured=set(),
            model_id="m",
            prompt_file=Path("unused"),
            token_envelope=100,
            prompt_override_text=lengthened,
        )

        assert changed_record.prompt_sha256 != shipped_record.prompt_sha256

    def test_text_above_the_first_section_marker_leaves_the_prompt_digest_unchanged(self) -> None:
        """``_load_prompt_sections`` discards everything above the first
        ``===`` sentinel (a behaviour-level header comment) before any
        section is picked up, so growing that header must not move the
        digest a run pins or compares against.
        """
        shipped_text = _shipped_anonymization_prompt_text()
        with_extra_header = "# an extra header comment, never read as a section\n" + shipped_text

        shipped_record = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )
        header_record = anonymizer_gate._current_run_record(
            configured=set(),
            model_id="m",
            prompt_file=Path("unused"),
            token_envelope=100,
            prompt_override_text=with_extra_header,
        )

        assert header_record.prompt_sha256 == shipped_record.prompt_sha256

    def test_prompt_file_is_recorded_when_given_and_none_for_the_shipped_home(
        self, tmp_path
    ) -> None:
        path = tmp_path / "a-variant.txt"

        overridden = anonymizer_gate._current_run_record(
            configured=set(),
            model_id="m",
            prompt_file=path,
            token_envelope=100,
            prompt_override_text=_shipped_anonymization_prompt_text(),
        )
        shipped = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        assert overridden.prompt_file == str(path)
        assert shipped.prompt_file is None

    def test_table_digest_changes_when_an_allow_row_is_edited(self, monkeypatch) -> None:
        """The table digest reflects the currently loaded anonymizer table
        by behaviour, not by re-deriving the expected hash with the same
        function under test: editing one allow row's description changes
        the digest a fresh record derives.
        """
        schema_before = {
            "anonymizer": {
                "scrub": [{"prefix": "Person", "entity_type": "person"}],
                "allow": [
                    {
                        "prefix": "Org",
                        "entity_type": "organization",
                        "description": "an organization, company or institution name",
                    }
                ],
            }
        }
        schema_after = {
            "anonymizer": {
                "scrub": [{"prefix": "Person", "entity_type": "person"}],
                "allow": [
                    {
                        "prefix": "Org",
                        "entity_type": "organization",
                        "description": "an organization, company, institution or team name",
                    }
                ],
            }
        }

        monkeypatch.setattr(anonymizer_gate, "load_schema_config", lambda: schema_before)
        record_before = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        monkeypatch.setattr(anonymizer_gate, "load_schema_config", lambda: schema_after)
        record_after = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        assert record_before.table_sha256 != record_after.table_sha256

    def test_corpus_digest_changes_on_an_entry_edit_but_not_on_an_about_edit(
        self, tmp_path, monkeypatch
    ) -> None:
        """``load_corpus`` returns only the ``entries`` list
        (:func:`anonymizer_gate.load_corpus`), so the corpus digest reads
        real behaviour: editing the fixture's ``_about`` text must never
        move it, and editing an entry must."""
        real_load_corpus = anonymizer_gate.load_corpus

        def _write_corpus(about: str, entries: list[dict]) -> Path:
            path = tmp_path / "corpus.json"
            path.write_text(json.dumps({"_about": about, "entries": entries}), encoding="utf-8")
            return path

        base_entries = [_valid_entry("e1")]

        monkeypatch.setattr(
            anonymizer_gate,
            "load_corpus",
            lambda: real_load_corpus(_write_corpus("original", base_entries)),
        )
        record_a = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        monkeypatch.setattr(
            anonymizer_gate,
            "load_corpus",
            lambda: real_load_corpus(
                _write_corpus("a different _about text entirely", base_entries)
            ),
        )
        record_b = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        assert record_a.corpus_sha256 == record_b.corpus_sha256

        changed_entries = [_valid_entry("e1", text="a different text entirely")]
        monkeypatch.setattr(
            anonymizer_gate,
            "load_corpus",
            lambda: real_load_corpus(_write_corpus("original", changed_entries)),
        )
        record_c = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        assert record_c.corpus_sha256 != record_a.corpus_sha256

    def test_corpus_digest_is_always_the_whole_corpus_regardless_of_a_limit_slice(
        self, monkeypatch
    ) -> None:
        """The run's own recorded test set (this function) is the WHOLE
        corpus a run belongs to — never a ``--limit`` slice — so a fresh
        pilot and its later full-corpus ``--resume`` continuation share the
        identical run identity and are never refused for a mismatched test
        set. (The OTHER "test set" role — what a scorecard was actually
        computed over — is :func:`_scored_inputs`'s own digest, built from
        the entries handed to scoring, never from this record.)"""
        full_corpus = [_valid_entry("e1"), _valid_entry("e2"), _valid_entry("e3")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: full_corpus)

        record = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=100
        )

        assert record.corpus_sha256 == anonymizer_gate._sha256_json(full_corpus)

    def test_token_envelope_is_recorded_as_given(self) -> None:
        """Unlike ``prompt_sha256``/``table_sha256``/``corpus_sha256``,
        ``token_envelope`` is not derived here — it is a plain integer this
        invocation's own caller (``main``) already read from
        ``server_cfg.consolidation.extraction_anonymize_token_envelope``
        and passes straight through."""
        record = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=4096
        )

        assert record.token_envelope == 4096


class TestSha256JsonRowAndKeyOrder:
    """``_sha256_json`` is what ``table_sha256`` hashes the anonymizer
    table with: a list's row order changes the digest, a dict's key order
    does not."""

    def test_row_order_change_changes_the_digest(self) -> None:
        table_a = {"scrub": [{"prefix": "A"}, {"prefix": "B"}]}
        table_b = {"scrub": [{"prefix": "B"}, {"prefix": "A"}]}

        assert anonymizer_gate._sha256_json(table_a) != anonymizer_gate._sha256_json(table_b)

    def test_dict_key_order_does_not_change_the_digest(self) -> None:
        table_a = {"scrub": [{"prefix": "A", "description": "d"}]}
        table_b = {"scrub": [{"description": "d", "prefix": "A"}]}

        assert anonymizer_gate._sha256_json(table_a) == anonymizer_gate._sha256_json(table_b)


class TestReadRunRecord:
    def test_a_record_missing_the_table_digest_key_reads_as_not_recorded(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "configured_prefixes": ["Person"],
                    "model_id": "m",
                    "prompt_file": None,
                    "prompt_sha256": "abc",
                }
            ),
            encoding="utf-8",
        )

        record = anonymizer_gate._read_run_record(run_dir)

        assert record.table_sha256 is None
        # The same record also carries no token_envelope key — read as
        # not recorded the same way the table digest above is.
        assert record.token_envelope is None

    def test_token_envelope_round_trips_through_write_and_read(self, tmp_path) -> None:
        """The budget a run pins is recorded in ``run.json`` and read back
        unchanged — the same round trip every other recorded input makes."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        record = anonymizer_gate._current_run_record(
            configured=set(), model_id="m", prompt_file=None, token_envelope=4096
        )

        anonymizer_gate._write_run_record(run_dir, record)
        read_back = anonymizer_gate._read_run_record(run_dir)

        assert read_back.token_envelope == 4096

    def test_a_record_missing_configured_prefixes_or_model_id_reads_as_not_recorded(
        self, tmp_path
    ) -> None:
        """``configured_prefixes`` and ``model_id`` are two of the six
        identity inputs (:data:`anonymizer_gate._INPUT_KEYS`) — a record
        on disk lacking either key reads it as ``None`` (not recorded),
        never a ``KeyError``, the same tolerance already given to
        ``prompt_sha256``/``table_sha256``/``corpus_sha256``/
        ``token_envelope`` above."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(
            json.dumps({"prompt_file": None, "prompt_sha256": "abc"}),
            encoding="utf-8",
        )

        record = anonymizer_gate._read_run_record(run_dir)

        assert record.configured_prefixes is None
        assert record.model_id is None


class TestWriteBaseline:
    """``write_baseline``'s scrubbed kinds and test set are the scope
    ``current`` — this accept invocation's own values — never the run
    directory's own recorded ``configured_prefixes``/``corpus_sha256``,
    which can differ (a complete run directory can be re-scored under a
    different ``--scrub``, or against a changed test set). Model,
    prompt, keyword table and token budget come from the accepted run's
    own record instead, since those four cannot vary the same way between
    the run and the scoring invocation.
    """

    def test_writes_source_scorecard_and_the_accepted_runs_own_inputs(self, tmp_path) -> None:
        run_record = anonymizer_gate.RunRecord(
            configured_prefixes=["City", "Person"],  # what the run itself recorded
            model_id="model-x",
            prompt_file=None,
            prompt_sha256="prompt-hash",
            table_sha256="table-hash",
            corpus_sha256="run-corpus-hash",  # the run's own whole-corpus identity; unread here
            token_envelope=4096,
        )
        current = anonymizer_gate.RunRecord(
            configured_prefixes=["Email", "Phone"],  # what THIS invocation scored with
            model_id="model-x",
            prompt_file=None,
            prompt_sha256="prompt-hash",
            table_sha256="table-hash",
            corpus_sha256="ignored-whole-corpus-hash",  # unread here; see entries below
            token_envelope=4096,
        )
        entries = [_valid_entry("e1")]  # the entries actually scored
        path = tmp_path / "baseline.json"

        anonymizer_gate.write_baseline(
            {"junk": 3, "failed": 0},
            source="run X",
            current=current,
            run_record=run_record,
            entries=entries,
            path=path,
        )

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["junk"] == 3
        assert data["failed"] == 0
        assert data["source"] == "run X"
        assert data["model_id"] == "model-x"
        assert data["prompt_sha256"] == "prompt-hash"
        assert data["table_sha256"] == "table-hash"
        assert data["token_envelope"] == 4096
        # The scrubbed kinds are this invocation's own scoring scope, never
        # the run's differing recorded set above. The test set is the
        # entries actually scored, hashed directly — neither RunRecord's
        # own (whole-corpus) digest.
        assert data["configured_prefixes"] == ["Email", "Phone"]
        assert data["corpus_sha256"] == anonymizer_gate._sha256_json(entries)

    def test_a_run_directory_with_no_record_gives_not_recorded_model_prompt_table_and_token_budget(
        self, tmp_path
    ) -> None:
        path = tmp_path / "baseline.json"
        current = anonymizer_gate.RunRecord(
            configured_prefixes=["Person"],
            model_id="ignored",
            prompt_file=None,
            prompt_sha256="ignored",
            table_sha256="ignored",
            corpus_sha256="ignored-whole-corpus-hash",
            token_envelope=8192,
        )
        entries = [_valid_entry("e1")]

        anonymizer_gate.write_baseline(
            {"junk": 3},
            source="run X",
            current=current,
            run_record=None,
            entries=entries,
            path=path,
        )

        data = json.loads(path.read_text(encoding="utf-8"))
        # The scrubbed kinds and test set still come from this invocation's
        # scope, run record or not.
        assert data["configured_prefixes"] == ["Person"]
        assert data["corpus_sha256"] == anonymizer_gate._sha256_json(entries)
        assert data["model_id"] is None
        assert data["prompt_sha256"] is None
        assert data["table_sha256"] is None
        assert data["token_envelope"] is None


def _run_record(**overrides) -> anonymizer_gate.RunRecord:
    base = dict(
        configured_prefixes=["Person"],
        model_id="m",
        prompt_file=None,
        prompt_sha256="p",
        table_sha256="t",
        corpus_sha256="c",
        token_envelope=100,
    )
    base.update(overrides)
    return anonymizer_gate.RunRecord(**base)


_IDENTITY_LINE_ENTRIES = [_valid_entry("identity-line-entry")]
_IDENTITY_LINE_ENTRIES_DIGEST = anonymizer_gate._sha256_json(_IDENTITY_LINE_ENTRIES)


class TestIdentityVerdictLine:
    """``_identity_verdict_line`` never refuses — it only names which of a
    scored run's recorded inputs differ from the accepted baseline's, in
    plain words. The test set compared is always the entries actually
    handed to scoring (``entries``, :func:`_scored_inputs`'s own reading),
    never a ``RunRecord``'s own ``corpus_sha256`` field — every test below
    that expects "test set" absent from the line sets the baseline's
    ``corpus_sha256`` to :data:`_IDENTITY_LINE_ENTRIES_DIGEST` and passes
    :data:`_IDENTITY_LINE_ENTRIES` as *entries*, so the two match."""

    def test_nothing_is_said_when_every_recorded_input_matches(self) -> None:
        baseline = {
            "configured_prefixes": ["Person"],
            "model_id": "m",
            "prompt_sha256": "p",
            "table_sha256": "t",
            "corpus_sha256": _IDENTITY_LINE_ENTRIES_DIGEST,
            "token_envelope": 100,
        }

        line = anonymizer_gate._identity_verdict_line(
            _run_record(), baseline, _run_record(), _IDENTITY_LINE_ENTRIES
        )

        assert line is None

    def test_each_differing_input_is_named_in_plain_words(self) -> None:
        baseline = {
            "configured_prefixes": ["Person"],
            "model_id": "a-different-model",
            "prompt_sha256": "a-different-prompt",
            "table_sha256": "t",
            "corpus_sha256": _IDENTITY_LINE_ENTRIES_DIGEST,
            "token_envelope": 999,
        }

        line = anonymizer_gate._identity_verdict_line(
            _run_record(), baseline, _run_record(), _IDENTITY_LINE_ENTRIES
        )

        assert line is not None
        assert "model" in line
        assert "prompt" in line
        assert "token budget" in line
        assert "keyword table" not in line
        assert "scrubbed kinds" not in line
        assert "test set" not in line

    def test_a_value_missing_from_the_baseline_is_reported_as_not_recorded_by_the_baseline(
        self,
    ) -> None:
        baseline = {
            "configured_prefixes": ["Person"],
            "model_id": "m",
            "prompt_sha256": "p",
            "table_sha256": None,
            "corpus_sha256": _IDENTITY_LINE_ENTRIES_DIGEST,
            "token_envelope": 100,
        }

        line = anonymizer_gate._identity_verdict_line(
            _run_record(), baseline, _run_record(), _IDENTITY_LINE_ENTRIES
        )

        assert line is not None
        assert "keyword table (not recorded by the baseline)" in line

    def test_a_run_with_no_record_reports_model_and_prompt_as_not_recorded_by_this_run(
        self,
    ) -> None:
        """A run directory with no record misses model/prompt/table/token
        budget — but the test set is always the entries actually scored
        (:func:`_scored_inputs`), so it reads as matching the baseline
        whenever the two share a gold corpus, never as unrecorded just
        because the run carries no ``run.json``."""
        baseline = {
            "configured_prefixes": ["Person"],
            "model_id": "m",
            "prompt_sha256": "p",
            "table_sha256": None,
            "corpus_sha256": _IDENTITY_LINE_ENTRIES_DIGEST,
            "token_envelope": 100,
        }

        line = anonymizer_gate._identity_verdict_line(
            None, baseline, _run_record(), _IDENTITY_LINE_ENTRIES
        )

        assert line is not None
        assert "model (not recorded by this run)" in line
        assert "prompt (not recorded by this run)" in line
        assert "token budget (not recorded by this run)" in line
        assert "test set" not in line

    def test_a_baseline_recording_none_of_the_inputs_names_them_all_by_their_plain_labels(
        self,
    ) -> None:
        line = anonymizer_gate._identity_verdict_line(_run_record(), {}, _run_record(), [])

        assert line == (
            "the accepted baseline records none of these inputs: "
            "scrubbed kinds, model, prompt, keyword table, test set, token budget"
        )

    def test_an_input_recorded_by_neither_side_is_reported_not_silently_a_match(self) -> None:
        """A model/prompt/table input missing from the baseline AND from
        the scored run's own record is reported explicitly, never silently
        treated as a match by the shared absence. (The test set never has
        this case: :func:`_scored_inputs` always hashes the entries
        actually scored, which is never absent — see
        :func:`_current_run_record` and :func:`_scored_inputs` for the two
        distinct "test set" roles.)"""
        baseline = {
            "configured_prefixes": ["Person"],
            "model_id": "m",
            "prompt_sha256": "p",
            "table_sha256": None,
            "corpus_sha256": _IDENTITY_LINE_ENTRIES_DIGEST,
            "token_envelope": 100,
        }

        line = anonymizer_gate._identity_verdict_line(
            _run_record(table_sha256=None), baseline, _run_record(), _IDENTITY_LINE_ENTRIES
        )

        assert line == "differs from the accepted baseline: keyword table (recorded by neither)"


class TestScoreAndReportIdentityNotice:
    """``_score_and_report`` prints the identity notice beside the
    regression verdict — a differing input against the accepted baseline
    is always a notice, never a refusal, on both tests below (neither
    passes ``accept=True``, the one branch that can still return 1)."""

    def test_prints_the_identity_notice_beside_the_regression_columns_and_still_succeeds(
        self, tmp_path, capsys
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        anonymizer_gate._write_run_record(
            run_dir,
            anonymizer_gate.RunRecord(
                configured_prefixes=["Person"],
                model_id="model-a",
                prompt_file=None,
                prompt_sha256="hash-a",
                table_sha256="hash-a",
                corpus_sha256="hash-a",
                token_envelope=100,
            ),
        )
        baseline_path = tmp_path / "baseline.json"
        anonymizer_gate.write_baseline(
            {"junk": 0, "failed": 0},
            source="prior run",
            current=anonymizer_gate.RunRecord(
                configured_prefixes=["Person"],
                model_id="model-b",
                prompt_file=None,
                prompt_sha256="hash-a",
                table_sha256="hash-a",
                corpus_sha256="hash-a",
                token_envelope=100,
            ),
            run_record=anonymizer_gate.RunRecord(
                configured_prefixes=["Person"],
                model_id="model-b",
                prompt_file=None,
                prompt_sha256="hash-a",
                table_sha256="hash-a",
                corpus_sha256="hash-a",
                token_envelope=100,
            ),
            entries=[],  # matches _score_and_report's own `entries=[]` below
            path=baseline_path,
        )

        current = anonymizer_gate.RunRecord(
            configured_prefixes=["Person"],
            model_id="model-a",
            prompt_file=None,
            prompt_sha256="hash-a",
            table_sha256="hash-a",
            corpus_sha256="hash-a",
            token_envelope=100,
        )

        code = anonymizer_gate._score_and_report(
            [],
            {},
            {"Person"},
            run_dir,
            baseline_path,
            accept=False,
            write_scorecard=False,
            current=current,
            run_record=anonymizer_gate._read_run_record(run_dir),
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "no regression against the accepted baseline" in out
        assert "differs from the accepted baseline: model" in out

    def test_the_notice_compares_this_invocations_own_scrubbed_kinds_not_the_runs_recorded_ones(
        self, tmp_path, capsys
    ) -> None:
        """A complete run directory can be re-scored under a different
        ``--scrub`` than it was originally run with (:func:`_score_from_disk`)
        — the identity notice must read the scrubbed kinds THIS scoring
        invocation actually used, never the run's own stale recorded set,
        or a re-score under a narrower or wider ``--scrub`` would silently
        compare against the wrong scope."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        anonymizer_gate._write_run_record(
            run_dir,
            anonymizer_gate.RunRecord(
                configured_prefixes=["Person"],
                model_id="model-a",
                prompt_file=None,
                prompt_sha256="hash-a",
                table_sha256="hash-a",
                corpus_sha256="hash-a",
                token_envelope=100,
            ),
        )
        baseline_path = tmp_path / "baseline.json"
        anonymizer_gate.write_baseline(
            {"junk": 0, "failed": 0},
            source="prior run",
            current=anonymizer_gate.RunRecord(
                configured_prefixes=["Person"],
                model_id="model-a",
                prompt_file=None,
                prompt_sha256="hash-a",
                table_sha256="hash-a",
                corpus_sha256="hash-a",
                token_envelope=100,
            ),
            run_record=anonymizer_gate.RunRecord(
                configured_prefixes=["Person"],
                model_id="model-a",
                prompt_file=None,
                prompt_sha256="hash-a",
                table_sha256="hash-a",
                corpus_sha256="hash-a",
                token_envelope=100,
            ),
            entries=[],  # matches _score_and_report's own `entries=[]` below
            path=baseline_path,
        )
        # This invocation's own current values carry a narrower scrubbed
        # scope than both the run's own recorded set and the baseline's —
        # the notice reads THIS invocation's own `current.configured_prefixes`
        # (`_scored_inputs`), never the run's stale recorded set.
        current = anonymizer_gate.RunRecord(
            configured_prefixes=[],
            model_id="model-a",
            prompt_file=None,
            prompt_sha256="hash-a",
            table_sha256="hash-a",
            corpus_sha256="hash-a",
            token_envelope=100,
        )

        code = anonymizer_gate._score_and_report(
            [],
            {},
            set(),
            run_dir,
            baseline_path,
            accept=False,
            write_scorecard=False,
            current=current,
            run_record=anonymizer_gate._read_run_record(run_dir),
        )

        out = capsys.readouterr().out
        assert code == 0
        assert "differs from the accepted baseline: scrubbed kinds" in out


class TestPromptFileFailsBeforeAnyRunDirectory:
    """An unusable ``--prompt-file`` is reported and returns 1 before a run
    directory is ever created, for every case ``main`` distinguishes: the
    file does not exist; it exists but cannot be read; its bytes are not
    UTF-8; and — via
    :func:`~paramem.graph.prompts.check_anonymization_prompt_sections` —
    a missing required section, a section missing one of its own required
    slots or carrying a slot the table does not list for it, and a section
    carrying a malformed placeholder.

    The positive control (a valid ``--prompt-file`` passing every case) is
    :class:`TestMainValidPromptFileReachesTheGuardedPath`.
    """

    def test_a_missing_prompt_file_returns_one_before_any_run_directory_is_created(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)

        code = anonymizer_gate.main(["--prompt-file", str(tmp_path / "missing.txt")])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable" in err
        assert "does not exist" in err
        assert not run_root.exists()

    def test_a_prompt_file_that_is_a_directory_is_reported_as_unreadable(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """A ``--prompt-file`` that exists but cannot be read as a file
        (e.g. a directory) is its own case, distinct from "does not
        exist"."""
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        prompt_dir = tmp_path / "not_a_file"
        prompt_dir.mkdir()

        code = anonymizer_gate.main(["--prompt-file", str(prompt_dir)])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable" in err
        assert "could not be read" in err
        assert not run_root.exists()

    def test_a_prompt_file_that_is_not_utf8_returns_one_before_any_run_directory_is_created(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        prompt_file = tmp_path / "anonymization.txt"
        prompt_file.write_bytes(b"\xff\xfe not valid utf-8 bytes")

        code = anonymizer_gate.main(["--prompt-file", str(prompt_file)])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable" in err
        assert "not valid UTF-8" in err
        assert not run_root.exists()

    def test_a_prompt_file_missing_a_section_returns_one_before_any_run_directory_is_created(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        prompt_file = tmp_path / "anonymization.txt"
        prompt_file.write_text(INVALID_ANONYMIZATION_SECTIONS_MISSING_SECTION, encoding="utf-8")

        code = anonymizer_gate.main(["--prompt-file", str(prompt_file)])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable: missing section ANCHOR-SYSTEM" in err
        assert not run_root.exists()

    def test_a_section_missing_a_required_slot_returns_one_before_any_run_directory_is_created(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        prompt_file = tmp_path / "anonymization.txt"
        prompt_file.write_text(INVALID_ANONYMIZATION_SECTIONS_MISSING_SLOT, encoding="utf-8")

        code = anonymizer_gate.main(["--prompt-file", str(prompt_file)])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable: section SCAN is missing slot {text}" in err
        assert not run_root.exists()

    def test_a_section_carrying_an_unknown_slot_returns_one_before_any_run_directory_is_created(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        prompt_file = tmp_path / "anonymization.txt"
        prompt_file.write_text(INVALID_ANONYMIZATION_SECTIONS_UNKNOWN_SLOT, encoding="utf-8")

        code = anonymizer_gate.main(["--prompt-file", str(prompt_file)])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable: section ANCHOR carries an unknown slot {extra}" in err
        assert not run_root.exists()

    def test_a_malformed_placeholder_returns_one_before_any_run_directory_is_created(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        prompt_file = tmp_path / "anonymization.txt"
        prompt_file.write_text(
            INVALID_ANONYMIZATION_SECTIONS_MALFORMED_PLACEHOLDER, encoding="utf-8"
        )

        code = anonymizer_gate.main(["--prompt-file", str(prompt_file)])

        err = capsys.readouterr().err
        assert code == 1
        assert "--prompt-file unusable: section SCAN: malformed placeholder" in err
        assert not run_root.exists()


def _stub_full_run(monkeypatch, tmp_path, calls: list[str] | None = None) -> list[str]:
    """Stub every side effect ``main``'s guarded (model-bearing) path
    touches — the GPU guard, the model load, the per-entry model calls and
    the server GPU restore — so a full ``main()`` invocation runs against a
    one-entry fake corpus with no GPU, no model and no server call.

    Shared by every test that exercises ``main``'s guarded path end to
    end: the run-record write timing, the resume identity refusal/notice
    split, and the ``--accept`` refusal. ``"write_run_record"`` and
    ``"acquire_gpu"`` are appended into the returned list, in call order,
    so a caller can assert the run record is written before the GPU is
    acquired; a caller with no interest in that order may ignore the
    return value. ``_run_corpus`` is stubbed to hand back a fake completed
    contract per entry WITHOUT writing an entry artifact to disk, so the
    run directory a bare ``main()`` call produces here always still needs
    its entries written by hand before ``_run_dir_complete`` reads it as
    complete.

    Args:
        monkeypatch: The pytest ``monkeypatch`` fixture.
        tmp_path: The pytest ``tmp_path`` fixture — houses the run root
            and baseline path this invocation writes under.
        calls: An existing list to append into, or ``None`` to start a
            fresh one.

    Returns:
        The list ``"write_run_record"``/``"acquire_gpu"`` are appended
        into, in call order.
    """
    if calls is None:
        calls = []
    corpus = [_valid_entry("e1")]
    monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)
    monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")

    real_write = anonymizer_gate._write_run_record

    def _tracking_write(run_dir, record):
        calls.append("write_run_record")
        real_write(run_dir, record)

    monkeypatch.setattr(anonymizer_gate, "_write_run_record", _tracking_write)

    @contextlib.contextmanager
    def fake_acquire_gpu(*, name, interactive):
        calls.append("acquire_gpu")
        yield

    _stub_gpu_guard(monkeypatch, fake_acquire_gpu)
    monkeypatch.setattr(anonymizer_gate, "_wait_for_cooldown", lambda: None)
    monkeypatch.setattr(
        "paramem.models.loader.load_base_model",
        lambda cfg, tiers: (object(), _CharCountTokenizer()),
    )
    monkeypatch.setattr(anonymizer_gate, "_restore_server_gpu", lambda **kwargs: None)
    monkeypatch.setattr(
        anonymizer_gate,
        "_run_corpus",
        lambda entries, model, tokenizer, **kwargs: {e["id"]: _fake_contract() for e in entries},
    )
    return calls


class TestFreshRunRecordAndResumeInputIdentity:
    """Exercises ``main``'s guarded (model-bearing) path end to end, with
    the GPU guard, the model load and every per-entry model call stubbed
    out, to pin the ``run.json`` write's timing and the resume identity
    check's refuse/proceed split."""

    def test_run_record_is_written_before_the_gpu_guard_is_acquired(
        self, monkeypatch, tmp_path
    ) -> None:
        calls = _stub_full_run(monkeypatch, tmp_path)

        code = anonymizer_gate.main([])

        assert code == 0
        assert calls == ["write_run_record", "acquire_gpu"]

    def test_resume_with_matching_inputs_never_rewrites_the_record_and_reaches_the_guard(
        self, monkeypatch, tmp_path
    ) -> None:
        calls = _stub_full_run(monkeypatch, tmp_path)
        assert anonymizer_gate.main([]) == 0
        calls.clear()

        code = anonymizer_gate.main(["--resume"])

        assert code == 0
        assert calls == ["acquire_gpu"]

    def test_resume_with_a_different_model_refuses_before_any_load(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        calls = _stub_full_run(monkeypatch, tmp_path)
        assert anonymizer_gate.main([]) == 0
        calls.clear()
        capsys.readouterr()

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError(
                "must not touch the GPU guard or load a tokenizer/model on refusal"
            )

        _stub_gpu_guard(monkeypatch, _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_tokenizer", _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_base_model", _must_not_be_called)

        code = anonymizer_gate.main(["--resume", "--model", "qwen"])

        out = capsys.readouterr().out
        assert code == 1
        assert calls == []
        assert "--resume refused: differs from the run: model" in out
        assert "start a fresh run (without --resume) instead" in out

    def test_resume_with_a_different_token_envelope_refuses_before_any_load(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """The per-call token budget has no ``--model``-shaped CLI flag —
        it comes from ``server_cfg.consolidation.
        extraction_anonymize_token_envelope`` — so this test changes it by
        wrapping ``load_server_config`` itself, the same production
        resolution ``main`` reads it through."""
        calls = _stub_full_run(monkeypatch, tmp_path)
        assert anonymizer_gate.main([]) == 0
        calls.clear()
        capsys.readouterr()

        from paramem.server.config import load_server_config as _real_load_server_config

        def _load_with_a_different_token_envelope(path):
            cfg = _real_load_server_config(path)
            cfg.consolidation.extraction_anonymize_token_envelope += 1
            return cfg

        monkeypatch.setattr(
            "paramem.server.config.load_server_config",
            _load_with_a_different_token_envelope,
        )

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError(
                "must not touch the GPU guard or load a tokenizer/model on refusal"
            )

        _stub_gpu_guard(monkeypatch, _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_tokenizer", _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_base_model", _must_not_be_called)

        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 1
        assert calls == []
        assert "--resume refused: differs from the run: token budget" in out
        assert "start a fresh run (without --resume) instead" in out


class TestLimitPilotThenUnlimitedResume:
    """A ``--limit`` pilot's run IDENTITY (:func:`_current_run_record`) is
    the whole corpus the run belongs to, never the sliced entries it
    happens to score — so a later, unlimited ``--resume`` onto the same
    run directory is recognized as the same run and reaches the guarded
    (model-bearing) path instead of being refused for a "mismatched" test
    set."""

    def test_limit_pilot_then_unlimited_resume_reaches_the_guarded_path(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        corpus = [_valid_entry("e1"), _valid_entry("e2"), _valid_entry("e3")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", tmp_path / "runs")
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")

        acquire_calls: list[int] = []

        @contextlib.contextmanager
        def fake_acquire_gpu(*, name, interactive):
            acquire_calls.append(1)
            yield

        _stub_gpu_guard(monkeypatch, fake_acquire_gpu)
        monkeypatch.setattr(anonymizer_gate, "_wait_for_cooldown", lambda: None)
        monkeypatch.setattr(
            "paramem.models.loader.load_base_model",
            lambda cfg, tiers: (object(), _CharCountTokenizer()),
        )
        monkeypatch.setattr(anonymizer_gate, "_restore_server_gpu", lambda **kwargs: None)

        def _fake_run_corpus(entries, model, tokenizer, *, run_dir, resume, **kwargs):
            # Writes a real artifact per entry — unlike _stub_full_run's
            # in-memory-only lambda — so a pilot's own entries persist to
            # disk and a later --resume can see the run as still
            # incomplete (one entry short) rather than already complete.
            contracts = {}
            entries_dir = run_dir / "entries"
            for entry in entries:
                existing = resume and anonymizer_gate._load_entry_artifact(entries_dir, entry["id"])
                if existing:
                    contracts[entry["id"]] = existing
                    continue
                contract = _fake_contract()
                anonymizer_gate._write_entry_artifact(entries_dir, entry["id"], contract, 0.01)
                contracts[entry["id"]] = contract
            return contracts

        monkeypatch.setattr(anonymizer_gate, "_run_corpus", _fake_run_corpus)

        # Pilot: --limit 2 writes real artifacts for e1/e2 only.
        assert anonymizer_gate.main(["--limit", "2"]) == 0

        capsys.readouterr()
        acquire_calls.clear()

        # Unlimited resume: e3 is still missing on disk, so the run is not
        # complete; its run identity (the whole corpus) still matches this
        # invocation's own, so it must proceed to the guarded path rather
        # than refuse.
        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 0
        assert "refused" not in out
        assert acquire_calls == [1]


class TestLimitPilotVerdictReportsTheChangedTestSet:
    """The scored-inputs test set (:func:`_scored_inputs`) is the entries
    actually handed to scoring — a ``--limit`` pilot's own sliced entries —
    so its verdict against a full-corpus baseline correctly reports the
    test set as differing, even though the pilot's run IDENTITY (model,
    prompt, table, and the whole-corpus digest in its own ``run.json``)
    matches the baseline's."""

    def test_limit_pilot_verdict_names_the_test_set_as_differing(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        corpus = [_valid_entry("e1"), _valid_entry("e2"), _valid_entry("e3")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", tmp_path / "runs")
        baseline_path = tmp_path / "baseline.json"
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", baseline_path)

        from paramem.server.config import load_server_config

        server_cfg = load_server_config("tests/fixtures/server.yaml")
        configured = {c.prefix for c in anonymizer_gate.resolve_categories(None)}
        record = anonymizer_gate._current_run_record(
            configured=configured,
            model_id=server_cfg.model_config.model_id,
            prompt_file=None,
            token_envelope=server_cfg.consolidation.extraction_anonymize_token_envelope,
        )
        # The baseline was accepted from a run scored over the WHOLE
        # corpus — same model/prompt/table/token-budget this invocation
        # will build.
        anonymizer_gate.write_baseline(
            {"junk": 0, "failed": 0},
            source="prior full run",
            current=record,
            run_record=record,
            entries=corpus,
            path=baseline_path,
        )

        @contextlib.contextmanager
        def fake_acquire_gpu(*, name, interactive):
            yield

        _stub_gpu_guard(monkeypatch, fake_acquire_gpu)
        monkeypatch.setattr(anonymizer_gate, "_wait_for_cooldown", lambda: None)
        monkeypatch.setattr(
            "paramem.models.loader.load_base_model",
            lambda cfg, tiers: (object(), _CharCountTokenizer()),
        )
        monkeypatch.setattr(anonymizer_gate, "_restore_server_gpu", lambda **kwargs: None)
        monkeypatch.setattr(
            anonymizer_gate,
            "_run_corpus",
            lambda entries, model, tokenizer, **kwargs: {
                e["id"]: _fake_contract() for e in entries
            },
        )

        code = anonymizer_gate.main(["--limit", "2"])

        out = capsys.readouterr().out
        assert code == 0
        # Model/prompt/table/scrubbed-kinds all match (same real config,
        # same fixture, no override) — only the test set differs, since
        # this pilot scored 2 of the corpus's 3 entries.
        assert "differs from the accepted baseline: test set" in out


def _delete_run_json(run_dir: Path) -> None:
    (run_dir / "run.json").unlink()


def _drop_key(key: str):
    """A tamper function that removes *key* entirely from ``run.json``."""

    def _tamper(run_dir: Path) -> None:
        run_json = run_dir / "run.json"
        data = json.loads(run_json.read_text(encoding="utf-8"))
        data.pop(key, None)
        run_json.write_text(json.dumps(data), encoding="utf-8")

    return _tamper


def _null_key(key: str):
    """A tamper function that sets *key* to an explicit ``null`` in ``run.json``."""

    def _tamper(run_dir: Path) -> None:
        run_json = run_dir / "run.json"
        data = json.loads(run_json.read_text(encoding="utf-8"))
        data[key] = None
        run_json.write_text(json.dumps(data), encoding="utf-8")

    return _tamper


class TestResumeRefusesWhenAnInputIsNotRecorded:
    """The "not recorded" half of the incomplete-resume refusal: a run
    directory whose recorded provenance is entirely missing, or missing
    one of its six inputs, refuses ``--resume`` before any load — exactly
    as a run whose recorded inputs differ does. Parametrized over five
    shapes a "not recorded" input takes: no ``run.json`` at all, and a
    present record individually missing ``table_sha256``,
    ``prompt_sha256`` (explicit ``null``), ``corpus_sha256`` or
    ``token_envelope``.
    """

    def _forbid_any_load(self, monkeypatch) -> None:
        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("a refused --resume must never load a tokenizer or a model")

        _stub_gpu_guard(monkeypatch, _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_tokenizer", _must_not_be_called)
        monkeypatch.setattr("paramem.models.loader.load_base_model", _must_not_be_called)

    @pytest.mark.parametrize(
        "tamper, expected_labels",
        [
            pytest.param(
                _delete_run_json,
                (
                    "scrubbed kinds",
                    "model",
                    "prompt",
                    "keyword table",
                    "test set",
                    "token budget",
                ),
                id="run_json_absent",
            ),
            pytest.param(
                _drop_key("table_sha256"),
                ("keyword table",),
                id="table_sha256_missing",
            ),
            pytest.param(
                _null_key("prompt_sha256"),
                ("prompt",),
                id="prompt_sha256_null",
            ),
            pytest.param(
                _drop_key("corpus_sha256"),
                ("test set",),
                id="corpus_sha256_missing",
            ),
            pytest.param(
                _drop_key("token_envelope"),
                ("token budget",),
                id="token_envelope_missing",
            ),
        ],
    )
    def test_refuses_before_any_load_and_names_the_unrecorded_inputs(
        self, monkeypatch, tmp_path, capsys, tamper, expected_labels
    ) -> None:
        _stub_full_run(monkeypatch, tmp_path)
        assert anonymizer_gate.main([]) == 0
        run_dir = anonymizer_gate._latest_run_dir(tmp_path / "runs")
        assert run_dir is not None
        assert anonymizer_gate._run_dir_complete([_valid_entry("e1")], run_dir) is False
        capsys.readouterr()

        tamper(run_dir)
        self._forbid_any_load(monkeypatch)

        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 1
        # The exact set of labels named, not merely their presence — a run
        # missing only one input must not also print the other five's
        # labels by accident of a different, unrelated line in the output.
        if expected_labels == (
            "scrubbed kinds",
            "model",
            "prompt",
            "keyword table",
            "test set",
            "token budget",
        ):
            assert (
                f"--resume refused: run {run_dir.name} has no run.json; none of "
                f"{', '.join(expected_labels)} are recorded" in out
            )
        else:
            assert f"--resume refused: differs from the run: {', '.join(expected_labels)}" in out
        assert "start a fresh run (without --resume) instead" in out


class TestMainAcceptRefusesOnDifferingInputs:
    """``--accept`` on a complete run directory writes the baseline when
    this invocation's own inputs match what the run itself recorded;
    refuses, without writing, when they differ."""

    def _complete_the_run(self, monkeypatch, tmp_path) -> Path:
        """Produce a complete run directory via a real, fully-stubbed
        ``main([])`` invocation (a real ``run.json`` with real digests),
        then write the one entry's artifact by hand so
        ``_run_dir_complete`` reads the directory as complete."""
        _stub_full_run(monkeypatch, tmp_path)
        assert anonymizer_gate.main([]) == 0
        run_dir = anonymizer_gate._latest_run_dir(tmp_path / "runs")
        assert run_dir is not None
        anonymizer_gate._write_entry_artifact(run_dir / "entries", "e1", _fake_contract(), 0.1)
        return run_dir

    def _stub_score_from_disk_load(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "paramem.models.loader.load_tokenizer", lambda cfg: _CharCountTokenizer()
        )

    def test_accept_writes_the_baseline_when_inputs_match(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_dir = self._complete_the_run(monkeypatch, tmp_path)
        self._stub_score_from_disk_load(monkeypatch)
        capsys.readouterr()

        code = anonymizer_gate.main(["--resume", "--accept"])

        out = capsys.readouterr().out
        assert code == 0
        assert "baseline accepted" in out
        run_record = anonymizer_gate._read_run_record(run_dir)
        baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
        assert baseline["source"] == f"run {run_dir.name}"
        assert "junk" in baseline
        assert "failed" in baseline
        assert baseline["model_id"] == run_record.model_id
        assert baseline["prompt_sha256"] == run_record.prompt_sha256
        assert baseline["table_sha256"] == run_record.table_sha256
        assert baseline["corpus_sha256"] == run_record.corpus_sha256
        assert baseline["token_envelope"] == run_record.token_envelope
        assert baseline["configured_prefixes"] == run_record.configured_prefixes

    def test_accept_refuses_and_writes_nothing_when_the_model_differs(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        run_dir = self._complete_the_run(monkeypatch, tmp_path)
        self._stub_score_from_disk_load(monkeypatch)
        baseline_path = tmp_path / "baseline.json"
        # A pre-existing scorecard.json must survive a refused --accept
        # untouched — the refusal writes neither the baseline nor a fresh
        # scorecard over whatever a prior run already recorded here.
        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text('{"sentinel": true}', encoding="utf-8")
        capsys.readouterr()

        code = anonymizer_gate.main(["--resume", "--accept", "--model", "qwen"])

        out = capsys.readouterr().out
        assert code == 1
        assert "--accept refused: differs from the run: model" in out
        assert not baseline_path.exists()
        assert scorecard_path.read_text(encoding="utf-8") == '{"sentinel": true}'

    def test_accept_refuses_and_writes_nothing_when_the_token_envelope_differs(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """The per-call token budget has no ``--model``-shaped CLI flag —
        it comes from ``server_cfg.consolidation.
        extraction_anonymize_token_envelope`` — so this test changes it by
        wrapping ``load_server_config`` itself, the same production
        resolution ``main`` reads it through."""
        run_dir = self._complete_the_run(monkeypatch, tmp_path)
        self._stub_score_from_disk_load(monkeypatch)
        baseline_path = tmp_path / "baseline.json"
        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text('{"sentinel": true}', encoding="utf-8")
        capsys.readouterr()

        from paramem.server.config import load_server_config as _real_load_server_config

        def _load_with_a_different_token_envelope(path):
            cfg = _real_load_server_config(path)
            cfg.consolidation.extraction_anonymize_token_envelope += 1
            return cfg

        monkeypatch.setattr(
            "paramem.server.config.load_server_config",
            _load_with_a_different_token_envelope,
        )

        code = anonymizer_gate.main(["--resume", "--accept"])

        out = capsys.readouterr().out
        assert code == 1
        assert "--accept refused: differs from the run: token budget" in out
        assert not baseline_path.exists()
        assert scorecard_path.read_text(encoding="utf-8") == '{"sentinel": true}'

    def test_accept_refuses_and_writes_nothing_when_the_run_never_recorded_its_inputs(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """A complete run directory with no ``run.json`` at all — built by
        hand rather than by this tool — has nothing this invocation can
        vouch for; ``--accept`` refuses it exactly as it refuses a run
        whose recorded inputs differ."""
        run_dir = self._complete_the_run(monkeypatch, tmp_path)
        (run_dir / "run.json").unlink()
        self._stub_score_from_disk_load(monkeypatch)
        baseline_path = tmp_path / "baseline.json"
        scorecard_path = run_dir / "scorecard.json"
        scorecard_path.write_text('{"sentinel": true}', encoding="utf-8")
        capsys.readouterr()

        code = anonymizer_gate.main(["--resume", "--accept"])

        out = capsys.readouterr().out
        assert code == 1
        assert (
            "--accept refused: run "
            f"{run_dir.name} has no run.json; none of scrubbed kinds, model, prompt, "
            "keyword table, test set, token budget are recorded" in out
        )
        assert not baseline_path.exists()
        assert scorecard_path.read_text(encoding="utf-8") == '{"sentinel": true}'


class TestMainValidPromptFileReachesTheGuardedPath:
    """A valid ``--prompt-file`` passes through ``main`` past the section/
    slot validator into the guarded (model-bearing) run — the positive
    control for ``TestPromptFileFailsBeforeAnyRunDirectory``, which only
    exercises the refusal cases."""

    def test_a_valid_prompt_file_reaches_the_guarded_run(self, monkeypatch, tmp_path) -> None:
        prompt_file = tmp_path / "variant.txt"
        prompt_file.write_text(VALID_ANONYMIZATION_SECTIONS, encoding="utf-8")
        calls = _stub_full_run(monkeypatch, tmp_path)

        code = anonymizer_gate.main(["--prompt-file", str(prompt_file)])

        assert code == 0
        assert "acquire_gpu" in calls
