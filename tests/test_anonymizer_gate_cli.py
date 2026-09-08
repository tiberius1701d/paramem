"""The anonymizer gate's CLI arms (``scripts/dev/anonymizer_gate.py``): the
``--accept``/``--limit`` precondition, the ``--cooldown-every`` parser
validation, the cooldown chunking's resume exemption, the skeleton
re-measurement print, and ``main``'s ``--dry-run``/``--resume`` dispatch —
no model, no GPU.

``test_anonymizer_gate_scorer.py`` covers the scorer itself
(``score_entry``/``score_corpus``/``Result``); this module covers the
argument parser and the corpus-run loop's own control flow.
"""

from __future__ import annotations

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
# RunRecord — the run directory's provenance (configured prefixes, model id,
# prompt sha256), written once at creation, read and compared by the
# score-from-disk path.
# ---------------------------------------------------------------------------


class TestRunRecord:
    def test_write_then_read_round_trips(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        record = anonymizer_gate.RunRecord(
            configured_prefixes=["Email", "Person"], model_id="model-x", prompt_file=None
        )

        anonymizer_gate._write_run_record(run_dir, record)

        assert anonymizer_gate._read_run_record(run_dir) == record

    def test_read_missing_returns_none(self, tmp_path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        assert anonymizer_gate._read_run_record(run_dir) is None

    def test_current_run_record_sorts_prefixes_and_stringifies_the_prompt_file(
        self, tmp_path
    ) -> None:
        record = anonymizer_gate._current_run_record(
            configured={"Person", "Email"},
            model_id="model-x",
            prompt_file=tmp_path / "variant.txt",
        )

        assert record.configured_prefixes == ["Email", "Person"]
        assert record.prompt_file == str(tmp_path / "variant.txt")

    def test_current_run_record_prompt_file_none_stays_none(self) -> None:
        record = anonymizer_gate._current_run_record(
            configured=set(), model_id="model-x", prompt_file=None
        )

        assert record.prompt_file is None

    def test_report_prints_no_record_message_when_run_json_is_absent(
        self, tmp_path, capsys
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        current = anonymizer_gate._current_run_record(
            configured={"Person"}, model_id="model-x", prompt_file=None
        )

        anonymizer_gate._report_run_record(run_dir, current)

        out = capsys.readouterr().out
        assert "carries no run.json record" in out

    def test_report_names_no_differing_fields_when_matching(self, tmp_path, capsys) -> None:
        """Same recorded and current values, digest included — a run whose
        ``--prompt-file`` content is unchanged from what was recorded never
        names ``prompt sha256`` as differing."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        current = anonymizer_gate._current_run_record(
            configured={"Person", "Email"},
            model_id="model-x",
            prompt_file=Path("variant.txt"),
            prompt_override_text="shared prompt text",
        )
        anonymizer_gate._write_run_record(run_dir, current)

        anonymizer_gate._report_run_record(run_dir, current)

        out = capsys.readouterr().out
        assert "recorded:" in out
        assert "differs from this invocation" not in out

    def test_report_names_each_differing_field(self, tmp_path, capsys) -> None:
        """Different recorded and current prompt override text yields a
        different ``prompt_sha256`` digest, named as differing."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        recorded = anonymizer_gate._current_run_record(
            configured={"Person"},
            model_id="model-old",
            prompt_file=None,
            prompt_override_text="old prompt text",
        )
        anonymizer_gate._write_run_record(run_dir, recorded)
        current = anonymizer_gate._current_run_record(
            configured={"Person", "Email"},
            model_id="model-new",
            prompt_file=Path("override.txt"),
            prompt_override_text="new prompt text",
        )

        anonymizer_gate._report_run_record(run_dir, current)

        out = capsys.readouterr().out
        assert "configured prefixes" in out
        assert "model id" in out
        assert "prompt sha256" in out


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
    the GPU guard as normal."""

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

        load_tokenizer_calls = self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        code = anonymizer_gate.main(["--resume"])

        out = capsys.readouterr().out
        assert code == 0
        assert "scored from disk" in out
        # This run directory was built by hand above with no run.json, so
        # the score-only path reports that.
        assert "carries no run.json record" in out
        assert len(load_tokenizer_calls) == 1
        scorecard = json.loads((run_dir / "scorecard.json").read_text(encoding="utf-8"))
        assert "failed" in scorecard

    def test_resume_never_rewrites_an_existing_run_json(self, monkeypatch, tmp_path) -> None:
        corpus = [_valid_entry("e1"), _valid_entry("e2")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e2", _fake_contract(), 0.1)

        original_record = anonymizer_gate.RunRecord(
            configured_prefixes=["Person"], model_id="original-model", prompt_file=None
        )
        anonymizer_gate._write_run_record(run_dir, original_record)

        self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        anonymizer_gate.main(["--resume"])

        # This invocation's own model id (the fixture's mistral entry)
        # differs from "original-model" — proof the record on disk is
        # untouched, not merely unchanged by coincidence.
        assert anonymizer_gate._read_run_record(run_dir) == original_record

    def test_limit_on_a_complete_run_directory_never_writes_the_scorecard(
        self, monkeypatch, tmp_path
    ) -> None:
        """``--limit`` never overwrites a complete run's own full-corpus
        ``scorecard.json`` from a slice."""
        corpus = [_valid_entry("e1"), _valid_entry("e2"), _valid_entry("e3")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e2", _fake_contract(), 0.1)

        self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        code = anonymizer_gate.main(["--resume", "--limit", "2"])

        assert code == 0
        assert not (run_dir / "scorecard.json").exists()

    def test_a_run_directory_missing_one_artifact_enters_the_guarded_path(
        self, monkeypatch, tmp_path
    ) -> None:
        corpus = [_valid_entry("e1"), _valid_entry("e2")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        # e2's artifact is deliberately absent -> the run directory is
        # incomplete -> the guarded (model-bearing) path must be entered.

        self._stub_load_tokenizer(monkeypatch)
        monkeypatch.setattr(anonymizer_gate, "_restore_server_gpu", lambda **kwargs: None)

        class _GuardedPathReached(Exception):
            pass

        def _acquire_gpu(*args, **kwargs):
            raise _GuardedPathReached("acquire_gpu was called — the guarded path was entered")

        _stub_gpu_guard(monkeypatch, _acquire_gpu)

        with pytest.raises(_GuardedPathReached):
            anonymizer_gate.main(["--resume"])

    def test_accept_writes_the_baseline_file_with_failed_and_source(
        self, monkeypatch, tmp_path
    ) -> None:
        corpus = [_valid_entry("e1"), _valid_entry("e2")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_dir = tmp_path / "run"
        monkeypatch.setattr(anonymizer_gate, "_latest_run_dir", lambda root=None: run_dir)
        entries_dir = run_dir / "entries"
        anonymizer_gate._write_entry_artifact(entries_dir, "e1", _fake_contract(), 0.1)
        anonymizer_gate._write_entry_artifact(entries_dir, "e2", _fake_contract(), 0.1)

        self._stub_load_tokenizer(monkeypatch)
        self._forbid_gpu_and_model(monkeypatch)

        # Redirect every disk access this run makes — the run root (unused
        # here since _latest_run_dir is stubbed above, but redirected for
        # the same reason _BASELINE_PATH is) and the baseline path — by
        # monkeypatching the module constants main() reads at call time,
        # never the real data/ha/ tree.
        tmp_run_root = tmp_path / "runs"
        tmp_baseline = tmp_path / "baseline.json"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", tmp_run_root)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_baseline)

        code = anonymizer_gate.main(["--resume", "--accept"])

        assert code == 0
        data = json.loads(tmp_baseline.read_text(encoding="utf-8"))
        assert "failed" in data
        assert data["source"] == f"run {run_dir.name}"

    def test_a_fresh_run_directory_writes_run_json_before_the_guarded_path(
        self, monkeypatch, tmp_path
    ) -> None:
        """A freshly created run directory (no ``--resume``, or ``--resume``
        with nothing yet on disk) writes ``run.json`` before the GPU guard
        is ever entered — the provenance a later score-from-disk run
        compares against."""
        corpus = [_valid_entry("e1")]
        monkeypatch.setattr(anonymizer_gate, "load_corpus", lambda: corpus)

        run_root = tmp_path / "runs"
        monkeypatch.setattr(anonymizer_gate, "_RUN_ROOT", run_root)
        monkeypatch.setattr(anonymizer_gate, "_BASELINE_PATH", tmp_path / "baseline.json")

        self._stub_load_tokenizer(monkeypatch)
        monkeypatch.setattr(anonymizer_gate, "_restore_server_gpu", lambda **kwargs: None)

        class _GuardedPathReached(Exception):
            pass

        def _acquire_gpu(*args, **kwargs):
            raise _GuardedPathReached("acquire_gpu was called — the guarded path was entered")

        _stub_gpu_guard(monkeypatch, _acquire_gpu)

        with pytest.raises(_GuardedPathReached):
            anonymizer_gate.main([])

        run_dirs = [d for d in run_root.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1
        record = anonymizer_gate._read_run_record(run_dirs[0])
        assert record is not None
        assert record.prompt_file is None
        assert record.model_id
        assert record.configured_prefixes == sorted(record.configured_prefixes)
