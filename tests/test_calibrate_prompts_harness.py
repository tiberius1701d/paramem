"""Regression tests for scripts/dev/calibrate_prompts.py.

Specifically guards the NameError that occurred when --stages normalize was
invoked with an empty chunk loop: params_base was assigned inside the
for-chunk loop and therefore never bound when chunks == [].

Also covers the auth gap: _post_stage must attach the Authorization header
and raise SystemExit with an actionable message on 401.

``TestSeedFromEnrichLoading`` covers the ``--seed-from --stages enrich`` leak
class: ``01_extract_chunk_N.json`` is written as a WRAPPER
(``{"stage", "chunk_index", "candidate_runs": [...]}``), unlike ``02_``/``03_``
which write the raw stage response directly — loading the wrapper as-is left
``prior_extract`` with no usable ``"parsed"`` graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make the script importable without installing it as a package.
_SCRIPTS_DEV = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(_SCRIPTS_DEV) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DEV))

import calibrate_prompts  # noqa: E402 (scripts/dev is not a package)

_CANNED_NORMALIZE_RESPONSE = {
    "stage": "normalize",
    "filtered": [],
    "filter_prompt_used": "normalize_filter.txt",
    "raw_output": "[]",
    # Every calibration response names the directory the SERVER wrote the
    # run's artifacts to; the client records that pointer rather than
    # keeping its own copy of the run.
    "artifact_dir": "/tmp/paramem-calibration/normalize_1",
}


class TestNormalizeStageNoNameError:
    """--stages normalize must not raise NameError when chunks is empty."""

    def test_normalize_records_the_run_in_the_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Invoke main(['--stages','normalize',...]) with _post_stage mocked.

        Asserts:
        - No NameError (params_base is bound before the chunk loop)
        - the run is recorded in ``runs.json`` by the directory the SERVER
          wrote it to — a single-seed run has nothing to compare, so the
          client writes no file of its own and keeps no copy of the response
        """
        # Minimal snapshot file — content is opaque to the harness (passed as a
        # path string to the server endpoint, not parsed locally).
        snapshot = tmp_path / "graph_merged_snapshot.json"
        snapshot.write_text(
            json.dumps(
                {
                    "directed": False,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [
                        {
                            "id": "alice",
                            "attributes": {"name": "Alice"},
                            "speaker_id": "speaker0",
                        }
                    ],
                    "links": [],
                }
            )
        )

        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()

        # Use the real configs/prompts dir so --prompts-dir validation passes.
        real_prompts_dir = Path(__file__).resolve().parents[1] / "configs" / "prompts"

        argv = [
            "--stages",
            "normalize",
            "--snapshot",
            str(snapshot),
            "--dump-dir",
            str(dump_dir),
            "--server",
            "http://localhost:8420",
            "--prompts-dir",
            str(real_prompts_dir),
            "--baseline",
            "none",
        ]

        with patch.object(
            calibrate_prompts, "_post_stage", return_value=_CANNED_NORMALIZE_RESPONSE
        ):
            rc = calibrate_prompts.main(argv)

        assert rc == 0, f"Expected rc=0, got {rc}"

        index = json.loads((dump_dir / "runs.json").read_text())
        assert index["normalize"] == {"None": _CANNED_NORMALIZE_RESPONSE["artifact_dir"]}
        # Single seed: nothing to compare, so no client-side file at all.
        assert not (dump_dir / "05_normalize.json").exists()

    def test_params_base_bound_before_chunk_loop(self):
        """Unit-level guard: params_base must be referenced in the module source
        BEFORE the for-chunk loop, not inside it.

        This is a structural assertion over the source text — it will catch any
        future accidental regression that re-introduces the assignment inside the
        loop.
        """
        source = Path(calibrate_prompts.__file__).read_text()
        lines = source.splitlines()

        # Find the line numbers of the two markers.
        params_base_line = None
        chunk_loop_line = None
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            next_line = lines[i] if i < len(lines) else ""
            if (
                params_base_line is None
                and stripped.startswith("params_base")
                and "temperature" in next_line
            ):
                params_base_line = i
            if stripped.startswith("for chunk in chunks:") and chunk_loop_line is None:
                chunk_loop_line = i

        assert params_base_line is not None, (
            "Could not locate 'params_base = ...' assignment in calibrate_prompts.py"
        )
        assert chunk_loop_line is not None, (
            "Could not locate 'for chunk in chunks:' loop in calibrate_prompts.py"
        )
        assert params_base_line < chunk_loop_line, (
            f"params_base (line {params_base_line}) must be assigned BEFORE "
            f"'for chunk in chunks:' (line {chunk_loop_line}). "
            "The NameError regression has been re-introduced."
        )


class TestLoadChunksTurnMarking:
    """_load_chunks must render ``text`` through the SAME production turn
    renderer (``SessionBuffer._format_turns``), never a hand-rolled
    marker — for every input shape (document AND transcript)."""

    def test_txt_document_chunk_is_turn_marked(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("Alex works at Brightfield Labs.", encoding="utf-8")

        chunks, source_type = calibrate_prompts._load_chunks(f, None)

        assert source_type == "document"
        assert len(chunks) >= 1
        assert chunks[0]["text"].startswith("[user] "), (
            f"Document chunk must be turn-marked: {chunks[0]['text']!r}"
        )
        assert "Alex works at Brightfield Labs." in chunks[0]["text"]

    def test_jsonl_transcript_preserves_role_alternation(self, tmp_path: Path):
        f = tmp_path / "session.jsonl"
        f.write_text(
            "\n".join(
                [
                    json.dumps({"role": "user", "text": "Hi there."}),
                    json.dumps({"role": "assistant", "text": "Hello, how can I help?"}),
                ]
            ),
            encoding="utf-8",
        )

        chunks, source_type = calibrate_prompts._load_chunks(f, None)

        assert source_type == "transcript"
        assert len(chunks) == 1
        text = chunks[0]["text"]
        assert text.startswith("[user] Hi there."), f"Got: {text!r}"
        assert "[assistant] Hello, how can I help?" in text

    def test_matches_sessionbuffer_format_turns_directly(self, tmp_path: Path):
        """The rendered text is byte-identical to calling the renderer
        directly — not a shape mimic (CLAUDE.md: no parallel renderer)."""
        from paramem.server.session_buffer import SessionBuffer

        f = tmp_path / "notes.txt"
        f.write_text("Some document content.", encoding="utf-8")

        chunks, _ = calibrate_prompts._load_chunks(f, None)

        expected_lines, _ = SessionBuffer._format_turns(
            [{"role": "user", "text": "Some document content."}]
        )
        assert chunks[0]["text"] == "\n".join(expected_lines)


class TestPostStageAuth:
    """_post_stage must attach an Authorization header and handle 401 gracefully."""

    def test_bearer_header_attached_when_token_present(self):
        """_post_stage sends Authorization: Bearer <token> when resolve_token returns a token."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}

        with (
            patch.object(calibrate_prompts, "resolve_token", return_value="test-secret-token"),
            patch("calibrate_prompts.requests.post", return_value=mock_response) as mock_post,
        ):
            result = calibrate_prompts._post_stage(
                "http://localhost:8420", "normalize", {"snapshot_path": "/tmp/snap.json"}
            )

        assert result == {"ok": True}
        _, kwargs = mock_post.call_args
        assert "headers" in kwargs, "_post_stage did not pass headers kwarg to requests.post"
        headers = kwargs["headers"]
        assert headers.get("Authorization") == "Bearer test-secret-token", (
            f"Expected 'Bearer test-secret-token', got: {headers.get('Authorization')!r}"
        )

    def test_401_raises_systemexit_with_actionable_message(self):
        """_post_stage raises SystemExit with a helpful message when the server returns 401."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with (
            patch.object(calibrate_prompts, "resolve_token", return_value=None),
            patch("calibrate_prompts.requests.post", return_value=mock_response),
        ):
            with pytest.raises(SystemExit) as exc_info:
                calibrate_prompts._post_stage(
                    "http://localhost:8420", "normalize", {"snapshot_path": "/tmp/snap.json"}
                )

        message = str(exc_info.value)
        assert "401" in message, f"Expected '401' in SystemExit message, got: {message!r}"
        assert "PARAMEM_API_TOKEN" in message, (
            f"Expected 'PARAMEM_API_TOKEN' in SystemExit message, got: {message!r}"
        )


class TestAnonymizeStageSpeakerName:
    """``--speaker`` must reach the
    ``/calibrate/anonymize`` request payload, not just
    ``/calibrate/extract`` — production's ``anonymize`` always
    threads the runtime-known speaker name into speaker-name seeding;
    omitting it here silently diverges calibration fidelity from
    production and can leave the speaker's real name un-scrubbed before
    a real ``/calibrate/enrich`` cloud call.
    """

    _REAL_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "prompts"

    def test_speaker_name_reaches_anonymize_payload(self, tmp_path: Path):
        input_path = tmp_path / "input.txt"
        input_path.write_text("Alex works as an engineer at Acme Corp.")
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()

        calls: list[tuple[str, dict]] = []

        def _fake_post_stage(server, stage, payload):
            calls.append((stage, payload))
            if stage == "extract":
                return {
                    "stage": "extract",
                    "raw_output": "{}",
                    "parsed": {
                        "session_id": "calib-chunk-0",
                        "timestamp": "2026-07-14T00:00:00Z",
                        "entities": [],
                        "relations": [],
                    },
                    "parse_error": None,
                }
            return {"stage": stage, "raw_output": "{}", "parsed": {}, "parse_error": None}

        argv = [
            "--input",
            str(input_path),
            "--source-type",
            "transcript",
            "--chunk",
            "0",
            "--stages",
            "extract,anonymize",
            "--speaker",
            "Alex",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with patch.object(calibrate_prompts, "_post_stage", side_effect=_fake_post_stage):
            rc = calibrate_prompts.main(argv)

        assert rc == 0
        anonymize_calls = [payload for stage, payload in calls if stage == "anonymize"]
        assert anonymize_calls, "expected an anonymize stage call"
        assert anonymize_calls[0]["speaker_name"] == "Alex"


class TestSeedFromEnrichLoading:
    """``--seed-from --stages enrich`` reads the prior EXTRACT dump off
    disk instead of re-running extraction.

    The graph is the only artifact the client hands over: every stage
    past ``local_extract`` is entered server-side with that graph as its
    seed, and the chain re-derives the anonymized facts, the anonymized
    transcript and the fail-closed decisions by running. The client no
    longer relays any of them.
    """

    _REAL_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "prompts"

    @staticmethod
    def _write_input(tmp_path: Path) -> Path:
        input_path = tmp_path / "input.txt"
        input_path.write_text("Alex works as an engineer at Acme Corp in Berlin.")
        return input_path

    @staticmethod
    def _recorded_run() -> dict:
        """The SERVER's record of an extract run — a calibration response."""
        return {
            "stage": "extract",
            "seed": None,
            "parsed": {
                "session_id": "calib-chunk-0",
                "timestamp": "2026-07-14T00:00:00Z",
                "entities": [
                    {"name": "Alex", "entity_type": "person", "speaker_id": "speaker0"},
                    {"name": "Acme Corp", "entity_type": "organization"},
                ],
                "relations": [
                    {
                        "subject": "speaker0",
                        "predicate": "works_at",
                        "object": "Acme Corp",
                        "relation_type": "factual",
                        "confidence": 0.9,
                        "speaker_id": "speaker0",
                    }
                ],
            },
        }

    def _run(self, tmp_path: Path, *, write_extract: bool = True) -> tuple[int, MagicMock]:
        seed_from = tmp_path / "seed_from"
        seed_from.mkdir()
        if write_extract:
            # The client's index points at the directory the SERVER wrote the
            # run to; the run itself lives there, in one copy.
            artifact_dir = tmp_path / "artifacts" / "extract_1"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "calibration_extract_1.json").write_text(
                json.dumps(self._recorded_run())
            )
            (seed_from / "runs.json").write_text(json.dumps({"extract": {"0": str(artifact_dir)}}))

        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        argv = [
            "--input",
            str(self._write_input(tmp_path)),
            "--source-type",
            "transcript",
            "--chunk",
            "0",
            "--stages",
            "enrich",
            "--seed-from",
            str(seed_from),
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        fake_post_stage = MagicMock(
            return_value={"stage": "enrich", "raw_output": "{}", "parsed": {}, "parse_error": None}
        )
        with patch.object(calibrate_prompts, "_post_stage", fake_post_stage):
            rc = calibrate_prompts.main(argv)
        return rc, fake_post_stage

    def test_seeded_stage_posts_the_extract_graph(self, tmp_path: Path):
        """The seeded stage hands over the graph from the extract dump and
        nothing else derived from it — no anon facts, no anonymized
        transcript, no client-side status check. The chain produces those
        by running.

        Mutation: reintroduce a client-side relay (posting ``facts`` built
        from the anonymize dump) -> this fails, because the payload would
        carry a fact list instead of the seed graph.
        """
        rc, fake_post_stage = self._run(tmp_path)
        assert rc == 0
        fake_post_stage.assert_called_once()
        args, _kwargs = fake_post_stage.call_args
        assert args[1] == "enrich"
        payload = args[2]
        assert payload["graph"] == self._recorded_run()["parsed"]
        assert payload["transcript"] == "[user] Alex works as an engineer at Acme Corp in Berlin."
        assert "facts" not in payload

    def test_seeded_stage_sends_only_variant_names(self, tmp_path: Path):
        """Prompt variants travel as ``{production: variant}`` names; the
        server resolves them against its own calibration prompt directory,
        so no filesystem path is posted."""
        rc, fake_post_stage = self._run(tmp_path)
        assert rc == 0
        payload = fake_post_stage.call_args[0][2]
        assert "prompts_dir" not in payload
        assert isinstance(payload["prompt_variants"], dict)
        assert all("/" not in name for name in payload["prompt_variants"].values())

    def test_missing_extract_dump_does_not_crash(self, tmp_path: Path):
        """A seed dir with no ``runs.json`` must not crash —
        ``prior_extract`` stays ``None``, the seeded stage is skipped for
        that chunk, and the operator is told what to run.

        Mutation: drop the ``prior_extract is None`` guard -> the payload
        build raises ``AttributeError`` -> this test fails.
        """
        rc, fake_post_stage = self._run(tmp_path, write_extract=False)
        assert rc == 0
        fake_post_stage.assert_not_called()


class TestRespondStage:
    """``respond`` runs one live serving turn — a bare utterance posted to
    ``/calibrate/respond``, not a chunk/graph/turns artifact."""

    _REAL_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "prompts"

    @staticmethod
    def _canned_response(raw_output: str = "Sure, here you go.") -> dict:
        return {
            "stage": "respond",
            "raw_output": raw_output,
            "parsed": {"escalated": False, "exit_via": "personal_probe"},
            "parse_error": None,
            "artifact_dir": "/tmp/paramem-calibration/respond_1",
            "phases": [],
        }

    def test_respond_posts_expected_payload_and_records_run(self, tmp_path: Path):
        """The candidate call posts exactly {text, speaker_id,
        conversation_id, prompt_variants} — no sampling params, since the
        endpoint takes none — and the run is recorded in the index."""
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        fake_post_stage = MagicMock(return_value=self._canned_response())

        argv = [
            "--stages",
            "respond",
            "--utterance",
            "What time is my dentist appointment?",
            "--speaker-id",
            "speaker0",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with patch.object(calibrate_prompts, "_post_stage", fake_post_stage):
            rc = calibrate_prompts.main(argv)

        assert rc == 0
        fake_post_stage.assert_called_once()
        args, _kwargs = fake_post_stage.call_args
        assert args[1] == "respond"
        payload = args[2]
        assert set(payload) == {"text", "speaker_id", "conversation_id", "prompt_variants"}
        assert payload["text"] == "What time is my dentist appointment?"
        assert payload["speaker_id"] == "speaker0"
        assert isinstance(payload["conversation_id"], str) and payload["conversation_id"]
        assert isinstance(payload["prompt_variants"], dict)

        index = json.loads((dump_dir / "runs.json").read_text())
        assert index["respond"] == {"0": self._canned_response()["artifact_dir"]}

        # 07_respond.json is written unconditionally now, even with
        # --baseline none where there is no comparison to report — the blob
        # is no longer built and then discarded.
        out_blob = json.loads((dump_dir / "07_respond.json").read_text())
        assert out_blob["stage"] == "respond"
        assert out_blob["artifact_dir"] == self._canned_response()["artifact_dir"]
        assert "baseline_artifact_dir" not in out_blob
        assert "reply_overlap" not in out_blob

    def test_missing_utterance_exits_with_actionable_message(self, tmp_path: Path):
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        argv = [
            "--stages",
            "respond",
            "--speaker-id",
            "speaker0",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with pytest.raises(SystemExit) as exc_info:
            calibrate_prompts.main(argv)
        message = str(exc_info.value)
        assert "--utterance" in message
        assert "respond" in message

    def test_utterance_without_respond_stage_is_rejected(self, tmp_path: Path):
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        snapshot = tmp_path / "graph_merged_snapshot.json"
        snapshot.write_text(
            json.dumps(
                {"directed": False, "multigraph": False, "graph": {}, "nodes": [], "links": []}
            )
        )
        argv = [
            "--stages",
            "normalize",
            "--utterance",
            "irrelevant here",
            "--snapshot",
            str(snapshot),
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with pytest.raises(SystemExit) as exc_info:
            calibrate_prompts.main(argv)
        assert "--utterance" in str(exc_info.value)

    def test_respond_combined_with_another_stage_is_rejected(self, tmp_path: Path):
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        input_path = tmp_path / "input.txt"
        input_path.write_text("Some document content.")
        argv = [
            "--stages",
            "respond,extract",
            "--utterance",
            "What's on my calendar?",
            "--input",
            str(input_path),
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with pytest.raises(SystemExit) as exc_info:
            calibrate_prompts.main(argv)
        message = str(exc_info.value)
        assert "respond" in message
        assert "combined" in message

    def test_multiple_seeds_with_respond_is_refused(self, tmp_path: Path):
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        argv = [
            "--stages",
            "respond",
            "--utterance",
            "What's the weather?",
            "--speaker-id",
            "speaker0",
            "--seeds",
            "1,2,3",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with pytest.raises(SystemExit) as exc_info:
            calibrate_prompts.main(argv)
        message = str(exc_info.value)
        assert "sampling parameters" in message or "seeds" in message.lower()

    def test_baseline_posted_when_serving_system_variant_exists(self, tmp_path: Path):
        """``--baseline auto`` (the default) runs a baseline call exactly
        when a ``calib_serving_system.txt`` variant exists in
        --prompts-dir. The baseline call shares the candidate's
        ``conversation_id`` — neither call appends to stored history
        (``handle_chat`` performs no session-buffer write), so sharing the
        id is side-effect-free, and a distinct id per leg would otherwise
        guarantee a spurious ``parsed_changed.conversation_id`` diff in
        every ``_phase_diff`` comparison."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "calib_serving_system.txt").write_text("A candidate system prompt.")

        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        calls: list[dict] = []

        def _fake_post_stage(server, stage, payload):
            calls.append(payload)
            reply = "Candidate reply." if len(calls) == 1 else "Baseline reply."
            return self._canned_response(raw_output=reply)

        argv = [
            "--stages",
            "respond",
            "--utterance",
            "What's on my schedule today?",
            "--speaker-id",
            "speaker0",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(prompts_dir),
        ]
        with patch.object(calibrate_prompts, "_post_stage", side_effect=_fake_post_stage):
            rc = calibrate_prompts.main(argv)

        assert rc == 0
        assert len(calls) == 2, "expected a candidate call AND a baseline call"
        candidate_payload, baseline_payload = calls
        assert candidate_payload["prompt_variants"] == {
            "serving_system.txt": "calib_serving_system.txt"
        }
        assert baseline_payload["prompt_variants"] == {}
        assert candidate_payload["conversation_id"] == baseline_payload["conversation_id"]

        out_blob = json.loads((dump_dir / "07_respond.json").read_text())
        assert out_blob["reply_overlap"]["salient_token_jaccard"] < 1.0

    def test_unexercised_variant_warning_printed(self, tmp_path: Path, capsys):
        """When the server reports a non-empty ``variants_unexercised``,
        the driver prints a loud warning naming the gap rather than
        silently accepting a variant that never got a chance to load."""
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        response = self._canned_response()
        response["variants_unexercised"] = ["recall_selection.txt"]
        fake_post_stage = MagicMock(return_value=response)

        argv = [
            "--stages",
            "respond",
            "--utterance",
            "What's my next appointment?",
            "--speaker-id",
            "speaker0",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with patch.object(calibrate_prompts, "_post_stage", fake_post_stage):
            rc = calibrate_prompts.main(argv)

        assert rc == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "recall_selection.txt" in captured.out

    def test_no_warning_when_all_variants_exercised(self, tmp_path: Path, capsys):
        """An empty (or absent) ``variants_unexercised`` prints no warning."""
        dump_dir = tmp_path / "dump"
        dump_dir.mkdir()
        fake_post_stage = MagicMock(return_value=self._canned_response())

        argv = [
            "--stages",
            "respond",
            "--utterance",
            "What's my next appointment?",
            "--speaker-id",
            "speaker0",
            "--dump-dir",
            str(dump_dir),
            "--prompts-dir",
            str(self._REAL_PROMPTS_DIR),
            "--baseline",
            "none",
        ]
        with patch.object(calibrate_prompts, "_post_stage", fake_post_stage):
            rc = calibrate_prompts.main(argv)

        assert rc == 0
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


class TestReplyOverlap:
    """``_reply_overlap`` — a minimal, deterministic content-drift signal
    between a baseline and a candidate serving reply."""

    def test_identical_replies_score_one(self):
        baseline = {"raw_output": "Your dentist appointment is at 3pm on Friday."}
        candidate = {"raw_output": "Your dentist appointment is at 3pm on Friday."}
        result = calibrate_prompts._reply_overlap(baseline, candidate)
        assert result["salient_token_jaccard"] == 1.0
        assert result["length_delta"] == 0

    def test_divergent_replies_score_below_one(self):
        baseline = {"raw_output": "Your dentist appointment is at 3pm on Friday."}
        candidate = {"raw_output": "I don't have any information about that."}
        result = calibrate_prompts._reply_overlap(baseline, candidate)
        assert result["salient_token_jaccard"] < 1.0
        assert result["salient_token_jaccard"] >= 0.0

    def test_both_empty_scores_one(self):
        result = calibrate_prompts._reply_overlap({"raw_output": ""}, {"raw_output": ""})
        assert result["salient_token_jaccard"] == 1.0
        assert result["length_delta"] == 0

    def test_one_empty_scores_zero(self):
        result = calibrate_prompts._reply_overlap(
            {"raw_output": ""}, {"raw_output": "Something concrete happened."}
        )
        assert result["salient_token_jaccard"] == 0.0
        assert result["length_delta"] > 0

    def test_length_delta_reflects_raw_output_byte_counts(self):
        baseline = {"raw_output": "short"}
        candidate = {"raw_output": "a somewhat longer reply here"}
        result = calibrate_prompts._reply_overlap(baseline, candidate)
        assert result["baseline_length"] == len("short")
        assert result["candidate_length"] == len("a somewhat longer reply here")
        assert result["length_delta"] == len("a somewhat longer reply here") - len("short")
