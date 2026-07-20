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
    "filtered": [],
    "filter_prompt_used": "normalize_filter.txt",
    "raw_output": "[]",
}


class TestNormalizeStageNoNameError:
    """--stages normalize must not raise NameError when chunks is empty."""

    def test_normalize_writes_output_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Invoke main(['--stages','normalize',...]) with _post_stage mocked.

        Asserts:
        - No NameError (params_base is bound before the chunk loop)
        - 05_normalize.json is written to the dump directory
        - The output file contains the expected 'stage' key
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

        out_file = dump_dir / "05_normalize.json"
        assert out_file.exists(), "05_normalize.json was not written"

        blob = json.loads(out_file.read_text())
        assert blob["stage"] == "normalize", f"Unexpected stage in output: {blob.get('stage')!r}"
        assert blob["snapshot_path"] == str(snapshot)

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
    ``/calibrate/extract`` — production's ``anonymize_for_cloud`` always
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
    """``--seed-from --stages enrich`` reads prior-stage dumps off disk
    instead of re-running ``extract``/``anonymize``.  All three regression
    cases share one real input file + real prompts dir; only the seed
    dump contents differ.
    """

    _REAL_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "configs" / "prompts"

    @staticmethod
    def _write_input(tmp_path: Path) -> Path:
        input_path = tmp_path / "input.txt"
        input_path.write_text("Alex works as an engineer at Acme Corp in Berlin.")
        return input_path

    @staticmethod
    def _extract_wrapper_blob() -> dict:
        """01_extract_chunk_0.json shape: a WRAPPER around candidate_runs."""
        return {
            "stage": "extract",
            "chunk_index": 0,
            "candidate_runs": [
                {
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
            ],
        }

    def _run(
        self, tmp_path: Path, *, anonymize_blob: dict, write_extract: bool = True
    ) -> tuple[int, MagicMock]:
        seed_from = tmp_path / "seed_from"
        seed_from.mkdir()
        if write_extract:
            (seed_from / "01_extract_chunk_0.json").write_text(
                json.dumps(self._extract_wrapper_blob())
            )
        (seed_from / "02_anonymize_chunk_0.json").write_text(json.dumps(anonymize_blob))

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

    def test_anon_facts_pass_through_to_enrich_payload(self, tmp_path: Path):
        """``/calibrate/anonymize``'s ``anon_facts`` — now fully assembled
        SERVER-side by ``anonymize_for_cloud`` — is fed STRAIGHT THROUGH
        to ``/calibrate/enrich``; there is no client-side re-derivation
        from ``01_extract_chunk_0.json``'s graph any more (that whole
        primitive-import + table-build + fact-build block was deleted).

        Mutation: reintroduce client-side fact building from
        ``prior_extract``'s graph -> this test fails (the facts payload
        would differ from — or ignore — ``anon_facts``).
        """
        rc, fake_post_stage = self._run(
            tmp_path,
            anonymize_blob={
                "parsed": {
                    "status": "ok",
                    "anonymized_transcript": "[user] Person_1 works as an engineer.",
                    "forward": {"Alex": "Person_1"},
                    "reverse": {"Person_1": "Alex"},
                    "anon_facts": [
                        {
                            "subject": "speaker0",
                            "predicate": "works_at",
                            "object": "Acme Corp",
                            "relation_type": "factual",
                            "confidence": 0.9,
                        }
                    ],
                    "norm_stats": {"inverted": 0, "dropped": 0},
                }
            },
        )
        assert rc == 0
        fake_post_stage.assert_called_once()
        args, _kwargs = fake_post_stage.call_args
        assert args[1] == "enrich"
        payload = args[2]
        assert payload["facts"] == [
            {
                "subject": "speaker0",
                "predicate": "works_at",
                "object": "Acme Corp",
                "relation_type": "factual",
                "confidence": 0.9,
            }
        ]
        assert payload["transcript"] == "[user] Person_1 works as an engineer."

    def test_missing_extract_dump_does_not_crash(self, tmp_path: Path):
        """A seed dir with ``02_anonymize_chunk_0.json`` but no
        ``01_extract_chunk_0.json`` must not crash — ``prior_extract``
        stays ``None`` and the enrich stage is skipped for that chunk.

        Mutation: drop the ``prior_extract is not None`` guard on the
        enrich stage -> ``AttributeError: 'NoneType' object has no
        attribute 'get'`` -> this test fails.
        """
        rc, fake_post_stage = self._run(
            tmp_path,
            anonymize_blob={
                "parsed": {
                    "forward": {"Alex": "speaker0"},
                    "anonymized_transcript": "[user] Person_1 works as an engineer.",
                }
            },
            write_extract=False,
        )
        assert rc == 0
        fake_post_stage.assert_not_called()

    def test_status_failed_aborts_without_a_cloud_call(self, tmp_path: Path):
        """``status == "failed"`` (anonymizer parse failure) must abort
        the chunk's enrich stage — no cloud call — matching production's
        fail-closed abort-on-``"failed"`` in ``_sota_pipeline``.

        Mutation: drop the ``status == "failed"`` gate -> the failure is
        silently treated as "proceed" and ``/calibrate/enrich`` (the
        cloud call) is posted to anyway -> this test fails.
        """
        rc, fake_post_stage = self._run(
            tmp_path,
            anonymize_blob={
                "parsed": {"status": "failed", "anon_facts": [], "anonymized_transcript": ""}
            },
        )
        assert rc == 0
        fake_post_stage.assert_not_called()

    def test_missing_anonymized_transcript_aborts_without_a_cloud_call(self, tmp_path: Path):
        """A ``status="ok"`` response with a missing/empty
        ``anonymized_transcript`` is ALSO fail-closed — the model never
        authored a safe transcript to send to the cloud, so the chunk's
        enrich stage must abort with no cloud call, exactly like the
        ``status == "failed"`` case.

        Mutation: gate only on ``status == "failed"`` (drop the
        ``anonymized_transcript`` check) -> ``/calibrate/enrich`` is
        posted to anyway -> this test fails.
        """
        rc, fake_post_stage = self._run(
            tmp_path,
            anonymize_blob={
                "parsed": {"status": "ok", "anon_facts": []}
            },  # no anonymized_transcript
        )
        assert rc == 0
        fake_post_stage.assert_not_called()
