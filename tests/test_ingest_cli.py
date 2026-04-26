"""Tests for scripts/ingest_docs.py.

Uses direct ``main()`` invocation with monkeypatched HTTP helpers so the tests
do not require a running server.  The ``post_json`` / ``get_json`` helpers
from ``paramem.cli.http_client`` are patched at the point of use inside the
scripts module.

Pattern: import the module with a sys.path insert (same as
tests/migrate/test_outputs_to_slot_dirs.py) and call ``main()`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so scripts/ is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paramem.cli.http_client import ServerHTTPError, ServerUnreachable  # noqa: E402  # isort: skip
from scripts.ingest_docs import main  # noqa: E402  # isort: skip


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_KNOWN_SPEAKER_ID = "abc12345"
_NOTE_TXT_CONTENT = "I work at Acme Corp.\n\nI graduated from TU Berlin in 2015."


def _make_txt(tmp_path: Path, content: str = _NOTE_TXT_CONTENT) -> Path:
    """Write a temporary .txt file and return its path."""
    p = tmp_path / "note.txt"
    p.write_text(content, encoding="utf-8")
    return p


def _successful_ingest_response(queued_count: int = 1) -> dict:
    """Build a minimal successful IngestSessionsResponse-shaped dict."""
    return {
        "queued": [f"doc-{i:08x}" for i in range(queued_count)],
        "total_chunks": queued_count,
        "registry_skipped": 0,
        "rejected_unknown_speaker": False,
        "rejected_no_speaker_id": False,
    }


def _status_response_with_speakers(speaker_id: str = _KNOWN_SPEAKER_ID) -> dict:
    """Build a minimal /status response with one enrolled speaker."""
    return {
        "speakers": [
            {
                "id": speaker_id,
                "name": "Alice",
                "embeddings": 3,
                "preferred_language": "en",
                "enroll_method": "voice",
            }
        ]
    }


# ---------------------------------------------------------------------------
# Happy path — non-interactive
# ---------------------------------------------------------------------------


class TestCliHappyPathNonInteractive:
    def test_exit_code_zero(self, tmp_path, monkeypatch, capsys):
        """Non-interactive run against a mocked server returns exit code 0."""
        path = _make_txt(tmp_path)

        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: _successful_ingest_response(1),
        )

        rc = main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert rc == 0

    def test_queued_count_printed(self, tmp_path, monkeypatch, capsys):
        """The number of queued sessions is printed to stdout."""
        path = _make_txt(tmp_path)

        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: _successful_ingest_response(1),
        )

        main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        out = capsys.readouterr().out
        assert "queued=1" in out

    def test_all_session_ids_printed(self, tmp_path, monkeypatch, capsys):
        """Every queued session id appears in stdout, not just the first.

        Operators running with --no-action need the full list to drive a
        later /ingest-sessions/cancel call.  Regression guard for the bug
        where only queued[0] was printed.
        """
        path = _make_txt(tmp_path)

        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: _successful_ingest_response(3),
        )

        main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        out = capsys.readouterr().out
        assert "doc-00000000" in out
        assert "doc-00000001" in out
        assert "doc-00000002" in out

    def test_registry_skipped_reported(self, tmp_path, monkeypatch, capsys):
        """registry_skipped count is reported in the output."""
        path = _make_txt(tmp_path)

        def mock_post(url, body=None, **kw):
            resp = _successful_ingest_response(0)
            resp["registry_skipped"] = 1
            resp["total_chunks"] = 1
            return resp

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        rc = main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "registry_skipped=1" in out

    def test_no_action_skips_post_ingest_prompt(self, tmp_path, monkeypatch, capsys):
        """--no-action exits after posting without any further prompts."""
        path = _make_txt(tmp_path)

        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: _successful_ingest_response(1),
        )

        # If input() were called it would raise EOFError; confirm it isn't.
        rc = main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert rc == 0

    def test_first_session_id_printed(self, tmp_path, monkeypatch, capsys):
        """The first session id from the response is printed."""
        path = _make_txt(tmp_path)

        def mock_post(url, body=None, **kw):
            return {
                "queued": ["doc-cafebabe"],
                "total_chunks": 1,
                "registry_skipped": 0,
                "rejected_unknown_speaker": False,
                "rejected_no_speaker_id": False,
            }

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        out = capsys.readouterr().out
        assert "doc-cafebabe" in out


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestCliUnknownSpeaker:
    def test_unknown_speaker_exits_nonzero(self, tmp_path, monkeypatch):
        """Server 404 for unknown speaker → CLI exits with non-zero code."""
        path = _make_txt(tmp_path)

        def mock_post(url, body=None, **kw):
            raise ServerHTTPError(404, url, '{"detail": "not found"}')

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        with pytest.raises(SystemExit) as exc_info:
            main([str(path), "--speaker", "unknown-id", "--non-interactive", "--no-action"])
        assert exc_info.value.code != 0

    def test_unknown_speaker_error_message(self, tmp_path, monkeypatch, capsys):
        """A 404 error prints a helpful message mentioning speaker not found."""
        path = _make_txt(tmp_path)

        def mock_post(url, body=None, **kw):
            raise ServerHTTPError(404, url, '{"detail": "not found"}')

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        with pytest.raises(SystemExit):
            main([str(path), "--speaker", "unknown-id", "--non-interactive", "--no-action"])
        err = capsys.readouterr().err
        assert "speaker" in err.lower() or "not found" in err.lower()


class TestCliUnsupportedFormat:
    def test_docx_exits_code_2(self, tmp_path, monkeypatch):
        """A .docx file causes exit code 2 with a friendly message."""
        path = tmp_path / "resume.docx"
        path.write_bytes(b"PK fake content")

        with pytest.raises(SystemExit) as exc_info:
            main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert exc_info.value.code == 2

    def test_docx_nothing_posted(self, tmp_path, monkeypatch):
        """post_json is never called for an unsupported format."""
        path = tmp_path / "resume.docx"
        path.write_bytes(b"PK fake content")

        post_called = []
        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: post_called.append(url) or {},
        )

        with pytest.raises(SystemExit):
            main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert post_called == []

    def test_unsupported_error_message(self, tmp_path, capsys):
        """Unsupported format error message mentions the format."""
        path = tmp_path / "resume.docx"
        path.write_bytes(b"PK fake content")

        with pytest.raises(SystemExit):
            main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        err = capsys.readouterr().err
        assert "ERROR" in err


class TestCliEmptyFile:
    def test_empty_txt_exits_code_2(self, tmp_path):
        """An empty .txt file causes exit code 2."""
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert exc_info.value.code == 2

    def test_empty_txt_nothing_posted(self, tmp_path, monkeypatch):
        """post_json is never called for an empty file."""
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")

        post_called = []
        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: post_called.append(url) or {},
        )

        with pytest.raises(SystemExit):
            main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert post_called == []


class TestCliNonExistentFile:
    def test_missing_file_exits_code_2(self, tmp_path):
        """A file that does not exist causes exit code 2."""
        path = tmp_path / "no_such_file.txt"

        rc = main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert rc == 2


class TestCliMissingNonInteractiveSpeaker:
    def test_non_interactive_without_speaker_exits_2(self, tmp_path):
        """--non-interactive without --speaker exits with code 2 (usage/file error)."""
        path = _make_txt(tmp_path)

        rc = main([str(path), "--non-interactive", "--no-action"])
        assert rc == 2


# ---------------------------------------------------------------------------
# Server unreachable
# ---------------------------------------------------------------------------


class TestCliServerUnreachable:
    def test_unreachable_server_exits_nonzero(self, tmp_path, monkeypatch):
        """ServerUnreachable during POST → non-zero exit."""
        path = _make_txt(tmp_path)

        def mock_post(url, body=None, **kw):
            raise ServerUnreachable("Connection refused")

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        with pytest.raises(SystemExit) as exc_info:
            main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Payload shape sent to server
# ---------------------------------------------------------------------------


class TestCliPayloadShape:
    def test_payload_matches_ingest_sessions_request(self, tmp_path, monkeypatch):
        """The body sent to POST /ingest-sessions matches IngestSessionsRequest schema."""
        path = _make_txt(tmp_path)
        captured_body = {}

        def mock_post(url, body=None, **kw):
            if body is not None:
                captured_body.update(body)
            return _successful_ingest_response(1)

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--non-interactive", "--no-action"])

        assert captured_body["speaker_id"] == _KNOWN_SPEAKER_ID
        assert "sessions" in captured_body
        assert len(captured_body["sessions"]) >= 1

        session = captured_body["sessions"][0]
        assert "source" in session
        assert "chunk" in session
        assert "chunk_index" in session
        assert session["source_type"] == "document"
        assert "doc_title" in session

    def test_payload_url_correct(self, tmp_path, monkeypatch):
        """POST is sent to /ingest-sessions on the configured server."""
        path = _make_txt(tmp_path)
        captured_url = []

        def mock_post(url, body=None, **kw):
            captured_url.append(url)
            return _successful_ingest_response(1)

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        main(
            [
                str(path),
                "--speaker",
                _KNOWN_SPEAKER_ID,
                "--non-interactive",
                "--no-action",
                "--server",
                "http://testserver:9999",
            ]
        )

        assert len(captured_url) == 1
        assert captured_url[0] == "http://testserver:9999/ingest-sessions"


# ---------------------------------------------------------------------------
# Interactive speaker picker (stdin-piped)
# ---------------------------------------------------------------------------


class TestCliInteractiveSpeakerPicker:
    def test_picker_lists_speakers_and_selects(self, tmp_path, monkeypatch, capsys):
        """Interactive flow: fetches speakers from /status, user selects #1."""
        path = _make_txt(tmp_path)

        def mock_get(url, **kw):
            return _status_response_with_speakers(_KNOWN_SPEAKER_ID)

        monkeypatch.setattr("scripts.ingest_docs.get_json", mock_get)

        def mock_post(url, body=None, **kw):
            return _successful_ingest_response(1)

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        # Pipe "1" as the speaker selection, then "S" for Schedule.
        inputs = iter(["1", "S"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        rc = main([str(path)])
        assert rc == 0

        captured_body = {}

        def mock_post2(url, body=None, **kw):
            if body is not None:
                captured_body.update(body)
            return _successful_ingest_response(1)

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post2)

        inputs2 = iter(["1", "S"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs2))

        rc2 = main([str(path)])
        assert rc2 == 0

    def test_no_action_skips_all_prompts_in_interactive_mode(self, tmp_path, monkeypatch):
        """--no-action skips BOTH speaker selection and post-ingest prompt when interactive."""
        path = _make_txt(tmp_path)

        # --speaker provided (but not --non-interactive) + --no-action
        # should not call input() at all.
        monkeypatch.setattr(
            "scripts.ingest_docs.post_json",
            lambda url, body=None, **kw: _successful_ingest_response(1),
        )
        monkeypatch.setattr(
            "scripts.ingest_docs.get_json", lambda url, **kw: _status_response_with_speakers()
        )

        input_called = []
        monkeypatch.setattr("builtins.input", lambda prompt="": input_called.append(prompt) or "1")

        rc = main([str(path), "--speaker", _KNOWN_SPEAKER_ID, "--no-action"])
        assert rc == 0
        # --no-action must suppress the [N/S/C] prompt; speaker was supplied so picker not called.
        nsc_prompts = [p for p in input_called if "N/S/C" in p]
        assert nsc_prompts == []


# ---------------------------------------------------------------------------
# Post-ingest action: Cancel path
# ---------------------------------------------------------------------------


class TestCliCancelAction:
    def test_cancel_posts_to_cancel_endpoint(self, tmp_path, monkeypatch):
        """Choosing [C] in the post-ingest prompt posts to /ingest-sessions/cancel.

        Uses interactive mode (no --non-interactive flag) so the [N/S/C] prompt
        is shown after queuing.  get_json is mocked to return a speaker list.
        """
        path = _make_txt(tmp_path)

        cancel_called = []

        def mock_post(url, body=None, **kw):
            if "cancel" in url:
                cancel_called.append(body)
                return {"cancelled": body["session_ids"], "not_found": []}
            return _successful_ingest_response(1)

        monkeypatch.setattr("scripts.ingest_docs.post_json", mock_post)

        # get_json is called for the speaker picker; --speaker bypasses it but
        # we still need it for status fetch if the code path calls it.
        monkeypatch.setattr(
            "scripts.ingest_docs.get_json",
            lambda url, **kw: _status_response_with_speakers(_KNOWN_SPEAKER_ID),
        )

        # Supply "C" as the [N/S/C] choice.  --speaker is provided so the picker
        # is bypassed; --no-action is NOT set so the prompt fires.
        inputs = iter(["C"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        rc = main([str(path), "--speaker", _KNOWN_SPEAKER_ID])
        assert rc == 0
        assert len(cancel_called) == 1
        assert "session_ids" in cancel_called[0]
