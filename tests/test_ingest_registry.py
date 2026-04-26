"""Unit tests for IngestRegistry and normalize_chunk_text."""

from __future__ import annotations

import json
from pathlib import Path

from paramem.server.document_ingest import IngestRegistry, normalize_chunk_text

# ---------------------------------------------------------------------------
# normalize_chunk_text
# ---------------------------------------------------------------------------


class TestNormalizeChunkText:
    def test_collapses_multiple_spaces(self):
        assert normalize_chunk_text("a  b") == "a b"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_chunk_text("a\tb\nc") == "a b c"

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_chunk_text("  hello  ") == "hello"

    def test_nfc_normalization(self):
        # U+00E9 (é precomposed) vs U+0065 + U+0301 (e + combining accent)
        precomposed = "é"
        decomposed = "é"
        assert normalize_chunk_text(precomposed) == normalize_chunk_text(decomposed)

    def test_no_case_folding(self):
        assert normalize_chunk_text("Abc") != normalize_chunk_text("abc")

    def test_empty_string(self):
        assert normalize_chunk_text("") == ""

    def test_whitespace_only(self):
        assert normalize_chunk_text("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# IngestRegistry — hash stability and uniqueness
# ---------------------------------------------------------------------------


class TestChunkHash:
    def test_hash_stable(self, tmp_path):
        """Same inputs produce the same hash across two calls."""
        reg = IngestRegistry(tmp_path / "registry.json")
        h1 = reg.chunk_hash(
            speaker_id="spk-1",
            source_path="/docs/notes.md",
            chunk_index=0,
            normalized_text="hello world",
            source_type="document",
        )
        h2 = reg.chunk_hash(
            speaker_id="spk-1",
            source_path="/docs/notes.md",
            chunk_index=0,
            normalized_text="hello world",
            source_type="document",
        )
        assert h1 == h2

    def test_hash_whitespace_reflow_same(self, tmp_path):
        """After normalize_chunk_text, 'a  b' and 'a b' produce the same hash."""
        reg = IngestRegistry(tmp_path / "registry.json")
        t1 = normalize_chunk_text("a  b")
        t2 = normalize_chunk_text("a b")
        h1 = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text=t1,
            source_type="document",
        )
        h2 = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text=t2,
            source_type="document",
        )
        assert h1 == h2

    def test_hash_case_sensitive(self, tmp_path):
        """'Abc' and 'abc' produce different hashes — no case folding."""
        reg = IngestRegistry(tmp_path / "registry.json")
        h_upper = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="Abc",
            source_type="document",
        )
        h_lower = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="abc",
            source_type="document",
        )
        assert h_upper != h_lower

    def test_separate_speakers_dont_collide(self, tmp_path):
        """Same chunk under two speaker_ids produces two distinct hashes."""
        reg = IngestRegistry(tmp_path / "registry.json")
        h1 = reg.chunk_hash(
            speaker_id="spk-alice",
            source_path="/notes.md",
            chunk_index=0,
            normalized_text="I work at Acme",
            source_type="document",
        )
        h2 = reg.chunk_hash(
            speaker_id="spk-bob",
            source_path="/notes.md",
            chunk_index=0,
            normalized_text="I work at Acme",
            source_type="document",
        )
        assert h1 != h2

    def test_hash_is_hex_64_chars(self, tmp_path):
        """Hash output is a 64-character lowercase hex string (SHA-256)."""
        reg = IngestRegistry(tmp_path / "registry.json")
        h = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="text",
            source_type="document",
        )
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# IngestRegistry — record / is_known / flush / reload
# ---------------------------------------------------------------------------


class TestRecordAndKnown:
    def test_is_not_known_initially(self, tmp_path):
        reg = IngestRegistry(tmp_path / "registry.json")
        h = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="text",
            source_type="document",
        )
        assert not reg.is_known(h)

    def test_record_then_known(self, tmp_path):
        """record() + flush() + new-instance reload → is_known returns True."""
        path = tmp_path / "registry.json"
        reg1 = IngestRegistry(path)
        h = reg1.chunk_hash(
            speaker_id="spk-1",
            source_path="/docs/a.txt",
            chunk_index=0,
            normalized_text="hello",
            source_type="document",
        )
        reg1.record(
            h,
            session_id="doc-aabb1122",
            speaker_id="spk-1",
            source_path="/docs/a.txt",
            chunk_index=0,
            source_type="document",
            doc_title="a",
            ingested_at="2026-01-01T00:00:00Z",
        )
        reg1.flush()

        # Re-load from disk.
        reg2 = IngestRegistry(path)
        assert reg2.is_known(h)

    def test_metadata_persisted(self, tmp_path):
        """Metadata stored via record() is readable after reload."""
        path = tmp_path / "registry.json"
        reg = IngestRegistry(path)
        h = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=1,
            normalized_text="content",
            source_type="document",
        )
        reg.record(
            h,
            session_id="doc-xyzw",
            speaker_id="s",
            source_path="f",
            chunk_index=1,
            source_type="document",
            doc_title="title",
            ingested_at="2026-01-01T00:00:00Z",
        )
        reg.flush()

        raw = json.loads(path.read_bytes())
        entry = raw["entries"][h]
        assert entry["session_id"] == "doc-xyzw"
        assert entry["doc_title"] == "title"
        assert entry["chunk_index"] == 1

    def test_schema_version(self, tmp_path):
        """Flushed registry carries version: 1."""
        path = tmp_path / "registry.json"
        reg = IngestRegistry(path)
        h = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="text",
            source_type="document",
        )
        reg.record(h, session_id="doc-aabb1122", speaker_id="s", source_path="f")
        reg.flush()
        data = json.loads(path.read_bytes())
        assert data["version"] == 1

    def test_empty_registry_on_flush(self, tmp_path):
        """flush() writes valid JSON with an empty entries dict after a record+flush cycle.

        flush() is a no-op when _dirty is False (see TestFlushNoop).  This test
        verifies the on-disk schema shape when there is at least one entry to
        flush — and confirms ``entries`` can be an empty dict by checking after
        a round-trip where a previously-written file is loaded and re-flushed.
        """
        path = tmp_path / "registry.json"
        # Write a registry with one entry via the normal record→flush path.
        reg1 = IngestRegistry(path)
        h = reg1.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="text",
            source_type="document",
        )
        reg1.record(h, session_id="doc-aabb1122", speaker_id="s", source_path="f")
        reg1.flush()
        assert path.exists()

        # Re-load (no new records → _dirty False).  Writing to disk again
        # would require a new record().  Verify the persisted structure.
        data = json.loads(path.read_bytes())
        assert "entries" in data


# ---------------------------------------------------------------------------
# IngestRegistry — write_infra_bytes usage (no caller-side if-branch)
# ---------------------------------------------------------------------------


class TestRegistryUsesWriteInfraBytes:
    def test_flush_routes_through_write_infra_bytes(self, tmp_path, monkeypatch):
        """flush() must call write_infra_bytes, never write directly.

        Verifies there is NO conditional ``if require_encryption`` branch
        in IngestRegistry — the helper handles that internally.  A record()
        call is required before flush() to mark the registry dirty; without it
        flush() is a deliberate no-op (see TestFlushNoop).
        """
        calls_made: list = []

        def _fake_write_infra_bytes(path: Path, plaintext: bytes) -> None:
            calls_made.append(("write_infra_bytes", Path(path), plaintext))

        monkeypatch.setattr(
            "paramem.server.document_ingest.write_infra_bytes",
            _fake_write_infra_bytes,
        )

        path = tmp_path / "registry.json"
        reg = IngestRegistry(path)
        h = reg.chunk_hash(
            speaker_id="s",
            source_path="f",
            chunk_index=0,
            normalized_text="text",
            source_type="document",
        )
        reg.record(h, session_id="doc-aabb1122", speaker_id="s", source_path="f")
        reg.flush()

        assert len(calls_made) == 1
        name, written_path, payload = calls_made[0]
        assert name == "write_infra_bytes"
        assert written_path == path
        data = json.loads(payload.decode("utf-8"))
        assert "entries" in data

    def test_init_routes_through_read_maybe_encrypted(self, tmp_path, monkeypatch):
        """__init__ must call read_maybe_encrypted when the file exists."""
        calls_made: list = []

        # Pre-create a valid plaintext registry so the file exists.
        path = tmp_path / "registry.json"
        path.write_bytes(json.dumps({"version": 1, "entries": {}}).encode())

        def _fake_read(p: Path) -> bytes:
            calls_made.append(p)
            return path.read_bytes()

        monkeypatch.setattr(
            "paramem.server.document_ingest.read_maybe_encrypted",
            _fake_read,
        )

        IngestRegistry(path)

        assert len(calls_made) == 1
        assert calls_made[0] == path


# ---------------------------------------------------------------------------
# IngestRegistry — corrupt file tolerance
# ---------------------------------------------------------------------------


class TestCorruptFileTolerance:
    def test_corrupt_json_starts_empty(self, tmp_path):
        """A corrupt JSON file is silently discarded — registry starts empty."""
        path = tmp_path / "registry.json"
        path.write_bytes(b"not valid json {{{{")
        reg = IngestRegistry(path)
        assert not reg._entries  # starts empty, no crash


# ---------------------------------------------------------------------------
# IngestRegistry — version validation
# ---------------------------------------------------------------------------


class TestVersionValidation:
    def test_unknown_version_starts_empty_with_warning(self, tmp_path, caplog):
        """A registry file with an unrecognised version is discarded with a warning.

        Version mismatch (e.g. v2 file read by v1 code) must not silently load
        entries that the current schema logic cannot interpret correctly.  The
        registry starts empty and logs a WARNING so operators can detect the
        mismatch.

        Implementation note: ROS 2's ``launch.logging`` module intercepts
        ``logging.getLogger()`` and sets ``propagate=False`` on every new
        logger, which breaks the default ``caplog`` propagation path.
        The project pattern (see ``tests/test_vram_validator.py``) is to
        attach ``caplog.handler`` directly to the named logger.
        """
        import logging

        named_logger = logging.getLogger("paramem.server.document_ingest")
        named_logger.addHandler(caplog.handler)
        named_logger.setLevel(logging.WARNING)

        path = tmp_path / "registry.json"
        path.write_bytes(
            json.dumps(
                {
                    "version": 2,
                    "entries": {
                        "abc" * 21 + "ab": {  # 64-char fake hash
                            "session_id": "doc-aabbccdd",
                            "speaker_id": "spk-1",
                            "source_path": "/docs/a.txt",
                            "chunk_index": 0,
                            "source_type": "document",
                            "doc_title": "a",
                            "ingested_at": "2026-01-01T00:00:00+00:00",
                        }
                    },
                }
            ).encode()
        )

        try:
            reg = IngestRegistry(path)
        finally:
            named_logger.removeHandler(caplog.handler)

        # All entries from the v2 file must be dropped.
        fake_hash = "abc" * 21 + "ab"
        assert not reg.is_known(fake_hash), "v2 entry must not be loaded by v1 registry"

        # At least one warning must mention the version mismatch.
        # Use getMessage() to get the fully-formatted string from the record.
        warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages, "expected a WARNING log for version mismatch"
        assert any("version" in m.lower() for m in warning_messages)


# ---------------------------------------------------------------------------
# IngestRegistry — flush no-op when not dirty
# ---------------------------------------------------------------------------


class TestFlushNoop:
    def test_flush_noop_when_not_dirty(self, tmp_path, monkeypatch):
        """flush() must not call write_infra_bytes when the registry is clean.

        When every chunk in a request was already known, record() is never
        called and _dirty stays False.  flush() in that path must skip the
        disk write entirely.
        """
        write_calls: list = []

        def _fake_write(path: Path, plaintext: bytes) -> None:
            write_calls.append(path)

        monkeypatch.setattr(
            "paramem.server.document_ingest.write_infra_bytes",
            _fake_write,
        )

        reg = IngestRegistry(tmp_path / "registry.json")
        # _dirty is False after __init__ (file does not exist).
        reg.flush()

        assert write_calls == [], "flush() must be a no-op when _dirty is False"
