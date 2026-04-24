"""Tests for paramem.backup.hashing."""

from __future__ import annotations

import hashlib

import pytest

from paramem.backup.hashing import (
    content_sha256_bytes,
    content_sha256_path,
)


class TestContentSha256Bytes:
    def test_content_sha256_raw_bytes_empty(self):
        """SHA-256 of empty bytes matches the known digest."""
        expected = hashlib.sha256(b"").hexdigest()
        assert content_sha256_bytes(b"") == expected

    def test_content_sha256_raw_bytes_known_value(self):
        """SHA-256 of a known byte string returns the expected digest."""
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert content_sha256_bytes(data) == expected

    def test_content_sha256_whitespace_sensitive(self):
        """Whitespace changes must produce different hashes (Resolved Decision 29)."""
        a = b"key: value\n"
        b_ = b"key:  value\n"  # extra space
        assert content_sha256_bytes(a) != content_sha256_bytes(b_)

    def test_content_sha256_key_order_sensitive(self):
        """Key-order changes in JSON/YAML produce different hashes (no canonicalization)."""
        a = b'{"a": 1, "b": 2}'
        b_ = b'{"b": 2, "a": 1}'
        assert content_sha256_bytes(a) != content_sha256_bytes(b_)

    def test_content_sha256_returns_lowercase_hex(self):
        """Digest must be a 64-character lowercase hex string."""
        digest = content_sha256_bytes(b"test")
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)  # must be valid hex


class TestContentSha256Path:
    def test_content_sha256_path_matches_bytes(self, tmp_path):
        """Hash of a file path must equal hash of the same bytes in memory."""
        data = b"some content for path test"
        f = tmp_path / "artifact.bin"
        f.write_bytes(data)
        assert content_sha256_path(f) == content_sha256_bytes(data)

    def test_content_sha256_path_missing_file(self, tmp_path):
        """Raises FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            content_sha256_path(tmp_path / "does_not_exist.bin")

    def test_content_sha256_path_large_file_matches_bytes(self, tmp_path):
        """Streaming hash of a large file matches the in-memory hash."""
        data = b"x" * (200 * 1024)  # 200 KiB — forces multiple chunks
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        assert content_sha256_path(f) == content_sha256_bytes(data)
