"""Tests for _atomic_write_file() in paramem.backup.backup.

``restore_bundle()``'s round trip (find_live_slot resolution, restored
adapter slot naming/content, corrupt-bundle detection, orphan interim/
main-tier pruning, simulate-only tier ordering) is covered in
``tests/backup/test_bundle_boot_binding.py``, not here. What remains here is
the generic ``_atomic_write_file`` stale-temp robustness unit, which does not
depend on manifest shape or slot layout.
"""

from __future__ import annotations

from paramem.backup.backup import _atomic_write_file

# ---------------------------------------------------------------------------
# _atomic_write_file — stale-temp robustness
# ---------------------------------------------------------------------------


class TestAtomicWriteFile:
    """Unit tests for _atomic_write_file (stale-temp robustness).

    ``_atomic_write_file`` writes to a uniquely-named temp file via
    ``tempfile.mkstemp`` and renames it into place, so a leftover temp file
    at *dst*'s destination from an earlier interrupted write never blocks a
    subsequent write.
    """

    def test_happy_path_writes_content(self, tmp_path) -> None:
        """_atomic_write_file writes the expected bytes to dst."""
        dst = tmp_path / "target.json"
        _atomic_write_file(b'{"ok": true}', dst)
        assert dst.read_bytes() == b'{"ok": true}'

    def test_parent_created_automatically(self, tmp_path) -> None:
        """dst.parent is created if absent."""
        dst = tmp_path / "subdir" / "target.json"
        _atomic_write_file(b"data", dst)
        assert dst.read_bytes() == b"data"

    def test_stale_temp_does_not_wedge(self, tmp_path) -> None:
        """An unrelated file at a fixed ``.restore-pending`` suffix path does not block a write.

        ``_atomic_write_file`` mints a unique temp name per call via
        ``tempfile.mkstemp``, so a file already present at that fixed suffix
        path is simply left alone.
        """
        dst = tmp_path / "target.json"
        # Plant a file at the fixed-suffix path _atomic_write_file does not use.
        stale_temp = tmp_path / "target.json.restore-pending"
        stale_temp.write_bytes(b"stale-content")

        # This must NOT raise even though a file exists at that fixed-suffix path.
        _atomic_write_file(b"fresh-content", dst)

        assert dst.read_bytes() == b"fresh-content"
        # The file at the fixed-suffix path is untouched (mkstemp uses a unique name).
        assert stale_temp.exists()

    def test_mode_0o600_applied(self, tmp_path) -> None:
        """dst has mode 0o600 after write (no world-readable plaintext window)."""
        import stat

        dst = tmp_path / "secret.json"
        _atomic_write_file(b"secret", dst)
        file_mode = stat.S_IMODE(dst.stat().st_mode)
        assert file_mode == 0o600, f"Expected mode 0o600, got 0o{file_mode:o}"

    def test_overwrites_existing_dst(self, tmp_path) -> None:
        """_atomic_write_file atomically replaces an existing dst file."""
        dst = tmp_path / "target.json"
        dst.write_bytes(b"old-content")
        _atomic_write_file(b"new-content", dst)
        assert dst.read_bytes() == b"new-content"
