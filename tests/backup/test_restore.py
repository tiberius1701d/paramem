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

    The prior implementation used a fixed ``.restore-pending`` suffix with
    ``O_CREAT | O_EXCL``: a crash between create and rename left a stale temp
    that wedged the next restore with ``FileExistsError``.  The fix uses
    ``tempfile.mkstemp`` which generates a unique name, so stale temps from a
    prior crash never block a subsequent restore.
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
        """A stale .restore-pending temp left by a prior crash must not cause FileExistsError.

        The fixed suffix + O_EXCL used to raise FileExistsError if the
        same-named temp already existed; mkstemp always picks a unique name
        so the stale file is ignored.
        """
        dst = tmp_path / "target.json"
        # Plant a stale temp file with the old fixed-suffix naming scheme.
        stale_temp = tmp_path / "target.json.restore-pending"
        stale_temp.write_bytes(b"stale-content")

        # This must NOT raise even though a stale temp exists at the old path.
        _atomic_write_file(b"fresh-content", dst)

        assert dst.read_bytes() == b"fresh-content"
        # The stale temp is untouched (mkstemp uses a unique name; the old file is orphaned).
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
