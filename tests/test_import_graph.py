"""Import-graph pins — `paramem.backup` and its lightweight consumers stay
free of torch/peft/transformers at module scope.

Each pin runs the import in a fresh subprocess (``sys.executable -c``) so
imports cached by earlier tests in the same process can never mask a
failure to stay clean. ``paramem.memory.interim_adapter`` imports PEFT ->
torch/transformers at module scope, so this file pins that ``paramem.backup``
never eagerly re-exports from ``paramem.backup.backup`` in a way that would
pull that module — and therefore the ML stack — into every ``paramem.backup.*``
import.
"""

from __future__ import annotations

import subprocess
import sys

_CHECK_SNIPPET = """
import sys
import {module}
heavy = {{"torch", "peft", "transformers"}} & set(sys.modules)
if heavy:
    raise SystemExit(f"heavy modules leaked into sys.modules: {{sorted(heavy)}}")
"""


def _assert_no_heavy_imports(module: str) -> None:
    """Import *module* in a fresh subprocess; fail if torch/peft/transformers load.

    Parameters
    ----------
    module:
        Dotted module path to import (e.g. ``"paramem.backup"``).

    Raises
    ------
    AssertionError
        If the subprocess exits non-zero (either the import itself failed,
        or the heavy-module check inside the snippet raised).
    """
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SNIPPET.format(module=module)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import {module!r} pulled in a heavy module or failed.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


class TestBackupImportStaysLight:
    def test_bare_backup_package_import_is_light(self):
        """``import paramem.backup`` alone must not load torch/peft/transformers.

        ``paramem/backup/__init__.py`` carries no re-export surface — it is
        a docstring only — so importing the bare package must not execute
        ``paramem.backup.backup`` (which imports
        ``paramem.memory.interim_adapter`` -> PEFT at module scope).
        """
        _assert_no_heavy_imports("paramem.backup")

    def test_backup_types_import_is_light(self):
        """``import paramem.backup.types`` must not load torch/peft/transformers.

        ``paramem.backup.types`` is pure-stdlib (dataclasses, enum,
        pathlib) and is imported at module scope by
        ``paramem.server.config`` — it must stay free of the ML stack so
        that config loading alone never forces a GPU-capable environment.
        """
        _assert_no_heavy_imports("paramem.backup.types")


class TestAdaptersImportStaysLight:
    def test_registry_binding_import_is_light(self):
        """``import paramem.adapters.registry_binding`` stays torch/peft-free.

        ``registry_binding`` now imports ``paramem.backup.hashing`` at
        module scope (promoted from a lazy import once ``paramem.backup``
        stopped eagerly pulling in PEFT) — this pin protects that promotion
        from silently regressing back to a heavy import chain.
        """
        _assert_no_heavy_imports("paramem.adapters.registry_binding")

    def test_manifest_import_is_light(self):
        """``import paramem.adapters.manifest`` stays torch/peft-free.

        Same promotion as ``registry_binding``: ``tier_registry_sha256``'s
        ``paramem.backup.hashing`` import moved to module scope.
        """
        _assert_no_heavy_imports("paramem.adapters.manifest")

    def test_slot_import_is_light(self):
        """``import paramem.adapters.slot`` stays torch/peft-free.

        Same promotion as ``registry_binding``/``manifest``:
        ``paramem.adapters.slot`` imports ``plaintext_sha256`` from
        ``paramem.backup.hashing`` at module scope — this pin protects that
        promotion from regressing into a heavy import chain.
        """
        _assert_no_heavy_imports("paramem.adapters.slot")


class TestServerAttentionImportStaysLight:
    def test_attention_import_is_light(self):
        """``import paramem.server.attention`` stays torch/peft-free.

        ``paramem.server.attention`` reaches ``paramem.server.config`` ->
        ``paramem.backup.types`` (pure stdlib) at module scope; since
        ``paramem.backup`` carries no re-export surface, that chain never
        touches ``paramem.backup.backup`` / ``paramem.memory.interim_adapter``
        (the PEFT/torch entry point). This is the module
        ``paramem.server.manifest_status``'s docstring names as staying
        torch/peft-free for the same reason — pinned here so a regression
        in either module's import chain is caught directly.
        """
        _assert_no_heavy_imports("paramem.server.attention")
