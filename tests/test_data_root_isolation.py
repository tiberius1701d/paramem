"""The test suite must never address the operator's live data root.

``<project root>/data/ha`` holds the household's real parametric memory —
adapter weights, the SimHash registry, session transcripts, speaker profiles,
backups — age-encrypted on a deployed host. A test that reads it is at best a
spurious failure (the daily passphrase is popped by
``conftest._isolate_paramem_security_env``, so an encrypted read raises) and at
worst a silent read or overwrite of personal data.

The boundary has two halves, and this module pins both:

* **Runtime** — ``conftest._isolate_data_root`` repoints
  :func:`paramem.server.config.default_data_dir` at a per-test tmp tree. That
  accessor is the single declaration of the data root, and every
  :class:`~paramem.server.config.PathsConfig` default plus every
  ``config is None`` fallback in the server resolves through it at call time,
  so the redirect covers the whole process without a per-call-site opt-in.
* **Structural** — nothing outside the declaration and the redirect may bind
  the underlying constant, because an import-time binding
  (``from paramem.server.config import DEFAULT_DATA_DIR``, the shape this
  module replaced) freezes the pre-redirect value and escapes the boundary.

Not covered, deliberately: a config that names its data paths EXPLICITLY keeps
them. ``tests/fixtures/server.yaml`` names the gitignored
``tests/fixtures/sandbox/data/ha`` (asserted below); the operator-local
``configs/server.yaml`` names the live tree, so a test that loads that file
gets live paths back. Those call sites are a separate decision, not something
this module can assert away.
"""

from __future__ import annotations

from pathlib import Path

from paramem.server.config import (
    PathsConfig,
    ServerConfig,
    default_data_dir,
    load_server_config,
)
from paramem.utils.paths import find_project_root

#: The live store, derived independently of the (redirected) config module so
#: the assertions below cannot be satisfied by the redirect moving too.
_REPO_ROOT = find_project_root(Path(__file__).resolve())
_LIVE_DATA_ROOT = _REPO_ROOT / "data" / "ha"

#: Path fields whose values live inside the data tree. ``prompts`` is excluded
#: on purpose — it addresses shipped read-only assets under ``configs/``, not
#: the knowledge store, and must stay pointed at the real repo.
_DATA_TREE_FIELDS = ("data", "sessions", "debug", "telemetry", "calibration")

#: Files allowed to name the private data-root constant: the declaration, the
#: redirect, and this guard. Every other module goes through
#: ``default_data_dir()``.
_SEAM_OWNERS = frozenset(
    {
        "paramem/server/config.py",
        "tests/conftest.py",
        "tests/test_data_root_isolation.py",
    }
)


def _is_inside(path: Path, root: Path) -> bool:
    """Return True when *path* is *root* itself or lives beneath it."""
    resolved = Path(path).resolve()
    return resolved == root.resolve() or root.resolve() in resolved.parents


def _iter_python_files() -> list[Path]:
    """Every tracked-source Python file under ``paramem/`` and ``tests/``."""
    return [
        p
        for root in (_REPO_ROOT / "paramem", _REPO_ROOT / "tests")
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    ]


def test_default_data_dir_is_redirected_away_from_the_live_store():
    """The accessor every fallback resolves through must not be the live root."""
    assert not _is_inside(default_data_dir(), _LIVE_DATA_ROOT), (
        f"default_data_dir() resolves to {default_data_dir()}, inside the operator's "
        f"live store {_LIVE_DATA_ROOT}. conftest._isolate_data_root is not in force — "
        "any test driving a filesystem path from config defaults now reads or writes "
        "real personal data."
    )


def test_default_paths_config_never_addresses_the_live_store():
    """A bare ``PathsConfig()`` — the shape three abstention tests reached the
    live registry through — must land entirely in the redirected tree."""
    paths = PathsConfig()
    for name in _DATA_TREE_FIELDS:
        value = getattr(paths, name)
        assert value.is_absolute(), f"PathsConfig().{name} is relative ({value})"
        assert not _is_inside(value, _LIVE_DATA_ROOT), (
            f"PathsConfig().{name} = {value} is inside the live store {_LIVE_DATA_ROOT}"
        )


def test_default_server_config_derived_paths_never_address_the_live_store():
    """The derived accessors (adapters, registry, key_metadata) follow the
    redirect too — those are the exact paths the memory source reads."""
    config = ServerConfig()
    for value in (
        config.adapter_dir,
        config.registry_path,
        config.key_metadata_path,
        config.paths.registry_dir,
        config.paths.calibration_prompts,
        config.paths.calibration_artifacts,
    ):
        assert not _is_inside(value, _LIVE_DATA_ROOT), (
            f"{value} is inside the live store {_LIVE_DATA_ROOT}"
        )


def test_prompts_default_still_points_at_the_shipped_assets():
    """The redirect covers the KNOWLEDGE store only. ``paths.prompts`` reads
    shipped, read-only templates under ``configs/prompts`` — redirecting it
    would break every prompt-contract test for no privacy gain."""
    prompts = PathsConfig().prompts
    assert prompts.is_absolute()
    assert prompts == _REPO_ROOT / "configs" / "prompts"


def test_test_fixture_config_paths_never_address_the_live_store():
    """``tests/fixtures/server.yaml`` is the sanctioned config for GPU /
    integration / contract tests. Every data-tree path it yields — including
    ``calibration``, which the YAML does not name and which therefore falls
    through to the default — must stay out of the live store."""
    config = load_server_config(_REPO_ROOT / "tests" / "fixtures" / "server.yaml")
    for name in _DATA_TREE_FIELDS:
        value = getattr(config.paths, name)
        assert not _is_inside(value, _LIVE_DATA_ROOT), (
            f"tests/fixtures/server.yaml resolves paths.{name} to {value}, inside the "
            f"operator's live store {_LIVE_DATA_ROOT}"
        )


def test_only_the_seam_owners_bind_the_data_root_constant():
    """Nothing may name ``_DATA_ROOT`` outside its declaration and the redirect.

    An import-time binding of the constant (rather than a call to
    ``default_data_dir()``) freezes the pre-redirect value in the importing
    module, which is precisely how a single module can escape the boundary
    while the rest of the suite looks isolated.
    """
    violations: list[str] = []
    for py in _iter_python_files():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        if rel in _SEAM_OWNERS:
            continue
        for line_num, line in enumerate(py.read_text().splitlines(), 1):
            if "_DATA_ROOT" in line:
                violations.append(f"{rel}:{line_num}: {line.strip()}")

    assert not violations, (
        "The data root must be read through paramem.server.config.default_data_dir() "
        "so the value resolves at call time and tests/conftest.py can redirect the "
        "whole process away from the operator's live store. Binding the constant "
        "escapes that redirect.\n\nViolations:\n  " + "\n  ".join(violations)
    )
