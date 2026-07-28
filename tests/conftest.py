"""Shared pytest configuration and fixtures."""

import os
import subprocess
import sys
import sysconfig

# --- Environment isolation gate (must run before any project import) --------
#
# FIX, if this gate fires:
#
#     conda activate paramem && unset PYTHONPATH && pytest tests/
#
# or make the environment clear it on every activation, once:
#
#     conda env config vars set PYTHONPATH= -n paramem
#
# Activating a conda env selects the interpreter and its site-packages; it
# does NOT sanitize PYTHONPATH, and CPython puts PYTHONPATH on sys.path AHEAD
# of the env's own site-packages.  A foreign tree there changes behaviour that
# nothing asserts on directly: ROS's ``launch.logging`` calls
# ``logging.setLoggerClass()``, so every logger built afterwards defaults to
# ``propagate=False``, records never reach the root logger where pytest's
# caplog listens, and log-assertion tests pass locally while CI reports them
# red.  The server does not share this exposure — systemd user services do not
# inherit the interactive shell.
_ENV_SITE_PACKAGES = sysconfig.get_paths()["purelib"]
_FOREIGN_SITE_PACKAGES = [
    entry
    for entry in sys.path
    if entry.endswith("site-packages") and not entry.startswith(sys.prefix)
]
if _FOREIGN_SITE_PACKAGES:
    raise RuntimeError(
        "pytest is not running in an isolated environment.\n"
        "Foreign site-packages on sys.path, ahead of this environment's own:\n  "
        + "\n  ".join(_FOREIGN_SITE_PACKAGES)
        + f"\nThis environment: {_ENV_SITE_PACKAGES}\n\n"
        "Fix:       conda activate paramem && unset PYTHONPATH && pytest tests/\n"
        "Permanent: conda env config vars set PYTHONPATH= -n paramem"
    )

# --- WSL2 / CUDA isolation gate (must run before any torch import) ----------
#
# On WSL2, ``import torch`` lazily loads ``libcuda.so`` and the first call to
# ``torch.cuda.is_available()`` opens ``/dev/dxg`` and allocates a cuBLAS
# workspace (~280 MiB) on the GPU.  Pytest collection imports every test
# module, and any decorator argument evaluated at collection time (e.g. a
# bare ``pytest.mark.skipif(not torch.cuda.is_available(), ...)``) will fire
# CUDA init even when ``-m "not gpu"`` would deselect the test.  That blocks
# parallel ML workloads on the same card.
#
# Default-off CUDA: set ``CUDA_VISIBLE_DEVICES=""`` before any test module
# imports torch.  ``torch.cuda.is_available()`` then returns ``False``
# without driver initialization, no ``/dev/dxg`` open, no workspace
# allocation.  GPU-marked tests opt back in by passing ``--gpu`` to pytest
# (this conftest re-exposes the device for those runs) or by being explicitly
# selected via ``-m gpu``.


def _gpu_explicitly_requested(argv: list[str]) -> bool:
    """True when the operator wants real CUDA exposed to this pytest run.

    Triggers: ``--gpu`` flag (already understood by ``pytest_addoption`` below),
    or ``-m`` selector that includes "gpu" without "not gpu".
    """
    if "--gpu" in argv:
        return True
    if "-m" in argv:
        idx = argv.index("-m")
        if idx + 1 < len(argv):
            expr = argv[idx + 1]
            return "gpu" in expr and "not gpu" not in expr
    return False


if not _gpu_explicitly_requested(sys.argv):
    # ``setdefault`` so an operator who exports ``CUDA_VISIBLE_DEVICES``
    # explicitly (e.g. ``CUDA_VISIBLE_DEVICES=0 pytest -m gpu``) is not
    # clobbered.  Setting to "" disables CUDA driver init without unloading
    # libcuda (the shared-lib mapping is unavoidable; the allocation/handle
    # acquisition is what actually contends).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # ``CUDA_VISIBLE_DEVICES=""`` hides the CUDA device but not the GPU
    # thermal sensor — wait_for_cooldown reads temperature via nvidia-smi,
    # which ignores CUDA visibility. Disable the cooldown gate outright so a
    # non-gpu run never blocks on (or even queries) the real GPU sensor.
    os.environ.setdefault("PARAMEM_COOLDOWN_DISABLED", "1")

# ``_api_token`` in app.py is captured at import time (module-level
# ``load_token_from_env()``).  Some test modules (e.g.
# ``test_dataset_probe_commit.py``) import experiment scripts that call
# ``load_dotenv`` at module level during pytest collection, re-seeding
# ``PARAMEM_API_TOKEN`` from ``.env`` AFTER conftest's module-level code but
# BEFORE app.py is first imported.  This causes BearerTokenMiddleware to be
# constructed with a real token and then 401 every test client that doesn't
# include it.
#
# Fix: pop the env var AND force-import app.py here so the middleware is
# constructed while the token is absent.  Python caches the module in
# ``sys.modules``; the later ``load_dotenv`` call sets the env var but the
# already-built middleware retains ``self._token = ""``.
os.environ.pop("PARAMEM_API_TOKEN", None)
# Prevent resolve_token() from reading the real repo .env or ~/.config/paramem/
# secrets during tests.  Auth-specific tests that need file-based resolution
# opt back in by passing allow_files=True against a temp tree.
os.environ["PARAMEM_CLI_NO_TOKEN_FILES"] = "1"

# ---------------------------------------------------------------------------

import pytest  # noqa: E402,I001  (must follow the CUDA gate above)
import paramem.server.app as _app_preload  # noqa: E402,F401  # force pre-collection import


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="Run GPU integration tests")
    parser.addoption(
        "--recall",
        action="store_true",
        default=False,
        help="Run memory recall tests (train ~30 epochs + probe; requires --gpu)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-deselect ``@pytest.mark.gpu`` tests unless ``--gpu`` or ``-m gpu``.

    Pairs with the CUDA isolation gate at the top of this file: by default,
    ``CUDA_VISIBLE_DEVICES=""`` is set and gpu-marked tests would crash on
    the first ``device='cuda'`` allocation.  Auto-skipping them keeps
    ``pytest tests/`` (no flags) usable while the gate is active.

    Operator opts in via:
      * ``pytest --gpu``      — runs every test including gpu-marked.
      * ``pytest -m gpu``     — runs only gpu-marked tests (also re-exposes CUDA).
      * default invocation    — every gpu-marked test is skipped.

    When ``-m`` is supplied explicitly, pytest's own marker filter takes
    precedence and we leave the items list alone (so ``-m "not gpu"`` and
    ``-m gpu`` continue to do exactly what the operator asked).
    """
    if config.getoption("--gpu"):
        return
    if config.getoption("-m"):
        # Operator passed an explicit marker expression — respect it.
        return
    skip_gpu = pytest.mark.skip(
        reason="gpu-marked test skipped by default; pass --gpu or -m gpu to run"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def gpu_base_model():
    """Load the base model exactly once for the entire GPU test session.

    Loads Mistral 7B (pinned via ``tests/fixtures/server.yaml``) on first
    request by any ``@pytest.mark.gpu`` test, keeps it alive for the full
    session, and on teardown unloads via :func:`safe_empty_cache` to release
    cuBLAS workspaces alongside the PyTorch allocator pool.

    Lazy CUDA init: all heavy imports (torch, transformers, PEFT) are
    deferred into the fixture body so that collection-time imports do not
    initialise the CUDA driver when ``CUDA_VISIBLE_DEVICES=""`` is active
    (the default when ``--gpu`` is absent).

    Yields:
        tuple[PreTrainedModel, PreTrainedTokenizer]: The base model and its
        tokenizer. The model is a plain ``PreTrainedModel`` on GPU; adapter
        state accumulated during a test module is cleaned up by that module's
        own teardown fixtures (see ``test_integration_gpu.py``).
    """
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

    from paramem.models.loader import load_base_model
    from paramem.server.config import load_server_config
    from paramem.utils.vram_guard import safe_empty_cache

    cfg = load_server_config("tests/fixtures/server.yaml")
    model, tokenizer = load_base_model(cfg.model_config)
    yield model, tokenizer

    # Teardown: release GPU memory using safe_empty_cache so cuBLAS
    # workspaces are freed alongside the PyTorch allocator pool.
    del model
    safe_empty_cache()


@pytest.fixture(autouse=True)
def _isolate_paramem_security_env(monkeypatch):
    """Pop security-relevant env vars for every test.

    python-dotenv's pytest plugin auto-loads .env at session start, which can
    seed ``PARAMEM_DAILY_PASSPHRASE`` into every test's environment when the
    live deployment has a daily passphrase wired up. Combined with a real
    ``~/.config/paramem/daily_key.age`` on disk, that makes
    :func:`paramem.backup.encryption.write_infra_bytes` pick the age branch
    in tests that implicitly assume "no key loaded" and expect plaintext.

    Pop the daily passphrase env var by default. Tests that need it set
    should use ``monkeypatch.setenv`` explicitly — the auto-rollback keeps
    other tests isolated.
    """
    monkeypatch.delenv("PARAMEM_DAILY_PASSPHRASE", raising=False)


@pytest.fixture(autouse=True)
def _isolate_data_root(monkeypatch, tmp_path):
    """Repoint the default data root at a per-test tmp tree.

    ``data/ha`` is the operator's LIVE personal-knowledge store: adapters,
    the SimHash registry, session transcripts, speaker profiles, backups.
    Every one of those files is age-encrypted on a deployed host, and
    ``_isolate_paramem_security_env`` above deliberately removes the daily
    passphrase — so a test that reaches the live tree either raises a decrypt
    error (how this defect was found: three ``test_abstention.py`` cases drove
    ``handle_chat`` with a default-constructed ``ServerConfig`` and hit
    ``data/ha/adapters/episodic/indexed_key_registry.json``) or, worse,
    silently reads or overwrites the household's real memory.

    :func:`paramem.server.config.default_data_dir` is the ONE place that root
    is declared, and every default in :class:`~paramem.server.config.PathsConfig`
    plus every ``config is None`` fallback in the server resolves through it at
    CALL time. Repointing the module-level ``_DATA_ROOT`` here therefore moves
    the whole process, whatever a test builds its config from and whichever
    module asks — no per-call-site opt-in, no importer list to maintain.

    Scope note (verified, not assumed): this covers DEFAULTS. A config that
    names its data paths explicitly keeps them — ``tests/fixtures/server.yaml``
    points at ``tests/fixtures/sandbox/data/ha`` (safe, gitignored), while the
    operator-local ``configs/server.yaml`` points at the live tree, so loading
    THAT file in a test still yields live paths. See
    ``tests/test_data_root_isolation.py`` for the invariant this fixture is
    pinned by.
    """
    from paramem.server import config as server_config

    monkeypatch.setattr(server_config, "_DATA_ROOT", tmp_path / "data" / "ha")


@pytest.fixture(autouse=True)
def _admin_scope_default():
    """Grant admin scope by default on the module-singleton ``app_module.app``.

    Endpoint tests across ``tests/server/*`` and a handful of root-level
    files (``test_consolidation.py``, ``test_consolidation_guard.py``,
    ``test_lang_id.py``) build ``TestClient(app_module.app)`` with no auth
    configured and call ``require_admin``-gated routes — they exist to test
    endpoint LOGIC, not the auth boundary. A fail-closed-admin security fix
    (unconfigured auth now stamps the non-admin ``chat`` scope instead of
    ``admin``) made every one of those calls 403 ``admin_scope_required``.

    Auth behavior itself is independently covered by
    ``tests/server/test_require_admin.py`` and
    ``tests/server/test_auth_middleware.py`` — both build their OWN
    ``FastAPI()`` instance (never ``app_module.app``), so overriding the
    dependency on the shared singleton cannot mask those.

    ``dependency_overrides`` lives on the shared ``app`` object, so the
    override is popped on teardown (not just cleared) to guarantee it never
    leaks into a test that deliberately wants the real admin gate.
    """
    from paramem.server.app import app, require_admin

    app.dependency_overrides[require_admin] = lambda: None
    yield
    app.dependency_overrides.pop(require_admin, None)


@pytest.fixture(autouse=True)
def _extraction_trace_scope():
    """Auto-wrap every test in an :func:`extraction_trace` scope.

    The phase-trace contract (:mod:`paramem.graph.phase_trace`) requires
    every :func:`phase_trace` call to fire inside an active
    :func:`extraction_trace` — production runs through ``extract_graph``
    which establishes that scope.  Tests that exercise pipeline
    internals (the ``anonymize``/``enrich`` stage bodies,
    ``anonymize_transcript``, etc.) directly would otherwise trip
    the "outside an active trace" guard.

    The fixture is no-op when nesting (``extraction_trace`` is
    re-entrant by design — see its docstring), so tests that wrap their
    own scope remain correct.
    """
    from paramem.graph.phase_trace import extraction_trace

    with extraction_trace():
        yield


@pytest.fixture(autouse=True)
def _isolate_systemd_timer_boundary(monkeypatch, tmp_path):
    """Redirect every systemd unit write away from the operator's real timer dir.

    ``_apply_config_live`` and the app lifespan (``paramem/server/app.py``)
    reconcile the consolidation timer (:mod:`paramem.server.systemd_timer`)
    and the backup timer (:mod:`paramem.backup.timer`) on every config apply.
    A test that calls either path without mocking the systemd boundary would
    write real ``paramem-consolidate.{timer,service}`` /
    ``paramem-backup.{timer,service}`` units into
    ``~/.config/systemd/user/`` and run a real ``systemctl enable --now``,
    arming a timer the operator has ordered kept off — this has already
    happened once from an unmocked test call site.

    Both timers share one reconciliation core
    (:func:`paramem.server.systemd_timer._reconcile_timer`), so redirecting
    each module's ``UNIT_DIR``/``TIMER_PATH``/``SERVICE_PATH`` into
    ``tmp_path`` and replacing the single ``_run_systemctl`` implementation
    with an inert stub covers both, regardless of which test file reaches
    ``reconcile()``.

    Tests that deliberately exercise timer behaviour (e.g.
    ``tests/test_systemd_timer_calendar.py``, ``tests/test_scheduler.py``,
    ``tests/test_app_lifespan.py``) monkeypatch these same names themselves.
    ``monkeypatch`` is function-scoped and shared with this fixture, so a
    later ``monkeypatch.setattr`` call in the test body simply overrides the
    defaults set here for the remainder of that test.
    """
    from paramem.backup import timer as backup_timer
    from paramem.server import systemd_timer

    def _inert_run_systemctl(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(systemd_timer, "_run_systemctl", _inert_run_systemctl)

    for module in (systemd_timer, backup_timer):
        monkeypatch.setattr(module, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(module, "TIMER_PATH", tmp_path / f"{module.TIMER_NAME}.timer")
        monkeypatch.setattr(module, "SERVICE_PATH", tmp_path / f"{module.TIMER_NAME}.service")
