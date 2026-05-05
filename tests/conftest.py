"""Shared pytest configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="Run GPU integration tests")
    parser.addoption(
        "--recall",
        action="store_true",
        default=False,
        help="Run memory recall tests (train ~30 epochs + probe; requires --gpu)",
    )


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
def _extraction_trace_scope():
    """Auto-wrap every test in an :func:`extraction_trace` scope.

    The phase-trace contract (:mod:`paramem.graph.phase_trace`) requires
    every :func:`phase_trace` call to fire inside an active
    :func:`extraction_trace` — production runs through ``extract_graph``
    which establishes that scope.  Tests that exercise pipeline
    internals (``_sota_pipeline``, ``_anonymize_with_local_model``, etc.)
    directly would otherwise trip the "outside an active trace" guard.

    The fixture is no-op when nesting (``extraction_trace`` is
    re-entrant by design — see its docstring), so tests that wrap their
    own scope remain correct.
    """
    from paramem.graph.phase_trace import extraction_trace

    with extraction_trace():
        yield
