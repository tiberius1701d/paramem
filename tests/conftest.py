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
