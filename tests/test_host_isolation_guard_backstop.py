"""Pins the host-isolation guard's teardown backstop against a swallowed violation.

The guard's central claim (see ``tests/conftest.py::_host_isolation_guard``)
is that a violation recorded during the test body but swallowed there — a
bare ``try/except BaseException`` around a triggering call, or
``TestClient(raise_server_exceptions=False)`` converting the raised
``HostIsolationViolation`` into a synthetic 500 — still fails the run at
teardown. A self-test running inside THIS process cannot pin that: swallowing
the exception inside the test body would just make the outer test pass,
proving nothing. This runs an inner pytest session as a real subprocess
(:mod:`_pytest.pytester`), loading the project's actual
``tests/conftest.py`` as a plugin, and asserts the inner run comes back red.
"""

from __future__ import annotations

import os
from pathlib import Path

pytest_plugins = ["pytester"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_teardown_fails_on_violation_swallowed_by_bare_except(pytester, monkeypatch):
    """A systemctl.spawn violation swallowed by the test body still fails at teardown.

    The inner test calls ``systemctl.spawn(...)`` (blocked by the guard) and
    immediately swallows the resulting ``HostIsolationViolation`` with a bare
    ``except BaseException: pass`` — mirroring what
    ``TestClient(raise_server_exceptions=False)`` does to a ``BaseException``
    raised inside an endpoint. The inner test body therefore PASSES on its
    own terms; the assertion here is that the outer run still reports it red
    (as a teardown error) because the guard's own teardown check inspects the
    recorded violations list independently of what the test body asserted.
    """
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, [str(_PROJECT_ROOT), os.environ.get("PYTHONPATH", "")])),
    )
    pytester.makepyfile(
        test_inner="""
        pytest_plugins = ["tests.conftest"]

        def test_swallowed_violation():
            from paramem.utils import systemctl

            try:
                systemctl.spawn("restart", "paramem-server")
            except BaseException:
                pass  # swallowed on purpose — the test body itself must not fail
        """
    )
    result = pytester.runpytest_subprocess()
    # The inner test body passes (the violation was swallowed there); the
    # guard's teardown check independently reports it as an error — this is
    # the backstop working, not a false negative.
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*Host-isolation guard recorded violation(s)*"])
    result.stdout.fnmatch_lines(["*systemctl.spawn*restart*paramem-server*"])
