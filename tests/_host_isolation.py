"""Host-isolation primitives shared by ``tests/conftest.py`` and the guard self-test.

A broken test mock can otherwise fire a real ``systemctl --user restart
paramem-server`` (restarting the operator's production server) or a real GET
against the live server on ``localhost:8420``.  This module provides the
inert/raising stubs for the :mod:`paramem.utils.systemctl` transport seam and
a socket-level block on the production port; ``tests/conftest.py`` installs
both as an autouse fixture.
"""

from __future__ import annotations

import socket
import subprocess

# Hardcoded deliberately, not read from any server/test config — the guard
# must not follow test config (a config-driven port would let a misconfigured
# test silently point the guard at the wrong port). A server running on a
# non-default port is outside this guard's protection.
PRODUCTION_PORT = 8420


class HostIsolationViolation(BaseException):
    """Raised when test code is about to touch the host: real systemctl or the live server.

    Subclasses ``BaseException`` (not ``Exception``) so that ``except
    Exception`` handlers in production code — e.g. the swallow-all policy
    around ``systemctl.spawn`` call sites — cannot hide it.
    """


def make_inert_run():
    """Return an inert stand-in for ``paramem.utils.systemctl.run``.

    Returns:
        A callable matching ``systemctl.run``'s signature that performs no
        host interaction and reports success.
    """

    def _run(*args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=("systemctl", "--user", *args), returncode=0, stdout="", stderr=""
        )

    return _run


def make_raising_spawn(violations: list[str]):
    """Return a raising stand-in for ``paramem.utils.systemctl.spawn``.

    Args:
        violations: List to append a description of the blocked call to
            before raising — read by the conftest teardown backstop.

    Returns:
        A callable matching ``systemctl.spawn``'s signature that records and
        raises :class:`HostIsolationViolation` instead of touching the host.
    """

    def _spawn(*args: str) -> None:
        description = f"systemctl.spawn{args!r}"
        violations.append(description)
        raise HostIsolationViolation(f"blocked host-touching systemctl call: {description}")

    return _spawn


def install_socket_guard(monkeypatch, violations: list[str]) -> None:
    """Block ``socket.socket`` connections to :data:`PRODUCTION_PORT`.

    Overrides ``connect``/``connect_ex`` on the ``socket.socket`` class so
    any attempt to reach the live server's port — from an unmocked CLI test
    or a broken HTTP mock — raises :class:`HostIsolationViolation` instead of
    reaching the operator's real server. Non-INET addresses (e.g. AF_UNIX
    ``str``/``bytes`` paths) and any other port pass through untouched.

    Args:
        monkeypatch: The pytest ``monkeypatch`` fixture (governs undo).
        violations: List to append a description of any blocked connection
            attempt to before raising.
    """
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _is_production_address(address) -> bool:
        if not isinstance(address, tuple) or len(address) < 2:
            return False
        port = address[1]
        return isinstance(port, int) and port == PRODUCTION_PORT

    def _guarded_connect(self, address):
        if _is_production_address(address):
            description = f"socket.connect({address!r})"
            violations.append(description)
            raise HostIsolationViolation(f"blocked host-touching connection: {description}")
        return original_connect(self, address)

    def _guarded_connect_ex(self, address):
        if _is_production_address(address):
            description = f"socket.connect_ex({address!r})"
            violations.append(description)
            raise HostIsolationViolation(f"blocked host-touching connection: {description}")
        return original_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
