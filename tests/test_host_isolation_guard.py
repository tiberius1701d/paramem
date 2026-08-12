"""Self-test for the autouse host-isolation guard in ``tests/conftest.py``.

Verifies the guard's three protections directly: the socket block on the
production port (``connect`` and ``connect_ex``, INET and INET6, with
non-tuple/non-production addresses passing through untouched), the raising
``systemctl.spawn`` stub, and the inert ``systemctl.run`` stub. Each
triggering test clears its recorded violation afterward so the autouse
teardown check does not fail the test twice.
"""

from __future__ import annotations

import socket

import pytest

from paramem.utils import systemctl
from tests._host_isolation import PRODUCTION_PORT, HostIsolationViolation


def test_socket_connect_to_production_port_raises_and_is_cleared(_host_isolation_violations):
    """A connect() to 127.0.0.1:PRODUCTION_PORT raises HostIsolationViolation."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(HostIsolationViolation):
            sock.connect(("127.0.0.1", PRODUCTION_PORT))
    finally:
        sock.close()
    assert _host_isolation_violations
    _host_isolation_violations.clear()


def test_connect_ex_to_production_port_raises_and_is_cleared(_host_isolation_violations):
    """connect_ex() to PRODUCTION_PORT also raises — not just connect()."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(HostIsolationViolation):
            sock.connect_ex(("127.0.0.1", PRODUCTION_PORT))
    finally:
        sock.close()
    assert _host_isolation_violations
    _host_isolation_violations.clear()


def test_af_unix_address_passes_through_untouched(tmp_path):
    """A str/bytes AF_UNIX address is not a production-port tuple — it passes through.

    Connecting to a nonexistent unix socket path raises a plain OSError from
    the real ``connect`` (never reached the guard's tuple-shaped check), not
    ``HostIsolationViolation``.
    """
    sock_path = tmp_path / "nonexistent.sock"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError) as exc_info:
            sock.connect(str(sock_path))
        assert not isinstance(exc_info.value, HostIsolationViolation)
    finally:
        sock.close()


def test_af_inet6_four_tuple_to_production_port_raises_and_is_cleared(_host_isolation_violations):
    """An AF_INET6 4-tuple address (host, port, flowinfo, scopeid) is still port-scoped.

    No real IPv6 connection is needed — the guard raises before any syscall.
    """
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        with pytest.raises(HostIsolationViolation):
            sock.connect(("::1", PRODUCTION_PORT, 0, 0))
    finally:
        sock.close()
    assert _host_isolation_violations
    _host_isolation_violations.clear()


def test_systemctl_spawn_restart_raises_and_is_cleared(_host_isolation_violations):
    """systemctl.spawn('restart', 'paramem-server') raises and is recorded."""
    with pytest.raises(HostIsolationViolation):
        systemctl.spawn("restart", "paramem-server")
    assert _host_isolation_violations
    _host_isolation_violations.clear()


def test_systemctl_run_show_environment_is_inert(_host_isolation_violations):
    """systemctl.run('show-environment') returns rc=0 and records nothing."""
    result = systemctl.run("show-environment")
    assert result.returncode == 0
    assert result.stdout == ""
    assert not _host_isolation_violations


def test_connect_to_ephemeral_port_listener_succeeds():
    """The guard is port-scoped: a loopback connect to a non-production port works."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    assert port != PRODUCTION_PORT

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(("127.0.0.1", port))
    finally:
        client.close()
        listener.close()
