"""Tests for assert_startup_posture and its lifespan wiring.

Semantics: uniform AUTO encryption — encrypt when the age daily identity
is loadable, plaintext otherwise. Operators opt into fail-loud via
``security.require_encryption: bool`` at startup via
:func:`paramem.server.security_posture.assert_startup_posture`.

Tests verify:
- ``require_encryption=False`` is a no-op regardless of key state.
- ``require_encryption=True`` with the daily identity loadable does not raise.
- ``require_encryption=True`` without the daily identity raises FatalConfigError.
- The lifespan source calls ``assert_startup_posture``.
"""

from __future__ import annotations

import inspect

import pytest

from paramem.backup.types import FatalConfigError
from paramem.server.security_posture import assert_startup_posture

# ---------------------------------------------------------------------------
# assert_startup_posture unit tests
# ---------------------------------------------------------------------------


class TestAssertStartupPosture:
    """Unit tests for assert_startup_posture covering the key-state paths."""

    def test_require_encryption_false_is_noop(self) -> None:
        """require_encryption=False → no-op regardless of key state."""
        # Must not raise even though no key is loadable.
        assert_startup_posture(
            require_encryption=False,
            daily_loadable=False,
        )

    def test_require_encryption_true_with_daily_ok(self) -> None:
        """require_encryption=True + daily_loadable=True → no exception."""
        assert_startup_posture(
            require_encryption=True,
            daily_loadable=True,
        )

    def test_require_encryption_true_daily_not_loadable_raises(self) -> None:
        """require_encryption=True, daily identity not loadable → FatalConfigError.

        The error message must mention ``require_encryption`` and the
        daily passphrase env var so operators know what to check.
        """
        from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR  # noqa: PLC0415

        with pytest.raises(FatalConfigError) as exc_info:
            assert_startup_posture(
                require_encryption=True,
                daily_loadable=False,
            )
        message = str(exc_info.value)
        assert "require_encryption" in message, (
            f"FatalConfigError must mention 'require_encryption': {message!r}"
        )
        assert DAILY_PASSPHRASE_ENV_VAR in message, (
            f"FatalConfigError must mention {DAILY_PASSPHRASE_ENV_VAR!r}: {message!r}"
        )


# ---------------------------------------------------------------------------
# Lifespan source smoke tests
# ---------------------------------------------------------------------------


class TestLifespanInvokesAssertStartupPosture:
    """Verify that the lifespan startup block calls assert_startup_posture."""

    def test_lifespan_invokes_assert_startup_posture(self) -> None:
        """assert_startup_posture must be called at lifespan entry.

        Inspects the lifespan source so a refactor that removes the call
        will fail here rather than silently. The function is imported from
        :mod:`paramem.server.security_posture`.
        """
        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert "assert_startup_posture(" in source, (
            "lifespan must call assert_startup_posture to enforce the "
            "require_encryption startup gate"
        )

    def test_lifespan_invokes_mode_consistency(self) -> None:
        """assert_mode_consistency must be called at lifespan entry."""
        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert "_assert_mode(" in source, (
            "lifespan must invoke assert_mode_consistency to enforce the "
            "SECURITY.md §4 four-case refuse"
        )


# ---------------------------------------------------------------------------
# Lifespan security-posture contract
# ---------------------------------------------------------------------------


class TestLifespanSecurityPosture:
    """Verify lifespan wires the SECURITY.md §4 posture gate correctly.

    Both ``assert_startup_posture`` and ``assert_mode_consistency`` are
    invoked at startup.  The former refuses the
    ``require_encryption=true + daily identity not loadable`` misconfiguration;
    the latter refuses on-disk mixed-state (plaintext alongside age) or a
    key-present-but-store-plaintext mismatch.
    """

    def test_primitive_is_importable(self) -> None:
        """assert_mode_consistency must exist and be callable from encryption."""
        from paramem.backup.encryption import assert_mode_consistency

        assert callable(assert_mode_consistency)

    def test_lifespan_invokes_mode_consistency(self) -> None:
        """assert_mode_consistency must be called at lifespan entry."""
        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert "_assert_mode(" in source, (
            "lifespan must invoke assert_mode_consistency to enforce the "
            "SECURITY.md §4 four-case refuse"
        )

    def test_lifespan_emits_security_posture_line(self) -> None:
        """lifespan must emit the SECURITY: ON/OFF log line per SECURITY.md §4.

        The line content lives in :mod:`paramem.server.security_posture`
        (factored out so the branching logic is a pure function of the three
        key-state booleans); the lifespan's job is to invoke that helper and
        route the result to the right log level. Pin both sides so neither
        refactor drift nor a stale inline literal sneaks through.
        """
        from paramem.server import app as app_module
        from paramem.server import security_posture as posture_module

        source = inspect.getsource(app_module.lifespan)
        assert "security_posture_log_line(" in source, (
            "lifespan must route through security_posture.security_posture_log_line"
        )

        posture_source = inspect.getsource(posture_module)
        assert "SECURITY: ON" in posture_source
        assert "SECURITY: OFF" in posture_source

    def test_lifespan_sets_encryption_state_field(self) -> None:
        """lifespan must populate _state['encryption'] so /status can expose
        the posture without re-probing the env on every request."""
        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert '_state["encryption"]' in source

    def test_status_response_surfaces_encryption(self) -> None:
        """StatusResponse must carry an ``encryption`` field and the /status
        handler must populate it from ``_state['encryption']`` — otherwise
        the lifespan setter is dead code (SECURITY.md and README both
        document that /status reports encryption: on|off)."""
        from paramem.server import app as app_module

        assert "encryption" in app_module.StatusResponse.model_fields, (
            "StatusResponse must declare an encryption field"
        )

        handler_source = inspect.getsource(app_module.status)
        assert (
            '_state.get("encryption"' in handler_source or '_state["encryption"]' in handler_source
        ), (  # noqa: E501
            "/status handler must read _state['encryption'] into the response"
        )
