"""Tests for assert_startup_posture and its lifespan wiring.

Semantics (post-refactor, 2026-04-24): uniform AUTO encryption — encrypt
when age daily OR ``PARAMEM_MASTER_KEY`` is loadable, plaintext otherwise.
Operators opt into fail-loud via ``security.require_encryption: bool`` at
startup via :func:`paramem.server.security_posture.assert_startup_posture`.

Tests verify:
- ``require_encryption=False`` is a no-op regardless of key state.
- ``require_encryption=True`` with a Fernet key loaded does not raise.
- ``require_encryption=True`` with a daily identity loadable does not raise.
- ``require_encryption=True`` with neither key raises FatalConfigError.
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
    """Unit tests for assert_startup_posture covering the four key-state paths."""

    def test_require_encryption_false_is_noop(self) -> None:
        """require_encryption=False → no-op regardless of key state."""
        # Must not raise even though neither key is loadable.
        assert_startup_posture(
            require_encryption=False,
            fernet_loaded=False,
            daily_loadable=False,
        )

    def test_require_encryption_true_with_fernet_ok(self) -> None:
        """require_encryption=True + fernet_loaded=True → no exception."""
        assert_startup_posture(
            require_encryption=True,
            fernet_loaded=True,
            daily_loadable=False,
        )

    def test_require_encryption_true_with_daily_ok(self) -> None:
        """require_encryption=True + daily_loadable=True → no exception."""
        assert_startup_posture(
            require_encryption=True,
            fernet_loaded=False,
            daily_loadable=True,
        )

    def test_require_encryption_true_neither_raises(self) -> None:
        """require_encryption=True, neither key loadable → FatalConfigError.

        The error message must mention both ``require_encryption`` and
        ``PARAMEM_MASTER_KEY`` so operators know what to check.
        """
        with pytest.raises(FatalConfigError) as exc_info:
            assert_startup_posture(
                require_encryption=True,
                fernet_loaded=False,
                daily_loadable=False,
            )
        message = str(exc_info.value)
        assert "require_encryption" in message, (
            f"FatalConfigError must mention 'require_encryption': {message!r}"
        )
        assert "PARAMEM_MASTER_KEY" in message, (
            f"FatalConfigError must mention 'PARAMEM_MASTER_KEY': {message!r}"
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
    invoked at startup.  The former refuses the ``require_encryption=true +
    no key`` misconfiguration; the latter refuses the four key × on-disk
    mismatch cases (set+plaintext, unset+ciphertext, mixed) and points the
    operator at the ``paramem encrypt-infra`` / ``paramem decrypt-infra
    --i-accept-plaintext`` migration commands.
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
        """lifespan must populate _state['encryption'] so a future /status
        surface can expose the posture without touching the env again."""
        from paramem.server import app as app_module

        source = inspect.getsource(app_module.lifespan)
        assert '_state["encryption"]' in source
