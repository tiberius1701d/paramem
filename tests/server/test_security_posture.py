"""Tests for :func:`paramem.server.security_posture.security_posture_log_line`.

Three posture buckets the server chooses between at startup, pinned so a
refactor cannot silently swap the operator-visible SECURITY banner.
"""

from __future__ import annotations

from paramem.server.security_posture import security_posture_log_line


class TestSecurityPostureLogLine:
    def test_age_with_recovery_is_multi_recipient_posture(self) -> None:
        """daily_loadable=True + recovery_available=True → multi-recipient ON."""
        line, is_on = security_posture_log_line(
            daily_loadable=True,
            recovery_available=True,
        )
        assert is_on is True
        assert "SECURITY: ON" in line
        assert "age daily identity loaded" in line
        assert "recovery recipient available" in line

    def test_age_without_recovery_warns_in_line(self) -> None:
        """daily_loadable=True + recovery_available=False → ON with warning."""
        line, is_on = security_posture_log_line(
            daily_loadable=True,
            recovery_available=False,
        )
        assert is_on is True
        assert "SECURITY: ON" in line
        assert "recovery recipient missing" in line
        assert "paramem generate-key" in line

    def test_no_keys_is_off(self) -> None:
        """No daily identity → SECURITY: OFF."""
        line, is_on = security_posture_log_line(
            daily_loadable=False,
            recovery_available=False,
        )
        assert is_on is False
        assert "SECURITY: OFF" in line
        assert "plaintext on disk" in line

    def test_recovery_available_without_daily_is_off(self) -> None:
        """recovery.pub alone (no daily passphrase + file) is not enough.

        The reader needs the daily identity to unwrap. Security is OFF
        when the daily identity is not loadable regardless of recovery.pub.
        """
        line, is_on = security_posture_log_line(
            daily_loadable=False,
            recovery_available=True,
        )
        assert is_on is False
        assert "SECURITY: OFF" in line
