"""Tests for :func:`paramem.server.security_posture.security_posture_log_line`.

Four posture buckets the server chooses between at startup, pinned so a
refactor cannot silently swap the operator-visible SECURITY banner.
"""

from __future__ import annotations

from paramem.backup.encryption import MASTER_KEY_ENV_VAR
from paramem.server.security_posture import security_posture_log_line


class TestSecurityPostureLogLine:
    def test_age_with_recovery_is_multi_recipient_posture(self) -> None:
        line, is_on = security_posture_log_line(
            fernet_loaded=False,
            daily_loadable=True,
            recovery_available=True,
        )
        assert is_on is True
        assert "SECURITY: ON" in line
        assert "age daily identity loaded" in line
        assert "recovery recipient available" in line

    def test_age_without_recovery_warns_in_line(self) -> None:
        line, is_on = security_posture_log_line(
            fernet_loaded=False,
            daily_loadable=True,
            recovery_available=False,
        )
        assert is_on is True
        assert "SECURITY: ON" in line
        assert "recovery recipient missing" in line
        assert "paramem generate-key" in line

    def test_fernet_only_legacy_posture(self) -> None:
        line, is_on = security_posture_log_line(
            fernet_loaded=True,
            daily_loadable=False,
            recovery_available=False,
        )
        assert is_on is True
        assert line == f"SECURITY: ON ({MASTER_KEY_ENV_VAR} set)"

    def test_no_keys_is_off(self) -> None:
        line, is_on = security_posture_log_line(
            fernet_loaded=False,
            daily_loadable=False,
            recovery_available=False,
        )
        assert is_on is False
        assert "SECURITY: OFF" in line
        assert "plaintext on disk" in line

    def test_age_wins_over_fernet_when_both_present(self) -> None:
        """Transitional state: both keys loaded during migration. The age
        posture wins because new writes will route through age."""
        line, is_on = security_posture_log_line(
            fernet_loaded=True,
            daily_loadable=True,
            recovery_available=True,
        )
        assert is_on is True
        assert "age daily identity" in line
        assert MASTER_KEY_ENV_VAR not in line

    def test_recovery_available_without_daily_falls_through_to_fernet(self) -> None:
        """recovery.pub alone (no daily passphrase + file) is not enough —
        the reader needs the daily identity to unwrap. Fall through to the
        Fernet posture when that's the only key loaded."""
        line, is_on = security_posture_log_line(
            fernet_loaded=True,
            daily_loadable=False,
            recovery_available=True,
        )
        assert is_on is True
        assert line == f"SECURITY: ON ({MASTER_KEY_ENV_VAR} set)"

    def test_recovery_available_without_daily_or_fernet_is_off(self) -> None:
        line, is_on = security_posture_log_line(
            fernet_loaded=False,
            daily_loadable=False,
            recovery_available=True,
        )
        assert is_on is False
        assert "SECURITY: OFF" in line
