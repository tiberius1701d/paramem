"""Tests for ``ServerBackupsConfig.__post_init__`` validation.

Covers both rules enforced at construction time:
- ``max_total_disk_gb`` must be strictly positive (the write-time disk-cap
  enforcement assumes a positive cap; a non-positive value would refuse every
  backup write).
- ``adapter_scope`` must be one of ``"live"`` / ``"main"``.
"""

from __future__ import annotations

import pytest

from paramem.server.config import ServerBackupsConfig


class TestAdapterScopeDefault:
    def test_default_is_live(self) -> None:
        """adapter_scope defaults to 'live'."""
        cfg = ServerBackupsConfig()
        assert cfg.adapter_scope == "live"


class TestAdapterScopeValidValues:
    @pytest.mark.parametrize("scope", ["live", "main"])
    def test_valid_scopes_accepted(self, scope: str) -> None:
        """Both 'live' and 'main' are valid adapter_scope values."""
        cfg = ServerBackupsConfig(adapter_scope=scope)
        assert cfg.adapter_scope == scope


class TestAdapterScopeInvalidValues:
    @pytest.mark.parametrize("bad_scope", ["auto", "all", "", "LIVE", "Live"])
    def test_invalid_scope_raises_value_error(self, bad_scope: str) -> None:
        """Any value other than 'live'/'main' raises ValueError."""
        with pytest.raises(ValueError, match="adapter_scope"):
            ServerBackupsConfig(adapter_scope=bad_scope)


class TestMaxTotalDiskGbDefault:
    def test_default_is_20(self) -> None:
        """max_total_disk_gb defaults to 20.0."""
        cfg = ServerBackupsConfig()
        assert cfg.max_total_disk_gb == 20.0


class TestMaxTotalDiskGbInvalidValues:
    @pytest.mark.parametrize("bad_cap", [0, 0.0, -1.0, -1e-9])
    def test_non_positive_cap_raises_value_error(self, bad_cap: float) -> None:
        """Zero or negative max_total_disk_gb raises ValueError.

        A non-positive cap would refuse every backup write at the disk-cap
        enforcement door — this is invalid configuration, not a policy choice.
        Operators who want no backups use ``schedule: "off"``.
        """
        with pytest.raises(ValueError, match="max_total_disk_gb"):
            ServerBackupsConfig(max_total_disk_gb=bad_cap)


class TestMaxTotalDiskGbValidValues:
    @pytest.mark.parametrize("good_cap", [20.0, 1e-12])
    def test_positive_cap_accepted(self, good_cap: float) -> None:
        """Any strictly positive value is accepted, including a sub-byte cap.

        ``1e-12`` documents that a sub-byte-but-positive cap is legal config:
        ``int(1e-12 * 1024**3) == 0`` still refuses every write at the door,
        but the config itself is valid.
        """
        cfg = ServerBackupsConfig(max_total_disk_gb=good_cap)
        assert cfg.max_total_disk_gb == good_cap
