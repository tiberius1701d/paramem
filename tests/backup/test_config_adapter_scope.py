"""Tests for the adapter_scope field added to ServerBackupsConfig.

Covers:
- Default value is "live".
- "live" and "main" are both valid.
- Any other value raises ValueError at construction time.
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
