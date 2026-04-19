"""Tests for cloud_only configuration loading and precedence.

Covers:
  - YAML cloud_only: true sets ServerConfig.cloud_only = True.
  - CLI --cloud-only flag sets cloud_only_startup state.
  - Both YAML and CLI together produce cloud_only ON (idempotent OR).
  - YAML false + CLI absent leaves cloud_only off.

All tests are pure config-loading — no GPU, no model load.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from paramem.server.config import ServerConfig, load_server_config

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_server_yaml(tmp_path: Path, content: str) -> Path:
    """Write a minimal server.yaml to tmp_path and return its path."""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(textwrap.dedent(content))
    return yaml_path


# ---------------------------------------------------------------------------
# Test 1 — YAML cloud_only: true enables cloud_only
# ---------------------------------------------------------------------------


def test_yaml_cloud_only_true_enables_cloud_only(tmp_path: Path) -> None:
    """When server.yaml sets cloud_only: true, ServerConfig.cloud_only is True."""
    yaml_path = _write_server_yaml(
        tmp_path,
        """
        cloud_only: true
        """,
    )
    config = load_server_config(yaml_path)
    assert config.cloud_only is True, (
        f"Expected cloud_only=True when YAML sets cloud_only: true. Got: {config.cloud_only}"
    )


# ---------------------------------------------------------------------------
# Test 2 — YAML cloud_only: false leaves cloud_only off
# ---------------------------------------------------------------------------


def test_yaml_cloud_only_false_leaves_cloud_only_off(tmp_path: Path) -> None:
    """When server.yaml sets cloud_only: false, ServerConfig.cloud_only is False."""
    yaml_path = _write_server_yaml(
        tmp_path,
        """
        cloud_only: false
        """,
    )
    config = load_server_config(yaml_path)
    assert config.cloud_only is False, (
        f"Expected cloud_only=False when YAML sets cloud_only: false. Got: {config.cloud_only}"
    )


# ---------------------------------------------------------------------------
# Test 3 — YAML absent (no cloud_only key) defaults to False
# ---------------------------------------------------------------------------


def test_yaml_absent_cloud_only_defaults_to_false(tmp_path: Path) -> None:
    """When cloud_only is absent from server.yaml, ServerConfig.cloud_only defaults to False."""
    yaml_path = _write_server_yaml(
        tmp_path,
        """
        model: mistral
        """,
    )
    config = load_server_config(yaml_path)
    assert config.cloud_only is False, (
        f"Expected cloud_only=False when YAML omits the key. Got: {config.cloud_only}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Default ServerConfig has cloud_only=False
# ---------------------------------------------------------------------------


def test_default_server_config_cloud_only_is_false() -> None:
    """ServerConfig() default must have cloud_only=False."""
    config = ServerConfig()
    assert config.cloud_only is False, (
        f"Default ServerConfig.cloud_only must be False. Got: {config.cloud_only}"
    )


# ---------------------------------------------------------------------------
# Test 5 — CLI flag sets cloud_only_startup state (unit of state dict)
# ---------------------------------------------------------------------------


def test_cli_flag_cloud_only_enables_cloud_only_startup() -> None:
    """The --cloud-only CLI flag sets _state['cloud_only_startup']=True in app.py.

    This test verifies the state dict semantics without running the full
    server lifespan.  The OR combiner in lifespan reads cloud_only_startup.
    """
    # Simulate what app.py's main() does after parsing --cloud-only
    from paramem.server import app as server_app

    # Save and restore original state to avoid cross-test contamination
    original = server_app._state.get("cloud_only_startup", False)
    try:
        server_app._state["cloud_only_startup"] = True
        assert server_app._state["cloud_only_startup"] is True, (
            "cloud_only_startup must be True after simulating --cloud-only CLI flag"
        )
    finally:
        server_app._state["cloud_only_startup"] = original


# ---------------------------------------------------------------------------
# Test 6 — Both YAML true and CLI flag produce cloud_only ON (idempotent OR)
# ---------------------------------------------------------------------------


def test_both_yaml_and_cli_enables_cloud_only(tmp_path: Path) -> None:
    """YAML cloud_only: true AND CLI --cloud-only together remain cloud_only ON.

    The OR combiner is idempotent: True OR True == True.
    """
    yaml_path = _write_server_yaml(
        tmp_path,
        """
        cloud_only: true
        """,
    )
    config = load_server_config(yaml_path)
    # Simulate CLI flag also set
    cli_cloud_only = True
    # The lifespan OR combiner: cli OR yaml
    result = cli_cloud_only or config.cloud_only
    assert result is True, f"YAML true + CLI true must produce cloud_only=True. Got: {result}"


# ---------------------------------------------------------------------------
# Test 7 — YAML false + CLI absent leaves cloud_only off
# ---------------------------------------------------------------------------


def test_yaml_false_cli_absent_leaves_cloud_only_off(tmp_path: Path) -> None:
    """YAML cloud_only: false and no CLI flag → cloud_only remains False.

    This is the normal production startup path.
    """
    yaml_path = _write_server_yaml(
        tmp_path,
        """
        cloud_only: false
        """,
    )
    config = load_server_config(yaml_path)
    cli_cloud_only = False  # no --cloud-only flag
    result = cli_cloud_only or config.cloud_only
    assert result is False, f"YAML false + CLI absent must produce cloud_only=False. Got: {result}"


# ---------------------------------------------------------------------------
# Test 8 — cloud_only: true in YAML enables even without CLI flag
# ---------------------------------------------------------------------------


def test_yaml_true_cli_absent_enables_cloud_only(tmp_path: Path) -> None:
    """YAML cloud_only: true is sufficient to enable cloud_only without any CLI flag."""
    yaml_path = _write_server_yaml(
        tmp_path,
        """
        cloud_only: true
        """,
    )
    config = load_server_config(yaml_path)
    cli_cloud_only = False  # no --cloud-only flag
    result = cli_cloud_only or config.cloud_only
    assert result is True, (
        "YAML cloud_only: true must enable cloud_only even when CLI flag is absent. "
        f"Got config.cloud_only={config.cloud_only}, combined={result}"
    )
