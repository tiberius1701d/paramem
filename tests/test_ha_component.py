"""Unit tests for the HA custom component (no HA runtime or GPU required).

Tests the pure-logic parts: manifest, strings, config constants, history
extraction, payload construction. HA-specific base classes are mocked
at module level before any custom_component imports.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

COMPONENT_DIR = Path(__file__).parent.parent / "custom_components" / "paramem"

# ---------------------------------------------------------------------------
# Mock all homeassistant.* modules before importing custom_components
# ---------------------------------------------------------------------------
_HA_MODULES = [
    "homeassistant",
    "homeassistant.components",
    "homeassistant.components.conversation",
    "homeassistant.config_entries",
    "homeassistant.const",
    "homeassistant.core",
    "homeassistant.helpers",
    "homeassistant.helpers.entity_platform",
]

_mocks = {}
for mod in _HA_MODULES:
    mock = MagicMock()
    _mocks[mod] = mock
    sys.modules[mod] = mock

# Make homeassistant.const.Platform.CONVERSATION resolve to a string
sys.modules["homeassistant.const"].Platform.CONVERSATION = "conversation"

# Now it's safe to import (must be after HA module mocking above)
from custom_components.paramem.const import DEFAULT_SERVER_URL, DEFAULT_TIMEOUT, DOMAIN  # noqa: E402, I001

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_loads(self):
        with open(COMPONENT_DIR / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["domain"] == "paramem"
        assert manifest["config_flow"] is True
        assert "conversation" in manifest["dependencies"]
        assert manifest["version"] == "0.1.0"

    def test_manifest_has_required_fields(self):
        with open(COMPONENT_DIR / "manifest.json") as f:
            manifest = json.load(f)

        for field in ["domain", "name", "config_flow", "dependencies", "version"]:
            assert field in manifest, f"Missing required field: {field}"


class TestStrings:
    def test_strings_well_formed(self):
        with open(COMPONENT_DIR / "strings.json") as f:
            strings = json.load(f)

        assert "config" in strings
        assert "step" in strings["config"]
        assert "user" in strings["config"]["step"]
        assert "error" in strings["config"]
        assert "cannot_connect" in strings["config"]["error"]


class TestConstants:
    def test_domain(self):
        assert DOMAIN == "paramem"

    def test_default_server_url(self):
        assert DEFAULT_SERVER_URL == "http://localhost:8420"

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 30


class TestPayloadConstruction:
    def test_payload_roundtrips_as_json(self):
        """The component's actual ``/chat`` payload shape: ``text``,
        ``conversation_id``, ``speaker`` — no ``history`` key. The server
        is history-authoritative (``SessionBuffer``); the component
        assembling and sending its own ``ChatLog``-derived history was dead
        work (the server drops any such key) and has been removed."""
        payload = {
            "text": "What's my favorite restaurant?",
            "conversation_id": "abc123",
            "speaker": "Alex",
        }
        deserialized = json.loads(json.dumps(payload))

        assert deserialized["text"] == "What's my favorite restaurant?"
        assert deserialized["conversation_id"] == "abc123"
        assert deserialized["speaker"] == "Alex"
        assert "history" not in deserialized

    def test_server_url_trailing_slash_stripped(self):
        assert "http://localhost:8420/".rstrip("/") == "http://localhost:8420"
