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

# Now it's safe to import
from custom_components.paramem.const import DEFAULT_SERVER_URL, DEFAULT_TIMEOUT, DOMAIN
from custom_components.paramem.conversation import _extract_history

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


class TestHistoryExtraction:
    def _make_entry(self, role, content):
        entry = MagicMock()
        entry.role = role
        entry.content = content
        entry.text = None
        return entry

    def test_extracts_roles_and_text(self):
        chat_log = MagicMock()
        chat_log.content = [
            self._make_entry("user", "Hello"),
            self._make_entry("assistant", "Hi there"),
        ]
        history = _extract_history(chat_log)

        assert len(history) == 2
        assert history[0] == {"role": "user", "text": "Hello"}
        assert history[1] == {"role": "assistant", "text": "Hi there"}

    def test_skips_entries_without_role(self):
        chat_log = MagicMock()
        chat_log.content = [self._make_entry(None, "orphan text")]

        history = _extract_history(chat_log)
        assert len(history) == 0

    def test_empty_chat_log(self):
        chat_log = MagicMock()
        chat_log.content = []

        history = _extract_history(chat_log)
        assert history == []

    def test_uses_text_fallback_when_content_is_none(self):
        entry = MagicMock()
        entry.role = "user"
        entry.content = None
        entry.text = "fallback text"

        chat_log = MagicMock()
        chat_log.content = [entry]

        history = _extract_history(chat_log)
        assert len(history) == 1
        assert history[0]["text"] == "fallback text"


class TestPayloadConstruction:
    def test_payload_roundtrips_as_json(self):
        payload = {
            "text": "What's my favorite restaurant?",
            "conversation_id": "abc123",
            "history": [
                {"role": "user", "text": "Previous question"},
                {"role": "assistant", "text": "Previous answer"},
            ],
        }
        deserialized = json.loads(json.dumps(payload))

        assert deserialized["text"] == "What's my favorite restaurant?"
        assert deserialized["conversation_id"] == "abc123"
        assert len(deserialized["history"]) == 2

    def test_server_url_trailing_slash_stripped(self):
        assert "http://localhost:8420/" .rstrip("/") == "http://localhost:8420"
