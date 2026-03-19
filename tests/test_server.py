"""Unit tests for the ParaMem server modules (no GPU required)."""

import json
from pathlib import Path

import pytest

from paramem.server.config import ServerConfig, load_server_config
from paramem.server.escalation import detect_escalation
from paramem.server.session_buffer import SessionBuffer


class TestConfig:
    def test_load_default_config(self):
        config = load_server_config("configs/server.yaml")
        assert config.model_name == "mistral"
        assert config.server.port == 8420
        assert config.adapter_dir == Path("data/ha/adapters")
        assert config.cloud.enabled is False

    def test_model_config_resolution(self):
        config = ServerConfig(model_name="mistral")
        mc = config.model_config
        assert mc.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
        assert mc.quantization == "nf4"

    def test_model_config_gemma(self):
        config = ServerConfig(model_name="gemma")
        mc = config.model_config
        assert mc.model_id == "google/gemma-2-9b-it"
        assert mc.cpu_offload is True

    def test_unknown_model_raises(self):
        config = ServerConfig(model_name="nonexistent")
        with pytest.raises(ValueError, match="Unknown model"):
            _ = config.model_config

    def test_adapter_config(self):
        config = ServerConfig()
        ac = config.adapter_config
        assert ac.rank == 8
        assert ac.alpha == 16

    def test_training_config(self):
        config = ServerConfig()
        tc = config.training_config
        assert tc.num_epochs == 30

    def test_missing_config_file_returns_defaults(self):
        config = load_server_config("nonexistent.yaml")
        assert config.model_name == "mistral"
        assert config.server.port == 8420


class TestEscalation:
    def test_no_escalation(self):
        should, query = detect_escalation("Paris is the capital of France.")
        assert should is False
        assert query == ""

    def test_escalation_detected(self):
        should, query = detect_escalation("[ESCALATE] What is the capital of France?")
        assert should is True
        assert query == "What is the capital of France?"

    def test_escalation_with_whitespace(self):
        should, query = detect_escalation("  [ESCALATE]   What is quantum computing?  ")
        assert should is True
        assert query == "What is quantum computing?"

    def test_escalation_mid_sentence_not_detected(self):
        should, query = detect_escalation("The team decided to [ESCALATE] the issue to management.")
        assert should is False

    def test_empty_response(self):
        should, query = detect_escalation("")
        assert should is False

    def test_escalation_tag_only(self):
        should, query = detect_escalation("[ESCALATE]")
        assert should is True
        assert query == ""


class TestSessionBuffer:
    def test_append_and_get_pending(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv1", "assistant", "Hi there!")

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == "conv1"
        assert "User: Hello" in pending[0]["transcript"]
        assert "Assistant: Hi there!" in pending[0]["transcript"]

    def test_multiple_conversations(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv2", "user", "Hi")

        pending = buffer.get_pending()
        assert len(pending) == 2

    def test_mark_consolidated(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv2", "user", "Hi")

        buffer.mark_consolidated(["conv1"])

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == "conv2"

        # Archived file should exist
        assert (tmp_path / "sessions" / "archive" / "conv1.jsonl").exists()

    def test_pending_count(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        assert buffer.pending_count == 0

        buffer.append("conv1", "user", "Hello")
        assert buffer.pending_count == 1

    def test_empty_buffer(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        assert buffer.get_pending() == []

    def test_turn_timestamps(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")

        path = tmp_path / "sessions" / "conv1.jsonl"
        with open(path) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert entry["role"] == "user"
        assert entry["text"] == "Hello"
