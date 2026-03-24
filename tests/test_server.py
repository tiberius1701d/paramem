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
        ac = config.episodic_adapter_config
        assert ac.rank == 8
        assert ac.alpha == 16
        assert ac.dropout == 0.0

    def test_training_config(self):
        config = ServerConfig()
        tc = config.training_config
        assert tc.num_epochs == 30
        assert tc.gradient_accumulation_steps == 2
        assert tc.max_seq_length == 1024

    def test_consolidation_config(self):
        config = ServerConfig()
        cc = config.consolidation_config
        assert cc.indexed_key_replay_enabled is True
        assert cc.promotion_threshold == 3

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

    def test_retain_sessions_false_deletes(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", retain_sessions=False)
        buffer.append("conv1", "user", "Hello")
        assert (tmp_path / "sessions" / "conv1.jsonl").exists()

        buffer.mark_consolidated(["conv1"])

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert not (tmp_path / "sessions" / "archive").exists()
        assert buffer.pending_count == 0

    def test_retain_sessions_true_archives(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", retain_sessions=True)
        buffer.append("conv1", "user", "Hello")

        buffer.mark_consolidated(["conv1"])

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert (tmp_path / "sessions" / "archive" / "conv1.jsonl").exists()

    def test_speaker_tracking(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        assert buffer.get_session_state("conv1") == "new"
        assert buffer.get_speaker("conv1") is None

        buffer.set_speaker("conv1", "Tobias")
        assert buffer.get_speaker("conv1") == "Tobias"
        assert buffer.get_session_state("conv1") == "identified"

    def test_speaker_in_transcript(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.set_speaker("conv1", "Tobias")
        buffer.append("conv1", "user", "I live in Amsterdam")

        pending = buffer.get_pending()
        assert "Tobias: I live in Amsterdam" in pending[0]["transcript"]


class TestKeyMetadata:
    def test_atomic_json_write(self, tmp_path):
        from paramem.server.consolidation import _atomic_json_write

        path = tmp_path / "test.json"
        _atomic_json_write({"key": "value"}, path)

        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}
        assert not (tmp_path / "test.tmp").exists()

    def test_atomic_json_write_list(self, tmp_path):
        from paramem.server.consolidation import _atomic_json_write

        path = tmp_path / "test.json"
        _atomic_json_write([1, 2, 3], path)

        with open(path) as f:
            data = json.load(f)
        assert data == [1, 2, 3]

    def test_load_key_metadata_missing(self, tmp_path):
        from paramem.server.consolidation import _load_key_metadata

        result = _load_key_metadata(tmp_path / "nonexistent.json")
        assert result is None

    def test_key_metadata_round_trip(self, tmp_path):
        from paramem.server.consolidation import (
            _atomic_json_write,
            _load_key_metadata,
        )

        metadata = {
            "cycle_count": 5,
            "promoted_keys": ["graph1", "graph2"],
            "keys": {
                "graph1": {"sessions_seen": 3},
                "graph3": {"sessions_seen": 1},
            },
        }
        path = tmp_path / "key_metadata.json"
        _atomic_json_write(metadata, path)

        loaded = _load_key_metadata(path)
        assert loaded["cycle_count"] == 5
        assert "graph1" in loaded["promoted_keys"]
        assert loaded["keys"]["graph3"]["sessions_seen"] == 1

    def test_voice_prompt_from_file(self, tmp_path):
        from paramem.server.config import VoiceConfig

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a test assistant.")

        vc = VoiceConfig(prompt_file=str(prompt_file))
        assert vc.load_prompt() == "You are a test assistant."

    def test_voice_prompt_fallback(self):
        from paramem.server.config import VoiceConfig

        vc = VoiceConfig(prompt_file="nonexistent.txt", system_prompt="Fallback prompt")
        assert vc.load_prompt() == "Fallback prompt"
