"""Unit tests for the ParaMem server modules (no GPU required)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.server.config import MODEL_REGISTRY, ServerConfig, load_server_config
from paramem.server.escalation import detect_escalation
from paramem.server.session_buffer import SessionBuffer


class TestConfig:
    def test_load_default_config(self):
        config = load_server_config("configs/server.yaml")
        assert config.model_name in MODEL_REGISTRY
        assert config.server.port == 8420
        assert config.adapter_dir == Path("data/ha/adapters").resolve()

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

    def test_env_var_interpolation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-secret-123")
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "agents:\n"
            "  sota:\n"
            "    enabled: true\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet\n"
            "    api_key: ${TEST_API_KEY}\n"
        )
        config = load_server_config(config_file)
        assert config.sota_agent.api_key == "sk-secret-123"
        assert config.sota_agent.provider == "anthropic"

    def test_env_var_missing_uses_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "agents:\n  sota:\n    enabled: true\n    api_key: ${NONEXISTENT_VAR}\n"
        )
        config = load_server_config(config_file)
        assert config.sota_agent.api_key == ""

    def test_prompts_path_loaded(self):
        config = load_server_config("configs/server.yaml")
        assert config.paths.prompts == Path("configs/prompts").resolve()


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

    def test_escalation_mid_sentence_detected(self):
        text = "I don't know the answer. [ESCALATE] What is the weather?"
        should, query = detect_escalation(text)
        assert should is True
        assert query == "What is the weather?"

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

    def test_mark_consolidated_debug_archives(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", debug=True)
        buffer.append("conv1", "user", "Hello")

        buffer.mark_consolidated(["conv1"])

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
        buffer = SessionBuffer(tmp_path / "sessions", debug=True)
        buffer.append("conv1", "user", "Hello")

        path = tmp_path / "sessions" / "conv1.jsonl"
        with open(path) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert entry["role"] == "user"
        assert entry["text"] == "Hello"

    def test_no_disk_writes_without_debug(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert buffer.pending_count == 1
        assert len(buffer.get_pending()) == 1

    def test_retain_sessions_false_deletes(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", retain_sessions=False, debug=True)
        buffer.append("conv1", "user", "Hello")
        assert (tmp_path / "sessions" / "conv1.jsonl").exists()

        buffer.mark_consolidated(["conv1"])

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert not (tmp_path / "sessions" / "archive").exists()
        assert buffer.pending_count == 0

    def test_retain_sessions_true_archives(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", retain_sessions=True, debug=True)
        buffer.append("conv1", "user", "Hello")

        buffer.mark_consolidated(["conv1"])

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert (tmp_path / "sessions" / "archive" / "conv1.jsonl").exists()

    def test_speaker_tracking(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        assert buffer.get_session_state("conv1") == "new"
        assert buffer.get_speaker("conv1") is None
        assert buffer.get_speaker_id("conv1") is None

        buffer.set_speaker("conv1", "spk_abc", "Alex")
        assert buffer.get_speaker("conv1") == "Alex"
        assert buffer.get_speaker_id("conv1") == "spk_abc"
        assert buffer.get_session_state("conv1") == "identified"

    def test_speaker_in_transcript(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.set_speaker("conv1", "spk_abc", "Alex")
        buffer.append("conv1", "user", "I live in Amsterdam")

        pending = buffer.get_pending()
        assert "Alex: I live in Amsterdam" in pending[0]["transcript"]
        assert pending[0]["speaker_id"] == "spk_abc"

    def test_snapshot_save_and_restore(self, tmp_path):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()

        # Populate buffer and save snapshot
        buf1 = SessionBuffer(tmp_path / "sessions", snapshot_key=key)
        buf1.set_speaker("conv1", "spk_abc", "Alex")
        buf1.append("conv1", "user", "I live in Amsterdam")
        buf1.append("conv1", "assistant", "That's nice!")
        assert buf1.save_snapshot()
        assert (tmp_path / "sessions" / "session_snapshot.enc").exists()

        # Restore into a fresh buffer
        buf2 = SessionBuffer(tmp_path / "sessions", snapshot_key=key)
        assert buf2.load_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

        pending = buf2.get_pending()
        assert len(pending) == 1
        assert "Alex: I live in Amsterdam" in pending[0]["transcript"]
        assert pending[0]["speaker_id"] == "spk_abc"
        assert buf2.get_speaker("conv1") == "Alex"

    def test_snapshot_wrong_key_discards(self, tmp_path):
        from cryptography.fernet import Fernet

        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()

        buf1 = SessionBuffer(tmp_path / "sessions", snapshot_key=key1)
        buf1.append("conv1", "user", "Secret data")
        buf1.save_snapshot()

        buf2 = SessionBuffer(tmp_path / "sessions", snapshot_key=key2)
        assert not buf2.load_snapshot()  # decryption fails, file deleted
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()
        assert buf2.pending_count == 0

    def test_snapshot_deleted_on_restore(self, tmp_path):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()

        buf1 = SessionBuffer(tmp_path / "sessions", snapshot_key=key)
        buf1.append("conv1", "user", "Hello")
        buf1.save_snapshot()
        assert (tmp_path / "sessions" / "session_snapshot.enc").exists()

        buf2 = SessionBuffer(tmp_path / "sessions", snapshot_key=key)
        buf2.load_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

    def test_snapshot_empty_buffer_no_file(self, tmp_path):
        from cryptography.fernet import Fernet

        key = Fernet.generate_key().decode()

        buffer = SessionBuffer(tmp_path / "sessions", snapshot_key=key)
        assert buffer.save_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()


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


class TestProbeAndReasonDispatch:
    """Test that _probe_and_reason dispatches to probe_keys_grouped_by_adapter."""

    def _make_plan(self, steps):
        """Build a RoutingPlan from a list of (adapter_name, keys) tuples."""
        from paramem.server.router import RoutingPlan, RoutingStep

        return RoutingPlan(
            steps=[RoutingStep(adapter_name=a, keys_to_probe=list(k)) for a, k in steps],
            strategy="direct",
            match_source="pa",
        )

    def _make_model(self, adapter_names):
        """Stub model with peft_config for the given adapter names."""
        model = MagicMock()
        model.peft_config = {name: MagicMock() for name in adapter_names}
        return model

    def test_dispatches_to_grouped_probe_with_correct_groups(self, monkeypatch, tmp_path):
        """_probe_and_reason builds keys_by_adapter in step order and calls
        probe_keys_grouped_by_adapter with those groups."""
        from paramem.server.config import ServerConfig, VoiceConfig

        captured = {}

        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            captured["keys_by_adapter"] = dict(keys_by_adapter)
            # Return all keys as successful probes.
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    results[k] = {"key": k, "answer": f"ans_{k}", "confidence": 1.0}
            return results

        # _probe_and_reason uses a lazy local import from paramem.training.indexed_memory,
        # so we patch at the source module.
        monkeypatch.setattr(
            "paramem.training.indexed_memory.probe_keys_grouped_by_adapter",
            fake_grouped,
        )

        # Stub out downstream calls.
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            lambda model, name: None,
        )
        monkeypatch.setattr(
            "paramem.server.inference._load_simhash_registry",
            lambda path: {},
        )
        monkeypatch.setattr(
            "paramem.server.inference.sanitize_for_cloud",
            lambda text, mode=None: (text, []),
        )
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer",
            lambda model, tokenizer, prompt, **kwargs: "final answer",
        )
        monkeypatch.setattr(
            "paramem.server.inference._build_messages",
            lambda text, history, system_prompt, tokenizer: [{"role": "user", "content": text}],
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"

        model = self._make_model(["episodic", "procedural"])
        # Disable PeftModel isinstance check so disable_adapter branch is skipped.
        monkeypatch.setattr(
            "paramem.server.inference.PeftModel",
            type(None),
            raising=False,
        )

        # Write a minimal voice prompt file.
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are an assistant.")
        config = ServerConfig()
        config.voice = VoiceConfig(prompt_file=str(prompt_file))

        plan = self._make_plan(
            [
                ("procedural", ["p1", "p2"]),
                ("episodic", ["e1"]),
            ]
        )

        from paramem.server.inference import _probe_and_reason

        _probe_and_reason(
            text="What do I like?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        assert "keys_by_adapter" in captured, "probe_keys_grouped_by_adapter was not called"
        kba = captured["keys_by_adapter"]
        # Both groups present.
        assert list(kba.keys()) == ["procedural", "episodic"], (
            f"Expected ['procedural', 'episodic'], got {list(kba.keys())}"
        )
        assert kba["procedural"] == ["p1", "p2"]
        assert kba["episodic"] == ["e1"]
