"""Unit tests for the ParaMem server modules (no GPU required)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.memory.store import MemoryStore as _MS
from paramem.server.config import MODEL_REGISTRY, ServerConfig, load_server_config
from paramem.server.escalation import detect_escalation
from paramem.server.session_buffer import SessionBuffer

_OPERATOR_CONFIG = Path("configs/server.yaml")
_SKIP_NO_OPERATOR = pytest.mark.skipif(
    not _OPERATOR_CONFIG.exists(),
    reason="operator-local configs/server.yaml absent (CI / fresh clone)",
)


class TestConfig:
    @_SKIP_NO_OPERATOR
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

    @_SKIP_NO_OPERATOR
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
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv1", "assistant", "Hi there!")

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == "conv1"
        assert "[user] Hello" in pending[0]["transcript"]
        assert "[assistant] Hi there!" in pending[0]["transcript"]

    def test_multiple_conversations(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv2", "user", "Hi")

        pending = buffer.get_pending()
        assert len(pending) == 2

    def test_mark_consolidated(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv2", "user", "Hi")

        buffer.mark_consolidated(["conv1"])

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == "conv2"

    def test_mark_consolidated_debug_archives(self, tmp_path):
        """With debug=True + retention_dir supplied, mark_consolidated moves the JSONL."""
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=True)
        buffer.append("conv1", "user", "Hello")

        retention = tmp_path / "archive"
        buffer.mark_consolidated(["conv1"], retention_dir=retention)

        assert (retention / "conv1.jsonl").exists()
        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()

    def test_pending_count(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert buffer.pending_count == 0

        buffer.append("conv1", "user", "Hello")
        assert buffer.pending_count == 1

    def test_empty_buffer(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert buffer.get_pending() == []

    def test_turn_timestamps(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=True)
        buffer.append("conv1", "user", "Hello")

        path = tmp_path / "sessions" / "conv1.jsonl"
        with open(path) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry
        assert entry["role"] == "user"
        assert entry["text"] == "Hello"

    def test_append_persists_unconditionally(self, tmp_path):
        """Pending sessions persist on disk even without debug
        (2026-05-14 invariant — survives restarts until consolidation
        consumes them)."""
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "Hello")

        assert (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert buffer.pending_count == 1
        assert len(buffer.get_pending()) == 1

    def test_retain_sessions_false_deletes(self, tmp_path):
        buffer = SessionBuffer(
            tmp_path / "sessions", state_dir=tmp_path / "state", retain_sessions=False, debug=True
        )
        buffer.append("conv1", "user", "Hello")
        assert (tmp_path / "sessions" / "conv1.jsonl").exists()

        buffer.mark_consolidated(["conv1"])

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert not (tmp_path / "sessions" / "archive").exists()
        assert buffer.pending_count == 0

    def test_retain_sessions_true_archives(self, tmp_path):
        """With retain_sessions=True + retention_dir, mark_consolidated moves the JSONL."""
        buffer = SessionBuffer(
            tmp_path / "sessions", state_dir=tmp_path / "state", retain_sessions=True, debug=True
        )
        buffer.append("conv1", "user", "Hello")

        retention = tmp_path / "archive"
        buffer.mark_consolidated(["conv1"], retention_dir=retention)

        assert not (tmp_path / "sessions" / "conv1.jsonl").exists()
        assert (retention / "conv1.jsonl").exists()

    def test_speaker_tracking(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert buffer.get_session_state("conv1") == "new"
        assert buffer.get_speaker("conv1") is None
        assert buffer.get_speaker_id("conv1") is None

        buffer.set_speaker("conv1", "spk_abc", "Alex")
        assert buffer.get_speaker("conv1") == "Alex"
        assert buffer.get_speaker_id("conv1") == "spk_abc"
        assert buffer.get_session_state("conv1") == "identified"

    def test_speaker_in_transcript(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.set_speaker("conv1", "spk_abc", "Alex")
        buffer.append("conv1", "user", "I live in Amsterdam")

        pending = buffer.get_pending()
        # Production format is [user] / [assistant] markers; speaker name
        # is bound via the {speaker_context} prompt directive, not inlined
        # in the transcript.  speaker_id continues to flow on the pending dict.
        assert "[user] I live in Amsterdam" in pending[0]["transcript"]
        assert pending[0]["speaker_id"] == "spk_abc"

    def _setup_daily(self, tmp_path, monkeypatch, passphrase="pw"):
        """Install a daily age identity so the envelope-encrypt path engages."""
        from paramem.backup.key_store import (
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
        monkeypatch.setenv("PARAMEM_DAILY_PASSPHRASE", passphrase)
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()
        return ident

    def test_snapshot_save_and_restore(self, tmp_path, monkeypatch):
        self._setup_daily(tmp_path, monkeypatch)

        buf1 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.set_speaker("conv1", "spk_abc", "Alex")
        buf1.append("conv1", "user", "I live in Amsterdam")
        buf1.append("conv1", "assistant", "That's nice!")
        assert buf1.save_snapshot()
        assert (tmp_path / "sessions" / "session_snapshot.enc").exists()

        # Snapshot body must be an age envelope (the current posture).
        from paramem.backup.age_envelope import AGE_MAGIC

        body = (tmp_path / "sessions" / "session_snapshot.enc").read_bytes()
        assert body.startswith(AGE_MAGIC), (
            "session snapshot must land as an age envelope under the daily posture"
        )

        # Restore into a fresh buffer.
        buf2 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert buf2.load_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

        pending = buf2.get_pending()
        assert len(pending) == 1
        assert "[user] I live in Amsterdam" in pending[0]["transcript"]
        assert pending[0]["speaker_id"] == "spk_abc"
        assert buf2.get_speaker("conv1") == "Alex"

    def test_snapshot_corrupted_envelope_discarded(self, tmp_path, monkeypatch):
        """Tampered snapshot → DecryptError caught → file unlinked, buffer empty."""
        self._setup_daily(tmp_path, monkeypatch)

        buf1 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.append("conv1", "user", "Secret data")
        buf1.save_snapshot()

        # Tamper: zero out bytes past the age header.
        snap_path = tmp_path / "sessions" / "session_snapshot.enc"
        raw = snap_path.read_bytes()
        snap_path.write_bytes(raw[:80] + bytes(len(raw) - 80))

        buf2 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert not buf2.load_snapshot()
        assert not snap_path.exists(), "corrupted snapshot must be unlinked on load failure"
        assert buf2.pending_count == 0

    def test_snapshot_deleted_on_successful_restore(self, tmp_path, monkeypatch):
        self._setup_daily(tmp_path, monkeypatch)

        buf1 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.append("conv1", "user", "Hello")
        buf1.save_snapshot()
        assert (tmp_path / "sessions" / "session_snapshot.enc").exists()

        buf2 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buf2.load_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

    def test_snapshot_empty_buffer_no_file(self, tmp_path, monkeypatch):
        self._setup_daily(tmp_path, monkeypatch)

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert buffer.save_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

    def test_snapshot_no_op_when_no_keys_loaded(self, tmp_path, monkeypatch):
        """Security OFF → save returns False; no snapshot file is written.
        Operator is not silently trusting a plaintext snapshot path."""
        # Explicitly clear any inherited env + point daily path at a missing file.
        monkeypatch.delenv("PARAMEM_DAILY_PASSPHRASE", raising=False)
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "state that would have been saved")
        assert buffer.save_snapshot() is False
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

    def test_snapshot_load_preserves_file_when_keys_absent(self, tmp_path, monkeypatch, caplog):
        """Snapshot file present but no key material loaded — must NOT unlink
        (operator may restore the key and recover), and must log a WARN."""
        import logging

        # First, write a snapshot with keys loaded.
        self._setup_daily(tmp_path, monkeypatch)
        buf1 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buf1.append("conv1", "user", "important mid-turn state")
        buf1.save_snapshot()
        snap_path = tmp_path / "sessions" / "session_snapshot.enc"
        assert snap_path.exists()

        # Now simulate "operator retired the key"
        from paramem.backup.key_store import _clear_daily_identity_cache

        monkeypatch.delenv("PARAMEM_DAILY_PASSPHRASE", raising=False)
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        _clear_daily_identity_cache()

        buf2 = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        with caplog.at_level(logging.WARNING, logger="paramem.server.session_buffer"):
            assert buf2.load_snapshot() is False
        # File must still be there — operator's chance to recover it.
        assert snap_path.exists(), (
            "snapshot must NOT be unlinked when keys are absent — operator may "
            "restore the key and recover"
        )


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
                "graph1": {"recurrence_count": 3},
                "graph3": {"recurrence_count": 1},
            },
        }
        path = tmp_path / "key_metadata.json"
        _atomic_json_write(metadata, path)

        loaded = _load_key_metadata(path)
        assert loaded["cycle_count"] == 5
        assert "graph1" in loaded["promoted_keys"]
        assert loaded["keys"]["graph3"]["recurrence_count"] == 1

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
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        return RoutingPlan(
            steps=[RoutingStep(adapter_name=a, keys_to_probe=list(k)) for a, k in steps],
            strategy="direct",
            intent=Intent.PERSONAL,
        )

    def _make_model(self, adapter_names):
        """Stub model with peft_config for the given adapter names."""
        model = MagicMock()
        model.peft_config = {name: MagicMock() for name in adapter_names}
        return model

    def test_dispatches_to_grouped_probe_with_correct_groups(self, monkeypatch, tmp_path):
        """_probe_and_reason builds keys_by_adapter in step order and passes
        them through to MemoryStore.probe → WeightMemorySource.probe in train
        mode → probe_keys_grouped_by_adapter."""
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

        monkeypatch.setattr(
            "paramem.memory.probe.probe_keys_grouped_by_adapter",
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
            memory_store=_MS(replay_enabled=False),
        )

        assert "keys_by_adapter" in captured, "probe_keys_grouped_by_adapter was not called"
        kba = captured["keys_by_adapter"]
        # Both groups present.
        assert list(kba.keys()) == ["procedural", "episodic"], (
            f"Expected ['procedural', 'episodic'], got {list(kba.keys())}"
        )
        assert kba["procedural"] == ["p1", "p2"]
        assert kba["episodic"] == ["e1"]

    def test_interim_episodic_facts_reach_prompt(self, monkeypatch, tmp_path):
        """Regression: facts probed under ``episodic_interim_<stamp>`` must
        appear in the augmented_text under the ``[Recent knowledge]`` layer.

        Before fix (R2): the hard-coded layer-iteration loop only checked
        ``["procedural", "episodic", "semantic"]`` and silently dropped any
        ``episodic_interim_<stamp>`` bucket from layers — so the cycle's
        freshly trained interim facts (attribute keys included) never
        reached Mistral's prompt despite ``Total recalled: N facts`` showing
        them as successfully probed.
        """
        from paramem.server.config import ServerConfig, VoiceConfig

        captured = {}

        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    results[k] = {
                        "key": k,
                        "fact_text": f"Mara has_attr_{k} value_{k}",
                        "confidence": 1.0,
                    }
            return results

        monkeypatch.setattr(
            "paramem.memory.probe.probe_keys_grouped_by_adapter",
            fake_grouped,
        )
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
            lambda model, tokenizer, prompt, **kwargs: "stub answer",
        )

        # Capture the augmented text reaching _build_messages — that's the
        # exact string handed to the chat template before tokenization.
        def capture_augmented(text, history, system_prompt, tokenizer):
            captured["augmented_text"] = text
            return [{"role": "user", "content": text}]

        monkeypatch.setattr(
            "paramem.server.inference._build_messages",
            capture_augmented,
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"

        model = self._make_model(["episodic", "procedural", "episodic_interim_20260516T1200"])
        monkeypatch.setattr(
            "paramem.server.inference.PeftModel",
            type(None),
            raising=False,
        )

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are an assistant.")
        config = ServerConfig()
        config.voice = VoiceConfig(prompt_file=str(prompt_file))

        plan = self._make_plan(
            [
                ("procedural", ["p1"]),
                ("episodic_interim_20260516T1200", ["phone_key", "email_key"]),
            ]
        )

        from paramem.server.inference import _probe_and_reason

        _probe_and_reason(
            text="What is my phone number?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=_MS(replay_enabled=False),
        )

        assert "augmented_text" in captured, "_build_messages was not called"
        text = captured["augmented_text"]

        # Procedural facts present.
        assert "Mara has_attr_p1 value_p1" in text, (
            f"procedural fact missing from prompt; augmented_text:\n{text}"
        )
        # Interim-episodic facts present — this is the regression check.
        assert "Mara has_attr_phone_key value_phone_key" in text, (
            f"episodic_interim phone fact missing from prompt; augmented_text:\n{text}"
        )
        assert "Mara has_attr_email_key value_email_key" in text, (
            f"episodic_interim email fact missing from prompt; augmented_text:\n{text}"
        )
        # Layer label is "Recent knowledge" (the canonical episodic-tier label),
        # not the bare adapter name — multiple interim slots collapse under one
        # heading.
        assert "[Recent knowledge]" in text, (
            f"interim facts should appear under [Recent knowledge]; got:\n{text}"
        )
        assert "[episodic_interim_20260516T1200]" not in text, (
            "interim adapter name should NOT leak as a section heading; "
            "merge them under [Recent knowledge] instead"
        )


# ---------------------------------------------------------------------------
# _build_store_contents — store-free builder (phase-2)
# ---------------------------------------------------------------------------


class TestBuildStoreContents:
    """_build_store_contents builds registry/entries/bookkeeping off-store."""

    def _make_config(self, tmp_path):
        """Minimal config stub sufficient for _build_store_contents."""
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        cfg.key_metadata_path = tmp_path / "key_metadata.json"
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 1
        cfg.inference.preload_cache = False
        return cfg

    def test_returns_four_tuple(self, tmp_path) -> None:
        """_build_store_contents returns (entries, registry, bookkeeping, stats)."""
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        cfg = self._make_config(tmp_path)
        result = _build_store_contents(cfg, model=None, tokenizer=None)
        assert len(result) == 4, "expected 4-tuple"
        new_e, new_r, new_b, stats = result
        assert isinstance(new_e, dict)
        assert isinstance(new_r, dict)
        assert isinstance(new_b, dict)
        assert isinstance(stats, dict)

    def test_stats_has_expected_keys(self, tmp_path) -> None:
        """stats dict carries boot_degraded and store_load_degraded."""
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        cfg = self._make_config(tmp_path)
        _, _, _, stats = _build_store_contents(cfg, model=None, tokenizer=None)
        assert "boot_degraded" in stats
        assert "store_load_degraded" in stats

    def test_preload_cache_off_entries_empty(self, tmp_path) -> None:
        """When preload_cache=False, new_entries is empty (intentional opt-out)."""
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        cfg = self._make_config(tmp_path)
        cfg.inference.preload_cache = False
        new_e, _, _, stats = _build_store_contents(cfg, model=None, tokenizer=None)
        assert new_e == {}, "entries must be empty when preload_cache=False"
        assert stats["boot_degraded"] is None

    def test_does_not_mutate_any_live_store(self, tmp_path) -> None:
        """_build_store_contents must not touch the live MemoryStore singleton."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        live = MemoryStore()
        live.put("episodic", "sentinel_key", {"key": "sentinel_key"})

        cfg = self._make_config(tmp_path)
        _build_store_contents(cfg, model=None, tokenizer=None)

        # The live store must be untouched.
        assert live.get("sentinel_key") is not None, "live store mutated by builder"

    def test_should_abort_accepted(self, tmp_path) -> None:
        """_build_store_contents accepts should_abort without raising."""
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        cfg = self._make_config(tmp_path)
        cfg.inference.preload_cache = True  # activate probe path
        cfg.consolidation.mode = "simulate"

        # should_abort=True; with simulate mode and no graph.json files the
        # probe returns quickly regardless, but the call must not raise.
        result = _build_store_contents(cfg, model=None, tokenizer=None, should_abort=lambda: True)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# _hydrate_memory_store_in_place — degraded-build swap guard (regression)
# ---------------------------------------------------------------------------


class TestHydrateMemoryStoreSwapGuard:
    """Degraded builder must not wipe a populated live store."""

    def _make_config(self, tmp_path):
        """Minimal config stub sufficient for _build_store_contents."""
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        cfg.key_metadata_path = tmp_path / "key_metadata.json"
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 1
        cfg.inference.preload_cache = False
        return cfg

    def test_degraded_build_does_not_swap(self, tmp_path) -> None:
        """When the builder returns store_load_degraded=True, swap is skipped.

        The live store's registry/bookkeeping must survive intact; only
        _state degraded flags are updated.
        """
        from unittest.mock import patch

        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        # Pre-populate the live store with a sentinel entry.
        live = MemoryStore()
        live.put("episodic", "pre_existing_key", {"key": "pre_existing_key", "tier": "episodic"})

        cfg = self._make_config(tmp_path)

        # Simulate read_registries_from_disk raising to trigger store_load_degraded.
        with patch(
            "paramem.memory.store.MemoryStore.read_registries_from_disk",
            side_effect=OSError("simulated disk failure"),
        ):
            _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)

        # The pre-existing entry must still be present — swap must not have run.
        assert live.get("pre_existing_key") is not None, (
            "degraded build wiped the live store: pre-existing entry lost"
        )

    def test_legitimate_empty_registry_does_swap(self, tmp_path) -> None:
        """A successful build with an empty registry (store_load_degraded=False) swaps.

        This verifies the guard is on the failure flag, not on len(registry)==0.
        """
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        # Pre-populate the live store with a sentinel entry.
        live = MemoryStore()
        live.put("episodic", "old_key", {"key": "old_key", "tier": "episodic"})

        cfg = self._make_config(tmp_path)
        # preload_cache=False + empty registry → successful empty build, should swap.
        cfg.inference.preload_cache = False

        _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)

        # The old entry must be gone — the swap replaced the store with the empty build.
        assert live.get("old_key") is None, (
            "legitimate empty build did not swap: old entry still present"
        )

    def test_degraded_build_sets_state_flag(self, tmp_path) -> None:
        """_state['store_load_degraded'] is set True when builder degrades."""
        from unittest.mock import patch

        import paramem.server.app as app_module
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        live = MemoryStore()
        cfg = self._make_config(tmp_path)

        original_state = app_module._state.copy()
        try:
            with patch(
                "paramem.memory.store.MemoryStore.read_registries_from_disk",
                side_effect=OSError("simulated disk failure"),
            ):
                _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)

            assert app_module._state["store_load_degraded"] is True, (
                "_state['store_load_degraded'] not set True after degraded build"
            )
        finally:
            # Restore _state so we do not leak into other tests.
            app_module._state.update(original_state)
