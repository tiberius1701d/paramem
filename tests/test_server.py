"""Unit tests for the ParaMem server modules (no GPU required)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.graph.prompts import prompt_overrides
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
        # Epochs and gradient accumulation are derived per fold from the
        # key-triple count via budget_for -- unconditional and unclamped,
        # no operator ceiling -- so they are not asserted here.
        assert tc.batch_size == 1
        assert tc.max_seq_length == 1024

    def test_consolidation_config(self):
        config = ServerConfig()
        cc = config.consolidation_config
        assert cc.indexed_key_replay is True
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
            "  cloud:\n"
            "    provider: anthropic\n"
            "    model: claude-sonnet\n"
            "    api_key: ${TEST_API_KEY}\n"
        )
        config = load_server_config(config_file)
        assert config.cloud_agent.api_key == "sk-secret-123"
        assert config.cloud_agent.provider == "anthropic"

    def test_env_var_missing_uses_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        config_file = tmp_path / "server.yaml"
        config_file.write_text("agents:\n  cloud:\n    api_key: ${NONEXISTENT_VAR}\n")
        config = load_server_config(config_file)
        assert config.cloud_agent.api_key == ""

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
        """append() mints a session_id for the conversation_id; both turns
        land in the same (single) pending session."""
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv1", "assistant", "Hi there!")

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"].startswith("conv1-")
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

        pending_before = buffer.get_pending()
        conv1_session_id = next(
            p["session_id"] for p in pending_before if p["session_id"].startswith("conv1-")
        )
        conv2_session_id = next(
            p["session_id"] for p in pending_before if p["session_id"].startswith("conv2-")
        )

        buffer.mark_consolidated([conv1_session_id])

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"] == conv2_session_id

    def test_mark_consolidated_debug_archives(self, tmp_path):
        """With debug=True + retention_dir supplied, mark_consolidated moves the JSONL."""
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=True)
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]

        retention = tmp_path / "archive"
        buffer.mark_consolidated([session_id], retention_dir=retention)

        assert (retention / f"{session_id}.jsonl").exists()
        assert not (tmp_path / "sessions" / f"{session_id}.jsonl").exists()

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
        session_id = buffer.get_pending()[0]["session_id"]

        path = tmp_path / "sessions" / f"{session_id}.jsonl"
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
        session_id = buffer.get_pending()[0]["session_id"]

        assert (tmp_path / "sessions" / f"{session_id}.jsonl").exists()
        assert buffer.pending_count == 1
        assert len(buffer.get_pending()) == 1

    def test_retain_sessions_false_deletes(self, tmp_path):
        buffer = SessionBuffer(
            tmp_path / "sessions", state_dir=tmp_path / "state", retain_sessions=False, debug=True
        )
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]
        assert (tmp_path / "sessions" / f"{session_id}.jsonl").exists()

        buffer.mark_consolidated([session_id])

        assert not (tmp_path / "sessions" / f"{session_id}.jsonl").exists()
        assert not (tmp_path / "sessions" / "archive").exists()
        assert buffer.pending_count == 0

    def test_retain_sessions_true_archives(self, tmp_path):
        """With retain_sessions=True + retention_dir, mark_consolidated moves the JSONL."""
        buffer = SessionBuffer(
            tmp_path / "sessions", state_dir=tmp_path / "state", retain_sessions=True, debug=True
        )
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]

        retention = tmp_path / "archive"
        buffer.mark_consolidated([session_id], retention_dir=retention)

        assert not (tmp_path / "sessions" / f"{session_id}.jsonl").exists()
        assert (retention / f"{session_id}.jsonl").exists()

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

    def test_get_conversation_turns_conversational(self, tmp_path):
        """Regression: a direct ``_turns[conversation_id]`` lookup is dead for
        the conversational case — ``append`` always mints a distinct
        session_id (``f"{conversation_id}-{timestamp}-{rand}"``), so
        ``get_conversation_turns`` must route through the ``_open`` indirection
        the same way ``append`` resolves it."""
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv1", "assistant", "Hi there!")

        turns = buffer.get_conversation_turns("conv1")
        assert [t["text"] for t in turns] == ["Hello", "Hi there!"]

    def test_get_conversation_turns_document_chunk_path(self, tmp_path):
        """Document-chunk sessions use session_id == the routing handle
        directly (``append_document_chunk`` never rotates) — the fallback
        to treating the id as a session id directly must keep this path
        working."""
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        buffer.set_speaker("doc-1-c000", "speaker0", "Alex")
        buffer.set_document_metadata("doc-1-c000", doc_id="doc-1", chunk_count=1)
        buffer.append_document_chunk("doc-1-c000", "user", "chunk text")

        turns = buffer.get_conversation_turns("doc-1-c000")
        assert [t["text"] for t in turns] == ["chunk text"]

    def test_get_conversation_turns_unknown_conversation_empty(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state")
        assert buffer.get_conversation_turns("never-seen") == []

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

    def test_snapshot_missing_open_key_discarded(self, tmp_path, monkeypatch):
        """A validly-encrypted payload lacking "open" is treated as corrupted:
        strict ``payload["open"]`` raises KeyError, caught by the same
        except that handles a bad envelope — unlink + discard (cold start),
        not a silent ``{}`` fill."""
        import json

        from paramem.backup.encryption import _atomic_write_bytes, envelope_encrypt_bytes

        self._setup_daily(tmp_path, monkeypatch)

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        snap_path = sessions_dir / "session_snapshot.enc"

        # Valid envelope, but the payload predates the "open" key.
        payload = {"turns": {"conv1": [{"role": "user", "text": "hi"}]}, "sessions": {}}
        _atomic_write_bytes(snap_path, envelope_encrypt_bytes(json.dumps(payload).encode()))

        buf = SessionBuffer(sessions_dir, state_dir=tmp_path / "state")
        assert not buf.load_snapshot()
        assert not snap_path.exists(), "payload missing 'open' must be discarded, not tolerated"
        assert buf.pending_count == 0

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
        from paramem.backup.encryption import write_infra_json

        path = tmp_path / "test.json"
        write_infra_json(path, {"key": "value"})

        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}
        assert not (tmp_path / "test.tmp").exists()

    def test_atomic_json_write_list(self, tmp_path):
        from paramem.backup.encryption import write_infra_json

        path = tmp_path / "test.json"
        write_infra_json(path, [1, 2, 3])

        with open(path) as f:
            data = json.load(f)
        assert data == [1, 2, 3]

    def test_load_key_metadata_missing(self, tmp_path):
        from paramem.server.consolidation import _load_key_metadata

        result = _load_key_metadata(tmp_path / "nonexistent.json")
        assert result is None

    def test_key_metadata_round_trip(self, tmp_path):
        from paramem.backup.encryption import write_infra_json
        from paramem.server.consolidation import _load_key_metadata

        metadata = {
            "cycle_count": 5,
            "promoted_keys": ["graph1", "graph2"],
            "keys": {
                "graph1": {"reinforcement_count": 3},
                "graph3": {"reinforcement_count": 1},
            },
        }
        path = tmp_path / "key_metadata.json"
        write_infra_json(path, metadata)

        loaded = _load_key_metadata(path)
        assert loaded["cycle_count"] == 5
        assert "graph1" in loaded["promoted_keys"]
        assert loaded["keys"]["graph3"]["reinforcement_count"] == 1


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

    def test_dispatches_to_grouped_probe_with_correct_groups(self, monkeypatch):
        """_probe_and_reason builds keys_by_adapter in step order and passes
        them through to MemoryStore.probe → WeightMemorySource.probe in train
        mode → probe_keys_grouped_by_adapter."""
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
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path, cached=False: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential",
            lambda text, **kwargs: False,
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

        config = ServerConfig()

        plan = self._make_plan(
            [
                ("procedural", ["p1", "p2"]),
                ("episodic", ["e1"]),
            ]
        )

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
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

    def test_per_turn_probe_requests_cached_registry(self, monkeypatch):
        """_probe_and_reason's on-miss source build must opt into the
        process-wide simhash-registry cache (``cached_registry=True``) —
        the per-turn probe is the one caller allowed to skip the disk
        re-walk; hydration callers (``app._build_store_contents``,
        ``ConsolidationLoop._hydrate_store_for_fold``) stay on the
        disk-truth default and are not this call site."""
        captured = {}

        class _FakeSource:
            def probe(self, keys_by_adapter, should_abort=None):
                results = {}
                for keys in keys_by_adapter.values():
                    for k in keys:
                        results[k] = {"key": k, "fact_text": f"fact about {k}", "confidence": 1.0}
                return results

        def fake_build_memory_source(**kwargs):
            captured.update(kwargs)
            return _FakeSource()

        monkeypatch.setattr(
            "paramem.memory.source.build_memory_source",
            fake_build_memory_source,
        )
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            lambda model, name: None,
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential",
            lambda text, **kwargs: False,
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

        model = self._make_model(["episodic"])

        config = ServerConfig()

        plan = self._make_plan([("episodic", ["e1"])])

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
            _probe_and_reason(
                text="What do I like?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=_MS(replay_enabled=False),
            )

        assert captured.get("cached_registry") is True

    def test_interim_episodic_facts_reach_prompt(self, monkeypatch):
        """Regression: facts probed under ``episodic_interim_<stamp>`` must
        appear in the augmented_text under the ``[Recent knowledge]`` layer.

        Before the fix, the hard-coded layer-iteration loop only checked
        ``["procedural", "episodic", "semantic"]`` and silently dropped any
        ``episodic_interim_<stamp>`` bucket from layers — so the cycle's
        freshly trained interim facts (attribute keys included) never
        reached Mistral's prompt despite ``Total recalled: N facts`` showing
        them as successfully probed.
        """
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
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path, cached=False: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential",
            lambda text, **kwargs: False,
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

        config = ServerConfig()

        plan = self._make_plan(
            [
                ("procedural", ["p1"]),
                ("episodic_interim_20260516T1200", ["phone_key", "email_key"]),
            ]
        )

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
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

    def _stub_common(self, monkeypatch, *, fact_prefix: str):
        """Shared stubbing for the two identity-prompt tests below — mirrors
        the pattern used by the two tests above, factored out since both new
        tests need identical mocking with only the ``speaker``/``speaker_id``
        arguments differing."""

        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    results[k] = {
                        "key": k,
                        "fact_text": f"{fact_prefix} likes {k}",
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
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path, cached=False: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential",
            lambda text, **kwargs: False,
        )
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer",
            lambda model, tokenizer, prompt, **kwargs: "final answer",
        )

    def test_local_reasoning_prompt_carries_speaker_token_not_name(self, monkeypatch):
        """The system prompt reaching _build_messages for a named speaker
        contains the raw speaker{N} token and ZERO occurrences of the display
        name — identity stays in token space on the LOCAL reasoning leg."""
        self._stub_common(monkeypatch, fact_prefix="speaker0")

        captured = {}

        def capture_messages(text, history, system_prompt, tokenizer):
            captured["system_prompt"] = system_prompt
            captured["augmented_text"] = text
            return [{"role": "user", "content": text}]

        monkeypatch.setattr("paramem.server.inference._build_messages", capture_messages)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self._make_model(["episodic"])

        config = ServerConfig()

        plan = self._make_plan([("episodic", ["e1"])])

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
            _probe_and_reason(
                text="What do I like?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=_MS(replay_enabled=False),
                speaker="Alice",
                speaker_id="speaker0",
            )

        assert "system_prompt" in captured, "_build_messages was not called"
        system_prompt = captured["system_prompt"]
        assert "speaker0" in system_prompt
        assert "Alice" not in system_prompt
        # The assembled reasoning prompt as a whole (prefix + recalled facts)
        # carries the token throughout and never the display name.
        full_prompt = system_prompt + captured["augmented_text"]
        assert "speaker0" in full_prompt
        assert "Alice" not in full_prompt

    def test_anonymous_speaker_prefix_from_speaker_id(self, monkeypatch):
        """Re-spec (B-form prefix from speaker_id presence): the local
        system-prompt identity line is now gated on ``speaker_id`` alone —
        anonymous/undisclosed speakers included.  ``speaker=None`` (the
        display name, still absent pre-disclosure) no longer suppresses it;
        only a ``speaker_id`` of ``None`` would.  The raw token is the
        payload — never the display name, which stays absent from the
        prompt regardless."""
        self._stub_common(monkeypatch, fact_prefix="speaker3")

        captured = {}

        def capture_messages(text, history, system_prompt, tokenizer):
            captured["system_prompt"] = system_prompt
            return [{"role": "user", "content": text}]

        monkeypatch.setattr("paramem.server.inference._build_messages", capture_messages)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self._make_model(["episodic"])

        config = ServerConfig()

        plan = self._make_plan([("episodic", ["e1"])])

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
            _probe_and_reason(
                text="What do I like?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=_MS(replay_enabled=False),
                speaker=None,
                speaker_id="speaker3",
            )

        assert "system_prompt" in captured, "_build_messages was not called"
        system_prompt = captured["system_prompt"]
        assert "You are speaking with speaker3." in system_prompt


class TestBaseModelAnswerSystemPrompt:
    """_base_model_answer's system-prompt assembly, mirrored against
    _probe_and_reason's via the shared ``_build_system_prompt`` helper
    (previously two byte-identical inline blocks with zero coverage on this
    leg — a drift between the two would have been invisible to CI)."""

    def test_speaker_token_and_language_reach_system_prompt(self, monkeypatch):
        captured = {}

        def capture_messages(text, history, system_prompt, tokenizer):
            captured["system_prompt"] = system_prompt
            return [{"role": "user", "content": text}]

        monkeypatch.setattr("paramem.server.inference._build_messages", capture_messages)
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer",
            lambda model, tokenizer, prompt, **kwargs: "a plain answer",
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = MagicMock()

        config = ServerConfig()

        from paramem.server.inference import _base_model_answer

        with prompt_overrides({"serving_system.txt": "Base voice prompt."}):
            result = _base_model_answer(
                text="hello",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                speaker="Alice",
                speaker_id="speaker0",
                language="de",
            )

        assert "system_prompt" in captured, "_build_messages was not called"
        system_prompt = captured["system_prompt"]
        assert "speaker0" in system_prompt
        assert "Alice" not in system_prompt
        assert "Respond in German" in system_prompt
        assert "Base voice prompt." in system_prompt
        assert result.text == "a plain answer"

    def test_anonymous_speaker_prefix_from_speaker_id(self, monkeypatch):
        """Re-spec (B-form prefix from speaker_id presence): ``speaker=None``
        no longer suppresses the identity token when ``speaker_id`` is set —
        the prefix is gated on ``speaker_id`` alone, anonymous included."""
        captured = {}

        def capture_messages(text, history, system_prompt, tokenizer):
            captured["system_prompt"] = system_prompt
            return [{"role": "user", "content": text}]

        monkeypatch.setattr("paramem.server.inference._build_messages", capture_messages)
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer",
            lambda model, tokenizer, prompt, **kwargs: "a plain answer",
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = MagicMock()

        config = ServerConfig()

        from paramem.server.inference import _base_model_answer

        with prompt_overrides({"serving_system.txt": "Base voice prompt."}):
            _base_model_answer(
                text="hello",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                speaker=None,
                speaker_id="speaker3",
            )

        system_prompt = captured["system_prompt"]
        assert "You are speaking with speaker3." in system_prompt


class TestBuildMessagesAlternationDefense:
    """``_build_messages``'s same-role merge and leading-assistant strip
    (inference.py) had ZERO behavioral coverage before this change — both
    prompt-capture tests above patch ``_build_messages`` out entirely.

    ``_run_chat_turn``'s user/assistant append pair
    (``paramem/server/app.py``) is NOT wrapped in a ``try/except``: the
    "user" append happens, then the "assistant" append happens as a
    separate synchronous call with its own file write/fsync.  A failure on
    the second append (e.g. disk full) after the first already succeeded —
    "an errored request that persisted a user turn without an assistant
    reply" — leaves a non-alternating history.  Alternation is therefore
    NOT structurally guaranteed, so the defense is KEPT (not deleted) and
    pinned here directly.
    """

    def test_consecutive_same_role_turns_merged(self, monkeypatch):
        from paramem.server.inference import _build_messages

        # Bypass adapt_messages/tokenizer template resolution entirely — it's
        # a separate concern (system-role folding) from the merge/strip logic
        # under test here.
        monkeypatch.setattr(
            "paramem.server.inference.adapt_messages",
            lambda messages, tokenizer: messages,
        )

        history = [
            {"role": "user", "text": "first"},
            {"role": "user", "text": "second"},
            {"role": "assistant", "text": "reply"},
        ]
        messages = _build_messages("question", history, "system prompt", tokenizer=MagicMock())

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[1]["content"] == "first\nsecond"
        assert messages[-1]["content"] == "question"

    def test_leading_assistant_turn_stripped(self, monkeypatch):
        from paramem.server.inference import _build_messages

        monkeypatch.setattr(
            "paramem.server.inference.adapt_messages",
            lambda messages, tokenizer: messages,
        )

        history = [
            {"role": "assistant", "text": "orphaned reply"},
            {"role": "user", "text": "hi"},
        ]
        messages = _build_messages("question", history, "system prompt", tokenizer=MagicMock())

        # The leading assistant turn is dropped; the surviving user turn ends
        # up last, so the current-turn text is appended onto it (see
        # _build_messages's final if/else) rather than becoming a new message.
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[-1]["content"] == "hi\nquestion"
        assert not any("orphaned reply" in m["content"] for m in messages)


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

    def test_meta_unbookkept_counts_active_keys_without_bookkeeping(self, tmp_path) -> None:
        """meta_unbookkept counts active registry keys with no bookkeeping row —
        the inverse of meta_orphaned (a bookkeeping row with no registry key).
        Detect-and-surface only: the build must still succeed (no raise, no
        degrade)."""
        from paramem.server.app import _build_store_contents
        from paramem.training.key_registry import KeyRegistry

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        reg = KeyRegistry()
        reg.add("graph1")
        reg.save(tmp_path / "episodic" / "indexed_key_registry.json")

        # key_metadata.json carries no row for graph1 -- registry/bookkeeping
        # divergence at boot.
        (tmp_path / "key_metadata.json").write_text(json.dumps({"keys": {}}))

        cfg = self._make_config(tmp_path)
        _, _, _, stats = _build_store_contents(cfg, model=None, tokenizer=None)

        assert stats["meta_unbookkept"] == 1
        assert stats["store_load_degraded"] is False

    def test_meta_unbookkept_zero_when_every_active_key_has_bookkeeping(self, tmp_path) -> None:
        """A registry/bookkeeping pair that agrees reports meta_unbookkept=0."""
        from paramem.server.app import _build_store_contents
        from paramem.training.key_registry import KeyRegistry

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        reg = KeyRegistry()
        reg.add("graph1")
        reg.save(tmp_path / "episodic" / "indexed_key_registry.json")

        (tmp_path / "key_metadata.json").write_text(
            json.dumps(
                {
                    "keys": {
                        "graph1": {
                            "speaker_id": "S0",
                            "relation_type": "factual",
                            "reinforcement_count": 1,
                            "last_reinforced_cycle": 0,
                            "last_seen": "",
                            "first_seen": "",
                        }
                    }
                }
            )
        )

        cfg = self._make_config(tmp_path)
        _, _, _, stats = _build_store_contents(cfg, model=None, tokenizer=None)

        assert stats["meta_unbookkept"] == 0


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
