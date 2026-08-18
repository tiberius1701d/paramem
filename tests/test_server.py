"""Unit tests for the ParaMem server modules (no GPU required)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.prompts import prompt_overrides
from paramem.memory.store import MemoryStore as _MS
from paramem.server.config import MODEL_REGISTRY, ServerConfig, load_server_config
from paramem.server.escalation import detect_escalation
from paramem.server.session_buffer import SessionBuffer
from tests._serving_door import seed_live_door_fingerprints

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
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv1", "assistant", "Hi there!")

        pending = buffer.get_pending()
        assert len(pending) == 1
        assert pending[0]["session_id"].startswith("conv1-")
        assert "[user] Hello" in pending[0]["transcript"]
        assert "[assistant] Hi there!" in pending[0]["transcript"]

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
        buffer = SessionBuffer(tmp_path / "sessions", debug=True)
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]

        retention = tmp_path / "archive"
        buffer.mark_consolidated([session_id], retention_dir=retention)

        assert (retention / f"{session_id}.jsonl").exists()
        assert not (tmp_path / "sessions" / f"{session_id}.jsonl").exists()

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
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]

        assert (tmp_path / "sessions" / f"{session_id}.jsonl").exists()
        assert buffer.pending_count == 1
        assert len(buffer.get_pending()) == 1

    def test_retain_sessions_false_deletes(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions", retain_sessions=False, debug=True)
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]
        assert (tmp_path / "sessions" / f"{session_id}.jsonl").exists()

        buffer.mark_consolidated([session_id])

        assert not (tmp_path / "sessions" / f"{session_id}.jsonl").exists()
        assert not (tmp_path / "sessions" / "archive").exists()
        assert buffer.pending_count == 0

    def test_retain_sessions_true_archives(self, tmp_path):
        """With retain_sessions=True + retention_dir, mark_consolidated moves the JSONL."""
        buffer = SessionBuffer(tmp_path / "sessions", retain_sessions=True, debug=True)
        buffer.append("conv1", "user", "Hello")
        session_id = buffer.get_pending()[0]["session_id"]

        retention = tmp_path / "archive"
        buffer.mark_consolidated([session_id], retention_dir=retention)

        assert not (tmp_path / "sessions" / f"{session_id}.jsonl").exists()
        assert (retention / f"{session_id}.jsonl").exists()

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
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "Hello")
        buffer.append("conv1", "assistant", "Hi there!")

        turns = buffer.get_conversation_turns("conv1")
        assert [t["text"] for t in turns] == ["Hello", "Hi there!"]

    def test_get_conversation_turns_document_chunk_path(self, tmp_path):
        """Document-chunk sessions use session_id == the routing handle
        directly (``append_document_chunk`` never rotates) — the fallback
        to treating the id as a session id directly must keep this path
        working."""
        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.set_speaker("doc-1-c000", "speaker0", "Alex")
        buffer.set_document_metadata("doc-1-c000", doc_id="doc-1", chunk_count=1)
        buffer.append_document_chunk("doc-1-c000", "user", "chunk text")

        turns = buffer.get_conversation_turns("doc-1-c000")
        assert [t["text"] for t in turns] == ["chunk text"]

    def test_get_conversation_turns_unknown_conversation_empty(self, tmp_path):
        buffer = SessionBuffer(tmp_path / "sessions")
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

        buf1 = SessionBuffer(tmp_path / "sessions")
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
        buf2 = SessionBuffer(tmp_path / "sessions")
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

        buf1 = SessionBuffer(tmp_path / "sessions")
        buf1.append("conv1", "user", "Secret data")
        buf1.save_snapshot()

        # Tamper: zero out bytes past the age header.
        snap_path = tmp_path / "sessions" / "session_snapshot.enc"
        raw = snap_path.read_bytes()
        snap_path.write_bytes(raw[:80] + bytes(len(raw) - 80))

        buf2 = SessionBuffer(tmp_path / "sessions")
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

        buf = SessionBuffer(sessions_dir)
        assert not buf.load_snapshot()
        assert not snap_path.exists(), "payload missing 'open' must be discarded, not tolerated"
        assert buf.pending_count == 0

    def test_snapshot_deleted_on_successful_restore(self, tmp_path, monkeypatch):
        self._setup_daily(tmp_path, monkeypatch)

        buf1 = SessionBuffer(tmp_path / "sessions")
        buf1.append("conv1", "user", "Hello")
        buf1.save_snapshot()
        assert (tmp_path / "sessions" / "session_snapshot.enc").exists()

        buf2 = SessionBuffer(tmp_path / "sessions")
        buf2.load_snapshot()
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

    def test_snapshot_empty_buffer_no_file(self, tmp_path, monkeypatch):
        self._setup_daily(tmp_path, monkeypatch)

        buffer = SessionBuffer(tmp_path / "sessions")
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

        buffer = SessionBuffer(tmp_path / "sessions")
        buffer.append("conv1", "user", "state that would have been saved")
        assert buffer.save_snapshot() is False
        assert not (tmp_path / "sessions" / "session_snapshot.enc").exists()

    def test_snapshot_load_preserves_file_when_keys_absent(self, tmp_path, monkeypatch, caplog):
        """Snapshot file present but no key material loaded — must NOT unlink
        (operator may restore the key and recover), and must log a WARN."""
        import logging

        # First, write a snapshot with keys loaded.
        self._setup_daily(tmp_path, monkeypatch)
        buf1 = SessionBuffer(tmp_path / "sessions")
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

        buf2 = SessionBuffer(tmp_path / "sessions")
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

    def _seeded_store(self, plan):
        """A real ``MemoryStore`` bookkeeping-seeded for every key in *plan*.

        The date-group selection stage (default ON) reads
        ``bookkeeping_for_key`` directly for every key a routing plan names
        to probe — the every-known-key-has-a-row invariant means a real
        probe never reaches this stage with an unbookkept key, so the test
        double must seed one row per key rather than leave the store empty."""
        store = _MS()
        for step in plan.steps:
            for key in step.keys_to_probe:
                store.set_bookkeeping(
                    key,
                    speaker_id="speaker0",
                    relation_type="factual",
                    first_seen="",
                    promoted=False,
                )
        return store

    def test_dispatches_to_grouped_probe_with_correct_groups(self, monkeypatch):
        """_probe_and_reason builds keys_by_adapter in step order and passes
        them through to MemoryStore.probe_source → WeightMemorySource.probe in
        train mode → probe_keys_grouped_by_adapter."""
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
            staticmethod(lambda path, *, cached=False: {}),
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
            lambda text, history, system_prompt: [{"role": "user", "content": text}],
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"

        model = self._make_model(["episodic", "procedural"])

        config = ServerConfig()
        # Exercise the live door directly -- the source is mocked below.
        config.inference.preload_cache = False

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
                memory_store=self._seeded_store(plan),
            )

        assert "keys_by_adapter" in captured, "probe_keys_grouped_by_adapter was not called"
        kba = captured["keys_by_adapter"]
        # Both groups present.
        assert list(kba.keys()) == ["procedural", "episodic"], (
            f"Expected ['procedural', 'episodic'], got {list(kba.keys())}"
        )
        assert kba["procedural"] == ["p1", "p2"]
        assert kba["episodic"] == ["e1"]

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

        def attribute_entry(key: str) -> dict:
            """This test's own fact shape — an attribute-flavoured triple.

            The live door verifies a source result against the fingerprint
            registered for the key, so this one builder feeds both sides:
            the stubbed source below and the seeding call further down.
            """
            return {
                "key": key,
                "subject": "Mara",
                "predicate": f"has_attr_{key}",
                "object": f"value_{key}",
            }

        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    results[k] = {
                        **attribute_entry(k),
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
            staticmethod(lambda path, *, cached=False: {}),
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
        def capture_augmented(text, history, system_prompt):
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
        # Exercise the live door directly -- the source is mocked below.
        config.inference.preload_cache = False

        plan = self._make_plan(
            [
                ("procedural", ["p1"]),
                ("episodic_interim_20260516T1200", ["phone_key", "email_key"]),
            ]
        )

        from paramem.server.inference import _probe_and_reason

        memory_store = self._seeded_store(plan)
        seed_live_door_fingerprints(memory_store, plan, entry_for=attribute_entry)

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
            _probe_and_reason(
                text="What is my phone number?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=memory_store,
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
            staticmethod(lambda path, *, cached=False: {}),
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

        def capture_messages(text, history, system_prompt):
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
                memory_store=self._seeded_store(plan),
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

        def capture_messages(text, history, system_prompt):
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
                memory_store=self._seeded_store(plan),
                speaker=None,
                speaker_id="speaker3",
            )

        assert "system_prompt" in captured, "_build_messages was not called"
        system_prompt = captured["system_prompt"]
        assert "You are speaking with speaker3." in system_prompt


class TestServingReadDoorExclusivity(TestProbeAndReasonDispatch):
    """The two serving read doors are exclusive, asserted in both
    directions: ``preload_cache=True`` never touches the model for
    probing and never builds a source; ``preload_cache=False`` never
    reads or writes the entry cache, even when the cache is populated
    with content that would answer differently."""

    def _stub_reasoning_and_prompt(self, monkeypatch, captured: dict) -> None:
        """Stub everything downstream of probing so the reasoning leg
        completes without a real model — shared by every test below."""
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential",
            lambda text, **kwargs: False,
        )

        def _fake_generate_answer(model, tokenizer, prompt, **kwargs):
            captured["generate_answer_calls"] = captured.get("generate_answer_calls", 0) + 1
            return "final answer"

        monkeypatch.setattr(
            "paramem.server.inference.generate_answer",
            _fake_generate_answer,
        )

        def capture_augmented(text, history, system_prompt):
            captured["augmented_text"] = text
            return [{"role": "user", "content": text}]

        monkeypatch.setattr(
            "paramem.server.inference._build_messages",
            capture_augmented,
        )

    def test_cache_true_builds_no_source_and_never_probes_the_model(self, monkeypatch):
        """``preload_cache=True`` over a warm cache: the reasoning leg is
        the only thing that would touch the model — probing itself never
        does. Asserted by spying on the one call the live door would have
        made (probe_keys_grouped_by_adapter, which WeightMemorySource
        wraps) and on switch_adapter (the live door's own post-probe
        restore) — neither fires — and on build_memory_source, which the
        cache arm never constructs."""
        captured: dict = {}
        self._stub_reasoning_and_prompt(monkeypatch, captured)

        grouped_probe = MagicMock()
        monkeypatch.setattr("paramem.memory.probe.probe_keys_grouped_by_adapter", grouped_probe)
        switch_adapter_mock = MagicMock()
        monkeypatch.setattr("paramem.models.loader.switch_adapter", switch_adapter_mock)
        build_source_spy = MagicMock(wraps=None)
        monkeypatch.setattr("paramem.memory.source.build_memory_source", build_source_spy)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self._make_model(["episodic"])
        config = ServerConfig()
        config.inference.preload_cache = True

        store = self._seeded_store(self._make_plan([("episodic", ["e1"])]))
        store.put(
            "episodic",
            "e1",
            {"key": "e1", "subject": "speaker0", "predicate": "likes", "object": "tea"},
        )

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
                memory_store=store,
            )

        grouped_probe.assert_not_called()
        switch_adapter_mock.assert_not_called()
        build_source_spy.assert_not_called()
        assert captured.get("generate_answer_calls") == 1, (
            "the reasoning leg is the sole model-touching call under the cache arm"
        )
        assert "speaker0 likes tea" in captured["augmented_text"]

    def test_cache_false_with_populated_cache_serves_from_source_and_leaves_cache_untouched(
        self, monkeypatch
    ):
        """``preload_cache=False`` over a POPULATED cache — the state a
        fold leaves behind: every recalled fact comes from the source, not
        the cache. Proven by seeding the cache with a triple that differs
        from the source's answer and reading which one the turn served;
        the cache is unchanged afterwards (no write-back)."""
        captured: dict = {}
        self._stub_reasoning_and_prompt(monkeypatch, captured)

        def source_entry(key: str) -> dict:
            """The triple the source answers with — and, below, the one whose
            fingerprint the store registers, since the live door verifies the
            served result against it."""
            return {"key": key, "subject": "speaker0", "predicate": "likes", "object": "coffee"}

        class _FakeSource:
            def probe(self, keys_by_adapter):
                return {
                    "e1": {
                        **source_entry("e1"),
                        "fact_text": "speaker0 likes coffee",
                    }
                }

        monkeypatch.setattr(
            "paramem.memory.source.build_memory_source",
            lambda **kwargs: _FakeSource(),
        )
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            lambda model, name: None,
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self._make_model(["episodic"])
        config = ServerConfig()
        config.inference.preload_cache = False

        plan = self._make_plan([("episodic", ["e1"])])
        store = self._seeded_store(plan)
        # The fingerprint on record is the SOURCE's fact — the live door
        # serves nothing it cannot prove, and the mirror's copy below is
        # deliberately the stale one.
        seed_live_door_fingerprints(store, plan, entry_for=source_entry)
        # Deliberately differs from the source's answer above (tea vs coffee).
        stale_entry = {"key": "e1", "subject": "speaker0", "predicate": "likes", "object": "tea"}
        store.put("episodic", "e1", dict(stale_entry))

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
            _probe_and_reason(
                text="What do I like?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=store,
            )

        assert "speaker0 likes coffee" in captured["augmented_text"], (
            "the source's answer must reach the prompt"
        )
        assert "speaker0 likes tea" not in captured["augmented_text"], (
            "the stale cached answer must never reach the prompt"
        )
        # No write-back: the cache still holds exactly the stale entry.
        assert store.get("e1") == stale_entry

    def test_cache_miss_for_an_active_key_is_a_no_fact_not_a_fault(self, monkeypatch):
        """A key the router hands down that the cache lacks answers None
        for that key alone — the turn completes through the existing
        layers with fewer facts, no raise, no incident, no status code."""
        captured: dict = {}
        self._stub_reasoning_and_prompt(monkeypatch, captured)
        monkeypatch.setattr(
            "paramem.models.loader.switch_adapter",
            lambda model, name: None,
        )

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self._make_model(["episodic"])
        config = ServerConfig()
        config.inference.preload_cache = True

        plan = self._make_plan([("episodic", ["missing_key"])])
        store = self._seeded_store(plan)  # bookkeeping-seeded, but no store.put — cache-cold.

        from paramem.server.inference import _probe_and_reason

        with prompt_overrides({"serving_system.txt": "You are an assistant."}):
            result = _probe_and_reason(
                text="What do I like?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=store,
            )

        # No raise reached this point; the turn completed via the reasoning
        # leg with zero recalled facts for the missing key.
        assert result is not None
        assert captured.get("generate_answer_calls") == 1
        assert "missing_key" not in captured.get("augmented_text", "")


class TestBaseModelAnswerSystemPrompt:
    """_base_model_answer's system-prompt assembly, mirrored against
    _probe_and_reason's via the shared ``_build_system_prompt`` helper
    (previously two byte-identical inline blocks with zero coverage on this
    leg — a drift between the two would have been invisible to CI)."""

    def test_speaker_token_and_language_reach_system_prompt(self, monkeypatch):
        captured = {}

        def capture_messages(text, history, system_prompt):
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

        def capture_messages(text, history, system_prompt):
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

        # _build_messages no longer calls adapt_messages itself — that
        # system-role-folding concern now lives in render_chat_prompt
        # (paramem.models.loader), applied by the production call site
        # AFTER _build_messages returns. Nothing to bypass here any more;
        # the merge/strip logic under test is the whole of what
        # _build_messages does.
        history = [
            {"role": "user", "text": "first"},
            {"role": "user", "text": "second"},
            {"role": "assistant", "text": "reply"},
        ]
        messages = _build_messages("question", history, "system prompt")

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[1]["content"] == "first\nsecond"
        assert messages[-1]["content"] == "question"

    def test_leading_assistant_turn_stripped(self, monkeypatch):
        from paramem.server.inference import _build_messages

        history = [
            {"role": "assistant", "text": "orphaned reply"},
            {"role": "user", "text": "hi"},
        ]
        messages = _build_messages("question", history, "system prompt")

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
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 1
        cfg.inference.preload_cache = False
        cfg.paths.data = tmp_path
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
        """stats dict carries tier_bindings."""
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        cfg = self._make_config(tmp_path)
        _, _, _, stats = _build_store_contents(cfg, model=None, tokenizer=None)
        assert "tier_bindings" in stats
        assert "store_load_degraded" not in stats

    def test_preload_cache_off_entries_empty(self, tmp_path) -> None:
        """When preload_cache=False, new_entries is empty (intentional opt-out)."""
        from paramem.server.app import _build_store_contents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        cfg = self._make_config(tmp_path)
        cfg.inference.preload_cache = False
        new_e, _, _, _ = _build_store_contents(cfg, model=None, tokenizer=None)
        assert new_e == {}, "entries must be empty when preload_cache=False"

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


# ---------------------------------------------------------------------------
# _build_store_contents — preload_recall_incomplete incident lifecycle
# ---------------------------------------------------------------------------


class TestPreloadRecallIncidentLifecycle:
    """`_build_store_contents` owns the `preload_recall_incomplete` incident
    across its five outcome arms (see the function's own docstring):
    preload disabled, no active keys, train-venue deferral, a probe
    shortfall, and a fully-hit probe."""

    def _make_config(self, tmp_path):
        """Minimal config stub sufficient for _build_store_contents."""
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 1
        cfg.inference.preload_cache = False
        cfg.paths.data = tmp_path
        return cfg

    def _seed_active_key(self, tmp_path, tier, key="graph1", *, keys=None):
        """Write a bookkeeping row for one or more active keys in *tier* and
        return the matching `TierBinding` to hand to a patched
        `verify_adapter_tree` — the every-known-key-has-a-row invariant means
        each key needs a real `key_metadata.json` row even though the
        binding itself is fabricated.

        *keys*, when given, seeds every listed key (in that order, so
        `TierBinding.registry.list_active()` preserves it) instead of the
        single *key*.
        """
        from paramem.adapters.registry_binding import VERIFIED, TierBinding
        from paramem.training.key_registry import KeyRegistry

        key_list = list(keys) if keys is not None else [key]
        tier_dir = tmp_path / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        (tier_dir / "key_metadata.json").write_text(
            json.dumps(
                {
                    "tier_cycle": 0,
                    "keys": {
                        k: {
                            "speaker_id": "speaker0",
                            "relation_type": "factual",
                            "reinforcement_count": 1,
                            "last_reinforced_cycle": 0,
                            "last_seen": "2026-01-01T00:00:00Z",
                            "first_seen": "2026-01-01T00:00:00Z",
                            "promoted": False,
                        }
                        for k in key_list
                    },
                }
            )
        )
        reg = KeyRegistry()
        for k in key_list:
            reg.add(k)
        return TierBinding(
            tier=tier,
            tier_root=tier_dir,
            status=VERIFIED,
            registry=reg,
            registry_present=True,
            slot=None,
            manifest=None,
            candidate_count=0,
            detail="",
        )

    def test_preload_cache_off_resolves_stale_incident(self, tmp_path) -> None:
        """Arm 1: `preload_cache=False` is a clean pass for the mirror — a
        stale `preload_recall_incomplete` incident from an earlier
        `preload_cache=True` run resolves."""
        from paramem.server.app import _build_store_contents
        from paramem.server.incidents import read_incidents, record_incident

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="preload_recall_incomplete",
            key="simulate",
            severity="warning",
            summary="stale",
            detail={},
        )

        cfg = self._make_config(tmp_path)
        cfg.inference.preload_cache = False
        _build_store_contents(cfg, model=None, tokenizer=None)

        matching = [
            i for i in read_incidents(state_dir) if i.id == "preload_recall_incomplete:simulate"
        ]
        assert len(matching) == 1
        assert matching[0].status == "resolved"

    def test_no_active_keys_resolves_stale_incident(self, tmp_path) -> None:
        """Arm 2: `preload_cache=True` with no active keys anywhere is also
        a clean pass — nothing to preload, so a stale incident resolves."""
        from paramem.server.app import _build_store_contents
        from paramem.server.incidents import read_incidents, record_incident

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="preload_recall_incomplete",
            key="simulate",
            severity="warning",
            summary="stale",
            detail={},
        )

        cfg = self._make_config(tmp_path)
        cfg.inference.preload_cache = True
        new_e, _, _, _ = _build_store_contents(cfg, model=None, tokenizer=None)

        assert new_e == {}, "precondition: no active key anywhere means nothing to preload"
        matching = [
            i for i in read_incidents(state_dir) if i.id == "preload_recall_incomplete:simulate"
        ]
        assert len(matching) == 1
        assert matching[0].status == "resolved"

    def test_train_venue_deferred_leaves_incident_untouched(self, tmp_path) -> None:
        """Arm 3: a train-venue call with no model resident defers the fill
        act entirely — it produces no new verdict, so an existing incident
        is left exactly as it was, and `stats["preload_complete"]` is False
        so the caller re-attempts once a model is resident."""
        import paramem.adapters.registry_binding as registry_binding_mod
        from paramem.server.app import _build_store_contents
        from paramem.server.incidents import read_incidents, record_incident

        binding = self._seed_active_key(tmp_path, "semantic")
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="preload_recall_incomplete",
            key="train",
            severity="warning",
            summary="pre-existing",
            detail={"hits": 0, "total": 1},
        )

        cfg = self._make_config(tmp_path)
        cfg.consolidation.mode = "train"
        cfg.inference.preload_cache = True

        with patch.object(
            registry_binding_mod, "verify_adapter_tree", return_value={"semantic": binding}
        ):
            _, _, _, stats = _build_store_contents(cfg, model=None, tokenizer=None)

        assert stats["preload_complete"] is False, (
            "a deferred train-venue fill must not report completion"
        )

        matching = [
            i for i in read_incidents(state_dir) if i.id == "preload_recall_incomplete:train"
        ]
        assert len(matching) == 1
        assert matching[0].status == "active", "deferral must leave the row exactly as it was"
        assert matching[0].count == 1, "deferral must not bump the untouched row"
        assert matching[0].summary == "pre-existing"

    def test_probe_shortfall_records_then_rebumps_incident(self, tmp_path) -> None:
        """Arm 4: an active key with no bound live slot is a genuine
        `DiskMemorySource` miss — the shortfall records
        `preload_recall_incomplete:<mode>` with hits/total/missed_by_tier in
        `detail`; a second short fill with the identical shortfall bumps the
        SAME row (count, last_seen) instead of creating a duplicate."""
        import paramem.adapters.registry_binding as registry_binding_mod
        from paramem.server.app import _build_store_contents
        from paramem.server.incidents import read_incidents

        binding = self._seed_active_key(tmp_path, "semantic", key="graph1")
        for tier in ("episodic", "procedural"):
            (tmp_path / tier).mkdir()
        state_dir = tmp_path / "state"
        incident_id = "preload_recall_incomplete:simulate"

        cfg = self._make_config(tmp_path)
        cfg.consolidation.mode = "simulate"
        cfg.inference.preload_cache = True

        with patch.object(
            registry_binding_mod, "verify_adapter_tree", return_value={"semantic": binding}
        ):
            _build_store_contents(cfg, model=None, tokenizer=None)

            first_matching = [i for i in read_incidents(state_dir) if i.id == incident_id]
            assert len(first_matching) == 1
            first = first_matching[0]
            assert first.status == "active"
            assert first.severity == "warning"
            assert first.count == 1
            assert first.detail["hits"] == 0
            assert first.detail["total"] == 1
            assert first.detail["missed_by_tier"] == {"semantic": ["graph1"]}

            # Second short fill hits the identical shortfall — same
            # (type, key) — so the existing row must bump, not duplicate.
            _build_store_contents(cfg, model=None, tokenizer=None)

        second_matching = [i for i in read_incidents(state_dir) if i.id == incident_id]
        assert len(second_matching) == 1, (
            "a repeat shortfall must bump the existing row, not duplicate it"
        )
        assert second_matching[0].count == 2
        assert second_matching[0].last_seen >= first.last_seen

    def test_all_keys_hit_resolves_incident(self, tmp_path) -> None:
        """Arm 5: every active key hits the probe — the fill is complete, so
        an existing `preload_recall_incomplete` incident resolves."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.memory.source as source_mod
        from paramem.server.app import _build_store_contents
        from paramem.server.incidents import read_incidents, record_incident

        binding = self._seed_active_key(tmp_path, "semantic", key="graph1")
        for tier in ("episodic", "procedural"):
            (tmp_path / tier).mkdir()
        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="preload_recall_incomplete",
            key="simulate",
            severity="warning",
            summary="stale",
            detail={"hits": 0, "total": 1},
        )

        cfg = self._make_config(tmp_path)
        cfg.consolidation.mode = "simulate"
        cfg.inference.preload_cache = True

        class _FakeAllHitSource:
            """Minimal DiskMemorySource stand-in — every requested key hits."""

            def __init__(self, *args, **kwargs):
                pass

            def probe(self, keys_by_adapter):
                return {
                    key: {"key": key, "subject": "s", "predicate": "p", "object": "o"}
                    for keys in keys_by_adapter.values()
                    for key in keys
                }

        with (
            patch.object(
                registry_binding_mod, "verify_adapter_tree", return_value={"semantic": binding}
            ),
            patch.object(source_mod, "DiskMemorySource", _FakeAllHitSource),
        ):
            new_e, _, _, _ = _build_store_contents(cfg, model=None, tokenizer=None)

        assert new_e["semantic"]["graph1"]["subject"] == "s"
        matching = [
            i for i in read_incidents(state_dir) if i.id == "preload_recall_incomplete:simulate"
        ]
        assert len(matching) == 1
        assert matching[0].status == "resolved"

    def test_admission_rules_reject_failure_marker_and_partial_entry(self, tmp_path) -> None:
        """`is_admissible_probe_result` governs the per-key admission
        decision inside the probe loop: of three keys in one tier, only the
        full four-field entry is admitted — a `failure_reason` marker (a
        confidence-gate drop) and an entry missing one content field are
        both misses, landing in the same tier's `missed_by_tier` list."""
        import paramem.adapters.registry_binding as registry_binding_mod
        import paramem.memory.source as source_mod
        from paramem.server.app import _build_store_contents
        from paramem.server.incidents import read_incidents

        binding = self._seed_active_key(tmp_path, "semantic", keys=["graph1", "graph2", "graph3"])
        for tier in ("episodic", "procedural"):
            (tmp_path / tier).mkdir()
        state_dir = tmp_path / "state"

        cfg = self._make_config(tmp_path)
        cfg.consolidation.mode = "simulate"
        cfg.inference.preload_cache = True

        class _FakeMixedAdmissionSource:
            """Minimal DiskMemorySource stand-in returning one full entry,
            one failure-marker miss, and one partial-entry miss."""

            def __init__(self, *args, **kwargs):
                pass

            def probe(self, keys_by_adapter):
                return {
                    "graph1": {
                        "key": "graph1",
                        "subject": "s",
                        "predicate": "p",
                        "object": "o",
                    },
                    "graph2": {
                        "raw_output": '{"key": "graph2"}',
                        "failure_reason": "low_confidence:0.4",
                    },
                    "graph3": {
                        # missing "object" — carries only three of the four
                        # content fields.
                        "key": "graph3",
                        "subject": "s3",
                        "predicate": "p3",
                    },
                }

        with (
            patch.object(
                registry_binding_mod, "verify_adapter_tree", return_value={"semantic": binding}
            ),
            patch.object(source_mod, "DiskMemorySource", _FakeMixedAdmissionSource),
        ):
            new_e, _, _, _ = _build_store_contents(cfg, model=None, tokenizer=None)

        assert set(new_e["semantic"]) == {"graph1"}, (
            "only the full four-field entry is admitted into the returned entries"
        )
        assert new_e["semantic"]["graph1"] == {
            "key": "graph1",
            "subject": "s",
            "predicate": "p",
            "object": "o",
        }

        matching = [
            i for i in read_incidents(state_dir) if i.id == "preload_recall_incomplete:simulate"
        ]
        assert len(matching) == 1
        assert matching[0].status == "active"
        assert matching[0].detail["hits"] == 1
        assert matching[0].detail["total"] == 3
        assert set(matching[0].detail["missed_by_tier"]["semantic"]) == {"graph2", "graph3"}, (
            "both the failure-marker miss and the partial-entry miss land in "
            "the same tier's missed list"
        )


# ---------------------------------------------------------------------------
# _hydrate_memory_store_in_place — degraded-build swap guard (regression)
# ---------------------------------------------------------------------------


class TestHydrateMemoryStoreSwapGuard:
    """Whole-store publish: an unverified tier quarantines the WHOLE store —
    there is no per-tier half-publish. A healthy tree (no unverified tier)
    still swaps normally."""

    def _make_config(self, tmp_path):
        """Minimal config stub sufficient for _build_store_contents."""
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        cfg.consolidation.mode = "simulate"
        cfg.consolidation.recall_probe_batch_size = 1
        cfg.inference.preload_cache = False
        cfg.paths.data = tmp_path
        return cfg

    def test_unverified_tier_quarantines_the_store_others_do_not_publish_either(
        self, tmp_path
    ) -> None:
        """A tier whose registry binding fails verification quarantines the
        WHOLE store — `store.swap` never runs, so even a healthy sibling
        tier (semantic, with a real active key) does NOT publish either.
        `_hydrate_memory_store_in_place` returns `False` and *live* is left
        completely untouched."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place
        from paramem.training.key_registry import KeyRegistry

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()

        # episodic: unparseable registry -> REGISTRY_UNREADABLE.
        (tmp_path / "episodic" / "indexed_key_registry.json").write_bytes(b"not json at all")

        # semantic: a real, healthy registry with one active key, and its
        # bookkeeping row -- every known key carries one (the
        # every-known-key-has-a-row invariant), so the boot load raises
        # loudly without it. Healthy on its own, but the quarantine is
        # whole-store, so it must not publish either.
        reg = KeyRegistry()
        reg.add("graph1")
        reg.save(tmp_path / "semantic" / "indexed_key_registry.json")
        (tmp_path / "semantic" / "key_metadata.json").write_text(
            json.dumps(
                {
                    "tier_cycle": 0,
                    "keys": {
                        "graph1": {
                            "speaker_id": "speaker0",
                            "relation_type": "factual",
                            "reinforcement_count": 1,
                            "last_reinforced_cycle": 0,
                            "last_seen": "2026-01-01T00:00:00Z",
                            "first_seen": "2026-01-01T00:00:00Z",
                            "promoted": False,
                        }
                    },
                }
            )
        )

        live = MemoryStore()
        cfg = self._make_config(tmp_path)

        ok = _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)

        assert ok is False
        # has_registry checked BEFORE any .registry(tier) call — that
        # accessor allocates a fresh empty registry via setdefault, which
        # would flip has_registry to True and mask the very thing under test.
        assert not live.has_registry("episodic")
        assert not live.has_registry("semantic"), (
            "a healthy sibling tier must not publish either — no per-tier half-publish"
        )

    def test_legitimate_empty_registry_does_swap(self, tmp_path) -> None:
        """A successful build with an empty (but verified) registry swaps.

        This verifies the guard is per-tier verification, not on
        len(registry)==0.
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

        ok = _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)

        assert ok is True
        # The old entry must be gone — the swap replaced the store with the empty build.
        assert live.get("old_key") is None, (
            "legitimate empty build did not swap: old entry still present"
        )

    def test_unverified_tier_records_the_store_quarantine_incident(self, tmp_path) -> None:
        """A tier whose registry binding fails verification records the ONE
        `store_quarantined` incident (never a per-tier
        `tier_registry_unverified` one — that type is reserved for
        post-fold drift detection, not the store-publish boundary), with
        the offending tier named in its cause."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place
        from paramem.server.incidents import read_incidents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()
        (tmp_path / "episodic" / "indexed_key_registry.json").write_bytes(b"not json at all")

        live = MemoryStore()
        cfg = self._make_config(tmp_path)

        _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)

        incidents = read_incidents(tmp_path / "state")
        assert not any(i.id == "tier_registry_unverified:episodic" for i in incidents)
        matching = [i for i in incidents if i.id == "store_quarantined:store"]
        assert len(matching) == 1
        assert matching[0].severity == "failed"
        assert "episodic" in str(matching[0].detail)

    def test_recovered_tree_clears_the_store_quarantine_incident(self, tmp_path) -> None:
        """A tree that was unverified and later becomes fully publishable
        again (e.g. the operator restored a healthy registry) has the
        `store_quarantined` incident resolved on the next hydrate — the
        lift's record-or-clear site is the same call, not a separate
        resolver a caller could forget."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place
        from paramem.server.incidents import read_incidents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()
        reg_path = tmp_path / "episodic" / "indexed_key_registry.json"
        reg_path.write_bytes(b"not json at all")

        live = MemoryStore()
        cfg = self._make_config(tmp_path)

        first = _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)
        assert first is False
        active = [
            i
            for i in read_incidents(tmp_path / "state")
            if i.id == "store_quarantined:store" and i.status == "active"
        ]
        assert len(active) == 1, "precondition: the store-quarantine incident must be active"

        # Restore a healthy (parseable, empty) registry and lift — same
        # primitive, re-invoked, no restart.
        reg_path.write_text('{"active_keys": [], "stale": [], "simhash": {}}')
        second = _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)
        assert second is True

        all_incidents = read_incidents(tmp_path / "state")
        resolved = [i for i in all_incidents if i.id == "store_quarantined:store"]
        assert len(resolved) == 1
        assert resolved[0].status == "resolved"

    def test_vanished_interim_tier_also_lifts_the_quarantine(self, tmp_path) -> None:
        """An unverified interim tier that later vanishes entirely (folded
        away and reaped by a full cycle) is another path to a fully
        publishable tree — `verify_adapter_tree` never enumerates it again,
        so the next hydrate is a normal successful build that lifts the
        quarantine exactly like a repaired registry does."""
        from paramem.memory.store import MemoryStore
        from paramem.server.app import _hydrate_memory_store_in_place
        from paramem.server.incidents import read_incidents

        for tier in ("episodic", "semantic", "procedural"):
            (tmp_path / tier).mkdir()
        interim_dir = tmp_path / "episodic" / "interim_20260421T0400"
        interim_dir.mkdir(parents=True)
        (interim_dir / "indexed_key_registry.json").write_bytes(b"not json at all")

        live = MemoryStore()
        cfg = self._make_config(tmp_path)

        first = _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)
        assert first is False
        active = [
            i
            for i in read_incidents(tmp_path / "state")
            if i.id == "store_quarantined:store" and i.status == "active"
        ]
        assert len(active) == 1, "precondition: the store-quarantine incident must be active"

        # The interim slot folds away entirely — a full cycle absorbed and
        # reaped it. It is never enumerated by verify_adapter_tree again.
        import shutil

        shutil.rmtree(interim_dir)

        second = _hydrate_memory_store_in_place(live, cfg, model=None, tokenizer=None)
        assert second is True

        resolved = [
            i for i in read_incidents(tmp_path / "state") if i.id == "store_quarantined:store"
        ]
        assert len(resolved) == 1
        assert resolved[0].status == "resolved"
