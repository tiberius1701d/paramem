"""Integration tests requiring GPU and model loading.

Run with: pytest tests/test_integration_gpu.py -v --gpu
Skip in CI: these tests are excluded by default (require --gpu flag).

Tests cover:
1. rollback_preparation
2. extract_procedural_graph
3. SOTA noise filter (anonymize → filter → de-anonymize)
4. get_home_context
5. _extract_and_start_training (mocked HA)
6. _training_scheduler (async)
7. BackgroundTrainer._train_adapter
8. Batch consolidation end-to-end
"""

import os
import time
from unittest.mock import MagicMock

import pytest

# Skip all tests in this file unless --gpu flag is passed
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU tests require --gpu flag",
    ),
]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load Mistral model once for all GPU tests, unload on teardown."""
    import gc

    import torch

    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    from paramem.models.loader import load_base_model
    from paramem.utils.config import load_config

    config = load_config()
    model, tokenizer = load_base_model(config.model)
    yield model, tokenizer

    # Release GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def training_config():
    from paramem.utils.config import load_config

    return load_config()


# --- 1. rollback_preparation ---


class TestRollbackPreparation:
    def _make_loop(self, tmp_path):
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        model = MagicMock()
        # Pretend all adapters (including staging) already exist to skip _ensure_adapters
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }
        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(indexed_key_replay_enabled=True),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            output_dir=tmp_path,
        )
        return loop

    def test_rollback_restores_state(self, tmp_path):
        """Verify prepare_training_data → rollback restores original state."""
        loop = self._make_loop(tmp_path)

        orig_cycle = loop.cycle_count
        orig_next_idx = loop._indexed_next_index
        orig_qa = dict(loop.indexed_key_qa)

        loop.prepare_training_data([], [])

        assert loop.cycle_count == orig_cycle + 1

        loop.rollback_preparation()
        assert loop.cycle_count == orig_cycle
        assert loop._indexed_next_index == orig_next_idx
        assert loop.indexed_key_qa == orig_qa

    def test_rollback_without_snapshot(self, tmp_path):
        """Rollback with no prior prepare should not crash."""
        loop = self._make_loop(tmp_path)
        loop.rollback_preparation()


# --- 2. extract_procedural_graph ---


class TestExtractProceduralGraph:
    def test_extracts_preferences(self, model_and_tokenizer):
        """Run procedural extraction on a transcript with preferences."""
        from paramem.graph.extractor import extract_procedural_graph

        model, tokenizer = model_and_tokenizer
        transcript = (
            "[user] I really enjoy listening to jazz music in the evening.\n"
            "[assistant] Jazz is a great choice for relaxing.\n"
            "[user] I also like drinking coffee in the morning."
        )
        graph = extract_procedural_graph(
            model,
            tokenizer,
            transcript,
            "test_proc",
            max_tokens=1024,
            stt_correction=False,
        )
        # Should extract at least something (preferences)
        # Quality depends on Mistral — may extract 0 or more
        assert graph.session_id == "test_proc"

    def test_empty_transcript(self, model_and_tokenizer):
        """Empty transcript should return empty graph."""
        from paramem.graph.extractor import extract_procedural_graph

        model, tokenizer = model_and_tokenizer
        graph = extract_procedural_graph(
            model,
            tokenizer,
            "[user] Stop.",
            "test_empty",
            max_tokens=512,
            stt_correction=False,
        )
        assert graph.session_id == "test_empty"


# --- 3. SOTA noise filter full flow ---


class TestSOTAFullFlow:
    def test_anonymize_with_local_model(self, model_and_tokenizer):
        """Test that local model can anonymize extracted facts."""
        from paramem.graph.extractor import _anonymize_with_local_model
        from paramem.graph.schema import Entity, Relation, SessionGraph

        model, tokenizer = model_and_tokenizer
        graph = SessionGraph(
            session_id="test",
            timestamp="2026-04-09T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                    confidence=1.0,
                ),
            ],
        )
        result, mapping, anon_transcript = _anonymize_with_local_model(graph, model, tokenizer)
        # Mistral may or may not produce valid anonymization
        # Just verify no crash and correct return types
        if result is not None:
            assert isinstance(result, list)
            assert isinstance(mapping, dict)
            assert isinstance(anon_transcript, str)


# --- 4. get_home_context ---


class TestGetHomeContext:
    def _make_client(self, states=None):
        from paramem.server.tools.ha_client import HAClient

        client = HAClient.__new__(HAClient)
        client._raw_states = states or []
        client._client = None
        client._rest_url = "http://fake:8123"
        client._token = "fake"
        client._timeout = 10
        return client

    def test_returns_context_with_mock_ha(self):
        """Test get_home_context with mocked HA API."""
        client = self._make_client(
            states=[
                {"entity_id": "zone.home", "attributes": {"friendly_name": "Home"}},
                {"entity_id": "zone.work", "attributes": {"friendly_name": "Work"}},
                {"entity_id": "light.living_room", "attributes": {"area_id": "living_room"}},
            ]
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "location_name": "Millfield",
            "time_zone": "Europe/Berlin",
            "latitude": 50.1,
            "longitude": 8.4,
        }
        mock_response.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_http.get.return_value = mock_response

        client._get_client = lambda: mock_http
        context = client.get_home_context()

        assert context["location_name"] == "Millfield"
        assert context["timezone"] == "Europe/Berlin"
        assert "Home" in context["zones"]
        assert "Work" in context["zones"]

    def test_graceful_on_connection_failure(self):
        """Test get_home_context when HA is unreachable."""
        import httpx

        client = self._make_client()
        mock_http = MagicMock()
        mock_http.get.side_effect = httpx.ConnectError("Connection refused")
        client._get_client = lambda: mock_http

        context = client.get_home_context()
        assert context["location_name"] == ""
        assert context["zones"] == []


# --- 5 & 6. Training scheduler and extract_and_start_training ---
# These are deeply integrated with the server and hard to unit test.
# Test the key behavior: consolidation flag management.


class TestTrainingSchedulerBehavior:
    @pytest.fixture(autouse=True)
    def _reset_state(self):
        """Snapshot _state before each test and restore after."""
        from paramem.server.app import _state

        saved = _state.get("consolidating")
        yield
        _state["consolidating"] = saved if saved is not None else False

    def test_consolidation_done_callback_clears_flag(self):
        """_consolidation_done_callback must reset _state['consolidating'] to False."""
        from paramem.server.app import _consolidation_done_callback, _state

        future = MagicMock()
        future.exception.return_value = None

        _state["consolidating"] = True
        _consolidation_done_callback(future)
        assert _state["consolidating"] is False

    def test_consolidation_done_callback_on_exception(self):
        """_consolidation_done_callback still clears the flag when the future raised."""
        from paramem.server.app import _consolidation_done_callback, _state

        future = MagicMock()
        future.exception.return_value = RuntimeError("simulated")

        _state["consolidating"] = True
        _consolidation_done_callback(future)
        assert _state["consolidating"] is False

    def test_training_scheduler_done_callback_on_exception(self):
        """_training_scheduler_done_callback clears the flag if extraction failed."""
        from paramem.server.app import _state, _training_scheduler_done_callback

        future = MagicMock()
        future.exception.return_value = RuntimeError("extraction failed")

        _state["consolidating"] = True
        _training_scheduler_done_callback(future)
        assert _state["consolidating"] is False


# --- 7. BackgroundTrainer._train_adapter ---


class TestBackgroundTrainerTraining:
    def test_train_adapter_with_real_model(self, model_and_tokenizer, training_config, tmp_path):
        """Test BackgroundTrainer can train an adapter on real model."""
        from paramem.models.loader import create_adapter
        from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
        from paramem.training.indexed_memory import assign_keys
        from paramem.utils.config import AdapterConfig, TrainingConfig

        model, tokenizer = model_and_tokenizer

        # Create adapter if not exists
        if not hasattr(model, "peft_config") or "episodic" not in model.peft_config:
            model = create_adapter(model, AdapterConfig(), "episodic")
        if "in_training" not in model.peft_config:
            model = create_adapter(model, AdapterConfig(), "in_training")

        qa = [
            {"question": "Where does Alex live?", "answer": "Alex lives in Millfield."},
        ]
        keyed = assign_keys(qa)

        tc = TrainingConfig(num_epochs=1, batch_size=1)
        bt = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=tc,
            output_dir=tmp_path,
        )

        completed = []
        job = TrainingJob(
            keyed_pairs=keyed,
            adapter_name="episodic",
            adapter_config=AdapterConfig(),
        )
        bt.start_jobs([job], on_complete=lambda: completed.append(True))

        # Wait for completion
        for _ in range(60):
            if not bt.is_training:
                break
            time.sleep(1)

        assert completed == [True]
        assert not bt.is_training

    def test_pause_resume(self, model_and_tokenizer, training_config, tmp_path):
        """Test pause/resume during training."""
        from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
        from paramem.training.indexed_memory import assign_keys
        from paramem.utils.config import AdapterConfig, TrainingConfig

        model, tokenizer = model_and_tokenizer

        # Build enough data for training to take a moment
        qa = [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(5)]
        keyed = assign_keys(qa)

        tc = TrainingConfig(num_epochs=3, batch_size=1)
        bt = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=tc,
            output_dir=tmp_path,
        )

        job = TrainingJob(
            keyed_pairs=keyed,
            adapter_name="episodic",
            adapter_config=AdapterConfig(),
        )
        bt.start_jobs([job])

        # Wait for training to start
        time.sleep(2)

        if bt.is_training:
            paused = bt.pause(timeout=10)
            assert paused
            bt.resume()

        bt.stop(timeout=30)
        assert not bt.is_training


# --- 7b. Staging adapter flow ---


@pytest.fixture(scope="class")
def staging_model(model_and_tokenizer):
    """PeftModel with episodic + in_training adapters, wrapped once per class.

    Using class scope avoids cross-test pollution of the module-level model
    fixture (which other tests may leave in an unexpected state).
    """
    from peft import PeftModel

    from paramem.models.loader import create_adapter
    from paramem.utils.config import AdapterConfig

    model, tokenizer = model_and_tokenizer
    cfg = AdapterConfig()

    # Wrap if not already a PeftModel
    if not isinstance(model, PeftModel):
        model = create_adapter(model, cfg, "episodic")

    if "episodic" not in model.peft_config:
        model = create_adapter(model, cfg, "episodic")
    if "in_training" not in model.peft_config:
        model = create_adapter(model, cfg, "in_training")

    return model, tokenizer


class TestStagingAdapterGPU:
    """End-to-end staging flow with real model and adapters."""

    def test_copy_adapter_weights_round_trip(self, staging_model):
        """copy_adapter_weights on real LoRA adapters preserves tensor values."""
        import torch

        from paramem.models.loader import copy_adapter_weights

        model, _ = staging_model

        # Perturb in_training so it differs from episodic
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".in_training.weight" in name and p.requires_grad:
                    p.data.fill_(0.5)

        # Copy episodic → in_training
        copy_adapter_weights(model, src="episodic", dst="in_training")

        # Every in_training tensor now equals its episodic counterpart
        params = dict(model.named_parameters())
        for name, p in params.items():
            if ".episodic.weight" in name:
                staging_name = name.replace(".episodic.weight", ".in_training.weight")
                assert staging_name in params
                assert torch.equal(p.data, params[staging_name].data)

    def test_staging_flow_inference_gets_production_weights(self, staging_model, tmp_path):
        """
        During a pause mid-training, inference must see the committed
        production weights, NOT the in_training mid-cycle state.
        """
        import torch

        from paramem.models.loader import switch_adapter
        from paramem.server.background_trainer import BackgroundTrainer
        from paramem.utils.config import TrainingConfig

        model, tokenizer = staging_model

        # Stamp episodic with a known signature. PEFT only marks the ACTIVE
        # adapter's tensors as requires_grad=True, so we don't filter on it.
        signature = 0.1234
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".episodic.weight" in name:
                    p.data.fill_(signature)

        # Stamp in_training with a different signature (simulating mid-training)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".in_training.weight" in name:
                    p.data.fill_(0.9999)

        bt = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=TrainingConfig(num_epochs=1),
            output_dir=tmp_path,
        )
        bt._is_training = True
        bt._current_adapter = "episodic"
        bt._training_paused.set()

        # Simulate mid-training state: in_training is active
        switch_adapter(model, "in_training")
        assert model.active_adapter == "in_training"

        # Pause for inference
        assert bt.pause(timeout=1.0) is True

        # During pause, active adapter must be episodic (production)
        assert model.active_adapter == "episodic"
        # And its weights must still be the signature value (not clobbered)
        checked = False
        for name, p in model.named_parameters():
            if ".episodic.weight" in name:
                assert torch.all(p.data == signature), (
                    f"Production episodic weights were modified during pause ({name})"
                )
                checked = True
                break
        assert checked, "No episodic weights found to verify"

        bt.resume()

    def test_commit_staging_to_production_on_gpu(self, staging_model, tmp_path):
        """Commit copies staging → production and saves atomically to disk."""
        import torch

        from paramem.server.background_trainer import BackgroundTrainer
        from paramem.utils.config import TrainingConfig

        model, tokenizer = staging_model

        # Stamp in_training
        target_value = 0.7777
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".in_training.weight" in name:
                    p.data.fill_(target_value)

        bt = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=TrainingConfig(num_epochs=1),
            output_dir=tmp_path,
        )
        bt._commit_staging_to_production("episodic")

        # Episodic in-memory now matches in_training
        checked = False
        for name, p in model.named_parameters():
            if ".episodic.weight" in name:
                assert torch.all(p.data == target_value), (
                    f"Commit did not copy staging weights to production ({name})"
                )
                checked = True
                break
        assert checked

        # Disk artifact written atomically (PEFT may save as .safetensors or .bin)
        target_dir = tmp_path / "episodic"
        assert target_dir.exists()
        weight_files = list(target_dir.glob("adapter_model.*"))
        assert weight_files, f"No adapter weight file in {target_dir}"
        # No leftover tmp/old dirs
        assert not list(tmp_path.glob("episodic.tmp.*"))
        assert not (tmp_path / "episodic.old").exists()


# --- 8. Batch consolidation end-to-end ---


class TestBatchConsolidationE2E:
    def test_extract_session_and_train(self, model_and_tokenizer, tmp_path):
        """End-to-end: extract sessions → prepare → train."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import (
            AdapterConfig,
            ConsolidationConfig,
            TrainingConfig,
        )

        model, tokenizer = model_and_tokenizer

        loop = ConsolidationLoop(
            model=model,
            tokenizer=tokenizer,
            consolidation_config=ConsolidationConfig(indexed_key_replay_enabled=True),
            training_config=TrainingConfig(num_epochs=2),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            output_dir=tmp_path,
            persist_graph=False,
            save_cycle_snapshots=False,
        )

        # Extract a session with personal facts
        transcript = (
            "[user] My name is Alex. I live in Millfield.\n"
            "[assistant] Nice to meet you, Alex. Millfield is a nice place."
        )
        episodic_qa, procedural_rels = loop.extract_session(
            transcript, "test_session", speaker_id="sp1"
        )

        # Should extract something (quality depends on Mistral)
        assert isinstance(episodic_qa, list)
        assert isinstance(procedural_rels, list)

        # If any QA pairs were extracted, verify training works
        if episodic_qa:
            result = loop.train_adapters(episodic_qa, procedural_rels, speaker_id="sp1")
            assert isinstance(result, dict)
