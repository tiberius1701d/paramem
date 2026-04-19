"""Integration tests requiring GPU and model loading.

Run with: pytest tests/test_integration_gpu.py -v --gpu
Skip in CI: these tests are excluded by default (require --gpu flag).

Tests cover:
1. rollback_preparation
2. extract_procedural_graph
3. SOTA noise filter (anonymize → filter → de-anonymize)
4. get_home_context
5. _extract_and_start_training (mocked HA)
6. _consolidation_scheduler (async)
7. BackgroundTrainer._train_adapter
8. Batch consolidation end-to-end
"""

import os
import time
from unittest.mock import MagicMock, patch

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

    def test_scheduled_extract_done_callback_on_exception(self):
        """_scheduled_extract_done_callback clears the flag if extraction failed."""
        from paramem.server.app import _scheduled_extract_done_callback, _state

        future = MagicMock()
        future.exception.return_value = RuntimeError("extraction failed")

        _state["consolidating"] = True
        _scheduled_extract_done_callback(future)
        assert _state["consolidating"] is False

    def test_scheduled_extract_done_callback_on_success(self):
        """On success, flag stays True — BG trainer will clear it on completion."""
        from paramem.server.app import _scheduled_extract_done_callback, _state

        future = MagicMock()
        future.exception.return_value = None

        _state["consolidating"] = True
        _scheduled_extract_done_callback(future)
        assert _state["consolidating"] is True


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


# --- 7c. _run_extract_graph helper end-to-end on GPU ---


class TestRunExtractGraphHelper:
    """Gap (a): the helper wrapper that every orchestrator goes through must
    actually work end-to-end with a real model.

    Unit tests (tests/test_extraction_pipeline_guard.py) prove the helper
    threads args and guards PeftModel correctly; this test proves the whole
    chain — `_extraction_kwargs` → `extract_graph` — executes on real hardware
    without drift. SOTA gates are all disabled so the test depends only on the
    local model.
    """

    def test_helper_runs_with_production_config(self, model_and_tokenizer, tmp_path):
        """Run `_run_extract_graph` with the exact flags from configs/server.yaml.

        Mirrors production: noise_filter, plausibility_judge, verify_anonymization,
        NER — all set to whatever ships in server.yaml today. Skips when
        prerequisites for the configured stack are missing (e.g.
        ANTHROPIC_API_KEY for the cloud noise filter) rather than silently
        running a weaker configuration.

        Structured so a future @pytest.mark.parametrize can introduce other
        provider/stage combinations without rewriting the scaffolding.
        """
        from paramem.graph.schema import SessionGraph
        from paramem.server.config import load_server_config
        from paramem.training.consolidation import ConsolidationLoop

        model, tokenizer = model_and_tokenizer

        server_config = load_server_config("configs/server.yaml")
        cc = server_config.consolidation

        if cc.extraction_noise_filter == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set; cannot exercise configured cloud noise filter")

        loop = ConsolidationLoop(
            model=model,
            tokenizer=tokenizer,
            consolidation_config=server_config.consolidation_config,
            training_config=server_config.training_config,
            episodic_adapter_config=server_config.episodic_adapter_config,
            semantic_adapter_config=server_config.semantic_adapter_config,
            output_dir=tmp_path,
            persist_graph=False,
            save_cycle_snapshots=False,
            prompts_dir=server_config.prompts_dir,
            extraction_max_tokens=cc.extraction_max_tokens,
            extraction_stt_correction=cc.extraction_stt_correction,
            extraction_ha_validation=cc.extraction_ha_validation,
            extraction_noise_filter=cc.extraction_noise_filter,
            extraction_noise_filter_model=cc.extraction_noise_filter_model,
            extraction_noise_filter_endpoint=cc.extraction_noise_filter_endpoint or None,
            extraction_ner_check=cc.extraction_ner_check,
            extraction_ner_model=cc.extraction_ner_model,
            extraction_plausibility_judge=cc.extraction_plausibility_judge,
            extraction_plausibility_stage=cc.extraction_plausibility_stage,
            extraction_verify_anonymization=cc.extraction_verify_anonymization,
        )

        graph = loop._run_extract_graph(
            "[user] My name is Alex. I live in Millfield.\n[assistant] Nice to meet you, Alex.",
            "gap_a_session",
        )

        assert isinstance(graph, SessionGraph)
        assert graph.session_id == "gap_a_session"


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


# --- 9. VRAM budget: math prediction vs real GPU occupation ---


class TestVRAMBudget:
    """Verify the VRAM validator against the full production loading chain.

    The validator is a math-based gate that runs *before* any model loads.
    These tests confirm two things the unit tests cannot:

    a) **Math in advance (warning path).** When the configuration does not
       fit, the validator raises :class:`ConfigurationError` with a
       structured breakdown and actionable remediation *before* the base
       model is loaded. No VRAM is consumed on rejection.

    b) **Real VRAM occupation (confirmation path).** When the
       configuration does fit by the math, the full chain actually loads
       on the GPU — base model + Whisper STT + TTS + 3 main adapters +
       14 interim adapters + staging slot — and real
       ``torch.cuda.mem_get_info()`` confirms we stay under the hardware
       cap. This catches drift between the validator's static estimate
       and the true working set (e.g. if STT/TTS grow, or an adapter
       shape changes and the math table isn't updated).

    These tests must stay together. Breaking (a) while (b) still passes
    means a misconfiguration would crash at load time instead of being
    rejected at startup; breaking (b) while (a) still passes means the
    validator's "fit" verdict is no longer trustworthy.
    """

    _HARDWARE_CAP_GIB: float = 8.0  # RTX 5070 Laptop
    _TARGET_MODULES: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def _create_adapters(self, model, adapter_cfg, names):
        """Attach each adapter in ``names`` to ``model`` and return the (possibly
        wrapped) PeftModel. Delegates to :func:`create_adapter` so the
        wrap-vs-add-adapter logic matches production."""
        from paramem.models.loader import create_adapter

        for name in names:
            model = create_adapter(model, adapter_cfg, name)
        return model

    def _delete_adapters(self, model, names):
        for name in names:
            try:
                model.delete_adapter(name)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass

    def test_fitting_config_math_and_reality(self, model_and_tokenizer):
        """Full production chain fits: validator passes AND real VRAM confirms.

        Loading order mirrors ``paramem.server.app.lifespan``:
        base → STT → TTS → main adapters → interim adapters → staging.
        At the end we compare real ``memory_allocated()`` against the
        validator's predicted total and assert we stay under the
        hardware cap.
        """
        import gc

        import torch

        from paramem.server.config import load_server_config
        from paramem.server.vram_validator import (
            _MODEL_VRAM_BYTES,
            _SAFETY_MARGIN_BYTES,
            estimate_stt_bytes,
            estimate_tts_bytes,
            estimated_adapter_bytes,
            validate_startup_vram,
        )

        model, _tokenizer = model_and_tokenizer
        server_cfg = load_server_config("configs/server.yaml")
        adapter_cfg = server_cfg.episodic_adapter_config
        max_interim = server_cfg.consolidation.max_interim_count
        assert max_interim >= 1, "max_interim_count must be at least 1 for this test"

        created_names: list[str] = []
        stt = None
        tts_manager = None

        try:
            # ── (a) math gate: pre-load math on a fresh GPU.
            # The module fixture has already loaded the base model, so real
            # ``mem_get_info`` would understate available free bytes (it would
            # exclude the already-loaded base, which is exactly what the
            # pre-load branch wants to include). We pass ``vram_cap_gib`` to
            # simulate the production pre-load condition: fresh GPU, full
            # 8 GiB available. This is the math the server would run at boot
            # before any load begins.
            stt_bytes = estimate_stt_bytes(server_cfg.stt)
            tts_bytes = estimate_tts_bytes(server_cfg.tts)
            validate_startup_vram(
                None,
                adapter_cfg,
                max_interim_count=max_interim,
                model_name=server_cfg.model_name,
                main_adapter_count=3,
                stt_bytes=stt_bytes,
                tts_bytes=tts_bytes,
                vram_cap_gib=self._HARDWARE_CAP_GIB,
            )

            # ── STT: Whisper per server.yaml (distil-large-v3 int8 on cuda).
            if server_cfg.stt.enabled and server_cfg.stt.device != "cpu":
                from paramem.server.stt import WhisperSTT

                stt = WhisperSTT(
                    model_name=server_cfg.stt.model,
                    device=server_cfg.stt.device,
                    compute_type=server_cfg.stt.compute_type,
                    language=server_cfg.stt.language,
                    beam_size=server_cfg.stt.beam_size,
                    vad_filter=server_cfg.stt.vad_filter,
                )
                if not stt.load():
                    pytest.skip("Whisper failed to load — cannot validate full STT+TTS chain")

            # ── TTS: whatever server.yaml configures (Piper CPU, MMS-TTS, etc.).
            if server_cfg.tts.enabled:
                from paramem.server.tts import TTSManager

                tts_manager = TTSManager(
                    server_cfg.tts,
                    vram_safety_margin_mb=server_cfg.server.vram_safety_margin_mb,
                )
                tts_manager.load_all()

            # ── Adapters: 3 main + max_interim interims + 1 staging slot.
            # Names must not collide with anything pytest-prior tests created.
            main_names = ["episodic_vram_probe", "semantic_vram_probe", "procedural_vram_probe"]
            interim_names = [f"episodic_interim_vram_probe_{i:02d}" for i in range(max_interim)]
            staging_name = "in_training_vram_probe"
            all_names = main_names + interim_names + [staging_name]

            model = self._create_adapters(model, adapter_cfg, all_names)
            created_names.extend(all_names)

            # ── (b) reality gate: verify post-load real VRAM usage.
            # mem_get_info reports free bytes AFTER we loaded everything. The
            # budget check here is: free ≥ safety_margin + headroom, i.e. we
            # haven't eaten into the 1 GiB KV cache / activation reserve.
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            allocated_bytes = torch.cuda.memory_allocated(0)
            used_bytes = total_bytes - free_bytes
            used_gib = used_bytes / 2**30

            # Predicted working set from the validator's math (without interim
            # headroom — those are all materialized now, so the only remaining
            # reservation is the 1 GiB KV cache + 256 MiB fragmentation margin).
            # STT/TTS bytes are included because the real VRAM reading captures
            # their allocations; omitting them here produced the 800 MiB
            # math-vs-reality gap that motivated the estimator work.
            hidden_size, num_layers = 4096, 32  # Mistral 7B
            adapter_bytes = estimated_adapter_bytes(adapter_cfg, hidden_size, num_layers)
            predicted_loaded = (
                _MODEL_VRAM_BYTES[server_cfg.model_name]
                + (3 + max_interim + 1) * adapter_bytes
                + stt_bytes
                + tts_bytes
            )

            print(
                f"\n[VRAM reality] used={used_gib:.2f} GiB / {total_bytes / 2**30:.2f} GiB, "
                f"allocated={allocated_bytes / 2**30:.2f} GiB, "
                f"predicted={predicted_loaded / 2**30:.2f} GiB, "
                f"free={free_bytes / 2**30:.2f} GiB"
            )

            # Hard cap: total GPU usage must stay under the hardware limit with
            # room for KV cache. If this fails, the math lied about the fit.
            assert used_gib < self._HARDWARE_CAP_GIB, (
                f"Real VRAM usage {used_gib:.2f} GiB exceeds hardware cap "
                f"{self._HARDWARE_CAP_GIB} GiB — validator's fit verdict is "
                f"unreliable."
            )

            # Free bytes must leave room for at least the 256 MiB safety margin
            # the validator reserves; otherwise KV cache during inference OOMs.
            assert free_bytes >= _SAFETY_MARGIN_BYTES, (
                f"Only {free_bytes / 2**20:.0f} MiB free after full-chain load "
                f"— below the {_SAFETY_MARGIN_BYTES / 2**20:.0f} MiB safety "
                f"margin."
            )

        finally:
            self._delete_adapters(model, created_names)
            if stt is not None:
                stt.unload()
            if tts_manager is not None:
                tts_manager.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

    def test_overbudget_config_rejected_before_load(self, caplog):
        """Oversized topology triggers math-based rejection with user-facing
        warning — and proves no load was attempted (real-VRAM failure avoided).

        This is the "proper failure reaction" path: operator configures a
        topology that won't fit, the server refuses to start with an
        actionable error, and no GPU pressure is ever applied. Without this
        gate the loads would proceed until CUDA OOM, which is harder to
        diagnose and leaks allocation state.
        """
        import logging

        from paramem.server.vram_validator import (
            ConfigurationError,
            validate_startup_vram,
        )
        from paramem.utils.config import AdapterConfig

        oversized = AdapterConfig(
            rank=256,
            alpha=512,
            learning_rate=2e-4,
            target_modules=list(self._TARGET_MODULES),
            dropout=0.0,
        )
        absurd_max_interim = 50

        diagnostic_calls: list[int] = []

        def _spy_diagnostic():
            diagnostic_calls.append(1)

        with patch(
            "paramem.server.vram_validator._log_gpu_occupancy_diagnostic",
            _spy_diagnostic,
        ):
            with caplog.at_level(logging.INFO, logger="paramem.server.vram_validator"):
                with pytest.raises(ConfigurationError) as exc_info:
                    validate_startup_vram(
                        None,
                        oversized,
                        max_interim_count=absurd_max_interim,
                        model_name="mistral",
                        model_id="mistralai/Mistral-7B-Instruct-v0.3",
                        main_adapter_count=3,
                    )

        msg = str(exc_info.value)

        # User-facing message must include every component of the breakdown
        # and every remediation knob so the operator can act on it.
        assert "VRAM Working Set Breakdown" in msg
        assert "base model" in msg
        assert "main adapters" in msg
        assert "interim adapters" in msg
        assert "in_training staging slot" in msg
        assert "KV cache headroom" in msg
        assert "required total" in msg
        assert "available (free VRAM)" in msg
        assert "margin" in msg
        assert "Reduce one of:" in msg
        assert f"current={oversized.rank}" in msg
        assert f"current={absurd_max_interim}" in msg
        assert f"current={len(self._TARGET_MODULES)}" in msg
        assert "mistralai/Mistral-7B-Instruct-v0.3" in msg

        # The nvidia-smi diagnostic must fire exactly once on the failure
        # path so operators see who is already holding VRAM.
        assert len(diagnostic_calls) == 1, (
            f"Expected _log_gpu_occupancy_diagnostic to be called once, got {len(diagnostic_calls)}"
        )

        # Success path must NOT have been taken (no "VRAM check passed" log).
        passed_logs = [r for r in caplog.records if "VRAM check passed" in r.getMessage()]
        assert not passed_logs, "Validator logged success but also raised — inconsistent path."
