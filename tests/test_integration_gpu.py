"""Integration tests requiring GPU and model loading.

Run with: pytest tests/test_integration_gpu.py -v --gpu
Skip in CI: these tests are excluded by default (require --gpu flag).

Tests cover:
1. extract_procedural_graph
2. SOTA noise filter (anonymize → filter → de-anonymize)
3. get_home_context
4. _extract_and_start_training (mocked HA)
5. _consolidation_scheduler (async)
6. BackgroundTrainer._train_adapter
7. Batch consolidation end-to-end
"""

import os
import time
from pathlib import Path
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


def _clear_lora_state(model) -> None:
    """Restore *model* to clean base-model inference state in-place.

    Two responsibilities, both required so the shared session model is safe
    for the next consumer (this module's later tests AND subsequent test
    files that receive the same session model):

    1. **Disable injected LoRA adapter layers.**  After tests that call
       :func:`~paramem.models.loader.create_adapter`, the base model's
       ``nn.Linear`` layers are replaced by PEFT ``LoraLinear`` modules
       whose ``_active_adapter`` may still point to a trained adapter.
       Setting ``_active_adapter = []`` on every ``LoraLinear`` (via PEFT's
       ``BaseTunerLayer.set_adapter``) makes each module's ``forward`` skip
       the LoRA path and return ``base_layer(x)`` — identical to the
       original linear behaviour.

    2. **Restore generation-clean training state.**  Training paths
       (``train_adapter`` via ``BackgroundTrainer`` / ``ConsolidationLoop``)
       call ``model.gradient_checkpointing_enable()`` and ``model.train()``
       and do not reliably revert them — in production every request goes
       through ``handle_chat`` which calls ``gradient_checkpointing_disable()``
       first, but contract tests that call the model directly inherit the
       dirty state.  HF silently disables the KV cache when gradient
       checkpointing is active (CLAUDE.md), which corrupts ``generate()``
       output (truncated/garbage responses).  Disable checkpointing and put
       the model back in ``eval()`` mode so downstream generation is clean.

    No GPU memory is freed here; the lora_A/lora_B parameters remain
    resident until the session model is deleted at session teardown.

    Args:
        model: The ``PreTrainedModel`` (possibly with injected PEFT layers)
            to clean.  A plain model with no LoRA layers still gets its
            checkpointing/eval state restored.
    """
    # (2) Always restore generation-clean state, even on a plain model with
    # no LoRA layers — training may have toggled these on the base model.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()

    # (1) Disable any injected LoRA adapters.
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        # PEFT not installed — no LoRA layers could have been injected.
        return
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.set_adapter([])


def _restore_generation_state(model) -> None:
    """Put *model* back in generation-clean state without touching adapters.

    Training paths leave gradient checkpointing enabled and the model in
    ``train()`` mode.  HF silently disables the KV cache while checkpointing
    is active (CLAUDE.md), which corrupts ``generate()`` output for any later
    test that reuses the shared session model.  Unlike :func:`_clear_lora_state`
    this does NOT deactivate LoRA adapters — the class-scoped ``staging_model``
    fixture owns the adapter lifecycle across its tests and must keep them
    attached between cases.
    """
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()


@pytest.fixture(autouse=True)
def _restore_shared_model_state(gpu_base_model):
    """Restore generation-clean state on the shared session model after each test.

    Training tests in this module mutate the shared model's checkpointing /
    train-mode flags (see :func:`_restore_generation_state`).  Without a
    per-test reset, a later test in the same module that calls ``generate()``
    (e.g. simulate-mode extraction, the extraction-helper end-to-end) inherits
    the dirty state and produces truncated/garbage output.  Adapter cleanup
    stays at module teardown via :func:`_clear_lora_state`.
    """
    model, _ = gpu_base_model
    yield
    _restore_generation_state(model)


@pytest.fixture(scope="module")
def model_and_tokenizer(gpu_base_model):
    """Expose the session-scoped base model to all GPU integration tests in this module.

    Delegates model loading to the session-scoped ``gpu_base_model`` fixture
    (``conftest.py``) so that Mistral 7B is loaded exactly once per
    ``pytest --gpu`` invocation rather than once per test file.

    On module teardown, any LoRA adapter state accumulated during this
    module's tests is cleared from the shared model: every ``LoraLinear``
    layer has its active adapter set to ``[]``, which makes it behave
    identically to the original ``nn.Linear`` for subsequent modules (see
    PEFT ``LoraLinear.forward`` — empty ``active_adapters`` skips the LoRA
    path entirely).  The lora_A/lora_B parameter tensors remain in GPU
    memory (LoRA at rank 8 is negligible) until session teardown frees the
    model.

    Uses ``load_server_config("tests/fixtures/server.yaml")`` to pin the
    calibration target. Test methods that need a different model should
    not be using this fixture — they should construct their own model
    explicitly, since this fixture is shared across the file.
    """
    model, tokenizer = gpu_base_model
    yield model, tokenizer

    # Module teardown: disable all injected LoRA adapters so subsequent
    # test modules that receive the same base model see clean inference
    # behaviour (no trained LoRA residuals from this module's tests).
    # Uses the PEFT private import path; ImportError is non-fatal since
    # if PEFT is unavailable no adapters could have been created.
    _clear_lora_state(model)


# --- 1. extract_procedural_graph ---


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
            speaker_id="Speaker0",
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
            speaker_id="Speaker0",
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
                    speaker_id="Speaker0",
                ),
            ],
        )
        result, mapping, anon_transcript, raw = _anonymize_with_local_model(graph, model, tokenizer)
        # Mistral may or may not produce valid anonymization
        # Just verify no crash and correct return types
        if result is not None:
            assert isinstance(result, list)
            assert isinstance(mapping, dict)
            assert isinstance(anon_transcript, str)
        assert isinstance(raw, str)


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
    def test_abort_for_inference(self, model_and_tokenizer, tmp_path):
        """abort_for_inference quiesces a submitted training callable at the next
        step boundary.

        The abort contract:
        - submit() enqueues a callable that calls train_adapter under the GPU lock.
        - abort_for_inference() sets the per-job abort flag, which the
          TrainingHooks shutdown predicate sees at the next on_step_end and sets
          control.should_training_stop=True.
        - After quiescence the submitted callable (and thus bt.is_training) is False.
        - The production adapter on disk is untouched.
        """
        from paramem.memory.entry import assign_keys
        from paramem.models.loader import create_adapter
        from paramem.server.background_trainer import BackgroundTrainer
        from paramem.training.trainer import train_adapter
        from paramem.utils.config import AdapterConfig, TrainingConfig

        model, tokenizer = model_and_tokenizer

        if not hasattr(model, "peft_config") or "episodic" not in model.peft_config:
            model = create_adapter(model, AdapterConfig(), "episodic")

        # Build enough data for training to take at least a few steps.
        keyed = assign_keys([("Alex", f"fact_{i}", f"value_{i}") for i in range(5)])

        tc = TrainingConfig(num_epochs=3, batch_size=1)
        bt = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=tc,
            output_dir=tmp_path,
        )

        hooks = bt.training_hooks_for_job()
        ac = AdapterConfig()

        def _do_train():
            from paramem.memory.entry import format_entry_training

            examples = format_entry_training(keyed, tokenizer)
            train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=examples,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path,
                hooks=hooks,
            )

        bt.submit(_do_train, inference_fallback_adapter="episodic")

        # Wait for training to start
        time.sleep(2)

        if bt.is_training:
            # Abort at the next step boundary; returns True when quiesced.
            aborted = bt.abort_for_inference(timeout=10)
            assert aborted, "abort_for_inference did not quiesce within 10 s"

        # Wait for the worker to fully drain
        if bt._worker_thread is not None:
            bt._worker_thread.join(timeout=30)
        assert not bt.is_training


# --- 7b. Staging adapter flow ---


@pytest.fixture(scope="class")
def staging_model(model_and_tokenizer):
    """PeftModel with episodic + in_training adapters, wrapped once per class.

    Using class scope avoids cross-test pollution of the module-level model
    fixture (which other tests may leave in an unexpected state).

    On teardown all adapters added by this fixture are deleted so that later
    tests and modules that share the same session model see a clean LoRA
    state.  Per CLAUDE.md: never ``delete_adapter`` followed immediately by
    ``create_adapter``; here the delete is final — no adapter is created
    afterward within this teardown path.
    """
    from peft import PeftModel

    from paramem.models.loader import create_adapter
    from paramem.utils.config import AdapterConfig

    model, tokenizer = model_and_tokenizer
    cfg = AdapterConfig()

    added: list[str] = []

    # Wrap if not already a PeftModel
    if not isinstance(model, PeftModel):
        model = create_adapter(model, cfg, "episodic")
        added.append("episodic")

    if "episodic" not in model.peft_config:
        model = create_adapter(model, cfg, "episodic")
        added.append("episodic")
    if "in_training" not in model.peft_config:
        model = create_adapter(model, cfg, "in_training")
        added.append("in_training")

    yield model, tokenizer

    # Teardown: delete the adapters this fixture created so the shared model
    # is left in a consistent state.  Adapters that already existed before
    # this fixture (not in ``added``) are intentionally left alone — the
    # module-level teardown in ``model_and_tokenizer`` handles final cleanup.
    for name in added:
        if name in model.peft_config:
            model.delete_adapter(name)


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

    def test_abort_for_inference_does_not_clobber_production_weights(self, staging_model, tmp_path):
        """After abort_for_inference(), the production episodic weights are untouched.

        The abort-then-skip contract: train_adapter returns metrics["aborted"]=True,
        and the caller skips its post-train commit / registry mutations so the
        production adapter on disk is untouched.

        Flow:
            1. Stamp episodic LoRA weights with a known signature.
            2. Submit a callable that calls train_adapter on episodic.
            3. Call abort_for_inference() after training starts.
            4. Assert: episodic weights still equal the original signature.
        """
        import torch

        from paramem.memory.entry import assign_keys
        from paramem.models.loader import switch_adapter
        from paramem.server.background_trainer import BackgroundTrainer
        from paramem.training.trainer import train_adapter
        from paramem.utils.config import AdapterConfig, TrainingConfig

        model, tokenizer = staging_model

        # Drop the fixture-pre-created in_training so train_adapter enters
        # _ensure_staging_slot via the first-time path (per AD-20: staging is
        # transient — must NOT pre-exist at training entry).  The fixture's
        # in_training was created for tests like test_copy_adapter_weights_round_trip
        # that need both slots present; this test exercises train_adapter end-to-end
        # so it owns the staging slot's full lifecycle.
        if "in_training" in model.peft_config:
            model.delete_adapter("in_training")

        # Stamp episodic LoRA A/B weights with a unique signature.
        signature = 0.1234
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".episodic." in name:
                    p.data.fill_(signature)

        switch_adapter(model, "episodic")

        # 15 keys × 30 epochs ≈ 450 steps so training stays in flight long enough
        # for abort_for_inference (called after a 2 s sleep below) to actually
        # interrupt mid-training rather than racing with normal completion.
        keyed = assign_keys([("Alex", f"fact_{i}", f"value_{i}") for i in range(15)])
        tc = TrainingConfig(num_epochs=30, batch_size=1)
        ac = AdapterConfig()

        bt = BackgroundTrainer(
            model=model,
            tokenizer=tokenizer,
            training_config=tc,
            output_dir=tmp_path,
        )
        abort_metrics: list[dict] = []

        def _do_train():
            # Build hooks INSIDE the submitted callable so training_hooks_for_job
            # captures the per-job abort event installed by _run_callable_queue
            # (BackgroundTrainer.py:316). Building outside before submit() would
            # capture _active_abort=None and the shutdown predicate would never
            # see abort_for_inference's signal — that's how production wires it.
            from paramem.memory.entry import format_entry_training

            hooks = bt.training_hooks_for_job()
            examples = format_entry_training(keyed, tokenizer)
            m = train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=examples,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path,
                hooks=hooks,
            )
            abort_metrics.append(m)

        bt.submit(_do_train, inference_fallback_adapter="episodic")

        # Let training reach at least one step.
        time.sleep(2)

        aborted = bt.abort_for_inference(timeout=15)
        assert aborted, "abort_for_inference did not quiesce within 15 s"

        # Wait for the worker to drain.
        if bt._worker_thread is not None:
            bt._worker_thread.join(timeout=30)

        # train_adapter must have reported abort.
        assert abort_metrics, "Training callable did not complete"
        assert abort_metrics[0].get("aborted") is True, (
            f"Expected metrics['aborted']=True after abort; got {abort_metrics[0]}"
        )

        # Episodic weights must not have been modified (abort skips commit).
        checked = False
        for name, p in model.named_parameters():
            if ".episodic." in name:
                assert torch.all(p.data == signature), (
                    f"Production episodic weights were modified during aborted training ({name})"
                )
                checked = True
                break
        assert checked, "No episodic weights found to verify"


# --- 7c. ExtractionPipeline.run chokepoint end-to-end on GPU ---


class TestRunExtractGraphHelper:
    """Gap (a): the chokepoint that every orchestrator goes through must
    actually work end-to-end with a real model.

    Unit tests (tests/test_extraction_pipeline_guard.py) prove the chokepoint
    threads args and guards PeftModel correctly; this test proves the whole
    chain — ``ExtractionPipeline.kwargs`` → ``extract_graph`` — executes on
    real hardware without drift. SOTA gates are all disabled so the test
    depends only on the local model.
    """

    @pytest.mark.skipif(
        not Path("configs/server.yaml").exists(),
        reason="operator-local configs/server.yaml absent (CI / fresh clone)",
    )
    def test_helper_runs_with_production_config(self, model_and_tokenizer, tmp_path):
        """Run ``loop.extraction.run`` with the exact flags from configs/server.yaml.

        Mirrors production: noise_filter, plausibility_judge, verify_anonymization,
        NER — all set to whatever ships in server.yaml today. Skips when
        prerequisites for the configured stack are missing (e.g.
        ANTHROPIC_API_KEY for the cloud noise filter) rather than silently
        running a weaker configuration.

        Structured so a future @pytest.mark.parametrize can introduce other
        provider/stage combinations without rewriting the scaffolding.
        """
        from paramem.graph.schema import SessionGraph
        from paramem.memory.store import MemoryStore
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
            memory_store=MemoryStore(replay_enabled=cc.indexed_key_replay),
            output_dir=tmp_path,
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

        graph = loop.extraction.run(
            "[user] My name is Alex. I live in Millfield.\n[assistant] Nice to meet you, Alex.",
            "gap_a_session",
            speaker_id="Speaker0",
        )

        assert isinstance(graph, SessionGraph)
        assert graph.session_id == "gap_a_session"


# --- 8. Batch consolidation end-to-end ---


class TestBatchConsolidationE2E:
    def test_extract_session_and_train(self, model_and_tokenizer, tmp_path):
        """End-to-end: extract sessions → prepare → train."""
        from paramem.memory.store import MemoryStore
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
            memory_store=MemoryStore(replay_enabled=True),
            output_dir=tmp_path,
            save_cycle_snapshots=False,
        )

        # Extract a session with personal facts
        transcript = (
            "[user] My name is Alex. I live in Millfield.\n"
            "[assistant] Nice to meet you, Alex. Millfield is a nice place."
        )
        episodic_rels, procedural_rels = loop.extract_session(
            transcript, "test_session", speaker_id="sp1"
        )

        # Should extract something (quality depends on Mistral)
        assert isinstance(episodic_rels, list)
        assert isinstance(procedural_rels, list)

        # If any relations were extracted, verify training works
        if episodic_rels:
            result = loop.train_adapters(episodic_rels, procedural_rels, speaker_id="sp1")
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

    @pytest.mark.skipif(
        not Path("configs/server.yaml").exists(),
        reason="operator-local configs/server.yaml absent (CI / fresh clone)",
    )
    def test_fitting_config_math_and_reality(self, model_and_tokenizer):
        """Full production chain fits: validator passes AND real VRAM confirms.

        Loading order mirrors ``paramem.server.app.lifespan``:
        base → STT → TTS → main adapters → interim adapters → staging.
        At the end we compare real ``memory_allocated()`` against the
        validator's predicted total and assert we stay under the
        hardware cap.
        """
        import torch

        from paramem.server.config import load_server_config
        from paramem.server.vram_predict import predict_base_bytes
        from paramem.server.vram_validator import (
            _SAFETY_MARGIN_BYTES,
            assess_topology,
            estimate_stt_bytes,
            estimate_tts_bytes,
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
            # predict_base_bytes reads from HF cache; may return None if not cached.
            # Falls back to known Mistral 7B measured value for the probe.
            _MISTRAL_7B_MEASURED = 4_308_428_800  # ~4,108 MiB NF4 (measured RTX 5070, 2026-04-19)
            base_pred = (
                predict_base_bytes(
                    server_cfg.model_config,
                    nf4_disk_to_runtime_factor=server_cfg.vram.nf4_disk_to_runtime_factor,
                )
                or _MISTRAL_7B_MEASURED
            )
            hidden_size, num_layers = 4096, 32  # Mistral 7B
            stt_bytes = estimate_stt_bytes(
                server_cfg.stt, workspace_factor=server_cfg.vram.stt_workspace_factor
            )
            tts_bytes = estimate_tts_bytes(
                server_cfg.tts,
                piper_ort_context_bytes=server_cfg.vram.tts_piper_ort_context_mib * 1024 * 1024,
            )
            lora_dtype_bytes = torch.tensor(
                [], dtype=getattr(torch, server_cfg.model_config.compute_dtype)
            ).element_size()
            peft_overhead_bytes = server_cfg.vram.peft_overhead_per_adapter_mib * 1024 * 1024
            assessment = assess_topology(
                adapter_cfg,
                max_interim_count=max_interim,
                base_bytes=base_pred,
                hidden_size=hidden_size,
                num_layers=num_layers,
                lora_dtype_bytes=lora_dtype_bytes,
                peft_overhead_bytes=peft_overhead_bytes,
                baseline_vram_gib=server_cfg.vram.baseline_vram_gib,
                model_id=server_cfg.model_config.model_id,
                main_adapter_count=3,
                headroom_gib=self._HARDWARE_CAP_GIB,
                stt_bytes=stt_bytes,
                tts_bytes=tts_bytes,
            )
            assert assessment.fits_baseline, (
                f"Topology must fit the configured baseline "
                f"({assessment.baseline_bytes / 2**30:.0f} GiB); "
                f"required {assessment.required_bytes / 2**30:.2f} GiB"
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

                tts_manager = TTSManager(server_cfg.tts)
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
            # budget check here mirrors the production post-load gate: used
            # must stay under the hardware cap, and free must leave room for
            # the 256 MiB fragmentation safety margin. The predictor
            # (base_pred, estimated_adapter_bytes) is informational; the
            # authoritative gate is enforce_post_load_budget against
            # live memory_allocated().
            torch.cuda.synchronize()
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            allocated_bytes = torch.cuda.memory_allocated(0)
            used_bytes = total_bytes - free_bytes
            used_gib = used_bytes / 2**30
            assert allocated_bytes > 0, "Expected non-zero VRAM allocation after full load"

            print(
                f"\n[VRAM reality] used={used_gib:.2f} GiB / {total_bytes / 2**30:.2f} GiB, "
                f"allocated={allocated_bytes / 2**30:.2f} GiB, "
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
            from paramem.server.vram_guard import safe_empty_cache

            safe_empty_cache()


# --- 11. Simulate-mode prompt-engineering iteration ---


class TestSimulateModePromptIteration:
    """Run the production extraction pipeline in simulate mode for fast
    prompt iteration.

    Architecture
    ------------
    The full pipeline up to and including `ExtractionPipeline.run` and
    `ExtractionPipeline.run_procedural` runs unchanged: chunking,
    `extract_session`, the 8-stage SOTA chain (when configured), QA
    generation, dedup, key assignment.  `run_consolidation_cycle` runs
    with `mode="simulate"` — persistence venue is `graph.json` under
    `adapter_dir/<tier>/`, not LoRA weights.  No
    `model.gradient_checkpointing_enable()` train cycle, no
    `_save_adapters`, no per-cycle adapter checkpoints.

    Why
    ---
    Prompt-engineering iteration on `extraction.txt` /
    `extraction_system.txt` / `extraction_procedural.txt` /
    `sota_enrichment.txt` / `sota_plausibility.txt` only exercises
    upstream phases.  Paying the ~10-minute training cost per iteration
    is wasteful — the simulate path produces the same per-session graph
    snapshots and per-phase raw_outputs that the operator inspects to
    judge a prompt edit, with full extraction quality visible.

    Use
    ---
    Run with --gpu; SOTA-chain phases skip silently when
    `ANTHROPIC_API_KEY` is unset.  Adjust the chunk text to match the
    domain you're calibrating prompts for.

    Hermetic
    --------
    All persistence directories (`paths.adapters`, `paths.debug`,
    `paths.sessions`, `paths.registry`, `paths.key_metadata`) redirect
    under `tmp_path`.  No production state is read or written.
    """

    @pytest.mark.parametrize("source_type", ["transcript", "document"])
    def test_simulate_run_skips_training_and_records_phases(
        self, model_and_tokenizer, tmp_path, source_type, monkeypatch
    ):
        """End-to-end simulate-mode run on real Mistral.

        * Loads `tests/fixtures/server.yaml` (CI-pinned config surface).
        * Monkey-patches `consolidation.mode = "simulate"` and redirects
          every `paths.*` to ``tmp_path``.
        * Seeds one chunk into a fresh `SessionBuffer` with the
          parametrized `source_type`.
        * Runs `_run_extraction_phase` end-to-end.

        Asserts:
            * Status is ``"simulated"``.
            * One session was processed.
            * Per-session ``graph_snapshot.json`` is on disk under
              ``tmp_path/debug/run_*/cycle_*/sessions/<sid>/``.
            * Phase trace contains at least ``local_extract``.
            * No LoRA weight files written (``adapter_model.safetensors``
              / ``.bin``) anywhere under ``tmp_path``.
            * Indexed-key registry contains at least one key (assigned
              by ``run_consolidation_cycle`` simulate path).
        """
        import json as _json
        from pathlib import Path as _Path

        import paramem.server.app as _app
        from paramem.memory.store import MemoryStore
        from paramem.server.config import load_server_config
        from paramem.server.consolidation import create_consolidation_loop
        from paramem.server.session_buffer import SessionBuffer

        cfg = load_server_config("tests/fixtures/server.yaml")

        # Mode override — the only thing we change about the production
        # pipeline.  Everything else stays at fixture values.
        monkeypatch.setattr(cfg.consolidation, "mode", "simulate")

        # Hermetic redirection of all persistence paths.  paths.adapters,
        # paths.registry, paths.key_metadata are derived from paths.data; the
        # directly-settable fields (sessions, debug) must be aimed under the
        # same root.  simulate_dir is gone — simulate mode now writes
        # graph.json under adapter_dir/<tier>/ alongside train mode.
        cfg.paths.data = tmp_path / "data"
        cfg.paths.debug = tmp_path / "data" / "debug"
        cfg.paths.sessions = tmp_path / "data" / "sessions"
        for path in (
            cfg.paths.data,
            cfg.paths.adapters,
            cfg.paths.debug,
            cfg.paths.sessions,
            cfg.paths.registry_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        # debug=True → per-session graph_snapshot.json gets written.
        # The whole point of this test for prompt iteration is to read
        # those snapshots; without debug, there is nothing to inspect.
        monkeypatch.setattr(cfg, "debug", True)

        # SOTA chain: production pipeline hits Anthropic if configured.
        # Skip its activation when no API key is present so the test
        # still exercises local_extract + key assignment without depending
        # on cloud.  Operators iterating on cloud prompts (sota_enrichment,
        # sota_plausibility) set ANTHROPIC_API_KEY and the chain runs.
        cc = cfg.consolidation
        if cc.extraction_noise_filter == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            monkeypatch.setattr(cc, "extraction_noise_filter", "")
            monkeypatch.setattr(cc, "extraction_plausibility_judge", "off")

        model, tokenizer = model_and_tokenizer

        buffer = SessionBuffer(
            session_dir=cfg.paths.sessions, state_dir=cfg.paths.data / "state", debug=False
        )
        buffer.set_speaker("sim-test-001", "Speaker0", "Alex")
        chunk_text = (
            "Alex is an Engineering Leader Software at Acme Corp in Germany. "
            "He led platform architecture for the Polaris XR-7 system."
        )
        buffer.append(
            conversation_id="sim-test-001",
            role="user",
            text=chunk_text,
            metadata={
                "source_type": source_type,
                "chunk_index": 0,
                "doc_title": "fictional_resume_test",
            },
        )

        # run_consolidation was deleted; use _run_extraction_phase via _state.
        # MemoryStore is lifespan-owned in production; construct it here with
        # the same replay flag the server derives from config.
        memory_store = MemoryStore(replay_enabled=cfg.consolidation.indexed_key_replay)
        loop = create_consolidation_loop(model, tokenizer, cfg, memory_store)
        prior_config = _app._state.get("config")
        prior_buffer = _app._state.get("session_buffer")
        prior_ha = _app._state.get("ha_client")
        prior_speaker = _app._state.get("speaker_store")
        _app._state["config"] = cfg
        _app._state["session_buffer"] = buffer
        _app._state["ha_client"] = None
        _app._state["speaker_store"] = None
        try:
            result = _app._run_extraction_phase(loop)
        finally:
            _app._state["config"] = prior_config
            _app._state["session_buffer"] = prior_buffer
            _app._state["ha_client"] = prior_ha
            _app._state["speaker_store"] = prior_speaker

        # --- Status assertions ---
        assert result["status"] == "simulated", (
            f"Expected status='simulated', got {result['status']!r} (full result: {result})"
        )
        assert result["sessions"] == 1
        assert result.get("simulated") is True
        assert result["episodic_rels"] >= 1, (
            f"Expected at least one episodic relation from simulate-mode run; "
            f"got episodic_rels={result['episodic_rels']}"
        )

        loop = result["loop"]
        assert loop is not None
        assert loop.indexed_key_registry is not None
        assert len(loop._all_active_keys()) >= 1, (
            "Indexed-key registry empty — run_consolidation_cycle (simulate mode) "
            "did not assign keys."
        )

        # --- Per-session snapshot present and structurally valid ---
        # Production layout (paramem/training/consolidation.py::snapshot_dir_for):
        #   paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/
        #     sessions/<sid>/graph_snapshot.json
        # Test setup pins cfg.paths.debug = tmp_path / "data" / "debug".
        snapshots = list(
            cfg.paths.debug.glob("episodic/**/sessions/sim-test-001/graph_snapshot.json")
        )
        assert len(snapshots) == 1, (
            f"Expected exactly one per-session graph_snapshot.json under "
            f"{cfg.paths.debug}; found {len(snapshots)}: {snapshots}"
        )
        graph = _json.loads(snapshots[0].read_text())
        assert graph["session_id"] == "sim-test-001"
        phase_records = graph.get("diagnostics", {}).get("phases", [])
        phase_names = [p["name"] for p in phase_records if isinstance(p, dict)]
        assert "local_extract" in phase_names, (
            f"local_extract phase did not fire (got {phase_names}). "
            "ExtractionPipeline.run did not invoke extract_graph."
        )
        local_extract = next(p for p in phase_records if p.get("name") == "local_extract")
        assert local_extract["outcome"] == "ok", (
            f"local_extract failed: {local_extract.get('reason')}"
        )
        assert local_extract.get("raw_output"), (
            "local_extract raw_output empty — model produced nothing."
        )

        # --- No LoRA writes happened anywhere under tmp_path ---
        lora_files = list((_Path(tmp_path)).rglob("adapter_model.safetensors"))
        lora_files += list((_Path(tmp_path)).rglob("adapter_model.bin"))
        assert not lora_files, (
            f"Simulate mode wrote LoRA weight files (should never happen): {lora_files}"
        )
