"""Unit tests for ConsolidationLoop.run_consolidation_cycle and related utilities.

Pure-Python, no GPU required.  All heavy dependencies (extraction, training,
adapter creation) are replaced with MagicMock objects so each test executes in
milliseconds and verifies only the orchestration logic.

Tests cover:
  1. Registry-last write order: adapter saved before registry.
  2. save_from_bytes guard (raises when called outside consolidation window).
  3. meta.json (manifest) written in the interim adapter slot.
  4. Inter-tier commit recoverability: crash during commit leaves session pending.
  5. session_ids provenance carry through _build_all_edge_entries_into.
  6. Recall-failed sessions stay pending, bounded retry, incident recording.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from paramem.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_loop(tmp_path: Path, *, adapter_names: list[str] | None = None):
    """Return a minimal ConsolidationLoop-like object for unit testing.

    We build a plain object (not a MagicMock) so we can attach real attribute
    behaviour while patching the methods we want to control.
    """
    from paramem.training.consolidation import ConsolidationLoop
    from paramem.training.key_registry import KeyRegistry

    if adapter_names is None:
        adapter_names = ["episodic", "semantic", "procedural", "in_training"]

    # Minimal mock model whose peft_config behaves like a dict.
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}

    tokenizer = MagicMock()

    # Build a minimal ConsolidationConfig and TrainingConfig.
    from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

    cons_config = ConsolidationConfig(
        indexed_key_replay=True,
    )
    training_config = TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
    )
    ep_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
    sem_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-5, target_modules=["q_proj"])

    # Use object.__setattr__ dance to avoid __init__ running (it loads models).
    # Instead, construct via __new__ and set attributes manually.
    loop = object.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = tokenizer
    loop.config = cons_config
    loop.training_config = training_config
    loop.episodic_config = ep_cfg
    loop.semantic_config = sem_cfg
    loop.procedural_config = None
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop._debug_base = None
    from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

    loop.extraction = ExtractionPipeline(
        model=model,
        tokenizer=tokenizer,
        config=ExtractionConfig(
            temperature=0.0,
            max_tokens=256,
            enrichment_provider="",
            enrichment_provider_model="",
            enrichment_provider_endpoint=None,
            plausibility_judge="off",
            plausibility_stage="deanon",
            scrub={"person name"},
        ),
        prompts_dir=None,
    )
    loop.store = MemoryStore(replay_enabled=True)
    for t in ("episodic", "semantic", "procedural"):
        loop.store.load_registry(t, KeyRegistry())
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.store.replace_simhashes_in_tier("episodic", {})
    loop.store.replace_simhashes_in_tier("semantic", {})
    loop.store.replace_simhashes_in_tier("procedural", {})
    loop.cycle_count = 0
    loop.promoted_keys = set()
    loop.shutdown_requested = False
    loop._thermal_policy = None
    loop.merger = MagicMock()
    # _build_all_edge_entries_into reads merger.graph.edges(data=True).
    # Provide a real NetworkX MultiDiGraph with two keyless episodic edges so the
    # graph-walk mints keys and training is triggered in tests that expect it.
    # The _materialize_consolidation_graph stub below skips reset_graph(), so the
    # graph survives intact through the keyed-walk step.
    _real_graph = nx.MultiDiGraph()
    _real_graph.add_node("subject1", attributes={"name": "Subject1"})
    _real_graph.add_node("object1", attributes={"name": "Object1"})
    _real_graph.add_edge("subject1", "object1", predicate="knows", relation_type="factual")
    _real_graph.add_node("subject2", attributes={"name": "Subject2"})
    _real_graph.add_node("object2", attributes={"name": "Object2"})
    _real_graph.add_edge("subject2", "object2", predicate="knows", relation_type="factual")
    loop.merger.graph = _real_graph
    # Graph-enrichment knobs. Default neighborhood hops for these unit tests.
    # Enrichment is off by default (refinement_enrichment="off", cloud master switch off in
    # ConsolidationConfig base defaults) so GraphTierRefiner.run_enrichment is never reached.
    loop.graph_enrichment_neighborhood_hops = 2
    loop.graph_enrichment_max_entities_per_pass = 50

    # Stub out the recall probe so tests with a MagicMock model do not
    # feed it into re.sub (which raises TypeError on non-string input).
    # These tests verify run_consolidation_cycle orchestration, not recall
    # gating; the probe is covered separately in
    # test_consolidation_recall_early_stop.py.
    loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}

    # Stub out _materialize_consolidation_graph so the materialize step does not
    # call reconstruct_graph / probe_entries on the MagicMock model.
    # The stub skips merger.reset_graph() so loop.merger.graph retains the
    # pre-populated keyless edges for the graph-walk keying step.
    # Materialize diagnostics: see test_consolidation.py::TestMaterializeInterimExtraRelations.
    loop._materialize_consolidation_graph = lambda **kw: (set(), [])

    return loop


def _matching_interim_config(adapter_config) -> MagicMock:
    """MagicMock double for a resident interim adapter's PEFT ``LoraConfig``.

    ``ensure_adapter_matching`` (the warm-init config-mismatch guard called
    from the interim mint branch) compares a resident adapter's ``r`` /
    ``lora_alpha`` / ``target_modules`` against the tier's target
    ``AdapterConfig`` — a bare ``MagicMock()`` placeholder never equals a
    real ``int``/``list``, so it always reads as a mismatch and falls through
    to a real (unpatched) ``create_adapter`` call. These tests simulate a
    resident interim adapter (re-fold within the cadence window) that must be
    recognised as MATCHING, so its fields are populated from *adapter_config*
    (interim adapters are always minted from ``self.episodic_config`` in
    production).
    """
    cfg = MagicMock()
    cfg.r = adapter_config.rank
    cfg.lora_alpha = adapter_config.alpha
    cfg.target_modules = list(adapter_config.target_modules)
    return cfg


def _fake_qa(n: int = 2) -> list[dict]:
    """Return n synthetic QA dicts."""
    return [
        {
            "question": f"What is fact {i}?",
            "answer": f"Fact {i} answer.",
            "subject": f"Subject{i}",
            "predicate": "knows",
            "object": f"Object{i}",
        }
        for i in range(1, n + 1)
    ]


def _make_mock_loop_with_procedural(tmp_path: Path):
    """Like _make_mock_loop but with procedural_config set (enables procedural-routing)."""
    from paramem.utils.config import AdapterConfig

    loop = _make_mock_loop(tmp_path)
    proc_cfg = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
    loop.procedural_config = proc_cfg
    return loop


def _fake_proc_rels(n: int = 2) -> list[dict]:
    """Return n synthetic procedural-relation dicts."""
    return [
        {
            "subject": f"Subject{i}",
            "predicate": "prefers",
            "object": f"Thing{i}",
            "relation_type": "preference",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Test 1 — registry-last write order and restart-time consistency
# ---------------------------------------------------------------------------


class TestRegistryLastWriteOrder:
    """Registry save must be the LAST disk write in run_consolidation_cycle;
    adapter-save failure must leave no registry entry on disk.
    """

    def test_registry_saved_after_adapter_weights(self, tmp_path: Path) -> None:
        """Registry save (save_from_bytes) must happen after save_adapter.

        The required call sequence is:
        save_bytes → hash → quads.json → build_manifest_for →
        save_adapter → save_from_bytes (registry-as-commit-signal).

        We verify: save_adapter precedes save_from_bytes in the call sequence.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = _matching_interim_config(
            loop.episodic_config
        )

        call_order: list[str] = []

        def _record_save_adapter(*args, **kwargs):
            call_order.append("save_adapter")

        def _record_save_from_bytes(payload, path, **kwargs):
            call_order.append("save_from_bytes")

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch(
                "paramem.models.loader.save_adapter",
                side_effect=_record_save_adapter,
            ),
            patch.object(KeyRegistry, "save_from_bytes", side_effect=_record_save_from_bytes),
        ):
            loop.run_consolidation_cycle(
                _fake_qa(2),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-001",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        # save_adapter must precede save_from_bytes (registry is the commit signal).
        assert "save_adapter" in call_order, "save_adapter was not called"
        assert "save_from_bytes" in call_order, "registry save_from_bytes was not called"
        last_save_adapter = max(i for i, c in enumerate(call_order) if c == "save_adapter")
        first_save_from_bytes = min(i for i, c in enumerate(call_order) if c == "save_from_bytes")
        assert last_save_adapter < first_save_from_bytes, (
            f"save_adapter must come before save_from_bytes; order was: {call_order}"
        )

    def test_interim_telemetry_records_derived_epochs(self, tmp_path: Path) -> None:
        """The interim fold's telemetry `record["epochs"]` must carry the
        DERIVED budget (paramem.utils.config.budget_for), not
        self.training_config.num_epochs -- computed before the try block so
        the finally-path record is correct even on failure.

        Derivation is the unconditional standard mechanism (no feature
        flag). One QA entry (N=1) falls in the smallest bucket (< 16 keys
        -> 80 epochs), which differs from the harness's configured
        training_config.num_epochs=1 -- an unmistakable signal that the
        recorded value is the derived one.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = _matching_interim_config(
            loop.episodic_config
        )

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter"),
            patch.object(KeyRegistry, "save_from_bytes"),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=True),
                reset_peak_memory_stats=MagicMock(),
                mem_get_info=MagicMock(return_value=(1_000_000_000, 2_000_000_000)),
                max_memory_allocated=MagicMock(return_value=500_000_000),
                max_memory_reserved=MagicMock(return_value=600_000_000),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            loop.run_consolidation_cycle(
                _fake_qa(1),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-003",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert mock_telemetry.called
        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "expected at least one interim_tier_train telemetry record"
        assert interim_calls[0].kwargs["record"]["epochs"] == 80, (
            "telemetry must record the DERIVED budget (80, smallest bucket for N=1), "
            f"got {interim_calls[0].kwargs['record']}"
        )

    def test_interim_telemetry_records_budget_fields_without_cuda(self, tmp_path: Path) -> None:
        """n_keys/accum/init/stale_keys must be recorded even when CUDA is
        unavailable -- proves these fields sit OUTSIDE the
        torch.cuda.is_available() gate (the CUDA-gate restructure). The mock
        model's named_parameters() carries a zero-valued lora_B tensor for
        the interim adapter so measured_adapter_init_state classifies
        init="cold" instead of degrading to an absent field.

        Donor seeding is unconditional and reachable at every measured-cold
        fold now, so it is explicitly stubbed to "no valid checkpoint" here
        -- this test is about the budget/telemetry fields, not the donor
        mechanism (covered separately by test_interim_telemetry_tags_donor_seeded_fold).
        """
        import torch as _torch

        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)
        loop.model.named_parameters.return_value = [
            (f"base_model.model.x.lora_B.{interim_name}.weight", _torch.zeros(2, 2)),
        ]

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_maybe_seed_from_donor", return_value=False),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter"),
            patch.object(KeyRegistry, "save_from_bytes"),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            loop.run_consolidation_cycle(
                _fake_qa(1),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-004",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "expected at least one interim_tier_train telemetry record"
        record = interim_calls[0].kwargs["record"]
        assert record["n_keys"] == 2
        assert record["accum"] == 1  # bucket <16 keys -> accum=1
        assert record["stale_keys"] == 0
        assert record["init"] == "cold"
        assert "free_before" not in record, "VRAM fields must stay absent when CUDA is unavailable"
        assert "peak_alloc" not in record
        for field, value in record.items():
            if isinstance(value, str):
                assert field in {"tier", "fold_stamp", "init"}, (
                    f"unexpected non-numeric string field {field!r}={value!r}"
                )
                if field == "init":
                    assert value in {"cold", "warm", "donor"}

    def test_interim_telemetry_tags_donor_seeded_fold(self, tmp_path: Path) -> None:
        """A cold interim adapter seeded from a VALID donor checkpoint must be
        tagged init="donor" at the interim telemetry call site too (not just
        the main-tier one) -- the funnel's returned metrics dict carries
        donor_seeded=True and the call site overrides its telemetry record
        from that, per the seeding hook's docstring (no second measurement).

        Donor seeding is unconditional (no feature flag).
        """
        import torch as _torch

        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        loop.model.get_base_model.return_value.config._name_or_path = "test/base-model"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)
        loop.model.named_parameters.return_value = [
            (f"base_model.model.x.lora_B.{interim_name}.weight", _torch.zeros(2, 2)),
        ]

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.multiple(
                "paramem.models.loader",
                switch_adapter=MagicMock(),
                copy_adapter_weights=MagicMock(),
            ),
            patch.multiple(
                "paramem.training.donor",
                donor_checkpoint_valid=MagicMock(return_value=True),
                load_donor_into_transient_slot=MagicMock(),
            ),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter"),
            patch.object(KeyRegistry, "save_from_bytes"),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            loop.run_consolidation_cycle(
                _fake_qa(1),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-004",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "expected at least one interim_tier_train telemetry record"
        record = interim_calls[0].kwargs["record"]
        assert record["init"] == "donor", (
            f"donor-seeded interim fold must be tagged 'donor': {record!r}"
        )

    def test_interim_telemetry_records_bind_fields_on_success(self, tmp_path: Path) -> None:
        """epochs_to_bind/steps_to_bind/hit_cap are derived from the
        recall_state returned by _train_tier_adapter on the success path."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.early_stop import _EarlyStopState
        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        fake_recall_state = _EarlyStopState(stop_epoch=5)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch.object(
                ConsolidationLoop,
                "_train_tier_adapter",
                return_value=({"aborted": False, "train_loss": 0.05}, fake_recall_state),
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter"),
            patch.object(KeyRegistry, "save_from_bytes"),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            loop.run_consolidation_cycle(
                _fake_qa(1),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-005",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "expected at least one interim_tier_train telemetry record"
        record = interim_calls[0].kwargs["record"]
        assert record["epochs_to_bind"] == 5
        assert record["hit_cap"] is False
        # n_keys=2 (the harness's two pre-populated graph edges) falls in the
        # smallest bucket (n_keys < 16 -> accum=1, unconditional derivation)
        # -> ceil(2/1) * 5 = 10 steps.
        assert record["steps_to_bind"] == 10

    def test_interim_telemetry_hit_cap_when_recall_never_fires(self, tmp_path: Path) -> None:
        """hit_cap=True and epochs_to_bind/steps_to_bind stay absent when
        the recall_state carries stop_epoch=None (the recall signal never
        fired within the derived epoch budget)."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.early_stop import _EarlyStopState
        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        fake_recall_state = _EarlyStopState(stop_epoch=None)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch.object(
                ConsolidationLoop,
                "_train_tier_adapter",
                return_value=({"aborted": False, "train_loss": 0.05}, fake_recall_state),
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter"),
            patch.object(KeyRegistry, "save_from_bytes"),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            loop.run_consolidation_cycle(
                _fake_qa(1),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-006",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "expected at least one interim_tier_train telemetry record"
        record = interim_calls[0].kwargs["record"]
        assert record["hit_cap"] is True
        assert "epochs_to_bind" not in record
        assert "steps_to_bind" not in record

    def test_interim_telemetry_omits_bind_fields_on_exception(self, tmp_path: Path) -> None:
        """A raise inside _train_tier_adapter still writes the
        interim_tier_train telemetry record (finally-path) with
        n_keys/accum/stale_keys populated but
        epochs_to_bind/steps_to_bind/hit_cap absent -- recall_state stays
        at its pre-declared None when the call never returns a value to
        unpack.
        """
        from paramem.training.consolidation import ConsolidationLoop

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch.object(
                ConsolidationLoop,
                "_train_tier_adapter",
                side_effect=RuntimeError("boom"),
            ),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                loop.run_consolidation_cycle(
                    _fake_qa(1),
                    [],
                    speaker_id="speaker0",
                    mode="train",
                    run_label="conv-i5-007",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "telemetry must still be written on the exception path"
        record = interim_calls[0].kwargs["record"]
        assert record["n_keys"] == 2
        assert record["aborted"] is False
        assert "epochs_to_bind" not in record
        assert "steps_to_bind" not in record
        assert "hit_cap" not in record

    def test_interim_telemetry_construction_failure_does_not_replace_in_flight_exception(
        self, tmp_path: Path
    ) -> None:
        """A raise while BUILDING the telemetry record (not just while
        writing it) must not replace the exception in flight from
        _train_tier_adapter. Forces _recall_bind_telemetry (called during
        record construction, inside the guarded try) to raise a distinct
        error while _train_tier_adapter raises its own distinct error --
        the ORIGINAL error must be what the caller sees.
        """
        from paramem.training.consolidation import ConsolidationLoop

        class _OriginalTrainingError(RuntimeError):
            pass

        class _ConstructionBoom(RuntimeError):
            pass

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch.object(
                ConsolidationLoop,
                "_train_tier_adapter",
                side_effect=_OriginalTrainingError("original training failure"),
            ),
            patch(
                "paramem.training.consolidation._recall_bind_telemetry",
                side_effect=_ConstructionBoom("record construction exploded"),
            ),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            with pytest.raises(_OriginalTrainingError, match="original training failure"):
                loop.run_consolidation_cycle(
                    _fake_qa(1),
                    [],
                    speaker_id="speaker0",
                    mode="train",
                    run_label="conv-i5-008",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # The telemetry write itself never reaches record_fold_telemetry --
        # construction raised first, and that failure is swallowed (logged)
        # rather than propagated or silently skipping the guard.
        assert not mock_telemetry.called

    def test_interim_telemetry_abort_path_sets_aborted_and_omits_hit_cap(
        self, tmp_path: Path
    ) -> None:
        """When _train_tier_adapter returns aborted=True (thermal throttle /
        operator pause -- a normal return, not a raise), the record carries
        aborted=True and OMITS hit_cap (which would otherwise misrepresent
        the abort as "ran to budget without binding").
        """
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.early_stop import _EarlyStopState

        loop = _make_mock_loop(tmp_path)
        loop._telemetry_dir = tmp_path / "telemetry"
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        # A trainer abort typically has no stop_epoch (recall never bound).
        fake_recall_state = _EarlyStopState(stop_epoch=None)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch.object(
                ConsolidationLoop,
                "_train_tier_adapter",
                return_value=({"aborted": True}, fake_recall_state),
            ),
            patch.multiple(
                "paramem.training.consolidation.torch.cuda",
                is_available=MagicMock(return_value=False),
            ),
            patch("paramem.training.consolidation.record_fold_telemetry") as mock_telemetry,
        ):
            result = loop.run_consolidation_cycle(
                _fake_qa(1),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-009",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert result.get("mode") == "aborted"
        interim_calls = [
            c for c in mock_telemetry.call_args_list if c.kwargs.get("kind") == "interim_tier_train"
        ]
        assert interim_calls, "expected an interim_tier_train telemetry record on the abort path"
        record = interim_calls[0].kwargs["record"]
        assert record["aborted"] is True
        assert "hit_cap" not in record
        assert "epochs_to_bind" not in record
        assert "steps_to_bind" not in record

    def test_adapter_save_failure_means_no_registry_entry(self, tmp_path: Path) -> None:
        """If save_adapter raises, registry must not be written to disk.

        save_from_bytes (the actual on-disk write) comes AFTER save_adapter.
        If save_adapter raises, the exception propagates before save_from_bytes
        is reached, so the registry file is never created.  save_bytes
        (the in-memory step) never touches disk.
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = _matching_interim_config(
            loop.episodic_config
        )
        registry_path = tmp_path / "indexed_key_registry.json"

        def _fail_save_adapter(*args, **kwargs):
            raise OSError("disk full")

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter", side_effect=_fail_save_adapter),
        ):
            with pytest.raises(OSError, match="disk full"):
                loop.run_consolidation_cycle(
                    _fake_qa(1),
                    [],
                    speaker_id="speaker0",
                    mode="train",
                    run_label="conv-i5-002",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )

        # Registry must not exist on disk (save_adapter failed before save_from_bytes).
        assert not registry_path.exists(), "Registry must not be written when adapter save fails"


# ---------------------------------------------------------------------------
# Test 1b — interim mint's warm-init else-branch (fold-level)
# ---------------------------------------------------------------------------


class TestInterimMintWarmInit:
    """The interim mint branch's ``else: ensure_adapter_matching(...)`` guard
    -- the second warm-init entrance, alongside the main-tier fold preamble
    -- exercised through the real fold (``run_consolidation_cycle``), not
    just the direct loader-level unit tests in
    ``test_interim_adapter_lifecycle.py``.
    """

    def test_resident_matching_slot_no_recreate_training_still_runs(self, tmp_path: Path) -> None:
        """A resident interim adapter whose config already matches is kept:
        ``ensure_adapter_matching`` IS invoked (proving the ``else`` branch
        fired, not merely that nothing happened), ``create_interim_adapter``
        is never called (the absent-only branch), ``delete_adapter`` is
        never called for it, and training still runs."""
        from paramem.models.loader import ensure_adapter_matching
        from paramem.training.key_registry import KeyRegistry

        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        interim_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        create_interim_spy = MagicMock()
        train_spy = MagicMock(return_value={"aborted": False})
        ensure_matching_spy = MagicMock(wraps=ensure_adapter_matching)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter", create_interim_spy),
            # ensure_adapter_matching is imported LOCALLY inside _run_fold
            # (fresh `from paramem.models.loader import ...` each call), so
            # the source module attribute is the one to patch -- patching a
            # paramem.training.consolidation attribute would be a no-op
            # (that name is never bound at module scope there).
            patch("paramem.models.loader.ensure_adapter_matching", ensure_matching_spy),
            patch("paramem.training.trainer.train_adapter", train_spy),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch("paramem.models.loader.save_adapter"),
            patch.object(KeyRegistry, "save_from_bytes"),
        ):
            loop.run_consolidation_cycle(
                _fake_qa(2),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-i5-warm",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert any(c.args[-1] == interim_name for c in ensure_matching_spy.call_args_list), (
            "expected ensure_adapter_matching to be called for the resident interim slot"
        )

        create_interim_spy.assert_not_called()
        delete_calls = [
            c for c in loop.model.delete_adapter.call_args_list if c.args == (interim_name,)
        ]
        assert not delete_calls, (
            f"expected no delete_adapter({interim_name!r}) call, got {delete_calls}"
        )
        assert train_spy.called, "the funnel must still train the warm-kept interim slot"


# ---------------------------------------------------------------------------
# Test 2 — save_from_bytes guard (raises when called outside consolidation window)
# ---------------------------------------------------------------------------


class TestSaveFromBytesGuard:
    """KeyRegistry.save_from_bytes raises when called outside consolidation window."""

    def test_save_from_bytes_raises_when_not_consolidating(self, tmp_path: Path) -> None:
        """_require_consolidating=True + consolidating=False → RuntimeError."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        payload = reg.save_bytes()
        path = tmp_path / "registry.json"

        with pytest.raises(RuntimeError, match="_require_consolidating"):
            reg.save_from_bytes(
                payload,
                path,
                _require_consolidating=True,
                consolidating=False,
            )

        # File must NOT have been written
        assert not path.exists()

    def test_save_from_bytes_succeeds_when_consolidating(self, tmp_path: Path) -> None:
        """_require_consolidating=True + consolidating=True → success."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        payload = reg.save_bytes()
        path = tmp_path / "registry.json"

        reg.save_from_bytes(payload, path, _require_consolidating=True, consolidating=True)
        assert path.exists()

    def test_save_from_bytes_opt_out_succeeds(self, tmp_path: Path) -> None:
        """_require_consolidating=False bypasses the guard (experiment path)."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k1")
        payload = reg.save_bytes()
        path = tmp_path / "registry.json"

        # Should succeed regardless of consolidating flag
        reg.save_from_bytes(
            payload,
            path,
            _require_consolidating=False,
            consolidating=False,
        )
        assert path.exists()

    def test_save_bytes_then_save_from_bytes_byte_identity(self, tmp_path: Path) -> None:
        """Bytes from save_bytes() written via save_from_bytes() must equal save() output."""
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("key1")
        reg.add("key2")

        path_a = tmp_path / "reg_a.json"
        path_b = tmp_path / "reg_b.json"

        payload = reg.save_bytes()
        reg.save(path_a)
        reg.save_from_bytes(payload, path_b, _require_consolidating=False)

        assert path_a.read_bytes() == path_b.read_bytes(), (
            "save_from_bytes must produce byte-identical output to save()"
        )


# ---------------------------------------------------------------------------
# Test 3 — meta.json written inside the interim adapter slot
# ---------------------------------------------------------------------------


class TestManifestWritten:
    """run_consolidation_cycle must embed meta.json in the interim adapter slot.

    Verifies that build_manifest_for is called and atomic_save_adapter
    writes meta.json alongside the adapter weights.  Uses a real
    atomic_save_adapter invocation (model.save_pretrained writes stub
    files) so the on-disk assertion is genuine.
    """

    def test_meta_json_written_in_interim_slot(self, tmp_path: Path) -> None:
        """meta.json must be present in the timestamped slot after run_consolidation_cycle."""
        from paramem.adapters.manifest import AdapterManifest, read_manifest

        loop = _make_mock_loop(tmp_path)
        stamp = "20260418T1430"
        adapter_name = f"episodic_interim_{stamp}"
        loop.model.peft_config[adapter_name] = MagicMock()

        # model.save_pretrained writes stub adapter files into the pending slot
        # so atomic_save_adapter can complete the six-step sequence.
        def _fake_save_pretrained(path, selected_adapters=None):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        loop.model.save_pretrained.side_effect = _fake_save_pretrained

        # Provide JSON-serialisable config attributes so build_manifest_for can
        # produce a valid manifest without fingerprinting real model weights.
        loop.model.config._name_or_path = "test-base-model"
        loop.model.config._commit_hash = None
        # base_model.model.state_dict() returns an empty dict → base_hash = UNKNOWN
        loop.model.base_model.model.state_dict.return_value = {}
        # Tokenizer: provide a name_or_path string.
        loop.tokenizer.name_or_path = "test-tokenizer"
        loop.tokenizer.backend_tokenizer = None
        loop.tokenizer.vocab_size = 32000
        # LoRA config attributes for the interim adapter.
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        loop.model.peft_config[adapter_name] = lora_cfg

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
        ):
            result = loop.run_consolidation_cycle(
                _fake_qa(2),
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-manifest-001",
                schedule="every 2h",
                max_interim_count=4,
                stamp=stamp,
            )

        assert result["mode"] == "trained"

        # Locate the timestamped slot created by atomic_save_adapter.
        # 2026-05-14 hierarchy: interim slots live under episodic/interim_<stamp>/.
        adapter_dir = tmp_path / "episodic" / f"interim_{stamp}"
        slots = [d for d in adapter_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, f"No slot dir created under {adapter_dir}"
        slot = slots[0]

        # meta.json must be present in the slot.
        assert (slot / "meta.json").exists(), f"meta.json missing from slot {slot}"

        # Manifest must be parseable and reference the correct adapter name.
        manifest = read_manifest(slot)
        assert isinstance(manifest, AdapterManifest)
        assert manifest.name == adapter_name


# ---------------------------------------------------------------------------
# Inter-tier commit recoverability
# ---------------------------------------------------------------------------


class TestInterTierCommitRecoverable:
    """A crash during ``commit_tier_slot`` must always be RECOVERABLE.

    The unified interim slot covers both episodic and procedural entries in a
    SINGLE ``commit_tier_slot`` call (step 12 of run_consolidation_cycle).
    If that commit crashes, the session is NOT marked consolidated — the production
    caller marks ``mark_consolidated`` ONLY after the cycle returns successfully.

    Reference: ``run_consolidation_cycle`` (consolidation.py step 12) commits the
    single interim slot; the production caller marks consolidated ONLY after return.
    """

    def test_commit_crash_leaves_session_pending(self, tmp_path: Path) -> None:
        """``commit_tier_slot`` raising must propagate out of the cycle so a
        caller's ``mark_consolidated`` is never reached.

        There is ONE commit for the unified interim slot.  A crash in it
        propagates; the session stays pending and is re-extracted on reboot.
        """
        loop = _make_mock_loop_with_procedural(tmp_path)
        stamp = "20260418T1430"
        loop.model.peft_config[f"episodic_interim_{stamp}"] = _matching_interim_config(
            loop.episodic_config
        )

        session_marked_consolidated: list[str] = []

        def _commit_side_effect(*args, **kwargs):
            raise RuntimeError("simulated crash during commit_tier_slot")

        def _caller_mark_consolidated(session_id: str) -> None:
            # The production caller only calls this AFTER the cycle returns.
            session_marked_consolidated.append(session_id)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
                side_effect=_commit_side_effect,
            ),
        ):
            try:
                loop.run_consolidation_cycle(
                    _fake_qa(2),
                    _fake_proc_rels(1),
                    speaker_id="speaker0",
                    mode="train",
                    run_label="conv-recover-001",
                    schedule="every 2h",
                    max_interim_count=4,
                    stamp=stamp,
                )
            except RuntimeError as exc:
                assert "commit_tier_slot" in str(exc)
            else:
                pytest.fail("commit crash must propagate out of the cycle")
            # The caller's mark_consolidated would run here ONLY on success.
            # The except branch above skipped it, mirroring production.

        # The session was NEVER marked consolidated → it stays pending →
        # re-extractable on the next cycle. This is the recoverability invariant.
        assert session_marked_consolidated == [], (
            "session must NOT be marked consolidated when a commit crashes mid-cycle — "
            "it must stay pending so the facts are re-extracted next cycle"
        )


# ---------------------------------------------------------------------------
# RecallGateRejected must drop the rejected interim slot from VRAM, not just
# roll back store state (owner decision 2026-07-26).
# ---------------------------------------------------------------------------


class TestRecallGateRejectedVramCleanup:
    """A post-save recall-gate rejection must leave NO trace of the rejected
    training in VRAM, matching the on-disk state ``commit_tier_slot``'s
    ``finally`` already leaves.

    Reference: ``_run_fold``'s ``except RecallGateRejected`` branch
    (``paramem/training/consolidation.py``).  Before this fix, the branch
    rolled back store state only and left the trained-but-rejected PEFT
    adapter resident and active, so a same-window retry silently
    warm-started from weights that failed the gate and existed nowhere on
    disk.
    """

    _STAMP = "20260418T1430"

    @staticmethod
    def _fake_create_interim_adapter(model, adapter_config, stamp):
        """Stand-in for ``create_interim_adapter``: mints the slot into the
        mock model's ``peft_config`` (mirrors the real function's effect)
        instead of actually building a LoRA adapter.
        """
        model.peft_config[f"episodic_interim_{stamp}"] = _matching_interim_config(adapter_config)
        return model

    def _run_cycle(self, loop, *, stamp: str):
        return loop.run_consolidation_cycle(
            _fake_qa(2),
            [],
            speaker_id="speaker0",
            mode="train",
            run_label="conv-rej-001",
            schedule="every 2h",
            max_interim_count=4,
            stamp=stamp,
        )

    def test_rejection_deletes_vram_slot_and_restores_episodic(self, tmp_path: Path) -> None:
        """A rejected fold must remove the interim adapter from
        ``model.peft_config`` and leave ``episodic`` as the active adapter.
        """
        from paramem.training.consolidation import RecallGateRejected

        loop = _make_mock_loop(tmp_path)
        interim_name = f"episodic_interim_{self._STAMP}"

        create_interim_spy = MagicMock(side_effect=self._fake_create_interim_adapter)

        def _delete_adapter(name: str) -> None:
            loop.model.peft_config.pop(name, None)

        loop.model.delete_adapter.side_effect = _delete_adapter

        def _commit_side_effect(*args, **kwargs):
            raise RecallGateRejected(
                "simulated post-save disk-integrity failure",
                recall_rate=0.5,
                threshold=1.0,
            )

        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                create_interim_spy,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
                side_effect=_commit_side_effect,
            ),
        ):
            result = self._run_cycle(loop, stamp=self._STAMP)

        assert result["mode"] == "recall_failed", f"expected recall_failed, got {result['mode']}"
        assert interim_name not in loop.model.peft_config, (
            "rejected interim adapter must be deleted from peft_config (VRAM), "
            f"got peft_config keys: {list(loop.model.peft_config)}"
        )
        delete_calls = [
            c for c in loop.model.delete_adapter.call_args_list if c.args == (interim_name,)
        ]
        assert delete_calls, f"expected delete_adapter({interim_name!r}) to be called"
        assert loop.model.set_adapter.call_args_list[-1].args == ("episodic",), (
            "episodic must be the active adapter after a rejection; last set_adapter "
            f"call was {loop.model.set_adapter.call_args_list[-1]}"
        )

    def test_same_window_retry_remints_fresh_slot_after_rejection(self, tmp_path: Path) -> None:
        """A retry within the same window (same stamp) after a rejection must
        recreate the interim slot from scratch (the mint guard sees it absent)
        and the retry must succeed once the gate passes.
        """
        from paramem.training.consolidation import RecallGateRejected

        loop = _make_mock_loop(tmp_path)
        interim_name = f"episodic_interim_{self._STAMP}"

        create_interim_spy = MagicMock(side_effect=self._fake_create_interim_adapter)

        def _delete_adapter(name: str) -> None:
            loop.model.peft_config.pop(name, None)

        loop.model.delete_adapter.side_effect = _delete_adapter

        _commit_calls = {"n": 0}

        def _commit_side_effect(*args, **kwargs):
            _commit_calls["n"] += 1
            if _commit_calls["n"] == 1:
                raise RecallGateRejected(
                    "simulated post-save disk-integrity failure",
                    recall_rate=0.5,
                    threshold=1.0,
                )
            return None

        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                create_interim_spy,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
                side_effect=_commit_side_effect,
            ),
        ):
            result_1 = self._run_cycle(loop, stamp=self._STAMP)
            assert result_1["mode"] == "recall_failed"
            assert interim_name not in loop.model.peft_config, (
                "slot must be gone from VRAM after the first (rejected) attempt"
            )

            result_2 = self._run_cycle(loop, stamp=self._STAMP)

        assert result_2["mode"] == "trained", (
            f"retry must succeed once the gate passes; got {result_2['mode']}"
        )
        assert create_interim_spy.call_count == 2, (
            "create_interim_adapter must be called again on the same-window retry "
            f"(fresh mint) — got {create_interim_spy.call_count} call(s)"
        )
        assert interim_name in loop.model.peft_config, (
            "the retry's freshly-minted slot must be resident after success"
        )

    def test_success_path_leaves_adapter_resident(self, tmp_path: Path) -> None:
        """A clean interim fold (no rejection) must leave the adapter resident
        in ``peft_config`` — the VRAM-cleanup handler must never fire on the
        success path.
        """
        loop = _make_mock_loop(tmp_path)
        interim_name = f"episodic_interim_{self._STAMP}"
        loop.model.peft_config[interim_name] = _matching_interim_config(loop.episodic_config)

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.memory.persistence.commit_tier_slot"),
        ):
            result = self._run_cycle(loop, stamp=self._STAMP)

        assert result["mode"] == "trained", f"expected trained, got {result['mode']}"
        assert interim_name in loop.model.peft_config, (
            "adapter must remain resident after a clean interim fold"
        )
        delete_calls = [
            c for c in loop.model.delete_adapter.call_args_list if c.args == (interim_name,)
        ]
        assert not delete_calls, (
            f"delete_adapter must not be called on the success path, got {delete_calls}"
        )

    def test_full_rejection_contract_then_successful_retry(self, tmp_path: Path) -> None:
        """Complete normal-return contract for a recall-gate rejection, in ONE flow.

        Drives a rejected interim fold and asserts, together, the full set of
        guarantees the individual tests above cover separately:

        1. the cycle returns normally with ``mode == "recall_failed"`` and
           ``recall_failed_session_ids`` carrying the pending session id
           (from ``episodic_rels``' ``session_id`` field — the
           ``_pending_session_ids_b`` provenance the ``except RecallGateRejected``
           branch actually uses, distinct from the per-key ``rec["session_ids"]``
           path covered by ``TestRecallFailedSessionStaysPending``);
        2. the interim slot is gone from ``model.peft_config`` (VRAM drop) and
           ``episodic`` is the active adapter;
        3. ``_indexed_next_index`` / ``_procedural_next_index`` did NOT advance;
        4. ``fold_resume.json`` is cleared;
        5. store rollback: a pre-existing key soft-staled by this cycle's
           subtractive-removal stage (``merger.removal_ledger``) is reactivated,
           and the freshly-minted (now rejected) interim tier is dropped
           wholesale from the store;
        6. a follow-up cycle at the SAME stamp re-mints a fresh slot and, once
           its gate passes, returns ``mode == "trained"`` with the counters
           advanced.
        """
        from paramem.training.consolidation import RecallGateRejected

        loop = _make_mock_loop(tmp_path)
        interim_name = f"episodic_interim_{self._STAMP}"

        create_interim_spy = MagicMock(side_effect=self._fake_create_interim_adapter)

        def _delete_adapter(name: str) -> None:
            loop.model.peft_config.pop(name, None)

        loop.model.delete_adapter.side_effect = _delete_adapter

        # A pre-existing episodic key that this cycle's subtractive-removal
        # stage soft-stales (mirrors a synonym/dedup collapse discovered
        # mid-fold) — exercises the reactivate half of the rollback.
        loop.store.put(
            "episodic",
            "graph_preexisting",
            {"key": "graph_preexisting", "subject": "X", "predicate": "knows", "object": "Y"},
            simhash=123,
        )
        loop.merger.removal_ledger = {"graph_preexisting": {"reason": "predicate_synonym_collapse"}}

        pending_session_id = "session-full-contract-001"
        episodic_rels = [
            {
                "subject": "placeholder",
                "predicate": "placeholder",
                "object": "placeholder",
                "relation_type": "factual",
                "speaker_id": "speaker0",
                "session_id": pending_session_id,
            }
        ]

        _commit_calls = {"n": 0}

        def _commit_side_effect(*args, **kwargs):
            _commit_calls["n"] += 1
            if _commit_calls["n"] == 1:
                raise RecallGateRejected(
                    "simulated post-save disk-integrity failure",
                    recall_rate=0.5,
                    threshold=1.0,
                )
            return None

        pre_indexed_index = loop._indexed_next_index
        pre_procedural_index = loop._procedural_next_index
        fold_resume_path = loop._fold_state_dir / "fold_resume.json"

        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                create_interim_spy,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.5, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
                side_effect=_commit_side_effect,
            ),
        ):
            result_1 = loop.run_consolidation_cycle(
                episodic_rels,
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-full-contract",
                schedule="every 2h",
                max_interim_count=4,
                stamp=self._STAMP,
            )

            # --- 1. Normal-return verdict ---
            assert result_1["mode"] == "recall_failed", (
                f"expected recall_failed, got {result_1['mode']}"
            )
            assert pending_session_id in result_1["recall_failed_session_ids"], (
                f"expected {pending_session_id!r} in recall_failed_session_ids, "
                f"got {result_1['recall_failed_session_ids']}"
            )

            # --- 2. VRAM slot removed, episodic active ---
            assert interim_name not in loop.model.peft_config, (
                "rejected interim adapter must be gone from peft_config (VRAM)"
            )
            assert loop.model.set_adapter.call_args_list[-1].args == ("episodic",), (
                "episodic must be the active adapter after a rejection"
            )

            # --- 3. Key counters did NOT advance ---
            assert loop._indexed_next_index == pre_indexed_index, (
                "_indexed_next_index must not advance on a rejected fold"
            )
            assert loop._procedural_next_index == pre_procedural_index, (
                "_procedural_next_index must not advance on a rejected fold"
            )

            # --- 4. fold_resume.json cleared ---
            assert not fold_resume_path.exists(), (
                "fold_resume.json must be cleared after a rejected fold"
            )

            # --- 5. Store rollback: reactivate + tier drop ---
            assert not loop.store.is_stale("graph_preexisting"), (
                "pre-existing key soft-staled this cycle must be reactivated on rollback"
            )
            assert "graph_preexisting" in loop.store.active_keys_in_tier("episodic"), (
                "reactivated key must be active again in its original tier"
            )
            assert loop.store.active_keys_in_tier(interim_name) == [], (
                "the rejected interim tier must be dropped wholesale from the store"
            )

            # --- 6. Same-stamp retry succeeds once the gate passes ---
            result_2 = loop.run_consolidation_cycle(
                episodic_rels,
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="conv-full-contract",
                schedule="every 2h",
                max_interim_count=4,
                stamp=self._STAMP,
            )

        assert result_2["mode"] == "trained", (
            f"retry must succeed once the gate passes; got {result_2['mode']}"
        )
        assert interim_name in loop.model.peft_config, (
            "the retry's freshly-minted slot must be resident after success"
        )
        assert loop._indexed_next_index > pre_indexed_index, (
            "counters must advance once the retry's fold commits successfully"
        )


# ---------------------------------------------------------------------------
# session_ids provenance carry through _build_all_edge_entries_into
# ---------------------------------------------------------------------------


class TestSessionIdsProvenanceCarry:
    """session_ids rides the in-RAM deferred-write record
    but is NEVER written to the persisted entry dict (store.put schema).

    These tests verify the carry-slot contract stated in _build_all_edge_entries_into:
    - rec["session_ids"] is present on the deferred-write record (sorted list of
      real contributing session ids, synthetic sentinels excluded).
    - rec["entry"] does NOT contain "session_ids" (the persisted dict schema
      stays unchanged).
    - speaker_id attribution is unchanged: the minted-key speaker_id comes from
      the subject node's speaker_id attribute, not from any session_ids field.
    """

    def _make_loop_with_sessions_in_graph(self, tmp_path: Path, *, session_ids: list[str]):
        """Build a minimal loop whose merger.graph has ONE keyless edge with
        the given list of real session ids in edge['sessions'].

        The edge is added with sessions=[real_id1, real_id2, ...] directly so
        we can test the harvest logic without going through the full extraction
        pipeline.
        """
        import networkx as nx

        loop = _make_mock_loop(tmp_path)
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("speaker0", speaker_id="speaker0", attributes={"name": "Alex"})
        real_graph.add_node("berlin", attributes={"name": "Berlin"})
        real_graph.add_edge(
            "speaker0",
            "berlin",
            predicate="lives_in",
            relation_type="factual",
            sessions=session_ids,
        )
        loop.merger.graph = real_graph
        return loop

    def test_rec_carries_session_ids_after_harvest(self, tmp_path: Path) -> None:
        """Deferred-write rec['session_ids'] contains real session ids from edge['sessions'].

        Synthetic sentinels (_SYNTHETIC_SESSION_IDS) are excluded; real ids survive.
        """
        from paramem.training.consolidation import _SYNTHETIC_SESSION_IDS

        real_ids = ["session-abc", "session-xyz"]
        # Include a synthetic sentinel to confirm it is filtered out.
        sessions_on_edge = real_ids + ["__interim_pending_sessions__"]
        loop = self._make_loop_with_sessions_in_graph(tmp_path, session_ids=sessions_on_edge)

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True)

        assert len(deferred_writes) == 1, f"Expected 1 deferred write; got {len(deferred_writes)}"
        rec = deferred_writes[0]
        assert "session_ids" in rec, "rec must carry session_ids (provenance plumbing)"
        result_ids = set(rec["session_ids"])
        assert "session-abc" in result_ids, f"session-abc missing from {result_ids}"
        assert "session-xyz" in result_ids, f"session-xyz missing from {result_ids}"
        # Synthetic sentinel must be excluded.
        for synthetic in _SYNTHETIC_SESSION_IDS:
            assert synthetic not in result_ids, (
                f"Synthetic sentinel {synthetic!r} must be excluded from rec['session_ids']"
            )

    def test_entry_dict_does_not_contain_session_ids(self, tmp_path: Path) -> None:
        """rec['entry'] (the persisted dict passed to store.put) must NOT contain session_ids.

        Provenance is transient/RAM-only; the persisted
        registry/bookkeeping schema stays unchanged.
        """
        loop = self._make_loop_with_sessions_in_graph(
            tmp_path, session_ids=["session-abc", "__full_consolidation_recon__"]
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True)

        assert deferred_writes, "Expected at least one deferred write"
        entry = deferred_writes[0]["entry"]
        assert "session_ids" not in entry, (
            "session_ids must NOT appear in the persisted entry dict — "
            "it is a transient rec-level field only"
        )

    def test_speaker_id_attribution_unchanged_by_session_ids(self, tmp_path: Path) -> None:
        """Minted-key speaker_id comes from the subject node, not from session_ids.

        Multi-session edge scenario: the edge carries real session ids from two
        sessions.  The minted entry's speaker_id must come from the subject node's
        speaker_id attribute ('speaker0'), not from any session_id value.
        """
        loop = self._make_loop_with_sessions_in_graph(
            tmp_path, session_ids=["session-A", "session-B"]
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True)

        assert deferred_writes, "Expected at least one deferred write"
        rec = deferred_writes[0]
        # speaker_id on the rec comes from the subject node attribute ("speaker0"),
        # not from any session_ids element.
        assert rec["speaker_id"] == "speaker0", (
            f"speaker_id must be 'speaker0' (from subject node attribute); "
            f"got {rec['speaker_id']!r}"
        )
        # The rec carries BOTH real session ids.
        result_ids = set(rec["session_ids"])
        assert "session-A" in result_ids and "session-B" in result_ids, (
            f"Both real session ids must be in rec['session_ids']; got {result_ids}"
        )

    def test_empty_sessions_on_edge_gives_empty_session_ids(self, tmp_path: Path) -> None:
        """Edge with no sessions list → rec['session_ids'] is empty (not an error)."""
        import networkx as nx

        loop = _make_mock_loop(tmp_path)
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("speaker0", speaker_id="speaker0", attributes={"name": "Alex"})
        real_graph.add_node("berlin", attributes={"name": "Berlin"})
        # No 'sessions' key on the edge (legacy graph or edge without stamps).
        real_graph.add_edge("speaker0", "berlin", predicate="lives_in", relation_type="factual")
        loop.merger.graph = real_graph

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True)

        assert deferred_writes, "Expected at least one deferred write"
        assert deferred_writes[0]["session_ids"] == [], (
            "Edge without sessions key → session_ids must be []"
        )

    def test_only_synthetic_sessions_gives_empty_session_ids(self, tmp_path: Path) -> None:
        """Edge whose sessions list contains ONLY synthetic sentinels → session_ids is [].

        This happens when a fold-only re-merge creates an edge with no real
        extraction-time session ids.
        """
        from paramem.training.consolidation import _SYNTHETIC_SESSION_IDS

        synthetic_only = list(_SYNTHETIC_SESSION_IDS)
        loop = self._make_loop_with_sessions_in_graph(tmp_path, session_ids=synthetic_only)

        tier_keyed: dict = {"episodic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True)

        assert deferred_writes, "Expected at least one deferred write"
        assert deferred_writes[0]["session_ids"] == [], (
            f"Only-synthetic sessions → session_ids must be []; "
            f"got {deferred_writes[0]['session_ids']}"
        )


# ---------------------------------------------------------------------------
# Keep recall-failed sessions pending + bounded retry + incident
# ---------------------------------------------------------------------------


class TestRecallFailedSessionStaysPending:
    """Acceptance tests for the keep-pending / bounded-retry / incident wiring.

    Validated without GPU or model weights.

    Test strategy:
    - Call run_consolidation_cycle directly so we can control the graph and
      probe stub precisely.
    - Set up the merger graph with an edge tagged with a real session id so
      rec["session_ids"] carries it through the harvest path.
    - Override _probe_passing_keys to exclude one key, triggering the drop
      site at step 11b.
    - Assert result["recall_failed_session_ids"] and downstream behavior.

    Conditional-assertion caveat:
    Under refinement_normalization="off", _pending_relations is None so the
    pending-session relations may not enter the merge graph and new episodic
    keys may not be minted.  The "off" arm asserts conditionally:
    if no new keys are minted, recall_failed_session_ids must be [] (the
    bug cannot manifest there); the non-empty "off" case awaits further GPU
    verification.  Procedural is asserted under "off" regardless (always
    fact-dict carrier).
    """

    # Shared patch list for run_consolidation_cycle without model weights.
    _CYCLE_PATCHES = [
        "paramem.memory.interim_adapter.create_interim_adapter",
        "paramem.training.trainer.train_adapter",
        "paramem.training.consolidation.format_entry_training",
        "paramem.models.loader.switch_adapter",
        "paramem.training.consolidation.build_registry",
        "paramem.models.loader.save_adapter",
    ]

    def _make_loop_with_session_edge(
        self,
        tmp_path: Path,
        *,
        session_id: str = "real-session-001",
        refinement_normalization: str = "off",
    ):
        """Build a loop whose merger.graph has a keyless edge with a real session id.

        The edge's sessions=[session_id] so _build_all_edge_entries_into
        harvests it onto rec["session_ids"].  Used for episodic recall-gate tests.
        """
        loop = _make_mock_loop(tmp_path)
        loop.config = loop.config.__class__(
            indexed_key_replay=True,
            refinement_normalization=refinement_normalization,
        )
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("alice", speaker_id="speaker0", attributes={"name": "Alice"})
        real_graph.add_node("paris", attributes={"name": "Paris"})
        real_graph.add_edge(
            "alice",
            "paris",
            predicate="lives_in",
            relation_type="factual",
            sessions=[session_id],
        )
        loop.merger.graph = real_graph
        # Use a fresh stamp so the adapter name is predictable.
        loop.model.peft_config["episodic_interim_20260617T0000"] = _matching_interim_config(
            loop.episodic_config
        )
        return loop

    def _run_cycle(self, loop, *, stamp: str = "20260617T0000", mode: str = "train"):
        """Call run_consolidation_cycle with the standard patch stack.

        Passes a single dummy episodic relation to bypass the no-relations guard
        (step 3 in run_consolidation_cycle).  The actual key minting comes from
        merger.graph (set up per test), not from this placeholder relation.
        """
        # One dummy relation to bypass guard step 3 (no episodic_rels → noop).
        _dummy_rel = [
            {
                "subject": "placeholder",
                "predicate": "placeholder",
                "object": "placeholder",
                "relation_type": "factual",
                "speaker_id": "speaker0",
            }
        ]
        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            return loop.run_consolidation_cycle(
                _dummy_rel,  # bypass guard step 3; real keys come from merger.graph
                [],  # procedural_rels: empty for episodic-only tests
                speaker_id="speaker0",
                mode=mode,
                run_label="test",
                schedule="",
                max_interim_count=4,
                stamp=stamp,
            )

    # ------------------------------------------------------------------
    # Test 1 — recall-failed key: session stays pending, key not registered
    # ------------------------------------------------------------------

    def test_episodic_recall_failure_populates_recall_failed_session_ids(
        self, tmp_path: Path
    ) -> None:
        """A new key that fails the recall gate → result contains its session id.

        The invariant (key unregistered AND session not consolidated) is
        verified by checking both the result dict and the store state.
        """
        session_id = "real-session-alpha"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, refinement_normalization="on"
        )

        # Override probe so it fails ALL new keys (empty passing set for new keys).
        # _recall_passing_keys returns None → falls through to _probe_passing_keys.
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="train")

        # The cycle must return recall_failed_session_ids with the contributing session.
        assert "recall_failed_session_ids" in result, "result must carry recall_failed_session_ids"
        assert session_id in result["recall_failed_session_ids"], (
            f"session {session_id!r} must be in recall_failed_session_ids; "
            f"got {result['recall_failed_session_ids']}"
        )

        # Invariant: key is NOT registered (drop site skipped store.put).
        active_keys = set(loop.store.all_active_keys())
        assert len(active_keys) == 0, (
            f"No key must be registered when recall gate fails; got {active_keys}"
        )

    def test_simulate_mode_produces_empty_recall_failed_session_ids(self, tmp_path: Path) -> None:
        """Simulate mode: recall gate is not run → recall_failed_session_ids is [].

        The simulate callsite does not run the recall gate; the result is always empty.
        """
        session_id = "real-session-sim"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, refinement_normalization="on"
        )
        # Even with a failing probe, simulate admits all without the gate.
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="simulate")

        assert result.get("recall_failed_session_ids", []) == [], (
            "Simulate mode must never produce recall_failed_session_ids"
        )

    def test_simulate_mode_never_invokes_donor_seeding_gate(self, tmp_path: Path) -> None:
        """Simulate mode (disk venue) must never reach _train_tier_adapter --
        the SOLE gate for donor seeding, which is unconditional (no feature
        flag; see paramem.training.donor).

        Every production _train_tier_adapter call site sits inside its
        enclosing `if scope.source == "weights":` branch
        (paramem.training.consolidation); this proves _run_fold really never
        crosses into it under mode="simulate", so "simulate venue never
        seeds" holds structurally rather than by an extra runtime check
        inside the funnel itself.
        """
        from paramem.training.consolidation import ConsolidationLoop

        session_id = "real-session-sim-donor"
        loop = self._make_loop_with_session_edge(tmp_path, session_id=session_id)

        with patch.object(ConsolidationLoop, "_train_tier_adapter") as spy:
            self._run_cycle(loop, mode="simulate")

        assert not spy.called, (
            "_train_tier_adapter (the donor-seeding gate) must never be called in simulate mode"
        )

    def test_off_refinement_episodic_arm_conditional(self, tmp_path: Path) -> None:
        """Under refinement_normalization='off', assert conditionally per the caveat above.

        If no new episodic keys are minted (pending-sessions path absent under off),
        recall_failed_session_ids is [] — the bug cannot manifest; no assertion
        beyond that.  If keys ARE minted (not expected from static analysis, but
        defensive), we assert the failing key's session is collected.

        NOTE: a GPU probe must establish the "off"-config minting source
        before asserting the non-empty "off" case.  This conditional arm is
        intentional — do not strengthen it without that GPU verification.
        """
        session_id = "real-session-off"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, refinement_normalization="off"
        )
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="train")

        new_keys = result.get("new_keys", [])
        failed = result.get("recall_failed_session_ids", [])

        if not new_keys:
            # Expected path: "off" mints no new episodic keys from the pending
            # graph, so the recall gate is never reached.  Bug cannot manifest.
            assert failed == [], (
                f"Under 'off' with no new keys, recall_failed_session_ids must be []; got {failed}"
            )
        else:
            # Defensive: if keys were minted, the failing session must be collected.
            assert session_id in failed, (
                f"Under 'off' with new keys, session {session_id!r} must appear in "
                f"recall_failed_session_ids; got {failed}"
            )

    # ------------------------------------------------------------------
    # Test 2 — bounded retry: cap → WARNING + incident + session released
    # ------------------------------------------------------------------

    def test_bump_retry_and_release_increments_counter(self, tmp_path: Path) -> None:
        """bump_retry_and_release increments recall_retry_count per session durably."""
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, state_dir=tmp_path / "state", consolidation_retry_cap=3)
        sid = "session-retry-001"
        buf._sessions[sid] = {"speaker": None, "state": "new"}

        released = buf.bump_retry_and_release({sid})

        assert released == [], "Count 1 < cap 3 — must not release"
        assert buf._sessions[sid]["recall_retry_count"] == 1

    def test_bump_retry_and_release_releases_at_cap(self, tmp_path: Path) -> None:
        """When retry count reaches cap, session is returned in released list."""
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, state_dir=tmp_path / "state", consolidation_retry_cap=3)
        sid = "session-retry-cap"
        # Seed in-memory count (matches durable — hydrate would do the same).
        buf._sessions[sid] = {"speaker": None, "state": "new", "recall_retry_count": 2}
        # Pre-seed durable store so bump sees count=2 before increment.
        from paramem.server.retry_state import bump_retry_count

        bump_retry_count(tmp_path / "state", sid)
        bump_retry_count(tmp_path / "state", sid)

        released = buf.bump_retry_and_release({sid})

        assert sid in released, f"Session must be released at cap; got {released}"
        assert buf._sessions[sid]["recall_retry_count"] == 3

    def test_bump_retry_and_release_skips_absent_ids(self, tmp_path: Path) -> None:
        """Guard: ids absent from _sessions are silently skipped."""
        from paramem.server.session_buffer import SessionBuffer

        buf = SessionBuffer(tmp_path, state_dir=tmp_path / "state", consolidation_retry_cap=3)
        # synthetic id that is not in _sessions
        released = buf.bump_retry_and_release({"__interim_pending_sessions__"})
        assert released == []

    def test_cap_release_records_incident_and_logs_warning(self, tmp_path: Path, caplog) -> None:
        """Hitting the cap → consolidation_retry_exhausted incident recorded + WARNING logged.

        The test simulates the _run_interim_training wiring by running
        bump_retry_and_release directly (the app.py closure is tested via
        integration; here we test the incident-record contract independently).
        """
        import logging

        from paramem.server.incidents import read_incidents, record_incident
        from paramem.server.retry_state import bump_retry_count
        from paramem.server.session_buffer import SessionBuffer

        state_dir = tmp_path / "state"
        buf = SessionBuffer(tmp_path, state_dir=state_dir, consolidation_retry_cap=2)
        sid = "session-cap-incident"
        buf._sessions[sid] = {"speaker": None, "state": "new", "recall_retry_count": 1}
        # Pre-seed durable store to count=1 so bump brings it to cap (2).
        bump_retry_count(state_dir, sid)

        caplog.set_level(logging.WARNING, logger="paramem.server.session_buffer")
        released = buf.bump_retry_and_release({sid})

        assert sid in released

        # Record the incident (mirrors _run_interim_training's for-loop).
        record_incident(
            state_dir,
            type="consolidation_retry_exhausted",
            key=sid,
            severity="warning",
            summary=(
                f"Session {sid}: facts could not be encoded after "
                f"{buf._consolidation_retry_cap} cycle(s)"
            ),
            detail={
                "session_id": sid,
                "consolidation_retry_cap": buf._consolidation_retry_cap,
                "cycle_mode": "trained",
            },
        )

        incidents = read_incidents(state_dir)
        retry_incidents = [i for i in incidents if i.type == "consolidation_retry_exhausted"]
        assert len(retry_incidents) == 1
        # Incident id is f"{type}:{key}" (dedup key = session id).
        assert retry_incidents[0].id == f"consolidation_retry_exhausted:{sid}"

        # WARNING logged by bump_retry_and_release.
        assert any("consolidation-retry cap" in r.message for r in caplog.records), (
            f"Expected WARNING about consolidation-retry cap; "
            f"got: {[r.message for r in caplog.records]}"
        )

    # ------------------------------------------------------------------
    # Test 3 — invariant guard: (key unregistered) ∧ (session NOT consolidated)
    # ------------------------------------------------------------------

    def test_invariant_no_state_where_key_unregistered_and_session_consolidated(
        self, tmp_path: Path
    ) -> None:
        """No state where (key unregistered) AND (session consolidated) can coexist.

        When a key fails the recall gate, it is NOT registered AND the session
        is NOT in the retire set returned by _completed_session_ids().  The
        invariant holds because the session stays in failed_session_ids.
        """
        session_id = "inv-session-001"
        loop = self._make_loop_with_session_edge(
            tmp_path, session_id=session_id, refinement_normalization="on"
        )
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        result = self._run_cycle(loop, mode="train")

        # Key must not be registered.
        assert len(list(loop.store.all_active_keys())) == 0, "Key must not be registered"
        # Session must appear in recall_failed_session_ids — the caller's
        # failed_session_ids.update() keeps it out of _completed_session_ids().
        assert session_id in result.get("recall_failed_session_ids", []), (
            "Session must be in recall_failed_session_ids to stay pending"
        )

    # ------------------------------------------------------------------
    # Test 4 — partial failure: passing key registered, failing not, session stays pending
    # ------------------------------------------------------------------

    def test_partial_failure_passing_key_registered_failing_not(self, tmp_path: Path) -> None:
        """Two new keys: one passes the recall gate, one fails.

        The passing key is registered; the failing key is not.  The session
        that contributed the failing key stays pending (appears in
        recall_failed_session_ids).  The passing key is not double-registered
        on a second run (store.put is idempotent via the store's own dedup).
        """
        # Add TWO edges with different sessions to the graph.
        loop = _make_mock_loop(tmp_path)
        loop.config = loop.config.__class__(
            indexed_key_replay=True,
            refinement_normalization="on",
        )
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("alice", speaker_id="speaker0", attributes={"name": "Alice"})
        real_graph.add_node("paris", attributes={"name": "Paris"})
        real_graph.add_node("london", attributes={"name": "London"})
        # Edge 1: contributing session "session-pass"
        real_graph.add_edge(
            "alice",
            "paris",
            predicate="lives_in",
            relation_type="factual",
            sessions=["session-pass"],
        )
        # Edge 2: contributing session "session-fail"
        real_graph.add_edge(
            "alice",
            "london",
            predicate="visited",
            relation_type="factual",
            sessions=["session-fail"],
        )
        loop.merger.graph = real_graph
        loop.model.peft_config["episodic_interim_20260617T0000"] = _matching_interim_config(
            loop.episodic_config
        )

        _minted_keys: list[str] = []

        def _probe_partial(adapter_name, entries):
            # Collect keys as they are minted; fail the second one.
            for e in entries:
                if e["key"] not in _minted_keys:
                    _minted_keys.append(e["key"])
            # Fail the last-minted key.
            return set(_minted_keys[:-1]) if _minted_keys else set()

        loop._probe_passing_keys = _probe_partial

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            # One dummy episodic relation to bypass guard step 3.
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "p",
                        "predicate": "p",
                        "object": "p",
                        "relation_type": "factual",
                        "speaker_id": "speaker0",
                    }
                ],
                [],
                speaker_id="speaker0",
                mode="train",
                run_label="test",
                schedule="",
                max_interim_count=4,
                stamp="20260617T0000",
            )

        failed = set(result.get("recall_failed_session_ids", []))
        # "session-fail" contributed the failing key → must appear.
        assert "session-fail" in failed, f"session-fail must be in failed; got {failed}"
        # "session-pass" contributed the passing key → must NOT appear.
        assert "session-pass" not in failed, f"session-pass must NOT be in failed; got {failed}"
        # The passing key is registered.
        active = set(loop.store.all_active_keys())
        assert len(active) == 1, f"Exactly 1 key must be registered; got {active}"

    # ------------------------------------------------------------------
    # Test 5 — procedural path: recall failure collects session id
    # ------------------------------------------------------------------

    def test_procedural_recall_failure_populates_recall_failed_session_ids(
        self, tmp_path: Path
    ) -> None:
        """New procedural key that fails the recall gate → session id collected.

        Proc facts flow through merger.graph (merged by extract_session/run_cycle).
        The session_id rides on the graph edge's ``sessions`` set (same path as
        episodic).  When _probe_passing_keys returns an empty set, every new
        key fails and the session id lands in recall_failed_session_ids.
        """
        loop = _make_mock_loop_with_procedural(tmp_path)
        loop.model.peft_config["episodic_interim_20260617T0000"] = _matching_interim_config(
            loop.episodic_config
        )

        proc_sid = "session-proc-fail"
        # Inject the procedural fact into merger.graph with the session_id on
        # the edge's sessions set — that's how extract_session delivers it in prod.
        loop.merger.graph.add_node("Alice", attributes={"name": "Alice"})
        loop.merger.graph.add_node("Tea", attributes={"name": "Tea"})
        loop.merger.graph.add_edge(
            "Alice",
            "Tea",
            predicate="prefers",
            relation_type="preference",
            confidence=1.0,
            sessions={proc_sid},
        )

        # Override probe to fail all keys — every deferred write stays pending.
        loop._probe_passing_keys = lambda adapter_name, entries: set()

        with (
            patch("paramem.memory.interim_adapter.create_interim_adapter"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"train_loss": 0.3, "aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.models.loader.save_adapter"),
        ):
            result = loop.run_consolidation_cycle(
                [],
                [{"relation_type": "preference"}],  # non-empty so no-relations guard passes
                speaker_id="speaker0",
                mode="train",
                run_label="test",
                schedule="",
                max_interim_count=4,
                stamp="20260617T0000",
            )

        failed = result.get("recall_failed_session_ids", [])
        assert proc_sid in failed, (
            f"Procedural recall-failed session {proc_sid!r} must be in "
            f"recall_failed_session_ids; got {failed}"
        )

    # ------------------------------------------------------------------
    # Conditional resolve: failing cycle keeps incident; clean cycle resolves
    # ------------------------------------------------------------------

    def test_s4_failing_cycle_does_not_resolve_recall_failure_incident(
        self, tmp_path: Path
    ) -> None:
        """A cycle returning non-empty recall_failed_session_ids must NOT resolve the incident.

        Ordering hazard: the recall-fail recorder records/bumps when non-empty;
        the success path MUST NOT wipe it in the same cycle.
        """
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

        state_dir = tmp_path / "state"
        sid = "session-s4-fail"
        # Pre-record an incident (simulates a prior cycle having recorded it).
        record_incident(
            state_dir,
            type="consolidation_retry_exhausted",
            key=sid,
            severity="warning",
            summary=f"Session {sid}: facts could not be encoded",
            detail={"session_id": sid},
        )

        # Simulate the conditional: result has a non-empty failed set.
        result_with_failures = {"recall_failed_session_ids": [sid]}
        if not result_with_failures.get("recall_failed_session_ids", []):
            resolve_incidents_by_type(state_dir, "consolidation_retry_exhausted")
        # Since failed is non-empty, we do NOT resolve — incident stays active.

        incidents = read_incidents(state_dir)
        recall_incidents = [i for i in incidents if i.type == "consolidation_retry_exhausted"]
        assert len(recall_incidents) == 1
        assert recall_incidents[0].status == "active", (
            "Incident must remain active when failing cycle runs"
        )

    def test_s4_clean_cycle_resolves_recall_failure_incident(self, tmp_path: Path) -> None:
        """A cycle returning empty recall_failed_session_ids RESOLVES the incident.

        Resolution rule: resolve consolidation_retry_exhausted ONLY when ZERO
        keys failed this cycle.
        """
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

        state_dir = tmp_path / "state"
        sid = "session-s4-clean"
        record_incident(
            state_dir,
            type="consolidation_retry_exhausted",
            key=sid,
            severity="warning",
            summary=f"Session {sid}: facts could not be encoded",
            detail={"session_id": sid},
        )

        # Simulate the conditional: result has an empty failed set.
        result_clean = {"recall_failed_session_ids": []}
        if not result_clean.get("recall_failed_session_ids", []):
            resolve_incidents_by_type(state_dir, "consolidation_retry_exhausted")

        incidents = read_incidents(state_dir)
        recall_incidents = [i for i in incidents if i.type == "consolidation_retry_exhausted"]
        assert len(recall_incidents) == 1
        assert recall_incidents[0].status == "resolved", (
            "Incident must be resolved when clean cycle runs"
        )


# ---------------------------------------------------------------------------
# Interim-tick epoch-level crash-resume (fold_resume.json for scope="interim_slot")
# ---------------------------------------------------------------------------


class TestInterimFoldResume:
    """Interim-tick crash-resume: the fold_resume.json marker persists the REAL
    train_assignment + pending-session scope (mirrors the main_tiers resume
    plumbing in TestFoldResumeHelpers, tests/test_consolidation.py), is read
    back on re-entry, and restores the persisted training set (skipping
    re-extraction) only when the pending-session scope is unchanged.
    """

    @staticmethod
    def _persisted_entry(
        key: str, subject: str, predicate: str, obj: str, speaker_id: str, tier: str
    ) -> dict:
        """Build one enriched persisted train_assignment entry (a "new" key)."""
        return {
            "key": key,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "speaker_id": speaker_id,
            "tier": tier,
            "relation_type": "factual" if tier == "episodic" else "preference",
            "session_ids": ["sess-1"],
            "last_seen": "2026-07-01T00:00:00+00:00",
            "first_seen": "2026-07-01T00:00:00+00:00",
        }

    def test_resume_restores_persisted_assignment_without_reextraction(
        self, tmp_path: Path
    ) -> None:
        """Matching marker (same fold_stamp + same pending-session scope):

        - training receives the persisted entries verbatim (same content, same
          order) — the resumed dataset fingerprints identically to the
          pre-crash one;
        - _materialize_consolidation_graph / _build_all_edge_entries_into /
          _capture_pending_relations are never invoked (no re-extraction);
        - the newly-minted keys still commit to the store post-training (the
          deferred-write metadata — relation_type/last_seen/first_seen —
          round-trips through the marker).
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260701T0000"
        adapter_name = f"episodic_interim_{stamp}"

        fold_stamp = loop._compute_fold_stamp(tier=adapter_name)
        persisted_entries = [
            self._persisted_entry("graph10", "Alice", "likes", "cats", "speaker1", "episodic"),
            self._persisted_entry("proc5", "Bob", "prefers", "coffee", "speaker2", "procedural"),
        ]
        loop._persist_fold_assignment(
            "interim_slot",
            fold_stamp,
            {adapter_name: persisted_entries},
            {adapter_name: "deadbeef"},
            pending_session_ids=["sess-1"],
        )

        trained_entries_calls: list[list[dict]] = []

        def _spy_train(_self, entries, *, adapter_name, **kwargs):
            trained_entries_calls.append(list(entries))
            return {"aborted": False}, None

        def _forbid_materialize(**kw):
            raise AssertionError("_materialize_consolidation_graph must not run on resume")

        def _forbid_build_entries(*a, **kw):
            raise AssertionError("_build_all_edge_entries_into must not run on resume")

        def _forbid_capture_pending():
            raise AssertionError("_capture_pending_relations must not run on resume")

        loop._materialize_consolidation_graph = _forbid_materialize
        loop._build_all_edge_entries_into = _forbid_build_entries
        loop._capture_pending_relations = _forbid_capture_pending

        from paramem.training.consolidation import ConsolidationLoop

        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch.object(ConsolidationLoop, "_train_tier_adapter", _spy_train),
        ):
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Alice",
                        "predicate": "likes",
                        "object": "cats",
                        "session_id": "sess-1",
                    }
                ],
                [
                    {
                        "subject": "Bob",
                        "predicate": "prefers",
                        "object": "coffee",
                        "session_id": "sess-1",
                    }
                ],
                speaker_id="speaker1",
                mode="train",
                run_label="tick-resume",
                stamp=stamp,
            )

        assert result["mode"] == "trained"
        assert len(trained_entries_calls) == 1, (
            f"_train_tier_adapter must be called exactly once; got {trained_entries_calls}"
        )
        got = trained_entries_calls[0]
        want = [
            {k: e[k] for k in ("key", "subject", "predicate", "object", "speaker_id")}
            for e in persisted_entries
        ]
        assert got == want, (
            f"resumed training set must match the persisted assignment verbatim"
            f" (same content, same order); got={got} want={want}"
        )
        # Newly-minted keys committed to the store post-training — proves the
        # deferred-write metadata (relation_type/last_seen/first_seen) round-
        # tripped correctly through the marker's enrichment.
        assert loop.store.get("graph10") is not None
        assert loop.store.get("proc5") is not None
        assert sorted(result["new_keys"]) == ["graph10", "proc5"]
        # Bookkeeping temporal fields must be the FIXTURE's real timestamp, not
        # a silently-dropped "" default — both sides of the round trip use
        # .get(..., "") so a dropped field would pass as "" == "" undetected;
        # asserting the actual non-empty fixture value closes that gap.
        for _key in ("graph10", "proc5"):
            _bk = loop.store.bookkeeping_for_key(_key)
            assert _bk is not None
            assert _bk["last_seen"] == "2026-07-01T00:00:00+00:00", (
                f"{_key}: last_seen must round-trip through the marker; got {_bk['last_seen']!r}"
            )
            assert _bk["first_seen"] == "2026-07-01T00:00:00+00:00", (
                f"{_key}: first_seen must round-trip through the marker; got {_bk['first_seen']!r}"
            )
        # Resume path leaves the marker untouched until commit; a clean commit
        # clears it (see TestInterimFoldResume.test_clean_commit_clears_marker
        # for the dedicated regression coverage of that step).
        assert not (loop._fold_state_dir / "fold_resume.json").exists()

    def test_scope_change_clears_marker_and_takes_fresh_path(self, tmp_path: Path) -> None:
        """Marker present but the pending-session set differs from the marker's:

        the marker is discarded and the fresh-derivation path runs — the stale
        (poisoned) persisted entries must NEVER reach training.
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260701T0000"
        adapter_name = f"episodic_interim_{stamp}"

        # fold_stamp matches (empty active keyset for a brand-new slot, same
        # value the fresh cycle below will compute) — isolates the test to the
        # pending-session-scope dimension specifically.
        fold_stamp = loop._compute_fold_stamp(tier=adapter_name)
        stale_entry = self._persisted_entry(
            "STALE", "Ghost", "haunts", "attic", "speakerX", "episodic"
        )
        loop._persist_fold_assignment(
            "interim_slot",
            fold_stamp,
            {adapter_name: [stale_entry]},
            {adapter_name: "stalefp"},
            pending_session_ids=["sess-OLD"],
        )
        marker_path = loop._fold_state_dir / "fold_resume.json"
        assert marker_path.exists()

        trained_entries_calls: list[list[dict]] = []

        def _spy_train(_self, entries, *, adapter_name, **kwargs):
            trained_entries_calls.append(list(entries))
            return {"aborted": False}, None

        from paramem.training.consolidation import ConsolidationLoop

        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch.object(ConsolidationLoop, "_train_tier_adapter", _spy_train),
            patch.object(
                ConsolidationLoop,
                "_probe_passing_keys",
                side_effect=lambda a, e: {x["key"] for x in e},
            ),
        ):
            loop.run_consolidation_cycle(
                # Different session id than the marker's ["sess-OLD"] — the
                # pending-session scope has changed.
                [
                    {
                        "subject": "Subject1",
                        "predicate": "knows",
                        "object": "Object1",
                        "session_id": "sess-NEW",
                    }
                ],
                [],
                speaker_id="speaker1",
                mode="train",
                run_label="tick-fresh",
                stamp=stamp,
            )

        all_trained_keys = [e["key"] for batch in trained_entries_calls for e in batch]
        assert "STALE" not in all_trained_keys, (
            "Stale (scope-mismatched) marker entries must NEVER reach training;"
            f" all_trained_keys={all_trained_keys}"
        )
        # The fresh path derives its training set from merger.graph (populated
        # by _make_mock_loop with two real keyless "knows" edges), not from the
        # discarded marker.
        assert all_trained_keys, "fresh-derivation path must still train on the real graph content"

    def test_clean_commit_clears_marker(self, tmp_path: Path) -> None:
        """A clean (fresh, non-resumed) interim commit writes a REAL marker
        during the cycle (dataset_fingerprint populated, not the pre-fix `{}`)
        and clears it on successful commit — regression coverage for the
        existing clear-on-success behavior plus the fingerprint fix.
        """
        loop = _make_mock_loop(tmp_path)
        stamp = "20260701T0100"
        adapter_name = f"episodic_interim_{stamp}"
        marker_path = loop._fold_state_dir / "fold_resume.json"
        assert not marker_path.exists()

        persist_calls: list[tuple] = []
        _orig_persist = loop._persist_fold_assignment

        def _spy_persist(scope_name, fold_stamp, train_assignment, dataset_fingerprints, **kw):
            persist_calls.append(
                (scope_name, fold_stamp, train_assignment, dataset_fingerprints, kw)
            )
            return _orig_persist(
                scope_name, fold_stamp, train_assignment, dataset_fingerprints, **kw
            )

        loop._persist_fold_assignment = _spy_persist

        from paramem.training.consolidation import ConsolidationLoop

        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, s: m,
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.memory.persistence.commit_tier_slot"),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.training.consolidation.format_entry_training", return_value=[{}]),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(
                ConsolidationLoop,
                "_probe_passing_keys",
                side_effect=lambda a, e: {x["key"] for x in e},
            ),
        ):
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Subject1",
                        "predicate": "knows",
                        "object": "Object1",
                        "session_id": "sess-clean",
                    }
                ],
                [],
                speaker_id="speaker1",
                mode="train",
                run_label="tick-clean",
                stamp=stamp,
            )

        assert result["mode"] == "trained"
        assert len(persist_calls) == 1, f"expected exactly one persist call; got {persist_calls}"
        _scope_name, _fold_stamp, _train_assignment, _dataset_fingerprints, _kw = persist_calls[0]
        assert _scope_name == "interim_slot"
        assert adapter_name in _train_assignment
        assert _train_assignment[adapter_name], "persisted assignment must not be empty"
        # The fingerprint bug fix: dataset_fingerprints must be REAL (non-empty),
        # not the pre-fix literal `{}`.
        assert _dataset_fingerprints, "dataset_fingerprints must be populated, not {}"
        assert adapter_name in _dataset_fingerprints
        assert _kw.get("pending_session_ids") == ["sess-clean"]
        # A clean commit clears the marker.
        assert not marker_path.exists(), "fold_resume.json must be cleared on a clean commit"
