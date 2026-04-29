"""Parity contract: experiment-scenario factory calls thread every
ServerConfig knob into ConsolidationLoop.

When a new pipeline knob is added to ConsolidationScheduleConfig, the
canonical create_consolidation_loop must thread it. This test fails
whenever the factory drops a knob, so experiments stay in sync with
production by construction.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_dotted(obj, dotted_key: str, value):
    """Set a dotted-path attribute on a nested dataclass tree.

    Walks the chain of getattr calls and sets the terminal attribute.
    """
    parts = dotted_key.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


# ---------------------------------------------------------------------------
# Parametrised scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    pytest.param(
        "dataset_probe",
        {
            "model_name": "mistral",
            "consolidation.max_epochs": 20,
            "consolidation.indexed_key_replay": True,
            "consolidation.extraction_noise_filter": "",
            "consolidation.extraction_plausibility_judge": "off",
            "adapters.procedural.enabled": False,
        },
        {
            "output_dir": "RUNDIR_SENTINEL",
            "save_cycle_snapshots": False,
            "persist_graph": False,
            "seed_state_from_disk": False,
        },
        id="dataset_probe",
    ),
    pytest.param(
        "step6_step7",
        {
            "model_name": "mistral",
            "consolidation.max_epochs": 10,
            "consolidation.indexed_key_replay": True,
            "consolidation.promotion_threshold": 3,
            "consolidation.extraction_stt_correction": False,
            "consolidation.extraction_ha_validation": False,
            "consolidation.extraction_noise_filter": "",
            "consolidation.extraction_plausibility_judge": "off",
            "consolidation.extraction_verify_anonymization": False,
            "adapters.episodic.rank": 8,
            "adapters.episodic.alpha": 16,
            "adapters.episodic.learning_rate": 1e-4,
            "adapters.semantic.rank": 8,
            "adapters.semantic.alpha": 16,
            "adapters.semantic.learning_rate": 1e-4,
            "adapters.procedural.enabled": True,
            "adapters.procedural.rank": 8,
            "adapters.procedural.alpha": 16,
            "adapters.procedural.learning_rate": 1e-4,
        },
        {
            "output_dir": "RUNDIR_SENTINEL",
            "save_cycle_snapshots": False,
            "persist_graph": False,
            "seed_state_from_disk": False,
        },
        id="step6_step7",
    ),
    pytest.param(
        "interim_rollover",
        {
            "model_name": "mistral",
            "consolidation.max_epochs": 10,
            "consolidation.indexed_key_replay": True,
            "consolidation.promotion_threshold": 3,
            "consolidation.extraction_stt_correction": False,
            "consolidation.extraction_ha_validation": False,
            "consolidation.extraction_noise_filter": "anthropic",
            "consolidation.extraction_noise_filter_model": "claude-sonnet-4-6",
            "consolidation.extraction_plausibility_judge": "off",
            "consolidation.extraction_verify_anonymization": False,
            "consolidation.graph_enrichment_enabled": True,
            "consolidation.graph_enrichment_neighborhood_hops": 1,
            "consolidation.graph_enrichment_max_entities_per_pass": 400,
            "consolidation.graph_enrichment_interim_enabled": True,
            "consolidation.graph_enrichment_min_triples_floor": 1,
            "adapters.episodic.rank": 8,
            "adapters.episodic.alpha": 16,
            "adapters.episodic.learning_rate": 1e-4,
            "adapters.semantic.rank": 8,
            "adapters.semantic.alpha": 16,
            "adapters.semantic.learning_rate": 1e-4,
            "adapters.procedural.enabled": True,
            "adapters.procedural.rank": 8,
            "adapters.procedural.alpha": 16,
            "adapters.procedural.learning_rate": 1e-4,
        },
        {
            "output_dir": "RUNDIR_SENTINEL",
            "save_cycle_snapshots": False,
            "persist_graph": False,
            "seed_state_from_disk": False,
        },
        id="interim_rollover",
    ),
]


@pytest.mark.parametrize("scenario_id, cfg_mutations, factory_kwargs", SCENARIOS)
def test_factory_threads_every_config_knob(
    scenario_id, cfg_mutations, factory_kwargs, tmp_path, monkeypatch
):
    """Factory passes every ServerConfig knob through to ConsolidationLoop.

    Each scenario represents an experiment callsite's cfg mutations and
    factory kwargs. The test asserts that the factory correctly threads
    all knobs — both the ones that were present before the migration and
    the ones added as part of it.
    """
    from paramem.server import consolidation as server_consolidation
    from paramem.server.config import load_server_config

    captured = {}

    def _fake_loop(**kwargs):
        captured.update(kwargs)
        instance = MagicMock()
        instance._kwargs = kwargs
        return instance

    monkeypatch.setattr(server_consolidation, "ConsolidationLoop", _fake_loop)

    cfg = load_server_config("tests/fixtures/server.yaml")
    # Re-route paths so the factory's seed-from-disk block has nothing to read.
    cfg.paths.data = tmp_path
    cfg.paths.simulate = tmp_path / "simulate"
    cfg.paths.debug = tmp_path / "debug"

    for dotted_key, value in cfg_mutations.items():
        _set_dotted(cfg, dotted_key, value)

    # Resolve the output_dir sentinel to a real tmp path.
    kw = dict(factory_kwargs)
    if kw.get("output_dir") == "RUNDIR_SENTINEL":
        kw["output_dir"] = tmp_path / "rundir"

    server_consolidation.create_consolidation_loop(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=cfg,
        **kw,
    )

    # --- Adapter configs ---
    assert captured["episodic_adapter_config"] == cfg.episodic_adapter_config
    assert captured["semantic_adapter_config"] == cfg.semantic_adapter_config
    if cfg.adapters.procedural.enabled:
        assert captured["procedural_adapter_config"] == cfg.procedural_adapter_config
    else:
        assert captured["procedural_adapter_config"] is None

    # --- Training and consolidation configs ---
    assert captured["training_config"] == cfg.training_config
    assert captured["consolidation_config"] == cfg.consolidation_config

    # --- output_dir override ---
    expected = kw["output_dir"] if kw.get("output_dir") is not None else cfg.adapter_dir
    assert captured["output_dir"] == expected

    # --- save_cycle_snapshots override ---
    expected_scs = (
        kw["save_cycle_snapshots"] if kw.get("save_cycle_snapshots") is not None else cfg.debug
    )
    assert captured["save_cycle_snapshots"] == expected_scs

    # --- persist_graph override ---
    expected_pg = kw["persist_graph"] if kw.get("persist_graph") is not None else False
    assert captured["persist_graph"] == expected_pg

    # --- Graph and enrichment knobs ---
    assert captured["graph_config"] == cfg.graph_config
    assert captured["graph_enrichment_enabled"] == cfg.consolidation.graph_enrichment_enabled
    assert (
        captured["graph_enrichment_neighborhood_hops"]
        == cfg.consolidation.graph_enrichment_neighborhood_hops
    )
    assert (
        captured["graph_enrichment_max_entities_per_pass"]
        == cfg.consolidation.graph_enrichment_max_entities_per_pass
    )
    assert (
        captured["graph_enrichment_interim_enabled"]
        == cfg.consolidation.graph_enrichment_interim_enabled
    )
    assert (
        captured["graph_enrichment_min_triples_floor"]
        == cfg.consolidation.graph_enrichment_min_triples_floor
    )

    # --- Extraction knobs ---
    assert captured["extraction_stt_correction"] == cfg.consolidation.extraction_stt_correction
    assert captured["extraction_ha_validation"] == cfg.consolidation.extraction_ha_validation
    assert captured["extraction_noise_filter"] == cfg.consolidation.extraction_noise_filter
    assert (
        captured["extraction_noise_filter_model"] == cfg.consolidation.extraction_noise_filter_model
    )
    assert (
        captured["extraction_noise_filter_endpoint"]
        == cfg.consolidation.extraction_noise_filter_endpoint
    )
    assert captured["extraction_ner_check"] == cfg.consolidation.extraction_ner_check
    assert captured["extraction_ner_model"] == cfg.consolidation.extraction_ner_model
    assert (
        captured["extraction_plausibility_judge"] == cfg.consolidation.extraction_plausibility_judge
    )
    assert (
        captured["extraction_plausibility_stage"] == cfg.consolidation.extraction_plausibility_stage
    )
    assert captured["extraction_verify_anonymization"] == (
        cfg.consolidation.extraction_verify_anonymization
    )
    assert captured["extraction_max_tokens"] == cfg.consolidation.extraction_max_tokens
    assert (
        captured["extraction_role_aware_grounding"]
        == cfg.consolidation.extraction_role_aware_grounding
    )
    assert captured["extraction_pii_scope"] == set(cfg.sanitization.cloud_scope)

    # --- Misc knobs ---
    assert captured["prompts_dir"] == cfg.prompts_dir
    assert captured["state_provider"] is None
    assert captured["extraction_temperature"] == 0.0


# ---------------------------------------------------------------------------
# Seed-gate tests
# ---------------------------------------------------------------------------


def test_factory_skips_seeding_when_seed_state_from_disk_false(tmp_path, monkeypatch):
    """When seed_state_from_disk=False, the factory does not call any
    seed_* method on the loop — even when disk paths exist with data.
    """
    from paramem.server import consolidation as server_consolidation
    from paramem.server.config import load_server_config

    # Place a sentinel keyed_pairs.json so the seeding block WOULD fire if
    # seed_state_from_disk were True.
    adapters_dir = tmp_path / "adapters"
    ep_dir = adapters_dir / "episodic"
    ep_dir.mkdir(parents=True)
    (ep_dir / "keyed_pairs.json").write_text('{"k1": {"question": "Q", "answer": "A"}}')

    loop_instance = MagicMock()

    def _fake_loop(**kwargs):
        return loop_instance

    monkeypatch.setattr(server_consolidation, "ConsolidationLoop", _fake_loop)

    cfg = load_server_config("tests/fixtures/server.yaml")
    cfg.paths.data = tmp_path
    cfg.paths.simulate = tmp_path / "simulate"
    cfg.paths.debug = tmp_path / "debug"

    server_consolidation.create_consolidation_loop(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=cfg,
        seed_state_from_disk=False,
    )

    loop_instance.seed_key_metadata.assert_not_called()
    loop_instance.seed_episodic_qa.assert_not_called()
    loop_instance.seed_semantic_qa.assert_not_called()
    loop_instance.seed_procedural_qa.assert_not_called()


def test_factory_seeds_from_disk_by_default(tmp_path, monkeypatch):
    """Production callers (no explicit seed_state_from_disk) get the
    historical seeding behaviour. A sentinel keyed_pairs.json under the
    adapter dir triggers seed_episodic_qa.
    """
    from paramem.server import consolidation as server_consolidation
    from paramem.server.config import load_server_config

    adapters_dir = tmp_path / "adapters"
    ep_dir = adapters_dir / "episodic"
    ep_dir.mkdir(parents=True)
    ep_payload = {"k1": {"question": "Who is Q?", "answer": "A person."}}
    (ep_dir / "keyed_pairs.json").write_text(json.dumps(ep_payload))

    loop_instance = MagicMock()

    def _fake_loop(**kwargs):
        return loop_instance

    monkeypatch.setattr(server_consolidation, "ConsolidationLoop", _fake_loop)

    cfg = load_server_config("tests/fixtures/server.yaml")
    cfg.paths.data = tmp_path
    cfg.paths.simulate = tmp_path / "simulate"
    cfg.paths.debug = tmp_path / "debug"

    server_consolidation.create_consolidation_loop(
        model=MagicMock(),
        tokenizer=MagicMock(),
        config=cfg,
        # seed_state_from_disk defaults to True
    )

    loop_instance.seed_episodic_qa.assert_called_once_with(ep_payload)
