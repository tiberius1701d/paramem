"""Tests for configuration loading.

Meta-test of ``paramem.utils.config.load_config`` -- the archived loader that
backs ``archive/experiments/*.py`` (Paper v1 reference scripts). Active
code paths do not call ``load_config``; enforced by
``tests/test_test_config_loader_usage.py``.
"""

from paramem.utils.config import ParaMemConfig, load_config

ARCHIVED_YAML = "archive/configs/default.yaml"


def test_load_default_config():
    config = load_config(ARCHIVED_YAML)
    assert isinstance(config, ParaMemConfig)
    assert config.model.model_id == "Qwen/Qwen2.5-3B"
    assert config.model.quantization == "nf4"


def test_adapters_loaded():
    config = load_config(ARCHIVED_YAML)
    assert "episodic" in config.adapters
    assert "semantic" in config.adapters
    assert config.adapters["episodic"].rank == 8
    assert config.adapters["semantic"].rank == 24


def test_training_defaults():
    config = load_config(ARCHIVED_YAML)
    assert config.training.batch_size == 1
    assert config.training.gradient_checkpointing is True


def test_missing_config_uses_defaults():
    config = load_config("nonexistent.yaml")
    assert isinstance(config, ParaMemConfig)
    assert config.model.model_id == "Qwen/Qwen2.5-3B"
