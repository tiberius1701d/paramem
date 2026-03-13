"""Tests for configuration loading."""

from paramem.utils.config import NeuroMemConfig, load_config


def test_load_default_config():
    config = load_config("configs/default.yaml")
    assert isinstance(config, NeuroMemConfig)
    assert config.model.model_id == "Qwen/Qwen2.5-3B"
    assert config.model.quantization == "nf4"


def test_adapters_loaded():
    config = load_config("configs/default.yaml")
    assert "episodic" in config.adapters
    assert "semantic" in config.adapters
    assert config.adapters["episodic"].rank == 8
    assert config.adapters["semantic"].rank == 24


def test_training_defaults():
    config = load_config("configs/default.yaml")
    assert config.training.batch_size == 1
    assert config.training.gradient_checkpointing is True


def test_missing_config_uses_defaults():
    config = load_config("nonexistent.yaml")
    assert isinstance(config, NeuroMemConfig)
    assert config.model.model_id == "Qwen/Qwen2.5-3B"
