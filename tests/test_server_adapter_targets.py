"""Tests for per-adapter `target_modules` in the server config.

Procedural adapter targets attention + MLP layers (representational imprinting
for persistent preferences). Episodic + semantic adapters target attention only
(indexed-key retrieval is a routing problem). The server deployment honours
per-adapter targeting from `ServerAdapterConfig` — previously hardcoded to
attention-only for all three.
"""

from __future__ import annotations

from paramem.server.config import ServerAdapterConfig, ServerConfig


def test_episodic_defaults_to_attention_only():
    cfg = ServerConfig()
    adapter = cfg._make_adapter_config(cfg.adapters.episodic)
    assert adapter.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]


def test_semantic_defaults_to_attention_only():
    cfg = ServerConfig()
    adapter = cfg._make_adapter_config(cfg.adapters.semantic)
    assert adapter.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]


def test_procedural_defaults_to_attention_plus_mlp():
    """Procedural adapter must target MLP layers in addition to attention —
    design intent for persistent preferences/habits."""
    cfg = ServerConfig()
    adapter = cfg._make_adapter_config(cfg.adapters.procedural)
    for m in ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        assert m in adapter.target_modules, f"procedural adapter missing {m}"


def test_override_per_adapter_targets():
    """User can override via explicit ServerAdapterConfig construction."""
    custom = ServerAdapterConfig(target_modules=["q_proj", "o_proj"])
    cfg = ServerConfig()
    cfg.adapters.procedural = custom
    adapter = cfg._make_adapter_config(cfg.adapters.procedural)
    assert adapter.target_modules == ["q_proj", "o_proj"]


def test_make_adapter_config_isolates_mutations():
    """Mutating the returned AdapterConfig's target_modules must not leak back
    into the ServerAdapterConfig source list."""
    cfg = ServerConfig()
    original = list(cfg.adapters.procedural.target_modules)
    built = cfg._make_adapter_config(cfg.adapters.procedural)
    built.target_modules.append("mutated")
    assert cfg.adapters.procedural.target_modules == original
