"""Smoke test for ``configs/server.yaml.example``.

Verifies the shipped template parses cleanly with no env vars set and
that all external-dependency optional services ship disabled-by-default
so a fresh consumer can boot the server without API keys, Home
Assistant, Wyoming STT/TTS endpoints, or pyannote installed.

Regression sentinel: if a future tweak to the example re-enables a
service or introduces a hard env-var requirement, these tests trip in
CI before the example reaches consumers.
"""

from __future__ import annotations

import pytest

from paramem.server.config import load_server_config


@pytest.fixture
def example_config(monkeypatch):
    """Load the example template with all referenced env vars unset.

    The example uses ``${ANTHROPIC_API_KEY}`` / ``${HA_URL}`` etc. as
    placeholders — parsing must not require any of them to be set.
    """
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "HA_URL",
        "HA_TOKEN",
        "PARAMEM_DAILY_PASSPHRASE",
        "PARAMEM_MASTER_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return load_server_config("configs/server.yaml.example")


def test_example_parses_without_env_vars(example_config):
    assert example_config is not None


def test_sota_agent_disabled_by_default(example_config):
    assert example_config.sota_agent.enabled is False


def test_all_sota_providers_disabled_by_default(example_config):
    for name, provider in example_config.sota_providers.items():
        assert provider.enabled is False, f"sota_providers[{name!r}] must default disabled"


def test_ha_agent_id_empty_by_default(example_config):
    assert example_config.ha_agent_id == ""


def test_speaker_disabled_by_default(example_config):
    assert example_config.speaker.enabled is False


def test_stt_disabled_by_default(example_config):
    assert example_config.stt.enabled is False


def test_tts_disabled_by_default(example_config):
    assert example_config.tts.enabled is False


def test_extraction_noise_filter_empty_by_default(example_config):
    assert example_config.consolidation.extraction_noise_filter == ""


def test_graph_enrichment_disabled_by_default(example_config):
    assert example_config.consolidation.graph_enrichment_enabled is False


def test_graph_enrichment_interim_disabled_by_default(example_config):
    assert example_config.consolidation.graph_enrichment_interim_enabled is False


def test_headless_boot_disabled_by_default(example_config):
    assert example_config.headless_boot is False


def test_abstention_remains_enabled(example_config):
    """Safety mechanism — must stay enabled."""
    assert example_config.abstention.enabled is True


def test_local_adapters_remain_enabled(example_config):
    cfg = example_config
    assert cfg.adapters.episodic.enabled is True
    assert cfg.adapters.semantic.enabled is True
    assert cfg.adapters.procedural.enabled is True


def test_cloud_mode_blocks_by_default(example_config):
    """Egress policy ships at the smallest cloud surface."""
    assert example_config.sanitization.cloud_mode == "block"


def test_intent_classifier_enabled(example_config):
    """Intent encoder is local-only (HF download on first use); ships enabled."""
    assert example_config.intent.enabled is True


def test_vram_cap_default(example_config):
    """Process-side VRAM safety net ships at 0.85 (15% headroom)."""
    assert example_config.vram.process_cap_fraction == 0.85


def test_debug_disabled_by_default(example_config):
    """debug=true writes plaintext fact copies to disk on every consolidation
    (SECURITY.md §debug carve-outs). The shipped template must default off so
    a fresh consumer doesn't get a plaintext-on-disk deployment by accident.
    """
    assert example_config.debug is False


def test_role_aware_grounding_off_by_default(example_config):
    """Role-aware grounding gate defaults to "off" (role-blind, backward-compatible).
    Operators measure false-positive rate via "diagnostic" before flipping to "active".
    """
    assert example_config.consolidation.extraction_role_aware_grounding == "off"


def test_no_inline_api_key_literals():
    """Defense-in-depth: the tracked example must never contain a literal-looking
    API key. Operators should always use ${VAR} env-var interpolation. Catches a
    careless edit that pastes a real key into the tracked file.
    """
    import re
    from pathlib import Path

    text = Path("configs/server.yaml.example").read_text()

    # Common provider key prefixes. Patterns require enough following entropy
    # (>=20 alphanumerics) to avoid matching documentation prose like "sk-...".
    # ${VAR} interpolation is allowed and explicitly excluded by the leading
    # character class.
    forbidden = [
        (r"(?<![${\w-])sk-[A-Za-z0-9_-]{20,}", "OpenAI / Anthropic SDK key prefix"),
        (r"(?<![${\w-])sk-ant-[A-Za-z0-9_-]{20,}", "Anthropic console key prefix"),
        (r"(?<![${\w-])AIza[A-Za-z0-9_-]{20,}", "Google API key prefix"),
        (r"(?<![${\w-])gsk_[A-Za-z0-9_-]{20,}", "Groq key prefix"),
        (r"(?<![${\w-])hf_[A-Za-z0-9_-]{20,}", "HuggingFace token prefix"),
    ]
    hits = []
    for pattern, label in forbidden:
        for match in re.finditer(pattern, text):
            hits.append(f"{label}: {match.group()[:12]}... at offset {match.start()}")
    assert not hits, (
        "configs/server.yaml.example contains literal-looking API key(s): "
        + "; ".join(hits)
        + ". Use ${ENV_VAR_NAME} interpolation instead."
    )
