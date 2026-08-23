"""Tests for ``paramem.server.config_store_validator``.

Covers the two checks :func:`check_config_against_store` moved out of
``_load_model_into_state`` — a populated interim ring under a disabled
episodic tier, and a disabled tier whose registry still holds active keys —
their positive counterparts (the same on-disk shapes are unremarkable when
the tier is ENABLED), the pass-silently behaviour against a store that does
not exist yet (the R-PATHS candidate case), and ``ConfigStoreMismatch``'s
``RuntimeError`` shape and required ``check`` parameter.

No GPU, no model — pure filesystem fixtures against a real
:class:`~paramem.server.config.ServerConfig` loaded from the project's
canonical test fixture, matching the pattern every other config-vs-disk
test in this suite uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paramem.server.config import load_server_config
from paramem.server.config_store_validator import ConfigStoreMismatch, check_config_against_store
from paramem.training.key_registry import KeyRegistry


def _server_config(tmp_path: Path):
    """Load the project's canonical test config with ``adapter_dir`` pointed
    at *tmp_path* — fresh per test, no shared on-disk state."""
    cfg = load_server_config("tests/fixtures/server.yaml")
    cfg.paths.data = tmp_path
    return cfg


def test_clean_store_returns_none(tmp_path: Path) -> None:
    """Every main tier enabled, no interim ring, no disabled-tier keys:
    the config does not contradict the store — ``check_config_against_store``
    returns ``None``."""
    cfg = _server_config(tmp_path)
    assert cfg.adapters.episodic.enabled
    assert check_config_against_store(cfg) is None


def test_store_that_does_not_exist_yet_returns_none(tmp_path: Path) -> None:
    """A candidate that re-points ``paths.data`` (an R-PATHS carve) names an
    ``adapter_dir`` nothing has ever been written to. Both checks must pass
    silently against a missing store — a missing store is a pass, not a
    refusal — even with a tier disabled, since there is nothing on disk for
    that tier to contradict."""
    cfg = _server_config(tmp_path / "never-written")
    cfg.adapters.semantic.enabled = False
    assert not (cfg.adapter_dir / "semantic").exists()
    assert check_config_against_store(cfg) is None


def test_interim_ring_under_a_disabled_episodic_tier_refuses_naming_both_remediations(
    tmp_path: Path,
) -> None:
    """Episodic disabled while a populated interim ring still sits on disk
    refuses, naming both operator remediations verbatim (the exact text that
    reaches the 4xx body, the incident detail, and the attention row)."""
    cfg = _server_config(tmp_path)
    cfg.adapters.episodic.enabled = False
    interim_dir = cfg.adapter_dir / "episodic" / "interim_20260101T0000"
    interim_dir.mkdir(parents=True)

    exc_holder: list[ConfigStoreMismatch] = []
    try:
        check_config_against_store(cfg)
    except ConfigStoreMismatch as exc:
        exc_holder.append(exc)

    assert exc_holder, "expected ConfigStoreMismatch to be raised"
    message = str(exc_holder[0])
    assert "adapters.episodic.enabled=false" in message
    assert "1 interim slot(s)" in message
    assert "POST /consolidate to drain the ring into episodic" in message
    assert "POST /interim/discard to destroy it" in message
    assert exc_holder[0].check == "interim_ring_without_episodic"


def test_disabled_tier_holding_active_keys_refuses_naming_the_key_count(
    tmp_path: Path,
) -> None:
    """A disabled tier whose registry still holds active keys refuses,
    naming the exact key count and both remediations."""
    cfg = _server_config(tmp_path)
    cfg.adapters.semantic.enabled = False
    tier_root = cfg.adapter_dir / "semantic"
    tier_root.mkdir(parents=True)
    registry = KeyRegistry()
    registry.add("graph1")
    registry.add("graph2")
    registry.save(tier_root / "indexed_key_registry.json")

    exc_holder: list[ConfigStoreMismatch] = []
    try:
        check_config_against_store(cfg)
    except ConfigStoreMismatch as exc:
        exc_holder.append(exc)

    assert exc_holder, "expected ConfigStoreMismatch to be raised"
    message = str(exc_holder[0])
    assert "adapters.semantic.enabled=false" in message
    assert "2 active key(s)" in message
    assert "Drain the tier (fold/promote its keys elsewhere)" in message
    assert "Erase its keys" in message
    assert exc_holder[0].check == "disabled_tier_active_keys:semantic"


def test_refusal_is_a_runtime_error_subclass(tmp_path: Path) -> None:
    """``ConfigStoreMismatch`` is a ``RuntimeError`` subclass — existing
    broad-exception handling (the boot lifespan's degrade-vs-abort
    classification) keeps its shape unchanged."""
    cfg = _server_config(tmp_path)
    cfg.adapters.episodic.enabled = False
    (cfg.adapter_dir / "episodic" / "interim_20260101T0000").mkdir(parents=True)

    raised: list[BaseException] = []
    try:
        check_config_against_store(cfg)
    except ConfigStoreMismatch as exc:
        raised.append(exc)

    assert raised, "expected ConfigStoreMismatch to be raised"
    assert isinstance(raised[0], RuntimeError)


def test_config_store_mismatch_check_is_required() -> None:
    """``check`` has no default — every construction site (production and
    test doubles alike) must name which check produced the refusal, so the
    incident dedup key downstream is never silently generic."""
    with pytest.raises(TypeError):
        ConfigStoreMismatch("refusal text")  # type: ignore[call-arg]


def test_config_store_mismatch_constructs_with_message_and_check() -> None:
    exc = ConfigStoreMismatch("refusal text", check="interim_ring_without_episodic")
    assert str(exc) == "refusal text"
    assert isinstance(exc, RuntimeError)
    assert exc.check == "interim_ring_without_episodic"


def test_interim_ring_under_an_enabled_episodic_tier_passes(tmp_path: Path) -> None:
    """A populated interim ring is exactly what an ENABLED episodic tier
    expects to find — it is the tier's own ring, not a contradiction.
    ``check_config_against_store`` must return ``None``."""
    cfg = _server_config(tmp_path)
    assert cfg.adapters.episodic.enabled
    interim_dir = cfg.adapter_dir / "episodic" / "interim_20260101T0000"
    interim_dir.mkdir(parents=True)
    assert check_config_against_store(cfg) is None


def test_disabled_tier_registry_active_keys_on_an_enabled_tier_passes(tmp_path: Path) -> None:
    """A tier registry holding active keys is unremarkable when that tier is
    ENABLED — the keys are reachable, mounted, and rebuilt at the next fold
    like any other. ``check_config_against_store`` must return ``None``."""
    cfg = _server_config(tmp_path)
    assert cfg.adapters.semantic.enabled
    tier_root = cfg.adapter_dir / "semantic"
    tier_root.mkdir(parents=True)
    registry = KeyRegistry()
    registry.add("graph1")
    registry.add("graph2")
    registry.save(tier_root / "indexed_key_registry.json")
    assert check_config_against_store(cfg) is None
