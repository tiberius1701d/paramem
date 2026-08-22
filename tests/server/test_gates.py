"""Tests for paramem.server.gates.

All tests run without GPU.  The model and tokenizer are MagicMocks.
``paramem.training.indexed_memory.probe_key`` is patched in every test that
exercises gates 3 or 4 in QA mode.  ``paramem.training.recall_eval.probe_entries``
is patched for quad-mode tests.  Registry files are written to ``tmp_path``.

Coverage:
  - Each gate's pass/fail/skip paths including the 2 new skip conditions.
  - Gate 4 deterministic sample stability.
  - Gate 4 retry seed produces a different list.
  - Phase categorizer: extraction exception → gate 1 FAIL, gate 2 SKIPPED.
  - Phase categorizer: training exception → gate 1 PASS, gate 2 FAIL.
  - Phase categorizer logs at WARNING.
  - Unmount: delete_adapter called when > 1 adapter mounted.
  - Unmount: delete_adapter NOT called when trial_probe is the sole adapter.
  - Unmount survives delete_adapter raising.
  - Gate 3: read_keyed_pairs + probe_entries → PASS / FAIL.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from peft import PeftModel

from paramem.server.gates import (
    _TRIAL_PROBE_ADAPTER_NAME,
    GATE_4_SAMPLE_SIZE,
    GateResult,
    _ensure_trial_probe_mounted,
    _find_tier_registry,
    _gate_1_extraction,
    _gate_2_training,
    _gate_3_reload_smoke,
    _is_training_marker,
    _sample_registry_keys,
    _unmount_trial_probe,
)
from paramem.training.key_registry import KeyRegistry
from paramem.utils.tiers import MAIN_TIERS


@pytest.fixture(autouse=True)
def _no_real_sleep_in_mount(monkeypatch):
    """Skip the WSL2 settle sleep in unit tests.

    `_settle_cuda_and_load_adapter` calls `time.sleep(_MOUNT_INITIAL_SETTLE_SECONDS)`
    (3s) before the first mount attempt to let the WSL2 driver recover after
    a heavy training pass. In tests with mocked GPU/model that wait is dead
    weight — patching it cuts ~30s off the gate suite.
    """
    monkeypatch.setattr("paramem.server.gates.time.sleep", lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trial_adapter(tmp_path: Path, with_registry: bool = True) -> Path:
    """Create a minimal trial adapter directory with placeholder files.

    Writes ``episodic/indexed_key_registry.json`` (the canonical probe-key
    source for gate 3) so gate 3 can locate the first key to probe without
    reading any ``quads.json`` sidecar.

    Parameters
    ----------
    with_registry:
        When True (default), writes ``episodic/indexed_key_registry.json``
        with two synthetic keys.  Pass False to test the no-registry path.
    """
    d = tmp_path / "trial_adapter"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
    if with_registry:
        episodic_dir = d / "episodic"
        episodic_dir.mkdir(parents=True, exist_ok=True)
        # Write new per-tier KeyRegistry schema.
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        (episodic_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())
    return d


def _make_mock_model(adapter_names: list[str] | None = None) -> MagicMock:
    """Build a MagicMock(spec=PeftModel) that mimics a PeftModel with named adapters.

    ``spec=PeftModel`` makes ``isinstance(model, PeftModel)`` True (required by
    ``mount_adapter``'s precondition) while still allowing every attribute the
    tests in this module configure directly (``peft_config``, ``active_adapter``,
    ``load_adapter``, ``delete_adapter``, ``set_adapter`` are all real PeftModel
    attributes/methods, so ``spec`` does not block them).
    """
    model = MagicMock(spec=PeftModel)
    if adapter_names is None:
        adapter_names = ["episodic"]
    # peft_config is a dict keyed by adapter name, auto-vivifying on lookup
    # of a name not yet present -- mirrors real PEFT's load_adapter, which
    # inserts the entry as a side effect before mount_adapter's own
    # base_model_name_or_path check reads it back.
    model.peft_config = defaultdict(MagicMock, {name: MagicMock() for name in adapter_names})
    model.active_adapter = adapter_names[0] if adapter_names else None
    # Attributes torch.nn.Module / transformers.PreTrainedModel expose on a
    # real (wrapped) PeftModel via dynamic __getattr__ delegation rather than
    # as class-level attributes -- MagicMock(spec=PeftModel) cannot see these
    # through dir(PeftModel), so gates.py's calls to them need an explicit
    # mock in place before first access.
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    return model


# ---------------------------------------------------------------------------
# GateResult.to_dict
# ---------------------------------------------------------------------------


class TestGateResultToDict:
    def test_to_dict_pass(self):
        r = GateResult(gate=1, name="extraction", status="pass", reason=None, metrics=None)
        d = r.to_dict()
        assert d["gate"] == 1
        assert d["name"] == "extraction"
        assert d["status"] == "pass"
        assert d["reason"] is None
        assert d["metrics"] is None

    def test_to_dict_fail_with_metrics(self):
        m = {"recalled": 15, "sampled": 20}
        r = GateResult(gate=4, name="live_registry_recall", status="fail", reason="low", metrics=m)
        d = r.to_dict()
        assert d["metrics"] == m
        assert d["status"] == "fail"


# ---------------------------------------------------------------------------
# _sample_registry_keys
# ---------------------------------------------------------------------------


class TestSampleRegistryKeys:
    def _make_content(self, n: int) -> tuple[bytes, list[str]]:
        """Build KeyRegistry JSON bytes + sorted population for n keys.

        Returns ``(registry_content, population)`` — the same pair
        :func:`_live_key_population` hands to :func:`_sample_registry_keys`.
        """
        reg = KeyRegistry()
        for i in range(1, n + 1):
            reg.add(f"graph{i}")
        return reg.save_bytes(), sorted(reg.list_active())

    def test_stable_same_input(self):
        """Same bytes → same list every time (deterministic)."""
        content, population = self._make_content(50)
        keys1 = _sample_registry_keys(content, population)
        keys2 = _sample_registry_keys(content, population)
        assert keys1 == keys2

    def test_sample_size_capped_at_20(self):
        content, population = self._make_content(50)
        keys = _sample_registry_keys(content, population)
        assert len(keys) == GATE_4_SAMPLE_SIZE

    def test_sample_size_less_than_20(self):
        content, population = self._make_content(10)
        keys = _sample_registry_keys(content, population)
        assert len(keys) == 10

    def test_retry_suffix_produces_different_list(self):
        """seed_suffix=b'|retry' must produce a different sample."""
        content, population = self._make_content(50)
        keys_first = _sample_registry_keys(content, population, seed_suffix=b"")
        keys_retry = _sample_registry_keys(content, population, seed_suffix=b"|retry")
        # Very unlikely to be identical with 50 keys and sample of 20.
        assert keys_first != keys_retry

    def test_sorted_population(self):
        """All returned keys must be from the registry's active_keys."""
        content, population = self._make_content(30)
        all_keys = set(population)
        keys = _sample_registry_keys(content, population)
        assert set(keys).issubset(all_keys)


# ---------------------------------------------------------------------------
# _is_training_marker
# ---------------------------------------------------------------------------


class TestIsTrainingMarker:
    def test_nan_loss_matches(self):
        exc = RuntimeError("train_loss nan detected at step 100")
        assert _is_training_marker(exc) is True

    def test_oom_matches(self):
        exc = RuntimeError("CUDA out of memory at layer 12")
        assert _is_training_marker(exc) is True

    def test_safetensors_matches(self):
        exc = OSError("failed to write adapter_model.safetensors")
        assert _is_training_marker(exc) is True

    def test_extraction_exception_no_match(self):
        exc = ValueError("JSON parse error in extraction output")
        assert _is_training_marker(exc) is False

    def test_logs_at_warning(self, caplog):
        """Phase categorizer must log at WARNING level."""
        exc = RuntimeError("something happened")
        import paramem.server.gates as gates_mod

        gates_mod.logger.propagate = True
        with caplog.at_level(logging.WARNING):
            _is_training_marker(exc)
        assert any("phase-categorizer" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Gate 1 — extraction
# ---------------------------------------------------------------------------


class TestGate1Extraction:
    def test_pass_no_exception(self):
        g = _gate_1_extraction(
            session_buffer_empty=False,
            summary={"status": "complete"},
            exc=None,
        )
        assert g.status == "pass"
        assert g.gate == 1

    def test_skipped_empty_buffer(self):
        g = _gate_1_extraction(
            session_buffer_empty=True,
            summary=None,
            exc=None,
        )
        assert g.status == "skipped"
        assert g.reason == "no_new_sessions"

    def test_fail_extraction_exception(self):
        exc = ValueError("JSON parse error in extraction output")
        g = _gate_1_extraction(
            session_buffer_empty=False,
            summary=None,
            exc=exc,
        )
        assert g.status == "fail"
        assert "ValueError" in g.reason

    def test_pass_training_exception_extraction_completed(self):
        """Training marker → gate 1 PASS (extraction ran, training failed)."""
        exc = RuntimeError("train_loss nan")
        g = _gate_1_extraction(
            session_buffer_empty=False,
            summary=None,
            exc=exc,
        )
        assert g.status == "pass"


# ---------------------------------------------------------------------------
# Gate 2 — training
# ---------------------------------------------------------------------------


class TestGate2Training:
    def test_pass_complete_with_adapter_files(self, tmp_path):
        trial_adapter_dir = _make_trial_adapter(tmp_path)
        g = _gate_2_training(
            session_buffer_empty=False,
            summary={"status": "complete"},
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "pass"
        assert g.gate == 2

    def test_pass_simulated_with_adapter_files(self, tmp_path):
        trial_adapter_dir = _make_trial_adapter(tmp_path)
        g = _gate_2_training(
            session_buffer_empty=False,
            summary={"status": "simulated"},
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "pass"

    def test_skipped_no_facts(self, tmp_path):
        """no_facts status → SKIPPED, not PASS."""
        trial_adapter_dir = tmp_path / "trial_adapter"
        g = _gate_2_training(
            session_buffer_empty=False,
            summary={"status": "no_facts"},
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "skipped"
        assert "no_training_attempted" in g.reason

    def test_skipped_no_pending(self, tmp_path):
        trial_adapter_dir = tmp_path / "trial_adapter"
        g = _gate_2_training(
            session_buffer_empty=False,
            summary={"status": "no_pending"},
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "skipped"

    def test_skipped_disabled(self, tmp_path):
        trial_adapter_dir = tmp_path / "trial_adapter"
        g = _gate_2_training(
            session_buffer_empty=False,
            summary={"status": "disabled"},
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "skipped"

    def test_skipped_empty_buffer(self, tmp_path):
        trial_adapter_dir = _make_trial_adapter(tmp_path)
        g = _gate_2_training(
            session_buffer_empty=True,
            summary=None,
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "skipped"
        assert g.reason == "no_new_sessions"

    def test_fail_training_exception(self, tmp_path):
        exc = RuntimeError("train_loss nan at step 50")
        trial_adapter_dir = _make_trial_adapter(tmp_path)
        g = _gate_2_training(
            session_buffer_empty=False,
            summary=None,
            exc=exc,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "fail"
        assert "training exception" in g.reason

    def test_fail_complete_but_no_adapter_files(self, tmp_path):
        """status==complete but empty adapter dir → FAIL."""
        trial_adapter_dir = tmp_path / "trial_adapter"
        trial_adapter_dir.mkdir()
        g = _gate_2_training(
            session_buffer_empty=False,
            summary={"status": "complete"},
            exc=None,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "fail"
        assert "empty or missing" in g.reason

    def test_skipped_when_extraction_exception(self, tmp_path):
        """Extraction exception → gate 2 SKIPPED (not reached)."""
        exc = ValueError("JSON parse error in extraction output")
        trial_adapter_dir = _make_trial_adapter(tmp_path)
        g = _gate_2_training(
            session_buffer_empty=False,
            summary=None,
            exc=exc,
            trial_adapter_dir=trial_adapter_dir,
        )
        assert g.status == "skipped"
        assert "extraction failed" in g.reason


# ---------------------------------------------------------------------------
# Gate 3 — adapter_reload smoke
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Gate 4 — live_registry_recall
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unmount helpers
# ---------------------------------------------------------------------------


class TestUnmountTrialProbe:
    def test_delete_called_when_multiple_adapters(self):
        """delete_adapter called when > 1 adapter is loaded."""
        model = _make_mock_model(["episodic", _TRIAL_PROBE_ADAPTER_NAME])
        mount_state = {"mounted": True, "pre_active_adapter": ["episodic"]}
        _unmount_trial_probe(model, mount_state)
        model.delete_adapter.assert_called_once_with("trial_probe")

    def test_unmount_survives_delete_raising(self):
        """delete_adapter raising must not propagate out of _unmount_trial_probe."""
        model = _make_mock_model(["episodic", "trial_probe"])
        model.delete_adapter.side_effect = RuntimeError("delete failed")
        mount_state = {"mounted": True, "pre_active_adapter": ["episodic"]}
        # Must not raise.
        _unmount_trial_probe(model, mount_state)
        assert mount_state["mounted"] is False

    def test_noop_when_not_mounted(self):
        """If mount_state['mounted'] is False, nothing happens."""
        model = _make_mock_model()
        mount_state = {"mounted": False, "pre_active_adapter": []}
        _unmount_trial_probe(model, mount_state)
        model.delete_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# _ensure_trial_probe_mounted
# ---------------------------------------------------------------------------


class TestEnsureTrialProbeMounted:
    def test_stores_pre_active_adapter(self, tmp_path):
        """_ensure_trial_probe_mounted records pre-mount state and mounts.

        The trial dir has episodic/indexed_key_registry.json and the mock model
        has peft_config["episodic"], so the in-memory path is taken:
        set_adapter is called with the kind name, not load_adapter from disk.
        """
        model = _make_mock_model(["episodic"])
        trial_dir = _make_trial_adapter(tmp_path)
        mount_state: dict = {}
        _ensure_trial_probe_mounted(model, trial_dir, mount_state)
        assert mount_state["pre_active_adapter"] == ["episodic"]
        assert mount_state["mounted"] is True
        assert mount_state["mounted_via"] == "set"
        model.set_adapter.assert_called_once_with("episodic")
        model.load_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# Phase categorizer integration: extraction vs. training exception routing
# ---------------------------------------------------------------------------


class TestPhaseCategorizer:
    def test_extraction_exc_gate1_fail_gate2_skipped(self, tmp_path):
        """Non-training exception → gate 1 FAIL, gate 2 SKIPPED."""
        trial_dir = _make_trial_adapter(tmp_path)
        exc = ValueError("JSON parse error in extraction output")

        g1 = _gate_1_extraction(session_buffer_empty=False, summary=None, exc=exc)
        g2 = _gate_2_training(
            session_buffer_empty=False,
            summary=None,
            exc=exc,
            trial_adapter_dir=trial_dir,
        )
        assert g1.status == "fail"
        assert g2.status == "skipped"

    def test_training_exc_gate1_pass_gate2_fail(self, tmp_path):
        """Training marker → gate 1 PASS, gate 2 FAIL."""
        trial_dir = _make_trial_adapter(tmp_path)
        exc = RuntimeError("train_loss nan at step 50")

        g1 = _gate_1_extraction(session_buffer_empty=False, summary=None, exc=exc)
        g2 = _gate_2_training(
            session_buffer_empty=False,
            summary=None,
            exc=exc,
            trial_adapter_dir=trial_dir,
        )
        assert g1.status == "pass"
        assert g2.status == "fail"

    def test_phase_categorizer_logs_warning(self, caplog):
        """Phase categorizer must log at WARNING regardless of match."""
        import paramem.server.gates as gates_mod

        gates_mod.logger.propagate = True
        exc = RuntimeError("something unusual")
        with caplog.at_level(logging.WARNING):
            _is_training_marker(exc)
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("phase-categorizer" in m for m in warning_msgs)


# ---------------------------------------------------------------------------
# gates module import hygiene
# ---------------------------------------------------------------------------


class TestGatesModuleNoGpuImport:
    def test_no_gpu_import_at_module_level(self):
        """Acceptance criterion C — gates module must not import torch at top level."""

        # If torch/peft/transformers were imported by gates, they appear in sys.modules
        # only if they were installed; the key test is that gates.py itself doesn't
        # unconditionally import them.  We verify by checking the module source.
        import inspect

        import paramem.server.gates as gates_mod

        source = inspect.getsource(gates_mod)
        # Top-level imports are before the first function/class definition.
        # The module must not have bare 'import torch' / 'import peft' lines.
        top_level_lines = []
        for line in source.splitlines():
            if line.startswith("def ") or line.startswith("class "):
                break
            top_level_lines.append(line)

        top_level_src = "\n".join(top_level_lines)
        assert "import torch" not in top_level_src
        assert "import peft" not in top_level_src
        assert "import transformers" not in top_level_src


# ---------------------------------------------------------------------------
# Gate 3
# ---------------------------------------------------------------------------


def _make_trial_adapter_quad(tmp_path: Path) -> Path:
    """Create a minimal trial adapter directory for gate 3 tests.

    Writes ``episodic/indexed_key_registry.json`` (the canonical key source
    for gate 3 in both QA and quad modes).
    """
    d = tmp_path / "trial_adapter_quad"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
    episodic_dir = d / "episodic"
    episodic_dir.mkdir(parents=True, exist_ok=True)
    # Write new per-tier KeyRegistry schema.
    reg = KeyRegistry()
    reg.add("graph1")
    reg.add("graph2")
    (episodic_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())
    return d


class TestGate3AdapterReloadQuad:
    """Gate 3 quad-path: indexed_key_registry.json + probe_entries dispatch."""

    def test_pass_successful_quad_probe(self, tmp_path):
        """Successful mount + probe_entries → PASS."""
        trial_dir = _make_trial_adapter_quad(tmp_path)
        model = _make_mock_model()
        tokenizer = MagicMock()

        probe_result = {
            "key": "graph1",
            "subject": "Alex",
            "predicate": "lives_in",
            "object": "Heilbronn",
            "confidence": 0.95,
            "raw_output": (
                '{"key": "graph1", "subject": "Alex", '
                '"predicate": "lives_in", "object": "Heilbronn"}'
            ),
        }

        def _probe_gen(m, tok, entries, **kw):
            for e in entries:
                yield e, probe_result

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            g = _gate_3_reload_smoke(
                session_buffer_empty=False,
                summary={"status": "complete"},
                model=model,
                tokenizer=tokenizer,
                trial_adapter_dir=trial_dir,
                mount_state={},
            )

        assert g.status == "pass"
        assert g.gate == 3

    def test_fail_probe_quad_returns_failure_reason(self, tmp_path):
        """probe_entries returning failure_reason dict → gate 3 FAIL."""
        trial_dir = _make_trial_adapter_quad(tmp_path)
        model = _make_mock_model()

        fail_result = {"raw_output": "", "failure_reason": "quad_parse_failure"}

        def _probe_gen(m, tok, entries, **kw):
            for e in entries:
                yield e, fail_result

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            g = _gate_3_reload_smoke(
                session_buffer_empty=False,
                summary={"status": "complete"},
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                mount_state={},
            )

        assert g.status == "fail"
        assert "quad_parse_failure" in g.reason

    def test_skipped_empty_buffer_quad(self, tmp_path):
        """session_buffer_empty=True → SKIPPED."""
        g = _gate_3_reload_smoke(
            session_buffer_empty=True,
            summary=None,
            model=_make_mock_model(),
            tokenizer=MagicMock(),
            trial_adapter_dir=tmp_path / "trial_adapter",
            mount_state={},
        )
        assert g.status == "skipped"

    def test_foreign_shaped_registry_fails_not_raises(self, tmp_path):
        """A foreign-shaped indexed_key_registry.json (missing 'simhash')
        must surface as GateResult(status="fail") with a read-failure
        reason — not propagate KeyRegistry.load's ValueError past the gate.

        Regression for the strict-load collapse in _gate_3_reload_smoke:
        the dead "legacy/flat registries" hand-parse fallback was deleted
        in favor of KeyRegistry.load being the single reader, so this shape
        must now fail at the FIRST read (all_keys), not silently produce a
        key that later dies at the simhash load."""
        d = tmp_path / "trial_adapter_foreign"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
        episodic_dir = d / "episodic"
        episodic_dir.mkdir(parents=True, exist_ok=True)
        # Foreign-shaped: active_keys present, but no simhash section.
        (episodic_dir / "indexed_key_registry.json").write_text(
            json.dumps({"active_keys": ["graph1", "graph2"]})
        )

        g = _gate_3_reload_smoke(
            session_buffer_empty=False,
            summary={"status": "complete"},
            model=_make_mock_model(),
            tokenizer=MagicMock(),
            trial_adapter_dir=d,
            mount_state={},
        )

        assert g.status == "fail"
        assert g.gate == 3
        assert "failed to read indexed_key_registry.json" in g.reason
        assert "simhash" in g.reason

    # test_default_qa_path_unchanged: removed with QA-format retirement.


# ---------------------------------------------------------------------------
# Gate 4
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# KeyRegistry.load_simhashes — the shared helper and the schema-mismatch bug
# it fixes
# ---------------------------------------------------------------------------


class TestLoadSimhashRegistry:
    """The gates' fingerprint reader must refuse to silently coerce a
    non-KeyRegistry file (e.g. key_metadata.json, accidentally pointed at)
    into a simhash map.

    The reader itself now lives on the class that owns the serialization
    (``KeyRegistry.load_simhashes``) — gates.py no longer carries its own
    copy — but the contract these assertions pin is unchanged.
    """

    def test_missing_file_returns_empty(self, tmp_path):
        assert KeyRegistry.load_simhashes(tmp_path / "missing.json") == {}

    def test_keyregistry_shape_extracts_simhash(self, tmp_path):
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 999)
        p = tmp_path / "indexed_key_registry.json"
        p.write_bytes(reg.save_bytes())
        assert KeyRegistry.load_simhashes(p) == {"graph1": 999}

    def test_gates_module_has_no_private_copy(self):
        """gates.py must not carry a second fingerprint-file reader.

        The duplicate ``_load_simhash_registry`` leaf was collapsed into
        ``KeyRegistry.load_simhashes``; re-adding one here would put the
        file-shape knowledge, the encryption read and the wrong-file guard in
        two places again.
        """
        import paramem.server.gates as gates_mod

        assert not hasattr(gates_mod, "_load_simhash_registry")


class TestGate3RealConfidenceVerification:
    """Gate 3 must actually verify recall content against the trial adapter's
    own SimHash fingerprint — not just parse JSON. Uses REAL
    verify_confidence (nothing mocked) to prove the fix is not vacuous, and
    that the extracted {key: simhash} map (not the raw KeyRegistry dict) is
    what gets wired in — the "obvious fix" the bug report warns is wrong
    would score every key 0.0, failing even matching content.
    """

    def _write_trial_registry(self, tmp_path: Path, key: str, fp: int) -> Path:
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
        episodic_dir = trial_dir / "episodic"
        episodic_dir.mkdir()
        reg = KeyRegistry()
        reg.add(key)
        reg.set_simhash(key, fp)
        (episodic_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())
        return trial_dir

    def test_matching_content_passes(self, tmp_path):
        from paramem.memory.entry import entry_simhash, finalize_recalled

        fp = entry_simhash(
            {"key": "graph1", "subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}
        )
        trial_dir = self._write_trial_registry(tmp_path, "graph1", fp)
        model = _make_mock_model()
        matching_raw = (
            '{"key": "graph1", "subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}'
        )

        def _probe_gen(m, tok, entries, registry=None, **kw):
            for e in entries:
                yield e, finalize_recalled(matching_raw, e["key"], registry, 0.75)

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            g = _gate_3_reload_smoke(
                session_buffer_empty=False,
                summary={"status": "complete"},
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                mount_state={},
            )

        assert g.status == "pass", g.reason

    def test_mismatched_donor_content_fails(self, tmp_path):
        """Warm-start-donor scenario named in the bug report: well-formed
        JSON, correct key, WRONG content — must FAIL, not pass."""
        from paramem.memory.entry import entry_simhash, finalize_recalled

        fp = entry_simhash(
            {"key": "graph1", "subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}
        )
        trial_dir = self._write_trial_registry(tmp_path, "graph1", fp)
        model = _make_mock_model()
        donor_raw = (
            '{"key": "graph1", "subject": "Priya", "predicate": "works_at", "object": "Globex"}'
        )

        def _probe_gen(m, tok, entries, registry=None, **kw):
            for e in entries:
                yield e, finalize_recalled(donor_raw, e["key"], registry, 0.75)

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            g = _gate_3_reload_smoke(
                session_buffer_empty=False,
                summary={"status": "complete"},
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                mount_state={},
            )

        assert g.status == "fail", "well-formed JSON with wrong content must be caught, not pass"
        assert "low_confidence" in g.reason


# ---------------------------------------------------------------------------
# Gate 3 per-kind subdir layout: _find_tier_registry helper
# ---------------------------------------------------------------------------


class TestGate3KindSubdirLayout:
    """Gate 3 must find indexed_key_registry.json under per-kind subdirs.

    Real trial training writes per-kind layout:
        <trial_adapter_dir>/episodic/indexed_key_registry.json
        <trial_adapter_dir>/semantic/indexed_key_registry.json
        <trial_adapter_dir>/procedural/indexed_key_registry.json

    Gate 3 uses ``_find_tier_registry`` to locate the registry file and probes
    the first key in it.  No quads.json sidecar is read.
    """

    def _registry_content(self) -> bytes:
        """Return minimal indexed_key_registry.json content bytes.

        Uses the per-tier KeyRegistry schema:
        ``{active_keys: [...], stale: {...}, simhash: {...}}``.
        """
        reg = KeyRegistry()
        reg.add("graph1")
        return reg.save_bytes()

    def test_find_tier_registry_episodic_subdir(self, tmp_path):
        """episodic/indexed_key_registry.json is found by _find_tier_registry."""
        d = tmp_path / "trial_adapter"
        d.mkdir()
        episodic_dir = d / "episodic"
        episodic_dir.mkdir()
        (episodic_dir / "indexed_key_registry.json").write_bytes(self._registry_content())
        result = _find_tier_registry(d)
        assert result is not None
        kind, path = result
        assert kind == "episodic"
        assert path == episodic_dir / "indexed_key_registry.json"

    def test_find_tier_registry_returns_none_when_absent(self, tmp_path):
        """Empty trial_adapter_dir → _find_tier_registry returns None."""
        d = tmp_path / "trial_adapter"
        d.mkdir()
        assert _find_tier_registry(d) is None

    def test_gate3_finds_registry_under_episodic_subdir(self, tmp_path):
        """Fixture: episodic/indexed_key_registry.json only.

        Gate 3 must PASS when the registry is present in the episodic subdir.
        """
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        # Adapter files at top-level (required by gate 3 mount step).
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
        # indexed_key_registry.json in episodic subdir.
        episodic_dir = trial_dir / "episodic"
        episodic_dir.mkdir()
        (episodic_dir / "indexed_key_registry.json").write_bytes(self._registry_content())

        model = _make_mock_model()
        probe_result = {
            "key": "graph1",
            "subject": "S",
            "predicate": "p",
            "object": "O",
            "confidence": 0.99,
            "raw_output": '{"key": "graph1", "subject": "S", "predicate": "p", "object": "O"}',
        }

        def _probe_gen(m, tok, entries, **kw):
            for e in entries:
                yield e, probe_result

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            g = _gate_3_reload_smoke(
                session_buffer_empty=False,
                summary={"status": "complete"},
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                mount_state={},
            )

        assert g.status == "pass", f"Expected pass, got {g.status}: {g.reason}"

    def test_gate3_finds_registry_under_semantic_when_episodic_missing(self, tmp_path):
        """Fixture: semantic/indexed_key_registry.json only (no episodic).

        Gate 3 must find the registry in the semantic subdir.
        """
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
        # indexed_key_registry.json only in semantic subdir.
        semantic_dir = trial_dir / "semantic"
        semantic_dir.mkdir()
        (semantic_dir / "indexed_key_registry.json").write_bytes(self._registry_content())

        model = _make_mock_model()
        probe_result = {
            "key": "graph1",
            "subject": "S",
            "predicate": "p",
            "object": "O",
            "confidence": 0.99,
            "raw_output": '{"key": "graph1", "subject": "S", "predicate": "p", "object": "O"}',
        }

        def _probe_gen(m, tok, entries, **kw):
            for e in entries:
                yield e, probe_result

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            g = _gate_3_reload_smoke(
                session_buffer_empty=False,
                summary={"status": "complete"},
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                mount_state={},
            )

        assert g.status == "pass", f"Expected pass from semantic subdir, got {g.status}: {g.reason}"

    def test_gate3_skips_when_no_kind_subdir_has_registry(self, tmp_path):
        """Empty trial_adapter_dir (no indexed_key_registry.json anywhere) → gate 3 SKIPPED.

        This matches the 'no kind-specific adapter trained' case — e.g. when
        extraction ran but produced no facts.
        """
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        # Create kind subdirs without indexed_key_registry.json.
        for kind in MAIN_TIERS:
            (trial_dir / kind).mkdir()

        model = _make_mock_model()
        g = _gate_3_reload_smoke(
            session_buffer_empty=False,
            summary={"status": "complete"},
            model=model,
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            mount_state={},
        )

        assert g.status == "skipped", (
            "Expected skipped when no indexed_key_registry.json in any location, "
            f"got {g.status}: {g.reason}"
        )
        assert "no kind-specific adapter trained" in (g.reason or "")


class TestTrialProbeMountResolvesKindSubdir:
    """PEFT ``load_adapter`` does not walk subdirs. The trial layout writes
    adapter_model.safetensors into per-kind subdirs (episodic/, semantic/,
    procedural/), so the mount path must resolve to the kind subdir, not the
    trial_adapter root.
    """

    def test_resolver_picks_episodic_slot_first(self, tmp_path):
        """Under the timestamped-slot layout: <kind>/<ts>/adapter_model.safetensors.
        Episodic wins over semantic/procedural; newest slot wins within episodic."""
        from paramem.server.gates import _resolve_adapter_mount_path

        trial_dir = tmp_path / "trial_adapter"
        for kind in ("episodic", "semantic", "procedural"):
            slot = trial_dir / kind / "20260423-100000"
            slot.mkdir(parents=True)
            (slot / "adapter_model.safetensors").write_bytes(b"\x00")
        assert _resolve_adapter_mount_path(trial_dir) == trial_dir / "episodic" / "20260423-100000"

    def test_resolver_picks_newest_slot(self, tmp_path):
        """Two slots in episodic/ → mtime-newest wins."""
        from paramem.server.gates import _resolve_adapter_mount_path

        trial_dir = tmp_path / "trial_adapter"
        old = trial_dir / "episodic" / "20260101-000000"
        new = trial_dir / "episodic" / "20260423-100000"
        for slot in (old, new):
            slot.mkdir(parents=True)
            (slot / "adapter_model.safetensors").write_bytes(b"\x00")
        # Force old to be older.
        import os
        import time

        past = time.time() - 3600
        os.utime(old / "adapter_model.safetensors", (past, past))
        os.utime(old, (past, past))
        assert _resolve_adapter_mount_path(trial_dir) == new

    def test_resolver_skips_pending(self, tmp_path):
        """``.pending/`` (in-progress write) must not be picked as a live slot."""
        from paramem.server.gates import _resolve_adapter_mount_path

        trial_dir = tmp_path / "trial_adapter"
        (trial_dir / "episodic" / ".pending").mkdir(parents=True)
        (trial_dir / "episodic" / ".pending" / "adapter_model.safetensors").write_bytes(b"\x00")
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")
        assert _resolve_adapter_mount_path(trial_dir) == slot

    def test_resolver_falls_back_to_flat_per_kind(self, tmp_path):
        """Legacy flat per-kind: <kind>/adapter_model.safetensors directly."""
        from paramem.server.gates import _resolve_adapter_mount_path

        trial_dir = tmp_path / "trial_adapter"
        (trial_dir / "semantic").mkdir(parents=True)
        (trial_dir / "semantic" / "adapter_model.safetensors").write_bytes(b"\x00")
        assert _resolve_adapter_mount_path(trial_dir) == trial_dir / "semantic"

    def test_resolver_falls_back_to_root_for_top_level_layout(self, tmp_path):
        """Top-level layout: trial_adapter/adapter_model.safetensors."""
        from paramem.server.gates import _resolve_adapter_mount_path

        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00")
        assert _resolve_adapter_mount_path(trial_dir) == trial_dir

    def test_mount_prefers_in_memory_set_adapter(self, tmp_path):
        """When trial just trained ``episodic`` in-memory and
        ``episodic/indexed_key_registry.json`` exists, mount must use
        ``set_adapter("episodic")`` instead of ``load_adapter``. This avoids
        the WSL2 CUDA driver instability that ``load_adapter`` triggers
        immediately after a heavy training pass."""
        from paramem.server.gates import _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")
        # _find_trained_kind_in_memory matches via indexed_key_registry.json (existence only).
        (trial_dir / "episodic").mkdir(exist_ok=True)
        _reg_ep1 = KeyRegistry()
        _reg_ep1.add("graph1")
        (trial_dir / "episodic" / "indexed_key_registry.json").write_bytes(_reg_ep1.save_bytes())

        model = _make_mock_model(adapter_names=["episodic"])
        mount_state: dict = {"mounted": False, "pre_active_adapter": []}
        _ensure_trial_probe_mounted(model, trial_dir, mount_state)

        model.set_adapter.assert_called_with("episodic")
        model.load_adapter.assert_not_called()
        assert mount_state["mounted_via"] == "set"
        assert mount_state["mounted_name"] == "episodic"

    def test_mount_disables_gradient_checkpointing_and_unmount_restores(self, tmp_path):
        """CLAUDE.md rule: gradient_checkpointing must be OFF during model.generate().
        Trial training leaves it ON; mount must disable + eval(), unmount restores
        the prior state. Without this, probe_key gets garbage output → parse_failure.
        """
        from paramem.server.gates import _ensure_trial_probe_mounted, _unmount_trial_probe

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")
        (trial_dir / "episodic").mkdir(exist_ok=True)
        _reg_ep2 = KeyRegistry()
        _reg_ep2.add("graph1")
        (trial_dir / "episodic" / "indexed_key_registry.json").write_bytes(_reg_ep2.save_bytes())

        model = _make_mock_model(adapter_names=["episodic"])
        # Trial training leaves the model in train mode + checkpointing enabled.
        model.is_gradient_checkpointing = True
        model.training = True

        mount_state: dict = {"mounted": False, "pre_active_adapter": []}
        _ensure_trial_probe_mounted(model, trial_dir, mount_state)

        # Mount must have disabled checkpointing and called eval().
        model.gradient_checkpointing_disable.assert_called()
        model.eval.assert_called()
        assert mount_state["pre_checkpointing"] is True
        assert mount_state["pre_training_mode"] is True

        # Unmount must restore the prior state.
        _unmount_trial_probe(model, mount_state)
        model.gradient_checkpointing_enable.assert_called()
        model.train.assert_called()

    def test_mount_picks_kind_matching_registry_location(self, tmp_path):
        """When indexed_key_registry.json is in procedural/ (only proc trained),
        mount must activate ``procedural`` — not the first kind alphabetically.
        Activating the wrong kind produces ``parse_failure`` because the
        probed adapter has never seen the key."""
        from paramem.server.gates import _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        for kind in ("episodic", "semantic", "procedural"):
            slot = trial_dir / kind / "20260423-100000"
            slot.mkdir(parents=True)
            (slot / "adapter_model.safetensors").write_bytes(b"\x00")
        # Only procedural has the registry — episodic/semantic do not.
        _reg_proc = KeyRegistry()
        _reg_proc.add("proc1")
        (trial_dir / "procedural" / "indexed_key_registry.json").write_bytes(_reg_proc.save_bytes())

        model = _make_mock_model(adapter_names=["episodic", "semantic", "procedural"])
        mount_state: dict = {"mounted": False, "pre_active_adapter": []}
        _ensure_trial_probe_mounted(model, trial_dir, mount_state)

        model.set_adapter.assert_called_with("procedural")
        assert mount_state["mounted_name"] == "procedural"

    def test_mount_falls_back_to_load_when_in_memory_missing(self, tmp_path):
        """When episodic isn't in peft_config (e.g. lifespan crash recovery
        rebuilt the model fresh), mount falls back to load_adapter from disk
        with the resolved per-kind slot path."""
        from unittest.mock import patch

        from paramem.server.gates import _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        # Empty peft_config — no in-memory trial adapter, must fall back.
        model = _make_mock_model(adapter_names=[])
        mount_state: dict = {"mounted": False, "pre_active_adapter": []}
        with patch("paramem.server.gates.time.sleep"):
            _ensure_trial_probe_mounted(model, trial_dir, mount_state)

        (called_path,) = model.load_adapter.call_args[0]
        assert called_path == str(slot)
        assert mount_state["mounted_via"] == "load"

    def test_mount_drains_cuda_before_load(self, tmp_path):
        """torch.cuda.synchronize() must be called before load_adapter so the
        WSL2 driver isn't caught mid-training-batch by the mount call.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        # Empty peft_config forces the load_adapter fallback path.
        model = _make_mock_model(adapter_names=[])
        call_order: list[str] = []
        model.load_adapter.side_effect = lambda *a, **k: call_order.append("load_adapter")

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.synchronize.side_effect = lambda: call_order.append("synchronize")

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.server.gates.time.sleep"),
        ):
            _ensure_trial_probe_mounted(
                model, trial_dir, {"mounted": False, "pre_active_adapter": []}
            )

        # synchronize fires twice (initial settle + per-attempt) before load_adapter.
        assert call_order[0] == "synchronize", (
            f"synchronize must precede load_adapter, got {call_order}"
        )
        assert "load_adapter" in call_order
        assert call_order.index("synchronize") < call_order.index("load_adapter")

    def test_mount_initial_settle_sleeps(self, tmp_path):
        """The pre-attempt settle sleep must fire (WSL2 driver needs wall-clock
        time after a heavy training pass; synchronize() returns too quickly).
        Tests the load-fallback path (in-memory adapter unavailable)."""
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import (
            _MOUNT_INITIAL_SETTLE_SECONDS,
            _ensure_trial_probe_mounted,
        )

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        # Force the load-fallback path.
        model = _make_mock_model(adapter_names=[])
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.server.gates.time.sleep") as sleep_mock,
        ):
            _ensure_trial_probe_mounted(
                model, trial_dir, {"mounted": False, "pre_active_adapter": []}
            )

        # First sleep call must be the initial settle.
        assert sleep_mock.call_args_list[0][0][0] == _MOUNT_INITIAL_SETTLE_SECONDS

    def test_mount_aborts_on_cuda_allocator_corruption(self, tmp_path):
        """CUDACachingAllocator INTERNAL ASSERT means PyTorch state is corrupt.
        Retries cannot recover — the loop must abort immediately.
        Tests the load-fallback path."""
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        model = _make_mock_model(adapter_names=[])
        model.load_adapter.side_effect = RuntimeError(
            'INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":419'
        )

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.server.gates.time.sleep"),
        ):
            try:
                _ensure_trial_probe_mounted(
                    model, trial_dir, {"mounted": False, "pre_active_adapter": []}
                )
                raised = False
            except RuntimeError:
                raised = True

        assert raised, "expected re-raise of allocator-corruption error"
        # Only ONE attempt should fire — the loop must abort on terminal markers.
        assert model.load_adapter.call_count == 1, (
            f"expected 1 attempt (no retries on allocator corruption), "
            f"got {model.load_adapter.call_count}"
        )

    def test_mount_retries_on_transient_cuda_failure(self, tmp_path):
        """First mount fails with 'device not ready', second succeeds — the
        retry loop must swallow the transient error and return cleanly."""
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import _MOUNT_RETRY_COUNT, _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        # Empty peft_config forces the load_adapter fallback path.
        model = _make_mock_model(adapter_names=[])
        attempts: list[int] = []

        def _flaky_load(*_a, **_k):
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise RuntimeError("CUDA driver error: device not ready")

        model.load_adapter.side_effect = _flaky_load

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.server.gates.time.sleep"),
        ):
            _ensure_trial_probe_mounted(
                model, trial_dir, {"mounted": False, "pre_active_adapter": []}
            )

        assert len(attempts) == 2, f"expected 2 attempts (1 fail + 1 retry), got {len(attempts)}"
        assert _MOUNT_RETRY_COUNT >= 2

    def test_mount_cleans_up_half_registered_adapter_between_retries(self, tmp_path):
        """PEFT registers the adapter name BEFORE moving weights to GPU. A
        CUDA failure during load_adapter leaves the name in peft_config,
        causing the retry to die with 'Adapter already exists'. The retry
        loop must delete the half-registered adapter before each attempt.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        # Empty peft_config forces the load_adapter fallback path.
        model = _make_mock_model(adapter_names=[])
        # Simulate PEFT half-registering: first call adds 'trial_probe' to
        # peft_config and then raises (mimicking the WSL2 CUDA path).
        attempt_count = 0

        def _flaky_load(*_a, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            model.peft_config["trial_probe"] = MagicMock()  # leave half-registered
            if attempt_count == 1:
                raise RuntimeError("CUDA driver error: device not ready")
            # On retry, no exception — but if cleanup didn't run, PEFT would
            # have raised "Adapter with name trial_probe already exists".

        model.load_adapter.side_effect = _flaky_load

        # delete_adapter must remove the half-registered entry.
        def _delete(name):
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.server.gates.time.sleep"),
        ):
            _ensure_trial_probe_mounted(
                model, trial_dir, {"mounted": False, "pre_active_adapter": []}
            )

        # delete_adapter must have been called between attempts to clean up
        # the partial registration from attempt 1.
        assert model.delete_adapter.called, (
            "delete_adapter must run before retry to clear half-registered name"
        )
        assert attempt_count == 2, f"expected 2 mount attempts, got {attempt_count}"

    def test_mount_raises_when_all_retries_exhausted(self, tmp_path):
        """If every attempt fails, the last exception is re-raised so gate 3
        FAILs with the operator-actionable error text."""
        from unittest.mock import MagicMock, patch

        from paramem.server.gates import _MOUNT_RETRY_COUNT, _ensure_trial_probe_mounted

        trial_dir = tmp_path / "trial_adapter"
        slot = trial_dir / "episodic" / "20260423-100000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"\x00")

        # Empty peft_config forces the load_adapter fallback path.
        model = _make_mock_model(adapter_names=[])
        model.load_adapter.side_effect = RuntimeError("persistent CUDA failure")

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True

        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.server.gates.time.sleep"),
        ):
            try:
                _ensure_trial_probe_mounted(
                    model, trial_dir, {"mounted": False, "pre_active_adapter": []}
                )
                raised = False
            except RuntimeError as exc:
                raised = True
                assert "persistent CUDA failure" in str(exc)

        assert raised, "expected RuntimeError after all retries exhausted"
        assert model.load_adapter.call_count == _MOUNT_RETRY_COUNT


# ---------------------------------------------------------------------------
# The trial-tree layout ConsolidationLoop.commit_main_tiers actually writes,
# read back by the real gate readers (_resolve_adapter_mount_path,
# KeyRegistry.load_simhashes) — closes the gap left by _make_trial_adapter*
# hand-building the layout by hand.
# ---------------------------------------------------------------------------


class TestGateReadersBindTheProductionWriterLayout:
    """The trial layout ``ConsolidationLoop.commit_main_tiers`` writes is
    exactly what ``_resolve_adapter_mount_path`` and
    ``KeyRegistry.load_simhashes`` bind — generated by the production writer
    itself, not a hand-built fixture, so a writer-side layout regression
    surfaces here."""

    def test_mount_path_and_simhash_reader_bind_a_real_commit_main_tiers_write(self, tmp_path):
        from paramem.memory.store import MemoryStore
        from paramem.server.gates import _resolve_adapter_mount_path
        from paramem.training.consolidation import ConsolidationLoop
        from tests._fold_fixtures import _FakeModel, _FakeTokenizer

        trial_root = tmp_path / "trial_adapter"

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = _FakeModel()
        loop.tokenizer = _FakeTokenizer()
        loop.output_dir = trial_root
        loop.fingerprint_cache = None
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.cycle_count = 0
        loop._keep_prior_slots = 5

        store = MemoryStore()
        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 123456789)
        store.load_registry("episodic", reg)
        store.put(
            "episodic",
            "graph1",
            {
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "sp1",
            },
            register=False,
        )
        store.set_bookkeeping(
            "graph1", speaker_id="sp1", relation_type="factual", first_seen="", promoted=False
        )
        loop.store = store

        committed = loop.commit_main_tiers(["episodic"], output_dir=trial_root)
        assert committed == {"episodic"}

        mount_path = _resolve_adapter_mount_path(trial_root)
        assert mount_path.parent == trial_root / "episodic", (
            f"the resolver must bind commit_main_tiers's own per-kind slot layout; got {mount_path}"
        )
        assert (mount_path / "adapter_model.safetensors").exists()

        loaded_simhash = KeyRegistry.load_simhashes(
            trial_root / "episodic" / "indexed_key_registry.json"
        )
        assert loaded_simhash == {"graph1": 123456789}, (
            "the gate's per-kind fingerprint read must bind commit_main_tiers's "
            f"own committed registry; got {loaded_simhash}"
        )


# ---------------------------------------------------------------------------
# _find_tier_registry — per-kind indexed_key_registry.json locator
# ---------------------------------------------------------------------------


class TestFindTierRegistry:
    """_find_tier_registry returns (kind, path) for the first kind subdir that
    contains an indexed_key_registry.json, or None when none exists."""

    def test_episodic_registry_found(self, tmp_path):
        """Returns (kind, path) when episodic/indexed_key_registry.json exists."""
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()

        ep_dir = trial_dir / "episodic"
        ep_dir.mkdir()
        registry = ep_dir / "indexed_key_registry.json"
        registry.write_text(json.dumps({"graph1": 0x1234567890ABCDEF}))

        result = _find_tier_registry(trial_dir)

        assert result is not None, "_find_tier_registry returned None when episodic registry exists"
        kind, path = result
        assert kind == "episodic"
        assert path == registry

    def test_semantic_fallback_when_no_episodic(self, tmp_path):
        """Returns semantic registry when only semantic/indexed_key_registry.json exists."""
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()

        sem_dir = trial_dir / "semantic"
        sem_dir.mkdir()
        registry = sem_dir / "indexed_key_registry.json"
        registry.write_text(json.dumps({"graph2": 0xFEDCBA9876543210}))

        result = _find_tier_registry(trial_dir)

        assert result is not None
        kind, path = result
        assert kind == "semantic"
        assert path == registry

    def test_none_returned_when_no_registry_anywhere(self, tmp_path):
        """Returns None when no indexed_key_registry.json exists in any tier subdir."""
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()

        result = _find_tier_registry(trial_dir)
        assert result is None


# ---------------------------------------------------------------------------
# Gate 4 — runs against the live per-tier registries, no global file anywhere
# ---------------------------------------------------------------------------


class TestGate4RunsAgainstPerTierRegistries:
    """Gate 4's population + GATE_4_MIN_REGISTRY_SIZE precondition are drawn
    from the live adapter store's per-tier ``indexed_key_registry.json``
    files (_live_key_population / iter_tier_roots) -- there is no global
    ``key_metadata.json`` sample-population file to read."""

    def _live_registry_with_n_keys(self, tmp_path: Path, n: int) -> Path:
        """A live adapter store whose episodic tier alone carries *n* active
        keys -- no key_metadata.json anywhere, main or interim."""
        live_dir = tmp_path / "live_adapters"
        episodic_dir = live_dir / "episodic"
        episodic_dir.mkdir(parents=True)
        reg = KeyRegistry()
        for i in range(n):
            reg.add(f"graph{i}")
        (episodic_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())
        assert not any(live_dir.rglob("key_metadata.json"))
        return live_dir

    def _trial_adapter_with_matching_registry(
        self, tmp_path: Path, key_fingerprints: dict[str, int]
    ) -> Path:
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
        episodic_dir = trial_dir / "episodic"
        episodic_dir.mkdir()
        reg = KeyRegistry()
        for k, fp in key_fingerprints.items():
            reg.add(k)
            reg.set_simhash(k, fp)
        (episodic_dir / "indexed_key_registry.json").write_bytes(reg.save_bytes())
        return trial_dir

    def test_runs_not_skipped_at_the_min_registry_size_threshold(self, tmp_path):
        import json as _json

        from paramem.memory.entry import entry_simhash, finalize_recalled
        from paramem.server.gates import GATE_4_MIN_REGISTRY_SIZE, _gate_4_recall_check

        keys = [f"graph{i}" for i in range(GATE_4_MIN_REGISTRY_SIZE)]
        live_dir = self._live_registry_with_n_keys(tmp_path, GATE_4_MIN_REGISTRY_SIZE)

        def _fact_for(key: str) -> dict:
            return {"key": key, "subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}

        key_fingerprints = {k: entry_simhash(_fact_for(k)) for k in keys}
        trial_dir = self._trial_adapter_with_matching_registry(tmp_path, key_fingerprints)

        def _probe_gen(m, tok, entries, registry=None, **kw):
            for e in entries:
                raw = _json.dumps(_fact_for(e["key"]))
                yield e, finalize_recalled(raw, e["key"], registry, 0.75)

        model = _make_mock_model(["episodic"])
        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            result = _gate_4_recall_check(
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                live_adapter_dir=live_dir,
                mount_state={},
                recall_probe_batch_size=4,
            )

        assert result.status != "skipped", (
            f"gate 4 must run at exactly GATE_4_MIN_REGISTRY_SIZE ({GATE_4_MIN_REGISTRY_SIZE}) "
            f"live keys with no global key_metadata.json anywhere; got status={result.status!r} "
            f"reason={result.reason!r}"
        )
        assert result.status == "pass", result.reason
        assert result.metrics["sampled"] == GATE_4_MIN_REGISTRY_SIZE

    def test_skipped_one_key_below_the_threshold(self, tmp_path):
        """One key short of the threshold still SKIPS -- the boundary is
        exact, not off-by-one, under the per-tier population."""
        from paramem.server.gates import GATE_4_MIN_REGISTRY_SIZE, _gate_4_recall_check

        live_dir = self._live_registry_with_n_keys(tmp_path, GATE_4_MIN_REGISTRY_SIZE - 1)
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)

        result = _gate_4_recall_check(
            model=_make_mock_model(["episodic"]),
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            live_adapter_dir=live_dir,
            mount_state={},
            recall_probe_batch_size=4,
        )

        assert result.status == "skipped"

    def test_write_only_trial_shape_with_no_registry_fails_zero_of_twenty(self, tmp_path):
        """The trial adapter has been written (weights on disk) but never
        published (no ``indexed_key_registry.json`` under any kind subdir --
        the shape a write-without-publish crash leaves behind). ``gate 4``
        must not skip: it runs, ``load_simhashes`` yields ``{}`` for a
        missing file, ``verify_confidence`` returns 0.0 for every key
        against that empty map, and the gate fails 0/20 rather than passing
        vacuously or crashing."""
        from paramem.server.gates import GATE_4_MIN_REGISTRY_SIZE, _gate_4_recall_check

        keys = [f"graph{i}" for i in range(GATE_4_MIN_REGISTRY_SIZE)]
        live_dir = self._live_registry_with_n_keys(tmp_path, GATE_4_MIN_REGISTRY_SIZE)

        # Written weights only -- no per-kind indexed_key_registry.json at all.
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)
        assert not any(trial_dir.rglob("indexed_key_registry.json"))

        def _probe_gen(m, tok, entries, registry=None, **kw):
            assert registry == {}
            for e in entries:
                yield e, {"key": e["key"], "subject": "x", "predicate": "y", "object": "z"}

        model = _make_mock_model(["episodic"])
        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            result = _gate_4_recall_check(
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                live_adapter_dir=live_dir,
                mount_state={},
                recall_probe_batch_size=4,
            )

        assert result.status == "fail"
        assert result.metrics["recalled"] == 0
        assert result.metrics["sampled"] == GATE_4_MIN_REGISTRY_SIZE
        assert keys  # sample population is non-empty; the failure is real, not vacuous

    def test_registry_read_os_error_fails_gate_4_instead_of_escaping(self, tmp_path):
        """A decrypt/permission ``OSError`` reading a live per-tier registry
        must surface as a gate-4 ``fail`` result -- not escape uncaught to
        the generic ``trial_exception`` path, which would leave the failure
        unattributed to the gate that actually hit it."""
        from paramem.server.gates import _gate_4_recall_check

        live_dir = self._live_registry_with_n_keys(tmp_path, 1)
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)

        with patch(
            "paramem.memory.store.MemoryStore.read_registries_from_disk",
            side_effect=OSError("permission denied"),
        ):
            result = _gate_4_recall_check(
                model=_make_mock_model(["episodic"]),
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                live_adapter_dir=live_dir,
                mount_state={},
                recall_probe_batch_size=4,
            )

        assert result.status == "fail"
        assert result.gate == 4
        assert "permission denied" in result.reason

    def test_population_excludes_interim_tier_keys(self, tmp_path):
        """An interim slot's active keys carry no fingerprint in the trial's
        per-tier SimHash map -- full-replay training only ever writes that
        map for the three main tiers (``MAIN_TIERS``), never for
        an interim slot.  ``_live_key_population`` must never draw an
        interim key into the sample population, or the probe would score an
        unverifiable key as a spurious miss."""
        from paramem.server.gates import GATE_4_MIN_REGISTRY_SIZE, _live_key_population

        live_dir = self._live_registry_with_n_keys(tmp_path, GATE_4_MIN_REGISTRY_SIZE)
        interim_dir = live_dir / "episodic" / "interim_20260101T0000"
        interim_dir.mkdir(parents=True)
        interim_reg = KeyRegistry()
        interim_reg.add("interim_only_key")
        (interim_dir / "indexed_key_registry.json").write_bytes(interim_reg.save_bytes())

        _content, population = _live_key_population(live_dir)

        assert "interim_only_key" not in population
        assert population == sorted(f"graph{i}" for i in range(GATE_4_MIN_REGISTRY_SIZE))

    def test_gate_4_never_samples_an_interim_only_key(self, tmp_path):
        """End-to-end: an interim-only key never reaches the deciding
        sample, even when it would otherwise be drawn -- the trial's
        SimHash map cannot score it, so the gate must never ask it to."""
        import json as _json

        from paramem.memory.entry import entry_simhash, finalize_recalled
        from paramem.server.gates import GATE_4_MIN_REGISTRY_SIZE, _gate_4_recall_check

        keys = [f"graph{i}" for i in range(GATE_4_MIN_REGISTRY_SIZE)]
        live_dir = self._live_registry_with_n_keys(tmp_path, GATE_4_MIN_REGISTRY_SIZE)
        interim_dir = live_dir / "episodic" / "interim_20260101T0000"
        interim_dir.mkdir(parents=True)
        interim_reg = KeyRegistry()
        interim_reg.add("interim_only_key")
        (interim_dir / "indexed_key_registry.json").write_bytes(interim_reg.save_bytes())

        def _fact_for(key: str) -> dict:
            return {"key": key, "subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}

        key_fingerprints = {k: entry_simhash(_fact_for(k)) for k in keys}
        trial_dir = self._trial_adapter_with_matching_registry(tmp_path, key_fingerprints)

        def _probe_gen(m, tok, entries, registry=None, **kw):
            for e in entries:
                raw = _json.dumps(_fact_for(e["key"]))
                yield e, finalize_recalled(raw, e["key"], registry, 0.75)

        model = _make_mock_model(["episodic"])
        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_gen):
            result = _gate_4_recall_check(
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                live_adapter_dir=live_dir,
                mount_state={},
                recall_probe_batch_size=4,
            )

        assert result.status == "pass", result.reason
        assert "interim_only_key" not in result.metrics["sampled_keys"]
