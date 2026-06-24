"""Tests for paramem.server.gates.

All tests run without GPU.  The model and tokenizer are MagicMocks.
``paramem.training.indexed_memory.probe_key`` is patched in every test that
exercises gates 3 or 4 in QA mode.  ``paramem.training.recall_eval.probe_entries``
is patched for quad-mode tests.  Registry files are written to ``tmp_path``.

Coverage:
  - Each gate's pass/fail/skip paths including the 2 new skip conditions.
  - NO_NEW_SESSIONS end-to-end (session_buffer_empty=True).
  - Gate 4 deterministic sample stability.
  - Gate 4 retry seed produces a different list.
  - Gate 4 ≥ 18/20 first sample → PASS, no retry.
  - Gate 4 17/20 first, 20/20 retry → PASS + cluster-variance warning.
  - Gate 4 17/20 both samples → FAIL.
  - Gate 4 < 20 keys → SKIPPED.
  - Gate 4 missing registry file → SKIPPED.
  - Gate 4 metrics includes sampled_keys.
  - Phase categorizer: extraction exception → gate 1 FAIL, gate 2 SKIPPED.
  - Phase categorizer: training exception → gate 1 PASS, gate 2 FAIL.
  - Phase categorizer logs at WARNING.
  - Unmount: delete_adapter called when > 1 adapter mounted.
  - Unmount: delete_adapter NOT called when trial_probe is the sole adapter.
  - Unmount survives delete_adapter raising.
  - Enriched registry format ({key: {"simhash": int, ...}}) works in gate 4.
  - Gate 3: read_keyed_pairs + probe_entries → PASS / FAIL.
  - Gate 4: probe_entries (batched) + verify_confidence → PASS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.gates import (
    _ADAPTER_KIND_SUBDIRS,
    _TRIAL_PROBE_ADAPTER_NAME,
    GATE_4_SAMPLE_SIZE,
    GateResult,
    _ensure_trial_probe_mounted,
    _find_tier_registry,
    _gate_1_extraction,
    _gate_2_training,
    _gate_3_reload_smoke,
    _gate_4_recall_check,
    _is_training_marker,
    _sample_registry_keys,
    _unmount_trial_probe,
    evaluate_gates,
)
from paramem.training.key_registry import KeyRegistry


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


def _make_registry(n: int, tmp_path: Path) -> Path:
    """Write a simple registry with ``n`` keys to tmp_path/registry.json."""
    registry = {f"graph{i}": i * 1000 + 1 for i in range(1, n + 1)}
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(registry))
    return p


def _make_enriched_registry(n: int, tmp_path: Path) -> Path:
    """Write an enriched registry with ``n`` keys."""
    registry = {
        f"graph{i}": {
            "simhash": i * 1000 + 1,
            "created_at": "2026-04-22T00:00:00+00:00",
            "last_seen_at": "2026-04-22T00:00:00+00:00",
            "session_id": "s1",
            "status": "active",
            "stale_since": None,
            "stale_cycles": 0,
        }
        for i in range(1, n + 1)
    }
    p = tmp_path / "registry_enriched.json"
    p.write_text(json.dumps(registry))
    return p


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
    """Build a MagicMock that mimics a PeftModel with named adapters."""
    model = MagicMock()
    if adapter_names is None:
        adapter_names = ["episodic"]
    # peft_config is a dict keyed by adapter name.
    model.peft_config = {name: MagicMock() for name in adapter_names}
    model.active_adapter = adapter_names[0] if adapter_names else None
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
    def _make_content(self, n: int) -> bytes:
        """Build KeyRegistry JSON bytes for n keys using the new per-tier schema."""
        reg = KeyRegistry()
        for i in range(1, n + 1):
            reg.add(f"graph{i}")
        return reg.save_bytes()

    def test_stable_same_input(self):
        """Same bytes → same list every time (deterministic)."""
        content = self._make_content(50)
        keys1 = _sample_registry_keys(content)
        keys2 = _sample_registry_keys(content)
        assert keys1 == keys2

    def test_sample_size_capped_at_20(self):
        content = self._make_content(50)
        keys = _sample_registry_keys(content)
        assert len(keys) == GATE_4_SAMPLE_SIZE

    def test_sample_size_less_than_20(self):
        content = self._make_content(10)
        keys = _sample_registry_keys(content)
        assert len(keys) == 10

    def test_retry_suffix_produces_different_list(self):
        """seed_suffix=b'|retry' must produce a different sample."""
        content = self._make_content(50)
        keys_first = _sample_registry_keys(content, seed_suffix=b"")
        keys_retry = _sample_registry_keys(content, seed_suffix=b"|retry")
        # Very unlikely to be identical with 50 keys and sample of 20.
        assert keys_first != keys_retry

    def test_sorted_population(self):
        """All returned keys must be from the registry's active_keys."""
        content = self._make_content(30)
        registry = json.loads(content)
        all_keys = set(registry["active_keys"])
        keys = _sample_registry_keys(content)
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
        """no_facts status → SKIPPED (REQUIRED FIX 2 — not PASS)."""
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

    def test_delete_not_called_when_sole_adapter(self, caplog):
        """delete_adapter must NOT be called when trial_probe is sole adapter."""
        import paramem.server.gates as gates_mod

        gates_mod.logger.propagate = True
        model = _make_mock_model(["trial_probe"])
        mount_state = {"mounted": True, "pre_active_adapter": []}
        with caplog.at_level(logging.WARNING):
            _unmount_trial_probe(model, mount_state)
        model.delete_adapter.assert_not_called()
        assert any("sole loaded adapter" in r.message for r in caplog.records)

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
# NO_NEW_SESSIONS end-to-end via evaluate_gates
# ---------------------------------------------------------------------------


class TestEvaluateGatesNoNewSessions:
    def test_all_gates_skipped_when_buffer_empty(self, tmp_path):
        """session_buffer_empty=True → all gates return skipped."""
        reg_path = _make_registry(25, tmp_path)
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        model = _make_mock_model()

        results = evaluate_gates(
            model=model,
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            live_registry_path=reg_path,
            session_buffer_empty=True,
            consolidation_summary=None,
            consolidation_exception=None,
            recall_probe_batch_size=16,
        )

        assert len(results) == 4
        # Gates 1/2/3 must be skipped.
        for r in results[:3]:
            assert r.status == "skipped", f"Gate {r.gate} expected skipped, got {r.status}"

    def test_trial_probe_not_in_adapters_after_evaluate(self, tmp_path):
        """Acceptance criterion D — trial_probe must not remain after evaluate_gates.

        The mock model starts with one adapter ("episodic").  The trial dir has
        episodic/indexed_key_registry.json so the in-memory path is taken:
        set_adapter is used (not load_adapter), and unmount restores the original
        adapter via set_adapter — delete_adapter is NOT called (we never added
        an extra adapter).  The peft_config keys remain unchanged (just "episodic").
        """
        reg_path = _make_registry(25, tmp_path)
        trial_dir = _make_trial_adapter(tmp_path)
        model = _make_mock_model(["episodic"])

        # Both gate 3 and gate 4 use probe_entries (generator yielding (entry, recalled) tuples).
        def _yield_recalled(m, tok, entries, registry=None, batch_size=1, **kw):
            for e in entries:
                yield (
                    e,
                    {
                        "key": e["key"],
                        "subject": "S",
                        "predicate": "p",
                        "object": "O",
                        "confidence": 1.0,
                        "raw_output": '{"key":"graph1","subject":"S","predicate":"p","object":"O"}',
                    },
                )

        with (
            patch("paramem.training.recall_eval.probe_entries", side_effect=_yield_recalled),
            patch("paramem.memory.entry.verify_confidence", return_value=1.0),
        ):
            evaluate_gates(
                model=model,
                tokenizer=MagicMock(),
                trial_adapter_dir=trial_dir,
                live_registry_path=reg_path,
                session_buffer_empty=False,
                consolidation_summary={"status": "complete"},
                consolidation_exception=None,
                recall_probe_batch_size=16,
            )

        # In-memory mount path: set_adapter was used, not load_adapter.
        # delete_adapter must NOT be called (no extra adapter was added).
        model.load_adapter.assert_not_called()
        model.delete_adapter.assert_not_called()
        # set_adapter was called at least twice: once to activate "episodic"
        # during mount, once to restore it during unmount.
        assert model.set_adapter.call_count >= 2

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

    # test_default_qa_path_unchanged: removed with QA-format retirement.


# ---------------------------------------------------------------------------
# Gate 4
# ---------------------------------------------------------------------------


def _make_quad_registry(n: int, tmp_path: Path, fname: str = "registry_quad.json") -> Path:
    """Write a flat registry with ``n`` keys."""
    registry = {f"qgraph{i}": i * 7919 + 3 for i in range(1, n + 1)}
    p = tmp_path / fname
    p.write_text(json.dumps(registry))
    return p


class TestGate4RecallCheckQuad:
    """Gate 4 quad-path: probe_entries (batched) + quad verify_confidence dispatch."""

    def test_pass_quad_20_of_20(self, tmp_path):
        """20/20 quad probes passing → PASS."""
        reg_path = _make_quad_registry(25, tmp_path)
        trial_dir = _make_trial_adapter_quad(tmp_path)
        model = _make_mock_model()

        def _probe_quad_batch(m, tok, entries, registry=None, batch_size=1, **kw):
            # Mirror probe_entries: a generator yielding (entry, recalled) tuples.
            for e in entries:
                key = e["key"]
                yield (
                    e,
                    {
                        "key": key,
                        "subject": "Alex",
                        "predicate": "lives_in",
                        "object": "Heilbronn",
                        "confidence": 1.0,
                        "raw_output": (
                            f'{{"key": "{key}", "subject": "Alex", '
                            '"predicate": "lives_in", "object": "Heilbronn"}'
                        ),
                    },
                )

        with patch("paramem.training.recall_eval.probe_entries", side_effect=_probe_quad_batch):
            with patch("paramem.memory.entry.verify_confidence", return_value=1.0):
                g = _gate_4_recall_check(
                    model=model,
                    tokenizer=MagicMock(),
                    trial_adapter_dir=trial_dir,
                    live_registry_path=reg_path,
                    mount_state={"mounted": True, "pre_active_adapter": ["episodic"]},
                    recall_probe_batch_size=16,
                )

        assert g.status == "pass"
        assert g.metrics["retried"] is False


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

        Uses the new per-tier KeyRegistry schema:
        ``{active_keys: [...], fidelity_history: {}, health: null}``.
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
        for kind in _ADAPTER_KIND_SUBDIRS:
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


# ---------------------------------------------------------------------------
# Gate 4 file-not-found → SKIPPED (2026-04-22 re-test fix)
# ---------------------------------------------------------------------------


class TestGate4RegistryFileNotFoundSkip:
    """Gate 4 must return SKIPPED (not FAIL) when the live registry file does
    not exist.

    Spec L381: "Skipped with — if the live registry has fewer than 20 keys
    (fresh install)."  File-not-found is the strongest form of that condition.
    Only corrupt / unparseable files (that *do* exist) should FAIL (missing-file
    case was changed to SKIPPED; corrupt-but-existing is still FAIL).
    """

    def test_gate4_skips_when_registry_file_does_not_exist(self, tmp_path):
        """live_registry_path points at a non-existent file → status='skipped'.

        Before the file-not-found fix, gate 4 returned FAIL with
        'live registry file not found'.  Fresh-install hosts have no
        production registry and must not be penalised.
        """
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)

        non_existent = tmp_path / "does_not_exist" / "key_metadata.json"
        # Do not create the file — it must be genuinely absent.
        assert not non_existent.exists()

        model = _make_mock_model()
        g = _gate_4_recall_check(
            model=model,
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            live_registry_path=non_existent,
            mount_state={"mounted": False, "pre_active_adapter": []},
            recall_probe_batch_size=16,
        )

        assert g.status == "skipped", (
            f"Gate 4 should return skipped on file-not-found, got {g.status}: {g.reason}"
        )
        assert "not found" in (g.reason or ""), f"Reason should mention 'not found': {g.reason!r}"
        assert "fresh install" in (g.reason or ""), (
            f"Reason should mention 'fresh install': {g.reason!r}"
        )

    def test_gate4_fails_when_registry_file_exists_but_unparseable(self, tmp_path):
        """File-not-found is SKIPPED; corrupt-but-existing is still FAIL.

        Pins the narrow scope of the fix: only the missing-file case was changed
        to SKIPPED.  A file that exists but contains invalid JSON must still
        return FAIL so genuine data corruption is not silently swallowed as a
        fresh-install skip.
        """
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)

        # Create a file that exists but contains invalid JSON.
        registry_path = tmp_path / "key_metadata.json"
        registry_path.write_text("not valid json {{{")
        assert registry_path.exists()

        model = _make_mock_model()
        g = _gate_4_recall_check(
            model=model,
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            live_registry_path=registry_path,
            mount_state={"mounted": False, "pre_active_adapter": []},
            recall_probe_batch_size=16,
        )

        assert g.status == "fail", (
            f"Gate 4 should FAIL for corrupt-but-existing registry, got {g.status}: {g.reason}"
        )
        # Reason must not mention fresh install — this is data corruption, not a fresh deploy.
        assert "fresh install" not in (g.reason or ""), (
            f"Corrupt-file FAIL reason must not mention 'fresh install': {g.reason!r}"
        )


# ---------------------------------------------------------------------------
# Gate 4 — SKIP unconditionally on missing live registry file
# ---------------------------------------------------------------------------


class TestGate4MissingRegistrySkips:
    """Gate 4 SKIPS on missing key_metadata.json — fresh install or pre-timestamped-slot
    layout. The CRITICAL #1 fix isolates trial registry writes to
    state/trial_registry/, so trial-induced corruption of the live file is no
    longer a concern that the gate needs to police.
    """

    def test_gate4_skip_when_file_missing(self, tmp_path):
        """Missing live registry → SKIPPED (no FAIL)."""
        missing_path = tmp_path / "nonexistent_registry.json"
        assert not missing_path.exists()

        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)

        g = _gate_4_recall_check(
            model=_make_mock_model(),
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            live_registry_path=missing_path,
            mount_state={"mounted": False, "pre_active_adapter": []},
            recall_probe_batch_size=16,
        )

        assert g.status == "skipped", (
            f"missing registry must be SKIPPED, got {g.status}: {g.reason}"
        )

    def test_evaluate_gates_skips_g4_on_missing(self, tmp_path):
        """evaluate_gates threads the missing-file SKIP all the way through."""
        missing_path = tmp_path / "nonexistent_registry.json"
        trial_dir = tmp_path / "trial_adapter"
        trial_dir.mkdir()
        (trial_dir / "adapter_config.json").write_text("{}")
        (trial_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4)

        from paramem.server.gates import evaluate_gates

        results = evaluate_gates(
            model=_make_mock_model(),
            tokenizer=MagicMock(),
            trial_adapter_dir=trial_dir,
            live_registry_path=missing_path,
            session_buffer_empty=False,
            consolidation_summary=None,
            consolidation_exception=None,
            recall_probe_batch_size=16,
        )
        assert results[3].status == "skipped"


# ---------------------------------------------------------------------------
# Bug A (live test) — trial-probe mount must use per-kind subdir
# ---------------------------------------------------------------------------


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
