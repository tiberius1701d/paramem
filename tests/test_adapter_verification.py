"""Tests for the post-save disk-integrity probe on ConsolidationLoop.

``_verify_saved_adapter_from_disk`` reloads a saved adapter slot from disk
into an isolated verify adapter and probes recall to catch silent partial
writes.  These tests cover the four contract points:

T1 - happy path: probe passes, returns recall >= threshold, no exception.
T2 - simulated corruption: probe detects degraded recall, raises RuntimeError.
T3 - in-RAM adapter preserved: original adapter is active after verify (pass or fail).
T4 - cleanup discipline: verify slot is dropped even when the probe raises.

All tests use a MagicMock model (no real GPU).  ``_run_recall_sanity_probe``
is patched directly on the loop instance so the tests are independent of
``experiments.utils.test_harness``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from peft import PeftModel

from paramem.memory.store import MemoryStore
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_peft_model(adapter_names: list[str] | None = None) -> MagicMock:
    """Return a MagicMock whose isinstance check against PeftModel passes.

    Uses ``spec=PeftModel`` so ``isinstance(model, PeftModel)`` is ``True``.
    Configures ``peft_config`` as a dict so the method's ``in``-membership
    and ``delete_adapter`` behaviour can be tracked.

    Args:
        adapter_names: Initial adapter names to populate in ``peft_config``.
            Defaults to ``["episodic"]``.
    """
    if adapter_names is None:
        adapter_names = ["episodic"]

    model = MagicMock(spec=PeftModel)
    lora_cfg = MagicMock()
    lora_cfg.base_model_name_or_path = "test-base-model"
    model.peft_config = {name: lora_cfg for name in adapter_names}

    # Simulate get_base_model().config._name_or_path
    model.get_base_model.return_value.config._name_or_path = "test-base-model"

    # set_adapter / delete_adapter side effects that mutate peft_config so
    # subsequent membership checks (``verify_name in model.peft_config``)
    # reflect the real operation.
    def _set_adapter(name: str) -> None:
        pass  # no-op; active adapter tracking not needed for these tests

    def _load_adapter(slot_path: str, *, adapter_name: str) -> None:
        lora_cfg_new = MagicMock()
        lora_cfg_new.base_model_name_or_path = "test-base-model"
        model.peft_config[adapter_name] = lora_cfg_new

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.set_adapter.side_effect = _set_adapter
    model.load_adapter.side_effect = _load_adapter
    model.delete_adapter.side_effect = _delete_adapter

    return model


def _make_verify_loop(tmp_path: Path, model: MagicMock | None = None) -> ConsolidationLoop:
    """Return a minimal ConsolidationLoop for _verify_saved_adapter_from_disk tests.

    Bypasses ``__init__`` via ``object.__new__`` following the pattern in
    ``tests/test_simulate_train_parity.py::_make_bare_loop`` and
    ``tests/test_consolidation.py::TestSaveAdaptersManifest._make_save_loop``.
    """
    if model is None:
        model = _make_peft_model()

    loop = object.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = MagicMock()
    loop.config = ConsolidationConfig()
    loop.training_config = TrainingConfig(num_epochs=1)
    loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.semantic_config = None
    loop.procedural_config = None
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop.store = MemoryStore(replay_enabled=True)
    loop.store.load_registry("episodic", KeyRegistry())
    loop.cycle_count = 0
    loop.merger = MagicMock()
    loop.store.replace_simhashes_in_tier("episodic", {})
    loop.store.replace_simhashes_in_tier("semantic", {})
    loop.store.replace_simhashes_in_tier("procedural", {})
    return loop


def _make_slot(tmp_path: Path, adapter_name: str = "episodic") -> Path:
    """Create a minimal (non-corrupt) slot directory for testing.

    Writes stub ``adapter_model.safetensors`` and ``adapter_config.json``
    mimicking what ``atomic_save_adapter`` produces after the flatten step.

    Args:
        tmp_path: Pytest temporary directory.
        adapter_name: Used to construct the slot directory name.
    """
    slot = tmp_path / f"slot_{adapter_name}"
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "adapter_model.safetensors").write_bytes(b"\x00\x01\x02\x03weights")
    (slot / "adapter_config.json").write_text('{"adapter_type": "lora"}')
    return slot


_SAMPLE_KEYED_PAIRS: list[dict] = [
    {
        "key": "graph1",
        "question": "Where does Alex live?",
        "answer": "Heilbronn.",
    },
    {
        "key": "graph2",
        "question": "What colour is the sky?",
        "answer": "Blue.",
    },
]


# ---------------------------------------------------------------------------
# T1 — happy path
# ---------------------------------------------------------------------------


class TestVerifySavedAdapterHappyPath:
    """T1: probe passes when the disk artifact is intact."""

    def test_returns_recall_above_threshold(self, tmp_path: Path) -> None:
        """Successful disk verify returns the measured recall rate.

        The verify slot is loaded, probed, the original adapter is restored,
        and the verify slot is dropped — all without raising.
        """
        model = _make_peft_model(["episodic"])
        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")

        # Patch the internal probe to return a passing rate (sharp recall).
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        rate = loop._verify_saved_adapter_from_disk(
            "episodic",
            slot,
            _SAMPLE_KEYED_PAIRS,
        )

        assert rate == pytest.approx(1.0)
        # load_adapter was called once with the slot path and verify name.
        model.load_adapter.assert_called_once_with(str(slot), adapter_name="episodic_verify")
        # The probe was invoked with the verify slot name.
        loop._run_recall_sanity_probe.assert_called_once_with(
            "episodic_verify",
            _SAMPLE_KEYED_PAIRS,
            max_probe=100,
            debug_phase="disk_verify",
        )

    def test_no_exception_when_recall_at_exact_threshold(self, tmp_path: Path) -> None:
        """Recall equal to threshold must NOT raise (boundary condition)."""
        loop = _make_verify_loop(tmp_path)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        # Should not raise — config default is 1.0.
        rate = loop._verify_saved_adapter_from_disk(
            "episodic",
            slot,
            _SAMPLE_KEYED_PAIRS,
        )
        assert rate == pytest.approx(1.0)

    def test_empty_keyed_pairs_returns_one_without_loading(self, tmp_path: Path) -> None:
        """Empty entries list is healthy by definition — no disk access needed."""
        model = _make_peft_model(["episodic"])
        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        rate = loop._verify_saved_adapter_from_disk(
            "episodic",
            slot,
            [],  # no pairs to verify
        )

        assert rate == pytest.approx(1.0)
        # No disk load occurred.
        model.load_adapter.assert_not_called()


# ---------------------------------------------------------------------------
# T2 — simulated corruption
# ---------------------------------------------------------------------------


class TestVerifySavedAdapterCorruption:
    """T2: degraded recall after reload-from-disk raises RuntimeError."""

    def test_raises_on_zero_recall(self, tmp_path: Path) -> None:
        """A completely corrupt artifact (recall=0.0) must raise RuntimeError.

        In production, safetensors parse failure causes ``_run_recall_sanity_probe``
        to catch the exception and return 0.0 (its defined failure semantics).
        We simulate that here by patching the probe to return 0.0 after
        writing bytes that would be rejected by a real safetensors parser.
        """
        model = _make_peft_model(["episodic"])
        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        # Corrupt the weights file after writing.
        (slot / "adapter_model.safetensors").write_bytes(b"\xff\xfe\x00\x01CORRUPT")

        # Simulate probe returning 0.0 (parse error or recall failure).
        loop._run_recall_sanity_probe = MagicMock(return_value=0.0)

        with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
            loop._verify_saved_adapter_from_disk(
                "episodic",
                slot,
                _SAMPLE_KEYED_PAIRS,
            )

    def test_raises_on_degraded_recall_below_threshold(self, tmp_path: Path) -> None:
        """Partially degraded recall (below threshold) raises RuntimeError.

        Models the case where the safetensors file parses successfully but the
        weights were partially overwritten — the model loads but answers wrong.
        """
        loop = _make_verify_loop(tmp_path)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.60)

        with pytest.raises(RuntimeError) as exc_info:
            loop._verify_saved_adapter_from_disk(
                "episodic",
                slot,
                _SAMPLE_KEYED_PAIRS,
            )

        msg = str(exc_info.value)
        assert "episodic" in msg
        assert "0.600" in msg or "0.60" in msg
        assert "1.00" in msg
        assert str(slot) in msg

    def test_error_message_mentions_slot_path(self, tmp_path: Path) -> None:
        """RuntimeError message must include the slot path for diagnosability."""
        loop = _make_verify_loop(tmp_path)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.0)

        with pytest.raises(RuntimeError) as exc_info:
            loop._verify_saved_adapter_from_disk(
                "episodic",
                slot,
                _SAMPLE_KEYED_PAIRS,
            )

        assert str(slot) in str(exc_info.value)


# ---------------------------------------------------------------------------
# T3 — in-RAM adapter preserved
# ---------------------------------------------------------------------------


class TestVerifySavedAdapterInRamPreserved:
    """T3: original adapter is active after verify (pass or fail)."""

    def test_original_adapter_active_after_successful_probe(self, tmp_path: Path) -> None:
        """After a passing probe, the original adapter is restored as active."""
        model = _make_peft_model(["episodic"])
        active_adapter: list[str] = []
        model.set_adapter.side_effect = lambda name: active_adapter.append(name)

        # Also wire delete to clean peft_config.
        def _delete(name: str) -> None:
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete

        # Reload: add verify slot to peft_config.
        def _load(slot_path: str, *, adapter_name: str) -> None:
            lora_cfg = MagicMock()
            lora_cfg.base_model_name_or_path = "test-base-model"
            model.peft_config[adapter_name] = lora_cfg

        model.load_adapter.side_effect = _load

        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        loop._verify_saved_adapter_from_disk("episodic", slot, _SAMPLE_KEYED_PAIRS)

        # The last set_adapter call must be for the original adapter, not verify.
        assert active_adapter[-1] == "episodic", (
            f"Expected last active adapter to be 'episodic', got {active_adapter}"
        )

    def test_original_adapter_active_after_failed_probe(self, tmp_path: Path) -> None:
        """After a failing probe, the original adapter is still restored via finally."""
        model = _make_peft_model(["episodic"])
        active_adapter: list[str] = []
        model.set_adapter.side_effect = lambda name: active_adapter.append(name)

        def _delete(name: str) -> None:
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete

        def _load(slot_path: str, *, adapter_name: str) -> None:
            lora_cfg = MagicMock()
            lora_cfg.base_model_name_or_path = "test-base-model"
            model.peft_config[adapter_name] = lora_cfg

        model.load_adapter.side_effect = _load

        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.0)

        with pytest.raises(RuntimeError):
            loop._verify_saved_adapter_from_disk("episodic", slot, _SAMPLE_KEYED_PAIRS)

        # Even after the exception, the finally block restored the original.
        assert "episodic" in active_adapter, (
            f"'episodic' was never set as active. set_adapter calls: {active_adapter}"
        )


# ---------------------------------------------------------------------------
# T4 — cleanup discipline
# ---------------------------------------------------------------------------


class TestVerifySavedAdapterCleanup:
    """T4: verify slot is dropped even when the probe raises."""

    def test_verify_slot_dropped_on_success(self, tmp_path: Path) -> None:
        """Verify slot is absent from peft_config after a passing probe."""
        model = _make_peft_model(["episodic"])

        def _load(slot_path: str, *, adapter_name: str) -> None:
            lora_cfg = MagicMock()
            lora_cfg.base_model_name_or_path = "test-base-model"
            model.peft_config[adapter_name] = lora_cfg

        def _delete(name: str) -> None:
            model.peft_config.pop(name, None)

        model.load_adapter.side_effect = _load
        model.delete_adapter.side_effect = _delete

        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        loop._verify_saved_adapter_from_disk("episodic", slot, _SAMPLE_KEYED_PAIRS)

        assert "episodic_verify" not in model.peft_config, (
            "Verify slot must be removed from peft_config after successful probe"
        )

    def test_verify_slot_dropped_when_probe_raises(self, tmp_path: Path) -> None:
        """Verify slot is absent from peft_config even when the probe raises RuntimeError."""
        model = _make_peft_model(["episodic"])

        def _load(slot_path: str, *, adapter_name: str) -> None:
            lora_cfg = MagicMock()
            lora_cfg.base_model_name_or_path = "test-base-model"
            model.peft_config[adapter_name] = lora_cfg

        def _delete(name: str) -> None:
            model.peft_config.pop(name, None)

        model.load_adapter.side_effect = _load
        model.delete_adapter.side_effect = _delete

        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.0)

        with pytest.raises(RuntimeError):
            loop._verify_saved_adapter_from_disk("episodic", slot, _SAMPLE_KEYED_PAIRS)

        assert "episodic_verify" not in model.peft_config, (
            "Verify slot must be removed from peft_config even after a failed probe"
        )

    def test_verify_slot_dropped_when_load_adapter_raises(self, tmp_path: Path) -> None:
        """Verify slot absent from peft_config when load_adapter itself raises.

        Simulates the case where PEFT cannot parse the safetensors file at all
        (e.g. completely truncated file).  The try/finally must still clean up
        any partial state.
        """
        model = _make_peft_model(["episodic"])

        # load_adapter raises without adding to peft_config.
        model.load_adapter.side_effect = ValueError("safetensors file corrupt")

        def _delete(name: str) -> None:
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete

        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        # When load_adapter raises the probe returns 0.0 (caught inside
        # _run_recall_sanity_probe), leading to RuntimeError from threshold gate.
        # Here load_adapter raises before we can call the probe, so the recall
        # stays 0.0 and the threshold raises.
        with pytest.raises((RuntimeError, ValueError)):
            loop._verify_saved_adapter_from_disk("episodic", slot, _SAMPLE_KEYED_PAIRS)

        # Regardless of which exception, the verify slot must not remain.
        assert "episodic_verify" not in model.peft_config

    def test_original_production_adapter_untouched(self, tmp_path: Path) -> None:
        """Production adapters (episodic, semantic, procedural) survive the verify cycle.

        After verify completes (pass or fail), the model's peft_config must
        still contain all production adapters that were present on entry.
        """
        model = _make_peft_model(["episodic", "semantic"])

        def _load(slot_path: str, *, adapter_name: str) -> None:
            lora_cfg = MagicMock()
            lora_cfg.base_model_name_or_path = "test-base-model"
            model.peft_config[adapter_name] = lora_cfg

        def _delete(name: str) -> None:
            model.peft_config.pop(name, None)

        model.load_adapter.side_effect = _load
        model.delete_adapter.side_effect = _delete

        loop = _make_verify_loop(tmp_path, model)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=1.0)

        loop._verify_saved_adapter_from_disk("episodic", slot, _SAMPLE_KEYED_PAIRS)

        assert "episodic" in model.peft_config, "Production episodic adapter must survive verify"
        assert "semantic" in model.peft_config, "Production semantic adapter must survive verify"
        assert "episodic_verify" not in model.peft_config, "Verify slot must be dropped"


# ---------------------------------------------------------------------------
# Integration: _save_adapters calls _verify_saved_adapter_from_disk
# ---------------------------------------------------------------------------


class TestSaveAdaptersCallsVerify:
    """_save_adapters must invoke _verify_saved_adapter_from_disk for each saved tier."""

    def _make_save_loop(self, tmp_path: Path) -> ConsolidationLoop:
        """Return a ConsolidationLoop wired for _save_adapters integration testing.

        Follows the pattern in ``TestSaveAdaptersManifest._make_save_loop``
        (tests/test_consolidation.py) exactly.
        """
        model = MagicMock()
        model.config._name_or_path = "test-base-model"
        model.config._commit_hash = None
        model.base_model.model.state_dict.return_value = {}
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        model.peft_config = {"episodic": lora_cfg}

        def _fake_save_pretrained(path: str, selected_adapters: list[str] | None = None) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = _fake_save_pretrained

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"
        tokenizer.backend_tokenizer = None
        tokenizer.vocab_size = 32000

        loop = object.__new__(ConsolidationLoop)
        loop.model = model
        loop.tokenizer = tokenizer
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = None
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._keep_prior_slots = 50  # high value so pruning is a no-op in these tests
        loop.store = MemoryStore(replay_enabled=True)
        loop.store.load_registry("episodic", KeyRegistry())
        loop.cycle_count = 0
        loop.merger = MagicMock()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        return loop

    def test_save_adapters_invokes_verify_probe(self, tmp_path: Path) -> None:
        """_save_adapters must call _verify_saved_adapter_from_disk for episodic tier."""
        loop = self._make_save_loop(tmp_path)
        calls: list[tuple[str, Path]] = []

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            calls.append((adapter_name, slot_path))
            return 1.0

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        loop._save_adapters()

        assert len(calls) >= 1, "_verify_saved_adapter_from_disk was never called"
        adapter_names = [c[0] for c in calls]
        assert "episodic" in adapter_names

    def test_save_adapters_verify_receives_slot_path(self, tmp_path: Path) -> None:
        """The slot_path passed to the verify probe must exist on disk."""
        loop = self._make_save_loop(tmp_path)
        received_slots: list[Path] = []

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            received_slots.append(slot_path)
            return 1.0

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        loop._save_adapters()

        assert received_slots, "No slots received by verify probe"
        for slot in received_slots:
            assert slot.exists(), f"Verify probe received non-existent slot path: {slot}"

    def test_save_adapters_propagates_verify_failure(self, tmp_path: Path) -> None:
        """When the verify probe raises RuntimeError, _save_adapters propagates it."""
        loop = self._make_save_loop(tmp_path)

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            raise RuntimeError("Post-save disk-integrity probe failed for adapter 'episodic'")

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
            loop._save_adapters()


# ---------------------------------------------------------------------------
# Pre-save probe removal + post-save slot cleanup
# ---------------------------------------------------------------------------


class TestPreSaveProbeRemoved:
    """_run_recall_sanity_probe must NOT be called in the interim-adapter and
    consolidate_interim_adapters paths (pre-save probes have been removed).
    Disk-integrity is now gated post-save by _save_adapters via
    _verify_saved_adapter_from_disk.
    """

    def _make_save_loop(self, tmp_path: Path) -> ConsolidationLoop:
        """Return a ConsolidationLoop wired for _save_adapters integration testing."""
        model = MagicMock()
        model.config._name_or_path = "test-base-model"
        model.config._commit_hash = None
        model.base_model.model.state_dict.return_value = {}
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        model.peft_config = {"episodic": lora_cfg}

        def _fake_save_pretrained(path: str, selected_adapters: list[str] | None = None) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = _fake_save_pretrained

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"
        tokenizer.backend_tokenizer = None
        tokenizer.vocab_size = 32000

        loop = object.__new__(ConsolidationLoop)
        loop.model = model
        loop.tokenizer = tokenizer
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = None
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._keep_prior_slots = 50  # high value so pruning is a no-op in these tests
        loop.store = MemoryStore(replay_enabled=True)
        loop.store.load_registry("episodic", KeyRegistry())
        loop.cycle_count = 0
        loop.merger = MagicMock()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        return loop

    def test_save_adapters_does_not_call_run_recall_sanity_probe(self, tmp_path: Path) -> None:
        """_run_recall_sanity_probe must not be called directly from _save_adapters.

        Post-save disk-integrity is handled by _verify_saved_adapter_from_disk,
        which calls _run_recall_sanity_probe internally on the reloaded slot.
        The old path that called _run_recall_sanity_probe on in-RAM weights
        before saving has been removed.
        """
        loop = self._make_save_loop(tmp_path)

        # Track whether the raw in-RAM probe is called directly from _save_adapters.
        probe_call_count: list[int] = [0]

        original_probe = getattr(loop, "_run_recall_sanity_probe", None)

        def _tracking_probe(adapter_name: str, entries: list[dict], **kwargs: object) -> float:
            probe_call_count[0] += 1
            if original_probe is not None:
                return original_probe(adapter_name, entries, **kwargs)  # type: ignore[call-arg]
            return 1.0

        # Also stub _verify_saved_adapter_from_disk so the probe inside it doesn't
        # fire (we only want to track calls that originate directly from _save_adapters,
        # not the ones that go through the disk-verify wrapper).
        loop._run_recall_sanity_probe = _tracking_probe  # type: ignore[assignment]
        loop._verify_saved_adapter_from_disk = MagicMock(return_value=1.0)  # type: ignore[assignment]

        loop._save_adapters()

        # _run_recall_sanity_probe must not be called directly from _save_adapters
        # (the verify wrapper is stubbed out, so if it IS called the count > 0).
        assert probe_call_count[0] == 0, (
            f"_run_recall_sanity_probe was called {probe_call_count[0]} time(s) directly "
            "from _save_adapters — the pre-save in-RAM probe must be removed"
        )


class TestPostSaveSlotCleanup:
    """When _verify_saved_adapter_from_disk raises, _save_adapters deletes
    the bad on-disk slot before re-raising RuntimeError."""

    def _make_save_loop(self, tmp_path: Path) -> ConsolidationLoop:
        """Return a ConsolidationLoop wired for slot-cleanup testing."""
        model = MagicMock()
        model.config._name_or_path = "test-base-model"
        model.config._commit_hash = None
        model.base_model.model.state_dict.return_value = {}
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        model.peft_config = {"episodic": lora_cfg}

        saved_slots: list[Path] = []

        def _fake_save_pretrained(path: str, selected_adapters: list[str] | None = None) -> None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = _fake_save_pretrained

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"
        tokenizer.backend_tokenizer = None
        tokenizer.vocab_size = 32000

        loop = object.__new__(ConsolidationLoop)
        loop.model = model
        loop.tokenizer = tokenizer
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = None
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._keep_prior_slots = 50  # high value so pruning is a no-op in these tests
        loop.store = MemoryStore(replay_enabled=True)
        loop.store.load_registry("episodic", KeyRegistry())
        loop.cycle_count = 0
        loop.merger = MagicMock()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        loop._saved_slots = saved_slots
        return loop

    def test_bad_slot_deleted_on_verify_failure(self, tmp_path: Path) -> None:
        """When the disk-verify probe raises, the saved slot directory is deleted.

        The verify stub receives the actual slot path that was created by
        atomic_save_adapter (via the real save_pretrained side effect on the
        mock model).  After RuntimeError propagates, the slot must not exist.
        """
        loop = self._make_save_loop(tmp_path)

        # Record which slot path the verify stub receives.
        received_slots: list[Path] = []

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            received_slots.append(slot_path)
            raise RuntimeError(
                f"Post-save disk-integrity probe failed for adapter '{adapter_name}'"
            )

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
            loop._save_adapters()

        # The slot that was passed to the verify probe must have been deleted.
        assert received_slots, "Verify probe was never called — no slot to check"
        for slot in received_slots:
            assert not slot.exists(), f"Bad slot {slot} must be deleted after failed disk-verify"

    def test_runtime_error_propagates_after_slot_cleanup(self, tmp_path: Path) -> None:
        """RuntimeError from the verify probe propagates to the caller after cleanup."""
        loop = self._make_save_loop(tmp_path)

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            raise RuntimeError("Post-save disk-integrity probe failed for adapter 'episodic'")

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
            loop._save_adapters()


# ---------------------------------------------------------------------------
# T5 — recall_sanity_threshold config knob
# ---------------------------------------------------------------------------


class TestRecallSanityThresholdKnob:
    """T5: the recall_sanity_threshold config knob default is 1.0 and flows
    through to _verify_saved_adapter_from_disk when no explicit override is
    passed.
    """

    def test_config_default_is_one(self) -> None:
        """ConsolidationConfig.recall_sanity_threshold defaults to 1.0."""
        cfg = ConsolidationConfig()
        assert cfg.recall_sanity_threshold == pytest.approx(1.0)

    def test_config_validation_rejects_out_of_range(self) -> None:
        """Values outside (0, 1] are rejected at construction time."""
        with pytest.raises(ValueError, match="recall_sanity_threshold"):
            ConsolidationConfig(recall_sanity_threshold=0.0)
        with pytest.raises(ValueError, match="recall_sanity_threshold"):
            ConsolidationConfig(recall_sanity_threshold=1.1)
        with pytest.raises(ValueError, match="recall_sanity_threshold"):
            ConsolidationConfig(recall_sanity_threshold=-0.1)

    def test_lower_threshold_flows_to_gate(self, tmp_path: Path) -> None:
        """When recall_sanity_threshold is lowered in config, the gate accepts
        a recall rate that would fail at the default 1.0.

        Concretely: with threshold=0.8, a probe returning 0.9 must NOT raise;
        without the config knob being read, the hardcoded 1.0 default would
        raise RuntimeError.
        """
        loop = _make_verify_loop(tmp_path)
        # Lower the threshold below the probe's return value.
        loop.config = ConsolidationConfig(recall_sanity_threshold=0.8)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.9)

        # With threshold=0.8, recall=0.9 must not raise.
        rate = loop._verify_saved_adapter_from_disk(
            "episodic",
            slot,
            _SAMPLE_KEYED_PAIRS,
        )
        assert rate == pytest.approx(0.9)

    def test_threshold_below_probe_still_raises(self, tmp_path: Path) -> None:
        """Recall below the configured threshold still raises RuntimeError."""
        loop = _make_verify_loop(tmp_path)
        loop.config = ConsolidationConfig(recall_sanity_threshold=0.8)
        slot = _make_slot(tmp_path, "episodic")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.7)

        with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
            loop._verify_saved_adapter_from_disk(
                "episodic",
                slot,
                _SAMPLE_KEYED_PAIRS,
            )


# ---------------------------------------------------------------------------
# save_adapter returns Path (M1 — COMMIT 2)
# ---------------------------------------------------------------------------


class TestSaveAdapterReturnsPath:
    """save_adapter must return the slot Path produced by atomic_save_adapter."""

    def test_save_adapter_returns_path(self, tmp_path: Path) -> None:
        """save_adapter is now declared ``-> Path`` and returns the slot directory.

        The return value is the timestamped slot subdir created by
        ``atomic_save_adapter`` — the same path that ``commit_tier_slot``
        captures and forwards to the verify callback.
        """
        from paramem.models.loader import save_adapter

        model = MagicMock()

        def _fake_save_pretrained(path: str, selected_adapters=None) -> None:
            from pathlib import Path as _Path

            p = _Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = _fake_save_pretrained

        result = save_adapter(model, tmp_path, "episodic")

        assert isinstance(result, Path), f"save_adapter must return Path, got {type(result)!r}"
        assert result.exists(), f"Returned slot path must exist on disk: {result}"
        assert result.parent == tmp_path or result.parent.parent == tmp_path, (
            "Slot must be a child (or grandchild via .pending) of target_dir"
        )


# ---------------------------------------------------------------------------
# _verify_committed_slot — entry shaping (M2) and delegation (COMMIT 2)
# ---------------------------------------------------------------------------


class TestVerifyCommittedSlot:
    """_verify_committed_slot must strip sentinel fields and delegate to
    _verify_saved_adapter_from_disk without mirroring its body.
    """

    def test_strips_new_sentinel_from_entries(self, tmp_path: Path) -> None:
        """The ``_new`` sentinel field must not reach _verify_saved_adapter_from_disk.

        ``all_interim_keyed`` carries ``_new=True`` (tag_new=True fold path).
        The probe entry builder must produce ONLY the four canonical SPO fields.
        """
        loop = _make_verify_loop(tmp_path)
        slot = _make_slot(tmp_path, "episodic_interim_20260101T0000")

        captured_entries: list[list[dict]] = []

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            captured_entries.append(entries)
            return 1.0

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        all_keyed_with_sentinels = [
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "_new": True,  # sentinel — must be stripped
                "extra_field": "ignored",  # any extra field must be stripped
            },
            {
                "key": "graph2",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme Corp",
                "_new": False,
            },
        ]

        loop._verify_committed_slot(
            "episodic_interim_20260101T0000",
            all_keyed_with_sentinels,
            slot,
        )

        assert len(captured_entries) == 1, "_verify_saved_adapter_from_disk called once"
        entries = captured_entries[0]
        assert len(entries) == 2
        for entry in entries:
            assert set(entry.keys()) == {"key", "subject", "predicate", "object"}, (
                f"Entry must have only 4 SPO fields, got: {set(entry.keys())!r}"
            )
            assert "_new" not in entry, "Sentinel field '_new' must be stripped"

    def test_delegates_to_verify_saved_adapter(self, tmp_path: Path) -> None:
        """_verify_committed_slot must call _verify_saved_adapter_from_disk, not copy it."""
        loop = _make_verify_loop(tmp_path)
        slot = _make_slot(tmp_path, "episodic_interim_20260101T0000")

        verify_called: list[tuple] = []

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            verify_called.append((adapter_name, slot_path))
            return 1.0

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        loop._verify_committed_slot(
            "episodic_interim_20260101T0000",
            [{"key": "g1", "subject": "A", "predicate": "b", "object": "C"}],
            slot,
        )

        assert len(verify_called) == 1, "_verify_saved_adapter_from_disk must be called once"
        assert verify_called[0][0] == "episodic_interim_20260101T0000"
        assert verify_called[0][1] == slot

    def test_propagates_verify_failure(self, tmp_path: Path) -> None:
        """A RuntimeError from _verify_saved_adapter_from_disk propagates unchanged."""
        loop = _make_verify_loop(tmp_path)
        slot = _make_slot(tmp_path, "episodic_interim_20260101T0000")
        loop._run_recall_sanity_probe = MagicMock(return_value=0.0)

        def _fake_verify(
            adapter_name: str,
            slot_path: Path,
            entries: list[dict],
            *,
            threshold: float | None = None,
            max_probe: int = 100,
        ) -> float:
            raise RuntimeError("Post-save disk-integrity probe failed for adapter 'x'")

        loop._verify_saved_adapter_from_disk = _fake_verify  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
            loop._verify_committed_slot(
                "episodic_interim_20260101T0000",
                [{"key": "g1", "subject": "A", "predicate": "b", "object": "C"}],
                slot,
            )


# ---------------------------------------------------------------------------
# commit_tier_slot verify callback (COMMIT 2)
# ---------------------------------------------------------------------------


class TestCommitTierSlotVerifyCallback:
    """commit_tier_slot must invoke the verify callback train-branch-only,
    between weight write and registry flush; a raise cleans up the orphan slot.
    """

    def _make_loop(self, tmp_path: Path) -> "ConsolidationLoop":
        """Minimal ConsolidationLoop for commit_tier_slot testing."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.output_dir = tmp_path
        loop.fingerprint_cache = None

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("graph1")
        store.load_registry("episodic_interim_20260101T0000", reg)
        store.put(
            "episodic_interim_20260101T0000",
            "graph1",
            {
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "sp1",
            },
            register=False,
        )
        loop.store = store
        return loop

    def test_verify_called_before_registry_flush_train(self, tmp_path: Path) -> None:
        """verify callback fires after weight write, before registry flush (train mode)."""
        from paramem.memory.persistence import commit_tier_slot

        loop = self._make_loop(tmp_path)

        call_order: list[str] = []

        def _fake_save_adapter(*args, **kwargs) -> Path:
            call_order.append("save_adapter")
            # Return a stub slot path (atomic_save_adapter would return real path).
            slot = tmp_path / "fake_slot"
            slot.mkdir(parents=True, exist_ok=True)
            (slot / "adapter_model.safetensors").write_bytes(b"weights")
            (slot / "adapter_config.json").write_text("{}")
            return slot

        from paramem.training.key_registry import KeyRegistry as _KR

        def _fake_save_from_bytes(payload, path, **kwargs) -> None:
            call_order.append("save_from_bytes")

        verify_calls: list[Path] = []

        def _verify(slot_path: Path) -> None:
            call_order.append("verify")
            verify_calls.append(slot_path)

        with (
            patch("paramem.models.loader.save_adapter", side_effect=_fake_save_adapter),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch.object(_KR, "save_from_bytes", side_effect=_fake_save_from_bytes),
        ):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic_interim_20260101T0000",
                stamp="20260101T0000",
                mode="train",
                all_keyed=[],
                output_dir=tmp_path,
                verify=_verify,
            )

        assert "verify" in call_order, "verify callback was not called"
        assert "save_from_bytes" in call_order, "registry flush was not called"
        verify_idx = call_order.index("verify")
        flush_idx = call_order.index("save_from_bytes")
        assert verify_idx < flush_idx, (
            f"verify must precede registry flush; order was: {call_order}"
        )
        assert len(verify_calls) == 1, "verify must be called exactly once"

    def test_verify_raise_removes_orphan_slot(self, tmp_path: Path) -> None:
        """When verify raises, the slot is removed and registry is NOT written.

        The existing _registry_flushed=False orphan-cleanup in commit_tier_slot's
        finally handles the removal — no second cleanup path needed.
        """
        from paramem.memory.interim_adapter import adapter_slot_root_for_name
        from paramem.memory.persistence import commit_tier_slot

        loop = self._make_loop(tmp_path)

        def _verify_that_raises(slot_path: Path) -> None:
            raise RuntimeError("Post-save disk-integrity probe failed for adapter 'x'")

        with (
            patch(
                "paramem.models.loader.save_adapter",
                return_value=tmp_path / "fake_slot",
            ),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            # Ensure slot_root exists before the call so we can check cleanup.
            slot_root = adapter_slot_root_for_name(tmp_path, "episodic_interim_20260101T0000")
            slot_root.mkdir(parents=True, exist_ok=True)

            with pytest.raises(RuntimeError, match="Post-save disk-integrity probe failed"):
                commit_tier_slot(
                    loop=loop,
                    tier="episodic",
                    adapter_name="episodic_interim_20260101T0000",
                    stamp="20260101T0000",
                    mode="train",
                    all_keyed=[],
                    output_dir=tmp_path,
                    verify=_verify_that_raises,
                )

        # Slot root must be removed by the orphan-cleanup finally.
        assert not slot_root.exists(), (
            f"Orphan slot dir must be removed after verify failure: {slot_root}"
        )

        # Registry file must NOT exist (registry flush never ran).
        registry_file = slot_root / "indexed_key_registry.json"
        assert not registry_file.exists(), (
            "Registry must not be written when verify raises before registry flush"
        )

    def test_verify_none_train_proceeds_unchanged(self, tmp_path: Path) -> None:
        """When verify=None, train mode commits exactly as before (no-op for verify)."""
        from paramem.memory.persistence import commit_tier_slot

        loop = self._make_loop(tmp_path)

        committed: list[bool] = []

        def _fake_save_from_bytes(payload, path, **kwargs) -> None:
            committed.append(True)

        from paramem.training.key_registry import KeyRegistry as _KR

        with (
            patch(
                "paramem.models.loader.save_adapter",
                return_value=tmp_path / "fake_slot",
            ),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
            patch.object(_KR, "save_from_bytes", side_effect=_fake_save_from_bytes),
        ):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic_interim_20260101T0000",
                stamp="20260101T0000",
                mode="train",
                all_keyed=[],
                output_dir=tmp_path,
                verify=None,
            )

        assert committed, "Registry flush (commit) must still happen when verify=None"

    def test_verify_not_called_in_simulate_mode(self, tmp_path: Path) -> None:
        """verify callback must NOT be called in simulate mode (no weights written)."""
        from paramem.memory.persistence import commit_tier_slot

        loop = self._make_loop(tmp_path)

        verify_called: list[bool] = []

        def _should_not_be_called(slot_path: Path) -> None:
            verify_called.append(True)

        with patch(
            "paramem.memory.persistence.save_memory_to_disk",
        ):
            commit_tier_slot(
                loop=loop,
                tier="episodic",
                adapter_name="episodic_interim_20260101T0000",
                stamp="20260101T0000",
                mode="simulate",
                all_keyed=[
                    {
                        "key": "graph1",
                        "subject": "Alice",
                        "predicate": "lives_in",
                        "object": "Berlin",
                        "speaker_id": "sp1",
                    }
                ],
                output_dir=tmp_path,
                verify=_should_not_be_called,
            )

        assert not verify_called, "verify must NOT be called in simulate mode"


# ---------------------------------------------------------------------------
# _persist_fold verify wiring (COMMIT 2)
# ---------------------------------------------------------------------------


class TestPersistFoldVerifyWiring:
    """_persist_fold must pass a non-None verify for interim-train and
    verify=None for interim-simulate.
    """

    def _make_persist_loop(self, tmp_path: Path) -> "ConsolidationLoop":
        """Return a ConsolidationLoop minimal enough to call _persist_fold directly."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.output_dir = tmp_path
        loop.fingerprint_cache = None
        loop.config = ConsolidationConfig()
        loop.merger = MagicMock()

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        store.load_registry("episodic_interim_20260601T0000", reg)
        loop.store = store
        return loop

    def test_interim_train_passes_non_none_verify(self, tmp_path: Path) -> None:
        """_persist_fold wires a verify callback when scope.source == 'weights'."""
        from paramem.training.consolidation import FoldScope

        loop = self._make_persist_loop(tmp_path)
        scope = FoldScope(
            name="interim",
            source="weights",
            persist="interim_slot",
        )

        captured_verify: list[object] = []

        def _fake_commit_tier_slot(**kwargs: object) -> None:
            captured_verify.append(kwargs.get("verify"))

        with patch(
            "paramem.memory.persistence.commit_tier_slot",
            side_effect=_fake_commit_tier_slot,
        ):
            loop._persist_fold(
                scope,
                adapter_name="episodic_interim_20260601T0000",
                stamp="20260601T0000",
                all_keyed=[{"key": "g1", "subject": "A", "predicate": "b", "object": "C"}],
            )

        assert len(captured_verify) == 1
        assert captured_verify[0] is not None, (
            "_persist_fold must pass a non-None verify for train interim (scope.source='weights')"
        )
        assert callable(captured_verify[0]), "verify must be callable"

    def test_interim_simulate_passes_none_verify(self, tmp_path: Path) -> None:
        """_persist_fold passes verify=None when scope.source != 'weights'."""
        from paramem.training.consolidation import FoldScope

        loop = self._make_persist_loop(tmp_path)
        scope = FoldScope(
            name="interim",
            source="disk",
            persist="interim_slot",
        )

        captured_verify: list[object] = []

        def _fake_commit_tier_slot(**kwargs: object) -> None:
            captured_verify.append(kwargs.get("verify"))

        with patch(
            "paramem.memory.persistence.commit_tier_slot",
            side_effect=_fake_commit_tier_slot,
        ):
            loop._persist_fold(
                scope,
                adapter_name="episodic_interim_20260601T0000",
                stamp="20260601T0000",
                all_keyed=[],
            )

        assert len(captured_verify) == 1
        assert captured_verify[0] is None, (
            "_persist_fold must pass verify=None for simulate interim (scope.source='disk')"
        )
