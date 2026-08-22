"""Unit tests for ``paramem.models.loader.render_chat_prompt`` — THE one
production chat-template renderer.

CPU-only, no GPU / real model weights.  No prior test file directly unit-
tests ``render_chat_prompt``'s own contract (return type, single
application of ``adapt_messages``, ``add_generation_prompt`` forwarding);
existing references (``tests/test_cloud_agent.py``,
``tests/test_extraction_pipeline.py``, ``tests/server/test_calibrate.py``,
``tests/test_encode_boundary_guard.py``, ``tests/test_prompts_present.py``)
only exercise it indirectly through higher-level integration paths or
patch ``adapt_messages``/``apply_chat_template`` as a side effect of testing
something else.

Covers:
- Return type is :class:`~paramem.utils.tokens.RenderedPrompt`.
- :func:`~paramem.models.loader.adapt_messages` is applied exactly once
  per call (never zero, never twice).
- ``add_generation_prompt`` is forwarded to ``apply_chat_template``
  unchanged, for both its default and an explicit override.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from peft import PeftModel

from paramem.models.loader import has_prior_trained_weights, render_chat_prompt, tier_backup_scope
from paramem.utils.config import AdapterConfig
from paramem.utils.tokens import RenderedPrompt


class _RecordingChatTemplateTokenizer:
    """Tokenizer stub whose ``apply_chat_template`` records every call's
    ``messages``/kwargs and returns a fixed string — no folding logic of
    its own, so tests using it pair with a patched ``adapt_messages``
    (identity) to isolate the render step from the fold step.
    """

    def __init__(self, rendered: str = "rendered-prompt-text"):
        self.rendered = rendered
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.rendered


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]


class TestRenderChatPromptReturnType:
    def test_returns_rendered_prompt_instance(self):
        tok = _RecordingChatTemplateTokenizer(rendered="<s>[INST] hi [/INST]")
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            result = render_chat_prompt(_messages(), tok)
        assert isinstance(result, RenderedPrompt)
        assert result == "<s>[INST] hi [/INST]"

    def test_plain_str_is_not_a_rendered_prompt(self):
        """Negative control — a caller comparing against a plain str would
        get an ``isinstance`` mismatch, proving the wrapping is load-bearing."""
        tok = _RecordingChatTemplateTokenizer(rendered="text")
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            result = render_chat_prompt(_messages(), tok)
        assert not isinstance("text", RenderedPrompt)
        assert isinstance(result, RenderedPrompt)


class TestRenderChatPromptAdaptMessagesAppliedOnce:
    def test_adapt_messages_called_exactly_once(self):
        tok = _RecordingChatTemplateTokenizer()
        messages = _messages()
        with patch(
            "paramem.models.loader.adapt_messages",
            side_effect=lambda msgs, t: msgs,
        ) as mock_adapt:
            render_chat_prompt(messages, tok)
        mock_adapt.assert_called_once_with(messages, tok)

    def test_adapted_output_is_what_gets_rendered_not_the_raw_input(self):
        """Proves render_chat_prompt renders adapt_messages's OUTPUT, not a
        second, independent copy of the raw input — a caller-visible way to
        catch a regression where adaptation is computed but discarded (or
        applied a second time downstream)."""
        tok = _RecordingChatTemplateTokenizer()
        folded = [{"role": "user", "content": "SYS\n\nUSR"}]
        with patch(
            "paramem.models.loader.adapt_messages",
            MagicMock(return_value=folded),
        ) as mock_adapt:
            render_chat_prompt(_messages(), tok)
        mock_adapt.assert_called_once()
        assert tok.calls[-1][0] == folded


class TestRenderChatPromptAddGenerationPromptForwarded:
    def test_default_true_is_forwarded(self):
        tok = _RecordingChatTemplateTokenizer()
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            render_chat_prompt(_messages(), tok)
        _, kwargs = tok.calls[-1]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is False

    def test_explicit_false_is_forwarded(self):
        tok = _RecordingChatTemplateTokenizer()
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            render_chat_prompt(_messages(), tok, add_generation_prompt=False)
        _, kwargs = tok.calls[-1]
        assert kwargs["add_generation_prompt"] is False


_FAKE_BACKUP_LAYER = "layer0.q_proj"


class _FakeAdapterConfig:
    """Stand-in for a PEFT ``LoraConfig`` entry in ``model.peft_config`` —
    ``create_adapter`` reads/writes ``base_model_name_or_path`` on it.
    """

    def __init__(self):
        self.base_model_name_or_path = None


class _FakeParam:
    """Stand-in for a model parameter — ``create_adapter`` sums
    ``.numel()``/``.requires_grad`` over ``model.parameters()`` for its
    trainable-vs-total logging line.
    """

    def __init__(self, requires_grad: bool = True):
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return 1


def _make_fake_backup_model(resident_tiers=()):
    """Minimal fake exercising exactly what ``tier_backup_scope`` — and the
    real ``create_adapter``/``copy_adapter_weights`` it calls — touch:
    ``isinstance(model, PeftModel)`` (via ``__class__``, the same
    ``MagicMock().__class__ = PeftModel`` pattern used elsewhere in this
    tree), ``peft_config`` (entries carrying ``base_model_name_or_path``),
    ``add_adapter``/``set_adapter``/``delete_adapter``, ``parameters``
    (for the trainable-param count), and ``named_parameters`` (for
    ``copy_adapter_weights``'s suffix-matched tensor copy).
    """
    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {}
    params: dict = {}

    def _seed_adapter(name):
        model.peft_config[name] = _FakeAdapterConfig()
        params[f"base_model.model.{_FAKE_BACKUP_LAYER}.lora_A.{name}.weight"] = MagicMock(
            data=_FakeTensor()
        )
        params[f"base_model.model.{_FAKE_BACKUP_LAYER}.lora_B.{name}.weight"] = MagicMock(
            data=_FakeTensor()
        )

    def _add_adapter(name, lora_config):
        _seed_adapter(name)

    def _set_adapter(name):
        model.active_adapter = name

    def _delete_adapter(name):
        model.peft_config.pop(name, None)
        for key in [k for k in params if f".{name}." in k]:
            del params[key]

    model.add_adapter.side_effect = _add_adapter
    model.set_adapter.side_effect = _set_adapter
    model.delete_adapter.side_effect = _delete_adapter
    model.named_parameters.side_effect = lambda: list(params.items())
    model.parameters.side_effect = lambda: [_FakeParam()]
    model._params = params  # exposed for the tests' own assertions

    for tier in resident_tiers:
        _seed_adapter(tier)
    model.active_adapter = resident_tiers[0] if resident_tiers else None
    return model


class _FakeTensor:
    """Stand-in for a ``torch.Tensor`` supporting only what this suite
    needs: ``.clone()``, ``.zero_()``, and ``==`` for equality assertions.
    """

    def __init__(self, value: int = 1):
        self.value = value

    def clone(self):
        return _FakeTensor(self.value)

    def zero_(self):
        self.value = 0
        return self

    def copy_(self, other):
        self.value = other.value
        return self

    def __eq__(self, other):
        return self.value == other.value


class TestTierBackupScope:
    """``tier_backup_scope`` snapshots exactly one tier's resident adapter
    into a transient ``<tier>_backup`` and restores it on any exception —
    the single-tier implementation directly (the tandem go-live design
    narrowed this from a multi-tier scope, since no tier's weights are
    live until the whole bundle has written, so this scope's only job is
    the unwind of a tier that never went live).
    """

    def test_requires_a_peft_model(self):
        config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        with pytest.raises(RuntimeError):
            with tier_backup_scope(object(), config, "episodic"):
                pass

    def test_yielded_scope_carries_no_model_attribute(self):
        """The base model's object identity never changes across the scope,
        so the yielded handle has nothing to resync -- it carries only
        ``vram``, never a ``model`` reference the caller could be tempted
        to read instead of its own."""
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])

        with tier_backup_scope(model, config, "semantic") as scope:
            assert not hasattr(scope, "model")

    def test_yielded_scope_carries_no_model_attribute_even_on_exception(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])
        captured_scope = None

        with pytest.raises(RuntimeError, match="boom"):
            with tier_backup_scope(model, config, "semantic") as scope:
                captured_scope = scope
                raise RuntimeError("boom")

        assert captured_scope is not None
        assert not hasattr(captured_scope, "model")

    def test_snapshots_and_frees_the_backup_on_clean_exit(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])

        with tier_backup_scope(model, config, "semantic"):
            assert "semantic_backup" in model.peft_config
            # Corrupt the resident tier as if training mutated it in place.
            for name in list(model._params):
                if ".semantic." in name and "semantic_backup" not in name:
                    model._params[name].data.zero_()

        # Clean exit: the backup is freed, the (mutated) tier is untouched.
        assert "semantic_backup" not in model.peft_config

    def test_restores_the_tier_from_its_backup_on_exception(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])
        original = {
            name: p.data.clone() for name, p in model._params.items() if ".semantic." in name
        }

        with pytest.raises(RuntimeError, match="boom"):
            with tier_backup_scope(model, config, "semantic"):
                for name in list(model._params):
                    if ".semantic." in name and "semantic_backup" not in name:
                        model._params[name].data.zero_()
                raise RuntimeError("boom")

        # The tier's weights are restored from the backup, and the backup
        # itself is freed on the way out.
        assert "semantic_backup" not in model.peft_config
        for name, tensor in original.items():
            assert model._params[name].data == tensor

    def test_non_resident_tier_is_a_no_op(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=[])

        with tier_backup_scope(model, config, "semantic"):
            assert "semantic_backup" not in model.peft_config
        assert "semantic_backup" not in model.peft_config


class TestTierBackupScopeDiscardsStaleBackup:
    """A ``<tier>_backup`` adapter leaked from a prior aborted event is
    discarded before this entry re-snapshots — a stale snapshot must never
    be the one a later restore replays."""

    def test_leaked_backup_is_replaced_by_a_fresh_snapshot_of_current_content(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])

        # Simulate a leaked backup from a prior aborted event, carrying
        # content that is NOT the tier's current content.
        model.add_adapter("semantic_backup", config)
        for name, param in list(model._params.items()):
            if ".semantic_backup." in name:
                param.data.value = 999

        with pytest.raises(RuntimeError, match="boom"):
            with tier_backup_scope(model, config, "semantic"):
                for name in list(model._params):
                    if ".semantic." in name and "semantic_backup" not in name:
                        model._params[name].data.zero_()
                raise RuntimeError("boom")

        # Restored from the FRESH snapshot (value 1, the tier's real prior
        # content) — never from the leaked stale backup's sentinel (999).
        for name, param in model._params.items():
            if ".semantic." in name and "semantic_backup" not in name:
                assert param.data.value == 1
        assert "semantic_backup" not in model.peft_config


class TestTierBackupScopePartialEnterDoubleFault:
    """An exception raised WHILE the snapshot itself is being taken (before
    ``snapshotted`` is ever set) must not trigger a restore — there is
    nothing complete to restore FROM, and the tier was never touched by the
    (never-entered) body."""

    def test_failure_during_copy_adapter_weights_skips_restore_but_still_cleans_up(
        self, monkeypatch
    ):
        import paramem.models.loader as loader_mod

        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])
        original = {
            name: p.data.clone() for name, p in model._params.items() if ".semantic." in name
        }

        def _boom(*args, **kwargs):
            raise RuntimeError("copy fail")

        monkeypatch.setattr(loader_mod, "copy_adapter_weights", _boom)

        with pytest.raises(RuntimeError, match="copy fail"):
            with tier_backup_scope(model, config, "semantic"):
                pass  # body never runs -- the failure is in the snapshot step

        # The tier itself was never mutated (the body never ran), so its
        # content is unchanged -- no restore was needed or attempted.
        for name, tensor in original.items():
            assert model._params[name].data == tensor
        # The partially-created backup (add_adapter ran, copy never did) is
        # still cleaned up by the unconditional finally block.
        assert "semantic_backup" not in model.peft_config


class TestTierBackupScopeTeardownNeverMasksTheInFlightException:
    """A teardown failure (switch-off or backup deletion, both best-effort)
    is logged and swallowed -- the ORIGINAL body exception always wins."""

    def test_switch_off_and_delete_adapter_failures_both_swallowed(self, monkeypatch):
        import paramem.models.loader as loader_mod

        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])
        original = {
            name: p.data.clone() for name, p in model._params.items() if ".semantic." in name
        }

        def _switch_off_boom(*args, **kwargs):
            raise ValueError("switch-off failed")

        monkeypatch.setattr(loader_mod, "_switch_off", _switch_off_boom)

        real_delete = model.delete_adapter.side_effect

        def _delete_boom(name):
            if name == "semantic_backup":
                raise ValueError("delete failed")
            real_delete(name)

        with pytest.raises(RuntimeError, match="boom"):
            with tier_backup_scope(model, config, "semantic"):
                model.delete_adapter.side_effect = _delete_boom
                for name in list(model._params):
                    if ".semantic." in name and "semantic_backup" not in name:
                        model._params[name].data.zero_()
                raise RuntimeError("boom")

        # The restore (which runs before the failing teardown) still landed.
        for name, tensor in original.items():
            assert model._params[name].data == tensor


class TestSwitchOffFallbackChain:
    """``_switch_off`` moves the active adapter onto the first resident
    fallback tier, in order -- and is a no-op both when the named adapter
    is not currently active and when no fallback tier is resident."""

    def test_no_op_when_adapter_name_is_not_currently_active(self):
        from paramem.models.loader import _switch_off

        model = _make_fake_backup_model(resident_tiers=["episodic"])
        model.active_adapter = "episodic"

        _switch_off(model, "semantic_backup", ("episodic", "semantic", "procedural"))

        model.set_adapter.assert_not_called()
        assert model.active_adapter == "episodic"

    def test_switches_to_the_first_resident_fallback_in_order(self):
        from paramem.models.loader import _switch_off

        model = _make_fake_backup_model(resident_tiers=["semantic", "procedural"])
        model.active_adapter = "semantic_backup"
        model.peft_config["semantic_backup"] = model.peft_config["semantic"]

        _switch_off(model, "semantic_backup", ("episodic", "semantic", "procedural"))

        model.set_adapter.assert_called_once_with("semantic")
        assert model.active_adapter == "semantic"

    def test_skips_non_resident_fallback_tiers_ahead_of_a_resident_one(self):
        from paramem.models.loader import _switch_off

        model = _make_fake_backup_model(resident_tiers=["procedural"])
        model.active_adapter = "procedural_backup"
        model.peft_config["procedural_backup"] = model.peft_config["procedural"]

        _switch_off(model, "procedural_backup", ("episodic", "semantic", "procedural"))

        model.set_adapter.assert_called_once_with("procedural")

    def test_no_op_when_no_fallback_tier_is_resident(self):
        from paramem.models.loader import _switch_off

        model = _make_fake_backup_model(resident_tiers=[])
        model.peft_config["orphan_backup"] = _FakeAdapterConfig()
        model.active_adapter = "orphan_backup"

        _switch_off(model, "orphan_backup", ("episodic", "semantic", "procedural"))

        model.set_adapter.assert_not_called()
        assert model.active_adapter == "orphan_backup"


class TestTierBackupScopeVram:
    """``tier_backup_scope`` wraps its own snapshot in ``vram_measure`` and
    exposes the result on ``_BackupScope.vram`` -- it records nothing
    itself; the telemetry write belongs to the caller."""

    def test_vram_populated_when_tier_resident(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=["semantic"])

        with tier_backup_scope(model, config, "semantic") as scope:
            assert set(scope.vram) == {"free_before", "free_after", "delta", "total"}

    def test_vram_empty_when_tier_not_resident(self):
        config = AdapterConfig(rank=8, alpha=16, target_modules=["q_proj"])
        model = _make_fake_backup_model(resident_tiers=[])

        with tier_backup_scope(model, config, "semantic") as scope:
            assert scope.vram == {}


class TestHasPriorTrainedWeights:
    """``has_prior_trained_weights`` -- the one predicate for "this tier has
    weights worth starting a staging slot from": resident in
    ``peft_config`` AND measuring warm."""

    def test_absent_adapter_is_false(self):
        model = MagicMock()
        model.peft_config = {}
        assert has_prior_trained_weights(model, "episodic") is False

    def test_resident_cold_adapter_is_false(self):
        import torch

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock()}
        model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", MagicMock(data=torch.zeros(2, 2))),
        ]
        assert has_prior_trained_weights(model, "episodic") is False

    def test_resident_warm_adapter_is_true(self):
        import torch

        model = MagicMock()
        model.peft_config = {"episodic": MagicMock()}
        model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", MagicMock(data=torch.ones(2, 2))),
        ]
        assert has_prior_trained_weights(model, "episodic") is True

    def test_no_peft_config_attribute_is_false(self):
        assert has_prior_trained_weights(object(), "episodic") is False
